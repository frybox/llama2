const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


var GS: usize = 0;

const RawConfig = extern struct {
  const Self = @This();
  dim: i32,
  ffndim: i32,
  nlayers: i32,
  nheads: i32,
  nkvheads: i32,
  nvocab: i32,
  ncontext: i32,

  fn cook(self: Self) Config {
    return Config{
      .dim = @intCast(self.dim),
      .ffndim = @intCast(self.ffndim),
      .nlayers = @intCast(self.nlayers),
      .nheads = @intCast(self.nheads),
      .nkvheads = @intCast(self.nkvheads),
      .nvocab = @intCast(self.nvocab),
      .ncontext = @intCast(self.ncontext),
    };
  }
};


const Config = struct {
  dim: usize,
  ffndim: usize,
  nlayers: usize,
  nheads: usize,
  nkvheads: usize,
  nvocab: usize,
  ncontext: usize,
};



const QuantizedTensor = struct {
  const Self = @This();
  q: []i8,
  s: []f32,

  fn init (allocator: Allocator, n: usize) !Self {
    var qt: Self = undefined;
    qt.q = try allocator.alloc(i8, n);
    qt.s = try allocator.alloc(f32, n/GS);
    return qt;
  }

  fn deinit (self: *Self, allocator: Allocator) void {
    allocator.free(self.q);
    allocator.free(self.s);
    self.* = undefined;
  }

  fn dequantize (self: *Self, x: []f32) void {
    assert(self.q.len == x.len);
    for (0..x.len) |i| {
      x[i] = self.q[i] * self.s[i/GS];
    }
  }

  fn quantize (self: *Self, x: []f32) void {
    assert(self.q.len == x.len);
    const ngroups: usize = x.len / GS;
    const qmax: f32 = 127.0;
    for (0..ngroups) |g| {
      const gx: []f32 = x[g*GS..][0..GS];
      var gq: []i8  = self.q[g*GS..][0..GS];
      var max: f32 = 0.0;
      for (0..GS) |i| {
        const v = @abs(gx[i]);
        if (v > max) max = v;
      }
      const s = max / qmax;
      for (0..GS) |i| {
        gq[i] = @intFromFloat(@round(gx[i]/s));
      }
      self.s[g] = s;
    }
  }
};


const Weights = struct {
  const Self = @This();
  qembeddings: []QuantizedTensor,
  embeddings: []f32,
  wrmsattn: [*]f32,
  wrmsffn: [*]f32,
  wrmsfinal: [*]f32,
  wq: []QuantizedTensor,
  wk: []QuantizedTensor,
  wv: []QuantizedTensor,
  wo: []QuantizedTensor,
  w1: []QuantizedTensor,
  w2: []QuantizedTensor,
  w3: []QuantizedTensor,

  fn init_quantized_tensors (ptr: *[*]i8, n: usize, size_each: usize, allocator: Allocator) ![]QuantizedTensor {
    var p: [*]i8 = ptr.*;
    var qts: []QuantizedTensor = try allocator.alloc(QuantizedTensor, n);
    for (0..n) |i| {
      var qp: [*]i8 = @ptrCast(p);
      qts[i].q = qp[0..size_each];
      var sp: [*]f32 = @ptrCast(@alignCast(qp+size_each));
      qts[i].s = sp[0..size_each/GS];
      p = @ptrCast(sp+size_each/GS);
    }
    ptr.* = p;
    return qts;
  }

  fn init (c: *const Config, data: []u8, allocator: Allocator) !Self {
    const dim: usize = c.dim;
    const ffndim: usize = c.ffndim;
    const head_size: usize = dim / c.nheads;
    const kvdim: usize = head_size * c.nkvheads;
    const nlayers: usize = c.nlayers;
    var w: Self = undefined;
    var fp: [*]f32 = @ptrCast(@alignCast(data));
    w.wrmsattn = fp;     fp += nlayers * dim;
    w.wrmsffn = fp;      fp += nlayers * dim;
    w.wrmsfinal = fp;    fp += dim;
    var up: [*]i8 = @ptrCast(fp);
    w.qembeddings = try init_quantized_tensors(&up, 1, c.nvocab*dim, allocator);
    w.embeddings = try allocator.alloc(f32, c.nvocab*dim);
    w.qembeddings[0].dequantize(w.embeddings);
    w.wq = try init_quantized_tensors(&up, nlayers, dim*dim, allocator);
    w.wk = try init_quantized_tensors(&up, nlayers, dim*kvdim, allocator);
    w.wv = try init_quantized_tensors(&up, nlayers, dim*kvdim, allocator);
    w.wo = try init_quantized_tensors(&up, nlayers, dim*dim, allocator);
    w.w1 = try init_quantized_tensors(&up, nlayers, dim*ffndim, allocator);
    w.w2 = try init_quantized_tensors(&up, nlayers, dim*ffndim, allocator);
    w.w3 = try init_quantized_tensors(&up, nlayers, dim*ffndim, allocator);
    return w;
  }

  fn deinit (self: *Self, allocator: Allocator) void {
    allocator.free(self.qembeddings);
    allocator.free(self.embeddings);
    allocator.free(self.wq);
    allocator.free(self.wk);
    allocator.free(self.wv);
    allocator.free(self.wo);
    allocator.free(self.w1);
    allocator.free(self.w2);
    allocator.free(self.w3);
    self.* = undefined;
  }
};


const State = struct {
  const Self = @This();
  x:      []f32,
  x1:     []f32,
  x2:     []f32,
  xq:     QuantizedTensor,
  h:      []f32,
  h1:     []f32,
  hq:     QuantizedTensor,
  q:      []f32,
  attn:   []f32,
  logits: []f32,
  logits_indexed: []IndexedProb,
  kcache: []f32,
  vcache: []f32,

  fn init(allocator: Allocator, c: *Config) !Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const kvdim = dim * c.nkvheads / c.nheads;
    var s: Self = undefined;
    s.x = try allocator.alloc(f32, dim);
    s.x1 = try allocator.alloc(f32, dim);
    s.x2 = try allocator.alloc(f32, dim);
    s.xq = try QuantizedTensor.init(allocator, dim);
    s.h = try allocator.alloc(f32, ffndim);
    s.h1 = try allocator.alloc(f32, ffndim);
    s.hq = try QuantizedTensor.init(allocator, ffndim);
    s.q = try allocator.alloc(f32, dim);
    s.attn = try allocator.alloc(f32, c.ncontext*c.nheads);
    s.logits = try allocator.alloc(f32, c.nvocab);
    s.logits_indexed = try allocator.alloc(IndexedProb, c.nvocab);
    s.kcache = try allocator.alloc(f32, c.nlayers*c.ncontext*kvdim);
    s.vcache = try allocator.alloc(f32, c.nlayers*c.ncontext*kvdim);
    return s;
  }

  fn deinit(self: *Self, allocator: Allocator) void {
    allocator.free(self.x);
    allocator.free(self.x1);
    allocator.free(self.x2);
    self.xq.deinit(allocator);
    allocator.free(self.h);
    allocator.free(self.h1);
    self.hq.deinit(allocator);
    allocator.free(self.q);
    allocator.free(self.attn);
    allocator.free(self.logits);
    allocator.free(self.logits_indexed);
    allocator.free(self.kcache);
    allocator.free(self.vcache);
    self.* = undefined;
  }
};


const Tokenizer = struct {
  const Self = @This();
  tokens: [][]u8,
  scores: []f32,
  max_token_len: u32,

  fn fromFile(path: []const u8, vocab_size: usize, allocator: Allocator, io: std.Io) !Self {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var reader = f.reader(io, &buf);
    const tokenizer = try Self.init(&reader.interface, allocator, vocab_size);
    return tokenizer;
  }

  fn init(r: *std.Io.Reader, allocator: Allocator, vocab_size: usize) !Self {
    var tokenizer: Self = undefined;
    tokenizer.tokens = try allocator.alloc([]u8, vocab_size);
    tokenizer.scores = try allocator.alloc(f32, vocab_size);
    tokenizer.max_token_len = try r.takeInt(@TypeOf(tokenizer.max_token_len), .little);
    for (0..vocab_size) |i| {
      tokenizer.scores[i] = @bitCast(try r.takeInt(u32, .little));
      const token_len = try r.takeInt(u32, .little);
      tokenizer.tokens[i] = try allocator.alloc(u8, token_len);
      try r.readSliceAll(tokenizer.tokens[i]);
    }
    return tokenizer;
  }

  fn deinit(self: *const Self, allocator: Allocator) void {
    for (self.tokens) |t| {
      allocator.free(t);
    }
    allocator.free(self.scores);
    allocator.free(self.tokens);
  }

  fn lookup(self: *const Self, str: []const u8) ?u32 {
    for (self.tokens, 0..) |token, i| {
      if (std.mem.eql(u8, token, str)) {
        return @intCast(i);
      }
    }
    return null;
  }

  fn encode(self: *const Self, input: []const u8, allocator: Allocator) ![]u32 {
    var tokens = try allocator.alloc(u32, input.len+3);
    const buf = try allocator.alloc(u8, self.max_token_len*2+1+2);
    defer allocator.free(buf);
    var fba_alloc = std.heap.FixedBufferAllocator.init(buf);
    const fba = fba_alloc.allocator();
    var utf8buf: [4]u8 = undefined;
    var ii: usize = 0;
    var ti: usize = 0;
    tokens[ti] = 1; //bos
    ti += 1;
    while (ii < input.len) {
      const l = try std.unicode.utf8ByteSequenceLength(input[ii]);
      const cp: u21 = try std.unicode.utf8Decode(input[ii..][0..l]);
      const el = try std.unicode.utf8Encode(cp, &utf8buf);
      tokens[ti] = self.lookup(utf8buf[0..el]) orelse {
        return error.TokenNotFound;
      };
      ii += l;
      ti += 1;
    }
    while (true) {
      var best_score: f32 = -1e10;
      var best_id: u32 = 0;
      var best_idx: ?usize = null;
      for (0..ti-1) |i| {
        const both = try std.mem.concat(fba, u8, &[_][]u8 {
            self.tokens[tokens[i]],
            self.tokens[tokens[i+1]],
        });
        defer fba.free(both);
        if (self.lookup(both)) |id| {
          if (self.scores[id] > best_score) {
            best_score = self.scores[id];
            best_id = id;
            best_idx = i;
          }
        }
      }
      if (best_idx) |i| {
        tokens[i] = best_id;
        for (i+1..ti-1) |j| {
          tokens[j] = tokens[j+1];
        }
        ti -= 1;
      } else {
        break;
      }
    }
    tokens = try allocator.realloc(tokens, ti);
    return tokens;
  }
};


fn matmul (o: []f32, w: QuantizedTensor, x: QuantizedTensor) void {
  for (0..o.len) |i| {
    var v: f32 = 0.0;
    const ii: usize = i * x.q.len;
    for (0..x.s.len) |j| {
      const jj = j*GS;
      var iv: i32 = 0;
      for (0..GS) |k| {
        iv += @as(i32, w.q[ii+jj+k]) * @as(i32, x.q[jj+k]);
      }
      v += @as(f32, @floatFromInt(iv)) * w.s[j] * x.s[j];
    }
    o[i] = v;
  }
}


fn rmsnorm (o: []f32, x: []f32, w: []f32) void {
  var sum: f32 = 0.0;
  for (0..x.len) |i| {
    sum += x[i] * x[i];
  }
  sum /= @floatFromInt(x.len);
  sum += 1e-5;
  sum = 1.0 / @sqrt(sum);
  for (0..x.len) |i| {
    o[i] = x[i] * sum * w[i];
  }
}


fn softmax (x: []f32) void {
  var max = x[0];
  for (x[1..]) |v| {
    if (v > max) max = v;
  }
  var sum: f32 = 0.0;
  for (0..x.len) |i| {
    x[i] = @exp(x[i]-max);
    sum += x[i];
  }
  for (0..x.len) |i| {
    x[i] /= sum;
  }
}



fn transformer (token: usize, pos: usize, c: *const Config, s: *State, w: *const Weights) void {
  const dim = c.dim;
  const ffndim = c.ffndim;
  const hsize = dim / c.nheads;
  const kvdim = hsize * c.nkvheads;
  const kvmul = c.nheads / c.nkvheads;
  const fhsize: f32 = @floatFromInt(hsize);
  const fpos: f32 = @floatFromInt(pos);
  const x = s.x;
  @memcpy(x, w.embeddings[token*dim..][0..dim]);
  for (0..c.nlayers) |l| {
    // 1. attention sublayer
    const loff = l * c.ncontext * kvdim;
    rmsnorm(s.x1, x, w.wrmsattn[l*dim..][0..dim]);
    s.xq.quantize(s.x1);
    var q = s.q;
    var k = s.kcache[loff+pos*kvdim..][0..kvdim];
    var v = s.vcache[loff+pos*kvdim..][0..kvdim];
    matmul(q, w.wq[l], s.xq);
    matmul(k, w.wk[l], s.xq);
    matmul(v, w.wv[l], s.xq);
    for (0..dim/2) |ii| {
      const i = ii * 2;
      const fhdim: f32 = @floatFromInt((i) % hsize);
      const freq = 1.0 / std.math.pow(f32, 10000.0, fhdim / fhsize);
      const val: f32 = fpos * freq;
      const fcr = @cos(val);
      const fci = @sin(val);
      var v0 = q[i];
      var v1 = q[i+1];
      q[i] = v0 * fcr - v1 * fci;
      q[i+1] = v0 * fci + v1 * fcr;
      if (i < kvdim) {
        v0 = k[i];
        v1 = k[i+1];
        k[i] = v0 * fcr - v1 * fci;
        k[i+1] = v0 * fci + v1 * fcr;
      }
    }
    for (0..c.nheads) |h| {
      q = s.q[h*hsize..][0..hsize];
      const attn = s.attn[h*c.ncontext..][0..c.ncontext];
      for (0..pos+1) |t| {
        k = s.kcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        var sum: f32 = 0.0;
        for (0..hsize) |i| {
          sum += q[i] * k[i];
        }
        attn[t] = sum / std.math.sqrt(fhsize);
      }
      softmax(attn[0..pos+1]);
      var x1 = s.x1[h*hsize..][0..hsize];
      @memset(x1, 0);
      for (0..pos+1) |t| {
        v = s.vcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        for (0..hsize) |i| {
          x1[i] += attn[t] * v[i];
        }
      }
    }
    s.xq.quantize(s.x1);
    matmul(s.x2, w.wo[l], s.xq);
    for (0 .. x.len) |i| {
      x[i] += s.x2[i];
    }

    // 2. ffn sublayer
    rmsnorm(s.x1, x, w.wrmsffn[l*dim..][0..dim]);
    s.xq.quantize(s.x1);
    matmul(s.h, w.w1[l], s.xq);
    matmul(s.h1, w.w3[l], s.xq);
    for (0..ffndim) |i| {
      var val = s.h[i];
      val = val * (1.0/(1.0+std.math.exp(-val))) * s.h1[i];
      s.h[i] = val;
    }
    s.hq.quantize(s.h);
    matmul(s.x1, w.w2[l], s.hq);
    for (0 .. x.len) |i| {
      x[i] += s.x1[i];
    }
  }
  rmsnorm(x, x, w.wrmsfinal[0..dim]);
  s.xq.quantize(x);
  matmul(s.logits, w.qembeddings[0], s.xq);
}


fn argmax(x: []f32) usize {
  var max: f32 = x[0];
  var maxi: usize = 0;
  for (1..x.len) |i| {
    if (x[i] > max) {
      max = x[i];
      maxi = i;
    }
  }
  return maxi;
}


fn sample(x: []f32) u32 {
  const random = prng.random();
  const r = random.float(f32);
  var cdf: f32 = 0.0;
  for (x, 0..) |val, i| {
    cdf += val;
    if (r < cdf) {
      return i;
    }
  }
  return x.len - 1;
}


const IndexedProb = struct {
  const Self = @This();
  index: u32,
  value: f32,
  fn desc (_: void, a: Self, b: Self) bool {
    return a.value > b.value;
  }
};


fn sample_topp (logits: []f32, p: f32, sorted: []IndexedProb) usize {
  const flen: f32 = @floatFromInt(logits.len);
  const cutoff: f32 = (1-p) / (flen - 1);
  var n: usize = 0;
  for (0..logits.len) |i| {
    if (logits[i] >= cutoff) {
      sorted[n].value = logits[i];
      sorted[n].index = @intCast(i);
      n += 1;
    }
  }
  std.sort.pdq(IndexedProb, sorted[0..n], {}, IndexedProb.desc);
  var acc: f32 = 0;
  var index: usize = n - 1;
  for (0..n) |i| {
    acc += sorted[i].value;
    if (acc > p) {
      index = i;
      break;
    }
  }
  const random = prng.random();
  const r = random.float(f32) * acc;
  var cdf: f32 = 0;
  for (0..index+1) |i| {
    cdf += sorted[i].value;
    if (cdf > r) {
      return sorted[i].index;
    }
  }
  return sorted[index].index;
}


var prng: std.Random.DefaultPrng = undefined;
fn log (comptime format: []const u8, args: anytype) void {
  std.debug.print(format, args);
}


pub fn main (init: std.process.Init) !void {
  const allocator = init.arena.allocator();
  const io = init.io;
  var stdout_buffer: [4096]u8 = undefined;
  var writer = std.Io.File.stdout().writer(io, &stdout_buffer);
  const stdout: *std.Io.Writer = &writer.interface;

  const checkpoint_path: []const u8 = "stories15M-q8.bin";
  const tokenizer_path: []const u8 = "tokenizer.bin";
  const temperature: f32 = 1.0;
  const topp: f32 = 0.9;
  const steps: usize = 256;
  const now_ns = std.Io.Clock.real.now(io).nanoseconds;
  prng = std.Random.DefaultPrng.init(@truncate(@as(u96, @bitCast(now_ns))));


  const checkpoint = try std.Io.Dir.cwd().openFile(io, checkpoint_path, .{});
  var buffer: [4096]u8 = undefined;
  var reader = checkpoint.reader(io, &buffer);
  const r: *std.Io.Reader = &reader.interface;
  const magic_number: u32 = try r.takeInt(u32, .little);
  if (magic_number != 0x616b3432) {
    return error.BadMagicNumber;
  }
  const version: u32 = try r.takeInt(u32, .little);
  if (version != 2) {
    return error.InvalidVersion;
  }
  var rawConfig = try r.takeStruct(RawConfig, .little);
  var config = rawConfig.cook();
  _ = try r.takeByte(); // bypass shared_classifier
  const group_size: u32 = try r.takeInt(u32, .little);
  GS = group_size;
  const file_size = (try checkpoint.stat(io)).size;
  log("Config: {any}\n", .{config});
  log("temperature: {d}\n", .{temperature});
  log("top-p: {d}\n", .{topp});
  log("\n", .{});
  const data = try std.posix.mmap(null, file_size, .{.READ=true}, .{.TYPE=.PRIVATE}, checkpoint.handle, 0);
  const header_size: usize = 256;
  var weights = try Weights.init(&config, data[header_size..], allocator);
  defer weights.deinit(allocator);
  const tokenizer = try Tokenizer.fromFile(tokenizer_path, config.nvocab, allocator, io);
  defer tokenizer.deinit(allocator);
  var state = try State.init(allocator, &config);
  defer state.deinit(allocator);
  const prompt: []u32 = try tokenizer.encode("", allocator);
  const prompt_len = prompt.len;
  var token: usize = prompt[0];
  var next: usize = undefined;
  var timer: ?std.Io.Timestamp = null;
  var pos: usize = 0;
  while (pos < steps) : (pos += 1) {
    transformer(token, pos, &config, &state, &weights);
    if (pos < prompt_len-1) {
      next = prompt[pos+1];
    } else {
      if (temperature == 0.0) {
        next = argmax(state.logits);
      } else {
        for (state.logits) |*val| {
          val.* /= temperature;
        }
        softmax(state.logits);
        if (topp == 0.0 or topp == 1.0) {
          next = sample(state.logits);
        } else {
          next = sample_topp(state.logits, topp, state.logits_indexed);
        }
      }
    }
    if (next == 1) {
      break;
    }
    const ch = if (token == 1 and tokenizer.tokens[next][0] == ' ')
                  tokenizer.tokens[next][1..]
               else tokenizer.tokens[next];
    try stdout.print("{s}", .{ch});
    try stdout.flush();
    token = next;
    if (timer == null) {
      timer = std.Io.Clock.awake.now(io);
    }
  }
  const elapsed_ns = timer.?.untilNow(io, .awake).nanoseconds;
  const tokens_per_sec: u32 = @intFromFloat(
            @as(f64, @floatFromInt(pos-1))*std.time.ns_per_s / @as(f64, @floatFromInt(elapsed_ns)));
  log("\n\n{d} tokens per second\n", .{tokens_per_sec});
}
