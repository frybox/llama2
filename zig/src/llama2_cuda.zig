const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


comptime {
  @setFloatMode(.optimized);
}


// ---------------------------------------------------------------------------
// CUDA runtime, as used from the host. Device addresses are CUdeviceptr
// values (unsigned long long == u64 here); a `float*`/`cudaStream_t` C
// parameter is ABI-identical to u64, so the launchers below are declared
// with u64 params and called with u64 device addresses. The default stream
// is 0 (what c/llama2_cuda.cu uses everywhere).
// ---------------------------------------------------------------------------

const CUresult = c_int;
const CUdeviceptr = u64;
const cudaMemcpyHostToDevice: c_int = 1;
const cudaMemcpyDeviceToHost: c_int = 2;

extern "C" fn cudaMalloc(p: *CUdeviceptr, size: usize) CUresult;
extern "C" fn cudaFree(p: CUdeviceptr) CUresult;
extern "C" fn cudaMemcpy(dst: CUdeviceptr, src: CUdeviceptr, size: usize, kind: c_int) CUresult;
extern "C" fn cudaGetErrorString(e: CUresult) ?[*:0]const u8;

fn checkR(res: CUresult, line: usize) void {
  if (res != 0) {
    const s = cudaGetErrorString(res) orelse "?";
    print("CUDA error: {s} (line {d})\n", .{ s, line });
    std.process.exit(1);
  }
}

// size of f32 in bytes, for device-space pointer arithmetic.
const F32SZ: u64 = 4;


// ---------------------------------------------------------------------------
// Kernel launchers, defined in src/llama2_cuda.cu (nvcc-compiled object
// linked into this executable). One per kernel; the final u64 param is the
// cudaStream_t (default stream = 0).
// ---------------------------------------------------------------------------

extern "C" fn llama2_cuda_init(x: CUdeviceptr, x1: CUdeviceptr, emb: CUdeviceptr, w: CUdeviceptr, dim: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_rmsnorm(o: CUdeviceptr, x: CUdeviceptr, w: CUdeviceptr, n: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_qkv(q: CUdeviceptr, k: CUdeviceptr, v: CUdeviceptr, wq: CUdeviceptr, wk: CUdeviceptr, wv: CUdeviceptr, x: CUdeviceptr, dim: c_int, kvdim: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_rope(q: CUdeviceptr, k: CUdeviceptr, dim: c_int, kvdim: c_int, hsize: c_int, pos: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_attn_score(attn: CUdeviceptr, q: CUdeviceptr, kcache: CUdeviceptr, nheads: c_int, pos: c_int, hsize: c_int, kvdim: c_int, kvmul: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_softmax_rows(attn: CUdeviceptr, nheads: c_int, pos: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_attn_value(x1: CUdeviceptr, attn: CUdeviceptr, vcache: CUdeviceptr, nheads: c_int, pos: c_int, hsize: c_int, kvdim: c_int, kvmul: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_matmul_axpy(o: CUdeviceptr, w: CUdeviceptr, x: CUdeviceptr, n: c_int, d: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_ffn_gate(h: CUdeviceptr, h1: CUdeviceptr, w1: CUdeviceptr, w3: CUdeviceptr, x: CUdeviceptr, dim: c_int, ffndim: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_silu_mul(h: CUdeviceptr, h1: CUdeviceptr, ffndim: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_matmul(o: CUdeviceptr, w: CUdeviceptr, x: CUdeviceptr, n: c_int, d: c_int, stream: CUdeviceptr) void;
extern "C" fn llama2_cuda_sample(e: CUdeviceptr, idx: CUdeviceptr, s_out: CUdeviceptr, logits: CUdeviceptr, n: c_int, inv_temp: f32, next: CUdeviceptr, topp: f32, coin: f32, stream: CUdeviceptr) void;


// ---------------------------------------------------------------------------
// Config / host weights / device state — same checkpoint layout as the C.
// ---------------------------------------------------------------------------

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

  pub fn kvdim(self: Config) usize {
    return self.dim * self.nkvheads / self.nheads;
  }
};


// Host-side view over the mmap'd checkpoint: float arrays, in file order.
const Weights = struct {
  const Self = @This();
  embeddings: [*]f32,
  wrmsattn: [*]f32,
  wrmsffn: [*]f32,
  wrmsfinal: [*]f32,
  wq: [*]f32,
  wk: [*]f32,
  wv: [*]f32,
  wo: [*]f32,
  w1: [*]f32,
  w2: [*]f32,
  w3: [*]f32,

  fn init(c: *const Config, data: []u8) Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const head_size = dim / c.nheads;
    const kvdim = head_size * c.nkvheads;
    const nlayers = c.nlayers;
    var w: Self = undefined;
    var p: [*]f32 = @ptrCast(@alignCast(data));
    w.embeddings = p; p += c.nvocab * dim;
    w.wrmsattn = p;   p += nlayers * dim;
    w.wq = p;         p += nlayers * dim * dim;
    w.wk = p;         p += nlayers * dim * kvdim;
    w.wv = p;         p += nlayers * dim * kvdim;
    w.wo = p;         p += nlayers * dim * dim;
    w.wrmsffn = p;    p += nlayers * dim;
    w.w1 = p;         p += nlayers * dim * ffndim;
    w.w2 = p;         p += nlayers * dim * ffndim;
    w.w3 = p;         p += nlayers * dim * ffndim;
    w.wrmsfinal = p;  p += dim;
    return w;
  }
};


fn initBuf(comptime T: type, n: usize) CUdeviceptr {
  var ptr: CUdeviceptr = 0;
  checkR(cudaMalloc(&ptr, @sizeOf(T) * n), @src().line);
  if (ptr == 0) {
    print("cudaMalloc state failed\n", .{});
    std.process.exit(1);
  }
  return ptr;
}

fn freeBuf(ptr: CUdeviceptr) void {
  _ = cudaFree(ptr);
}

fn copyToDev(dst: CUdeviceptr, src: [*]f32, n: usize) void {
  checkR(cudaMemcpy(dst, @intFromPtr(src), @sizeOf(f32) * n, cudaMemcpyHostToDevice), @src().line);
}


// Device weights, in checkpoint order.
const DevWeights = struct {
  const Self = @This();
  embeddings: CUdeviceptr,
  wrmsattn: CUdeviceptr,
  wrmsffn: CUdeviceptr,
  wrmsfinal: CUdeviceptr,
  wq: CUdeviceptr,
  wk: CUdeviceptr,
  wv: CUdeviceptr,
  wo: CUdeviceptr,
  w1: CUdeviceptr,
  w2: CUdeviceptr,
  w3: CUdeviceptr,

  fn upload(c: *const Config, w: *const Weights) Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const head_size = dim / c.nheads;
    const kvdim = head_size * c.nkvheads;
    var d: Self = undefined;
    d.embeddings = initBuf(f32, c.nvocab * dim);
    d.wrmsattn = initBuf(f32, c.nlayers * dim);
    d.wrmsffn = initBuf(f32, c.nlayers * dim);
    d.wrmsfinal = initBuf(f32, dim);
    d.wq = initBuf(f32, c.nlayers * dim * dim);
    d.wk = initBuf(f32, c.nlayers * dim * kvdim);
    d.wv = initBuf(f32, c.nlayers * dim * kvdim);
    d.wo = initBuf(f32, c.nlayers * dim * dim);
    d.w1 = initBuf(f32, c.nlayers * dim * ffndim);
    d.w2 = initBuf(f32, c.nlayers * dim * ffndim);
    d.w3 = initBuf(f32, c.nlayers * dim * ffndim);
    copyToDev(d.embeddings, w.embeddings, c.nvocab * dim);
    copyToDev(d.wrmsattn, w.wrmsattn, c.nlayers * dim);
    copyToDev(d.wq, w.wq, c.nlayers * dim * dim);
    copyToDev(d.wk, w.wk, c.nlayers * dim * kvdim);
    copyToDev(d.wv, w.wv, c.nlayers * dim * kvdim);
    copyToDev(d.wo, w.wo, c.nlayers * dim * dim);
    copyToDev(d.wrmsffn, w.wrmsffn, c.nlayers * dim);
    copyToDev(d.w1, w.w1, c.nlayers * dim * ffndim);
    copyToDev(d.w2, w.w2, c.nlayers * dim * ffndim);
    copyToDev(d.w3, w.w3, c.nlayers * dim * ffndim);
    copyToDev(d.wrmsfinal, w.wrmsfinal, dim);
    return d;
  }

  fn deinit(self: *Self) void {
    freeBuf(self.embeddings);
    freeBuf(self.wrmsattn);
    freeBuf(self.wrmsffn);
    freeBuf(self.wrmsfinal);
    freeBuf(self.wq);
    freeBuf(self.wk);
    freeBuf(self.wv);
    freeBuf(self.wo);
    freeBuf(self.w1);
    freeBuf(self.w2);
    freeBuf(self.w3);
    self.* = undefined;
  }
};


// Device state: activation buffers + top-p sampling workspace.
const State = struct {
  const Self = @This();
  x: CUdeviceptr,
  x1: CUdeviceptr,
  h: CUdeviceptr,
  h1: CUdeviceptr,
  q: CUdeviceptr,
  attn: CUdeviceptr,
  logits: CUdeviceptr,
  kcache: CUdeviceptr,
  vcache: CUdeviceptr,
  sample_e: CUdeviceptr,
  sample_idx: CUdeviceptr,
  sample_S: CUdeviceptr,
  next_token: CUdeviceptr,

  fn init(c: *const Config) Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const kvdim = c.kvdim();
    var s: Self = undefined;
    s.x = initBuf(f32, dim);
    s.x1 = initBuf(f32, dim);
    s.h = initBuf(f32, ffndim);
    s.h1 = initBuf(f32, ffndim);
    s.q = initBuf(f32, dim);
    s.attn = initBuf(f32, c.nheads * c.ncontext);
    s.logits = initBuf(f32, c.nvocab);
    s.kcache = initBuf(f32, c.nlayers * c.ncontext * kvdim);
    s.vcache = initBuf(f32, c.nlayers * c.ncontext * kvdim);
    s.sample_e = initBuf(f32, c.nvocab);
    s.sample_idx = initBuf(i32, c.nvocab);
    s.sample_S = initBuf(f32, 1);
    s.next_token = initBuf(i32, 1);
    return s;
  }

  fn deinit(self: *Self) void {
    freeBuf(self.x);
    freeBuf(self.x1);
    freeBuf(self.h);
    freeBuf(self.h1);
    freeBuf(self.q);
    freeBuf(self.attn);
    freeBuf(self.logits);
    freeBuf(self.kcache);
    freeBuf(self.vcache);
    freeBuf(self.sample_e);
    freeBuf(self.sample_idx);
    freeBuf(self.sample_S);
    freeBuf(self.next_token);
    self.* = undefined;
  }
};


const Transformer = struct {
  const Self = @This();
  c: Config,
  s: State,
  dev_w: DevWeights,

  fn init(c: *const Config, w: *const Weights) Self {
    return Self{
      .c = c.*,
      .s = State.init(c),
      .dev_w = DevWeights.upload(c, w),
    };
  }

  fn deinit(self: *Self) void {
    self.s.deinit();
    self.dev_w.deinit();
  }

  // One forward pass; mirrors forward() in c/llama2_cuda.cu.
  fn forward(self: Self, token: usize, pos: usize) void {
    const c = self.c;
    const dim = c.dim;
    const ffndim = c.ffndim;
    const kvdim = c.kvdim();
    const kvmul = c.nheads / c.nkvheads;
    const hsize = dim / c.nheads;
    const s = self.s;
    const dw = self.dev_w;
    const x = s.x;
    const stream: CUdeviceptr = 0;

    // token embedding: x = emb[token]; x1 = rmsnorm(x) * wrmsattn[0]
    llama2_cuda_init(x, s.x1, dw.embeddings + token * dim * F32SZ, dw.wrmsattn, @intCast(dim), stream);

    for (0..c.nlayers) |l| {
      // 1. self-attention sublayer (l==0's rmsnorm is fused into init above)
      if (l > 0) {
        llama2_cuda_rmsnorm(s.x1, x, dw.wrmsattn + @as(u64, @intCast(l)) * dim * F32SZ, @intCast(dim), stream);
      }
      const loff: u64 = @as(u64, @intCast(l)) * c.ncontext * kvdim;
      const k_off = s.kcache + (loff + @as(u64, @intCast(pos)) * kvdim) * F32SZ;
      const v_off = s.vcache + (loff + @as(u64, @intCast(pos)) * kvdim) * F32SZ;
      llama2_cuda_qkv(s.q, k_off, v_off,
          dw.wq + @as(u64, @intCast(l)) * dim * dim * F32SZ,
          dw.wk + @as(u64, @intCast(l)) * dim * kvdim * F32SZ,
          dw.wv + @as(u64, @intCast(l)) * dim * kvdim * F32SZ,
          s.x1, @intCast(dim), @intCast(kvdim), stream);
      llama2_cuda_rope(s.q, k_off, @intCast(dim), @intCast(kvdim), @intCast(hsize), @intCast(pos), stream);
      llama2_cuda_attn_score(s.attn, s.q, s.kcache + loff * F32SZ, @intCast(c.nheads), @intCast(pos), @intCast(hsize), @intCast(kvdim), @intCast(kvmul), stream);
      llama2_cuda_softmax_rows(s.attn, @intCast(c.nheads), @intCast(pos), stream);
      llama2_cuda_attn_value(s.x1, s.attn, s.vcache + loff * F32SZ, @intCast(c.nheads), @intCast(pos), @intCast(hsize), @intCast(kvdim), @intCast(kvmul), stream);
      llama2_cuda_matmul_axpy(x, dw.wo + @as(u64, @intCast(l)) * dim * dim * F32SZ, s.x1, @intCast(dim), @intCast(dim), stream);

      // 2. ffn sublayer
      llama2_cuda_rmsnorm(s.x1, x, dw.wrmsffn + @as(u64, @intCast(l)) * dim * F32SZ, @intCast(dim), stream);
      llama2_cuda_ffn_gate(s.h, s.h1,
          dw.w1 + @as(u64, @intCast(l)) * dim * ffndim * F32SZ,
          dw.w3 + @as(u64, @intCast(l)) * dim * ffndim * F32SZ,
          s.x1, @intCast(dim), @intCast(ffndim), stream);
      llama2_cuda_silu_mul(s.h, s.h1, @intCast(ffndim), stream);
      llama2_cuda_matmul_axpy(x, dw.w2 + @as(u64, @intCast(l)) * ffndim * dim * F32SZ, s.h, @intCast(ffndim), @intCast(dim), stream);
    }

    llama2_cuda_rmsnorm(x, x, dw.wrmsfinal, @intCast(dim), stream);
    llama2_cuda_matmul(s.logits, dw.embeddings, x, @intCast(dim), @intCast(c.nvocab), stream);
  }

  // top-p sample from device logits; mirrors sample_device() in the C file.
  fn sampleDevice(self: Self, temperature: f32, topp: f32, coin: f32) usize {
    const s = self.s;
    llama2_cuda_sample(s.sample_e, s.sample_idx, s.sample_S, s.logits, @intCast(self.c.nvocab), 1.0 / temperature, s.next_token, topp, coin, 0);
    var tok: i32 = 0;
    checkR(cudaMemcpy(@intFromPtr(&tok), s.next_token, @sizeOf(i32), cudaMemcpyDeviceToHost), @src().line);
    return @intCast(tok);
  }
};


// ---------------------------------------------------------------------------
// Tokenizer: identical to the CPU implementation.
// ---------------------------------------------------------------------------

const Tokenizer = struct {
  const Self = @This();
  tokens: [][]u8,
  scores: []f32,
  max_token_len: u32,
  byte_pieces: [512]u8,

  fn fromFile(path: []const u8, vocab_size: usize, allocator: Allocator, io: std.Io) !Self {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var reader = f.reader(io, &buf);
    return Self.init(&reader.interface, allocator, vocab_size);
  }

  fn init(r: *std.Io.Reader, allocator: Allocator, vocab_size: usize) !Self {
    var tokenizer: Self = undefined;
    for (0..256) |i| {
      tokenizer.byte_pieces[i*2] = @intCast(i);
      tokenizer.byte_pieces[i*2+1] = 0;
    }
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
      if (mem.eql(u8, token, str)) {
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
        const both = try mem.concat(fba, u8, &[_][]u8 {
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

  /// C: char *decode(Tokenizer *t, int prev_token, int token)
  fn decode(self: *const Self, prev_token: u32, token: u32) []const u8 {
    var piece: []const u8 = self.tokens[token];
    if (prev_token == 1 and piece.len > 0 and piece[0] == ' ') piece = piece[1..];
    if (Self.parseBytePiece(piece)) |bytev| {
      const off: usize = @as(usize, bytev) * 2;
      piece = self.byte_pieces[off..off + 1];
    }
    return piece;
  }

  fn parseBytePiece(piece: []const u8) ?u8 {
    if (piece.len != 6 or piece[0] != '<' or piece[1] != '0'
     or piece[2] != 'x' or piece[5] != '>') return null;
    return std.fmt.parseInt(u8, piece[3..5], 16) catch null;
  }
};


// ---------------------------------------------------------------------------
// Sampler: xorshift64* RNG, same seed handling as the C main().
// ---------------------------------------------------------------------------

const Sampler = struct {
  const Self = @This();
  state: u64,

  fn init(seed: u64) Self {
    return Self{ .state = seed };
  }

  fn randomU32(self: *Self) u32 {
    self.state ^= self.state >> 12;
    self.state ^= self.state << 25;
    self.state ^= self.state >> 27;
    const prod: u128 = @as(u128, self.state) * @as(u128, 0x2545F4914F6CDD1D);
    return @truncate(prod >> 32);
  }

  fn randomF32(self: *Self) f32 {
    return @as(f32, @floatFromInt(self.randomU32() >> 8)) / 16777216.0;
  }
};


// ---------------------------------------------------------------------------
// generate: mirrors generate() in c/llama2_cuda.cu.
// ---------------------------------------------------------------------------

fn generate(transformer: *const Transformer, tokenizer: *const Tokenizer, sampler: *Sampler, steps: usize, stdout: *std.Io.Writer, io: std.Io) void {
  const prompt_tokens: []const u32 = &[_]u32{1}; // just BOS, like the C empty prompt
  const num_prompt_tokens: usize = prompt_tokens.len;
  assert(num_prompt_tokens >= 1);

  var token: usize = prompt_tokens[0];
  var next: usize = 0;
  var pos: usize = 0;
  var timer: ?std.Io.Timestamp = null;
  while (pos < steps) {
    transformer.forward(token, pos);
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      const coin = sampler.randomF32();
      next = transformer.sampleDevice(1.0, 0.9, coin);
    }
    pos += 1;
    if (next == 1) break;
    const piece = tokenizer.decode(@intCast(token), @intCast(next));
    if (piece.len > 0) {
      stdout.print("{s}", .{piece}) catch {};
      stdout.flush() catch {};
    }
    token = next;
    if (timer == null) {
      timer = std.Io.Clock.awake.now(io);
    }
  }
  stdout.print("\n", .{}) catch {};
  if (pos > 1) {
    const elapsed_ns = timer.?.untilNow(io, .awake).nanoseconds;
    const speed: f64 = @as(f64, @floatFromInt(pos - 1)) * std.time.ns_per_s / @as(f64, @floatFromInt(elapsed_ns));
    print("\ntotal {d} tokens, speed {d:.1} token/s\n\n\n", .{ pos - 1, speed });
  }
}


pub fn main(init: std.process.Init) !void {
  const allocator = init.arena.allocator();
  const io = init.io;

  var stdout_buffer: [4096]u8 = undefined;
  var writer = std.Io.File.stdout().writer(io, &stdout_buffer);
  const stdout: *std.Io.Writer = @ptrCast(&writer.interface);
  defer stdout.flush() catch {};

  // RUNCUDA_SEED, same as the C main(); otherwise the current nanosecond time.
  const now_ns = std.Io.Clock.real.now(io).nanoseconds;
  var rng_seed: u64 = @truncate(@as(u96, @bitCast(now_ns)));
  if (init.environ_map.get("RUNCUDA_SEED")) |seed_str| {
    rng_seed = std.fmt.parseInt(u64, seed_str, 10) catch rng_seed;
  }

  const checkpoint_path: []const u8 = "stories15M.bin";
  const tokenizer_path: []const u8 = "tokenizer.bin";
  const steps: usize = 256;

  const checkpoint = try std.Io.Dir.cwd().openFile(io, checkpoint_path, .{});
  defer checkpoint.close(io);
  var buffer: [4096]u8 = undefined;
  var reader = checkpoint.reader(io, &buffer);
  var rawConfig = try reader.interface.takeStruct(RawConfig, .little);
  const config = rawConfig.cook();
  const file_size = (try checkpoint.stat(io)).size;
  print("Config: {any}\n", .{config});
  print("\n", .{});

  const data = try std.posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, checkpoint.handle, 0);
  const weights = Weights.init(&config, data[@sizeOf(RawConfig)..]);
  var transformer = Transformer.init(&config, &weights);
  defer transformer.deinit();
  const tokenizer = try Tokenizer.fromFile(tokenizer_path, config.nvocab, allocator, io);
  defer tokenizer.deinit(allocator);
  var sampler = Sampler.init(rng_seed);

  generate(&transformer, &tokenizer, &sampler, steps, stdout, io);
}
