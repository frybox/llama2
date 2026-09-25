const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


var GS: usize = 0;


// ---------------------------------------------------------------------------
// Runtime CPU feature detection (CPUID), mirroring c/runq.c runq_cpu().
//
// This toolchain's stage2-c backend provides `zig_x86_cpuid` (the same extern
// the compiler's std/zig/system/x86.zig declares for detectNative), which
// lowers to a real CPUID instruction. The avx2 GEMV kernel is the C
// translation in src/gemv.c (bit-identical to c/runq.c's
// matmul_avx2_impl), specialized for GS == 32 and only selected when the
// CPU has AVX2; otherwise matmul() falls back to the scalar path, so
// mainqv is correct on any x86-64 CPU.
//
//   level 0 "scalar" : the original mainq matmul (exact integer group sums)
//   level 1 "avx2"   : C vpmaddwd kernel from src/gemv.c (bit-identical to
//                      c/runq.c's matmul_avx2_impl)
//
// Override with env MAINQV_KERNEL=scalar|avx2 (mirrors RUNQ_KERNEL in
// c/runq.c); the chosen kernel is printed to stderr at startup.
// ---------------------------------------------------------------------------
extern fn zig_x86_cpuid(leaf_id: u32, subid: u32, eax: *u32, ebx: *u32, ecx: *u32, edx: *u32) callconv(.c) void;

const CpuCaps = struct {
  has_fma: bool,
  has_avx2: bool,
  level: u32,
  kernel: []const u8,
};

fn cpuidAll(leaf: u32, sub: u32) struct { eax: u32, ebx: u32, ecx: u32, edx: u32 } {
  var eax: u32 = 0;
  var ebx: u32 = 0;
  var ecx: u32 = 0;
  var edx: u32 = 0;
  zig_x86_cpuid(leaf, sub, &eax, &ebx, &ecx, &edx);
  return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

var g_cpu: CpuCaps = .{ .has_fma = false, .has_avx2 = false, .level = 0, .kernel = "scalar" };

fn detectCpu(env: *std.process.Environ.Map) void {
  var caps = CpuCaps{ .has_fma = false, .has_avx2 = false, .level = 0, .kernel = "scalar" };
  const id1 = cpuidAll(1, 0);
  caps.has_fma = (id1.ecx >> 12) & 1 != 0;        // CPUID(1).ECX[12] = FMA
  const max_leaf = cpuidAll(0, 0).eax;
  if (max_leaf >= 7) {
    const id7 = cpuidAll(7, 0);
    caps.has_avx2 = (id7.ebx >> 5) & 1 != 0;     // CPUID(7,0).EBX[5] = AVX2
  }
  // kernel level selection: 0 scalar < 1 avx2 (8-lane int8 GEMV).
  var level: u32 = 0;
  var kernel: []const u8 = "scalar";
  if (caps.has_avx2) {
    level = 1;
    kernel = "avx2";
  }
  if (GS != 32) {                                 // kernel is specialized for GS==32
    level = 0;
    kernel = "scalar";
  }
  // manual override, like RUNQ_KERNEL in c/runq.c
  if (env.get("MAINQV_KERNEL")) |e| {
    if (mem.eql(u8, e, "scalar")) {
      level = 0;
      kernel = "scalar";
    } else if (mem.eql(u8, e, "avx2")) {
      level = 1;
      kernel = "avx2";
    } else {
      std.debug.print("MAINQV_KERNEL: unknown value '{s}', ignoring\n", .{e});
    }
  }
  caps.level = level;
  caps.kernel = kernel;
  g_cpu = caps;
}

fn printCpu() void {
  std.debug.print("cpu: avx2={d} fma={d}  -> matmul kernel: {s} (GS={d})\n", .{
    @as(u8, @intFromBool(g_cpu.has_avx2)),
    @as(u8, @intFromBool(g_cpu.has_fma)),
    g_cpu.kernel,
    GS,
  });
}

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
  byte_pieces: [512]u8,

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

  /// C: sscanf(piece, "<0x%02hhX>", &bytev) == 1
  fn parseBytePiece(piece: []const u8) ?u8 {
    if (piece.len != 6 or piece[0] != '<' or piece[1] != '0'
     or piece[2] != 'x' or piece[5] != '>') return null;
    return std.fmt.parseInt(u8, piece[3..5], 16) catch null;
  }
};


// ---------------------------------------------------------------------------
// Shared vector plumbing (same idioms as mainv.zig): 8-lane f32 vectors =
// one 256-bit AVX2 ymm. @Vector is a builtin and cannot be const-aliased, so
// provide a callable name for `Vector(n, T)` in the kernels below.
// ---------------------------------------------------------------------------
const default_vector_width: usize = std.simd.suggestVectorLength(f32) orelse 4;
const dot_vec_width: usize = 8;
const exp_vec_width: usize = 8;

fn Vector (comptime len: usize, comptime T: type) type {
  return @Vector(len, T);
}

// rope angle table for the position currently being decoded: filled once per
// token by transformer(), read by every layer (mainq recomputed pow+cos+sin
// nlayers * dim/2 = 864 times per token for nothing).
const rope_table_len: usize = 512;
var rope_cos: [rope_table_len]f32 = undefined;
var rope_sin: [rope_table_len]f32 = undefined;

// vector_exp8: exp() for 8 lanes at once.
//   why  : softmax over the 32000 logits and the swiglu activation call
//          exp() thousands of times per token; each scalar std.math.exp is
//          an out-of-line libm call. Same kernel as mainv.zig:
//          exp(x) = 2^k * 2^r, k = round(x * log2(e)) split off with
//          @floor(t + 0.5), r = x - k*ln2 (|r| <= ln2/2), 2^k assembled
//          from the f32 exponent field ((k + 127) << 23, exact for
//          -126 <= k <= 127), 2^r = degree-5 Taylor in Horner form.
//   error: max relative error ~3.3e-6 over x in [-120, 0] (dropped r^6/720
//          term). Downstream effect: softmax probabilities differ by at most
//          ~1.6e-7, which does not change greedy decoding; stochastic runs
//          differ from a libm-based softmax only in the tail of the cdf.
//   range: inputs clamped to [-87, 88], where the (k+127)<<23 trick still
//          resolves denormals / finite f32.
inline fn vector_exp8 (a: @Vector(exp_vec_width, f32)) @Vector(exp_vec_width, f32) {
  const V = @Vector(exp_vec_width, f32);
  const x = @min(@max(a, @as(V, @splat(-87.0))), @as(V, @splat(88.0)));
  // k = round(x * log2 e), r = x - k * ln2  (|r| <= ln2/2)
  const k = @floor(@mulAdd(V, x, @splat(1.4426950408889634), @splat(0.5)));
  const r = @mulAdd(V, k, @splat(-0.6931471805599453), x);
  // exp(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 (Horner, 5 FMAs)
  var p: V = @splat(1.0 / 120.0);
  p = @mulAdd(V, p, r, @splat(1.0 / 24.0));
  p = @mulAdd(V, p, r, @splat(1.0 / 6.0));
  p = @mulAdd(V, p, r, @splat(0.5));
  p = @mulAdd(V, p, r, @splat(1.0));
  p = @mulAdd(V, p, r, @splat(1.0));
  // scale by 2^k, built as the exponent field of a f32
  const kint: @Vector(exp_vec_width, i32) = @intFromFloat(k);
  const scale: V = @bitCast((kint + @as(@Vector(exp_vec_width, i32), @splat(127))) <<
                            @as(@Vector(exp_vec_width, u5), @splat(23)));
  return p * scale;
}

// ---------------------------------------------------------------------------
// int8 GEMV, dispatched at runtime by detectCpu() (see the CPUID block at
// the top of this file). Both kernels compute, per group, the EXACT integer
// sum of the GS-wide group of w.q * x.q products (int32, no overflow:
// 32*127*127 = 516128 < 2^31), then fold per group, j ascending, per row i:
//
//   v  += float(iv) * w.s[i,j] * x.s[j]
//
// The avx2 kernel is qv_matmul_avx2() from src/gemv.c - the C translation of
// c/runq.c's matmul_avx2_impl (vpmaddwd), so this path is bit-identical to
// c/runq's matmul. The scalar path keeps its original left-to-right float
// accumulation; the C kernel accumulates 4 groups per 4-lane vector add, so
// the two paths may disagree by ~1 ulp (same exact integer sums).
// ---------------------------------------------------------------------------
extern fn qv_matmul_avx2 (o: [*]f32, wq: [*]i8, ws: [*]f32, xq: [*]i8,
                          xs: [*]f32, n: c_int, d: c_int) callconv(.c) void;

fn matmul (o: []f32, w: QuantizedTensor, x: QuantizedTensor) void {
  if (g_cpu.level == 1) {
    qv_matmul_avx2(o.ptr, w.q.ptr, w.s.ptr, x.q.ptr, x.s.ptr,
      @intCast(x.q.len), @intCast(o.len));
  } else {
    matmul_scalar(o, w, x);
  }
}


fn matmul_scalar (o: []f32, w: QuantizedTensor, x: QuantizedTensor) void {
  for (0..o.len) |i| {
    var v: f32 = 0.0;
    const ii: usize = i * x.q.len;
    for (0..x.s.len) |j| {
      const jj = j*GS;
      var iv: i32 = 0;
      for (0..GS) |k| {
        iv += @as(i32, w.q[ii+jj+k]) * @as(i32, x.q[jj+k]);
      }
      v += @as(f32, @floatFromInt(iv)) * w.s[i * x.s.len + j] * x.s[j];
    }
    o[i] = v;
  }
}





// vector_dot_product: 8-lane f32 dot with FOUR independent accumulators
// (32 floats per iteration), pairwise tail fold, then the < 32 leftovers
// scalar. Same kernel as mainv.zig vector_dot_product: one @mulAdd has to
// wait for the previous one (FMA latency), so a single accumulator retires
// at most one FMA per ~4 cycles - four independent chains keep the FMA
// units busy and give the load unit four independent 32-byte loads in
// flight. All GEMV lengths in this model (288, 768) are multiples of 32, so
// the scalar tail is dead there; it keeps the function correct for the
// attn_qk dots (hsize = 48).
fn vector_dot_product (w: []const f32, x: []const f32) f32 {
  assert(w.len == x.len);
  var a0: Vector(dot_vec_width, f32) = @splat(0.0);
  var a1: Vector(dot_vec_width, f32) = @splat(0.0);
  var a2: Vector(dot_vec_width, f32) = @splat(0.0);
  var a3: Vector(dot_vec_width, f32) = @splat(0.0);
  var j: usize = 0;
  while (j + 4*dot_vec_width <= w.len) : (j += 4*dot_vec_width) {
    a0 = @mulAdd(Vector(dot_vec_width, f32), w[j..][0..dot_vec_width].*, x[j..][0..dot_vec_width].*, a0);
    a1 = @mulAdd(Vector(dot_vec_width, f32), w[j+dot_vec_width..][0..dot_vec_width].*, x[j+dot_vec_width..][0..dot_vec_width].*, a1);
    a2 = @mulAdd(Vector(dot_vec_width, f32), w[j+2*dot_vec_width..][0..dot_vec_width].*, x[j+2*dot_vec_width..][0..dot_vec_width].*, a2);
    a3 = @mulAdd(Vector(dot_vec_width, f32), w[j+3*dot_vec_width..][0..dot_vec_width].*, x[j+3*dot_vec_width..][0..dot_vec_width].*, a3);
  }
  const vsum = (a0 + a1) + (a2 + a3);
  var sum: f32 = undefined;
  if (j == w.len) {
    sum = @reduce(.Add, vsum);
  } else {
    var t0: Vector(dot_vec_width, f32) = @splat(0.0);
    while (j + dot_vec_width <= w.len) : (j += dot_vec_width) {
      t0 = @mulAdd(Vector(dot_vec_width, f32), w[j..][0..dot_vec_width].*, x[j..][0..dot_vec_width].*, t0);
    }
    sum = @reduce(.Add, vsum + t0);
  }
  while (j < w.len) : (j += 1) {
    sum = @mulAdd(f32, w[j], x[j], sum);
  }
  return sum;
}


/// o[i] = (x[i] / rms(x)) * w[i]. Same formula as mainq's rmsnorm; both
/// scalar loops are 8-lane vectors: pass 1 accumulates sum(x*x) in one
/// 8-wide accumulator (reduced at the end, divided by x.len), pass 2 scales
/// with one broadcast: x * scale * w. (Same kernel as mainv.zig rmsnorm;
/// here, unlike the GEMV, it is a tiny share of per-token time - vectorised
/// for symmetry, not for the speed-up.)
fn rmsnorm(o: []f32, x: []f32, w: []f32) void {
  assert(o.len == x.len);
  assert(x.len == w.len);
  const vecwidth = default_vector_width;
  const nvectors = x.len / vecwidth;
  const remstart = nvectors * vecwidth;
  var product: @Vector(vecwidth, f32) = @splat(0.0);
  for (0..nvectors) |i| {
    const xvec: @Vector(vecwidth, f32) = x[vecwidth*i..][0..vecwidth].*;
    product = @mulAdd(@Vector(vecwidth, f32), xvec, xvec, product);
  }
  var sum = @reduce(.Add, product);
  for (remstart..x.len) |i| {
    sum = @mulAdd(f32, x[i], x[i], sum);
  }
  sum /= @floatFromInt(x.len);
  sum += 1e-5;
  sum = 1.0 / std.math.sqrt(sum);
  const scale: @Vector(vecwidth, f32) = @splat(sum);
  for (0..nvectors) |i| {
    const xvec: @Vector(vecwidth, f32) = x[i*vecwidth..][0..vecwidth].*;
    const wvec: @Vector(vecwidth, f32) = w[i*vecwidth..][0..vecwidth].*;
    o[i*vecwidth..][0..vecwidth].* = xvec * scale * wvec;
  }
  for (remstart..o.len) |i| {
    o[i] = x[i] * sum * w[i];
  }
}


/// softmax over x, in place: pass 1 finds the maximum, pass 2 writes
/// exp(x-max) and accumulates the sum, pass 3 divides every element by it.
/// Same three passes as mainq; pass 2 changed: the per-element scalar
/// std.math.exp / @exp (an out-of-line libm call, 32000 per token) becomes
/// vector_exp8 over 8 lanes, and the running sum stays an 8-lane vector
/// with one partial sum per lane, reduced once after the loop. (Same kernel
/// as mainv.zig softmax.)
fn softmax (x: []f32) void {
  const V = @Vector(exp_vec_width, f32);
  var vm0: V = @splat(x[0]);
  var vm1: V = vm0;
  var vm2: V = vm0;
  var vm3: V = vm0;
  var i: usize = 0;
  const W = exp_vec_width;
  while (i + 4 * W <= x.len) : (i += 4 * W) {
    vm0 = @max(vm0, @as(V, x[i..][0..W].*));
    vm1 = @max(vm1, @as(V, x[i + W..][0..W].*));
    vm2 = @max(vm2, @as(V, x[i + 2 * W..][0..W].*));
    vm3 = @max(vm3, @as(V, x[i + 3 * W..][0..W].*));
  }
  while (i + W <= x.len) : (i += W) {
    vm0 = @max(vm0, @as(V, x[i..][0..W].*));
  }
  var max: f32 = @reduce(.Max, @max(@max(vm0, vm1), @max(vm2, vm3)));
  while (i < x.len) : (i += 1) {
    if (x[i] > max) max = x[i];
  }
  const maxv: V = @splat(max);
  var vs0: V = @splat(0.0);
  var vs1: V = vs0;
  var vs2: V = vs0;
  var vs3: V = vs0;
  i = 0;
  while (i + 4 * W <= x.len) : (i += 4 * W) {
    const e0 = vector_exp8(@as(V, x[i..][0..W].*) - maxv);
    const e1 = vector_exp8(@as(V, x[i + W..][0..W].*) - maxv);
    const e2 = vector_exp8(@as(V, x[i + 2 * W..][0..W].*) - maxv);
    const e3 = vector_exp8(@as(V, x[i + 3 * W..][0..W].*) - maxv);
    x[i..][0..W].* = e0;
    x[i + W..][0..W].* = e1;
    x[i + 2 * W..][0..W].* = e2;
    x[i + 3 * W..][0..W].* = e3;
    vs0 += e0;
    vs1 += e1;
    vs2 += e2;
    vs3 += e3;
  }
  while (i + W <= x.len) : (i += W) {
    const e = vector_exp8(@as(V, x[i..][0..W].*) - maxv);
    x[i..][0..W].* = e;
    vs0 += e;
  }
  var sum: f32 = @reduce(.Add, (vs0 + vs1) + (vs2 + vs3));
  while (i < x.len) : (i += 1) {
    x[i] = std.math.exp(x[i] - max);
    sum += x[i];
  }
  const inv: f32 = 1.0 / sum;
  for (0..x.len) |j| {
    x[j] *= inv;
  }
}



fn transformer (token: usize, pos: usize, c: *const Config, s: *State, w: *const Weights) void {
  const dim = c.dim;
  const ffndim = c.ffndim;
  const hsize = dim / c.nheads;
  const kvdim = hsize * c.nkvheads;
  // GQA/MQA head mapping: query head h reads the k/v head h/kvmul.
  const kvmul = c.nheads / c.nkvheads;
  const fhsize: f32 = @floatFromInt(hsize);
  const fpos: f32 = @floatFromInt(pos);
  const x = s.x;
  @memcpy(x, w.embeddings[token*dim..][0..dim]);
  // rope angles depend only on (pos, i), identical in all nlayers layers:
  // fill the table once per token (144 pow+cos+sin triples) instead of
  // nlayers * dim/2 = 864 per token like mainq.
  assert(dim / 2 <= rope_table_len);
  for (0..dim/2) |ii| {
    const i = ii * 2;
    const fhdim: f32 = @floatFromInt(i % hsize);
    const freq = 1.0 / std.math.pow(f32, 10000.0, fhdim / fhsize);
    const val: f32 = fpos * freq;
    rope_cos[ii] = std.math.cos(val);
    rope_sin[ii] = std.math.sin(val);
  }
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
      const fcr = rope_cos[ii];
      const fci = rope_sin[ii];
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
        attn[t] = vector_dot_product(q, k) / std.math.sqrt(fhsize);
      }
      softmax(attn[0..pos+1]);
      var x1 = s.x1[h*hsize..][0..hsize];
      @memset(x1, 0);
      // attn @ v: x1[i] = sum_t attn[t] * v[t][i] -- a rank-1 update per
      // (layer, head, position), written as explicit 8-lane @mulAdd like
      // mainv (the scalar form is scalarised to one addss per element).
      const V = @Vector(dot_vec_width, f32);
      for (0..pos+1) |t| {
        v = s.vcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        const av: V = @splat(attn[t]);
        var i: usize = 0;
        while (i + dot_vec_width <= hsize) : (i += dot_vec_width) {
          const xv: V = x1[i..][0..dot_vec_width].*;
          const vv: V = v[i..][0..dot_vec_width].*;
          x1[i..][0..dot_vec_width].* = @mulAdd(V, av, vv, xv);
        }
        while (i < hsize) : (i += 1) {
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
    // swiglu: h[i] = h[i] * silu(h[i]) * h1[i]. The per-element std.math.exp
    // (an out-of-line libm call, 4*dim per layer) becomes vector_exp8 over
    // 8 lanes; 8 divides 4*dim = 1152 exactly, tail keeps any ffndim ok.
    const V = @Vector(exp_vec_width, f32);
    var hi: usize = 0;
    while (hi + exp_vec_width <= ffndim) : (hi += exp_vec_width) {
      const hv: V = s.h[hi..][0..exp_vec_width].*;
      const h1v: V = s.h1[hi..][0..exp_vec_width].*;
      const one: V = @splat(1.0);
      const sig = one / (one + vector_exp8(-hv));
      s.h[hi..][0..exp_vec_width].* = hv * sig * h1v;
    }
    while (hi < ffndim) : (hi += 1) {
      var val = s.h[hi];
      val = val * (1.0/(1.0+std.math.exp(-val))) * s.h1[hi];
      s.h[hi] = val;
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
  // Deterministic PRNG seed (mainq seeds from the wall clock): the matmul
  // kernels are bit-identical, but a fixed seed makes the stochastic
  // top-p sampling reproducible too, so scalar vs avx2 produce the exact
  // same text. Override with env MAINQV_SEED=<int> to vary it.
  var seed: u64 = 0x9e3779b97f4a7c15; // 2^64 * phi
  if (init.environ_map.get("MAINQV_SEED")) |s| {
    seed = std.fmt.parseInt(u64, s, 10) catch seed;
  }
  prng = std.Random.DefaultPrng.init(seed);


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
  detectCpu(init.environ_map);
  printCpu();
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
    const piece = tokenizer.decode(@intCast(token), @intCast(next));
    try stdout.print("{s}", .{piece});
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
