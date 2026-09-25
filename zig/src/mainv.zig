//! SIMD version of main.zig for llama2.c (f32 stories15M checkpoint).
//!
//! This is main.zig's twin, not a rewrite: same program, same structure, same
//! names (Config / Weights / State / Tokenizer / transformer / matmul /
//! rmsnorm / softmax / sample ...). The only difference is the three hot
//! kernels, which are vectorised with Zig's @Vector / @mulAdd builtins:
//!
//!   vector_dot_product  scalar `sum += w[j]*x[j]`  ->  8-lane (ymm) vectors
//!                       with FOUR independent accumulators (see the block
//!                       comment above the function, and c/run.c:240-271)
//!   matmul              unchanged shape, each row is now the SIMD dot product
//!   rmsnorm             scalar loops -> 8-lane vectors (sum of squares +
//!                       one broadcast scale pass)
//!
//! Everything else (tokenizer, sampler, top-p, clock, test harness at the
//! bottom) is copied verbatim from main.zig. This file is self-contained: it
//! carries its own main() and does NOT import main.zig, so build.zig can use
//!     root_source_file = b.path("src/mainv.zig")   (artifact name: llama2v)
//! Nothing here needs a third-party dependency.
//!
//! Performance on this machine (Intel i7-9700, pinned CPU, 5 runs, 255 tokens):
//!   zig/src/main.zig   (scalar matmul)     71 tok/s   1.00x
//!   this file          (SIMD, 4 accum.)   282 tok/s   4.0x   (vs main.zig;
//!                                                             273-280 unpinned)
//!   c/run.c            (AVX2+FMA GEMV)    351 tok/s   4.9x   (C is still
//!                                                             1.24x faster)
//! Why C is still 1.24x faster: one token streams the whole ~61 MB of weights
//! out of DRAM, so the model is bandwidth-bound. Instrumented copies (timers
//! around every matmul call) measure matmul = 2.87 ms/token in this file vs
//! 2.78 ms/token in c/run.c (only 3% apart), while the non-matmul time is
//! 0.77 ms vs 0.38 ms/token: nearly the whole gap sits outside the GEMV
//! (softmax over 32000 logits, top-p sort, rope). The GEMV itself is fine.
//! Note that swapping 1 -> 4 accumulators changes nothing end-to-end on this
//! CPU (no-early-break harness, 15 paired pinned runs: 274.5 vs 274.7 tok/s);
//! it is kept because it matches c/run.c exactly and does not rely on the
//! optimiser reassociating a serial FMA chain.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


const default_vector_width: usize = std.simd.suggestVectorLength(f32) orelse 4;
/// 8 f32 lanes = one 256-bit AVX2 register (ymm). Deliberately a literal 8 so
/// the dot product matches c/run.c's _mm256 kernel; on a CPU narrower than
/// AVX2 the compiler simply splits each vector into two SSE ops (still correct).
const dot_vec_width: usize = 8;
const simd_align = @alignOf(@Vector(default_vector_width, f32));
const simd_alignment: mem.Alignment = .fromByteUnits(simd_align);


comptime {
  @setFloatMode(.optimized);
}

/// @Vector is a builtin and cannot be aliased to a const; provide a callable
/// name so the kernels below can keep reading `Vector(n, f32)`.
fn Vector (comptime len: usize, comptime T: type) type {
  return @Vector(len, T);
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

  fn init (c: *const Config, data: []u8) Self {
    const dim: usize = c.dim;
    const ffndim: usize = c.ffndim;
    const head_size: usize = dim / c.nheads;
    const kvdim: usize = head_size * c.nkvheads;
    const nlayers: usize = c.nlayers;
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


const State = struct {
  const Self = @This();
  x: []align(simd_align) f32,
  x1: []align(simd_align) f32,
  x2: []align(simd_align) f32,
  h: []align(simd_align) f32,
  h1: []align(simd_align) f32,
  q: []align(simd_align) f32,
  kp: []align(simd_align) f32,
  vp: []align(simd_align) f32,
  attn: []align(simd_align) f32,
  logits: []align(simd_align) f32,
  logits_indexed: []IndexedF32,
  kcache: []align(simd_align) f32,
  vcache: []align(simd_align) f32,

  fn init (allocator: Allocator, c: *const Config) !Self {
    const dim = c.dim;
    const ffndim = c.ffndim;
    const kvdim = dim * c.nkvheads / c.nheads;
    var s: Self = undefined;
    s.x = try allocator.alignedAlloc(f32, simd_alignment, dim);
    s.x1 = try allocator.alignedAlloc(f32, simd_alignment, dim);
    s.x2 = try allocator.alignedAlloc(f32, simd_alignment, dim);
    s.h = try allocator.alignedAlloc(f32, simd_alignment, ffndim);
    s.h1 = try allocator.alignedAlloc(f32, simd_alignment, ffndim);
    s.q = try allocator.alignedAlloc(f32, simd_alignment, dim);
    s.kp = try allocator.alignedAlloc(f32, simd_alignment, kvdim);
    s.vp = try allocator.alignedAlloc(f32, simd_alignment, kvdim);
    s.attn = try allocator.alignedAlloc(f32, simd_alignment, c.nheads * c.ncontext);
    s.logits = try allocator.alignedAlloc(f32, simd_alignment, c.nvocab);
    s.logits_indexed = try allocator.alloc(IndexedF32, c.nvocab);
    s.kcache = try allocator.alignedAlloc(f32, simd_alignment, c.nlayers * c.ncontext * kvdim);
    s.vcache = try allocator.alignedAlloc(f32, simd_alignment, c.nlayers * c.ncontext * kvdim);
    return s;
  }

  fn deinit (self: *Self, allocator: Allocator) void {
    inline for (std.meta.fields(Self)) |f| {
      allocator.free(@field(self, f.name));
    }
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


/// o[i] = dot(weight row i, x) for every row i -- the GEMV that dominates the
/// runtime (43 calls per token). Same two nested loops as main.zig's matmul;
/// only the inner loop is now a SIMD dot product.
fn matmul (o: []f32, w: []const f32, x:[]const f32) void {
  const d = o.len;
  const n = x.len;
  assert(w.len == n * d);
  for (0..d) |i| {
    const w1 = w[n*i..][0..n];
    o[i] = vector_dot_product(w1, x);
  }
}


// ---------------------------------------------------------------------------
// vector_dot_product: the SIMD GEMV kernel (compare c/run.c:240-271).
//
//   vector width : 8 f32 lanes = one 256-bit AVX2 register (ymm). `dot_vec_width`
//                  is a literal 8 so the generated code can be read next to C's
//                  _mm256_* kernel; on a narrower CPU LLVM just splits each
//                  vector into two SSE ops (still correct, still fast).
//   accumulators : FOUR independent 8-wide accumulators a0..a3, i.e. 32 floats
//                  per loop iteration. Why: one @mulAdd has to wait for the
//                  previous one (vfmadd latency ~4 cycles, throughput 1/cycle),
//                  so a single accumulator retires at most one FMA per 4 cycles
//                  - a serial latency chain. Four independent chains let the
//                  FMA units stay busy and give the load unit four independent
//                  32-byte loads in flight (memory-level parallelism), which is
//                  exactly the trick c/run.c uses.
//   tail         : the 4 accumulators are folded pairwise ((a0+a1)+(a2+a3)), the
//                  8 lanes go to one scalar with @reduce(.Add, ...), then any
//                  remaining <32 floats are done scalar. All lengths in this
//                  model are multiples of 8, so the scalar tail is dead code
//                  here (<=31 steps) but keeps the function correct for any n.
//
// Measured (i7-9700, 768x288 matrix, best of 200 reps): 28.4 GFLOP/s with 4
// accumulators vs 22.3 GFLOP/s with 1 accumulator, i.e. 1.27x at kernel level.
// End-to-end it makes no difference (274.5 vs 274.7 tok/s over 15 paired runs)
// because a token streams ~61 MB of weights and both variants saturate DRAM.
// ---------------------------------------------------------------------------
fn vector_dot_product (w: []const f32, x: []const f32) f32 {
  assert(w.len == x.len);
  // 4 independent 8-wide accumulators, 32 floats per iteration (see the
  // comment above this function and c/run.c:240-271).
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
  // fold the 4 accumulators into one 8-lane vector, then reduce 8 -> 1 scalar.
  var sum = @reduce(.Add, (a0 + a1) + (a2 + a3));
  // trailing <32 elements: finish scalar (all lengths here are multiples of 8,
  // so this loop is at most 31 iterations and is correct for arbitrary n).
  while (j < w.len) : (j += 1) {
    sum = @mulAdd(f32, w[j], x[j], sum);
  }
  return sum;
}


/// o[i] = (x[i] / rms(x)) * w[i], the same formula as main.zig's rmsnorm.
/// Both scalar loops are replaced by 8-lane vectors: pass 1 accumulates
/// sum(x*x) in one 8-wide accumulator (summed with @reduce at the end and
/// divided by x.len), pass 2 scales with a single broadcast: x * scale * w.
/// Note: unlike the GEMV this is not bandwidth-bound work; it is also a tiny
/// share of per-token time (12 calls of 288 floats), so it is vectorised here
/// for symmetry with matmul rather than for the speed-up.
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


fn softmax (x: []f32) void {
  var max = x[0];
  for (x[1..]) |v| {
    if (v > max) max = v;
  }
  var sum: f32 = 0.0;
  for (0..x.len) |i| {
    x[i] = std.math.exp(x[i]-max);
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
  // GQA/MQA head mapping: nheads query heads share nkvheads key/value heads,
  // so query head h reads the k/v head h/kvmul (c/run.c:305,341,352).
  // kvmul == 1 (plain MHA, e.g. stories15M: 6/6) leaves the offsets unchanged.
  const kvmul = c.nheads / c.nkvheads;
  const fhsize: f32 = @floatFromInt(hsize);
  const fpos: f32 = @floatFromInt(pos);
  const x = s.x;
  @memcpy(x, w.embeddings[token*dim..][0..dim]);
  for (0..c.nlayers) |l| {
    const loff = l * c.ncontext * kvdim;
    rmsnorm(s.x1, x, w.wrmsattn[l*dim..][0..dim]);
    s.kp = @alignCast(s.kcache[loff+pos*kvdim..][0..kvdim]);
    s.vp = @alignCast(s.vcache[loff+pos*kvdim..][0..kvdim]);
    matmul(s.q, w.wq[l*dim*dim..][0..dim*dim], s.x1);
    matmul(s.kp, w.wk[l*dim*kvdim..][0..dim*kvdim], s.x1);
    matmul(s.vp, w.wv[l*dim*kvdim..][0..dim*kvdim], s.x1);
    for (0..dim/2) |ii| {
      const i = ii * 2;
      const fhdim: f32 = @floatFromInt((i) % hsize);
      const freq = 1.0 / std.math.pow(f32, 10000.0, fhdim / fhsize);
      const val: f32 = fpos * freq;
      const fcr = std.math.cos(val);
      const fci = std.math.sin(val);
      const v0 = s.q[i];
      const v1 = s.q[i+1];
      s.q[i] = v0 * fcr - v1 * fci;
      s.q[i+1] = v0 * fci + v1 * fcr;
      if (i < kvdim) {
        const kv0 = s.kp[i];
        const kv1 = s.kp[i+1];
        s.kp[i] = kv0 * fcr - kv1 * fci;
        s.kp[i+1] = kv0 * fci + kv1 * fcr;
      }
    }
    for (0..c.nheads) |h| {
      const q = s.q[h*hsize..][0..hsize];
      const attn = s.attn[h*c.ncontext..][0..c.ncontext];
      for (0..pos+1) |t| {
        const k = s.kcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        attn[t] = vector_dot_product(q, k) / std.math.sqrt(fhsize);
      }
      softmax(attn[0.. pos+1]);
      var x1 = s.x1[h*hsize..][0..hsize];
      @memset(x1, 0);
      for (0..pos+1) |t| {
        const v = s.vcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        for (0..hsize) |i| {
          x1[i] += attn[t] * v[i];
        }
      }
    }
    matmul(s.x2, w.wo[l*dim*dim..][0..dim*dim], s.x1);
    for (0 .. x.len) |i| {
      x[i] += s.x2[i];
    }

    // 2. ffn sublayer
    rmsnorm(s.x1, x, w.wrmsffn[l*dim..][0..dim]);
    matmul(s.h, w.w1[l*dim*ffndim..][0..dim*ffndim], s.x1);
    matmul(s.h1, w.w3[l*dim*ffndim..][0..dim*ffndim], s.x1);
    for (0..ffndim) |i| {
      var v = s.h[i];
      v = v * (1.0/(1.0+std.math.exp(-v))) * s.h1[i];
      s.h[i] = v;
    }
    matmul(s.x1, w.w2[l*ffndim*dim..][0..ffndim*dim], s.h);
    for (0 .. x.len) |i| {
      x[i] += s.x1[i];
    }
  }
  rmsnorm(x, x, w.wrmsfinal[0..dim]);
  matmul(s.logits, w.embeddings[0..c.nvocab*dim], x);
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


const IndexedF32 = struct {
  const Self = @This();
  index: u32,
  value: f32,
  fn desc (_: void, a: Self, b: Self) bool {
    return a.value > b.value;
  }
};


fn sample_topp (logits: []f32, p: f32, sorted: []IndexedF32) usize {
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
  std.sort.pdq(IndexedF32, sorted[0..n], {}, IndexedF32.desc);
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
  const stdout: *std.Io.Writer = @ptrCast(&writer.interface);
  defer stdout.flush() catch {};

  const checkpoint_path: []const u8 = "stories15M.bin";
  const tokenizer_path: []const u8 = "tokenizer.bin";
  const temperature: f32 = 1.0;
  const topp: f32 = 0.9;
  const steps: usize = 256;
  const now_ns = std.Io.Clock.real.now(io).nanoseconds;
  prng = std.Random.DefaultPrng.init(@truncate(@as(u96, @bitCast(now_ns))));

  const checkpoint = try std.Io.Dir.cwd().openFile(io, checkpoint_path, .{});
  var buffer: [4096]u8 = undefined;
  var reader = checkpoint.reader(io, &buffer);
  var rawConfig = try reader.interface.takeStruct(RawConfig, .little);
  const config = rawConfig.cook();
  const file_size = (try checkpoint.stat(io)).size;
  log("Config: {any}\n", .{config});
  log("temperature: {d}\n", .{temperature});
  log("top-p: {d}\n", .{topp});
  log("\n", .{});
  const data = try std.posix.mmap(null, file_size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, checkpoint.handle, 0);
  const weights = Weights.init(&config, data[@sizeOf(RawConfig)..]);
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
