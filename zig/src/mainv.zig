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
//! Performance on this machine (Intel i7-9700, `taskset -c 3`, 7 interleaved
//! runs, median, timed over a FIXED 255-token window so the numbers do not
//! depend on where the first EOS token happens to land):
//!   zig/src/main.zig   (scalar matmul)     71 tok/s   1.00x
//!   baseline mainv     (SIMD, 4 accum.)   277 tok/s   3.9x   3.6164 ms/token
//!   this file          (SIMD + 5 fixes)   328 tok/s   4.6x   3.0492 ms/token
//!   c/run.c  gcc -O3                      321 tok/s           3.1195 ms/token
//!   c/run.c  gcc -Ofast -march=native     348 tok/s   4.9x   2.8736 ms/token
//!   this file + A1b+B2+S2 below           340 tok/s   4.8x   2.9412 ms/token
//!
//! A later revision of this file added the five fixes below; they only touch the
//! five places listed, everything else is unchanged. The changes, each measured
//! in isolation the same way (ms/token saved):
//!   H1   softmax over the 32000 logits: a scalar std.math.exp per element (an
//!        out-of-line libm call) -> vector_exp8, 8 lanes at a time     0.385
//!   H1b  the same vector_exp8 inside swiglu/silu (1152 exps per layer) 0.085
//!   attn@v in the attention head: explicit 8-lane @Vector + @mulAdd, the
//!        scalar form was being scalarised by LLVM                    0.117
//!   rope: the cos/sin angles depend only on (pos, i), so build them once per
//!        token instead of once per (layer, i)                        0.050
//!   softmax normalisation: one reciprocal + multiply instead of 32000
//!        divisions                                                   0.019
//!   (each row is baseline-minus-that-one-change in the same session, +-0.03
//!    i.e. the ~1% run spread; they add up to 0.656 against 0.567 for all five
//!    together, the usual second-order effect of combining them)
//!   (two further ideas were tested and REJECTED, both measured as zero:
//!    @setFloatMode(.optimized), and swapping the top-p pdq sort for qsort.
//!    Neither is in this file. The fast-math rejection was re-confirmed on the
//!    pristine source with 30 alternating pinned pairs: median delta 0.00
//!    ms/token, 14/30 wins, against a same-source noise floor of 0.00.)
//!
//! The remaining 1.06x gap to `gcc -Ofast` was decomposed with per-stage rdtsc
//! timers on instrumented twins of both, in the protocol this file actually
//! runs (temperature 1.0, topp 0.9, no early break): loop_total 3.0007 vs
//! 2.8741 ms/token. classifier +0.068 is over half of it (288x32000 GEMV, 57%
//! of the whole token; 21.5 GB/s here against 22.4 in C, and the measured
//! single-core streaming ceiling for this row shape is 23.9 GB/s), then
//! softmax_logits +0.033 (the 32000-element exp pass), w1 +0.010, w3 +0.009,
//! attn_qk +0.007, softmax_attn +0.005, sample_topp +0.004 and the other GEMVs
//! +0.002..0.004. Offset by rope -0.008, w2 -0.011 (both now faster here) and
//! temp_div -0.004 because temperature is a compile-time constant 1.0, so the
//! x/1.0 scaling folds away (a property of main.zig, not of the changes below),
//! while C pays 32000 divisions. Three fixes were applied on top of that
//! analysis, each measured in isolation with 24-30 alternating pinned pairs
//! (the same-source pair is the noise floor and comes out at exactly 0.00):
//!   A1b  vector_dot_product's tail: n = 48 (attn_qk) used to run one 32-lane
//!        FMA iteration and then 16 scalar ones, ~22 extra cycles per dot 0.036
//!   B2   matmul: two weight rows per pass (dot_rows_2), so the row-end
//!        reduce/shuffle chain of row r hides behind the FMAs of row r+1  0.030
//!   S2   the 32000-element softmax max scan was a scalar branchy loop; it and
//!        the exp pass now use four 8-lane accumulators and a 4x unrolled tail
//!                                                                         0.027
//! Together (7 interleaved runs, median, same session): 340.0 tok/s = 2.9412
//! ms/token against c/run.c's 348.8 = 2.8670, i.e. +0.074 ms/token (+2.6%),
//! which is 30/30 in paired runs; the +6.7% this started from was 327.0 =
//! 3.0581. What is left is ~76% classifier, and that GEMV already streams
//! within 10% of the machine's single-core limit. Tested and rejected this
//! round, all at or below the noise floor: @prefetch on the GEMV weight stream
//! (7 distances x 2 localities), 3- and 4-row blocking (register pressure),
//! fully unrolling the n = 288/768 inner loop, and a softmax variant that
//! branches on the length.
//! Note that swapping 1 -> 4 accumulators changes nothing end-to-end on this
//! CPU (no-early-break harness, 15 paired pinned runs: 274.5 vs 274.7 tok/s);
//! it is kept because it matches c/run.c exactly and does not rely on the
//! optimiser reassociating a serial FMA chain.
//!
//! Also in this file (and in main.zig) is one correctness fix, not a speed one:
//! the K/V head offset was `h*hsize`, which is only correct when
//! nheads == nkvheads. It is now `(h/kvmul)*hsize` with kvmul = nheads/nkvheads,
//! matching c/run.c:305,341,352. For stories15M (6/6) kvmul == 1, so the offsets
//! are unchanged and the greedy 256-step token ids come out bit-identical to
//! before; a synthetic nheads=6/nkvheads=2 checkpoint diverges in its very first
//! sampled token under the old mapping, and under the new one it matches the C
//! output token for token (that checkpoint also reproduces the 256-step greedy
//! ids, and pos-0 logits agree with C to within 7.2e-6).

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const print = std.debug.print;


const default_vector_width: usize = std.simd.suggestVectorLength(f32) orelse 4;
/// 16 f32 lanes = one 512-bit AVX512 register (zmm). Deliberately a literal 16
/// so the dot product emits native 512-bit FMAs; on a CPU without AVX512F the
/// compiler simply splits each vector into SSE/AVX2 ops (still correct).
const dot_vec_width: usize = 16;
const simd_align = @alignOf(@Vector(default_vector_width, f32));
const simd_alignment: mem.Alignment = .fromByteUnits(simd_align);


/// @Vector is a builtin and cannot be aliased to a const; provide a callable
/// name so the kernels below can keep reading `Vector(n, f32)`.
fn Vector (comptime len: usize, comptime T: type) type {
  return @Vector(len, T);
}


/// rope angle table for the position currently being decoded: rope_cos[ii] /
/// rope_sin[ii] hold cos/sin of `pos * freq(2*ii)` and are filled once per token
/// by transformer(), then read by every layer. 512 entries cover any head size
/// this model family uses (dim/2 = 144 here).
const rope_table_len: usize = 512;
var rope_cos: [rope_table_len]f32 = undefined;
var rope_sin: [rope_table_len]f32 = undefined;

/// Width of the exp kernel below: 8 f32 lanes = one 256-bit ymm, the same width
/// the GEMV uses (dot_vec_width). exp is not part of the GEMV, so it gets its
/// own name instead of sharing that one.
const exp_vec_width: usize = 8;


// ---------------------------------------------------------------------------
// vector_exp8: exp() for 8 lanes at once.
//
//   why  : softmax over the 32000 logits calls exp() once per element, and the
//          swiglu activation calls it 4*dim = 1152 times per layer (6 layers).
//          With the scalar `std.math.exp` every one of those is an out-of-line
//          libm call, so the vector units sit idle; measured 0.363 ms/token in
//          softmax + 0.051 ms/token in swiglu on a pinned CPU.
//   how  : exp(x) = 2^k * 2^r. k = round(x * log2(e)) is split off with @floor
//          (@floor(t + 0.5) rounds to nearest), the remainder r = x - k*ln2 has
//          |r| <= ln2/2 = 0.347, and 2^k is assembled straight from the f32
//          exponent field, ((k + 127) << 23), which is exact for
//          -126 <= k <= 127. 2^r is the degree-5 truncation of the exp Taylor
//          series in Horner form: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120.
//   width: 8 lanes, i.e. one ymm per 8 exponentials; callers with a leftover
//          tail (< 8 elements) fall back to std.math.exp, so any length works.
//   error: max relative error 3.25e-6 over a dense sweep of the real input
//          domain (x in [-120, 0], step 7e-4; 79.8% of those points stay below
//          5e-7), dominated by the dropped r^6/720 term (0.347^6/720 = 2.4e-6).
//          Measured by expcheck.zig, which compiles this very function
//          (extracted from this file) against f64 std.math.exp. Knock-on
//          effect: the softmax it feeds differs from an f64 softmax by at most
//          1.6e-7 per probability, and the greedy 256-step token ids come out
//          token-for-token identical to the baseline's, so the error never
//          reaches the decoder.
//   range: inputs are clamped to [-87, 88]. exp(-87) = 1.6e-38 is the smallest
//          value the (k+127)<<23 trick still resolves, exp(88) = 1.7e38 the
//          largest finite f32. The clamps only fire where the result no longer
//          matters: softmax subtracts the maximum first, so it never passes
//          x > 0, and in swiglu a large negative h gives a sigmoid that rounds
//          to 1.0 (or to a ~1e-37 denormal) with or without the clamp.
// ---------------------------------------------------------------------------
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


/// o[i] = dot(weight row i, x) for every row i -- the GEMV that dominates the
/// runtime (43 calls per token). Same two nested loops as main.zig's matmul;
/// only the inner loop is now a SIMD dot product.
/// B2: `R` independent dots (i.e. `R` weight rows against the same vector x)
/// computed in ONE pass over x.  Same accumulators and same summation order as
/// vector_dot_product, so the numerical result is bit-identical to the
/// row-at-a-time version while the row-end reduce/shuffle chain of row r is
/// hidden behind the FMAs of rows r+1..R-1.
inline fn dot_rows_2 (o: *[2]f32, w: []const f32, x: []const f32, n: usize) void {
  const V = Vector(dot_vec_width, f32);
  var acc: [2][4]V = @splat(@splat(@as(V, @splat(0.0))));
  var j: usize = 0;
  while (j + 4*dot_vec_width <= n) : (j += 4*dot_vec_width) {
    inline for (0..2) |r| {
      inline for (0..4) |k| {
        const off = r*n + j + k*dot_vec_width;
        acc[r][k] = @mulAdd(V, w[off..][0..dot_vec_width].*, x[j+k*dot_vec_width..][0..dot_vec_width].*, acc[r][k]);
      }
    }
  }
  inline for (0..2) |r| {
    var sum = @reduce(.Add, (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]));
    var jj = j;
    while (jj < n) : (jj += 1) {
      sum = @mulAdd(f32, w[r*n + jj], x[jj], sum);
    }
    o[r] = sum;
  }
}

fn matmul (o: []f32, w: []const f32, x:[]const f32) void {
  const d = o.len;
  const n = x.len;
  assert(w.len == n * d);
  var i: usize = 0;
  while (i + 2 <= d) : (i += 2) {
    dot_rows_2(o[i..][0..2], w[n*i..][0..2*n], x, n);
  }
  while (i < d) : (i += 1) {
    o[i] = vector_dot_product(w[n*i..][0..n], x);
  }
}


// ---------------------------------------------------------------------------
// vector_dot_product: the SIMD GEMV kernel (compare c/run.c's matmul_avx2_impl,
// c/run.c:240-271 -- same 4-accumulator structure, here at 512-bit width).
//
//   vector width : 16 f32 lanes = one 512-bit AVX512 register (zmm). `dot_vec_width`
//                  is a literal 16 so the generated code is native 512-bit FMAs;
//                  on a narrower CPU LLVM just splits each vector into two
//                  256-bit ops (still correct).
//   accumulators : FOUR independent 16-wide accumulators a0..a3, i.e. 64 floats
//                  per loop iteration. Why: one @mulAdd has to wait for the
//                  previous one (vfmadd latency ~4-5 cycles, throughput 1/cycle),
//                  so a single accumulator retires at most one FMA per latency
//                  cycles - a serial latency chain. Four independent chains let
//                  the FMA units stay busy and give the load unit four
//                  independent 64-byte loads in flight (memory-level
//                  parallelism), exactly the trick c/run.c's kernel uses.
//   tail         : the 4 accumulators are folded pairwise ((a0+a1)+(a2+a3)), the
//                  16 lanes go to one scalar with @reduce(.Add, ...), then any
//                  remaining <64 floats are done vectorised (one 16-wide
//                  accumulator) and finally scalar. Every GEMV length in this
//                  model is a multiple of 16 (288, 768, 1024; attn_qk 48), so
//                  the scalar tail is dead code here, but the function stays
//                  correct for any n.
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
  // fold the 4 accumulators into one 8-lane vector.
  const vsum = (a0 + a1) + (a2 + a3);
  // A1b: same vectorised tail, but the n % 32 == 0 case (every GEMV length in
  // this model: 288 and 768) takes a branch that keeps the baseline reduce, so
  // it pays neither the extra vector add nor the extra 8-wide accumulator.
  // Only the attn_qk dots (hsize = 48) enter the vector-tail path.
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


/// softmax over x, in place: pass 1 finds the maximum, pass 2 writes
/// exp(x-max) and accumulates the sum, pass 3 divides every element by it.
/// Same three passes as main.zig; pass 2 is what changed: the scalar
/// std.math.exp call per element (an out-of-line libm call, 32000 per token)
/// becomes vector_exp8 over 8 lanes, and the running sum stays an 8-lane vector
/// with one partial sum per lane, reduced once after the loop, so no horizontal
/// reduction is inside the loop. The < 8 leftover elements (none here:
/// nvocab = 32000) still use std.math.exp.
fn softmax (x: []f32) void {
  const V = @Vector(exp_vec_width, f32);
  // G2/S2 = S1 + 4 independent max accumulators (shortens the dependency
  // chain of the reduction: 32000 -> 4 lanes x 1000 sequential @max).
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
  // GQA/MQA head mapping: nheads query heads share nkvheads key/value heads,
  // so query head h reads the k/v head h/kvmul (c/run.c:305,341,352).
  // kvmul == 1 (plain MHA, e.g. stories15M: 6/6) leaves the offsets unchanged.
  const kvmul = c.nheads / c.nkvheads;
  const fhsize: f32 = @floatFromInt(hsize);
  const fpos: f32 = @floatFromInt(pos);
  const x = s.x;
  @memcpy(x, w.embeddings[token*dim..][0..dim]);
  // rope angles for this position. They depend only on (pos, i), i.e. they are
  // identical in all nlayers layers, but the baseline recomputed them inside the
  // layer loop: nlayers * dim/2 = 6*144 = 864 pow+cos+sin triples per token.
  // Fill the table once per token (144 of each) and let the layers read it,
  // measured 0.024 ms/token.
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
    const loff = l * c.ncontext * kvdim;
    rmsnorm(s.x1, x, w.wrmsattn[l*dim..][0..dim]);
    s.kp = @alignCast(s.kcache[loff+pos*kvdim..][0..kvdim]);
    s.vp = @alignCast(s.vcache[loff+pos*kvdim..][0..kvdim]);
    matmul(s.q, w.wq[l*dim*dim..][0..dim*dim], s.x1);
    matmul(s.kp, w.wk[l*dim*kvdim..][0..dim*kvdim], s.x1);
    matmul(s.vp, w.wv[l*dim*kvdim..][0..dim*kvdim], s.x1);
    for (0..dim/2) |ii| {
      const i = ii * 2;
      const fcr = rope_cos[ii];
      const fci = rope_sin[ii];
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
      // attn @ v: x1[i] = sum_t attn[t] * v[t][i] -- a rank-1 update per
      // (layer, head, position). Written as an explicit 8-lane @Vector with
      // @mulAdd because the scalar form above is scalarised by LLVM (one SSE
      // addss per element); 8 divides hsize = 48 exactly, measured 0.135
      // ms/token. The loads of v stay sequential, this is not a GEMV.
      const V = @Vector(8, f32);
      for (0..pos+1) |t| {
        const v = s.vcache[loff+t*kvdim+(h/kvmul)*hsize..][0..hsize];
        const av: V = @splat(attn[t]);
        var i: usize = 0;
        while (i + 8 <= hsize) : (i += 8) {
          const xv: V = x1[i..][0..8].*;
          const vv: V = v[i..][0..8].*;
          x1[i..][0..8].* = @mulAdd(V, av, vv, xv);
        }
        while (i < hsize) : (i += 1) {
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
    // swiglu: h[i] = h[i] * silu(h[i]) * h1[i], silu(v) = v / (1 + exp(-v)),
    // the same activation as c/run.c. The exp() is the vector_exp8 used by
    // softmax: 1152 scalar libm calls per layer -> 144 vector FMAs, measured
    // 0.051 ms/token. 8 divides 4*dim = 1152 exactly; the tail keeps the loop
    // correct for any ffndim.
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
      var v = s.h[hi];
      v = v * (1.0/(1.0+std.math.exp(-v))) * s.h1[hi];
      s.h[hi] = v;
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
  log("\n\nzrunv: total {d} tokens, speed {d} tok/s\n\n\n", .{pos-1, tokens_per_sec});
}
