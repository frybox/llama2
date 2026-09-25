// AVX2 int8 GEMV kernel for llama2qv (mainqv.zig), exported through a flat
// C ABI.
//
// This is a direct C translation of c/runq.c's matmul_avx2_impl (same
// instructions, same operations, same order - only the QuantizedTensor
// struct accessors become plain pointers), so llama2qv's avx2 matmul is
// bit-identical to c/runq's: the 8-10% speed gap seen in benchmarks is
// gone. mainqv.zig's runtime CPUID dispatch calls qv_matmul_avx2() when
// the CPU has AVX2 and the checkpoint group size is 32; otherwise it uses
// its scalar Zig matmul (same exact integer group sums; its float
// accumulation tree differs from this kernel's 4-lane vector one, so the
// two paths may disagree by ~1 ulp - text output is identical in
// practice).
//
// Contract (same layout as c/runq.c, dequantized GEMV):
//
//   o[i] = sum_g iv_{i,g} * ws[i*n/G + g] * xs[g]
//   iv   = exact integer sum of the 32 products
//          wq[i*n + 32g + 0 .. 32) * xq[32g + 0 .. 32)   (int32, no overflow:
//          32*127*127 = 516128 < 2^31)
//   G    = 32 (dispatch guarantees GS == 32; anything else falls back to
//          the scalar kernel)
//   n    = x quantized length (input groups * 32)
//   d    = number of output rows (o)
//
// No -mavx2 build flag needed: the kernel carries
// __attribute__((target("avx2"))), and it is only ever called after a
// runtime CPUID probe confirmed AVX2 (see mainqv.zig detectCpu).
#include <stdint.h>
#include <immintrin.h>

static void __attribute__((target("avx2")))
qv_matmul_avx2_impl (float *o, const int8_t *wq, const float *ws,
                     const int8_t *xq, const float *xs, int n, int d) {
  const int G = 32; // group size (GS); dispatch guarantees 32
  for (int i = 0; i < d; i++) {
    const int8_t *wrow = wq + (size_t)i * n;
    // 4-lane float accumulator: one vector add per 4 groups (4-way ILP), and a
    // single int32->float conversion per 4 groups.
    __m128 acc = _mm_setzero_ps();
    int j = 0;
    for (; j + 4 * G <= n; j += 4 * G) {
      int32_t s[4];
      for (int g = 0; g < 4; g++) {
        int jg = j + g * G;
        __m128i wl = _mm_loadu_si128((const __m128i *)(wrow + jg));
        __m128i wh = _mm_loadu_si128((const __m128i *)(wrow + jg + 16));
        __m128i xl = _mm_loadu_si128((const __m128i *)(xq + jg));
        __m128i xh = _mm_loadu_si128((const __m128i *)(xq + jg + 16));
        __m256i Wl = _mm256_cvtepi8_epi16(wl);
        __m256i Wh = _mm256_cvtepi8_epi16(wh);
        __m256i Xl = _mm256_cvtepi8_epi16(xl);
        __m256i Xh = _mm256_cvtepi8_epi16(xh);
        __m256i Pl = _mm256_madd_epi16(Wl, Xl); // 8 int32 (16 prods, elems 0..15)
        __m256i Ph = _mm256_madd_epi16(Wh, Xh); // 8 int32 (16 prods, elems 16..31)
        __m256i P  = _mm256_add_epi32(Pl, Ph);  // 16 int32 (all 32 prods, exact)
        // exact horizontal sum of each 128-bit lane (4 int32) to its scalar sum.
        // Note: _MM_SHUFFLE(a,b,c,d) places src[d] into dst[0], src[c]->dst[1], etc.
        //   S[0]=P0+P3, S[1]=P1+P2, S[2]=P2+P0, S[3]=P3+P1  (per 128-bit lane)
        //   U[0]=S[0]+S[1]=P0+P1+P2+P3  (verified against a concrete [1,2,3,4] case)
        __m256i S = _mm256_add_epi32(P, _mm256_shuffle_epi32(P, _MM_SHUFFLE(1, 0, 3, 2)));
        __m256i U = _mm256_add_epi32(S, _mm256_shuffle_epi32(S, _MM_SHUFFLE(3, 3, 3, 3)));
        int32_t lo = _mm_extract_epi32(_mm256_castsi256_si128(U), 0);   // sum of 16 prods (elems 0..15)
        int32_t hi = _mm_extract_epi32(_mm256_extracti128_si256(U, 1), 0); // sum of 16 prods (elems 16..31)
        s[g] = lo + hi; // exact sum of all 32 products in group g
      }
      // 4 group int32 sums -> 4 float, one vector multiply-accumulate
      __m128i si = _mm_set_epi32(s[3], s[2], s[1], s[0]);
      __m128 fv = _mm_cvtepi32_ps(si); // 4 int32 -> 4 float (vcvtdq2ps)
      __m128 wsv = _mm_loadu_ps(ws + (i * n + j) / G); // 4 w scales
      __m128 xsv = _mm_loadu_ps(xs + j / G);           // 4 x scales
      acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(fv, wsv), xsv));
    }
    // horizontal-sum the 4-lane float accumulator.
    // NOTE: _mm_shuffle_ps uses a DIFFERENT imm encoding than _mm256_shuffle_epi32
    // (each dst lane can only pick from {a0,a1,b0,b1}), so use a scalar reduce here.
    float v;
    {
      float fv[4];
      _mm_storeu_ps(fv, acc);
      v = (fv[0] + fv[1]) + (fv[2] + fv[3]);
    }
    // trailing groups after the 4-group batches: continue from the batch loop's
    // final j (NOT (n/G)*G — that would skip a group when n%4G != 0, e.g. n=288).
    for (int j2 = j; j2 < n; j2 += G) {
      int32_t iv = 0;
      for (int k = 0; k < G; k++) iv += (int32_t)wrow[j2 + k] * (int32_t)xq[j2 + k];
      v += (float)iv * ws[(i * n + j2) / G] * xs[j2 / G];
    }
    o[i] = v;
  }
}


void qv_matmul_avx2 (float *o, const int8_t *wq, const float *ws,
                     const int8_t *xq, const float *xs, int n, int d) {
  qv_matmul_avx2_impl(o, wq, ws, xq, xs, n, d);
}
