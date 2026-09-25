// Bit-exactness test for the int8 GEMV kernels in c/runqv.c.
//
// The three kernels below are COPIES of the ones in runqv.c (matmul_scalar,
// matmul_avx2_impl, matmul_avx512_impl) with the QuantizedTensor struct
// replaced by plain pointers (q = int8, s = float scales) and GS fixed to 32.
// The numeric contract is the one the kernels in runqv.c implement: avx2/avx512
// must agree with matmul_scalar to ~1ulp (different float accumulation order),
// and avx512 must be BIT-EXACT with avx2 (same float accumulation, only the
// exact int32 group sums come from 512-bit instructions). This is checked
// across the real model shapes plus odd sizes (tail paths).
//
// A kernel is only RUN when the CPU reports the features it needs (the
// avx512 kernel is skipped, not simulated, on a CPU without AVX512F/BW/VL/DQ),
// so the test also passes on non-AVX512 machines.
//
//   gcc -O3 -o c/test_matmul c/test_matmul.c -lm && ./c/test_matmul
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <cpuid.h>

#define GS 32

static int cpu_has_avx2 (void) {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  __cpuid(1, eax, ebx, ecx, edx);
  int has_fma = (ecx >> 12) & 1;
  int has_avx2 = 0;
  if (__get_cpuid_max(0, 0) >= 7) {
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    has_avx2 = (ebx >> 5) & 1;
  }
  return has_fma && has_avx2;
}

static int cpu_has_avx512 (void) {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid_max(0, 0) < 7) return 0;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  int f  = (ebx >> 16) & 1; // AVX512F
  int dq = (ebx >> 17) & 1; // AVX512DQ
  int bw = (ebx >> 30) & 1; // AVX512BW
  int vl = (ebx >> 31) & 1; // AVX512VL
  return f && dq && bw && vl;
}

static void matmul_scalar(float *o, const int8_t *wq, const float *ws,
                          const int8_t *xq, const float *xs, int n, int d) {
  for (int i = 0; i < d; i++) {
    float v = 0.0f;
    for (int j = 0; j < n; j += GS) {
      int32_t iv = 0;
      int in = i * n;
      for (int k = 0; k < GS; k++) iv += (int32_t)wq[in + j + k] * (int32_t)xq[j + k];
      v += ((float)iv) * ws[(in + j) / GS] * xs[j / GS];
    }
    o[i] = v;
  }
}

static void __attribute__((target("avx2")))
matmul_avx2_impl(float *o, const int8_t *wq, const float *ws,
                 const int8_t *xq, const float *xs, int n, int d) {
  const int G = GS;
  for (int i = 0; i < d; i++) {
    const int8_t *wrow = wq + (size_t)i * n;
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
        __m256i Pl = _mm256_madd_epi16(Wl, Xl);
        __m256i Ph = _mm256_madd_epi16(Wh, Xh);
        __m256i P  = _mm256_add_epi32(Pl, Ph);
        __m256i S = _mm256_add_epi32(P, _mm256_shuffle_epi32(P, _MM_SHUFFLE(1, 0, 3, 2)));
        __m256i U = _mm256_add_epi32(S, _mm256_shuffle_epi32(S, _MM_SHUFFLE(3, 3, 3, 3)));
        int32_t lo = _mm_extract_epi32(_mm256_castsi256_si128(U), 0);
        int32_t hi = _mm_extract_epi32(_mm256_extracti128_si256(U, 1), 0);
        s[g] = lo + hi;
      }
      __m128i si = _mm_set_epi32(s[3], s[2], s[1], s[0]);
      __m128 fv = _mm_cvtepi32_ps(si);
      __m128 wsv = _mm_loadu_ps(ws + (i * n + j) / G);
      __m128 xsv = _mm_loadu_ps(xs + j / G);
      acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(fv, wsv), xsv));
    }
    float v;
    { float fv[4]; _mm_storeu_ps(fv, acc); v = (fv[0] + fv[1]) + (fv[2] + fv[3]); }
    for (int j2 = j; j2 < n; j2 += G) {
      int32_t iv = 0;
      for (int k = 0; k < G; k++) iv += (int32_t)wrow[j2 + k] * (int32_t)xq[j2 + k];
      v += (float)iv * ws[(i * n + j2) / G] * xs[j2 / G];
    }
    o[i] = v;
  }
}

static void __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
matmul_avx512_impl(float *o, const int8_t *wq, const float *ws,
                   const int8_t *xq, const float *xs, int n, int d) {
  const int G = GS;
  for (int i = 0; i < d; i++) {
    const int8_t *wrow = wq + (size_t)i * n;
    __m128 acc = _mm_setzero_ps();
    int j = 0;
    for (; j + 4 * G <= n; j += 4 * G) {
      int32_t s[4];
      for (int g = 0; g < 4; g++) {
        int jg = j + g * G;
        __m256i wl = _mm256_loadu_si256((const __m256i *)(wrow + jg));
        __m256i xl = _mm256_loadu_si256((const __m256i *)(xq + jg));
        __m512i W = _mm512_cvtepi8_epi16(wl);
        __m512i X = _mm512_cvtepi8_epi16(xl);
        __m512i P = _mm512_madd_epi16(W, X);
        s[g] = _mm512_reduce_add_epi32(P);
      }
      __m128i si = _mm_set_epi32(s[3], s[2], s[1], s[0]);
      __m128 fv = _mm_cvtepi32_ps(si);
      __m128 wsv = _mm_loadu_ps(ws + (i * n + j) / G);
      __m128 xsv = _mm_loadu_ps(xs + j / G);
      acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(fv, wsv), xsv));
    }
    float v;
    { float fv[4]; _mm_storeu_ps(fv, acc); v = (fv[0] + fv[1]) + (fv[2] + fv[3]); }
    for (int j2 = j; j2 < n; j2 += G) {
      int32_t iv = 0;
      for (int k = 0; k < G; k++) iv += (int32_t)wrow[j2 + k] * (int32_t)xq[j2 + k];
      v += (float)iv * ws[(i * n + j2) / G] * xs[j2 / G];
    }
    o[i] = v;
  }
}

static void bench_one(const char *name, void (*k)(float*,const int8_t*,const float*,const int8_t*,const float*,int,int),
                     int n, int d, int reps);
static void fill(int8_t *q, int len, uint64_t *st);
static void fillf(float *s, int len, uint64_t *st);

static void bench_one(const char *name, void (*k)(float*,const int8_t*,const float*,const int8_t*,const float*,int,int),
                     int n, int d, int reps) {
  uint64_t st = 777;
  int ng = n / GS;
  int8_t *wq = malloc(n * d * 4); int8_t *xq = malloc(n * 4);
  float *ws = malloc(d * ng * 4); float *xs = malloc(ng * 4);
  float *o = malloc(d * 4);
  fill(wq, n * d, &st); fill(xq, n, &st); fillf(ws, d * ng, &st); fillf(xs, ng, &st);
  k(o, wq, ws, xq, xs, n, d); // warm
  double t0 = (double)clock() / CLOCKS_PER_SEC;
  for (int r = 0; r < reps; r++) k(o, wq, ws, xq, xs, n, d);
  double t1 = (double)clock() / CLOCKS_PER_SEC;
  double sec = (t1 - t0) / reps;
  double gflops = 2.0 * n * d / sec / 1e9;
  printf("  %-8s n=%d d=%d  %8.2f ms/iter  %8.1f GFLOP/s\n", name, n, d, sec * 1e3, gflops);
  free(wq); free(xq); free(ws); free(xs); free(o);
}

static void fill(int8_t *q, int len, uint64_t *st) {
  for (int i = 0; i < len; i++) {
    *st = *st * 6364136223846793005ULL + 1442695040888963407ULL;
    q[i] = (int8_t)((*st >> 32) & 127) - 128; // -128..127
  }
}
static void fillf(float *s, int len, uint64_t *st) {
  for (int i = 0; i < len; i++) {
    *st = *st * 6364136223846793005ULL + 1442695040888963407ULL;
    s[i] = (float)((*st >> 40) % 1000) / 1000.0f + 0.25f;
  }
}

static double maxrel(const float *a, const float *b, int d) {
  double m = 0;
  for (int i = 0; i < d; i++) {
    double den = fabs((double)a[i]) > 1.0 ? fabs((double)a[i]) : 1.0;
    double r = fabs((double)a[i] - (double)b[i]) / den;
    if (r > m) m = r;
  }
  return m;
}

// Numerical contract (same as runqv.c):
//   - avx2/avx512 accumulate the groups in float in a different order than
//     matmul_scalar, so vs scalar only ~1ulp agreement is expected;
//   - avx512 uses the SAME float accumulation as avx2 (only the exact int32
//     group sums are computed with 512-bit ops), so avx512 == avx2 BIT-EXACT.
static int check(int n, int d) {
  static int8_t *wq, *xq; static float *ws, *xs; static float *o1, *o2, *o3;
  uint64_t st = 12345 + n * 7 + d;
  static int caps_known = 0, has_avx2 = 0, has_avx512 = 0;
  if (!caps_known) { caps_known = 1; has_avx2 = cpu_has_avx2(); has_avx512 = cpu_has_avx512(); }
  int ng = n / GS, dg = d * ng;
  wq = malloc(n * d * 4); xq = malloc(n * 4); ws = malloc(dg * 4); xs = malloc(ng * 4);
  o1 = malloc(d * 4); o2 = malloc(d * 4); o3 = malloc(d * 4);
  fill(wq, n * d, &st); fill(xq, n, &st); fillf(ws, dg, &st); fillf(xs, ng, &st);
  matmul_scalar(o1, wq, ws, xq, xs, n, d);
  // kernels whose CPUID features the CPU lacks are SKIPPED, not run (they would
  // SIGILL); a skipped kernel counts as passed.
  int ok2 = 1, ok5 = 1;
  double rel2 = 0.0, rel5 = 0.0;
  if (has_avx2) {
    matmul_avx2_impl(o2, wq, ws, xq, xs, n, d);
    rel2 = maxrel(o1, o2, d);
    if (rel2 > 1e-5) ok2 = 0;
  }
  if (has_avx512) {
    matmul_avx512_impl(o3, wq, ws, xq, xs, n, d);
    if (has_avx2) {                       // must be bit-exact with avx2
      for (int i = 0; i < d; i++) if (o2[i] != o3[i]) ok5 = 0;
    } else {                              // no avx2 to compare against: ~1ulp
      rel5 = maxrel(o1, o3, d);
      if (rel5 > 1e-5) ok5 = 0;
    }
  }
  printf("  n=%-5d d=%-5d", n, d);
  if (has_avx2)        printf("  avx2_vs_scalar rel=%.2e(ok=%d)", rel2, ok2);
  else                 printf("  avx2=skipped(no avx2+fma)");
  if (has_avx512 && has_avx2) printf("  avx512_vs_avx2 bitexact=%d", ok5);
  else if (has_avx512)        printf("  avx512_vs_scalar rel=%.2e(ok=%d)", rel5, ok5);
  else                        printf("  avx512=skipped(no avx512f/bw/vl/dq)");
  printf("\n");
  free(wq); free(xq); free(ws); free(xs); free(o1); free(o2); free(o3);
  return ok2 && ok5;
}

int main(void) {
  // real model shapes (dim=288, ffndim=1024, kvdim=288) + odd/tail sizes
  int all = 1;
  all &= check(288, 288);  // wq/wo/ffn-gate square-ish
  all &= check(288, 1024); // w1/w3
  all &= check(1024, 288); // w2
  all &= check(288, 4096); // embeddings-like (nvocab rows)
  all &= check(64, 64);
  all &= check(32, 32);
  all &= check(96, 48);    // odd multiple of 32 (not of 128) -> tail in float loop
  all &= check(160, 32);
  printf(all ? "ALL PASS\n" : "FAIL\n");
  if (getenv("BENCH")) {
    printf("\nbench (n=288 d=1024, reps=200000):\n");
    bench_one("scalar", matmul_scalar, 288, 1024, 200000);
    if (cpu_has_avx2())   bench_one("avx2",   matmul_avx2_impl, 288, 1024, 200000);
    if (cpu_has_avx512()) bench_one("avx512", matmul_avx512_impl, 288, 1024, 200000);
    printf("bench (n=1024 d=288, reps=200000):\n");
    bench_one("scalar", matmul_scalar, 1024, 288, 200000);
    if (cpu_has_avx2())   bench_one("avx2",   matmul_avx2_impl, 1024, 288, 200000);
    if (cpu_has_avx512()) bench_one("avx512", matmul_avx512_impl, 1024, 288, 200000);
  }
  return all ? 0 : 1;
}
