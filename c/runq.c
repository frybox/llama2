
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <immintrin.h>
#include <cpuid.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif


int GS = 0;

typedef struct {
  int dim;
  int ffndim;
  int nlayers;
  int nheads;
  int nkvheads;
  int nvocab;
  int ncontext;
} Config;


typedef struct {
  int8_t *q;
  float *s;
} QuantizedTensor;


typedef struct {
  QuantizedTensor *q_tokens;
  float *embeddings;
  float *wrmsattn;
  float *wrmsffn;
  float *wrmsfinal;
  QuantizedTensor *wq;
  QuantizedTensor *wk;
  QuantizedTensor *wv;
  QuantizedTensor *wo;
  QuantizedTensor *w1;
  QuantizedTensor *w2;
  QuantizedTensor *w3;
} Weights;


typedef struct {
  float *x;
  float *x1;
  float *x2;
  QuantizedTensor xq;
  float *h;
  float *h1;
  QuantizedTensor hq;
  float *q;
  float *attn;
  float *logits;
  float *kcache;
  float *vcache;
} State;


typedef struct {
  Config c;
  Weights w;
  State s;
  int fd;
  float *data;
  ssize_t fsize;
} Transformer;


static void fexit (FILE *f, const char *msg) {
  if (f) fclose(f);
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}


static void mexit (const char *msg) {
  fexit(NULL, msg);
}


void malloc_state (State *s, Config *c) {
  int dim = c->dim;
  int ffndim = c->ffndim;
  int kvdim = dim * c->nkvheads / c->nheads;
  s->x      = calloc(dim, sizeof(*s->x));
  s->x1     = calloc(dim, sizeof(*s->x1));
  s->x2     = calloc(dim, sizeof(*s->x2));
  s->xq     = (QuantizedTensor){ .q=calloc(dim, sizeof(int8_t)), .s=calloc(dim/GS, sizeof(float)) };
  s->h      = calloc(ffndim, sizeof(*s->h));
  s->h1     = calloc(ffndim, sizeof(*s->h1));
  s->hq     = (QuantizedTensor){ .q=calloc(ffndim, sizeof(int8_t)), .s=calloc(ffndim/GS, sizeof(float)) };
  s->q      = calloc(c->dim, sizeof(*s->q));
  s->attn   = calloc(c->nheads * c->ncontext, sizeof(*s->attn));
  s->logits = calloc(c->nvocab, sizeof(*s->logits));
  s->kcache = calloc(c->nlayers * c->ncontext * kvdim, sizeof(*s->kcache));
  s->vcache = calloc(c->nlayers * c->ncontext * kvdim, sizeof(*s->vcache));
  if (!s->x || !s->x1 || !s->x2 || !s->xq.q || !s->xq.s ||
      !s->h || !s->h1 || !s->hq.q || !s->hq.s || !s->q ||
      !s->attn || !s->logits || !s->kcache || !s->vcache) {
    mexit("malloc state failed!");
  }
}


void free_state (State *s) {
  free(s->x);
  free(s->x1);
  free(s->x2);
  free(s->xq.q);
  free(s->xq.s);
  free(s->h);
  free(s->h1);
  free(s->hq.q);
  free(s->hq.s);
  free(s->q);
  free(s->attn);
  free(s->logits);
  free(s->kcache);
  free(s->vcache);
}


void dequantize (QuantizedTensor *qx, float *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = qx->q[i] * qx->s[i/GS];
  }
}


void quantize (QuantizedTensor *qx, float *x, int n) {
  int ngroups = n / GS;
  float Q_MAX = 127.0f;
  for (int g=0; g<ngroups; g++) {
    float *px = x + g*GS;
    int8_t *pq = qx->q + g*GS;
    float max = 0.0f;
    for (int i=0; i<GS; i++) {
      float v = fabs(px[i]);
      if (v > max) max = v;
    }
    float s = max / Q_MAX;
    for (int i=0; i<GS; i++) {
      pq[i] = (int8_t)round(px[i]/s);
    }
    qx->s[g] = s;
  }
}


QuantizedTensor *init_quantized_tensors (void **ptr, int n, int size_each) {
  void *p = *ptr;
  QuantizedTensor *qts = malloc(n*sizeof(QuantizedTensor));
  for (int i=0; i<n; i++) {
    qts[i].q = (int8_t*)p;
    p = (int8_t*)p + size_each;
    qts[i].s = (float*)p;
    p = (float*)p + size_each/GS;
  }
  *ptr = p;
  return qts;
}


void mmap_weights (Weights *w, Config *c, void *ptr) {
  int dim = c->dim;
  int ffndim = c->ffndim;
  int head_size = dim / c->nheads;
  int kvdim = head_size * c->nkvheads;
  unsigned long long nlayers = c->nlayers;
  float *fp = (float*)ptr;
  w->wrmsattn = fp;
  fp += nlayers * c->dim;
  w->wrmsffn = fp;
  fp += nlayers * c->dim;
  w->wrmsfinal = fp;
  fp += c->dim;

  ptr = (void*)fp;
  w->q_tokens = init_quantized_tensors(&ptr, 1, c->nvocab*c->dim);
  w->embeddings = malloc(c->nvocab * dim * sizeof(float));
  dequantize(w->q_tokens, w->embeddings, c->nvocab*dim);
  w->wq = init_quantized_tensors(&ptr, c->nlayers, dim*dim);
  w->wk = init_quantized_tensors(&ptr, c->nlayers, dim*kvdim);
  w->wv = init_quantized_tensors(&ptr, c->nlayers, dim*kvdim);
  w->wo = init_quantized_tensors(&ptr, c->nlayers, dim*dim);
  w->w1 = init_quantized_tensors(&ptr, c->nlayers, dim*ffndim);
  w->w2 = init_quantized_tensors(&ptr, c->nlayers, ffndim*dim);
  w->w3 = init_quantized_tensors(&ptr, c->nlayers, dim*ffndim);
}


void read_checkpoint (char *path, Config *c, Weights *w, int *fd, float **data, ssize_t *fsize) {
  FILE *f = fopen(path, "rb");
  if (!f) { mexit("Can't open file"); }
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, f) != 1) {
    fexit(f, "Can't read checkpoint magic number");
  }
  if (magic_number != 0x616b3432) {
    fexit(f, "Bad magic number");
  }
  int version;
  if (fread(&version, sizeof(int), 1, f) != 1) {
    fexit(f, "Can't read checkpiont version");
  }
  if (version != 2) {
    fexit(f, "version should be 2");
  }
  if (fread(c, sizeof(Config), 1, f) != 1) {
    fexit(f, "Can't read checkpoint config");
  }
  uint8_t shared_classifier;
  if (fread(&shared_classifier, sizeof(uint8_t), 1, f) != 1) {
    fexit(f, "Can't read shared classifier");
  }
  int group_size;
  if (fread(&group_size, sizeof(int), 1, f) != 1) {
    fexit(f, "Can't read group size");
  }
  GS = group_size;
  fseek(f, 0, SEEK_END);
  *fsize = ftell(f);
  fclose(f);

  int header_size = 256;
  *fd = open(path, O_RDONLY);
  if (*fd == -1) { mexit("open checkpoint file failed!"); }
  *data = mmap(NULL, *fsize, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { mexit("mmap failed!"); }
  void *weights = ((char*)*data) + header_size;
  mmap_weights(w, c, weights);
}


void build_transformer (Transformer *tr, char *path) {
  read_checkpoint(path, &tr->c, &tr->w, &tr->fd, &tr->data, &tr->fsize);
  malloc_state(&tr->s, &tr->c);
}


void  free_transformer (Transformer *tr) {
  free(tr->w.q_tokens);
  free(tr->w.embeddings);
  free(tr->w.wq);
  free(tr->w.wk);
  free(tr->w.wv);
  free(tr->w.wo);
  free(tr->w.w1);
  free(tr->w.w2);
  free(tr->w.w3);
  if (tr->data != MAP_FAILED) {
    munmap(tr->data, tr->fsize);
  }
  if (tr->fd != -1) {
    close(tr->fd);
  }
  free_state(&tr->s);
}


void rmsnorm (float *o, float *x, float *w, int size) {
  float ss = 0.0f;
  for (int i=0; i<size; i++) {
    ss += x[i] * x[i];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int i=0; i<size; i++) {
    o[i] = w[i] * ss * x[i];
  }
}


void softmax (float *x, int size) {
  float maxv = x[0];
  for (int i=1; i<size; i++) {
    if (x[i] > maxv) {
      maxv = x[i];
    }
  }
  float sum = 0.0f;
  for (int i=0; i<size; i++) {
    x[i] = expf(x[i] - maxv);
    sum += x[i];
  }
  for (int i=0; i<size; i++) {
    x[i] /= sum;
  }
}


// AVX2 int8 GEMV kernel for runq.c, plus dispatch wrapper + scalar baseline.
//
// matmul(o, w, x, n, d):  o[i] = sum_j w[i*n+j] * x[j]   (dequantized)
//   w: QuantizedTensor  q (n*d int8), s (d*(n/GS) float scales, row-major)
//   x: QuantizedTensor  q (n int8),   s (n/GS float scales)
//   dequant: o[i] = sum_{g} (int) (sum_{k=0}^{GS-1} w->q[i*n+j+k]*x->q[j+k])
//                        * w->s[(i*n+j)/GS] * x->s[j/GS]
//
// AVX2 kernel (matmul_avx2_impl), per output row i, one quantization group of
// GS==32 elements at a time:
//   - load 32 int8 of w(row i) and 32 int8 of x (two 128-bit loads each)
//   - sign-extend to 16 int16 each (vpunpcklbw/vpunpckhbw via _mm256_cvtepi8_epi16)
//   - P = vpmaddwd: 8 int32, lane k = W[2k]*X[2k] + W[2k+1]*X[2k+1]
//     (int8*int8 fits int16 exactly; no overflow)
//   - exact horizontal sum of the 8 int32 -> int32 group sum `iv`
//     (int32 add is exact & associative, |iv| <= 32*127*127 = 516128 < 2^31)
//   - v += (float)iv * w->s[...] * x->s[...]   (same float ops/order as scalar)
//
// Bit-exact with matmul_scalar when GS==32 (and for any n, trailing handled).
// Compiled with plain -O3/-Ofast (no -mavx2 needed): the kernel carries
// __attribute__((target("avx2"))); dispatch probes CPUID at runtime (see
// runq_cpu below), requires GS==32, and RUNQ_KERNEL=scalar|avx2|avx512|amx
// (default: auto) can force a specific kernel.

// ---------------------------------------------------------------- scalar -----
static void matmul_scalar (float *o, QuantizedTensor *w, QuantizedTensor *x, int n, int d) {
  for (int i = 0; i < d; i++) {
    float v = 0.0f;
    for (int j = 0; j < n; j += GS) {
      int32_t iv = 0;
      int in = i * n;
      for (int k = 0; k < GS; k++) {
        iv += (int32_t)w->q[in + j + k] * (int32_t)x->q[j + k];
      }
      v += ((float)iv) * w->s[(in + j) / GS] * x->s[j / GS];
    }
    o[i] = v;
  }
}

// ------------------------------------------------------------------ avx2 -----
static void __attribute__((target("avx2")))
matmul_avx2_impl (float *o, QuantizedTensor *w, QuantizedTensor *x, int n, int d) {
  const int8_t *wq = w->q;
  const int8_t *xq = x->q;
  const float *ws = w->s;
  const float *xs = x->s;
  const int G = GS; // assumed 32 (checked in dispatch)
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

static void matmul_avx2 (float *o, QuantizedTensor *w, QuantizedTensor *x, int n, int d) {
  matmul_avx2_impl(o, w, x, n, d);
}

// --------------------------------------------------------------- dispatch ----
// CPU feature table (x86-64). Everything is probed at RUNTIME via CPUID, so
// no -mavx2/-mavx512f build flags are needed: each kernel carries
// __attribute__((target("..."))), and this table only DECIDES which kernel
// to call.
//
//   feature      CPUID leaf:reg[bit]             notes
//   AVX2         CPUID(7,0).EBX[5]               256-bit integer+float
//   FMA          CPUID(1).ECX[12]                fused multiply-add
//   AVX512F      CPUID(7,0).EBX[16]              512-bit vector state (ZMM)
//   AVX512BW     CPUID(7,0).EBX[30]              512-bit byte/word
//   AVX512VL     CPUID(7,0).EBX[31]              512-bit instrs on 128/256 regs
//   AVX512VNNI   CPUID(7,0).ECX[11]              512-bit vdpbusd int8 dot (future)
//   AMX-TILE     CPUID(7,0).EDX[24]              AMX tile config (future GEMM)
//   AMX-BF16     CPUID(7,0).EDX[22]              AMX bf16 (future GEMM)
//   AMX-INT8     CPUID(7,0).EDX[25]              AMX int8 (int8 GEMM, future)
//
//   (512-bit int8 GEMV needs F+BW+VL together; an AMX GEMM needs AMX-TILE plus
//   the element-type feature (AMX-INT8 for int8).  Verified against the Linux
//   kernel cpufeatures.h, which populates /proc/cpuinfo on this host.)
//
// Selection picks the HIGHEST level that is (a) supported by the CPU and
// (b) has a kernel implemented here:  scalar < avx2 < avx512 < amx.
// Every vector kernel is specialized for GS==32, so other group sizes fall
// back to scalar.  Override: RUNQ_KERNEL=scalar|avx2|avx512|amx (forcing a
// kernel your CPU lacks will crash — that is on you).
typedef struct {
  int has_fma;
  int has_avx2;
  int has_avx512f;
  int has_avx512bw;
  int has_avx512vl;
  int has_avx512vnni;
  int has_amx_tile;
  int has_amx_bf16;
  int has_amx_int8;
  int level;        // selected kernel: 0 scalar, 1 avx2, 2 avx512, 3 amx
  const char *kernel;
} CpuCaps;

static CpuCaps runq_cpu (void) {
  static CpuCaps caps;
  static int done = 0;
  if (done) return caps;
  done = 1;

  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  unsigned int maxid = __get_cpuid_max(0, 0); // max standard CPUID leaf

  __cpuid(1, eax, ebx, ecx, edx);
  caps.has_fma  = (ecx >> 12) & 1;   // CPUID(1).ECX[12] = FMA

  if (maxid >= 7) {
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    caps.has_avx2        = (ebx >> 5)  & 1;   // CPUID(7,0).EBX[5]
    caps.has_avx512f     = (ebx >> 16) & 1;   // CPUID(7,0).EBX[16]
    caps.has_avx512bw    = (ebx >> 30) & 1;   // CPUID(7,0).EBX[30]
    caps.has_avx512vl    = (ebx >> 31) & 1;   // CPUID(7,0).EBX[31]
    caps.has_avx512vnni  = (ecx >> 11) & 1;   // CPUID(7,0).ECX[11]
    caps.has_amx_bf16    = (edx >> 22) & 1;   // CPUID(7,0).EDX[22]
    caps.has_amx_tile    = (edx >> 24) & 1;   // CPUID(7,0).EDX[24]
    caps.has_amx_int8    = (edx >> 25) & 1;   // CPUID(7,0).EDX[25]
  }

  // pick highest supported level.  Levels 2 (avx512) and 3 (amx) have no
  // dedicated kernel yet, so auto-selection caps at the highest IMPLEMENTED
  // level (RUNQ_KERNEL=avx512|amx can still force it for a CPU that has it).
  int level = 0;
  const char *kernel = "scalar";
  if (caps.has_avx2) { level = 1; kernel = "avx2"; }
  if (caps.has_avx512f && caps.has_avx512bw && caps.has_avx512vl) { level = 2; kernel = "avx512"; }
  if (caps.has_amx_tile && (caps.has_amx_bf16 || caps.has_amx_int8)) { level = 3; kernel = "amx"; }
  if (GS != 32) { level = 0; kernel = "scalar"; } // kernels specialized for GS==32

  const int level_implemented = 1; // avx2 is the highest kernel written so far
  if (level > level_implemented) { level = level_implemented; kernel = "avx2"; }

  // manual override
  const char *e = getenv("RUNQ_KERNEL");
  if (e) {
    if      (!strcmp(e, "scalar")) { level = 0; kernel = "scalar"; }
    else if (!strcmp(e, "avx2"))   { level = 1; kernel = "avx2";   }
    else if (!strcmp(e, "avx512")) { level = 2; kernel = "avx512"; }
    else if (!strcmp(e, "amx"))    { level = 3; kernel = "amx";    }
    else fprintf(stderr, "RUNQ_KERNEL: unknown value '%s', ignoring\n", e);
  }
  caps.level = level;
  caps.kernel = kernel;
  return caps;
}

static void runq_print_cpu (void) {
  CpuCaps c = runq_cpu();
  fprintf(stderr,
    "cpu: avx2=%d fma=%d avx512f=%d avx512bw=%d avx512vl=%d avx512vnni=%d"
    " amx_tile=%d amx_bf16=%d amx_int8=%d  -> matmul kernel: %s (GS=%d)\n",
    c.has_avx2, c.has_fma, c.has_avx512f, c.has_avx512bw, c.has_avx512vl,
    c.has_avx512vnni, c.has_amx_tile, c.has_amx_bf16, c.has_amx_int8,
    c.kernel, GS);
}

void matmul (float *o, QuantizedTensor *w, QuantizedTensor *x, int n, int d) {
  CpuCaps c = runq_cpu();
  switch (c.level) {
    case 1: matmul_avx2(o, w, x, n, d); break;
    case 2: matmul_avx2(o, w, x, n, d); break; // TODO: 512-bit kernel (AVX512F+BW+VL)
    case 3: matmul_avx2(o, w, x, n, d); break; // TODO: AMX tile kernel (AMX-TILE+INT8)
    default: matmul_scalar(o, w, x, n, d);
  }
}


float *forward (Transformer *tr, int token, int pos) {
  Config *c = &tr->c;
  Weights *w = &tr->w;
  State *s = &tr->s;
  float *x = s->x;
  int dim = c->dim;
  int kvdim = dim * c->nkvheads / c->nheads;
  int kvmul = c->nheads / c->nkvheads;
  int ffndim = c->ffndim;
  int hsize = dim / c->nheads;

  float *content = w->embeddings + token*dim;
  memcpy(x, content, dim*sizeof(*x));

  for (unsigned long long l=0; l<c->nlayers; l++) {
    unsigned long long loff = l * c->ncontext * kvdim;

    // 1. self-attention sublayer
    rmsnorm(s->x1, x, w->wrmsattn+l*dim, dim);
    float *q = s->q;
    float *k = s->kcache + loff + pos*kvdim;
    float *v = s->vcache + loff + pos*kvdim;
    quantize(&s->xq, s->x1, dim);
    matmul(q, w->wq+l, &s->xq, dim, dim);
    matmul(k, w->wk+l, &s->xq, dim, kvdim);
    matmul(v, w->wv+l, &s->xq, dim, kvdim);

    for (int i=0; i<dim; i+=2) {
      int hdim = i % hsize;
      float freq = 1.0f / powf(10000.0f, hdim/(float)hsize);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      float v0 = q[i], v1 = q[i+1];
      q[i]   = v0*fcr - v1*fci;
      q[i+1] = v0*fci + v1*fcr;
      if (i < kvdim) {
        v0 = k[i]; v1 = k[i+1];
        k[i]   = v0*fcr - v1*fci;
        k[i+1] = v0*fci + v1*fcr;
      }
    }
    // 可按h并行处理所有head
    for (int h=0; h<c->nheads; h++) {
      q = s->q + h*hsize;
      float *attn = s->attn + h*c->ncontext;
      for (int t=0; t<=pos; t++) {
        k = s->kcache + loff + t*kvdim + (h/kvmul)*hsize;
        float score = 0.0f;
        for (int i=0; i<hsize; i++) {
          score += q[i] * k[i];
        }
        attn[t] = score / sqrtf(hsize);
      }
      softmax(attn, pos+1);
      float *x1 = s->x1 + h*hsize;
      memset(x1, 0, hsize*sizeof(*x1));
      for (int t=0; t<=pos; t++) {
        v = s->vcache + loff + t*kvdim + (h/kvmul)*hsize;
        for (int i=0; i<hsize; i++) {
          x1[i] += attn[t] * v[i];
        }
      }  
    }
    quantize(&s->xq, s->x1, dim);
    matmul(s->x2, w->wo+l, &s->xq, dim, dim);
    for (int i=0; i<dim; i++) {
      x[i] += s->x2[i];
    }

    // 2. ffn sublayer
    rmsnorm(s->x1, x, w->wrmsffn+l*dim, dim);
    quantize(&s->xq, s->x1, dim);
    matmul(s->h, w->w1+l, &s->xq, dim, ffndim);
    matmul(s->h1, w->w3+l, &s->xq, dim, ffndim);
    for (int i=0; i<ffndim; i++) {
      float val = s->h[i];
      val *= (1.0f / (1.0f + expf(-val)));
      val *= s->h1[i];
      s->h[i] = val;
    }
    quantize(&s->hq, s->h, ffndim);
    matmul(s->x1, w->w2+l, &s->hq, ffndim, dim);
    for (int i=0; i<dim; i++) {
      x[i] += s->x1[i];
    }
  }

  rmsnorm(x, x, w->wrmsfinal, dim);
  quantize(&s->xq, x, dim);
  matmul(s->logits, w->q_tokens, &s->xq, dim, c->nvocab);
  return s->logits;
}


typedef struct {
  char *str;
  int id;
} TokenIndex;


typedef struct {
  char **vocab;
  float *scores;
  TokenIndex *sorted;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;


int compare_tokens (const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


void build_tokenizer (Tokenizer *t, char *path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char **)malloc(vocab_size * sizeof(char*));
  t->scores = (float*)malloc(vocab_size *sizeof(float));
  t->sorted = NULL;
  for (int i=0; i<256; i++) {
    t->byte_pieces[i*2] = (unsigned char)i;
    t->byte_pieces[i*2+1] = '\0';
  }
  FILE *f = fopen(path, "rb");
  if (!f) { mexit("Couldn't load file"); }
  if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
    fexit(f, "failed read file");
  }
  for (int i=0; i<vocab_size; i++) {
    if (fread(t->scores+i, sizeof(float), 1, f) != 1) fexit(f, "failed read score");
    int len;
    if (fread(&len, sizeof(int), 1, f) != 1) fexit(f, "failed read len");
    t->vocab[i] = (char *)malloc(len+1);
    if (fread(t->vocab[i], len, 1, f) != 1) fexit(f, "failed read vocab");
    t->vocab[i][len] = '\0';
  }  
  fclose(f);
}


void free_tokenizer (Tokenizer *t) {
  for (int i=0; i<t->vocab_size; i++) free(t->vocab[i]);
  free(t->vocab);
  free(t->scores);
  free(t->sorted);
}


char *decode (Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  if (prev_token == 1 && piece[0] == ' ') piece++;
  unsigned char bytev;
  if (sscanf(piece, "<0x%02hhX>", &bytev) == 1) {
    piece = (char *)t->byte_pieces + bytev*2;
  }
  return piece;
}


void safe_printf (char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}


int str_lookup (char *str, TokenIndex *sorted, int vocab_size) {
  TokenIndex tok = { .str = str };
  TokenIndex *res = bsearch(&tok, sorted, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}


void encode (Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *ntokens) {
  if (!text) mexit("cannot encode NULL text");
  if (!t->sorted) {
    t->sorted = malloc(t->vocab_size*sizeof(TokenIndex));
    for (int i=0; i<t->vocab_size; i++) {
      t->sorted[i].str = t->vocab[i];
      t->sorted[i].id = i;
    }
    qsort(t->sorted, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }
  char *strbuf = malloc((t->max_token_length*2+1+2) * sizeof(char));
  size_t strlen = 0;
  int n = 0;
  if (bos) tokens[n++] = 1;
  if (text[0] != '\0') {
    int dummy_suffix = str_lookup(" ", t->sorted, t->vocab_size);
    tokens[n++] = dummy_suffix;
  }
  for (char *c=text; *c!='\0'; c++) {
    if ((*c & 0xC0) != 0x80) strlen = 0;
    strbuf[strlen++] = *c;
    strbuf[strlen] = '\0';
    if ((*(c+1)&0xC0) == 0x80 && strlen < 4) continue;
    int id = str_lookup(strbuf, t->sorted, t->vocab_size);
    if (id != -1) {
      tokens[n++] = id;
    } else {
      for (int i=0; i<strlen; i++) {
        tokens[n++] = (unsigned char)strbuf[i] + 3;
      }
    }
    strlen = 0;
  }
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    for (int i=0; i<(n-1); i++) {
      sprintf(strbuf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(strbuf, t->sorted, t->vocab_size);
      if (id != -1 && t->scores[id] > best_score) {
        best_score = t->scores[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_idx == -1) break;
    tokens[best_idx] = best_id;
    for (int i=best_idx+1; i<n-1; i++) {
      tokens[i] = tokens[i+1];
    }
    n--;
  }
  if (eos) tokens[n++] = 2;
  *ntokens = n;
  free(strbuf);
}


typedef struct {
  float prob;
  int index;
} ProbIndex;


typedef struct {
  int vocab_size;
  ProbIndex *probindex;
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;


int sample_argmax (float *p, int n) {
  int maxi = 0;
  float maxp = p[0];
  for (int i=1; i<n; i++) {
    if (p[i] > maxp) {
      maxi = i;
      maxp = p[i];
    }
  }
  return maxi;
}


int sample_mult (float *p, int n, float coin) {
  float cdf = 0.0f;
  for (int i=0; i<n; i++) {
    cdf += p[i];
    if (coin < cdf) return i;
  }
  return n-1;
}


int compare (const void *a, const void *b) {
  ProbIndex *pa = (ProbIndex*) a;
  ProbIndex *pb = (ProbIndex*) b;
  if (pa->prob > pb->prob) return -1;
  if (pa->prob < pb->prob) return 1;
  return 0;
}


int sample_topp (float *p, int n, float topp, ProbIndex *pi, float coin) {
  int n0 = 0;
  const float cutoff = (1.0f-topp) / (n-1);
  for (int i=0; i<n; i++) {
    if (p[i] >= cutoff) {
      pi[n0].index = i;
      pi[n0++].prob = p[i];
    }
  }
  qsort(pi, n0, sizeof(ProbIndex), compare);
  float sum = 0.0f;
  int last_idx = n0-1;
  for (int i=0; i<n0; i++) {
    sum += pi[i].prob;
    if (sum > topp) {
      last_idx = i;
      break;
    }
  }
  float r = coin * sum;
  float cdf = 0.0f;
  for (int i=0; i<=last_idx; i++) {
    cdf += pi[i].prob;
    if (r<cdf) return pi[i].index;
  }
  return pi[last_idx].index;
}


void build_sampler (Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}


void free_sampler (Sampler *sampler) {
  free(sampler->probindex);
}


unsigned int random_u32 (unsigned long long *state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}


float random_f32 (unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}


int sample (Sampler *sampler, float *logits) {
  int next;
  if (sampler->temperature == 0.0f) {
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    for (int i=0; i<sampler->vocab_size; i++) {
      logits[i] /= sampler->temperature;
    }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}


long time_in_ms () {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void generate (Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (!prompt) prompt = empty_prompt;
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int*)malloc((strlen(prompt)+3)*sizeof(int));
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    mexit("something iswrong, expected at least 1 prompt token");
  }
  long start = 0;
  int next;
  int token = prompt_tokens[0];
  int pos = 0;
  while (pos < steps) {
    float *logits = forward(transformer, token, pos);
    if (pos < num_prompt_tokens-1) {
      next = prompt_tokens[pos+1];
    } else {
      next = sample(sampler, logits);
    }
    pos++;
    if (next == 1) break;
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);
    token = next;
    if (start == 0) start = time_in_ms();
  }
  printf("\n");
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "\ntotal %d tokens, speed %.1f token/s\n", pos-1, (pos-1) / (double)(end-start)*1000);
  }
  free(prompt_tokens);
}


int main (int argc, char *argv[]) {
  char *checkpoint_path = "stories15M-q8.bin";
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  char *prompt = NULL;
  unsigned long long rng_seed = (unsigned int)time(NULL);
  if (getenv("RUNQ_SEED")) rng_seed = (unsigned long long)atoll(getenv("RUNQ_SEED"));
  if (argc >= 2) {
    checkpoint_path = argv[1];
  }
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  runq_print_cpu();
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.c.nvocab);
  Sampler sampler;
  build_sampler(&sampler, transformer.c.nvocab, temperature, topp, rng_seed);
  generate(&transformer, &tokenizer, &sampler, prompt, steps);
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
