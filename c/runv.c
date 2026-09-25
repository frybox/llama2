
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <cpuid.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

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
  float *embeddings;
  float *wrmsattn;
  float *wrmsffn;
  float *wrmsfinal;
  float *wq;
  float *wk;
  float *wv;
  float *wo;
  float *w1;
  float *w2;
  float *w3;
} Weights;


typedef struct {
  float *x;
  float *x1;
  float *x2;
  float *h;
  float *h1;
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


static void fexit(FILE *f, const char *msg) {
  if (f) fclose(f);
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}


static void mexit(const char *msg) {
  fexit(NULL, msg);
}


void malloc_state(State *s, Config *c) {
  int kvdim = c->dim * c->nkvheads / c->nheads;
  s->x      = calloc(c->dim, sizeof(*s->x));
  s->x1     = calloc(c->dim, sizeof(*s->x1));
  s->x2     = calloc(c->dim, sizeof(*s->x2));
  s->h      = calloc(c->ffndim, sizeof(*s->h));
  s->h1     = calloc(c->ffndim, sizeof(*s->h1));
  s->q      = calloc(c->dim, sizeof(*s->q));
  s->attn   = calloc(c->nheads * c->ncontext, sizeof(*s->attn));
  s->logits = calloc(c->nvocab, sizeof(*s->logits));
  s->kcache = calloc(c->nlayers * c->ncontext * kvdim, sizeof(*s->kcache));
  s->vcache = calloc(c->nlayers * c->ncontext * kvdim, sizeof(*s->vcache));
  if (!s->x || !s->x1 || !s->x2 || !s->h ||
      !s->h1 || !s->q || !s->attn || !s->logits ||
      !s->kcache || !s->vcache) {
    mexit("malloc state failed!");
  }
}


void free_state(State *s) {
  free(s->x);
  free(s->x1);
  free(s->x2);
  free(s->h);
  free(s->h1);
  free(s->q);
  free(s->attn);
  free(s->logits);
  free(s->kcache);
  free(s->vcache);
}


void mmap_weights(Weights *w, Config *c, float *p) {
  int head_size = c->dim / c->nheads;
  int kvdim = head_size * c->nkvheads;
  unsigned long long nlayers = c->nlayers;
  w->embeddings = p;
  p += c->nvocab * c->dim;
  w->wrmsattn = p;
  p += nlayers * c->dim;
  w->wq = p;
  p += nlayers * c->dim * c->dim;
  w->wk = p;
  p += nlayers * c->dim * kvdim;
  w->wv = p;
  p += nlayers * c->dim * kvdim;
  w->wo = p;
  p += nlayers * c->dim * c->dim;
  w->wrmsffn = p;
  p += nlayers * c->dim;
  w->w1 = p;
  p += nlayers * c->dim * c->ffndim;
  w->w2 = p;
  p += nlayers * c->ffndim * c->dim;
  w->w3 = p;
  p += nlayers * c->dim * c->ffndim;
  w->wrmsfinal = p;
  p += c->dim;
}


void read_checkpoint(char *path, Config *c, Weights *w, int *fd, float **data, ssize_t *fsize) {
  FILE *f = fopen(path, "rb");
  if (!f) { mexit("Can't open file"); }
  if (fread(c, sizeof(*c), 1, f) != 1) { fexit(f, "Invalid file"); }
  c->nvocab = abs(c->nvocab);
  fseek(f, 0, SEEK_END);
  *fsize = ftell(f);
  fclose(f);
  *fd = open(path, O_RDONLY);
  if (*fd == -1) { mexit("open failed!"); }
  *data = mmap(NULL, *fsize, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { mexit("mmap failed!"); }
  float *weights = *data + sizeof(Config)/sizeof(float);
  mmap_weights(w, c, weights);
}


void build_transformer(Transformer *tr, char *path) {
  read_checkpoint(path, &tr->c, &tr->w, &tr->fd, &tr->data, &tr->fsize);
  malloc_state(&tr->s, &tr->c);
}


void  free_transformer(Transformer *tr) {
  if (tr->data != MAP_FAILED) {
    munmap(tr->data, tr->fsize);
  }
  if (tr->fd != -1) {
    close(tr->fd);
  }
  free_state(&tr->s);
}


void rmsnorm(float *o, float *x, float *w, int size) {
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


void softmax(float *x, int size) {
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


// ------------------------------------------------------------------ scalar ----
// o[i] = dot(w + i*n, x, n).  Row-major, x is the (small) vector reused for every
// row: this is a GEMV.  Bit-exact reference; also the fallback for CPUs without
// AVX2+FMA.
static void matmul_scalar(float *o, float *w, float *x, int n, int d) {
  for (int i=0; i<d; i++) {
    float v = 0.0f;
    for(int j=0; j<n; j++) {
      v += w[i*n+j] * x[j];
    }
    o[i] = v;
  }
}


// -------------------------------------------------------------------- avx2 ----
// AVX2/FMA GEMV.  For each output row, the j inner loop is vectorized in 8-float
// steps: load 8 floats of the w row and 8 floats of x, accumulate with
// _mm256_fmadd_ps into a single 256-bit accumulator, then reduce the 8 lanes to
// a scalar and finish the tail (<8 elems) scalar.
//
// FMA folds each pair into one fused multiply-add, so results are NOT bit-exact
// with matmul_scalar (fma(x,y,z) != x*y+z in general) -- that is fine here (the
// model samples, it is non-deterministic anyway); relative error stays tiny.
// Compiled with plain -O3 (no -mavx2 needed): the kernel carries
// __attribute__((target("avx2","fma"))); the dispatch (below) checks CPUID at
// runtime, and RUN_KERNEL=scalar|avx2|avx512 (default: highest supported) can
// force one of the three levels.
static void __attribute__((target("avx2,fma")))
matmul_avx2_impl(float *o, float *w, float *x, int n, int d) {
  for (int i=0; i<d; i++) {
    const float *row = w + (size_t)i*n;
    // 4 independent 256-bit accumulators -> 4-way ILP breaks the FMA latency
    // chain (one 256-bit FMA per cycle, 4-cycle latency); unroll 32 floats.
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int j = 0;
    for (; j+32 <= n; j += 32) {
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row+j),      _mm256_loadu_ps(x+j),      a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row+j+8),    _mm256_loadu_ps(x+j+8),    a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row+j+16),   _mm256_loadu_ps(x+j+16),   a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row+j+24),   _mm256_loadu_ps(x+j+24),   a3);
    }
    // reduce 4 accumulators (32 lanes) to a scalar.
    a0 = _mm256_add_ps(a0, a1);
    a2 = _mm256_add_ps(a2, a3);
    a0 = _mm256_add_ps(a0, a2);
    float buf[8];
    _mm256_storeu_ps(buf, a0);
    float v = ((buf[0]+buf[1])+(buf[2]+buf[3])) + ((buf[4]+buf[5])+(buf[6]+buf[7]));
    // trailing <32 elements: finish scalar (n is a multiple of 8 in practice,
    // so this loop is at most 31 iterations; correct for arbitrary n).
    for (; j<n; j++) {
      v += row[j] * x[j];
    }
    o[i] = v;
  }
}


// ------------------------------------------------------------------ avx512 ----
// Same 4-accumulator structure as matmul_avx2_impl, but at 512-bit width:
// four independent 16-lane (zmm) accumulators, 64 floats per iteration.
// Needs only AVX512F (FMA). Carries its own target attribute; the dispatch
// checks CPUID before this is ever called.
static void __attribute__((target("avx512f")))
matmul_avx512_impl(float *o, float *w, float *x, int n, int d) {
  for (int i=0; i<d; i++) {
    const float *row = w + (size_t)i*n;
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int j = 0;
    for (; j+64 <= n; j += 64) {
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(row+j),    _mm512_loadu_ps(x+j),    a0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(row+j+16), _mm512_loadu_ps(x+j+16), a1);
      a2 = _mm512_fmadd_ps(_mm512_loadu_ps(row+j+32), _mm512_loadu_ps(x+j+32), a2);
      a3 = _mm512_fmadd_ps(_mm512_loadu_ps(row+j+48), _mm512_loadu_ps(x+j+48), a3);
    }
    // fold the 4 accumulators (64 lanes) to a scalar with a pairwise tree.
    __m512 a = (a0 + a1) + (a2 + a3);
    float buf[16];
    _mm512_storeu_ps(buf, a);
    float t[8]; for (int k=0;k<8;k++)  t[k] = buf[k] + buf[k+8];
    float u[4]; for (int k=0;k<4;k++)  u[k] = t[k] + t[k+4];
    float v = (u[0]+u[1]) + (u[2]+u[3]);
    for (; j<n; j++) v += row[j] * x[j];
    o[i] = v;
  }
}


// ------------------------------------------------------------ dispatch --------
// Cached CPUID check.  Three kernel levels, lowest first:
//   0 scalar, 1 avx2 (needs FMA+AVX2), 2 avx512 (needs FMA+AVX2+AVX512F).
// RUN_KERNEL=scalar|avx2|avx512 can force a level; the default is the highest
// the CPU supports (avx512 when present).  Forcing a kernel the CPU lacks
// crashes (the kernels are plain code under a target attribute) -- that is on
// you, same as in the other runners.
static int run_kernel_level(void) {
  static int cached = -1;
  if (cached >= 0) return cached;
  int level = 0;
  const char *e = getenv("RUN_KERNEL");
  int forced = 0;
  if (e) {
    if (strcmp(e, "scalar") == 0)      { level = 0; forced = 1; }
    else if (strcmp(e, "avx2") == 0)   { level = 1; forced = 1; }
    else if (strcmp(e, "avx512") == 0) { level = 2; forced = 1; }
    else fprintf(stderr, "RUN_KERNEL: unknown value '%s', ignoring\n", e);
  }
  if (!forced) {
    // CPUID bits: FMA = (1).ECX[12], AVX2 = (7,0).EBX[5], AVX512F = (7,0).EBX[16].
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid(1, eax, ebx, ecx, edx);
    int has_fma  = (ecx >> 12) & 1;
    int has_avx2 = 0, has_avx512f = 0;
    if (__get_cpuid_max(0, 0) >= 7) {
      unsigned int e7a = 0, e7b = 0, e7c = 0, e7d = 0;
      __cpuid_count(7, 0, e7a, e7b, e7c, e7d);
      has_avx2    = (e7b >> 5)  & 1;
      has_avx512f = (e7b >> 16) & 1;
    }
    if (has_fma && has_avx2) level = 1;
    if (has_fma && has_avx2 && has_avx512f) level = 2;
  }
  cached = level;
  return cached;
}


void matmul(float *o, float *w, float *x, int n, int d) {
  switch (run_kernel_level()) {
    case 2: matmul_avx512_impl(o, w, x, n, d); break;
    case 1: matmul_avx2_impl(o, w, x, n, d); break;
    default: matmul_scalar(o, w, x, n, d); break;
  }
}


float *forward(Transformer *tr, int token, int pos) {
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
    // 1. self-attention sublayer
    rmsnorm(s->x1, x, w->wrmsattn+l*dim, dim);
    unsigned long long loff = l * c->ncontext * kvdim;
    float *k = s->kcache + loff + pos*kvdim;
    float *v = s->vcache + loff + pos*kvdim;
    matmul(s->q, w->wq + l*dim*dim, s->x1, dim, dim);
    matmul(k, w->wk + l*dim*kvdim, s->x1, dim, kvdim);
    matmul(v, w->wv + l*dim*kvdim, s->x1, dim, kvdim);
    for (int i=0; i<dim; i+=2) {
      int hdim = i % hsize;
      float freq = 1.0f / powf(10000.0f, hdim/(float)hsize);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      float v0 = s->q[i], v1 = s->q[i+1];
      s->q[i]   = v0*fcr - v1*fci;
      s->q[i+1] = v0*fci + v1*fcr;
      if (i < kvdim) {
        v0 = k[i]; v1 = k[i+1];
        k[i]   = v0*fcr - v1*fci;
        k[i+1] = v0*fci + v1*fcr;
      }
    }
    // 可按h并行处理所有head
    for (int h=0; h<c->nheads; h++) {
      float *q = s->q + h*hsize;
      float *attn = s->attn + h*c->ncontext;
      for (int t=0; t<=pos; t++) {
        float *k = s->kcache + loff + t*kvdim + (h/kvmul)*hsize;
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
        float *v = s->vcache + loff + t*kvdim + (h/kvmul)*hsize;
        for (int i=0; i<hsize; i++) {
          x1[i] += attn[t] * v[i];
        }
      }  
    }
    matmul(s->x2, w->wo+l*dim*dim, s->x1, dim, dim);
    for (int i=0; i<dim; i++) {
      x[i] += s->x2[i];
    }

    // 2. ffn sublayer
    rmsnorm(s->x1, x, w->wrmsffn+l*dim, dim);
    matmul(s->h, w->w1+l*dim*ffndim, s->x1, dim, ffndim);
    matmul(s->h1, w->w3+l*dim*ffndim, s->x1, dim, ffndim);
    for (int i=0; i<ffndim; i++) {
      float val = s->h[i];
      val *= (1.0f / (1.0f + expf(-val)));
      val *= s->h1[i];
      s->h[i] = val;
    }
    matmul(s->x1, w->w2+l*ffndim*dim, s->h, ffndim, dim);
    for (int i=0; i<dim; i++) {
      x[i] += s->x1[i];
    }
  }

  rmsnorm(x, x, w->wrmsfinal, dim);
  matmul(s->logits, w->embeddings, x, dim, c->nvocab);
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


int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


void build_tokenizer(Tokenizer *t, char *path, int vocab_size) {
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


void free_tokenizer(Tokenizer *t) {
  for (int i=0; i<t->vocab_size; i++) free(t->vocab[i]);
  free(t->vocab);
  free(t->scores);
  free(t->sorted);
}


char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  if (prev_token == 1 && piece[0] == ' ') piece++;
  unsigned char bytev;
  if (sscanf(piece, "<0x%02hhX>", &bytev) == 1) {
    piece = (char *)t->byte_pieces + bytev*2;
  }
  return piece;
}


void safe_printf(char *piece) {
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


int str_lookup(char *str, TokenIndex *sorted, int vocab_size) {
  TokenIndex tok = { .str = str };
  TokenIndex *res = bsearch(&tok, sorted, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}


void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *ntokens) {
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


int sample_argmax(float *p, int n) {
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


int sample_mult(float *p, int n, float coin) {
  float cdf = 0.0f;
  for (int i=0; i<n; i++) {
    cdf += p[i];
    if (coin < cdf) return i;
  }
  return n-1;
}


int compare(const void *a, const void *b) {
  ProbIndex *pa = (ProbIndex*) a;
  ProbIndex *pb = (ProbIndex*) b;
  if (pa->prob > pb->prob) return -1;
  if (pa->prob < pb->prob) return 1;
  return 0;
}


int sample_topp(float *p, int n, float topp, ProbIndex *pi, float coin) {
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


void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}


void free_sampler(Sampler *sampler) {
  free(sampler->probindex);
}


unsigned int random_u32(unsigned long long *state) {
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}


float random_f32(unsigned long long *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}


int sample(Sampler *sampler, float *logits) {
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


long time_in_ms() {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
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
    fprintf(stderr, "\ncrunv: total %d tokens, speed %.1f token/s\n\n\n", pos-1, (pos-1) / (double)(end-start)*1000);
  }
  free(prompt_tokens);
}


int main(int argc, char *argv[]) {
  char *checkpoint_path = "stories15M.bin";
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  char *prompt = NULL;
  unsigned long long rng_seed = (unsigned int)time(NULL);
  if (getenv("RUN_SEED")) rng_seed = (unsigned long long)atoll(getenv("RUN_SEED"));
  if (argc >= 2) {
    checkpoint_path = argv[1];
  }
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
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
