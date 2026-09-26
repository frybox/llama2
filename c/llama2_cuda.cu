
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <cuda_runtime.h>


#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s: %s (line %d)\n", cudaGetErrorString(err), #x, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


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
  // device weights
  float *dev_embeddings;
  float *dev_wrmsattn;
  float *dev_wrmsffn;
  float *dev_wrmsfinal;
  float *dev_wq;
  float *dev_wk;
  float *dev_wv;
  float *dev_wo;
  float *dev_w1;
  float *dev_w2;
  float *dev_w3;
  // device state
  float *dev_x;
  float *dev_x1;
  float *dev_x2;
  float *dev_h;
  float *dev_h1;
  float *dev_q;
  float *dev_attn;
  float *dev_logits;
  float *dev_kcache;
  float *dev_vcache;
} Transformer;


static void fexit(FILE *f, const char *msg) {
  if (f) fclose(f);
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}


static void mexit(const char *msg) {
  fexit(NULL, msg);
}


void malloc_state(Transformer *tr, State *s, Config *c) {
  int dim = c->dim;
  int ffndim = c->ffndim;
  int kvdim = dim * c->nkvheads / c->nheads;
  CUDA_CHECK(cudaMalloc(&s->x, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&s->x1, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&s->x2, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&s->h, sizeof(float) * ffndim));
  CUDA_CHECK(cudaMalloc(&s->h1, sizeof(float) * ffndim));
  CUDA_CHECK(cudaMalloc(&s->q, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&s->attn, sizeof(float) * (size_t)c->nheads * c->ncontext));
  CUDA_CHECK(cudaMalloc(&s->logits, sizeof(float) * c->nvocab));
  CUDA_CHECK(cudaMalloc(&s->kcache, sizeof(float) * (size_t)c->nlayers * c->ncontext * kvdim));
  CUDA_CHECK(cudaMalloc(&s->vcache, sizeof(float) * (size_t)c->nlayers * c->ncontext * kvdim));
  if (!s->x || !s->x1 || !s->x2 || !s->h || !s->h1 || !s->q ||
      !s->attn || !s->logits || !s->kcache || !s->vcache) {
    mexit("cudaMalloc state failed!");
  }
  tr->dev_x = s->x;
  tr->dev_x1 = s->x1;
  tr->dev_x2 = s->x2;
  tr->dev_h = s->h;
  tr->dev_h1 = s->h1;
  tr->dev_q = s->q;
  tr->dev_attn = s->attn;
  tr->dev_logits = s->logits;
  tr->dev_kcache = s->kcache;
  tr->dev_vcache = s->vcache;
}


void free_state(State *s) {
  cudaFree(s->x);
  cudaFree(s->x1);
  cudaFree(s->x2);
  cudaFree(s->h);
  cudaFree(s->h1);
  cudaFree(s->q);
  cudaFree(s->attn);
  cudaFree(s->logits);
  cudaFree(s->kcache);
  cudaFree(s->vcache);
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


void read_checkpoint(const char *path, Config *c, Weights *w, int *fd, float **data, ssize_t *fsize) {
  FILE *f = fopen(path, "rb");
  if (!f) { mexit("Can't open file"); }
  if (fread(c, sizeof(*c), 1, f) != 1) { fexit(f, "Invalid file"); }
  c->nvocab = abs(c->nvocab);
  fseek(f, 0, SEEK_END);
  *fsize = ftell(f);
  fclose(f);
  *fd = open(path, O_RDONLY);
  if (*fd == -1) { mexit("open failed!"); }
  *data = (float*)mmap(NULL, *fsize, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { mexit("mmap failed!"); }
  float *weights = *data + sizeof(Config)/sizeof(float);
  mmap_weights(w, c, weights);
}


void upload_weights(Transformer *tr) {
  Config *c = &tr->c;
  int dim = c->dim;
  int ffndim = c->ffndim;
  int head_size = dim / c->nheads;
  int kvdim = head_size * c->nkvheads;
  size_t dim2 = (size_t)dim * dim;
  size_t dimkv = (size_t)dim * kvdim;
  size_t dimff = (size_t)dim * ffndim;
  size_t ffn_dim = (size_t)ffndim * dim;
  CUDA_CHECK(cudaMalloc(&tr->dev_embeddings, sizeof(float) * (size_t)c->nvocab * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsattn, sizeof(float) * (size_t)c->nlayers * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsffn, sizeof(float) * (size_t)c->nlayers * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsfinal, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wq, sizeof(float) * (size_t)c->nlayers * dim2));
  CUDA_CHECK(cudaMalloc(&tr->dev_wk, sizeof(float) * (size_t)c->nlayers * dimkv));
  CUDA_CHECK(cudaMalloc(&tr->dev_wv, sizeof(float) * (size_t)c->nlayers * dimkv));
  CUDA_CHECK(cudaMalloc(&tr->dev_wo, sizeof(float) * (size_t)c->nlayers * dim2));
  CUDA_CHECK(cudaMalloc(&tr->dev_w1, sizeof(float) * (size_t)c->nlayers * dimff));
  CUDA_CHECK(cudaMalloc(&tr->dev_w2, sizeof(float) * (size_t)c->nlayers * ffn_dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_w3, sizeof(float) * (size_t)c->nlayers * dimff));
  CUDA_CHECK(cudaMemcpy(tr->dev_embeddings, tr->w.embeddings, sizeof(float) * (size_t)c->nvocab * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsattn, tr->w.wrmsattn, sizeof(float) * (size_t)c->nlayers * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wq, tr->w.wq, sizeof(float) * (size_t)c->nlayers * dim2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wk, tr->w.wk, sizeof(float) * (size_t)c->nlayers * dimkv, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wv, tr->w.wv, sizeof(float) * (size_t)c->nlayers * dimkv, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wo, tr->w.wo, sizeof(float) * (size_t)c->nlayers * dim2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsffn, tr->w.wrmsffn, sizeof(float) * (size_t)c->nlayers * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_w1, tr->w.w1, sizeof(float) * (size_t)c->nlayers * dimff, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_w2, tr->w.w2, sizeof(float) * (size_t)c->nlayers * ffn_dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_w3, tr->w.w3, sizeof(float) * (size_t)c->nlayers * dimff, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsfinal, tr->w.wrmsfinal, sizeof(float) * dim, cudaMemcpyHostToDevice));
}


void build_transformer(Transformer *tr, const char *path) {
  read_checkpoint(path, &tr->c, &tr->w, &tr->fd, &tr->data, &tr->fsize);
  malloc_state(tr, &tr->s, &tr->c);
  upload_weights(tr);
}


void  free_transformer(Transformer *tr) {
  cudaFree(tr->dev_embeddings);
  cudaFree(tr->dev_wrmsattn);
  cudaFree(tr->dev_wrmsffn);
  cudaFree(tr->dev_wrmsfinal);
  cudaFree(tr->dev_wq);
  cudaFree(tr->dev_wk);
  cudaFree(tr->dev_wv);
  cudaFree(tr->dev_wo);
  cudaFree(tr->dev_w1);
  cudaFree(tr->dev_w2);
  cudaFree(tr->dev_w3);
  if (tr->data != MAP_FAILED) {
    munmap(tr->data, tr->fsize);
  }
  if (tr->fd != -1) {
    close(tr->fd);
  }
  free_state(&tr->s);
}


// o[i] = x[i] * w[i] * (1/sqrt(mean(x^2)+eps)); o and x may alias;
// single block, whole rmsnorm on-device (no host round-trip)
__global__ void rmsnorm_kernel(float *o, const float *x, const float *w, int size) {
  extern __shared__ float red[];
  float sum = 0.0f;
  for (int i=threadIdx.x; i<size; i+=blockDim.x) {
    sum += x[i] * x[i];
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int i=blockDim.x/2; i>0; i>>=1) {
    if (threadIdx.x < i) {
      red[threadIdx.x] += red[threadIdx.x+i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float s = 1.0 / sqrtf(red[0] / (float)size + 1e-5f);
    red[0] = s;
  }
  __syncthreads();
  float s = red[0];
  for (int i=threadIdx.x; i<size; i+=blockDim.x) {
    o[i] = x[i] * w[i] * s;
  }
}


// out[i] = sum_j w[i*n + j] * x[j]; one block per output row, n elements
__global__ void matmul_kernel(float *o, const float *w, const float *x, int n, int d) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const float *wi = w + (size_t)i * n;
  float v = 0.0f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    v += wi[j] * x[j];
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    o[i] = red[0];
  }
}


// o[i] += sum_j w[i*n + j] * x[j]; residual epilogue fused (one block per row)
__global__ void matmul_axpy_kernel(float *o, const float *w, const float *x, int n, int d) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const float *wi = w + (size_t)i * n;
  float v = 0.0f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    v += wi[j] * x[j];
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    o[i] += red[0];
  }
}


// fused q/k/v projection: rows 0..dim-1 -> q, dim..dim+kvdim-1 -> k, rest -> v
__global__ void qkv_kernel(float *q, float *k, float *v,
                           const float *wq, const float *wk, const float *wv,
                           const float *x, int dim, int kvdim) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const float *w;
  float *o;
  if (i < dim) {
    w = wq; o = q;
  } else if (i < dim + kvdim) {
    w = wk; o = k; i -= dim;
  } else {
    w = wv; o = v; i -= dim + kvdim;
  }
  const float *wi = w + (size_t)i * dim;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < dim; j += blockDim.x) {
    acc += wi[j] * x[j];
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    o[i] = red[0];
  }
}


// fused ffn gate/up projection: rows 0..ffndim-1 -> h (w1), rest -> h1 (w3)
__global__ void ffn_gate_kernel(float *h, float *h1,
                                const float *w1, const float *w3,
                                const float *x, int dim, int ffndim) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const float *w;
  float *o;
  if (i < ffndim) {
    w = w1; o = h;
  } else {
    w = w3; o = h1; i -= ffndim;
  }
  const float *wi = w + (size_t)i * dim;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < dim; j += blockDim.x) {
    acc += wi[j] * x[j];
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    o[i] = red[0];
  }
}


// softmax of `rows` contiguous rows of length `size`, one block per row
__global__ void softmax_rows_kernel(float *x, int rows, int size) {
  extern __shared__ float red[];
  float *xr = x + (size_t)blockIdx.x * size;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    m = fmaxf(m, xr[i]);
  }
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    }
    __syncthreads();
  }
  __syncthreads();
  float maxv = red[0];
  float v = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    v += expf(xr[i] - maxv);
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  __syncthreads();
  float inv = 1.0f / red[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    xr[i] = expf(xr[i] - maxv) * inv;
  }
}


// o[i] += x[i]
__global__ void axpy_kernel(float *o, const float *x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    o[idx] += x[idx];
  }
}


// SiLU(gate) * up, elementwise in place
__global__ void silu_mul_kernel(float *h, const float *h1, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = h[idx];
    val *= (1.0f / (1.0f + expf(-val)));
    h[idx] = val * h1[idx];
  }
}


// apply rotary embeddings to q (and to k for the first kvdim/2 pairs),
// one thread per pair (2i, 2i+1)
__global__ void rope_kernel(float *q, float *k, int dim, int kvdim, int hsize, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 2 >= dim) return;
  int hdim = 2 * (i % (hsize / 2));
  float freq = 1.0f / powf(10000.0f, hdim / (float)hsize);
  float val = (float)pos * freq;
  float fcr = cosf(val);
  float fci = sinf(val);
  float v0 = q[2 * i], v1 = q[2 * i + 1];
  q[2 * i]     = v0 * fcr - v1 * fci;
  q[2 * i + 1] = v0 * fci + v1 * fcr;
  if (2 * i < kvdim) {
    v0 = k[2 * i]; v1 = k[2 * i + 1];
    k[2 * i]     = v0 * fcr - v1 * fci;
    k[2 * i + 1] = v0 * fci + v1 * fcr;
  }
}


// attn[h][t] = dot(q_head, k_head) / sqrt(hsize); one block per (head, t)
__global__ void attn_score_kernel(float *attn, const float *q, const float *kcache, int nheads, int pos, int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  int t = blockIdx.y;
  const float *qh = q + h * hsize;
  const float *kh = kcache + t * kvdim + (h / kvmul) * hsize;
  float v = 0.0f;
  for (int i = threadIdx.x; i < hsize; i += blockDim.x) {
    v += qh[i] * kh[i];
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    attn[h * (pos + 1) + t] = red[0] / sqrtf((float)hsize);
  }
}


// x1[head, i] = sum_t attn[h][t] * v[head, t, i]; one block per head
__global__ void attn_value_kernel(float *x1, const float *attn, const float *vcache, int nheads, int pos, int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  int i = blockIdx.y;
  const float *ah = attn + h * (pos + 1);
  float v = 0.0f;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    v += ah[t] * vcache[(size_t)t * kvdim + (h / kvmul) * hsize + i];
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    x1[h * hsize + i] = red[0];
  }
}


float *forward(Transformer *tr, int token, int pos) {
  Config *c = &tr->c;
  State *s = &tr->s;
  float *x = s->x;
  int dim = c->dim;
  int kvdim = dim * c->nkvheads / c->nheads;
  int kvmul = c->nheads / c->nkvheads;
  int ffndim = c->ffndim;
  int hsize = dim / c->nheads;
  int threads = 256;
  size_t shared = threads * sizeof(float);

  {
    float *content = tr->w.embeddings + (size_t)token * dim;
    CUDA_CHECK(cudaMemcpy(x, content, sizeof(float) * dim, cudaMemcpyHostToDevice));
  }

  for (unsigned long long l = 0; l < c->nlayers; l++) {
    // 1. self-attention sublayer
    rmsnorm_kernel<<<1, threads, shared>>>(s->x1, x, tr->dev_wrmsattn + l * dim, dim);
    float *k = s->kcache + l * (size_t)c->ncontext * kvdim + pos * kvdim;
    float *v = s->vcache + l * (size_t)c->ncontext * kvdim + pos * kvdim;
    qkv_kernel<<<dim + 2 * kvdim, threads, shared>>>(s->q, k, v,
        tr->dev_wq + l * (size_t)dim * dim, tr->dev_wk + l * (size_t)dim * kvdim,
        tr->dev_wv + l * (size_t)dim * kvdim, s->x1, dim, kvdim);
    {
      dim3 blocks(dim / 2);
      rope_kernel<<<blocks, threads>>>(s->q, k, dim, kvdim, hsize, pos);
    }
    {
      dim3 blocks(c->nheads, pos + 1);
      attn_score_kernel<<<blocks, threads, shared>>>(s->attn, s->q, s->kcache + l * (size_t)c->ncontext * kvdim, c->nheads, pos, hsize, kvdim, kvmul);
    }
    softmax_rows_kernel<<<c->nheads, threads, shared>>>(s->attn, c->nheads, pos + 1);
    {
      dim3 blocks(c->nheads, hsize);
      attn_value_kernel<<<blocks, threads, shared>>>(s->x1, s->attn, s->vcache + l * (size_t)c->ncontext * kvdim, c->nheads, pos, hsize, kvdim, kvmul);
    }
    matmul_axpy_kernel<<<dim, threads, shared>>>(x, tr->dev_wo + l * (size_t)dim * dim, s->x1, dim, dim);

    // 2. ffn sublayer
    rmsnorm_kernel<<<1, threads, shared>>>(s->x1, x, tr->dev_wrmsffn + l * dim, dim);
    ffn_gate_kernel<<<2 * ffndim, threads, shared>>>(s->h, s->h1,
        tr->dev_w1 + l * (size_t)dim * ffndim, tr->dev_w3 + l * (size_t)dim * ffndim,
        s->x1, dim, ffndim);
    silu_mul_kernel<<<(ffndim + threads - 1) / threads, threads>>>(s->h, s->h1, ffndim);
    matmul_axpy_kernel<<<dim, threads, shared>>>(x, tr->dev_w2 + l * (size_t)ffndim * dim, s->h, ffndim, dim);
  }

  rmsnorm_kernel<<<1, threads, shared>>>(x, x, tr->dev_wrmsfinal, dim);
  matmul_kernel<<<c->nvocab, threads, shared>>>(s->logits, tr->dev_embeddings, x, dim, c->nvocab);
  return s->logits;
}


typedef struct {
  const char *str;
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


void build_tokenizer(Tokenizer *t, const char *path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char **)malloc(vocab_size * sizeof(char*));
  t->scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted = NULL;
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i*2] = (unsigned char)i;
    t->byte_pieces[i*2+1] = '\0';
  }
  FILE *f = fopen(path, "rb");
  if (!f) { mexit("Couldn't load file"); }
  if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
    fexit(f, "failed read file");
  }
  for (int i = 0; i < vocab_size; i++) {
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
  for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
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


int str_lookup(const char *str, TokenIndex *sorted, int vocab_size) {
  TokenIndex tok = { .str = str };
  TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}


void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *ntokens) {
  if (!text) mexit("cannot encode NULL text");
  if (!t->sorted) {
    t->sorted = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted[i].str = t->vocab[i];
      t->sorted[i].id = i;
    }
    qsort(t->sorted, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }
  char *strbuf = (char*)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t strlen = 0;
  int n = 0;
  if (bos) tokens[n++] = 1;
  if (text[0] != '\0') {
    int dummy_suffix = str_lookup(" ", t->sorted, t->vocab_size);
    tokens[n++] = dummy_suffix;
  }
  for (const char *c = text; *c != '\0'; c++) {
    if ((*c & 0xC0) != 0x80) strlen = 0;
    strbuf[strlen++] = *c;
    strbuf[strlen] = '\0';
    if ((*(c+1) & 0xC0) == 0x80 && strlen < 4) continue;
    int id = str_lookup(strbuf, t->sorted, t->vocab_size);
    if (id != -1) {
      tokens[n++] = id;
    } else {
      for (int i = 0; i < strlen; i++) {
        tokens[n++] = (unsigned char)strbuf[i] + 3;
      }
    }
    strlen = 0;
  }
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    for (int i = 0; i < (n-1); i++) {
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
    for (int i = best_idx + 1; i < n - 1; i++) {
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
  for (int i = 1; i < n; i++) {
    if (p[i] > maxp) {
      maxi = i;
      maxp = p[i];
    }
  }
  return maxi;
}


int sample_mult(float *p, int n, float coin) {
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += p[i];
    if (coin < cdf) return i;
  }
  return n-1;
}


int compare(const void *a, const void *b) {
  ProbIndex *pa = (ProbIndex*)a;
  ProbIndex *pb = (ProbIndex*)b;
  if (pa->prob > pb->prob) return -1;
  if (pa->prob < pb->prob) return 1;
  return 0;
}


int sample_topp(float *p, int n, float topp, ProbIndex *pi, float coin) {
  int n0 = 0;
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (p[i] >= cutoff) {
      pi[n0].index = i;
      pi[n0++].prob = p[i];
    }
  }
  qsort(pi, n0, sizeof(ProbIndex), compare);
  float sum = 0.0f;
  int last_idx = n0 - 1;
  for (int i = 0; i < n0; i++) {
    sum += pi[i].prob;
    if (sum > topp) {
      last_idx = i;
      break;
    }
  }
  float r = coin * sum;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += pi[i].prob;
    if (r < cdf) return pi[i].index;
  }
  return pi[last_idx].index;
}


void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
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
    for (int i = 0; i < sampler->vocab_size; i++) {
      logits[i] /= sampler->temperature;
    }
    {
      float maxv = logits[0];
      for (int i = 1; i < sampler->vocab_size; i++) {
        if (logits[i] > maxv) maxv = logits[i];
      }
      float sum = 0.0f;
      for (int i = 0; i < sampler->vocab_size; i++) {
        logits[i] = expf(logits[i] - maxv);
        sum += logits[i];
      }
      for (int i = 0; i < sampler->vocab_size; i++) {
        logits[i] /= sum;
      }
    }
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


static int dump_done = 0;

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
  const char *empty_prompt = "";
  if (!prompt) prompt = empty_prompt;
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    mexit("something is wrong, expected at least 1 prompt token");
  }
  float *host_logits = (float*)malloc(sizeof(float) * transformer->c.nvocab);
  long start = 0;
  int next;
  int token = prompt_tokens[0];
  int pos = 0;
  while (pos < steps) {
    float *logits = forward(transformer, token, pos);
    CUDA_CHECK(cudaMemcpy(host_logits, logits, sizeof(float) * transformer->c.nvocab, cudaMemcpyDeviceToHost));
    if (pos == 0 && getenv("CUDADUMP") && !dump_done) {
      dump_done = 1;
      for (int i = 0; i < 12; i++) fprintf(stderr, "CUDALOGIT %.6f ", host_logits[i]);
      fprintf(stderr, "\n");
    }
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      next = sample(sampler, host_logits);
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
    fprintf(stderr, "\ntotal %d tokens, speed %.1f token/s\n\n\n", pos - 1, (pos - 1) / (double)(end - start) * 1000);
  }
  free(host_logits);
  free(prompt_tokens);
}


int main(int argc, char *argv[]) {
  const char *checkpoint_path = "stories15M.bin";
  const char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  const char *prompt = NULL;
  unsigned long long rng_seed = (unsigned int)time(NULL);
  if (getenv("RUNCUDA_SEED")) rng_seed = (unsigned long long)atoll(getenv("RUNCUDA_SEED"));
  if (argc >= 2) {
    checkpoint_path = argv[1];
  }
  Transformer transformer;
  memset(&transformer, 0, sizeof(transformer));
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
