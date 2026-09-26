
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
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
  int8_t *dev_xq_q;
  float *dev_xq_s;
  int8_t *dev_hq_q;
  float *dev_hq_s;
  // device weights
  float *dev_embeddings;
  float *dev_wrmsattn;
  float *dev_wrmsffn;
  float *dev_wrmsfinal;
  int8_t *dev_wq;
  float *dev_wq_s;
  int8_t *dev_wk;
  float *dev_wk_s;
  int8_t *dev_wv;
  float *dev_wv_s;
  int8_t *dev_wo;
  float *dev_wo_s;
  int8_t *dev_w1;
  float *dev_w1_s;
  int8_t *dev_w2;
  float *dev_w2_s;
  int8_t *dev_w3;
  float *dev_w3_s;
  int8_t *dev_qtok;
  float *dev_qtok_s;
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
  CUDA_CHECK(cudaMalloc(&tr->dev_xq_q, sizeof(int8_t) * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_xq_s, sizeof(float) * dim / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_hq_q, sizeof(int8_t) * ffndim));
  CUDA_CHECK(cudaMalloc(&tr->dev_hq_s, sizeof(float) * ffndim / GS));
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


void free_state(Transformer *tr, State *s) {
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
  cudaFree(tr->dev_xq_q);
  cudaFree(tr->dev_xq_s);
  cudaFree(tr->dev_hq_q);
  cudaFree(tr->dev_hq_s);
}


// pull x from the device, quantize on the host, push the result back
void quantize_upload(Transformer *tr, const float *dx, int n,
                     int8_t **dq, float **ds) {
  float hostbuf[8192];
  int8_t hq[8192];
  float hs[8192 / GS];
  if (n > (int)(sizeof(hostbuf) / sizeof(float))) mexit("quantize_upload: n too big");
  CUDA_CHECK(cudaMemcpy(hostbuf, dx, sizeof(float) * n, cudaMemcpyDeviceToHost));
  QuantizedTensor t = { .q = hq, .s = hs };
  int ngroups = n / GS;
  float Q_MAX = 127.0f;
  for (int g = 0; g < ngroups; g++) {
    float *px = hostbuf + g * GS;
    int8_t *pq = t.q + g * GS;
    float max = 0.0f;
    for (int i = 0; i < GS; i++) {
      float v = fabsf(px[i]);
      if (v > max) max = v;
    }
    float s = max / Q_MAX;
    for (int i = 0; i < GS; i++) {
      pq[i] = (int8_t)roundf(px[i] / s);
    }
    t.s[g] = s;
  }
  CUDA_CHECK(cudaMemcpy(*dq, t.q, sizeof(int8_t) * n, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*ds, t.s, sizeof(float) * n / GS, cudaMemcpyHostToDevice));
}


QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
  void *p = *ptr;
  QuantizedTensor *qts = (QuantizedTensor*)malloc(n * sizeof(QuantizedTensor));
  for (int i = 0; i < n; i++) {
    qts[i].q = (int8_t*)p;
    p = (int8_t*)p + size_each;
    qts[i].s = (float*)p;
    p = (float*)p + size_each / GS;
  }
  *ptr = p;
  return qts;
}


void mmap_weights(Weights *w, Config *c, void *ptr) {
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
  w->q_tokens = init_quantized_tensors(&ptr, 1, c->nvocab * c->dim);
  w->embeddings = (float*)malloc(c->nvocab * dim * sizeof(float));
  for (int i = 0; i < c->nvocab * dim; i++) {
    w->embeddings[i] = w->q_tokens->q[i] * w->q_tokens->s[i / GS];
  }
  w->wq = init_quantized_tensors(&ptr, c->nlayers, dim * dim);
  w->wk = init_quantized_tensors(&ptr, c->nlayers, dim * kvdim);
  w->wv = init_quantized_tensors(&ptr, c->nlayers, dim * kvdim);
  w->wo = init_quantized_tensors(&ptr, c->nlayers, dim * dim);
  w->w1 = init_quantized_tensors(&ptr, c->nlayers, dim * ffndim);
  w->w2 = init_quantized_tensors(&ptr, c->nlayers, ffndim * dim);
  w->w3 = init_quantized_tensors(&ptr, c->nlayers, dim * ffndim);
}


void read_checkpoint(const char *path, Config *c, Weights *w, int *fd, float **data, ssize_t *fsize) {
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
    fexit(f, "Can't read checkpoint version");
  }
  if (version != 2) {
    fexit(f, "version should be 2");
  }
  if (fread(c, sizeof(Config), 1, f) != 1) {
    fexit(f, "Can't read checkpoint config");
  }
  c->nvocab = abs(c->nvocab);
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
  *data = (float*)mmap(NULL, *fsize, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) { mexit("mmap failed!"); }
  void *weights = ((char*)*data) + header_size;
  mmap_weights(w, c, weights);
}


void upload_weights(Transformer *tr) {
  Config *c = &tr->c;
  int dim = c->dim;
  int ffndim = c->ffndim;
  int head_size = dim / c->nheads;
  int kvdim = head_size * c->nkvheads;
  unsigned long long nlayers = c->nlayers;
  size_t dim2 = (size_t)dim * dim;
  size_t dimkv = (size_t)dim * kvdim;
  size_t dimff = (size_t)dim * ffndim;
  size_t ffn_dim = (size_t)ffndim * dim;
  CUDA_CHECK(cudaMalloc(&tr->dev_embeddings, sizeof(float) * (size_t)c->nvocab * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsattn, sizeof(float) * (size_t)nlayers * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsffn, sizeof(float) * (size_t)nlayers * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wrmsfinal, sizeof(float) * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_wq, sizeof(int8_t) * (size_t)nlayers * dim2));
  CUDA_CHECK(cudaMalloc(&tr->dev_wq_s, sizeof(float) * (size_t)nlayers * dim2 / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_wk, sizeof(int8_t) * (size_t)nlayers * dimkv));
  CUDA_CHECK(cudaMalloc(&tr->dev_wk_s, sizeof(float) * (size_t)nlayers * dimkv / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_wv, sizeof(int8_t) * (size_t)nlayers * dimkv));
  CUDA_CHECK(cudaMalloc(&tr->dev_wv_s, sizeof(float) * (size_t)nlayers * dimkv / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_wo, sizeof(int8_t) * (size_t)nlayers * dim2));
  CUDA_CHECK(cudaMalloc(&tr->dev_wo_s, sizeof(float) * (size_t)nlayers * dim2 / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_w1, sizeof(int8_t) * (size_t)nlayers * dimff));
  CUDA_CHECK(cudaMalloc(&tr->dev_w1_s, sizeof(float) * (size_t)nlayers * dimff / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_w2, sizeof(int8_t) * (size_t)nlayers * ffn_dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_w2_s, sizeof(float) * (size_t)nlayers * ffn_dim / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_w3, sizeof(int8_t) * (size_t)nlayers * dimff));
  CUDA_CHECK(cudaMalloc(&tr->dev_w3_s, sizeof(float) * (size_t)nlayers * dimff / GS));
  CUDA_CHECK(cudaMalloc(&tr->dev_qtok, sizeof(int8_t) * (size_t)c->nvocab * dim));
  CUDA_CHECK(cudaMalloc(&tr->dev_qtok_s, sizeof(float) * (size_t)c->nvocab * dim / GS));
  CUDA_CHECK(cudaMemcpy(tr->dev_embeddings, tr->w.embeddings, sizeof(float) * (size_t)c->nvocab * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsattn, tr->w.wrmsattn, sizeof(float) * (size_t)nlayers * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsffn, tr->w.wrmsffn, sizeof(float) * (size_t)nlayers * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_wrmsfinal, tr->w.wrmsfinal, sizeof(float) * dim, cudaMemcpyHostToDevice));
  // Host layout from init_quantized_tensors is per-layer interleaved:
  // [l0.q][l0.s][l1.q][l1.s]...  — copy each layer's q and s separately.
  for (unsigned long long l = 0; l < nlayers; l++) {
    CUDA_CHECK(cudaMemcpy(tr->dev_wq + l * dim2, tr->w.wq[l].q, sizeof(int8_t) * dim2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wq_s + l * dim2 / GS, tr->w.wq[l].s, sizeof(float) * dim2 / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wk + l * dimkv, tr->w.wk[l].q, sizeof(int8_t) * dimkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wk_s + l * dimkv / GS, tr->w.wk[l].s, sizeof(float) * dimkv / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wv + l * dimkv, tr->w.wv[l].q, sizeof(int8_t) * dimkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wv_s + l * dimkv / GS, tr->w.wv[l].s, sizeof(float) * dimkv / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wo + l * dim2, tr->w.wo[l].q, sizeof(int8_t) * dim2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_wo_s + l * dim2 / GS, tr->w.wo[l].s, sizeof(float) * dim2 / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w1 + l * dimff, tr->w.w1[l].q, sizeof(int8_t) * dimff, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w1_s + l * dimff / GS, tr->w.w1[l].s, sizeof(float) * dimff / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w2 + l * ffn_dim, tr->w.w2[l].q, sizeof(int8_t) * ffn_dim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w2_s + l * ffn_dim / GS, tr->w.w2[l].s, sizeof(float) * ffn_dim / GS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w3 + l * dimff, tr->w.w3[l].q, sizeof(int8_t) * dimff, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tr->dev_w3_s + l * dimff / GS, tr->w.w3[l].s, sizeof(float) * dimff / GS, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(tr->dev_qtok, tr->w.q_tokens->q, sizeof(int8_t) * (size_t)c->nvocab * dim, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(tr->dev_qtok_s, tr->w.q_tokens->s, sizeof(float) * (size_t)c->nvocab * dim / GS, cudaMemcpyHostToDevice));
}


void build_transformer(Transformer *tr, const char *path) {
  read_checkpoint(path, &tr->c, &tr->w, &tr->fd, &tr->data, &tr->fsize);
  malloc_state(tr, &tr->s, &tr->c);
  upload_weights(tr);
}


void free_transformer(Transformer *tr) {
  cudaFree(tr->dev_embeddings);
  cudaFree(tr->dev_wrmsattn);
  cudaFree(tr->dev_wrmsffn);
  cudaFree(tr->dev_wrmsfinal);
  cudaFree(tr->dev_wq);
  cudaFree(tr->dev_wq_s);
  cudaFree(tr->dev_wk);
  cudaFree(tr->dev_wk_s);
  cudaFree(tr->dev_wv);
  cudaFree(tr->dev_wv_s);
  cudaFree(tr->dev_wo);
  cudaFree(tr->dev_wo_s);
  cudaFree(tr->dev_w1);
  cudaFree(tr->dev_w1_s);
  cudaFree(tr->dev_w2);
  cudaFree(tr->dev_w2_s);
  cudaFree(tr->dev_w3);
  cudaFree(tr->dev_w3_s);
  cudaFree(tr->dev_qtok);
  cudaFree(tr->dev_qtok_s);
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
  free_state(tr, &tr->s);
}


// *o[i] = w[i] * x[i] * ss, ss = 1/sqrt(mean(x^2)+eps); o and x may alias
__global__ void rms_scale_kernel(float *o, const float *x, const float *w, int size, float ss) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    o[idx] = w[idx] * ss * x[idx];
  }
}


// x[i] -= val
__global__ void sub_kernel(float *x, int size, float val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x[idx] -= val;
  }
}


// x[i] = expf(x[i])
__global__ void exp_kernel(float *x, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x[idx] = expf(x[idx]);
  }
}


// x[i] /= val
__global__ void div_kernel(float *x, int size, float val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    x[idx] /= val;
  }
}


void rmsnorm(float *o, float *x, float *w, int size) {
  float hostbuf[8192];
  float ss = 0.0f;
  if (size > (int)(sizeof(hostbuf) / sizeof(float))) mexit("rmsnorm: size too big");
  CUDA_CHECK(cudaMemcpy(hostbuf, x, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; i++) {
    ss += hostbuf[i] * hostbuf[i];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    rms_scale_kernel<<<blocks, threads>>>(o, x, w, size, ss);
  }
}


// softmax in place; reductions done on the host (sizes are small here)
void softmax(float *x, int size) {
  float hostbuf[8192];
  float maxv;
  float sum;
  if (size > (int)(sizeof(hostbuf) / sizeof(float))) mexit("softmax: size too big");
  CUDA_CHECK(cudaMemcpy(hostbuf, x, sizeof(float) * size, cudaMemcpyDeviceToHost));
  maxv = hostbuf[0];
  for (int i = 1; i < size; i++) {
    if (hostbuf[i] > maxv) maxv = hostbuf[i];
  }
  sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += expf(hostbuf[i] - maxv);
  }
  {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sub_kernel<<<blocks, threads>>>(x, size, maxv);
    exp_kernel<<<blocks, threads>>>(x, size);
    div_kernel<<<blocks, threads>>>(x, size, sum);
  }
}




// out[i] = sum_g (sum_k wq[in+j+k]*xq[j+k]) * ws[(in+j)/GS] * xs[j/GS]
// one block per output row; n and d must be divisible by GS
__global__ void qmatmul_kernel(float *o, const int8_t *wq, const float *ws,
                              const int8_t *xq, const float *xs, int n, int d, int gs) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  int nfull = (n / gs) * gs; // same as the CPU reference: remainder groups are skipped
  float v = 0.0f;
  for (int j = threadIdx.x * gs; j < nfull; j += blockDim.x * gs) {
    int in = i * n;
    int32_t iv = 0;
    for (int k = 0; k < gs; k++) {
      iv += (int32_t)wq[in + j + k] * (int32_t)xq[j + k];
    }
    v += (float)iv * ws[(in + j) / gs] * xs[j / gs];
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


__global__ void axpy_kernel(float *o, const float *x, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    o[idx] += x[idx];
  }
}


__global__ void silu_mul_kernel(float *h, const float *h1, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = h[idx];
    val *= (1.0f / (1.0f + expf(-val)));
    h[idx] = val * h1[idx];
  }
}


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


__global__ void attn_score_kernel(float *attn, const float *q, const float *kcache, int nheads, int pos, int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  int t = blockIdx.y;
  const float *qh = q + h * hsize;
  const float *kh = kcache + (size_t)t * kvdim + (h / kvmul) * hsize;
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
    attn[(size_t)h * (pos + 1) + t] = red[0] / sqrtf((float)hsize);
  }
}


__global__ void attn_value_kernel(float *x1, const float *attn, const float *vcache, int nheads, int pos, int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  int i = blockIdx.y;
  const float *ah = attn + (size_t)h * (pos + 1);
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
  float *x = tr->dev_x;
  int dim = c->dim;
  int kvdim = dim * c->nkvheads / c->nheads;
  int kvmul = c->nheads / c->nkvheads;
  int ffndim = c->ffndim;
  int hsize = dim / c->nheads;
  int threads = 256;
  size_t dim2 = (size_t)dim * dim;
  size_t dimkv = (size_t)dim * kvdim;
  size_t dimff = (size_t)dim * ffndim;
  size_t ffn_dim = (size_t)ffndim * dim;

  {
    float *content = tr->dev_embeddings + (size_t)token * dim;
    CUDA_CHECK(cudaMemcpy(x, content, sizeof(float) * dim, cudaMemcpyDeviceToDevice));
  }

  for (unsigned long long l = 0; l < c->nlayers; l++) {
    // 1. self-attention sublayer
    rmsnorm(tr->dev_x1, x, tr->dev_wrmsattn + l * dim, dim);
    float *k = tr->dev_kcache + l * (size_t)c->ncontext * kvdim + pos * kvdim;
    float *v = tr->dev_vcache + l * (size_t)c->ncontext * kvdim + pos * kvdim;
    quantize_upload(tr, tr->dev_x1, dim, &tr->dev_xq_q, &tr->dev_xq_s);
    qmatmul_kernel<<<dim, threads, threads * sizeof(float)>>>(tr->dev_q, tr->dev_wq + l * dim2, tr->dev_wq_s + l * dim2 / GS, tr->dev_xq_q, tr->dev_xq_s, dim, dim, GS);
    qmatmul_kernel<<<kvdim, threads, threads * sizeof(float)>>>(k, tr->dev_wk + l * dimkv, tr->dev_wk_s + l * dimkv / GS, tr->dev_xq_q, tr->dev_xq_s, dim, kvdim, GS);
    qmatmul_kernel<<<kvdim, threads, threads * sizeof(float)>>>(v, tr->dev_wv + l * dimkv, tr->dev_wv_s + l * dimkv / GS, tr->dev_xq_q, tr->dev_xq_s, dim, kvdim, GS);
    {
      dim3 blocks(dim / 2);
      rope_kernel<<<blocks, threads>>>(tr->dev_q, k, dim, kvdim, hsize, pos);
    }
    {
      dim3 blocks(c->nheads, pos + 1);
      attn_score_kernel<<<blocks, threads, threads * sizeof(float)>>>(tr->dev_attn, tr->dev_q, tr->dev_kcache + l * (size_t)c->ncontext * kvdim, c->nheads, pos, hsize, kvdim, kvmul);
    }
    for (int h = 0; h < c->nheads; h++) {
      softmax(tr->dev_attn + h * (pos + 1), pos + 1);
    }
    {
      dim3 blocks(c->nheads, hsize);
      attn_value_kernel<<<blocks, threads, threads * sizeof(float)>>>(tr->dev_x1, tr->dev_attn, tr->dev_vcache + l * (size_t)c->ncontext * kvdim, c->nheads, pos, hsize, kvdim, kvmul);
    }
    quantize_upload(tr, tr->dev_x1, dim, &tr->dev_xq_q, &tr->dev_xq_s);
    qmatmul_kernel<<<dim, threads, threads * sizeof(float)>>>(tr->dev_x2, tr->dev_wo + l * dim2, tr->dev_wo_s + l * dim2 / GS, tr->dev_xq_q, tr->dev_xq_s, dim, dim, GS);
    axpy_kernel<<<(dim + threads - 1) / threads, threads>>>(x, tr->dev_x2, dim);

    // 2. ffn sublayer
    rmsnorm(tr->dev_x1, x, tr->dev_wrmsffn + l * dim, dim);
    quantize_upload(tr, tr->dev_x1, dim, &tr->dev_xq_q, &tr->dev_xq_s);
    qmatmul_kernel<<<ffndim, threads, threads * sizeof(float)>>>(tr->dev_h, tr->dev_w1 + l * dimff, tr->dev_w1_s + l * dimff / GS, tr->dev_xq_q, tr->dev_xq_s, dim, ffndim, GS);
    qmatmul_kernel<<<ffndim, threads, threads * sizeof(float)>>>(tr->dev_h1, tr->dev_w3 + l * dimff, tr->dev_w3_s + l * dimff / GS, tr->dev_xq_q, tr->dev_xq_s, dim, ffndim, GS);
    silu_mul_kernel<<<(ffndim + threads - 1) / threads, threads>>>(tr->dev_h, tr->dev_h1, ffndim);
    quantize_upload(tr, tr->dev_h, ffndim, &tr->dev_hq_q, &tr->dev_hq_s);
    qmatmul_kernel<<<dim, threads, threads * sizeof(float)>>>(tr->dev_x1, tr->dev_w2 + l * ffn_dim, tr->dev_w2_s + l * ffn_dim / GS, tr->dev_hq_q, tr->dev_hq_s, ffndim, dim, GS);
    axpy_kernel<<<(dim + threads - 1) / threads, threads>>>(x, tr->dev_x1, dim);
  }

  rmsnorm(x, x, tr->dev_wrmsfinal, dim);
  quantize_upload(tr, x, dim, &tr->dev_xq_q, &tr->dev_xq_s);
  qmatmul_kernel<<<c->nvocab, threads, threads * sizeof(float)>>>(tr->dev_logits, tr->dev_qtok, tr->dev_qtok_s, tr->dev_xq_q, tr->dev_xq_s, dim, c->nvocab, GS);
  return tr->dev_logits;
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
    fprintf(stderr, "\ntotal %d tokens, speed %.1f tok/s\n\n\n", pos - 1, (pos - 1) / (double)(end - start) * 1000);
  }
  free(host_logits);
  free(prompt_tokens);
}


int main(int argc, char *argv[]) {
  const char *checkpoint_path = "stories15M-q8.bin";
  const char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  char *prompt = NULL;
  unsigned long long rng_seed = (unsigned int)time(NULL);
  if (getenv("RUNQCUDA_SEED")) rng_seed = (unsigned long long)atoll(getenv("RUNQCUDA_SEED"));
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
