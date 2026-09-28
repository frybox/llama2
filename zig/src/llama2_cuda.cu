// CUDA kernels for the llama2 CUDA implementation, shared with the zig host
// (src/llama2_cuda.zig). Each kernel body is a verbatim copy of the matching
// __global__ from c/llama2_cuda.cu; each is exposed to the zig host through a
// flat extern "C" launcher (zig cannot form <<< >>> launches), plus the
// device-side top-p sampling (prep + thrust sort + pick).

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define LLAMA2_CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { printf("CUDA error %s (line %d)\n", cudaGetErrorString(_e), __LINE__); exit(1); } \
} while (0)


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

__global__ void init_rmsnorm_kernel(float *x, float *x1, const float *emb, const float *w, int dim) {
  extern __shared__ float red[];
  float sum = 0.0f;
  for (int i=threadIdx.x; i<dim; i+=blockDim.x) sum += emb[i]*emb[i];
  red[threadIdx.x]=sum; __syncthreads();
  for (int i=blockDim.x/2;i>0;i>>=1){ if(threadIdx.x<i) red[threadIdx.x]+=red[threadIdx.x+i]; __syncthreads(); }
  if (threadIdx.x==0) red[0]=1.0f/sqrtf(red[0]/(float)dim+1e-5f);
  __syncthreads();
  float s=red[0];
  for (int i=threadIdx.x;i<dim;i+=blockDim.x){ float e=emb[i]; x[i]=e; x1[i]=e*w[i]*s; }
}

__global__ void sample_prep_kernel(float *e, int *idx, float *out_S, const float *logits, int n, float inv_temp) {
  // inv_temp = 1/temperature passed in: FP32 division per element is the kernel's
  // bottleneck; the multiply is bit-exact enough (verified: identical tokens).
  extern __shared__ float red[];
  float maxv = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float l = logits[i] * inv_temp;
    if (l > maxv) maxv = l;
  }
  red[threadIdx.x] = maxv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  float m = red[0];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float ev = expf(logits[i] * inv_temp - m);
    e[i] = ev;
    idx[i] = i;
    sum += ev;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out_S = red[0];
}

__global__ void sample_pick_kernel(int *next, const float *e, const int *idx, int n,
                                   float topp, float coin, const float *out_S) {
  if (threadIdx.x == 0) {
    float S = *out_S;
    float cutoff_e = (1.0f - topp) / (n - 1) * S;
    float slast = 0.0f;
    for (int i = n - 1; i >= 0; i--) {
      if (e[i] < cutoff_e) break;
      slast += e[i];
      if (slast > topp * S) break;
    }
    float r = coin * slast;
    float cum = 0.0f;
    int token = 0;
    for (int i = n - 1; i >= 0; i--) {
      if (e[i] < cutoff_e) break;
      cum += e[i];
      token = idx[i];
      if (r < cum) break;
    }
    *next = token;
  }
}

// ------------------------------------------------------------------
// extern "C" launchers for the zig host (zig cannot form <<< >>> launches).

extern "C" void llama2_cuda_init(float* x, float* x1, const float* emb, const float* w, int dim, cudaStream_t stream) {
  const int threads = 256;
  init_rmsnorm_kernel<<<1, threads, threads * sizeof(float), stream>>>(x, x1, emb, w, dim);
}

extern "C" void llama2_cuda_rmsnorm(float* o, const float* x, const float* w, int n, cudaStream_t stream) {
  const int threads = 256;
  rmsnorm_kernel<<<1, 256, 256*sizeof(float), stream>>>(o, x, w, n);
}

extern "C" void llama2_cuda_qkv(float* q, float* k, float* v, const float* wq, const float* wk, const float* wv, const float* x, int dim, int kvdim, cudaStream_t stream) {
  const int threads = 256;
  qkv_kernel<<<dim + 2*kvdim, threads, threads*sizeof(float), stream>>>(q, k, v, wq, wk, wv, x, dim, kvdim);
}

extern "C" void llama2_cuda_rope(float* q, float* k, int dim, int kvdim, int hsize, int pos, cudaStream_t stream) {
  const int threads = 256;
  rope_kernel<<<dim/2, threads, 0, stream>>>(q, k, dim, kvdim, hsize, pos);
}

extern "C" void llama2_cuda_attn_score(float* attn, const float* q, const float* kcache, int nheads, int pos, int hsize, int kvdim, int kvmul, cudaStream_t stream) {
  const int threads = 256;
  attn_score_kernel<<<dim3(nheads, pos+1), threads, threads*sizeof(float), stream>>>(attn, q, kcache, nheads, pos, hsize, kvdim, kvmul);
}

extern "C" void llama2_cuda_softmax_rows(float* attn, int nheads, int pos, cudaStream_t stream) {
  const int threads = 256;
  softmax_rows_kernel<<<nheads, threads, threads*sizeof(float), stream>>>(attn, nheads, pos+1);
}

extern "C" void llama2_cuda_attn_value(float* x1, const float* attn, const float* vcache, int nheads, int pos, int hsize, int kvdim, int kvmul, cudaStream_t stream) {
  const int threads = 256;
  attn_value_kernel<<<dim3(nheads, hsize), threads, threads*sizeof(float), stream>>>(x1, attn, vcache, nheads, pos, hsize, kvdim, kvmul);
}

extern "C" void llama2_cuda_matmul_axpy(float* o, const float* w, const float* x, int n, int d, cudaStream_t stream) {
  const int threads = 256;
  matmul_axpy_kernel<<<d, threads, threads*sizeof(float), stream>>>(o, w, x, n, d);
}

extern "C" void llama2_cuda_ffn_gate(float* h, float* h1, const float* w1, const float* w3, const float* x, int dim, int ffndim, cudaStream_t stream) {
  const int threads = 256;
  ffn_gate_kernel<<<2*ffndim, threads, threads*sizeof(float), stream>>>(h, h1, w1, w3, x, dim, ffndim);
}

extern "C" void llama2_cuda_silu_mul(float* h, const float* h1, int ffndim, cudaStream_t stream) {
  const int threads = 256;
  silu_mul_kernel<<<(ffndim+threads-1)/threads, threads, 0, stream>>>(h, h1, ffndim);
}

extern "C" void llama2_cuda_matmul(float* o, const float* w, const float* x, int n, int d, cudaStream_t stream) {
  const int threads = 256;
  matmul_kernel<<<d, threads, threads*sizeof(float), stream>>>(o, w, x, n, d);
}

// One call: device prep (inv_temp folds softmax), thrust sort by value,
// pick the token from the top-p prefix. e/idx/s_out/next are device bufs.
extern "C" void llama2_cuda_sample(float *e, int *idx, float *s_out, float *logits,
                                  int n, float inv_temp, int *next,
                                  float topp, float coin, cudaStream_t stream) {
  const int threads = 256;
  sample_prep_kernel<<<1, threads, threads * sizeof(float), stream>>>(e, idx, s_out, logits, n, inv_temp);
  thrust::sort_by_key(thrust::device_pointer_cast(e), thrust::device_pointer_cast(e + n), thrust::device_pointer_cast(idx));
  sample_pick_kernel<<<1, 1>>>(next, e, idx, n, topp, coin, s_out);
}

