// Generated verbatim from /home/wisedu/llmengine/llama2/c/llama2q_cuda.cu
// (19 __global__ kernels, function bodies unchanged) + one extern "C" launcher
// per kernel, mirroring zig/src/llama2_cuda.cu style. stream is always 0 here
// (the zig host drives synchronization with its own stream); grid/block/shared
// match the C forward() call sites. No global GS: group size is a kernel arg.
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define LLAMA2Q_CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s: %s (line %d)\n", cudaGetErrorString(err), #x, __LINE__); \
    exit(1); \
  } \
} while (0)


__global__ void quantize_kernel(const float *x, int n, int8_t *q, float *s, int gs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int g = i / gs;
  const float *pg = x + g * gs;
  float maxv = 0.0f;
  for (int j = 0; j < gs; j++) {
    float v = fabsf(pg[j]);
    if (v > maxv) maxv = v;
  }
  float sc = maxv / 127.0f;
  if (i % gs == 0) s[g] = sc; // first element of each group writes its scale (exactly once)
  q[i] = (int8_t)roundf(x[i] / sc);
}

__global__ void load_emb_kernel(float *x, const float *embeddings, int dim, const int *gp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  x[i] = embeddings[(size_t)gp[0] * dim + i];
}

__global__ void rms_scale_kernel(float *o, const float *x, const float *w, int size, float ss) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    o[idx] = w[idx] * ss * x[idx];
  }
}

__global__ void rmsnorm_kernel(float *o, const float *x, const float *w, int size) {
  extern __shared__ float red[];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    sum += x[i] * x[i];
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      red[threadIdx.x] += red[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float s = 1.0 / sqrtf(red[0] / (float)size + 1e-5f);
    red[0] = s;
  }
  __syncthreads();
  float s = red[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    o[i] = x[i] * w[i] * s;
  }
}

__global__ void softmax_rows_kernel(float *x, const int *gp, int ncontext) {
  extern __shared__ float red[];
  float *xr = x + (size_t)blockIdx.x * ncontext;
  int n = gp[1] + 1;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
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
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    v += expf(xr[i] - maxv);
  }
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  __syncthreads();
  float inv = 1.0f / red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    xr[i] = expf(xr[i] - maxv) * inv;
  }
}

__global__ void rmsnorm_quant_kernel(float *o, const float *x, const float *w, int size,
                                     int8_t *q, float *s, int gs) {
  extern __shared__ float red[];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) sum += x[i] * x[i];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) red[threadIdx.x] += red[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0) red[0] = 1.0f / sqrtf(red[0] / (float)size + 1e-5f);
  __syncthreads();
  float sc = red[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) o[i] = x[i] * w[i] * sc;
  __syncthreads();
  int nfull = (size / gs) * gs;
  for (int i = threadIdx.x; i < nfull; i += blockDim.x) {
    int g = i / gs;
    const float *pg = o + g * gs;
    float maxv = 0.0f;
    for (int j = 0; j < gs; j++) { float v = fabsf(pg[j]); if (v > maxv) maxv = v; }
    float gsc = maxv / 127.0f;
    if (i % gs == 0) s[g] = gsc;
    q[i] = (int8_t)roundf(o[i] / gsc);
  }
}

__global__ void axpy_rmsnorm_quant_kernel(float *o, float *x, const float *y, const float *w,
                                          int size, int8_t *q, float *s, int gs) {
  extern __shared__ float red[];
  for (int i = threadIdx.x; i < size; i += blockDim.x) x[i] += y[i];
  __syncthreads();
  float sum = 0.0f;
  for (int i = threadIdx.x; i < size; i += blockDim.x) sum += x[i] * x[i];
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) red[threadIdx.x] += red[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0) red[0] = 1.0f / sqrtf(red[0] / (float)size + 1e-5f);
  __syncthreads();
  float sc = red[0];
  for (int i = threadIdx.x; i < size; i += blockDim.x) o[i] = x[i] * w[i] * sc;
  __syncthreads();
  int nfull = (size / gs) * gs;
  for (int i = threadIdx.x; i < nfull; i += blockDim.x) {
    int g = i / gs;
    const float *pg = o + g * gs;
    float maxv = 0.0f;
    for (int j = 0; j < gs; j++) { float v = fabsf(pg[j]); if (v > maxv) maxv = v; }
    float gsc = maxv / 127.0f;
    if (i % gs == 0) s[g] = gsc;
    q[i] = (int8_t)roundf(o[i] / gsc);
  }
}

__global__ void silu_mul_quant_kernel(float *h, const float *h1, int n,
                                      int8_t *q, float *s, int gs) {
  int nfull = (n / gs) * gs;
  for (int i = threadIdx.x; i < nfull; i += blockDim.x) {
    float val = h[i];
    h[i] = val * (1.0f / (1.0f + expf(-val))) * h1[i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nfull; i += blockDim.x) {
    int g = i / gs;
    const float *pg = h + g * gs;
    float maxv = 0.0f;
    for (int j = 0; j < gs; j++) { float v = fabsf(pg[j]); if (v > maxv) maxv = v; }
    float gsc = maxv / 127.0f;
    if (i % gs == 0) s[g] = gsc;
    q[i] = (int8_t)roundf(h[i] / gsc);
  }
}

__global__ void qmatmul_qkv_kernel(float *q, float *k, float *v,
                                   const int8_t *wq, const int8_t *wk, const int8_t *wv,
                                   const float *wsq, const float *wks, const float *wsv,
                                   const int8_t *xq, const float *xs,
                                   int dim, int kvdim, const int *gp, int gs) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const int8_t *wr; const float *wsc; float *o; int idx;
  if (i < dim) { wr = wq; wsc = wsq; o = q; idx = i; }
  else if (i < 2 * dim) { int j = i - dim; wr = wk; wsc = wks; o = k + (size_t)gp[1] * kvdim; idx = j; }
  else { int j = i - 2 * dim; wr = wv; wsc = wsv; o = v + (size_t)gp[1] * kvdim; idx = j; }
  int in = idx * dim;
  int nfull = (dim / gs) * gs;
  float vv = 0.0f;
  for (int j = threadIdx.x * gs; j < nfull; j += blockDim.x * gs) {
    int32_t iv = 0;
    for (int kk = 0; kk < gs; kk++) {
      iv += (int32_t)wr[in + j + kk] * (int32_t)xq[j + kk];
    }
    vv += (float)iv * wsc[(in + j) / gs] * xs[j / gs];
  }
  red[threadIdx.x] = vv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) o[idx] = red[0];
}

__global__ void attn_rope_score_kernel(float *attn, const float *q, const float *kcache,
                                       const int *gp, int nheads, int ncontext,
                                       int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[]; // [0,hsize) q slice, [hsize,2hsize) k slice, then reduction
  int h = blockIdx.x;
  int t = blockIdx.y;
  int pos = gp[1];
  if (t > pos) return;
  const float *qsrc = q + h * hsize;
  const float *ksrc = kcache + (size_t)t * kvdim + (h / kvmul) * hsize;
  for (int i = threadIdx.x; i < hsize; i += blockDim.x) {
    red[i] = qsrc[i];
    red[hsize + i] = ksrc[i];
  }
  __syncthreads();
  for (int p = threadIdx.x; p < hsize / 2; p += blockDim.x) {
    int i = 2 * p;
    float freq = 1.0f / powf(10000.0f, (float)i / (float)hsize);
    float crq = cosf((float)pos * freq), ciq = sinf((float)pos * freq);
    float crk = cosf((float)t * freq), cik = sinf((float)t * freq);
    float v0 = red[i], v1 = red[i + 1];
    red[i] = v0 * crq - v1 * ciq;
    red[i + 1] = v0 * ciq + v1 * crq;
    v0 = red[hsize + i]; v1 = red[hsize + i + 1];
    red[hsize + i] = v0 * crk - v1 * cik;
    red[hsize + i + 1] = v0 * cik + v1 * crk;
  }
  __syncthreads();
  float v = 0.0f;
  for (int i = threadIdx.x; i < hsize; i += blockDim.x) {
    v += red[i] * red[hsize + i];
  }
  __syncthreads(); // q/k slices no longer needed before reduction overwrites red[]
  red[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    attn[(size_t)h * ncontext + t] = red[0] / sqrtf((float)hsize);
  }
}

__global__ void qmatmul_w1w3_kernel(float *h, float *h1, const int8_t *w1, const int8_t *w3,
                                    const float *ws1, const float *ws3,
                                    const int8_t *xq, const float *xs, int dim, int ffndim, int gs) {
  extern __shared__ float red[];
  int i = blockIdx.x;
  const int8_t *wr; const float *wsc; float *o; int idx;
  if (i < ffndim) { wr = w1; wsc = ws1; o = h; idx = i; }
  else { idx = i - ffndim; wr = w3; wsc = ws3; o = h1; }
  int in = idx * dim;
  int nfull = (dim / gs) * gs;
  float vv = 0.0f;
  for (int j = threadIdx.x * gs; j < nfull; j += blockDim.x * gs) {
    int32_t iv = 0;
    for (int kk = 0; kk < gs; kk++) {
      iv += (int32_t)wr[in + j + kk] * (int32_t)xq[j + kk];
    }
    vv += (float)iv * wsc[(in + j) / gs] * xs[j / gs];
  }
  red[threadIdx.x] = vv;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) o[idx] = red[0];
}

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

__global__ void attn_value_kernel(float *x1, const float *attn, const float *vcache,
                                  const int *gp, int nheads, int ncontext,
                                  int hsize, int kvdim, int kvmul) {
  extern __shared__ float red[];
  int h = blockIdx.x;
  int i = blockIdx.y;
  int pos = gp[1];
  const float *ah = attn + (size_t)h * ncontext;
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

__global__ void sample_prep_kernel(float *e, int *idx, float *out_S, const float *logits, int n, float inv_temp) {
  // inv_temp = 1/temperature passed in: FP32 division per element (~200 cycles) is
  // the kernel's bottleneck; the multiply is ~5x cheaper and bit-exact enough
  // (verified: identical sampled tokens across seeds).
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
  extern __shared__ float red[]; // [threads] chunk sums
  const int threads = blockDim.x;
  float *cs = red;
  float S = *out_S;
  float cutoff_e = (1.0f - topp) / (n - 1) * S;
  int L = (n + threads - 1) / threads;
  int b = threadIdx.x;
  int lo = b * L, hi = min(n, lo + L);
  float v = 0.0f;
  for (int j = lo; j < hi; j++) {
    if (e[j] >= cutoff_e) v += e[j];
  }
  cs[b] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    // truncation: walk from top, slast = cum at first element where cum > topp*S (or full F sum)
    float suf = 0.0f;
    int tstar = -1;
    for (int t = threads - 1; t >= 0; t--) {
      if (suf <= topp * S && suf + cs[t] > topp * S) { tstar = t; break; }
      suf += cs[t];
    }
    float slast;
    if (tstar < 0) {
      slast = suf;
    } else {
      float cum = 0.0f;
      int hi2 = min(n, (tstar + 1) * L);
      for (int j = hi2 - 1; j >= tstar * L; j--) {
        if (e[j] < cutoff_e) continue;
        cum += e[j];
        if (suf + cum > topp * S) break;
      }
      slast = suf + cum;
    }
    float r = coin * slast;
    // crossing: first element (from top) where cum > r
    float suf2 = 0.0f;
    int tsel = -1;
    for (int t = threads - 1; t >= 0; t--) {
      if (r < suf2 + cs[t]) { tsel = t; break; }
      suf2 += cs[t];
    }
    int token = 0;
    if (tsel >= 0) {
      float cum2 = 0.0f;
      int lo2 = tsel * L, hi2 = min(n, lo2 + L);
      for (int j = hi2 - 1; j >= lo2; j--) {
        if (e[j] < cutoff_e) continue;
        cum2 += e[j];
        token = idx[j];
        if (r < suf2 + cum2) break;
      }
    } else {
      for (int j = n - 1; j >= 0; j--) if (e[j] >= cutoff_e) { token = idx[j]; break; }
    }
    *next = token;
  }
}

// ---- extern "C" launchers (one per kernel) ----
// Every launcher takes a trailing cudaStream_t: the zig host captures the
// full forward as one CUDA graph on its own stream (see llama2q_cuda.zig),
// and the sampling launchers pass 0 (the legacy default stream, which the
// driver serializes with the graph stream implicitly).

extern "C" void llama2q_cuda_quantize(const float *x, int n, int8_t *q, float *s, int gs, cudaStream_t stream) {
  quantize_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n, q, s, gs);
}

extern "C" void llama2q_cuda_load_emb(float *x, const float *embeddings, int dim, const int *gp, cudaStream_t stream) {
  load_emb_kernel<<<(dim + 255) / 256, 256, 0, stream>>>(x, embeddings, dim, gp);
}

extern "C" void llama2q_cuda_rms_scale(float *o, const float *x, const float *w, int size, float ss, cudaStream_t stream) {
  int nblk = (size + 255) / 256;
  rms_scale_kernel<<<nblk, 256, 0, stream>>>(o, x, w, size, ss);
}

extern "C" void llama2q_cuda_rmsnorm(float *o, const float *x, const float *w, int size, cudaStream_t stream) {
  rmsnorm_kernel<<<1, 256, 256 * sizeof(float), stream>>>(o, x, w, size);
}

extern "C" void llama2q_cuda_softmax_rows(float *attn, const int *gp, int ncontext, int nheads, cudaStream_t stream) {
  softmax_rows_kernel<<<nheads, 256, 256 * sizeof(float), stream>>>(attn, gp, ncontext);
}

extern "C" void llama2q_cuda_rmsnorm_quant(float *o, const float *x, const float *w, int size, int8_t *q, float *s, int gs, cudaStream_t stream) {
  rmsnorm_quant_kernel<<<1, 256, 256 * sizeof(float), stream>>>(o, x, w, size, q, s, gs);
}

extern "C" void llama2q_cuda_axpy_rmsnorm_quant(float *o, float *x, const float *y, const float *w, int size, int8_t *q, float *s, int gs, cudaStream_t stream) {
  axpy_rmsnorm_quant_kernel<<<1, 256, 256 * sizeof(float), stream>>>(o, x, y, w, size, q, s, gs);
}

extern "C" void llama2q_cuda_silu_mul_quant(float *h, const float *h1, int ffndim, int8_t *q, float *s, int gs, cudaStream_t stream) {
  silu_mul_quant_kernel<<<(ffndim + 255) / 256, 256, 0, stream>>>(h, h1, ffndim, q, s, gs);
}

extern "C" void llama2q_cuda_qmatmul_qkv(float *q, float *k, float *v, const int8_t *wq, const int8_t *wk, const int8_t *wv, const float *wsq, const float *wks, const float *wsv, const int8_t *xq, const float *xs, int dim, int kvdim, const int *gp, int gs, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  qmatmul_qkv_kernel<<<3 * dim, 256, shared, stream>>>(q, k, v, wq, wk, wv, wsq, wks, wsv, xq, xs, dim, kvdim, gp, gs);
}

extern "C" void llama2q_cuda_attn_rope_score(float *attn, const float *q, const float *kcache, const int *gp, int nheads, int ncontext, int hsize, int kvdim, int kvmul, cudaStream_t stream) {
  size_t shared = (256 + 2 * hsize) * sizeof(float);
  attn_rope_score_kernel<<<dim3(nheads, ncontext), 256, shared, stream>>>(attn, q, kcache, gp, nheads, ncontext, hsize, kvdim, kvmul);
}

extern "C" void llama2q_cuda_qmatmul_w1w3(float *h, float *h1, const int8_t *w1, const int8_t *w3, const float *ws1, const float *ws3, const int8_t *xq, const float *xs, int dim, int ffndim, int gs, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  qmatmul_w1w3_kernel<<<2 * ffndim, 256, shared, stream>>>(h, h1, w1, w3, ws1, ws3, xq, xs, dim, ffndim, gs);
}

extern "C" void llama2q_cuda_qmatmul(float *o, const int8_t *wq, const float *ws, const int8_t *xq, const float *xs, int n, int d, int gs, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  qmatmul_kernel<<<d, 256, shared, stream>>>(o, wq, ws, xq, xs, n, d, gs);
}

extern "C" void llama2q_cuda_axpy(float *o, const float *x, int dim, cudaStream_t stream) {
  axpy_kernel<<<(dim + 255) / 256, 256, 0, stream>>>(o, x, dim);
}

extern "C" void llama2q_cuda_silu_mul(float *h, const float *h1, int ffndim, cudaStream_t stream) {
  silu_mul_kernel<<<(ffndim + 255) / 256, 256, 0, stream>>>(h, h1, ffndim);
}

extern "C" void llama2q_cuda_rope(float *q, float *k, int dim, int kvdim, int hsize, int pos, cudaStream_t stream) {
  rope_kernel<<<dim, 256, 0, stream>>>(q, k, dim, kvdim, hsize, pos);
}

extern "C" void llama2q_cuda_attn_score(float *attn, const float *q, const float *kcache, int nheads, int pos, int hsize, int kvdim, int kvmul, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  attn_score_kernel<<<dim3(nheads, pos + 1), 256, shared, stream>>>(attn, q, kcache, nheads, pos, hsize, kvdim, kvmul);
}

extern "C" void llama2q_cuda_attn_value(float *x1, const float *attn, const float *vcache, const int *gp, int nheads, int ncontext, int hsize, int kvdim, int kvmul, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  attn_value_kernel<<<dim3(nheads, hsize), 256, shared, stream>>>(x1, attn, vcache, gp, nheads, ncontext, hsize, kvdim, kvmul);
}

extern "C" void llama2q_cuda_sample_prep(float *e, int *idx, float *out_S, const float *logits, int n, float inv_temp, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  sample_prep_kernel<<<1, 256, shared, stream>>>(e, idx, out_S, logits, n, inv_temp);
}

extern "C" void llama2q_cuda_sample_pick(int *next, const float *e, const int *idx, int n, float topp, float coin, const float *out_S, cudaStream_t stream) {
  size_t shared = 256 * sizeof(float);
  sample_pick_kernel<<<1, 256, shared, stream>>>(next, e, idx, n, topp, coin, out_S);
}

// device-side sort of e (keys) by idx (values); mirrors C sample_device.
// thrust uses its own (default) stream; the parameter is kept for signature
// uniformity with the other launchers.
extern "C" void llama2q_cuda_sample_sort(float *e, int *idx, int n, cudaStream_t stream) {
  (void)stream;
  thrust::sort_by_key(thrust::device_pointer_cast(e),
                      thrust::device_pointer_cast(e + n),
                      thrust::device_pointer_cast(idx));
}

// ---- CUDA graph: capture the whole forward as one graph (C build_graph) ----
// All kernel launches below read {token,pos} from the device gparams buffer,
// so every launch's arguments are fixed for the lifetime of the graph; the
// per-token values travel in through the pinned host gparams_h copied by the
// H2D memcpy node recorded first in the capture.

struct llama2q_graph_params {
  // output (filled in by build)
  cudaGraphExec_t *out_exec;
  cudaGraph_t *out_graph;
  // host pinned {token,pos}
  const int *gparams_h;
  // device state
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
  int8_t *xq;
  float *xq_s;
  int8_t *hq;
  float *hq_s;
  int *gparams;
  // weights
  const float *embeddings;
  const float *wrmsattn;
  const float *wrmsffn;
  const float *wrmsfinal;
  const int8_t *wq;
  const int8_t *wk;
  const int8_t *wv;
  const int8_t *wo;
  const int8_t *w1;
  const int8_t *w2;
  const int8_t *w3;
  const float *wq_s;
  const float *wk_s;
  const float *wv_s;
  const float *wo_s;
  const float *w1_s;
  const float *w2_s;
  const float *w3_s;
  const int8_t *qtok;
  const float *qtok_s;
  // config
  int dim, kvdim, ffndim, nheads, ncontext, hsize, kvmul, gs, nlayers, nvocab;
};

extern "C" void llama2q_cuda_build_graph(struct llama2q_graph_params *p) {
  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return;
  cudaMemcpyAsync(p->gparams, p->gparams_h, 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
  const int threads = 256;
  size_t shared = (size_t)threads * sizeof(float);
  size_t dim2 = (size_t)p->dim * p->dim;
  size_t dimkv = (size_t)p->dim * p->kvdim;
  size_t dimff = (size_t)p->dim * p->ffndim;
  size_t ffn_dim = (size_t)p->ffndim * p->dim;

  load_emb_kernel<<<(p->dim + threads - 1) / threads, threads, 0, stream>>>(p->x, p->embeddings, p->dim, p->gparams);
  for (int l = 0; l < p->nlayers; l++) {
    size_t li = (size_t)l;
    rmsnorm_quant_kernel<<<1, threads, shared, stream>>>(p->x1, p->x, p->wrmsattn + li * p->dim, p->dim, p->xq, p->xq_s, p->gs);
    qmatmul_qkv_kernel<<<3 * p->dim, threads, shared, stream>>>(p->q, p->kcache + li * p->ncontext * p->kvdim, p->vcache + li * p->ncontext * p->kvdim,
        p->wq + li * dim2, p->wk + li * dimkv, p->wv + li * dimkv,
        p->wq_s + li * dim2 / p->gs, p->wk_s + li * dimkv / p->gs, p->wv_s + li * dimkv / p->gs,
        p->xq, p->xq_s, p->dim, p->kvdim, p->gparams, p->gs);
    attn_rope_score_kernel<<<dim3(p->nheads, p->ncontext), threads, shared + 2 * p->hsize * sizeof(float), stream>>>(
        p->attn, p->q, p->kcache + li * p->ncontext * p->kvdim, p->gparams, p->nheads, p->ncontext, p->hsize, p->kvdim, p->kvmul);
    softmax_rows_kernel<<<p->nheads, threads, shared, stream>>>(p->attn, p->gparams, p->ncontext);
    attn_value_kernel<<<dim3(p->nheads, p->hsize), threads, shared, stream>>>(p->x1, p->attn, p->vcache + li * p->ncontext * p->kvdim,
        p->gparams, p->nheads, p->ncontext, p->hsize, p->kvdim, p->kvmul);
    quantize_kernel<<<(p->dim + threads - 1) / threads, threads, 0, stream>>>(p->x1, p->dim, p->xq, p->xq_s, p->gs);
    qmatmul_kernel<<<p->dim, threads, shared, stream>>>(p->x2, p->wo + li * dim2, p->wo_s + li * dim2 / p->gs, p->xq, p->xq_s, p->dim, p->dim, p->gs);

    axpy_rmsnorm_quant_kernel<<<1, threads, shared, stream>>>(p->x1, p->x, p->x2, p->wrmsffn + li * p->dim, p->dim, p->xq, p->xq_s, p->gs);
    qmatmul_w1w3_kernel<<<2 * p->ffndim, threads, shared, stream>>>(p->h, p->h1, p->w1 + li * dimff, p->w3 + li * dimff,
        p->w1_s + li * dimff / p->gs, p->w3_s + li * dimff / p->gs, p->xq, p->xq_s, p->dim, p->ffndim, p->gs);
    silu_mul_quant_kernel<<<(p->ffndim + threads - 1) / threads, threads, 0, stream>>>(p->h, p->h1, p->ffndim, p->hq, p->hq_s, p->gs);
    qmatmul_kernel<<<p->dim, threads, shared, stream>>>(p->x1, p->w2 + li * ffn_dim, p->w2_s + li * ffn_dim / p->gs, p->hq, p->hq_s, p->ffndim, p->dim, p->gs);
    axpy_kernel<<<(p->dim + threads - 1) / threads, threads, 0, stream>>>(p->x, p->x1, p->dim);
  }
  rmsnorm_quant_kernel<<<1, threads, shared, stream>>>(p->x, p->x, p->wrmsfinal, p->dim, p->xq, p->xq_s, p->gs);
  qmatmul_kernel<<<p->nvocab, threads, shared, stream>>>(p->logits, p->qtok, p->qtok_s, p->xq, p->xq_s, p->dim, p->nvocab, p->gs);
  cudaGraph_t graph;
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
    cudaStreamDestroy(stream);
    return;
  }
  cudaGraphExec_t exec;
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return;
  }
  // the capture stream is no longer needed once the graph exists; the graph
  // itself is launched on the default stream (see graph_launch).
  cudaStreamDestroy(stream);
  *p->out_graph = graph;
  *p->out_exec = exec;
}

// launch the captured forward on the default stream (0): the sampling
// launchers also use stream 0, so everything serializes in submission order.
extern "C" void llama2q_cuda_graph_launch(cudaGraphExec_t exec) {
  cudaGraphLaunch(exec, 0);
}

extern "C" void llama2q_cuda_graph_destroy(cudaGraphExec_t exec, cudaGraph_t graph) {
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
}

// free pinned host memory (the zig host calls this instead of cudaFreeHost
// directly, so the symbol is resolved by nvcc against libcudart here).
extern "C" void llama2q_cuda_free_host(void *p) {
  cudaFreeHost(p);
}

// pinned host allocation; returns the device-pointer value (0 on failure).
extern "C" void *llama2q_cuda_alloc_host(size_t bytes) {
  void *p = 0;
  if (cudaHostAlloc(&p, bytes, 0) != cudaSuccess) return 0;
  return p;
}
