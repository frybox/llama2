// Correctness test for c/runv.c fp32 GEMV kernels (scalar / avx2 / avx512).
// Kernels are copies of the runv.c kernels; FMA means avx2/avx512 are NOT
// bit-exact with scalar, so we check NEAR-exactness (max relative error small).
// A kernel is only RUN when the CPU reports the features it needs (the avx512
// kernel is skipped on a CPU without AVX512F), so this also works on non-AVX512
// machines.
//   gcc -O3 -o c/test_fp32 c/test_fp32.c -lm && ./c/test_fp32
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <cpuid.h>

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
  return (ebx >> 16) & 1; // AVX512F (the fp32 kernel needs only F)
}

static void m_scalar(float *o, float *w, float *x, int n, int d) {
  for (int i=0;i<d;i++){ float v=0; for(int j=0;j<n;j++) v+=w[i*n+j]*x[j]; o[i]=v; }
}
static void __attribute__((target("avx2,fma")))
m_avx2(float *o, float *w, float *x, int n, int d) {
  for (int i=0;i<d;i++){
    const float *row=w+(size_t)i*n;
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
    int j=0;
    for(;j+32<=n;j+=32){
      a0=_mm256_fmadd_ps(_mm256_loadu_ps(row+j),_mm256_loadu_ps(x+j),a0);
      a1=_mm256_fmadd_ps(_mm256_loadu_ps(row+j+8),_mm256_loadu_ps(x+j+8),a1);
      a2=_mm256_fmadd_ps(_mm256_loadu_ps(row+j+16),_mm256_loadu_ps(x+j+16),a2);
      a3=_mm256_fmadd_ps(_mm256_loadu_ps(row+j+24),_mm256_loadu_ps(x+j+24),a3);
    }
    a0=_mm256_add_ps(a0,a1); a2=_mm256_add_ps(a2,a3); a0=_mm256_add_ps(a0,a2);
    float buf[8]; _mm256_storeu_ps(buf,a0);
    float v=((buf[0]+buf[1])+(buf[2]+buf[3]))+((buf[4]+buf[5])+(buf[6]+buf[7]));
    for(;j<n;j++) v+=row[j]*x[j];
    o[i]=v;
  }
}
static void __attribute__((target("avx512f")))
m_avx512(float *o, float *w, float *x, int n, int d) {
  for (int i=0;i<d;i++){
    const float *row=w+(size_t)i*n;
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps();
    int j=0;
    for(;j+64<=n;j+=64){
      a0=_mm512_fmadd_ps(_mm512_loadu_ps(row+j),_mm512_loadu_ps(x+j),a0);
      a1=_mm512_fmadd_ps(_mm512_loadu_ps(row+j+16),_mm512_loadu_ps(x+j+16),a1);
      a2=_mm512_fmadd_ps(_mm512_loadu_ps(row+j+32),_mm512_loadu_ps(x+j+32),a2);
      a3=_mm512_fmadd_ps(_mm512_loadu_ps(row+j+48),_mm512_loadu_ps(x+j+48),a3);
    }
    __m512 a=(a0+a1)+(a2+a3);
    float buf[16]; _mm512_storeu_ps(buf,a);
    float t[8]; for(int k=0;k<8;k++) t[k]=buf[k]+buf[k+8];
    float u[4]; for(int k=0;k<4;k++) u[k]=t[k]+t[k+4];
    float v=(u[0]+u[1])+(u[2]+u[3]);
    for(;j<n;j++) v+=row[j]*x[j];
    o[i]=v;
  }
}

static double maxrel(float *a, float *b, int d){
  double m=0; for(int i=0;i<d;i++){ double d1=fabs(a[i]-b[i]); double den=fmax(1.0,fabs(a[i])); double r=d1/den; if(r>m)m=r; }
  return m;
}
int main(void){
  int ns[4]={288,1024,288,288}, ds[4]={288,288,1024,288};
  int has2 = cpu_has_avx2(), has5 = cpu_has_avx512();
  int all = 1;
  for(int ci=0;ci<4;ci++){
    int n=ns[ci], d=ds[ci];
    float *w=malloc(n*d*4), *x=malloc(n*4), *o1=malloc(d*4),*o2=malloc(d*4),*o3=malloc(d*4);
    srand(123+ci);
    for(int i=0;i<n*d;i++) w[i]=(float)((rand()%1000)/500.0-1.0)*0.3f;
    for(int i=0;i<n;i++)  x[i]=(float)((rand()%1000)/500.0-1.0)*0.3f;
    // kernels the CPU cannot execute are skipped (they would SIGILL)
    m_scalar(o1,w,x,n,d);
    double r2 = -1.0, r5 = -1.0, r52 = -1.0;
    if(has2) { m_avx2(o2,w,x,n,d); r2 = maxrel(o1,o2,d); }
    if(has5) { m_avx512(o3,w,x,n,d); r5 = maxrel(o1,o3,d); if(has2) r52 = maxrel(o2,o3,d); }
    if(has2 && r2 > 1e-5) all = 0;
    if(has5 && r5 > 1e-5) all = 0;
    printf("n=%d d=%d  ",n,d);
    if(has2) printf("avx2_vs_scalar rel=%.2e  ", r2); else printf("avx2=skipped(no avx2+fma)  ");
    if(has5) printf("avx512_vs_scalar rel=%.2e  ", r5); else printf("avx512=skipped(no avx512f)  ");
    if(has2 && has5) printf("avx512_vs_avx2 rel=%.2e", r52); else printf("avx512_vs_avx2=n/a");
    printf("\n");
    free(w);free(x);free(o1);free(o2);free(o3);
  }
  printf(all ? "ALL PASS (rel <= 1e-5)\n" : "FAIL\n");
  return all ? 0 : 1;
}
