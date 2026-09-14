#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
// Instruction-sized primitives only. All loops, addressing and algorithms are TileLang.
__device__ __forceinline__ unsigned nr_ld32(const unsigned char* p) { return *(const unsigned*)p; }
__device__ __forceinline__ void nr_st32(unsigned char* p, unsigned v) { *(unsigned*)p=v; }
__device__ __forceinline__ unsigned nr_vload32(const unsigned* p) {return *(const volatile unsigned*)p;}
__device__ __forceinline__ void nr_vstore32(unsigned* p,unsigned v) {*(volatile unsigned*)p=v;}
__device__ __forceinline__ unsigned nr_hbits(float h) { return __half_as_ushort(__float2half_rn(h)); }
__device__ __forceinline__ float nr_half(unsigned h) { return __half2float(__ushort_as_half((unsigned short)h)); }
__device__ __forceinline__ float nr_dec(unsigned char v) { return __half2float(half(__nv_cvt_fp8_to_halfraw(v,__NV_E4M3))); }
__device__ __forceinline__ unsigned char nr_enc(float v) { return __nv_cvt_halfraw_to_fp8((__half_raw)__float2half_rn(v),__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ unsigned nr_qpair(unsigned v) { union {unsigned u;half2 h;} x; x.u=v;return __nv_cvt_halfraw2_to_fp8x2((__half2_raw)x.h,__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ unsigned nr_decode2(unsigned v) { union {unsigned u;half2 h;} x; x.h=half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)v,__NV_E4M3));return x.u; }
__device__ __forceinline__ float nr_add(float a,float b) {return __half2float(__hadd(__float2half_rn(a),__float2half_rn(b)));}
__device__ __forceinline__ float nr_mul(float a,float b) {return __half2float(__hmul(__float2half_rn(a),__float2half_rn(b)));}
__device__ __forceinline__ float nr_fma(float a,float b,float c) {return __half2float(__hfma(__float2half_rn(a),__float2half_rn(b),__float2half_rn(c)));}
__device__ __forceinline__ unsigned nr_add2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hadd2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_mul2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmul2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_fma2(unsigned a,unsigned b,unsigned c) { union {unsigned u;half2 h;} x,y,z;x.u=a;y.u=b;z.u=c;x.h=__hfma2(x.h,y.h,z.h);return x.u; }
__device__ __forceinline__ unsigned nr_min2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmin2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_max2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmax2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_dup(float v) { union {unsigned u;half2 h;} x;x.h=__float2half2_rn(v);return x.u; }
__device__ __forceinline__ unsigned nr_shuffle(unsigned v,int lane) {return __shfl_sync(0xffffffff,v,lane);}
__device__ __forceinline__ unsigned nr_xor(unsigned v,int mask) {return __shfl_xor_sync(0xffffffff,v,mask);}
__device__ __forceinline__ unsigned nr_perm(unsigned a,unsigned b,unsigned s) {return __byte_perm(a,b,s);}
__device__ __forceinline__ float nr_rcp(float x) {float y;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float nr_rsqrt(float x) {float y;asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ void nr_mma(unsigned* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(d[0]),"+r"(d[1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ AH, const half_t* __restrict__ BH, int* __restrict__ C0, int* __restrict__ C1, const int* __restrict__ M, uchar* __restrict__ O, const int* __restrict__ P, int count, int n0, int n1, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int v0, int v1);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ AH, const half_t* __restrict__ BH, int* __restrict__ C0, int* __restrict__ C1, const int* __restrict__ M, uchar* __restrict__ O, const int* __restrict__ P, int count, int n0, int n1, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int v0, int v1) {
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < count) {
    int condval_1;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_1 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_1 = 0;
    }
    int condval_2;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_2 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_2 = 0;
    }
    half_t condval;
    if (((0 <= condval_1) && (condval_2 < (sizes_1 >> 1)))) {
      int64_t condval_3;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_3 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_3 = (int64_t)0;
      }
      int64_t condval_4;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_4 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_4 = (int64_t)0;
      }
      condval = BH[condval_4];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_5;
    if (((((((int)blockIdx.x) * 256) + (count * 3)) + ((int)threadIdx.x)) < (sizes_0 >> 1))) {
      condval_5 = AH[(((((int64_t)((int)blockIdx.x)) * (int64_t)256) + (((int64_t)count) * (int64_t)3)) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_5 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t a = ((half_t)nr_add(((float)condval), ((float)condval_5)));
    half_t a_1 = cutlass::half_t(__hmax((cutlass::half_t(__hmin((a).to_half(), (half_t(0x1p+2f/*4.000000e+00*/)).to_half()))).to_half(), (half_t(-0x1p+2f/*-4.000000e+00*/)).to_half()));
    half_t a_2 = __habs(cutlass::half_t(__hmax((cutlass::half_t(__hmin((a).to_half(), (half_t(0x1p+2f/*4.000000e+00*/)).to_half()))).to_half(), (half_t(-0x1p+2f/*-4.000000e+00*/)).to_half())));
    half_t b = ((half_t)nr_fma(((float)__habs(cutlass::half_t(__hmax((cutlass::half_t(__hmin((a).to_half(), (half_t(0x1p+2f/*4.000000e+00*/)).to_half()))).to_half(), (half_t(-0x1p+2f/*-4.000000e+00*/)).to_half())))), -0x1.cap-5f/*-5.590820e-02*/, 0x1.cap-2f/*4.472656e-01*/));
    half_t b_1 = ((half_t)nr_fma(((float)cutlass::half_t(__hmax((cutlass::half_t(__hmin((a).to_half(), (half_t(0x1p+2f/*4.000000e+00*/)).to_half()))).to_half(), (half_t(-0x1p+2f/*-4.000000e+00*/)).to_half()))), ((float)b), 0x1.cap-1f/*8.945312e-01*/));
    int condval_6;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_4)) {
      condval_6 = P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_6 = 0;
    }
    if (0 <= condval_6) {
      int condval_7;
      if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_4)) {
        condval_7 = P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_7 = 0;
      }
      if (condval_7 < sizes_3) {
        int64_t condval_8;
        if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_4))) {
          condval_8 = ((int64_t)P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
        } else {
          condval_8 = (int64_t)0;
        }
        int64_t condval_9;
        if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_4))) {
          condval_9 = ((int64_t)P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
        } else {
          condval_9 = (int64_t)0;
        }
        O[condval_9] = nr_enc(((float)a));
      }
    }
  }
  if (((int)blockIdx.x) == 0) {
    for (int j = 0; j < ((n0 + 255) >> 8); ++j) {
      if (((j * 256) + ((int)threadIdx.x)) < n0) {
        C0[((((int64_t)j) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))] = v0;
      }
    }
    for (int j_1 = 0; j_1 < ((n1 + 255) >> 8); ++j_1) {
      if (((j_1 * 256) + ((int)threadIdx.x)) < n1) {
        C1[((((int64_t)j_1) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))] = v1;
      }
    }
  }
}

