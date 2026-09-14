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

extern "C" __global__ void main_kernel(const half_t* __restrict__ AH, const uchar* __restrict__ B, int* __restrict__ C0, int* __restrict__ C1, const half_t* __restrict__ G, const int* __restrict__ M, uchar* __restrict__ O, const int* __restrict__ P, int count, int n0, int n1, int poolwidth, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int v0, int v1, int width);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ AH, const uchar* __restrict__ B, int* __restrict__ C0, int* __restrict__ C1, const half_t* __restrict__ G, const int* __restrict__ M, uchar* __restrict__ O, const int* __restrict__ P, int count, int n0, int n1, int poolwidth, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int v0, int v1, int width) {
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
    uchar condval;
    if (((0 <= condval_1) && (condval_2 < sizes_1))) {
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
      condval = B[condval_4];
    } else {
      condval = (uchar)0;
    }
    half_t a = ((half_t)nr_dec(condval));
    int condval_6;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_6 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_6 = 0;
    }
    half_t condval_5;
    if (((condval_6 & 511) < sizes_5)) {
      int condval_7;
      if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
        condval_7 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_7 = 0;
      }
      int condval_8;
      if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
        condval_8 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_8 = 0;
      }
      condval_5 = G[(condval_8 & 511)];
    } else {
      condval_5 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t b = condval_5;
    int condval_10;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_10 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_10 = 0;
    }
    int rmod = ((condval_10 >> 9) % width);
    int condval_11;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_11 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_11 = 0;
    }
    int rmod_1 = ((condval_11 >> 9) % width);
    int condval_12;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_12 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_12 = 0;
    }
    int rdiv = ((condval_12 >> 9) / width);
    int condval_13;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_13 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_13 = 0;
    }
    int rmod_2 = ((condval_13 >> 9) % width);
    int condval_14;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_14 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_14 = 0;
    }
    int rmod_3 = ((condval_14 >> 9) % width);
    int condval_15;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_15 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_15 = 0;
    }
    int rdiv_1 = ((condval_15 >> 9) / width);
    int condval_16;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_2)) {
      condval_16 = M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_16 = 0;
    }
    half_t condval_9;
    if (((0 <= ((((((0 <= width) && (0 <= rmod)) || ((width < 0) && (rmod <= 0))) ? rmod : (rmod + width)) >> 1) + ((((((0 <= width) && (0 <= rmod_1)) || ((width < 0) && (rmod_1 <= 0))) ? rdiv : (rdiv - 1)) >> 1) * poolwidth))) && (((((((((0 <= width) && (0 <= rmod_2)) || ((width < 0) && (rmod_2 <= 0))) ? rmod_2 : (rmod_2 + width)) >> 1) * 512) + (((((((0 <= width) && (0 <= rmod_3)) || ((width < 0) && (rmod_3 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) >> 1) * poolwidth) * 512)) + (condval_16 & 511)) < (sizes_0 >> 1)))) {
      int64_t condval_17;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_17 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_17 = (int64_t)0;
      }
      int64_t rmod_4 = ((condval_17 >> (int64_t)9) % ((int64_t)width));
      int64_t condval_18;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_18 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_18 = (int64_t)0;
      }
      int64_t rmod_5 = ((condval_18 >> (int64_t)9) % ((int64_t)width));
      int64_t condval_19;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_19 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_19 = (int64_t)0;
      }
      int64_t rdiv_2 = ((condval_19 >> (int64_t)9) / ((int64_t)width));
      int64_t condval_20;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_20 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_20 = (int64_t)0;
      }
      int64_t condval_21;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_21 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_21 = (int64_t)0;
      }
      int64_t rmod_6 = ((condval_21 >> (int64_t)9) % ((int64_t)width));
      int64_t condval_22;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_22 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_22 = (int64_t)0;
      }
      int64_t rmod_7 = ((condval_22 >> (int64_t)9) % ((int64_t)width));
      int64_t condval_23;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_23 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_23 = (int64_t)0;
      }
      int64_t rdiv_3 = ((condval_23 >> (int64_t)9) / ((int64_t)width));
      int64_t condval_24;
      if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_2))) {
        condval_24 = ((int64_t)M[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
      } else {
        condval_24 = (int64_t)0;
      }
      condval_9 = AH[(((((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_6)) || ((((int64_t)width) < (int64_t)0) && (rmod_6 <= (int64_t)0))) ? rmod_6 : (rmod_6 + ((int64_t)width))) >> (int64_t)1) * (int64_t)512) + ((((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_7)) || ((((int64_t)width) < (int64_t)0) && (rmod_7 <= (int64_t)0))) ? rdiv_3 : (rdiv_3 - (int64_t)1)) >> (int64_t)1) * ((int64_t)poolwidth)) * (int64_t)512)) + (condval_24 & (int64_t)511))];
    } else {
      condval_9 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t c = condval_9;
    half_t a_1 = ((half_t)nr_fma(((float)a), ((float)b), ((float)c)));
    int condval_25;
    if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_4)) {
      condval_25 = P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_25 = 0;
    }
    if (0 <= condval_25) {
      int condval_26;
      if ((((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < sizes_4)) {
        condval_26 = P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];
      } else {
        condval_26 = 0;
      }
      if (condval_26 < sizes_3) {
        int64_t condval_27;
        if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_4))) {
          condval_27 = ((int64_t)P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
        } else {
          condval_27 = (int64_t)0;
        }
        int64_t condval_28;
        if ((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) < ((int64_t)sizes_4))) {
          condval_28 = ((int64_t)P[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))]);
        } else {
          condval_28 = (int64_t)0;
        }
        O[condval_28] = nr_enc(((float)a_1));
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

