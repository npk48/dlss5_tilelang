#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ unsigned long long wide_mma(unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned c0,unsigned c1) {
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};" : "+r"(c0),"+r"(c1) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 return ((unsigned long long)c1<<32)|c0;
}
__device__ __forceinline__ half2 wide_h2(unsigned x) { union {unsigned u; half2 h;} v;v.u=x;return v.h; }
__device__ __forceinline__ unsigned wide_bits(half2 x) { union {unsigned u; half2 h;} v;v.h=x;return v.u; }
__device__ __forceinline__ unsigned wide_add(unsigned a,unsigned b) {return wide_bits(__hadd2(wide_h2(a),wide_h2(b)));}
__device__ __forceinline__ unsigned wide_mul(unsigned a,unsigned b) {return wide_bits(__hmul2(wide_h2(a),wide_h2(b)));}
__device__ __forceinline__ unsigned wide_fma(unsigned a,unsigned b,unsigned c) {return wide_bits(__hfma2(wide_h2(a),wide_h2(b),wide_h2(c)));}
__device__ __forceinline__ unsigned wide_min(unsigned a,unsigned b) {return wide_bits(__hmin2(wide_h2(a),wide_h2(b)));}
__device__ __forceinline__ unsigned wide_max(unsigned a,unsigned b) {return wide_bits(__hmax2(wide_h2(a),wide_h2(b)));}
__device__ __forceinline__ unsigned wide_abs(unsigned a) {return wide_bits(__habs2(wide_h2(a)));}
__device__ __forceinline__ unsigned wide_splat(float a) {return wide_bits(__float2half2_rn(a));}
__device__ __forceinline__ unsigned wide_shfl(unsigned a,int lane) {return __shfl_sync(0xffffffff,a,lane);}
__device__ __forceinline__ unsigned wide_shfl_pair(unsigned a,unsigned b,int lane,bool select_b) {
 unsigned x,y;
 // One unconditional asm block: do not predicate either source-bank shuffle.
 asm volatile("{ .reg .b32 sx,sy;\n\t"
              "shfl.sync.idx.b32 sx, %2, %4, 31, 0xffffffff;\n\t"
              "shfl.sync.idx.b32 sy, %3, %4, 31, 0xffffffff;\n\t"
              "mov.b32 %0, sx; mov.b32 %1, sy; }"
              : "=r"(x),"=r"(y) : "r"(a),"r"(b),"r"(lane));
 return select_b ? y : x;
}
__device__ __forceinline__ unsigned wide_decode(unsigned a) {return wide_bits(__half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)a,__NV_E4M3)));}
__device__ __forceinline__ unsigned wide_encode(unsigned a,unsigned b) {
 return unsigned(__nv_cvt_halfraw2_to_fp8x2((__half2_raw)wide_h2(a),__NV_SATFINITE,__NV_E4M3)) | (unsigned(__nv_cvt_halfraw2_to_fp8x2((__half2_raw)wide_h2(b),__NV_SATFINITE,__NV_E4M3))<<16);
}
__device__ __forceinline__ unsigned wide_rsqrt(unsigned a) { float y;float x=__half2float(__ushort_as_half(a));asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return wide_splat(y); }
__device__ __forceinline__ unsigned wide_rcp(unsigned a) { float y;float x=__half2float(__ushort_as_half(a));asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return wide_splat(y); }

__device__ __forceinline__ unsigned wide_packet_prmt(unsigned a,unsigned b,unsigned s) { unsigned r; asm("prmt.b32 %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s)); return r; }
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

extern "C" __global__ void kernel_kernel(const int* __restrict__ IM, int* __restrict__ Proj, const int* __restrict__ R, const int* __restrict__ W, int imn, int inp, int pn, int rn, int rows, int wn);
extern "C" __global__ void __launch_bounds__(32, 1) kernel_kernel(const int* __restrict__ IM, int* __restrict__ Proj, const int* __restrict__ R, const int* __restrict__ W, int imn, int inp, int pn, int rn, int rows, int wn) {
  uint acc[2];
  uint a[4];
  for (int n = 0; n < 4; ++n) {
    acc[0] = (uint)0;
    acc[1] = (uint)0;
    for (int kp = 0; kp < 8; ++kp) {
      #pragma unroll
      for (int word = 0; word < 4; ++word) {
        a[word] = (uint)0;
        if ((((((int)blockIdx.x) * 16) + ((word & 1) * 8)) + (((int)threadIdx.x) >> 2)) < rows) {
          int condval;
          if ((((((((int)blockIdx.x) * 4096) + ((word & 1) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (((((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1))) < imn)) {
            condval = IM[((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)word) & (int64_t)1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)-15) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)2) << (int64_t)2)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)4) >> (int64_t)1)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)8) >> (int64_t)1)))];
          } else {
            condval = 0;
          }
          if (0 <= condval) {
            int condval_2;
            if ((((((((int)blockIdx.x) * 4096) + ((word & 1) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (((((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1))) < imn)) {
              condval_2 = IM[((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)word) & (int64_t)1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)-15) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)2) << (int64_t)2)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)4) >> (int64_t)1)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)8) >> (int64_t)1)))];
            } else {
              condval_2 = 0;
            }
            int condval_3;
            if ((((((((int)blockIdx.x) * 4096) + ((word & 1) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (((((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1))) < imn)) {
              condval_3 = IM[((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)word) & (int64_t)1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)-15) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)2) << (int64_t)2)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)4) >> (int64_t)1)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)8) >> (int64_t)1)))];
            } else {
              condval_3 = 0;
            }
            int condval_1;
            if (((0 <= (inp + condval_2)) && (((inp + condval_3) >> 2) < (rn >> 2)))) {
              int condval_4;
              if ((((((((int)blockIdx.x) * 4096) + ((word & 1) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (((((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1))) < imn)) {
                condval_4 = IM[((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)word) & (int64_t)1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)-15) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)2) << (int64_t)2)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)4) >> (int64_t)1)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)8) >> (int64_t)1)))];
              } else {
                condval_4 = 0;
              }
              int condval_5;
              if ((((((((int)blockIdx.x) * 4096) + ((word & 1) * 2048)) + ((((int)threadIdx.x) >> 2) * 256)) + (((((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((kp * 32) + ((word >> 1) * 16)) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1))) < imn)) {
                condval_5 = IM[((((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + ((((int64_t)word) & (int64_t)1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)256)) + (((((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)-15) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)2) << (int64_t)2)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)4) >> (int64_t)1)) | (((((((int64_t)kp) * (int64_t)32) + ((((int64_t)word) >> (int64_t)1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)) & (int64_t)8) >> (int64_t)1)))];
              } else {
                condval_5 = 0;
              }
              condval_1 = R[((inp + condval_5) >> 2)];
            } else {
              condval_1 = 0;
            }
            int _reinterpret_tmp = condval_1;
            a[word] = (*(uint *)(&(_reinterpret_tmp)));
          }
        }
      }
      int condval_6;
      if ((((((((kp * 1024) + (((int)blockIdx.y) * 256)) + ((n >> 1) * 128)) + (((int)threadIdx.x) * 4)) + ((n & 1) * 2)) + 24576) < wn)) {
        condval_6 = W[((((((kp * 1024) + (((int)blockIdx.y) * 256)) + ((n >> 1) * 128)) + (((int)threadIdx.x) * 4)) + ((n & 1) * 2)) + 24576)];
      } else {
        condval_6 = 0;
      }
      int condval_7;
      if ((((((((kp * 1024) + (((int)blockIdx.y) * 256)) + ((n >> 1) * 128)) + (((int)threadIdx.x) * 4)) + ((n & 1) * 2)) + 24577) < wn)) {
        condval_7 = W[((((((kp * 1024) + (((int)blockIdx.y) * 256)) + ((n >> 1) * 128)) + (((int)threadIdx.x) * 4)) + ((n & 1) * 2)) + 24577)];
      } else {
        condval_7 = 0;
      }
      uint64_t r = wide_mma(a[0], a[1], a[2], a[3], condval_6, condval_7, acc[0], acc[1]);
      acc[0] = ((uint)r);
      acc[1] = ((uint)(r >> (uint64_t)32));
    }
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
      if ((((((int)blockIdx.x) * 16) + (j * 8)) + (((int)threadIdx.x) >> 2)) < rows) {
        if (((((((((int)blockIdx.x) * 1024) + (j * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((int)blockIdx.y) * 16)) + (n * 4)) + (((int)threadIdx.x) & 3)) < pn) {
          uint v_ = acc[j];
          Proj[((((((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)j) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)64)) + (((int64_t)((int)blockIdx.y)) * (int64_t)16)) + (((int64_t)n) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) & (int64_t)3))] = (*(int *)(&(v_)));
        }
      }
    }
  }
}

