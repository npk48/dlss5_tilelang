#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
namespace nr_tl_shallow {
union H2 { unsigned u; half2 h; __device__ H2(unsigned v):u(v){} __device__ H2(half2 v):h(v){} };
// Float ABI accepts both CUDA half and TileLang's cutlass::half_t. Every
// conversion is explicitly rounded to Half before the intrinsic executes.
__device__ __forceinline__ unsigned pack(float a, float b) { return unsigned(__half_as_ushort(__float2half_rn(a))) | (unsigned(__half_as_ushort(__float2half_rn(b))) << 16); }
__device__ __forceinline__ float unpack(unsigned a, int i) { return __half2float(__ushort_as_half(a >> (16*i))); }
__device__ __forceinline__ unsigned add(unsigned a,unsigned b) { return H2(__hadd2(H2(a).h,H2(b).h)).u; }
__device__ __forceinline__ unsigned mul(unsigned a,unsigned b) { return H2(__hmul2(H2(a).h,H2(b).h)).u; }
__device__ __forceinline__ unsigned fma(unsigned a,unsigned b,unsigned c) { return H2(__hfma2(H2(a).h,H2(b).h,H2(c).h)).u; }
__device__ __forceinline__ unsigned min(unsigned a,unsigned b) { return H2(__hmin2(H2(a).h,H2(b).h)).u; }
__device__ __forceinline__ unsigned max(unsigned a,unsigned b) { return H2(__hmax2(H2(a).h,H2(b).h)).u; }
__device__ __forceinline__ unsigned abs(unsigned a) { return H2(__habs2(H2(a).h)).u; }
__device__ __forceinline__ unsigned e4pair(unsigned a) { return __nv_cvt_halfraw2_to_fp8x2(H2(a).h,__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ float une4(unsigned a) { return __half2float(half(__nv_cvt_fp8_to_halfraw((unsigned char)a,__NV_E4M3))); }
__device__ __forceinline__ unsigned e4(float a) { return __nv_cvt_halfraw_to_fp8(__float2half_rn(a),__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ unsigned shfl(unsigned a,int lane) { return __shfl_sync(0xffffffff,a,lane); }
__device__ __forceinline__ unsigned transpose(unsigned a) { unsigned d; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;":"=r"(d):"r"(a)); return d; }
__device__ __forceinline__ float hadd(float a,float b) { return __half2float(__hadd(__float2half_rn(a),__float2half_rn(b))); }
__device__ __forceinline__ float hmul(float a,float b) { return __half2float(__hmul(__float2half_rn(a),__float2half_rn(b))); }
__device__ __forceinline__ float hfma(float a,float b,float c) { return __half2float(__hfma(__float2half_rn(a),__float2half_rn(b),__float2half_rn(c))); }
__device__ __forceinline__ float rsqrt(float a) { float d; asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(d):"f"(a)); return __half2float(__float2half_rn(d)); }
__device__ __forceinline__ float rcp(float a) { float d; asm("rcp.approx.ftz.f32 %0,%1;":"=f"(d):"f"(a)); return __half2float(__float2half_rn(d)); }
__device__ __forceinline__ int mma8(unsigned *c,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};" : "+r"(c[0]),"+r"(c[1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); return 0;
}
__device__ __forceinline__ int mma16(unsigned *c,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};" : "+r"(c[0]),"+r"(c[1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); return 0;
}
__device__ __forceinline__ int status_or(int *p,int v) { atomicOr(p,v); return 0; }
__device__ __forceinline__ int sync() { __syncwarp(); return 0; }
}

#pragma once
// Only individual memory instructions. Ownership, guards, routing, arithmetic,
// MMA scheduling and all producer/consumer loops are in shallow_joint.py.
namespace nr_tl_shallow_joint {
__device__ __forceinline__ int ld128(unsigned *q, const void *p) {
 asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(q[0]),"=r"(q[1]),"=r"(q[2]),"=r"(q[3]) : "l"(p) : "memory"); return 0;
}
__device__ __forceinline__ int st128(void *p, unsigned a, unsigned b, unsigned c, unsigned d) {
 asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p),"r"(a),"r"(b),"r"(c),"r"(d) : "memory"); return 0;
}
__device__ __forceinline__ int st32(void *p, unsigned a) {
 asm volatile("st.global.u32 [%0], %1;" :: "l"(p),"r"(a) : "memory"); return 0;
}
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

extern "C" __global__ void main_kernel(const half_t* __restrict__ G0, const half_t* __restrict__ G1, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const half_t* __restrict__ G0, const half_t* __restrict__ G1, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X) {
  uint raw[32];
  uint pool_a[4];
  uint aa[16];
  uint ff[32];
  uint mp[4];
  uint canonical[1];
  uint ha_[16];
  uint hidden[16];
  uint b[4];
  uint b_1[4];
  uint inv[8];
  uint qa[16];
  uint kb[16];
  uint vb[16];
  uint z[32];
  uint b_2[4];
  uint logits[32];
  uint seed[4];
  uint pa[16];
  uint attended[16];
  uint out[16];
  uint b_3[4];
  #pragma unroll
  for (int j = 0; j < 32; ++j) {
    raw[j] = (uint)0;
  }
  #pragma unroll
  for (int j_1 = 0; j_1 < 4; ++j_1) {
    pool_a[j_1] = (uint)0;
  }
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      mp[j_2] = (uint)0;
    }
    int condval_1;
    if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
      int condval_2;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_2 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_2 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_3;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_3 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_3 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_4;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_4 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_4 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_5;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_5 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_5 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_6;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_6 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_6 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_7;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_7 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_7 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_8;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_8 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_8 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_9;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_9 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_9 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_10;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_10 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_10 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_11;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_11 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_11 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_12;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_12 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_12 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_13;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_13 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_13 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_14;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_14 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_14 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_15;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_15 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_15 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_16;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_16 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_16 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_17;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_17 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_17 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_18;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_18 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_18 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_19;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_19 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_19 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int condval_20;
      if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        condval_20 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        condval_20 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      condval_1 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_2 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_3 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_4 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_5 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_6 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_7 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_8 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_9 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_10 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_11 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_12 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_13 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_14 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_15 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_16 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_17 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_18 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_19 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_20 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
    } else {
      condval_1 = -1;
    }
    int condval;
    if ((0 <= condval_1)) {
      int condval_21;
      if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
        int condval_22;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_22 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_22 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_23;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_23 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_23 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_24;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_24 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_24 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_25;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_25 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_25 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_26;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_26 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_26 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_27;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_27 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_27 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_28;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_28 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_28 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_29;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_29 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_29 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_30;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_30 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_30 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_31;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_31 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_31 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_32;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_32 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_32 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_33;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_33 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_33 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_34;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_34 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_34 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_35;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_35 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_35 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_36;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_36 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_36 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_37;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_37 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_37 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_38;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_38 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_38 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_39;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_39 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_39 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_40;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_40 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_40 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        condval_21 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_22 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_23 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_24 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_25 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_26 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_27 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_28 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_29 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_30 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_31 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_32 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_33 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_34 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_35 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_36 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_37 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_38 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_39 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_40 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
      } else {
        condval_21 = -1;
      }
      condval = (condval_21 + 10485760);
    } else {
      condval = -1;
    }
    if (0 <= condval) {
      int condval_42;
      if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
        int condval_43;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_43 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_43 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_44;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_44 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_44 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_45;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_45 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_45 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_46;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_46 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_46 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_47;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_47 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_47 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_48;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_48 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_48 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_49;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_49 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_49 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_50;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_50 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_50 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_51;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_51 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_51 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_52;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_52 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_52 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_53;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_53 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_53 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_54;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_54 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_54 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_55;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_55 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_55 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_56;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_56 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_56 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_57;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_57 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_57 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_58;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_58 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_58 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_59;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_59 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_59 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_60;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_60 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_60 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        int condval_61;
        if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
          condval_61 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
        } else {
          condval_61 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
        }
        condval_42 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_43 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_44 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_45 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_46 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_47 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_48 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_49 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_50 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_51 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_52 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_53 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_54 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_55 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_56 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_57 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_58 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_59 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_60 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_61 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
      } else {
        condval_42 = -1;
      }
      int condval_41;
      if ((0 <= condval_42)) {
        int condval_62;
        if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
          int condval_63;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_63 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_63 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_64;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_64 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_64 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_65;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_65 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_65 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_66;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_66 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_66 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_67;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_67 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_67 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_68;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_68 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_68 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_69;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_69 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_69 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_70;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_70 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_70 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_71;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_71 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_71 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_72;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_72 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_72 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_73;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_73 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_73 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_74;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_74 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_74 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_75;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_75 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_75 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_76;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_76 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_76 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_77;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_77 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_77 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_78;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_78 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_78 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_79;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_79 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_79 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_80;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_80 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_80 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_81;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_81 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_81 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          condval_62 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_63 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_64 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_65 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_66 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_67 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_68 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_69 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_70 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_71 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_72 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_73 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_74 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_75 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_76 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_77 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_78 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_79 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_80 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_81 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_62 = -1;
        }
        condval_41 = (condval_62 + 10485760);
      } else {
        condval_41 = -1;
      }
      if (condval_41 < 60358048) {
        int condval_83;
        if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
          int condval_84;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_84 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_84 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_85;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_85 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_85 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_86;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_86 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_86 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_87;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_87 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_87 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_88;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_88 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_88 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_89;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_89 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_89 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_90;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_90 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_90 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_91;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_91 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_91 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_92;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_92 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_92 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_93;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_93 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_93 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_94;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_94 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_94 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_95;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_95 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_95 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_96;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_96 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_96 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_97;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_97 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_97 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_98;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_98 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_98 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_99;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_99 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_99 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_100;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_100 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_100 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_101;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_101 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_101 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_102;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_102 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_102 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          condval_83 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_84 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_85 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_86 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_87 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_88 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_89 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_90 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_91 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_92 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_93 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_94 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_95 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_96 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_97 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_98 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_99 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_100 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_101 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_102 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_83 = -1;
        }
        int condval_82;
        if ((0 <= condval_83)) {
          int condval_103;
          if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
            int condval_104;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_104 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_104 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_105;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_105 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_105 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_106;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_106 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_106 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_107;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_107 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_107 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_108;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_108 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_108 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_109;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_109 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_109 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_110;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_110 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_110 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_111;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_111 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_111 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_112;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_112 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_112 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_113;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_113 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_113 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_114;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_114 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_114 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_115;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_115 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_115 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_116;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_116 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_116 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_117;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_117 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_117 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_118;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_118 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_118 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_119;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_119 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_119 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_120;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_120 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_120 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_121;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_121 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_121 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_122;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_122 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_122 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            condval_103 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_104 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_105 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_106 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_107 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_108 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_109 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_110 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_111 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_112 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_113 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_114 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_115 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_116 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_117 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_118 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_119 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_120 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_121 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_122 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
          } else {
            condval_103 = -1;
          }
          condval_82 = (condval_103 + 10485760);
        } else {
          condval_82 = -1;
        }
        int condval_124;
        if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
          int condval_125;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_125 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_125 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_126;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_126 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_126 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_127;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_127 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_127 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_128;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_128 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_128 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_129;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_129 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_129 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_130;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_130 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_130 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_131;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_131 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_131 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_132;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_132 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_132 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_133;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_133 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_133 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_134;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_134 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_134 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_135;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_135 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_135 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_136;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_136 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_136 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_137;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_137 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_137 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_138;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_138 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_138 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_139;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_139 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_139 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_140;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_140 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_140 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_141;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_141 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_141 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_142;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_142 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_142 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_143;
          if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
            condval_143 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
          } else {
            condval_143 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
          }
          condval_124 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_125 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_126 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_127 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_128 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_129 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_130 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_131 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_132 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_133 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_134 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_135 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_136 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_137 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_138 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_139 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_140 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_141 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_142 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_143 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_124 = -1;
        }
        int condval_123;
        if ((0 <= condval_124)) {
          int condval_144;
          if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
            int condval_145;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_145 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_145 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_146;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_146 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_146 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_147;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_147 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_147 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_148;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_148 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_148 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_149;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_149 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_149 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_150;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_150 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_150 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_151;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_151 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_151 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_152;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_152 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_152 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_153;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_153 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_153 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_154;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_154 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_154 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_155;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_155 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_155 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_156;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_156 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_156 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_157;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_157 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_157 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_158;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_158 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_158 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_159;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_159 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_159 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_160;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_160 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_160 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_161;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_161 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_161 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_162;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_162 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_162 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_163;
            if ((0 < ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
              condval_163 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
            } else {
              condval_163 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
            }
            condval_144 = ((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_145 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_146 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_147 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_148 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_149 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_150 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_151 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_152 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_153 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_154 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_155 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_156 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_157 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_158 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_159 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_160 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_161 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_162 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_163 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
          } else {
            condval_144 = -1;
          }
          condval_123 = (condval_144 + 10485760);
        } else {
          condval_123 = -1;
        }
        nr_tl_shallow_joint::ld128((&(mp[0])), (&(X[condval_123])));
      }
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint s0 = nr_tl_shallow::shfl(mp[0], (((i * 16) + ((((int)threadIdx.x) >> 2) * 2)) + p));
        uint s1 = nr_tl_shallow::shfl(mp[1], (((i * 16) + ((((int)threadIdx.x) >> 2) * 2)) + p));
        uint s2 = nr_tl_shallow::shfl(mp[2], (((i * 16) + ((((int)threadIdx.x) >> 2) * 2)) + p));
        uint s3 = nr_tl_shallow::shfl(mp[3], (((i * 16) + ((((int)threadIdx.x) >> 2) * 2)) + p));
        uint condval_164;
        if (((((int)threadIdx.x) % 4) == 0)) {
          condval_164 = s0;
        } else {
          uint condval_165;
          if (((((int)threadIdx.x) & 3) == 1)) {
            condval_165 = s1;
          } else {
            uint condval_166;
            if (((((int)threadIdx.x) & 3) == 2)) {
              condval_166 = s2;
            } else {
              condval_166 = s3;
            }
            condval_165 = condval_166;
          }
          condval_164 = condval_165;
        }
        uint word = condval_164;
        uint condval_167;
        if (((((int)threadIdx.x) % 4) == 0)) {
          condval_167 = s0;
        } else {
          uint condval_168;
          if (((((int)threadIdx.x) & 3) == 1)) {
            condval_168 = s1;
          } else {
            uint condval_169;
            if (((((int)threadIdx.x) & 3) == 2)) {
              condval_169 = s2;
            } else {
              condval_169 = s3;
            }
            condval_168 = condval_169;
          }
          condval_167 = condval_168;
        }
        canonical[0] = condval_167;
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          uint condval_170;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_170 = s0;
          } else {
            uint condval_171;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_171 = s1;
            } else {
              uint condval_172;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_172 = s2;
              } else {
                condval_172 = s3;
              }
              condval_171 = condval_172;
            }
            condval_170 = condval_171;
          }
          uint v = ((condval_170 >> ((uint)(byte * 8))) & (uint)255);
          uint condval_173;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_173 = s0;
          } else {
            uint condval_174;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_174 = s1;
            } else {
              uint condval_175;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_175 = s2;
              } else {
                condval_175 = s3;
              }
              condval_174 = condval_175;
            }
            condval_173 = condval_174;
          }
          if ((((condval_173 >> ((uint)(byte * 8))) & (uint)255) & (uint)127) == (uint)127) {
            uint condval_176;
            if (((((int)threadIdx.x) % 4) == 0)) {
              condval_176 = s0;
            } else {
              uint condval_177;
              if (((((int)threadIdx.x) & 3) == 1)) {
                condval_177 = s1;
              } else {
                uint condval_178;
                if (((((int)threadIdx.x) & 3) == 2)) {
                  condval_178 = s2;
                } else {
                  condval_178 = s3;
                }
                condval_177 = condval_178;
              }
              condval_176 = condval_177;
            }
            canonical[0] = ((canonical[0] & ((uint)4294967295 ^ ((uint)255 << ((uint)(byte * 8))))) | (nr_tl_shallow::e4(((half_t)nr_tl_shallow::une4(((condval_176 >> ((uint)(byte * 8))) & (uint)255)))) << ((uint)(byte * 8))));
          }
        }
        aa[(((m * 4) + (p * 2)) + i)] = canonical[0];
        #pragma unroll
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          uint condval_179;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_179 = s0;
          } else {
            uint condval_180;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_180 = s1;
            } else {
              uint condval_181;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_181 = s2;
              } else {
                condval_181 = s3;
              }
              condval_180 = condval_181;
            }
            condval_179 = condval_180;
          }
          uint condval_182;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_182 = s0;
          } else {
            uint condval_183;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_183 = s1;
            } else {
              uint condval_184;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_184 = s2;
              } else {
                condval_184 = s3;
              }
              condval_183 = condval_184;
            }
            condval_182 = condval_183;
          }
          uint raw_1 = nr_tl_shallow::pack(((half_t)nr_tl_shallow::une4((condval_179 >> ((uint)(j_3 * 16))))), ((half_t)nr_tl_shallow::une4((condval_182 >> ((uint)((j_3 * 16) + 8))))));
          ff[((((m * 8) + (p * 4)) + (j_3 * 2)) + i)] = nr_tl_shallow::mul(raw_1, nr_tl_shallow::pack(G0[(((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2))], G0[((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
        }
      }
    }
  }
  ushort v_ = (ushort)13962;
  half_t scale = (*(half_t *)(&(v_)));
  for (int part = 0; part < 4; ++part) {
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      #pragma unroll
      for (int j_4 = 0; j_4 < 16; ++j_4) {
        hidden[j_4] = (uint)0;
      }
      nr_tl_shallow_joint::ld128((&(b[0])), (&(Wt0[(((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_1 = 0; m_1 < 4; ++m_1) {
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          nr_tl_shallow::mma8((&(hidden[((m_1 * 4) + (n * 2))])), aa[(m_1 * 4)], aa[((m_1 * 4) + 1)], aa[((m_1 * 4) + 2)], aa[((m_1 * 4) + 3)], b[(n * 2)], b[((n * 2) + 1)]);
        }
      }
      #pragma unroll
      for (int m_2 = 0; m_2 < 4; ++m_2) {
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          ha_[(((m_2 * 4) + (pair * 2)) + i_1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[((m_2 * 4) + i_1)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_2 * 4) + i_1)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_2 * 4) + i_1)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) | (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[(((m_2 * 4) + i_1) + 2)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_2 * 4) + i_1) + 2)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_2 * 4) + i_1) + 2)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; ++p_1) {
      nr_tl_shallow_joint::ld128((&(b_1[0])), (&(Wt1[(((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_3 = 0; m_3 < 4; ++m_3) {
        #pragma unroll
        for (int n_1 = 0; n_1 < 2; ++n_1) {
          nr_tl_shallow::mma8((&(ff[(((m_3 * 8) + (p_1 * 4)) + (n_1 * 2))])), ha_[(m_3 * 4)], ha_[((m_3 * 4) + 1)], ha_[((m_3 * 4) + 2)], ha_[((m_3 * 4) + 3)], b_1[(n_1 * 2)], b_1[((n_1 * 2) + 1)]);
        }
      }
    }
  }
  #pragma unroll
  for (int m_4 = 0; m_4 < 4; ++m_4) {
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; ++p_2) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        aa[(((m_4 * 4) + (p_2 * 2)) + i_2)] = (nr_tl_shallow::e4pair(ff[(((m_4 * 8) + (p_2 * 4)) + i_2)]) | (nr_tl_shallow::e4pair(ff[((((m_4 * 8) + (p_2 * 4)) + i_2) + 2)]) << (uint)16));
      }
    }
  }
  #pragma unroll
  for (int component = 0; component < 3; ++component) {
    #pragma unroll
    for (int j_5 = 0; j_5 < 32; ++j_5) {
      z[j_5] = (uint)0;
    }
    #pragma unroll
    for (int p_3 = 0; p_3 < 2; ++p_3) {
      nr_tl_shallow_joint::ld128((&(b_2[0])), (&(Wq[(((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_5 = 0; m_5 < 4; ++m_5) {
        #pragma unroll
        for (int n_2 = 0; n_2 < 2; ++n_2) {
          nr_tl_shallow::mma8((&(z[(((m_5 * 8) + (p_3 * 4)) + (n_2 * 2))])), aa[(m_5 * 4)], aa[((m_5 * 4) + 1)], aa[((m_5 * 4) + 2)], aa[((m_5 * 4) + 3)], b_2[(n_2 * 2)], b_2[((n_2 * 2) + 1)]);
        }
      }
    }
    if (component < 2) {
      #pragma unroll
      for (int row = 0; row < 8; ++row) {
        uint s = nr_tl_shallow::add(nr_tl_shallow::fma(z[(((row >> 1) * 8) + (row & 1))], z[(((row >> 1) * 8) + (row & 1))], nr_tl_shallow::mul(z[((((row >> 1) * 8) + (row & 1)) + 4)], z[((((row >> 1) * 8) + (row & 1)) + 4)])), nr_tl_shallow::fma(z[((((row >> 1) * 8) + (row & 1)) + 2)], z[((((row >> 1) * 8) + (row & 1)) + 2)], nr_tl_shallow::mul(z[((((row >> 1) * 8) + (row & 1)) + 6)], z[((((row >> 1) * 8) + (row & 1)) + 6)])));
        uint t = nr_tl_shallow::add(nr_tl_shallow::add(nr_tl_shallow::shfl(s, (((int)threadIdx.x) & 28)), nr_tl_shallow::shfl(s, ((((int)threadIdx.x) & 28) + 2))), nr_tl_shallow::add(nr_tl_shallow::shfl(s, ((((int)threadIdx.x) & 28) + 1)), nr_tl_shallow::shfl(s, ((((int)threadIdx.x) & 28) + 3))));
        half_t h = cutlass::half_t(__hmax((((half_t)nr_tl_shallow::hadd(((half_t)nr_tl_shallow::unpack(t, 0)), ((half_t)nr_tl_shallow::unpack(t, 1))))).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half()));
        inv[row] = nr_tl_shallow::pack(((half_t)nr_tl_shallow::rsqrt(h)), ((half_t)nr_tl_shallow::rsqrt(h)));
      }
      #pragma unroll
      for (int j_6 = 0; j_6 < 32; ++j_6) {
        z[j_6] = nr_tl_shallow::mul(z[j_6], inv[(((j_6 >> 3) * 2) + (j_6 & 1))]);
        if (component == 0) {
          ushort v__1 = (ushort)13962;
          z[j_6] = nr_tl_shallow::mul(z[j_6], nr_tl_shallow::pack((*(half_t *)(&(v__1))), (*(half_t *)(&(v__1)))));
        }
      }
    }
    #pragma unroll
    for (int m_6 = 0; m_6 < 4; ++m_6) {
      #pragma unroll
      for (int p_4 = 0; p_4 < 2; ++p_4) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          if (component == 0) {
            int64_t condval_185;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_185 = (int64_t)0;
            } else {
              int64_t condval_186;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_186 = (int64_t)3;
              } else {
                condval_186 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_185 = condval_186;
            }
            int64_t condval_187;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_187 = (int64_t)0;
            } else {
              int64_t condval_188;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_188 = (int64_t)3;
              } else {
                condval_188 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_187 = condval_188;
            }
            int64_t condval_189;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_189 = (int64_t)0;
            } else {
              int64_t condval_190;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_190 = (int64_t)3;
              } else {
                condval_190 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_189 = condval_190;
            }
            int64_t condval_191;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_191 = (int64_t)0;
            } else {
              int64_t condval_192;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_192 = (int64_t)3;
              } else {
                condval_192 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_191 = condval_192;
            }
            qa[(((m_6 * 4) + (p_4 * 2)) + i_3)] = (nr_tl_shallow::e4pair(z[(((condval_187 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_191 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_193;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_193 = (int64_t)0;
            } else {
              int64_t condval_194;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_194 = (int64_t)3;
              } else {
                condval_194 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_193 = condval_194;
            }
            int64_t condval_195;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_195 = (int64_t)0;
            } else {
              int64_t condval_196;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_196 = (int64_t)3;
              } else {
                condval_196 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_195 = condval_196;
            }
            int64_t condval_197;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_197 = (int64_t)0;
            } else {
              int64_t condval_198;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_198 = (int64_t)3;
              } else {
                condval_198 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_197 = condval_198;
            }
            int64_t condval_199;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_199 = (int64_t)0;
            } else {
              int64_t condval_200;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_200 = (int64_t)3;
              } else {
                condval_200 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_199 = condval_200;
            }
            kb[(((m_6 * 4) + (i_3 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_195 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_199 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          int64_t condval_201;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_201 = (int64_t)0;
          } else {
            condval_201 = (int64_t)1;
          }
          int64_t condval_202;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_202 = (int64_t)0;
          } else {
            condval_202 = (int64_t)1;
          }
          int64_t condval_203;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_203 = (int64_t)0;
          } else {
            condval_203 = (int64_t)1;
          }
          int64_t condval_204;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_204 = (int64_t)0;
          } else {
            condval_204 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_3 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_202 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_204 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_205;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_205 = (int64_t)3;
          } else {
            condval_205 = (int64_t)2;
          }
          int64_t condval_206;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_206 = (int64_t)3;
          } else {
            condval_206 = (int64_t)2;
          }
          int64_t condval_207;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_207 = (int64_t)3;
          } else {
            condval_207 = (int64_t)2;
          }
          int64_t condval_208;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_208 = (int64_t)3;
          } else {
            condval_208 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_3 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_206 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_208 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
        }
      }
    }
  }
  #pragma unroll
  for (int slab = 0; slab < 2; ++slab) {
    #pragma unroll
    for (int m_7 = 0; m_7 < 2; ++m_7) {
      #pragma unroll
      for (int p_5 = 0; p_5 < 4; ++p_5) {
        nr_tl_shallow_joint::ld128((&(seed[0])), (&(Wq[(((((slab * 1024) + (m_7 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768)])));
        #pragma unroll
        for (int n_4 = 0; n_4 < 2; ++n_4) {
          logits[(((m_7 * 16) + (p_5 * 4)) + (n_4 * 2))] = seed[(n_4 * 2)];
          logits[((((m_7 * 16) + (p_5 * 4)) + (n_4 * 2)) + 1)] = seed[((n_4 * 2) + 1)];
          nr_tl_shallow::mma8((&(logits[(((m_7 * 16) + (p_5 * 4)) + (n_4 * 2))])), qa[((slab * 8) + (m_7 * 4))], qa[(((slab * 8) + (m_7 * 4)) + 1)], qa[(((slab * 8) + (m_7 * 4)) + 2)], qa[(((slab * 8) + (m_7 * 4)) + 3)], kb[((p_5 * 4) + (n_4 * 2))], kb[(((p_5 * 4) + (n_4 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int j_7 = 0; j_7 < 32; ++j_7) {
      logits[j_7] = (((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_7], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) & (uint)65535) << (uint)5) + (uint)32768) & (uint)65535) | ((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_7], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) >> (uint)16) << (uint)5) + (uint)32768) << (uint)16));
    }
    #pragma unroll
    for (int row_1 = 0; row_1 < 4; ++row_1) {
      uint s0_1 = nr_tl_shallow::add(logits[(((row_1 >> 1) * 16) + (row_1 & 1))], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 2)]);
      uint s1_1 = nr_tl_shallow::add(s0_1, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 4)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 6)]));
      uint s2_1 = nr_tl_shallow::add(s1_1, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 8)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 10)]));
      uint s_1 = nr_tl_shallow::add(s2_1, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 12)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 14)]));
      uint t_1 = nr_tl_shallow::add(nr_tl_shallow::add(nr_tl_shallow::add(nr_tl_shallow::shfl(s_1, (((int)threadIdx.x) & 28)), nr_tl_shallow::shfl(s_1, ((((int)threadIdx.x) & 28) + 1))), nr_tl_shallow::shfl(s_1, ((((int)threadIdx.x) & 28) + 2))), nr_tl_shallow::shfl(s_1, ((((int)threadIdx.x) & 28) + 3)));
      half_t h_1 = cutlass::half_t(__hmax((((half_t)nr_tl_shallow::hadd(((half_t)nr_tl_shallow::unpack(t_1, 0)), ((half_t)nr_tl_shallow::unpack(t_1, 1))))).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half()));
      inv[row_1] = nr_tl_shallow::pack(((half_t)nr_tl_shallow::rcp(h_1)), ((half_t)nr_tl_shallow::rcp(h_1)));
    }
    #pragma unroll
    for (int j_8 = 0; j_8 < 32; ++j_8) {
      logits[j_8] = nr_tl_shallow::mul(logits[j_8], inv[(((j_8 >> 4) * 2) + (j_8 & 1))]);
    }
    #pragma unroll
    for (int part_2 = 0; part_2 < 2; ++part_2) {
      #pragma unroll
      for (int m_8 = 0; m_8 < 2; ++m_8) {
        #pragma unroll
        for (int p_6 = 0; p_6 < 2; ++p_6) {
          #pragma unroll
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            pa[((((part_2 * 8) + (m_8 * 4)) + (p_6 * 2)) + i_4)] = (nr_tl_shallow::e4pair(logits[((((m_8 * 16) + (part_2 * 8)) + (p_6 * 4)) + i_4)]) | (nr_tl_shallow::e4pair(logits[(((((m_8 * 16) + (part_2 * 8)) + (p_6 * 4)) + i_4) + 2)]) << (uint)16));
          }
        }
      }
    }
    #pragma unroll
    for (int j_9 = 0; j_9 < 16; ++j_9) {
      attended[j_9] = (uint)0;
    }
    #pragma unroll
    for (int part_3 = 0; part_3 < 2; ++part_3) {
      #pragma unroll
      for (int n_5 = 0; n_5 < 4; ++n_5) {
        #pragma unroll
        for (int m_9 = 0; m_9 < 2; ++m_9) {
          nr_tl_shallow::mma8((&(attended[((m_9 * 8) + (n_5 * 2))])), pa[((part_3 * 8) + (m_9 * 4))], pa[(((part_3 * 8) + (m_9 * 4)) + 1)], pa[(((part_3 * 8) + (m_9 * 4)) + 2)], pa[(((part_3 * 8) + (m_9 * 4)) + 3)], vb[((part_3 * 8) + (n_5 * 2))], vb[(((part_3 * 8) + (n_5 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int m_10 = 0; m_10 < 2; ++m_10) {
      #pragma unroll
      for (int p_7 = 0; p_7 < 2; ++p_7) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          aa[(((m_10 * 4) + (p_7 * 2)) + i_5)] = (nr_tl_shallow::e4pair(attended[(((m_10 * 8) + (p_7 * 4)) + i_5)]) | (nr_tl_shallow::e4pair(attended[((((m_10 * 8) + (p_7 * 4)) + i_5) + 2)]) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int j_10 = 0; j_10 < 16; ++j_10) {
      int64_t condval_209;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_209 = (int64_t)0;
      } else {
        int64_t condval_210;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_210 = (int64_t)3;
        } else {
          condval_210 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_209 = condval_210;
      }
      int64_t condval_211;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_211 = (int64_t)0;
      } else {
        int64_t condval_212;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_212 = (int64_t)3;
        } else {
          condval_212 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_211 = condval_212;
      }
      out[j_10] = nr_tl_shallow::mul(ff[((condval_211 * (int64_t)8) + (((int64_t)j_10) & (int64_t)7))], nr_tl_shallow::pack(G1[((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))], G1[(((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
    }
    #pragma unroll
    for (int p_8 = 0; p_8 < 2; ++p_8) {
      nr_tl_shallow_joint::ld128((&(b_3[0])), (&(Wp[((p_8 * 128) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_11 = 0; m_11 < 2; ++m_11) {
        #pragma unroll
        for (int n_6 = 0; n_6 < 2; ++n_6) {
          nr_tl_shallow::mma8((&(out[(((m_11 * 8) + (p_8 * 4)) + (n_6 * 2))])), aa[(m_11 * 4)], aa[((m_11 * 4) + 1)], aa[((m_11 * 4) + 2)], aa[((m_11 * 4) + 3)], b_3[(n_6 * 2)], b_3[((n_6 * 2) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int m_12 = 0; m_12 < 2; ++m_12) {
      int condval_213;
      if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
        int condval_214;
        if ((((slab * 2) + m_12) == 0)) {
          condval_214 = 0;
        } else {
          int condval_215;
          if ((((slab * 2) + m_12) == 1)) {
            condval_215 = 3;
          } else {
            condval_215 = (((slab * 2) + m_12) - 1);
          }
          condval_214 = condval_215;
        }
        int condval_216;
        if ((((slab * 2) + m_12) == 0)) {
          condval_216 = 0;
        } else {
          int condval_217;
          if ((((slab * 2) + m_12) == 1)) {
            condval_217 = 3;
          } else {
            condval_217 = (((slab * 2) + m_12) - 1);
          }
          condval_216 = condval_217;
        }
        int condval_218;
        if ((((slab * 2) + m_12) == 0)) {
          condval_218 = 0;
        } else {
          int condval_219;
          if ((((slab * 2) + m_12) == 1)) {
            condval_219 = 3;
          } else {
            condval_219 = (((slab * 2) + m_12) - 1);
          }
          condval_218 = condval_219;
        }
        int condval_220;
        if ((((slab * 2) + m_12) == 0)) {
          condval_220 = 0;
        } else {
          int condval_221;
          if ((((slab * 2) + m_12) == 1)) {
            condval_221 = 3;
          } else {
            condval_221 = (((slab * 2) + m_12) - 1);
          }
          condval_220 = condval_221;
        }
        int condval_222;
        if ((((slab * 2) + m_12) == 0)) {
          condval_222 = 0;
        } else {
          int condval_223;
          if ((((slab * 2) + m_12) == 1)) {
            condval_223 = 3;
          } else {
            condval_223 = (((slab * 2) + m_12) - 1);
          }
          condval_222 = condval_223;
        }
        int condval_224;
        if ((((slab * 2) + m_12) == 0)) {
          condval_224 = 0;
        } else {
          int condval_225;
          if ((((slab * 2) + m_12) == 1)) {
            condval_225 = 3;
          } else {
            condval_225 = (((slab * 2) + m_12) - 1);
          }
          condval_224 = condval_225;
        }
        condval_213 = (((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((condval_214 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_216 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((condval_218 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_220 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_222 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_224 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
      } else {
        condval_213 = -1;
      }
      if (0 <= condval_213) {
        int condval_226;
        if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
          int condval_227;
          if ((((slab * 2) + m_12) == 0)) {
            condval_227 = 0;
          } else {
            int condval_228;
            if ((((slab * 2) + m_12) == 1)) {
              condval_228 = 3;
            } else {
              condval_228 = (((slab * 2) + m_12) - 1);
            }
            condval_227 = condval_228;
          }
          int condval_229;
          if ((((slab * 2) + m_12) == 0)) {
            condval_229 = 0;
          } else {
            int condval_230;
            if ((((slab * 2) + m_12) == 1)) {
              condval_230 = 3;
            } else {
              condval_230 = (((slab * 2) + m_12) - 1);
            }
            condval_229 = condval_230;
          }
          int condval_231;
          if ((((slab * 2) + m_12) == 0)) {
            condval_231 = 0;
          } else {
            int condval_232;
            if ((((slab * 2) + m_12) == 1)) {
              condval_232 = 3;
            } else {
              condval_232 = (((slab * 2) + m_12) - 1);
            }
            condval_231 = condval_232;
          }
          int condval_233;
          if ((((slab * 2) + m_12) == 0)) {
            condval_233 = 0;
          } else {
            int condval_234;
            if ((((slab * 2) + m_12) == 1)) {
              condval_234 = 3;
            } else {
              condval_234 = (((slab * 2) + m_12) - 1);
            }
            condval_233 = condval_234;
          }
          int condval_235;
          if ((((slab * 2) + m_12) == 0)) {
            condval_235 = 0;
          } else {
            int condval_236;
            if ((((slab * 2) + m_12) == 1)) {
              condval_236 = 3;
            } else {
              condval_236 = (((slab * 2) + m_12) - 1);
            }
            condval_235 = condval_236;
          }
          int condval_237;
          if ((((slab * 2) + m_12) == 0)) {
            condval_237 = 0;
          } else {
            int condval_238;
            if ((((slab * 2) + m_12) == 1)) {
              condval_238 = 3;
            } else {
              condval_238 = (((slab * 2) + m_12) - 1);
            }
            condval_237 = condval_238;
          }
          condval_226 = (((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((condval_227 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_229 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((condval_231 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_233 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_235 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_237 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
        } else {
          condval_226 = -1;
        }
        if (condval_226 < 47250848) {
          int condval_239;
          if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
            int condval_240;
            if ((((slab * 2) + m_12) == 0)) {
              condval_240 = 0;
            } else {
              int condval_241;
              if ((((slab * 2) + m_12) == 1)) {
                condval_241 = 3;
              } else {
                condval_241 = (((slab * 2) + m_12) - 1);
              }
              condval_240 = condval_241;
            }
            int condval_242;
            if ((((slab * 2) + m_12) == 0)) {
              condval_242 = 0;
            } else {
              int condval_243;
              if ((((slab * 2) + m_12) == 1)) {
                condval_243 = 3;
              } else {
                condval_243 = (((slab * 2) + m_12) - 1);
              }
              condval_242 = condval_243;
            }
            int condval_244;
            if ((((slab * 2) + m_12) == 0)) {
              condval_244 = 0;
            } else {
              int condval_245;
              if ((((slab * 2) + m_12) == 1)) {
                condval_245 = 3;
              } else {
                condval_245 = (((slab * 2) + m_12) - 1);
              }
              condval_244 = condval_245;
            }
            int condval_246;
            if ((((slab * 2) + m_12) == 0)) {
              condval_246 = 0;
            } else {
              int condval_247;
              if ((((slab * 2) + m_12) == 1)) {
                condval_247 = 3;
              } else {
                condval_247 = (((slab * 2) + m_12) - 1);
              }
              condval_246 = condval_247;
            }
            int condval_248;
            if ((((slab * 2) + m_12) == 0)) {
              condval_248 = 0;
            } else {
              int condval_249;
              if ((((slab * 2) + m_12) == 1)) {
                condval_249 = 3;
              } else {
                condval_249 = (((slab * 2) + m_12) - 1);
              }
              condval_248 = condval_249;
            }
            int condval_250;
            if ((((slab * 2) + m_12) == 0)) {
              condval_250 = 0;
            } else {
              int condval_251;
              if ((((slab * 2) + m_12) == 1)) {
                condval_251 = 3;
              } else {
                condval_251 = (((slab * 2) + m_12) - 1);
              }
              condval_250 = condval_251;
            }
            condval_239 = (((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((condval_240 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_242 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((condval_244 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_246 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_248 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_250 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
          } else {
            condval_239 = -1;
          }
          int condval_252;
          if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
            int condval_253;
            if ((((slab * 2) + m_12) == 0)) {
              condval_253 = 0;
            } else {
              int condval_254;
              if ((((slab * 2) + m_12) == 1)) {
                condval_254 = 3;
              } else {
                condval_254 = (((slab * 2) + m_12) - 1);
              }
              condval_253 = condval_254;
            }
            int condval_255;
            if ((((slab * 2) + m_12) == 0)) {
              condval_255 = 0;
            } else {
              int condval_256;
              if ((((slab * 2) + m_12) == 1)) {
                condval_256 = 3;
              } else {
                condval_256 = (((slab * 2) + m_12) - 1);
              }
              condval_255 = condval_256;
            }
            int condval_257;
            if ((((slab * 2) + m_12) == 0)) {
              condval_257 = 0;
            } else {
              int condval_258;
              if ((((slab * 2) + m_12) == 1)) {
                condval_258 = 3;
              } else {
                condval_258 = (((slab * 2) + m_12) - 1);
              }
              condval_257 = condval_258;
            }
            int condval_259;
            if ((((slab * 2) + m_12) == 0)) {
              condval_259 = 0;
            } else {
              int condval_260;
              if ((((slab * 2) + m_12) == 1)) {
                condval_260 = 3;
              } else {
                condval_260 = (((slab * 2) + m_12) - 1);
              }
              condval_259 = condval_260;
            }
            int condval_261;
            if ((((slab * 2) + m_12) == 0)) {
              condval_261 = 0;
            } else {
              int condval_262;
              if ((((slab * 2) + m_12) == 1)) {
                condval_262 = 3;
              } else {
                condval_262 = (((slab * 2) + m_12) - 1);
              }
              condval_261 = condval_262;
            }
            int condval_263;
            if ((((slab * 2) + m_12) == 0)) {
              condval_263 = 0;
            } else {
              int condval_264;
              if ((((slab * 2) + m_12) == 1)) {
                condval_264 = 3;
              } else {
                condval_264 = (((slab * 2) + m_12) - 1);
              }
              condval_263 = condval_264;
            }
            condval_252 = (((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((condval_253 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_255 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((condval_257 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_259 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_261 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_263 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
          } else {
            condval_252 = -1;
          }
          nr_tl_shallow_joint::st128((&(X[(condval_252 + 13107200)])), (nr_tl_shallow::e4pair(out[(m_12 * 8)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 2)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 1)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 3)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 4)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 6)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 5)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 7)]) << (uint)16)));
        }
      }
    }
  }
  if (((int)threadIdx.x) == 0) {
    nr_tl_shallow_joint::st32((&(X[((((int)blockIdx.x) * 4) + 60293120)])), (uint)0);
  }
}

