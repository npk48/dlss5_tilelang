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

extern "C" __global__ void main_kernel(const half_t* __restrict__ G0, const half_t* __restrict__ G1, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, int H0, int W0, int counter, int gx, int gy, int in_offset, int out_offset, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const half_t* __restrict__ G0, const half_t* __restrict__ G1, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, int H0, int W0, int counter, int gx, int gy, int in_offset, int out_offset, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6) {
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
    int rmod = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_1 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_1 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_2 = (((int)blockIdx.x) % (W0 >> 4));
    int rmod_3 = (((int)blockIdx.x) % (W0 >> 4));
    int condval;
    if (((((0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod)) || (((W0 >> 4) < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_1)) || (((W0 >> 4) < 0) && (rmod_1 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) < (H0 >> 3))) & (0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod_2)) || (((W0 >> 4) < 0) && (rmod_2 <= 0))) ? rmod_2 : (rmod_2 + (W0 >> 4))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_3)) || (((W0 >> 4) < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) < (W0 >> 3)))) {
      int rmod_4 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_2 = (((int)blockIdx.x) / (W0 >> 4));
      int rmod_5 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_3 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_1;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_5)) || (((W0 >> 4) < 0) && (rmod_5 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_6 = (((int)blockIdx.x) % (W0 >> 4));
        condval_1 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_6)) || (((W0 >> 4) < 0) && (rmod_6 <= 0))) ? rmod_6 : (rmod_6 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_7 = (((int)blockIdx.x) % (W0 >> 4));
        condval_1 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_7)) || (((W0 >> 4) < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_8 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_4 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_2;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_8)) || (((W0 >> 4) < 0) && (rmod_8 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_9 = (((int)blockIdx.x) % (W0 >> 4));
        condval_2 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_9)) || (((W0 >> 4) < 0) && (rmod_9 <= 0))) ? rmod_9 : (rmod_9 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_10 = (((int)blockIdx.x) % (W0 >> 4));
        condval_2 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_10)) || (((W0 >> 4) < 0) && (rmod_10 <= 0))) ? rmod_10 : (rmod_10 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_11 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_5 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_3;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_11)) || (((W0 >> 4) < 0) && (rmod_11 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_12 = (((int)blockIdx.x) % (W0 >> 4));
        condval_3 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_12)) || (((W0 >> 4) < 0) && (rmod_12 <= 0))) ? rmod_12 : (rmod_12 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_13 = (((int)blockIdx.x) % (W0 >> 4));
        condval_3 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_13)) || (((W0 >> 4) < 0) && (rmod_13 <= 0))) ? rmod_13 : (rmod_13 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_14 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_6 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_4;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_14)) || (((W0 >> 4) < 0) && (rmod_14 <= 0))) ? rdiv_6 : (rdiv_6 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_15 = (((int)blockIdx.x) % (W0 >> 4));
        condval_4 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_15)) || (((W0 >> 4) < 0) && (rmod_15 <= 0))) ? rmod_15 : (rmod_15 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_16 = (((int)blockIdx.x) % (W0 >> 4));
        condval_4 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_16)) || (((W0 >> 4) < 0) && (rmod_16 <= 0))) ? rmod_16 : (rmod_16 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_17 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_7 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_5;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_17)) || (((W0 >> 4) < 0) && (rmod_17 <= 0))) ? rdiv_7 : (rdiv_7 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_18 = (((int)blockIdx.x) % (W0 >> 4));
        condval_5 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_18)) || (((W0 >> 4) < 0) && (rmod_18 <= 0))) ? rmod_18 : (rmod_18 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_19 = (((int)blockIdx.x) % (W0 >> 4));
        condval_5 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_19)) || (((W0 >> 4) < 0) && (rmod_19 <= 0))) ? rmod_19 : (rmod_19 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_20 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_8 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_6;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_20)) || (((W0 >> 4) < 0) && (rmod_20 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_21 = (((int)blockIdx.x) % (W0 >> 4));
        condval_6 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_21)) || (((W0 >> 4) < 0) && (rmod_21 <= 0))) ? rmod_21 : (rmod_21 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_22 = (((int)blockIdx.x) % (W0 >> 4));
        condval_6 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_22)) || (((W0 >> 4) < 0) && (rmod_22 <= 0))) ? rmod_22 : (rmod_22 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_23 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_9 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_7;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_23)) || (((W0 >> 4) < 0) && (rmod_23 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_24 = (((int)blockIdx.x) % (W0 >> 4));
        condval_7 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_24)) || (((W0 >> 4) < 0) && (rmod_24 <= 0))) ? rmod_24 : (rmod_24 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_25 = (((int)blockIdx.x) % (W0 >> 4));
        condval_7 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_25)) || (((W0 >> 4) < 0) && (rmod_25 <= 0))) ? rmod_25 : (rmod_25 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_26 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_27 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_10 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_8;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_27)) || (((W0 >> 4) < 0) && (rmod_27 <= 0))) ? rdiv_10 : (rdiv_10 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_28 = (((int)blockIdx.x) % (W0 >> 4));
        condval_8 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_28)) || (((W0 >> 4) < 0) && (rmod_28 <= 0))) ? rmod_28 : (rmod_28 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_29 = (((int)blockIdx.x) % (W0 >> 4));
        condval_8 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_29)) || (((W0 >> 4) < 0) && (rmod_29 <= 0))) ? rmod_29 : (rmod_29 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_30 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_11 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_9;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_30)) || (((W0 >> 4) < 0) && (rmod_30 <= 0))) ? rdiv_11 : (rdiv_11 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_31 = (((int)blockIdx.x) % (W0 >> 4));
        condval_9 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_31)) || (((W0 >> 4) < 0) && (rmod_31 <= 0))) ? rmod_31 : (rmod_31 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_32 = (((int)blockIdx.x) % (W0 >> 4));
        condval_9 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_32)) || (((W0 >> 4) < 0) && (rmod_32 <= 0))) ? rmod_32 : (rmod_32 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_33 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_12 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_10;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_33)) || (((W0 >> 4) < 0) && (rmod_33 <= 0))) ? rdiv_12 : (rdiv_12 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_34 = (((int)blockIdx.x) % (W0 >> 4));
        condval_10 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_34)) || (((W0 >> 4) < 0) && (rmod_34 <= 0))) ? rmod_34 : (rmod_34 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_35 = (((int)blockIdx.x) % (W0 >> 4));
        condval_10 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_35)) || (((W0 >> 4) < 0) && (rmod_35 <= 0))) ? rmod_35 : (rmod_35 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_36 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_13 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_11;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_36)) || (((W0 >> 4) < 0) && (rmod_36 <= 0))) ? rdiv_13 : (rdiv_13 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_37 = (((int)blockIdx.x) % (W0 >> 4));
        condval_11 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_37)) || (((W0 >> 4) < 0) && (rmod_37 <= 0))) ? rmod_37 : (rmod_37 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_38 = (((int)blockIdx.x) % (W0 >> 4));
        condval_11 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_38)) || (((W0 >> 4) < 0) && (rmod_38 <= 0))) ? rmod_38 : (rmod_38 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_39 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_14 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_12;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_39)) || (((W0 >> 4) < 0) && (rmod_39 <= 0))) ? rdiv_14 : (rdiv_14 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_40 = (((int)blockIdx.x) % (W0 >> 4));
        condval_12 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_40)) || (((W0 >> 4) < 0) && (rmod_40 <= 0))) ? rmod_40 : (rmod_40 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_41 = (((int)blockIdx.x) % (W0 >> 4));
        condval_12 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_41)) || (((W0 >> 4) < 0) && (rmod_41 <= 0))) ? rmod_41 : (rmod_41 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_42 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_15 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_13;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_42)) || (((W0 >> 4) < 0) && (rmod_42 <= 0))) ? rdiv_15 : (rdiv_15 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_43 = (((int)blockIdx.x) % (W0 >> 4));
        condval_13 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_43)) || (((W0 >> 4) < 0) && (rmod_43 <= 0))) ? rmod_43 : (rmod_43 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_44 = (((int)blockIdx.x) % (W0 >> 4));
        condval_13 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_44)) || (((W0 >> 4) < 0) && (rmod_44 <= 0))) ? rmod_44 : (rmod_44 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_45 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_46 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_16 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_14;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_46)) || (((W0 >> 4) < 0) && (rmod_46 <= 0))) ? rdiv_16 : (rdiv_16 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_47 = (((int)blockIdx.x) % (W0 >> 4));
        condval_14 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_47)) || (((W0 >> 4) < 0) && (rmod_47 <= 0))) ? rmod_47 : (rmod_47 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_48 = (((int)blockIdx.x) % (W0 >> 4));
        condval_14 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_48)) || (((W0 >> 4) < 0) && (rmod_48 <= 0))) ? rmod_48 : (rmod_48 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_49 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_17 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_15;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_49)) || (((W0 >> 4) < 0) && (rmod_49 <= 0))) ? rdiv_17 : (rdiv_17 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_50 = (((int)blockIdx.x) % (W0 >> 4));
        condval_15 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_50)) || (((W0 >> 4) < 0) && (rmod_50 <= 0))) ? rmod_50 : (rmod_50 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_51 = (((int)blockIdx.x) % (W0 >> 4));
        condval_15 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_51)) || (((W0 >> 4) < 0) && (rmod_51 <= 0))) ? rmod_51 : (rmod_51 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_52 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_18 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_16;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_52)) || (((W0 >> 4) < 0) && (rmod_52 <= 0))) ? rdiv_18 : (rdiv_18 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_53 = (((int)blockIdx.x) % (W0 >> 4));
        condval_16 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_53)) || (((W0 >> 4) < 0) && (rmod_53 <= 0))) ? rmod_53 : (rmod_53 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_54 = (((int)blockIdx.x) % (W0 >> 4));
        condval_16 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_54)) || (((W0 >> 4) < 0) && (rmod_54 <= 0))) ? rmod_54 : (rmod_54 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_55 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_19 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_17;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_55)) || (((W0 >> 4) < 0) && (rmod_55 <= 0))) ? rdiv_19 : (rdiv_19 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_56 = (((int)blockIdx.x) % (W0 >> 4));
        condval_17 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_56)) || (((W0 >> 4) < 0) && (rmod_56 <= 0))) ? rmod_56 : (rmod_56 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_57 = (((int)blockIdx.x) % (W0 >> 4));
        condval_17 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_57)) || (((W0 >> 4) < 0) && (rmod_57 <= 0))) ? rmod_57 : (rmod_57 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_58 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_20 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_18;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_58)) || (((W0 >> 4) < 0) && (rmod_58 <= 0))) ? rdiv_20 : (rdiv_20 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_59 = (((int)blockIdx.x) % (W0 >> 4));
        condval_18 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_59)) || (((W0 >> 4) < 0) && (rmod_59 <= 0))) ? rmod_59 : (rmod_59 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_60 = (((int)blockIdx.x) % (W0 >> 4));
        condval_18 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_60)) || (((W0 >> 4) < 0) && (rmod_60 <= 0))) ? rmod_60 : (rmod_60 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_61 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_21 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_19;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_61)) || (((W0 >> 4) < 0) && (rmod_61 <= 0))) ? rdiv_21 : (rdiv_21 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_62 = (((int)blockIdx.x) % (W0 >> 4));
        condval_19 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_62)) || (((W0 >> 4) < 0) && (rmod_62 <= 0))) ? rmod_62 : (rmod_62 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_63 = (((int)blockIdx.x) % (W0 >> 4));
        condval_19 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_63)) || (((W0 >> 4) < 0) && (rmod_63 <= 0))) ? rmod_63 : (rmod_63 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      condval = ((((((((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_4)) || (((W0 >> 4) < 0) && (rmod_4 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_1 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) + ((H0 >> 4) * ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_2 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_3 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_4 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_5 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_6 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_7 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10))) * (W0 >> 3)) * 512) + ((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_26)) || (((W0 >> 4) < 0) && (rmod_26 <= 0))) ? rmod_26 : (rmod_26 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + (((W0 >> 6) * (((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_8 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_9 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_10 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_11 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_12 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_13 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + (((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_45)) || (((W0 >> 4) < 0) && (rmod_45 <= 0))) ? rmod_45 : (rmod_45 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_14 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_15 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_16 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_17 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_18 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_19 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
    } else {
      condval = -1;
    }
    int rmod_64 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_22 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_65 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_23 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_66 = (((int)blockIdx.x) % (W0 >> 4));
    int rmod_67 = (((int)blockIdx.x) % (W0 >> 4));
    int condval_20;
    if (((((0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod_64)) || (((W0 >> 4) < 0) && (rmod_64 <= 0))) ? rdiv_22 : (rdiv_22 - 1))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_65)) || (((W0 >> 4) < 0) && (rmod_65 <= 0))) ? rdiv_23 : (rdiv_23 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) < (H0 >> 3))) & (0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod_66)) || (((W0 >> 4) < 0) && (rmod_66 <= 0))) ? rmod_66 : (rmod_66 + (W0 >> 4))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_67)) || (((W0 >> 4) < 0) && (rmod_67 <= 0))) ? rmod_67 : (rmod_67 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) < (W0 >> 3)))) {
      int rmod_68 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_24 = (((int)blockIdx.x) / (W0 >> 4));
      int rmod_69 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_25 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_21;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_69)) || (((W0 >> 4) < 0) && (rmod_69 <= 0))) ? rdiv_25 : (rdiv_25 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_70 = (((int)blockIdx.x) % (W0 >> 4));
        condval_21 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_70)) || (((W0 >> 4) < 0) && (rmod_70 <= 0))) ? rmod_70 : (rmod_70 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_71 = (((int)blockIdx.x) % (W0 >> 4));
        condval_21 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_71)) || (((W0 >> 4) < 0) && (rmod_71 <= 0))) ? rmod_71 : (rmod_71 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_72 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_26 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_22;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_72)) || (((W0 >> 4) < 0) && (rmod_72 <= 0))) ? rdiv_26 : (rdiv_26 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_73 = (((int)blockIdx.x) % (W0 >> 4));
        condval_22 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_73)) || (((W0 >> 4) < 0) && (rmod_73 <= 0))) ? rmod_73 : (rmod_73 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_74 = (((int)blockIdx.x) % (W0 >> 4));
        condval_22 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_74)) || (((W0 >> 4) < 0) && (rmod_74 <= 0))) ? rmod_74 : (rmod_74 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_75 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_27 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_23;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_75)) || (((W0 >> 4) < 0) && (rmod_75 <= 0))) ? rdiv_27 : (rdiv_27 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_76 = (((int)blockIdx.x) % (W0 >> 4));
        condval_23 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_76)) || (((W0 >> 4) < 0) && (rmod_76 <= 0))) ? rmod_76 : (rmod_76 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_77 = (((int)blockIdx.x) % (W0 >> 4));
        condval_23 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_77)) || (((W0 >> 4) < 0) && (rmod_77 <= 0))) ? rmod_77 : (rmod_77 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_78 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_28 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_24;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_78)) || (((W0 >> 4) < 0) && (rmod_78 <= 0))) ? rdiv_28 : (rdiv_28 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_79 = (((int)blockIdx.x) % (W0 >> 4));
        condval_24 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_79)) || (((W0 >> 4) < 0) && (rmod_79 <= 0))) ? rmod_79 : (rmod_79 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_80 = (((int)blockIdx.x) % (W0 >> 4));
        condval_24 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_80)) || (((W0 >> 4) < 0) && (rmod_80 <= 0))) ? rmod_80 : (rmod_80 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_81 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_29 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_25;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_81)) || (((W0 >> 4) < 0) && (rmod_81 <= 0))) ? rdiv_29 : (rdiv_29 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_82 = (((int)blockIdx.x) % (W0 >> 4));
        condval_25 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_82)) || (((W0 >> 4) < 0) && (rmod_82 <= 0))) ? rmod_82 : (rmod_82 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_83 = (((int)blockIdx.x) % (W0 >> 4));
        condval_25 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_83)) || (((W0 >> 4) < 0) && (rmod_83 <= 0))) ? rmod_83 : (rmod_83 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_84 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_30 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_26;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_84)) || (((W0 >> 4) < 0) && (rmod_84 <= 0))) ? rdiv_30 : (rdiv_30 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_85 = (((int)blockIdx.x) % (W0 >> 4));
        condval_26 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_85)) || (((W0 >> 4) < 0) && (rmod_85 <= 0))) ? rmod_85 : (rmod_85 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_86 = (((int)blockIdx.x) % (W0 >> 4));
        condval_26 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_86)) || (((W0 >> 4) < 0) && (rmod_86 <= 0))) ? rmod_86 : (rmod_86 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_87 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_31 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_27;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_87)) || (((W0 >> 4) < 0) && (rmod_87 <= 0))) ? rdiv_31 : (rdiv_31 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_88 = (((int)blockIdx.x) % (W0 >> 4));
        condval_27 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_88)) || (((W0 >> 4) < 0) && (rmod_88 <= 0))) ? rmod_88 : (rmod_88 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_89 = (((int)blockIdx.x) % (W0 >> 4));
        condval_27 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_89)) || (((W0 >> 4) < 0) && (rmod_89 <= 0))) ? rmod_89 : (rmod_89 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_90 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_91 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_32 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_28;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_91)) || (((W0 >> 4) < 0) && (rmod_91 <= 0))) ? rdiv_32 : (rdiv_32 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_92 = (((int)blockIdx.x) % (W0 >> 4));
        condval_28 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_92)) || (((W0 >> 4) < 0) && (rmod_92 <= 0))) ? rmod_92 : (rmod_92 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_93 = (((int)blockIdx.x) % (W0 >> 4));
        condval_28 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_93)) || (((W0 >> 4) < 0) && (rmod_93 <= 0))) ? rmod_93 : (rmod_93 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_94 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_33 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_29;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_94)) || (((W0 >> 4) < 0) && (rmod_94 <= 0))) ? rdiv_33 : (rdiv_33 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_95 = (((int)blockIdx.x) % (W0 >> 4));
        condval_29 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_95)) || (((W0 >> 4) < 0) && (rmod_95 <= 0))) ? rmod_95 : (rmod_95 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_96 = (((int)blockIdx.x) % (W0 >> 4));
        condval_29 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_96)) || (((W0 >> 4) < 0) && (rmod_96 <= 0))) ? rmod_96 : (rmod_96 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_97 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_34 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_30;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_97)) || (((W0 >> 4) < 0) && (rmod_97 <= 0))) ? rdiv_34 : (rdiv_34 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_98 = (((int)blockIdx.x) % (W0 >> 4));
        condval_30 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_98)) || (((W0 >> 4) < 0) && (rmod_98 <= 0))) ? rmod_98 : (rmod_98 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_99 = (((int)blockIdx.x) % (W0 >> 4));
        condval_30 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_99)) || (((W0 >> 4) < 0) && (rmod_99 <= 0))) ? rmod_99 : (rmod_99 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_100 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_35 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_31;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_100)) || (((W0 >> 4) < 0) && (rmod_100 <= 0))) ? rdiv_35 : (rdiv_35 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_101 = (((int)blockIdx.x) % (W0 >> 4));
        condval_31 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_101)) || (((W0 >> 4) < 0) && (rmod_101 <= 0))) ? rmod_101 : (rmod_101 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_102 = (((int)blockIdx.x) % (W0 >> 4));
        condval_31 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_102)) || (((W0 >> 4) < 0) && (rmod_102 <= 0))) ? rmod_102 : (rmod_102 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_103 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_36 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_32;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_103)) || (((W0 >> 4) < 0) && (rmod_103 <= 0))) ? rdiv_36 : (rdiv_36 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_104 = (((int)blockIdx.x) % (W0 >> 4));
        condval_32 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_104)) || (((W0 >> 4) < 0) && (rmod_104 <= 0))) ? rmod_104 : (rmod_104 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_105 = (((int)blockIdx.x) % (W0 >> 4));
        condval_32 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_105)) || (((W0 >> 4) < 0) && (rmod_105 <= 0))) ? rmod_105 : (rmod_105 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_106 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_37 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_33;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_106)) || (((W0 >> 4) < 0) && (rmod_106 <= 0))) ? rdiv_37 : (rdiv_37 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_107 = (((int)blockIdx.x) % (W0 >> 4));
        condval_33 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_107)) || (((W0 >> 4) < 0) && (rmod_107 <= 0))) ? rmod_107 : (rmod_107 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_108 = (((int)blockIdx.x) % (W0 >> 4));
        condval_33 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_108)) || (((W0 >> 4) < 0) && (rmod_108 <= 0))) ? rmod_108 : (rmod_108 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_109 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_110 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_38 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_34;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_110)) || (((W0 >> 4) < 0) && (rmod_110 <= 0))) ? rdiv_38 : (rdiv_38 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_111 = (((int)blockIdx.x) % (W0 >> 4));
        condval_34 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_111)) || (((W0 >> 4) < 0) && (rmod_111 <= 0))) ? rmod_111 : (rmod_111 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_112 = (((int)blockIdx.x) % (W0 >> 4));
        condval_34 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_112)) || (((W0 >> 4) < 0) && (rmod_112 <= 0))) ? rmod_112 : (rmod_112 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_113 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_39 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_35;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_113)) || (((W0 >> 4) < 0) && (rmod_113 <= 0))) ? rdiv_39 : (rdiv_39 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_114 = (((int)blockIdx.x) % (W0 >> 4));
        condval_35 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_114)) || (((W0 >> 4) < 0) && (rmod_114 <= 0))) ? rmod_114 : (rmod_114 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_115 = (((int)blockIdx.x) % (W0 >> 4));
        condval_35 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_115)) || (((W0 >> 4) < 0) && (rmod_115 <= 0))) ? rmod_115 : (rmod_115 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_116 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_40 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_36;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_116)) || (((W0 >> 4) < 0) && (rmod_116 <= 0))) ? rdiv_40 : (rdiv_40 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_117 = (((int)blockIdx.x) % (W0 >> 4));
        condval_36 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_117)) || (((W0 >> 4) < 0) && (rmod_117 <= 0))) ? rmod_117 : (rmod_117 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_118 = (((int)blockIdx.x) % (W0 >> 4));
        condval_36 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_118)) || (((W0 >> 4) < 0) && (rmod_118 <= 0))) ? rmod_118 : (rmod_118 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_119 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_41 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_37;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_119)) || (((W0 >> 4) < 0) && (rmod_119 <= 0))) ? rdiv_41 : (rdiv_41 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_120 = (((int)blockIdx.x) % (W0 >> 4));
        condval_37 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_120)) || (((W0 >> 4) < 0) && (rmod_120 <= 0))) ? rmod_120 : (rmod_120 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_121 = (((int)blockIdx.x) % (W0 >> 4));
        condval_37 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_121)) || (((W0 >> 4) < 0) && (rmod_121 <= 0))) ? rmod_121 : (rmod_121 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_122 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_42 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_38;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_122)) || (((W0 >> 4) < 0) && (rmod_122 <= 0))) ? rdiv_42 : (rdiv_42 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_123 = (((int)blockIdx.x) % (W0 >> 4));
        condval_38 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_123)) || (((W0 >> 4) < 0) && (rmod_123 <= 0))) ? rmod_123 : (rmod_123 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_124 = (((int)blockIdx.x) % (W0 >> 4));
        condval_38 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_124)) || (((W0 >> 4) < 0) && (rmod_124 <= 0))) ? rmod_124 : (rmod_124 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      int rmod_125 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_43 = (((int)blockIdx.x) / (W0 >> 4));
      int condval_39;
      if ((0 < (((((((0 <= (W0 >> 4)) && (0 <= rmod_125)) || (((W0 >> 4) < 0) && (rmod_125 <= 0))) ? rdiv_43 : (rdiv_43 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) & 1))) {
        int rmod_126 = (((int)blockIdx.x) % (W0 >> 4));
        condval_39 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_126)) || (((W0 >> 4) < 0) && (rmod_126 <= 0))) ? rmod_126 : (rmod_126 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) + 1);
      } else {
        int rmod_127 = (((int)blockIdx.x) % (W0 >> 4));
        condval_39 = ((((((((0 <= (W0 >> 4)) && (0 <= rmod_127)) || (((W0 >> 4) < 0) && (rmod_127 <= 0))) ? rmod_127 : (rmod_127 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) & 1) * 3);
      }
      condval_20 = ((((((((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_68)) || (((W0 >> 4) < 0) && (rmod_68 <= 0))) ? rdiv_24 : (rdiv_24 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2)))) >> 1) * 8) + (condval_21 * 2)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1)) >> 3) + ((H0 >> 4) * ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_22 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_23 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_24 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_25 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_26 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_27 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10))) * (W0 >> 3)) * 512) + ((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_90)) || (((W0 >> 4) < 0) && (rmod_90 <= 0))) ? rmod_90 : (rmod_90 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 5) * 512)) + (((W0 >> 6) * (((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_28 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_29 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_30 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_31 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_32 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_33 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + (((((((((((((0 <= (W0 >> 4)) && (0 <= rmod_109)) || (((W0 >> 4) < 0) && (rmod_109 <= 0))) ? rmod_109 : (rmod_109 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 4) == 3)))) >> 1) * 8) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_34 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_35 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_36 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 8)) ^ (((((((condval_37 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_38 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32)) ^ (((((((condval_39 & 3) * 16) + (((((m * 16) + (((int)threadIdx.x) >> 1)) >> 2) & 1) * 8)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) & 3) << 1)) + ((((m * 16) + (((int)threadIdx.x) >> 1)) >> 3) & 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127));
    } else {
      condval_20 = -1;
    }
    int base = max(condval, condval_20);
    int condval_40;
    if ((0 <= base)) {
      condval_40 = (in_offset + base);
    } else {
      condval_40 = -1;
    }
    int condval_41;
    if ((0 <= base)) {
      condval_41 = (in_offset + base);
    } else {
      condval_41 = -1;
    }
    int base_1 = max(condval_40, condval_41);
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      mp[j_2] = (uint)0;
    }
    if (0 <= base_1) {
      if (base_1 < sizes_6) {
        nr_tl_shallow_joint::ld128((&(mp[0])), (&(X[((int64_t)base_1)])));
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
        uint condval_42;
        if (((((int)threadIdx.x) % 4) == 0)) {
          condval_42 = s0;
        } else {
          uint condval_43;
          if (((((int)threadIdx.x) & 3) == 1)) {
            condval_43 = s1;
          } else {
            uint condval_44;
            if (((((int)threadIdx.x) & 3) == 2)) {
              condval_44 = s2;
            } else {
              condval_44 = s3;
            }
            condval_43 = condval_44;
          }
          condval_42 = condval_43;
        }
        uint word = condval_42;
        uint condval_45;
        if (((((int)threadIdx.x) % 4) == 0)) {
          condval_45 = s0;
        } else {
          uint condval_46;
          if (((((int)threadIdx.x) & 3) == 1)) {
            condval_46 = s1;
          } else {
            uint condval_47;
            if (((((int)threadIdx.x) & 3) == 2)) {
              condval_47 = s2;
            } else {
              condval_47 = s3;
            }
            condval_46 = condval_47;
          }
          condval_45 = condval_46;
        }
        canonical[0] = condval_45;
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          uint condval_48;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_48 = s0;
          } else {
            uint condval_49;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_49 = s1;
            } else {
              uint condval_50;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_50 = s2;
              } else {
                condval_50 = s3;
              }
              condval_49 = condval_50;
            }
            condval_48 = condval_49;
          }
          uint v = ((condval_48 >> ((uint)(byte * 8))) & (uint)255);
          uint condval_51;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_51 = s0;
          } else {
            uint condval_52;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_52 = s1;
            } else {
              uint condval_53;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_53 = s2;
              } else {
                condval_53 = s3;
              }
              condval_52 = condval_53;
            }
            condval_51 = condval_52;
          }
          if ((((condval_51 >> ((uint)(byte * 8))) & (uint)255) & (uint)127) == (uint)127) {
            uint condval_54;
            if (((((int)threadIdx.x) % 4) == 0)) {
              condval_54 = s0;
            } else {
              uint condval_55;
              if (((((int)threadIdx.x) & 3) == 1)) {
                condval_55 = s1;
              } else {
                uint condval_56;
                if (((((int)threadIdx.x) & 3) == 2)) {
                  condval_56 = s2;
                } else {
                  condval_56 = s3;
                }
                condval_55 = condval_56;
              }
              condval_54 = condval_55;
            }
            canonical[0] = ((canonical[0] & ((uint)4294967295 ^ ((uint)255 << ((uint)(byte * 8))))) | (nr_tl_shallow::e4(((half_t)nr_tl_shallow::une4(((condval_54 >> ((uint)(byte * 8))) & (uint)255)))) << ((uint)(byte * 8))));
          }
        }
        aa[(((m * 4) + (p * 2)) + i)] = canonical[0];
        #pragma unroll
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          uint condval_57;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_57 = s0;
          } else {
            uint condval_58;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_58 = s1;
            } else {
              uint condval_59;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_59 = s2;
              } else {
                condval_59 = s3;
              }
              condval_58 = condval_59;
            }
            condval_57 = condval_58;
          }
          uint condval_60;
          if (((((int)threadIdx.x) % 4) == 0)) {
            condval_60 = s0;
          } else {
            uint condval_61;
            if (((((int)threadIdx.x) & 3) == 1)) {
              condval_61 = s1;
            } else {
              uint condval_62;
              if (((((int)threadIdx.x) & 3) == 2)) {
                condval_62 = s2;
              } else {
                condval_62 = s3;
              }
              condval_61 = condval_62;
            }
            condval_60 = condval_61;
          }
          uint raw_1 = nr_tl_shallow::pack(((half_t)nr_tl_shallow::une4((condval_57 >> ((uint)(j_3 * 16))))), ((half_t)nr_tl_shallow::une4((condval_60 >> ((uint)((j_3 * 16) + 8))))));
          half_t condval_63;
          if (((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) < sizes_4)) {
            condval_63 = G0[(((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2))];
          } else {
            condval_63 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          half_t condval_64;
          if ((((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_4)) {
            condval_64 = G0[((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
          } else {
            condval_64 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          ff[((((m * 8) + (p * 4)) + (j_3 * 2)) + i)] = nr_tl_shallow::mul(raw_1, nr_tl_shallow::pack(condval_63, condval_64));
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
      if ((((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4)) < sizes_0) {
        nr_tl_shallow_joint::ld128((&(b[0])), (&(Wt0[(((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
      if ((((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4)) < sizes_1) {
        nr_tl_shallow_joint::ld128((&(b_1[0])), (&(Wt1[(((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
      if ((((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4)) < sizes_2) {
        nr_tl_shallow_joint::ld128((&(b_2[0])), (&(Wq[(((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
            int64_t condval_65;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_65 = (int64_t)0;
            } else {
              int64_t condval_66;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_66 = (int64_t)3;
              } else {
                condval_66 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_65 = condval_66;
            }
            int64_t condval_67;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_67 = (int64_t)0;
            } else {
              int64_t condval_68;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_68 = (int64_t)3;
              } else {
                condval_68 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_67 = condval_68;
            }
            int64_t condval_69;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_69 = (int64_t)0;
            } else {
              int64_t condval_70;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_70 = (int64_t)3;
              } else {
                condval_70 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_69 = condval_70;
            }
            int64_t condval_71;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_71 = (int64_t)0;
            } else {
              int64_t condval_72;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_72 = (int64_t)3;
              } else {
                condval_72 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_71 = condval_72;
            }
            qa[(((m_6 * 4) + (p_4 * 2)) + i_3)] = (nr_tl_shallow::e4pair(z[(((condval_67 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_71 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_73;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_73 = (int64_t)0;
            } else {
              int64_t condval_74;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_74 = (int64_t)3;
              } else {
                condval_74 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_73 = condval_74;
            }
            int64_t condval_75;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_75 = (int64_t)0;
            } else {
              int64_t condval_76;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_76 = (int64_t)3;
              } else {
                condval_76 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_75 = condval_76;
            }
            int64_t condval_77;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_77 = (int64_t)0;
            } else {
              int64_t condval_78;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_78 = (int64_t)3;
              } else {
                condval_78 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_77 = condval_78;
            }
            int64_t condval_79;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_79 = (int64_t)0;
            } else {
              int64_t condval_80;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_80 = (int64_t)3;
              } else {
                condval_80 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_79 = condval_80;
            }
            kb[(((m_6 * 4) + (i_3 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_75 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_79 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          int64_t condval_81;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_81 = (int64_t)0;
          } else {
            condval_81 = (int64_t)1;
          }
          int64_t condval_82;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_82 = (int64_t)0;
          } else {
            condval_82 = (int64_t)1;
          }
          int64_t condval_83;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_83 = (int64_t)0;
          } else {
            condval_83 = (int64_t)1;
          }
          int64_t condval_84;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_84 = (int64_t)0;
          } else {
            condval_84 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_3 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_82 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_84 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_85;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_85 = (int64_t)3;
          } else {
            condval_85 = (int64_t)2;
          }
          int64_t condval_86;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_86 = (int64_t)3;
          } else {
            condval_86 = (int64_t)2;
          }
          int64_t condval_87;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_87 = (int64_t)3;
          } else {
            condval_87 = (int64_t)2;
          }
          int64_t condval_88;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_88 = (int64_t)3;
          } else {
            condval_88 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_3 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_86 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_88 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
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
        if ((((((slab * 1024) + (m_7 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768) < sizes_2) {
          nr_tl_shallow_joint::ld128((&(seed[0])), (&(Wq[(((((slab * 1024) + (m_7 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768)])));
        }
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
      int64_t condval_89;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_89 = (int64_t)0;
      } else {
        int64_t condval_90;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_90 = (int64_t)3;
        } else {
          condval_90 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_89 = condval_90;
      }
      int64_t condval_91;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_91 = (int64_t)0;
      } else {
        int64_t condval_92;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_92 = (int64_t)3;
        } else {
          condval_92 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_91 = condval_92;
      }
      half_t condval_93;
      if ((((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_5)) {
        condval_93 = G1[((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
      } else {
        condval_93 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      half_t condval_94;
      if (((((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_5)) {
        condval_94 = G1[(((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
      } else {
        condval_94 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      out[j_10] = nr_tl_shallow::mul(ff[((condval_91 * (int64_t)8) + (((int64_t)j_10) & (int64_t)7))], nr_tl_shallow::pack(condval_93, condval_94));
    }
    #pragma unroll
    for (int p_8 = 0; p_8 < 2; ++p_8) {
      if (((p_8 * 128) + (((int)threadIdx.x) * 4)) < sizes_3) {
        nr_tl_shallow_joint::ld128((&(b_3[0])), (&(Wp[((p_8 * 128) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_11 = 0; m_11 < 2; ++m_11) {
        #pragma unroll
        for (int n_6 = 0; n_6 < 2; ++n_6) {
          nr_tl_shallow::mma8((&(out[(((m_11 * 8) + (p_8 * 4)) + (n_6 * 2))])), aa[(m_11 * 4)], aa[((m_11 * 4) + 1)], aa[((m_11 * 4) + 2)], aa[((m_11 * 4) + 3)], b_3[(n_6 * 2)], b_3[((n_6 * 2) + 1)]);
        }
      }
    }
    int H = max((H0 >> 1), (H0 >> 1));
    int W = max((W0 >> 1), (W0 >> 1));
    #pragma unroll
    for (int m_12 = 0; m_12 < 2; ++m_12) {
      int rmod_128 = (((int)blockIdx.x) % (W >> 3));
      int rdiv_44 = (((int)blockIdx.x) / (W >> 3));
      int rmod_129 = (((int)blockIdx.x) % (W >> 3));
      int rdiv_45 = (((int)blockIdx.x) / (W >> 3));
      int condval_96;
      if ((((slab * 2) + m_12) == 0)) {
        condval_96 = 0;
      } else {
        int condval_97;
        if ((((slab * 2) + m_12) == 1)) {
          condval_97 = 3;
        } else {
          condval_97 = (((slab * 2) + m_12) - 1);
        }
        condval_96 = condval_97;
      }
      int condval_98;
      if ((((slab * 2) + m_12) == 0)) {
        condval_98 = 0;
      } else {
        int condval_99;
        if ((((slab * 2) + m_12) == 1)) {
          condval_99 = 3;
        } else {
          condval_99 = (((slab * 2) + m_12) - 1);
        }
        condval_98 = condval_99;
      }
      int rmod_130 = (((int)blockIdx.x) % (W >> 3));
      int rmod_131 = (((int)blockIdx.x) % (W >> 3));
      int condval_100;
      if ((((slab * 2) + m_12) == 0)) {
        condval_100 = 0;
      } else {
        int condval_101;
        if ((((slab * 2) + m_12) == 1)) {
          condval_101 = 3;
        } else {
          condval_101 = (((slab * 2) + m_12) - 1);
        }
        condval_100 = condval_101;
      }
      int condval_102;
      if ((((slab * 2) + m_12) == 0)) {
        condval_102 = 0;
      } else {
        int condval_103;
        if ((((slab * 2) + m_12) == 1)) {
          condval_103 = 3;
        } else {
          condval_103 = (((slab * 2) + m_12) - 1);
        }
        condval_102 = condval_103;
      }
      int condval_95;
      if (((((0 <= ((((0 <= (W >> 3)) && (0 <= rmod_128)) || (((W >> 3) < 0) && (rmod_128 <= 0))) ? rdiv_44 : (rdiv_44 - 1))) & (((((((0 <= (W >> 3)) && (0 <= rmod_129)) || (((W >> 3) < 0) && (rmod_129 <= 0))) ? rdiv_45 : (rdiv_45 - 1)) * 2) + ((int)(((((condval_96 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_98 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H >> 2))) & (0 <= ((((0 <= (W >> 3)) && (0 <= rmod_130)) || (((W >> 3) < 0) && (rmod_130 <= 0))) ? rmod_130 : (rmod_130 + (W >> 3))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_131)) || (((W >> 3) < 0) && (rmod_131 <= 0))) ? rmod_131 : (rmod_131 + (W >> 3))) * 2) + ((int)(((((condval_100 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_102 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < (W >> 2)))) {
        int rmod_132 = (((int)blockIdx.x) % (W >> 3));
        int rmod_133 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_46 = (((int)blockIdx.x) / (W >> 3));
        int condval_104;
        if ((((slab * 2) + m_12) == 0)) {
          condval_104 = 0;
        } else {
          int condval_105;
          if ((((slab * 2) + m_12) == 1)) {
            condval_105 = 3;
          } else {
            condval_105 = (((slab * 2) + m_12) - 1);
          }
          condval_104 = condval_105;
        }
        int condval_106;
        if ((((slab * 2) + m_12) == 0)) {
          condval_106 = 0;
        } else {
          int condval_107;
          if ((((slab * 2) + m_12) == 1)) {
            condval_107 = 3;
          } else {
            condval_107 = (((slab * 2) + m_12) - 1);
          }
          condval_106 = condval_107;
        }
        int condval_108;
        if ((((slab * 2) + m_12) == 0)) {
          condval_108 = 0;
        } else {
          int condval_109;
          if ((((slab * 2) + m_12) == 1)) {
            condval_109 = 3;
          } else {
            condval_109 = (((slab * 2) + m_12) - 1);
          }
          condval_108 = condval_109;
        }
        int condval_110;
        if ((((slab * 2) + m_12) == 0)) {
          condval_110 = 0;
        } else {
          int condval_111;
          if ((((slab * 2) + m_12) == 1)) {
            condval_111 = 3;
          } else {
            condval_111 = (((slab * 2) + m_12) - 1);
          }
          condval_110 = condval_111;
        }
        int condval_112;
        if ((((slab * 2) + m_12) == 0)) {
          condval_112 = 0;
        } else {
          int condval_113;
          if ((((slab * 2) + m_12) == 1)) {
            condval_113 = 3;
          } else {
            condval_113 = (((slab * 2) + m_12) - 1);
          }
          condval_112 = condval_113;
        }
        int condval_114;
        if ((((slab * 2) + m_12) == 0)) {
          condval_114 = 0;
        } else {
          int condval_115;
          if ((((slab * 2) + m_12) == 1)) {
            condval_115 = 3;
          } else {
            condval_115 = (((slab * 2) + m_12) - 1);
          }
          condval_114 = condval_115;
        }
        condval_95 = (((((((((((0 <= (W >> 3)) && (0 <= rmod_132)) || (((W >> 3) < 0) && (rmod_132 <= 0))) ? rmod_132 : (rmod_132 + (W >> 3))) * 1024) + ((((((((0 <= (W >> 3)) && (0 <= rmod_133)) || (((W >> 3) < 0) && (rmod_133 <= 0))) ? rdiv_46 : (rdiv_46 - 1)) * 2) + ((int)(((((condval_104 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_106 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W >> 2)) * 512)) + (((int)(((((condval_108 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_110 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_112 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_114 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
      } else {
        condval_95 = -1;
      }
      int rmod_134 = (((int)blockIdx.x) % (W >> 3));
      int rdiv_47 = (((int)blockIdx.x) / (W >> 3));
      int rmod_135 = (((int)blockIdx.x) % (W >> 3));
      int rdiv_48 = (((int)blockIdx.x) / (W >> 3));
      int condval_117;
      if ((((slab * 2) + m_12) == 0)) {
        condval_117 = 0;
      } else {
        int condval_118;
        if ((((slab * 2) + m_12) == 1)) {
          condval_118 = 3;
        } else {
          condval_118 = (((slab * 2) + m_12) - 1);
        }
        condval_117 = condval_118;
      }
      int condval_119;
      if ((((slab * 2) + m_12) == 0)) {
        condval_119 = 0;
      } else {
        int condval_120;
        if ((((slab * 2) + m_12) == 1)) {
          condval_120 = 3;
        } else {
          condval_120 = (((slab * 2) + m_12) - 1);
        }
        condval_119 = condval_120;
      }
      int rmod_136 = (((int)blockIdx.x) % (W >> 3));
      int rmod_137 = (((int)blockIdx.x) % (W >> 3));
      int condval_121;
      if ((((slab * 2) + m_12) == 0)) {
        condval_121 = 0;
      } else {
        int condval_122;
        if ((((slab * 2) + m_12) == 1)) {
          condval_122 = 3;
        } else {
          condval_122 = (((slab * 2) + m_12) - 1);
        }
        condval_121 = condval_122;
      }
      int condval_123;
      if ((((slab * 2) + m_12) == 0)) {
        condval_123 = 0;
      } else {
        int condval_124;
        if ((((slab * 2) + m_12) == 1)) {
          condval_124 = 3;
        } else {
          condval_124 = (((slab * 2) + m_12) - 1);
        }
        condval_123 = condval_124;
      }
      int condval_116;
      if (((((0 <= ((((0 <= (W >> 3)) && (0 <= rmod_134)) || (((W >> 3) < 0) && (rmod_134 <= 0))) ? rdiv_47 : (rdiv_47 - 1))) & (((((((0 <= (W >> 3)) && (0 <= rmod_135)) || (((W >> 3) < 0) && (rmod_135 <= 0))) ? rdiv_48 : (rdiv_48 - 1)) * 2) + ((int)(((((condval_117 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_119 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H >> 2))) & (0 <= ((((0 <= (W >> 3)) && (0 <= rmod_136)) || (((W >> 3) < 0) && (rmod_136 <= 0))) ? rmod_136 : (rmod_136 + (W >> 3))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_137)) || (((W >> 3) < 0) && (rmod_137 <= 0))) ? rmod_137 : (rmod_137 + (W >> 3))) * 2) + ((int)(((((condval_121 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_123 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < (W >> 2)))) {
        int rmod_138 = (((int)blockIdx.x) % (W >> 3));
        int rmod_139 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_49 = (((int)blockIdx.x) / (W >> 3));
        int condval_125;
        if ((((slab * 2) + m_12) == 0)) {
          condval_125 = 0;
        } else {
          int condval_126;
          if ((((slab * 2) + m_12) == 1)) {
            condval_126 = 3;
          } else {
            condval_126 = (((slab * 2) + m_12) - 1);
          }
          condval_125 = condval_126;
        }
        int condval_127;
        if ((((slab * 2) + m_12) == 0)) {
          condval_127 = 0;
        } else {
          int condval_128;
          if ((((slab * 2) + m_12) == 1)) {
            condval_128 = 3;
          } else {
            condval_128 = (((slab * 2) + m_12) - 1);
          }
          condval_127 = condval_128;
        }
        int condval_129;
        if ((((slab * 2) + m_12) == 0)) {
          condval_129 = 0;
        } else {
          int condval_130;
          if ((((slab * 2) + m_12) == 1)) {
            condval_130 = 3;
          } else {
            condval_130 = (((slab * 2) + m_12) - 1);
          }
          condval_129 = condval_130;
        }
        int condval_131;
        if ((((slab * 2) + m_12) == 0)) {
          condval_131 = 0;
        } else {
          int condval_132;
          if ((((slab * 2) + m_12) == 1)) {
            condval_132 = 3;
          } else {
            condval_132 = (((slab * 2) + m_12) - 1);
          }
          condval_131 = condval_132;
        }
        int condval_133;
        if ((((slab * 2) + m_12) == 0)) {
          condval_133 = 0;
        } else {
          int condval_134;
          if ((((slab * 2) + m_12) == 1)) {
            condval_134 = 3;
          } else {
            condval_134 = (((slab * 2) + m_12) - 1);
          }
          condval_133 = condval_134;
        }
        int condval_135;
        if ((((slab * 2) + m_12) == 0)) {
          condval_135 = 0;
        } else {
          int condval_136;
          if ((((slab * 2) + m_12) == 1)) {
            condval_136 = 3;
          } else {
            condval_136 = (((slab * 2) + m_12) - 1);
          }
          condval_135 = condval_136;
        }
        condval_116 = (((((((((((0 <= (W >> 3)) && (0 <= rmod_138)) || (((W >> 3) < 0) && (rmod_138 <= 0))) ? rmod_138 : (rmod_138 + (W >> 3))) * 1024) + ((((((((0 <= (W >> 3)) && (0 <= rmod_139)) || (((W >> 3) < 0) && (rmod_139 <= 0))) ? rdiv_49 : (rdiv_49 - 1)) * 2) + ((int)(((((condval_125 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_127 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W >> 2)) * 512)) + (((int)(((((condval_129 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_131 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_133 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_135 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
      } else {
        condval_116 = -1;
      }
      int dest = max(condval_95, condval_116);
      if (0 <= dest) {
        if (0 <= (out_offset + dest)) {
          if ((out_offset + dest) < sizes_6) {
            nr_tl_shallow_joint::st128((&(X[(((int64_t)out_offset) + ((int64_t)dest))])), (nr_tl_shallow::e4pair(out[(m_12 * 8)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 2)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 1)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 3)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 4)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 6)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 5)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 7)]) << (uint)16)));
          }
        }
      }
    }
  }
  if (0 <= counter) {
    if (((int)threadIdx.x) == 0) {
      if (((((int)blockIdx.x) * 4) + counter) < sizes_6) {
        nr_tl_shallow_joint::st32((&(X[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + ((int64_t)counter))])), (uint)0);
      }
    }
  }
}

