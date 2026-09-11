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
  uint q[4];
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
    if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
      condval_1 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
    } else {
      condval_1 = -1;
    }
    int condval;
    if ((0 <= condval_1)) {
      int condval_2;
      if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
        condval_2 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
      } else {
        condval_2 = -1;
      }
      condval = (condval_2 + 54886400);
    } else {
      condval = -1;
    }
    if (0 <= condval) {
      int condval_4;
      if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
        condval_4 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
      } else {
        condval_4 = -1;
      }
      int condval_3;
      if ((0 <= condval_4)) {
        int condval_5;
        if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
          condval_5 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
        } else {
          condval_5 = -1;
        }
        condval_3 = (condval_5 + 54886400);
      } else {
        condval_3 = -1;
      }
      if (condval_3 < 60358048) {
        int condval_7;
        if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
          condval_7 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
        } else {
          condval_7 = -1;
        }
        int condval_6;
        if ((0 <= condval_7)) {
          int condval_8;
          if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
            condval_8 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
          } else {
            condval_8 = -1;
          }
          condval_6 = (condval_8 + 54886400);
        } else {
          condval_6 = -1;
        }
        int condval_10;
        if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
          condval_10 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
        } else {
          condval_10 = -1;
        }
        int condval_9;
        if ((0 <= condval_10)) {
          int condval_11;
          if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
            condval_11 = ((((((((((((int)blockIdx.x) / 40) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 40) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 40960);
          } else {
            condval_11 = -1;
          }
          condval_9 = (condval_11 + 54886400);
        } else {
          condval_9 = -1;
        }
        nr_tl_shallow_joint::ld128((&(mp[0])), (&(X[condval_9])));
      }
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint word = mp[((p * 2) + i)];
        canonical[0] = word;
        #pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          uint v = ((word >> ((uint)(byte * 8))) & (uint)255);
          if ((((word >> ((uint)(byte * 8))) & (uint)255) & (uint)127) == (uint)127) {
            canonical[0] = ((canonical[0] & ((uint)4294967295 ^ ((uint)255 << ((uint)(byte * 8))))) | (nr_tl_shallow::e4(((half_t)nr_tl_shallow::une4(((word >> ((uint)(byte * 8))) & (uint)255)))) << ((uint)(byte * 8))));
          }
        }
        aa[(((m * 4) + (p * 2)) + i)] = canonical[0];
        #pragma unroll
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          uint raw_1 = nr_tl_shallow::pack(((half_t)nr_tl_shallow::une4((word >> ((uint)(j_3 * 16))))), ((half_t)nr_tl_shallow::une4((word >> ((uint)((j_3 * 16) + 8))))));
          ff[((((m * 8) + (p * 4)) + (j_3 * 2)) + i)] = nr_tl_shallow::mul(raw_1, nr_tl_shallow::pack(G0[(((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2))], G0[((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
        }
      }
    }
  }
  ushort v_ = (ushort)18829;
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
          ushort v__1 = (ushort)18829;
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
            int64_t condval_12;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_12 = (int64_t)0;
            } else {
              int64_t condval_13;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_13 = (int64_t)3;
              } else {
                condval_13 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_12 = condval_13;
            }
            int64_t condval_14;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_14 = (int64_t)0;
            } else {
              int64_t condval_15;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_15 = (int64_t)3;
              } else {
                condval_15 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_14 = condval_15;
            }
            int64_t condval_16;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_16 = (int64_t)0;
            } else {
              int64_t condval_17;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_17 = (int64_t)3;
              } else {
                condval_17 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_16 = condval_17;
            }
            int64_t condval_18;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_18 = (int64_t)0;
            } else {
              int64_t condval_19;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_19 = (int64_t)3;
              } else {
                condval_19 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_18 = condval_19;
            }
            qa[(((m_6 * 4) + (p_4 * 2)) + i_3)] = (nr_tl_shallow::e4pair(z[(((condval_14 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_18 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_20;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_20 = (int64_t)0;
            } else {
              int64_t condval_21;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_21 = (int64_t)3;
              } else {
                condval_21 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_20 = condval_21;
            }
            int64_t condval_22;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_22 = (int64_t)0;
            } else {
              int64_t condval_23;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_23 = (int64_t)3;
              } else {
                condval_23 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_22 = condval_23;
            }
            int64_t condval_24;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_24 = (int64_t)0;
            } else {
              int64_t condval_25;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_25 = (int64_t)3;
              } else {
                condval_25 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_24 = condval_25;
            }
            int64_t condval_26;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_26 = (int64_t)0;
            } else {
              int64_t condval_27;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_27 = (int64_t)3;
              } else {
                condval_27 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_26 = condval_27;
            }
            kb[(((m_6 * 4) + (i_3 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_22 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_26 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          int64_t condval_28;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_28 = (int64_t)0;
          } else {
            condval_28 = (int64_t)1;
          }
          int64_t condval_29;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_29 = (int64_t)0;
          } else {
            condval_29 = (int64_t)1;
          }
          int64_t condval_30;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_30 = (int64_t)0;
          } else {
            condval_30 = (int64_t)1;
          }
          int64_t condval_31;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_31 = (int64_t)0;
          } else {
            condval_31 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_3 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_29 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_31 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_32;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_32 = (int64_t)3;
          } else {
            condval_32 = (int64_t)2;
          }
          int64_t condval_33;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_33 = (int64_t)3;
          } else {
            condval_33 = (int64_t)2;
          }
          int64_t condval_34;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_34 = (int64_t)3;
          } else {
            condval_34 = (int64_t)2;
          }
          int64_t condval_35;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_35 = (int64_t)3;
          } else {
            condval_35 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_3 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_33 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_35 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
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
      uint s0 = nr_tl_shallow::add(logits[(((row_1 >> 1) * 16) + (row_1 & 1))], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 2)]);
      uint s1 = nr_tl_shallow::add(s0, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 4)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 6)]));
      uint s2 = nr_tl_shallow::add(s1, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 8)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 10)]));
      uint s_1 = nr_tl_shallow::add(s2, nr_tl_shallow::add(logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 12)], logits[((((row_1 >> 1) * 16) + (row_1 & 1)) + 14)]));
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
      int64_t condval_36;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_36 = (int64_t)0;
      } else {
        int64_t condval_37;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_37 = (int64_t)3;
        } else {
          condval_37 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_36 = condval_37;
      }
      int64_t condval_38;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_38 = (int64_t)0;
      } else {
        int64_t condval_39;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_39 = (int64_t)3;
        } else {
          condval_39 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_38 = condval_39;
      }
      out[j_10] = nr_tl_shallow::mul(ff[((condval_38 * (int64_t)8) + (((int64_t)j_10) & (int64_t)7))], nr_tl_shallow::pack(G1[((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))], G1[(((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
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
      #pragma unroll
      for (int p_9 = 0; p_9 < 2; ++p_9) {
        uint lo = (nr_tl_shallow::e4pair(out[((m_12 * 8) + (p_9 * 4))]) | (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 2)]) << (uint)16));
        uint hi = (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 1)]) | (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 3)]) << (uint)16));
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint l = nr_tl_shallow::shfl(lo, (((((int)threadIdx.x) & 7) * 4) + k));
          uint h_2 = nr_tl_shallow::shfl(hi, (((((int)threadIdx.x) & 7) * 4) + k));
          uint condval_40;
          if ((((int)threadIdx.x) < 8)) {
            condval_40 = l;
          } else {
            condval_40 = h_2;
          }
          q[k] = condval_40;
        }
        int condval_42;
        if ((((slab * 2) + m_12) == 0)) {
          condval_42 = 0;
        } else {
          int condval_43;
          if ((((slab * 2) + m_12) == 1)) {
            condval_43 = 3;
          } else {
            condval_43 = (((slab * 2) + m_12) - 1);
          }
          condval_42 = condval_43;
        }
        int condval_44;
        if ((((slab * 2) + m_12) == 0)) {
          condval_44 = 0;
        } else {
          int condval_45;
          if ((((slab * 2) + m_12) == 1)) {
            condval_45 = 3;
          } else {
            condval_45 = (((slab * 2) + m_12) - 1);
          }
          condval_44 = condval_45;
        }
        int condval_46;
        if ((((slab * 2) + m_12) == 0)) {
          condval_46 = 0;
        } else {
          int condval_47;
          if ((((slab * 2) + m_12) == 1)) {
            condval_47 = 3;
          } else {
            condval_47 = (((slab * 2) + m_12) - 1);
          }
          condval_46 = condval_47;
        }
        int condval_48;
        if ((((slab * 2) + m_12) == 0)) {
          condval_48 = 0;
        } else {
          int condval_49;
          if ((((slab * 2) + m_12) == 1)) {
            condval_49 = 3;
          } else {
            condval_49 = (((slab * 2) + m_12) - 1);
          }
          condval_48 = condval_49;
        }
        int condval_41;
        if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_42 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_44 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_46 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_48 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
          int condval_51;
          if ((((slab * 2) + m_12) == 0)) {
            condval_51 = 0;
          } else {
            int condval_52;
            if ((((slab * 2) + m_12) == 1)) {
              condval_52 = 3;
            } else {
              condval_52 = (((slab * 2) + m_12) - 1);
            }
            condval_51 = condval_52;
          }
          int condval_53;
          if ((((slab * 2) + m_12) == 0)) {
            condval_53 = 0;
          } else {
            int condval_54;
            if ((((slab * 2) + m_12) == 1)) {
              condval_54 = 3;
            } else {
              condval_54 = (((slab * 2) + m_12) - 1);
            }
            condval_53 = condval_54;
          }
          int condval_50;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_51 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_53 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_55;
            if ((((slab * 2) + m_12) == 0)) {
              condval_55 = 0;
            } else {
              int condval_56;
              if ((((slab * 2) + m_12) == 1)) {
                condval_56 = 3;
              } else {
                condval_56 = (((slab * 2) + m_12) - 1);
              }
              condval_55 = condval_56;
            }
            int condval_57;
            if ((((slab * 2) + m_12) == 0)) {
              condval_57 = 0;
            } else {
              int condval_58;
              if ((((slab * 2) + m_12) == 1)) {
                condval_58 = 3;
              } else {
                condval_58 = (((slab * 2) + m_12) - 1);
              }
              condval_57 = condval_58;
            }
            condval_50 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_55 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_57 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_59;
            if ((((slab * 2) + m_12) == 0)) {
              condval_59 = 0;
            } else {
              int condval_60;
              if ((((slab * 2) + m_12) == 1)) {
                condval_60 = 3;
              } else {
                condval_60 = (((slab * 2) + m_12) - 1);
              }
              condval_59 = condval_60;
            }
            int condval_61;
            if ((((slab * 2) + m_12) == 0)) {
              condval_61 = 0;
            } else {
              int condval_62;
              if ((((slab * 2) + m_12) == 1)) {
                condval_62 = 3;
              } else {
                condval_62 = (((slab * 2) + m_12) - 1);
              }
              condval_61 = condval_62;
            }
            condval_50 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_59 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_61 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_63;
          if ((((slab * 2) + m_12) == 0)) {
            condval_63 = 0;
          } else {
            int condval_64;
            if ((((slab * 2) + m_12) == 1)) {
              condval_64 = 3;
            } else {
              condval_64 = (((slab * 2) + m_12) - 1);
            }
            condval_63 = condval_64;
          }
          int condval_65;
          if ((((slab * 2) + m_12) == 0)) {
            condval_65 = 0;
          } else {
            int condval_66;
            if ((((slab * 2) + m_12) == 1)) {
              condval_66 = 3;
            } else {
              condval_66 = (((slab * 2) + m_12) - 1);
            }
            condval_65 = condval_66;
          }
          int condval_67;
          if ((((slab * 2) + m_12) == 0)) {
            condval_67 = 0;
          } else {
            int condval_68;
            if ((((slab * 2) + m_12) == 1)) {
              condval_68 = 3;
            } else {
              condval_68 = (((slab * 2) + m_12) - 1);
            }
            condval_67 = condval_68;
          }
          int condval_70;
          if ((((slab * 2) + m_12) == 0)) {
            condval_70 = 0;
          } else {
            int condval_71;
            if ((((slab * 2) + m_12) == 1)) {
              condval_71 = 3;
            } else {
              condval_71 = (((slab * 2) + m_12) - 1);
            }
            condval_70 = condval_71;
          }
          int condval_72;
          if ((((slab * 2) + m_12) == 0)) {
            condval_72 = 0;
          } else {
            int condval_73;
            if ((((slab * 2) + m_12) == 1)) {
              condval_73 = 3;
            } else {
              condval_73 = (((slab * 2) + m_12) - 1);
            }
            condval_72 = condval_73;
          }
          int condval_69;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_70 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_72 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_74;
            if ((((slab * 2) + m_12) == 0)) {
              condval_74 = 0;
            } else {
              int condval_75;
              if ((((slab * 2) + m_12) == 1)) {
                condval_75 = 3;
              } else {
                condval_75 = (((slab * 2) + m_12) - 1);
              }
              condval_74 = condval_75;
            }
            int condval_76;
            if ((((slab * 2) + m_12) == 0)) {
              condval_76 = 0;
            } else {
              int condval_77;
              if ((((slab * 2) + m_12) == 1)) {
                condval_77 = 3;
              } else {
                condval_77 = (((slab * 2) + m_12) - 1);
              }
              condval_76 = condval_77;
            }
            condval_69 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_74 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_76 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_78;
            if ((((slab * 2) + m_12) == 0)) {
              condval_78 = 0;
            } else {
              int condval_79;
              if ((((slab * 2) + m_12) == 1)) {
                condval_79 = 3;
              } else {
                condval_79 = (((slab * 2) + m_12) - 1);
              }
              condval_78 = condval_79;
            }
            int condval_80;
            if ((((slab * 2) + m_12) == 0)) {
              condval_80 = 0;
            } else {
              int condval_81;
              if ((((slab * 2) + m_12) == 1)) {
                condval_81 = 3;
              } else {
                condval_81 = (((slab * 2) + m_12) - 1);
              }
              condval_80 = condval_81;
            }
            condval_69 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_78 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_80 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_82;
          if ((((slab * 2) + m_12) == 0)) {
            condval_82 = 0;
          } else {
            int condval_83;
            if ((((slab * 2) + m_12) == 1)) {
              condval_83 = 3;
            } else {
              condval_83 = (((slab * 2) + m_12) - 1);
            }
            condval_82 = condval_83;
          }
          int condval_84;
          if ((((slab * 2) + m_12) == 0)) {
            condval_84 = 0;
          } else {
            int condval_85;
            if ((((slab * 2) + m_12) == 1)) {
              condval_85 = 3;
            } else {
              condval_85 = (((slab * 2) + m_12) - 1);
            }
            condval_84 = condval_85;
          }
          int condval_86;
          if ((((slab * 2) + m_12) == 0)) {
            condval_86 = 0;
          } else {
            int condval_87;
            if ((((slab * 2) + m_12) == 1)) {
              condval_87 = 3;
            } else {
              condval_87 = (((slab * 2) + m_12) - 1);
            }
            condval_86 = condval_87;
          }
          int condval_89;
          if ((((slab * 2) + m_12) == 0)) {
            condval_89 = 0;
          } else {
            int condval_90;
            if ((((slab * 2) + m_12) == 1)) {
              condval_90 = 3;
            } else {
              condval_90 = (((slab * 2) + m_12) - 1);
            }
            condval_89 = condval_90;
          }
          int condval_91;
          if ((((slab * 2) + m_12) == 0)) {
            condval_91 = 0;
          } else {
            int condval_92;
            if ((((slab * 2) + m_12) == 1)) {
              condval_92 = 3;
            } else {
              condval_92 = (((slab * 2) + m_12) - 1);
            }
            condval_91 = condval_92;
          }
          int condval_88;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_89 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_91 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_93;
            if ((((slab * 2) + m_12) == 0)) {
              condval_93 = 0;
            } else {
              int condval_94;
              if ((((slab * 2) + m_12) == 1)) {
                condval_94 = 3;
              } else {
                condval_94 = (((slab * 2) + m_12) - 1);
              }
              condval_93 = condval_94;
            }
            int condval_95;
            if ((((slab * 2) + m_12) == 0)) {
              condval_95 = 0;
            } else {
              int condval_96;
              if ((((slab * 2) + m_12) == 1)) {
                condval_96 = 3;
              } else {
                condval_96 = (((slab * 2) + m_12) - 1);
              }
              condval_95 = condval_96;
            }
            condval_88 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_93 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_95 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_97;
            if ((((slab * 2) + m_12) == 0)) {
              condval_97 = 0;
            } else {
              int condval_98;
              if ((((slab * 2) + m_12) == 1)) {
                condval_98 = 3;
              } else {
                condval_98 = (((slab * 2) + m_12) - 1);
              }
              condval_97 = condval_98;
            }
            int condval_99;
            if ((((slab * 2) + m_12) == 0)) {
              condval_99 = 0;
            } else {
              int condval_100;
              if ((((slab * 2) + m_12) == 1)) {
                condval_100 = 3;
              } else {
                condval_100 = (((slab * 2) + m_12) - 1);
              }
              condval_99 = condval_100;
            }
            condval_88 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_97 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_99 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_101;
          if ((((slab * 2) + m_12) == 0)) {
            condval_101 = 0;
          } else {
            int condval_102;
            if ((((slab * 2) + m_12) == 1)) {
              condval_102 = 3;
            } else {
              condval_102 = (((slab * 2) + m_12) - 1);
            }
            condval_101 = condval_102;
          }
          int condval_103;
          if ((((slab * 2) + m_12) == 0)) {
            condval_103 = 0;
          } else {
            int condval_104;
            if ((((slab * 2) + m_12) == 1)) {
              condval_104 = 3;
            } else {
              condval_104 = (((slab * 2) + m_12) - 1);
            }
            condval_103 = condval_104;
          }
          int condval_105;
          if ((((slab * 2) + m_12) == 0)) {
            condval_105 = 0;
          } else {
            int condval_106;
            if ((((slab * 2) + m_12) == 1)) {
              condval_106 = 3;
            } else {
              condval_106 = (((slab * 2) + m_12) - 1);
            }
            condval_105 = condval_106;
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
          int condval_107;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_108 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_110 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
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
            condval_107 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_112 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_114 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_116;
            if ((((slab * 2) + m_12) == 0)) {
              condval_116 = 0;
            } else {
              int condval_117;
              if ((((slab * 2) + m_12) == 1)) {
                condval_117 = 3;
              } else {
                condval_117 = (((slab * 2) + m_12) - 1);
              }
              condval_116 = condval_117;
            }
            int condval_118;
            if ((((slab * 2) + m_12) == 0)) {
              condval_118 = 0;
            } else {
              int condval_119;
              if ((((slab * 2) + m_12) == 1)) {
                condval_119 = 3;
              } else {
                condval_119 = (((slab * 2) + m_12) - 1);
              }
              condval_118 = condval_119;
            }
            condval_107 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_116 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_118 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_120;
          if ((((slab * 2) + m_12) == 0)) {
            condval_120 = 0;
          } else {
            int condval_121;
            if ((((slab * 2) + m_12) == 1)) {
              condval_121 = 3;
            } else {
              condval_121 = (((slab * 2) + m_12) - 1);
            }
            condval_120 = condval_121;
          }
          int condval_122;
          if ((((slab * 2) + m_12) == 0)) {
            condval_122 = 0;
          } else {
            int condval_123;
            if ((((slab * 2) + m_12) == 1)) {
              condval_123 = 3;
            } else {
              condval_123 = (((slab * 2) + m_12) - 1);
            }
            condval_122 = condval_123;
          }
          int condval_124;
          if ((((slab * 2) + m_12) == 0)) {
            condval_124 = 0;
          } else {
            int condval_125;
            if ((((slab * 2) + m_12) == 1)) {
              condval_125 = 3;
            } else {
              condval_125 = (((slab * 2) + m_12) - 1);
            }
            condval_124 = condval_125;
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
          int condval_126;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_127 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_129 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
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
            condval_126 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_131 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_133 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
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
            int condval_137;
            if ((((slab * 2) + m_12) == 0)) {
              condval_137 = 0;
            } else {
              int condval_138;
              if ((((slab * 2) + m_12) == 1)) {
                condval_138 = 3;
              } else {
                condval_138 = (((slab * 2) + m_12) - 1);
              }
              condval_137 = condval_138;
            }
            condval_126 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_135 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_137 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_139;
          if ((((slab * 2) + m_12) == 0)) {
            condval_139 = 0;
          } else {
            int condval_140;
            if ((((slab * 2) + m_12) == 1)) {
              condval_140 = 3;
            } else {
              condval_140 = (((slab * 2) + m_12) - 1);
            }
            condval_139 = condval_140;
          }
          int condval_141;
          if ((((slab * 2) + m_12) == 0)) {
            condval_141 = 0;
          } else {
            int condval_142;
            if ((((slab * 2) + m_12) == 1)) {
              condval_142 = 3;
            } else {
              condval_142 = (((slab * 2) + m_12) - 1);
            }
            condval_141 = condval_142;
          }
          int condval_143;
          if ((((slab * 2) + m_12) == 0)) {
            condval_143 = 0;
          } else {
            int condval_144;
            if ((((slab * 2) + m_12) == 1)) {
              condval_144 = 3;
            } else {
              condval_144 = (((slab * 2) + m_12) - 1);
            }
            condval_143 = condval_144;
          }
          int condval_146;
          if ((((slab * 2) + m_12) == 0)) {
            condval_146 = 0;
          } else {
            int condval_147;
            if ((((slab * 2) + m_12) == 1)) {
              condval_147 = 3;
            } else {
              condval_147 = (((slab * 2) + m_12) - 1);
            }
            condval_146 = condval_147;
          }
          int condval_148;
          if ((((slab * 2) + m_12) == 0)) {
            condval_148 = 0;
          } else {
            int condval_149;
            if ((((slab * 2) + m_12) == 1)) {
              condval_149 = 3;
            } else {
              condval_149 = (((slab * 2) + m_12) - 1);
            }
            condval_148 = condval_149;
          }
          int condval_145;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_146 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_148 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_150;
            if ((((slab * 2) + m_12) == 0)) {
              condval_150 = 0;
            } else {
              int condval_151;
              if ((((slab * 2) + m_12) == 1)) {
                condval_151 = 3;
              } else {
                condval_151 = (((slab * 2) + m_12) - 1);
              }
              condval_150 = condval_151;
            }
            int condval_152;
            if ((((slab * 2) + m_12) == 0)) {
              condval_152 = 0;
            } else {
              int condval_153;
              if ((((slab * 2) + m_12) == 1)) {
                condval_153 = 3;
              } else {
                condval_153 = (((slab * 2) + m_12) - 1);
              }
              condval_152 = condval_153;
            }
            condval_145 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_150 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_152 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_154;
            if ((((slab * 2) + m_12) == 0)) {
              condval_154 = 0;
            } else {
              int condval_155;
              if ((((slab * 2) + m_12) == 1)) {
                condval_155 = 3;
              } else {
                condval_155 = (((slab * 2) + m_12) - 1);
              }
              condval_154 = condval_155;
            }
            int condval_156;
            if ((((slab * 2) + m_12) == 0)) {
              condval_156 = 0;
            } else {
              int condval_157;
              if ((((slab * 2) + m_12) == 1)) {
                condval_157 = 3;
              } else {
                condval_157 = (((slab * 2) + m_12) - 1);
              }
              condval_156 = condval_157;
            }
            condval_145 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_154 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_156 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_158;
          if ((((slab * 2) + m_12) == 0)) {
            condval_158 = 0;
          } else {
            int condval_159;
            if ((((slab * 2) + m_12) == 1)) {
              condval_159 = 3;
            } else {
              condval_159 = (((slab * 2) + m_12) - 1);
            }
            condval_158 = condval_159;
          }
          int condval_160;
          if ((((slab * 2) + m_12) == 0)) {
            condval_160 = 0;
          } else {
            int condval_161;
            if ((((slab * 2) + m_12) == 1)) {
              condval_161 = 3;
            } else {
              condval_161 = (((slab * 2) + m_12) - 1);
            }
            condval_160 = condval_161;
          }
          int condval_162;
          if ((((slab * 2) + m_12) == 0)) {
            condval_162 = 0;
          } else {
            int condval_163;
            if ((((slab * 2) + m_12) == 1)) {
              condval_163 = 3;
            } else {
              condval_163 = (((slab * 2) + m_12) - 1);
            }
            condval_162 = condval_163;
          }
          int condval_164;
          if ((((slab * 2) + m_12) == 0)) {
            condval_164 = 0;
          } else {
            int condval_165;
            if ((((slab * 2) + m_12) == 1)) {
              condval_165 = 3;
            } else {
              condval_165 = (((slab * 2) + m_12) - 1);
            }
            condval_164 = condval_165;
          }
          int condval_166;
          if ((((slab * 2) + m_12) == 0)) {
            condval_166 = 0;
          } else {
            int condval_167;
            if ((((slab * 2) + m_12) == 1)) {
              condval_167 = 3;
            } else {
              condval_167 = (((slab * 2) + m_12) - 1);
            }
            condval_166 = condval_167;
          }
          int condval_169;
          if ((((slab * 2) + m_12) == 0)) {
            condval_169 = 0;
          } else {
            int condval_170;
            if ((((slab * 2) + m_12) == 1)) {
              condval_170 = 3;
            } else {
              condval_170 = (((slab * 2) + m_12) - 1);
            }
            condval_169 = condval_170;
          }
          int condval_171;
          if ((((slab * 2) + m_12) == 0)) {
            condval_171 = 0;
          } else {
            int condval_172;
            if ((((slab * 2) + m_12) == 1)) {
              condval_172 = 3;
            } else {
              condval_172 = (((slab * 2) + m_12) - 1);
            }
            condval_171 = condval_172;
          }
          int condval_168;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_169 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_171 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_173;
            if ((((slab * 2) + m_12) == 0)) {
              condval_173 = 0;
            } else {
              int condval_174;
              if ((((slab * 2) + m_12) == 1)) {
                condval_174 = 3;
              } else {
                condval_174 = (((slab * 2) + m_12) - 1);
              }
              condval_173 = condval_174;
            }
            int condval_175;
            if ((((slab * 2) + m_12) == 0)) {
              condval_175 = 0;
            } else {
              int condval_176;
              if ((((slab * 2) + m_12) == 1)) {
                condval_176 = 3;
              } else {
                condval_176 = (((slab * 2) + m_12) - 1);
              }
              condval_175 = condval_176;
            }
            condval_168 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_173 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_175 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_177;
            if ((((slab * 2) + m_12) == 0)) {
              condval_177 = 0;
            } else {
              int condval_178;
              if ((((slab * 2) + m_12) == 1)) {
                condval_178 = 3;
              } else {
                condval_178 = (((slab * 2) + m_12) - 1);
              }
              condval_177 = condval_178;
            }
            int condval_179;
            if ((((slab * 2) + m_12) == 0)) {
              condval_179 = 0;
            } else {
              int condval_180;
              if ((((slab * 2) + m_12) == 1)) {
                condval_180 = 3;
              } else {
                condval_180 = (((slab * 2) + m_12) - 1);
              }
              condval_179 = condval_180;
            }
            condval_168 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_177 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_179 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_181;
          if ((((slab * 2) + m_12) == 0)) {
            condval_181 = 0;
          } else {
            int condval_182;
            if ((((slab * 2) + m_12) == 1)) {
              condval_182 = 3;
            } else {
              condval_182 = (((slab * 2) + m_12) - 1);
            }
            condval_181 = condval_182;
          }
          int condval_184;
          if ((((slab * 2) + m_12) == 0)) {
            condval_184 = 0;
          } else {
            int condval_185;
            if ((((slab * 2) + m_12) == 1)) {
              condval_185 = 3;
            } else {
              condval_185 = (((slab * 2) + m_12) - 1);
            }
            condval_184 = condval_185;
          }
          int condval_186;
          if ((((slab * 2) + m_12) == 0)) {
            condval_186 = 0;
          } else {
            int condval_187;
            if ((((slab * 2) + m_12) == 1)) {
              condval_187 = 3;
            } else {
              condval_187 = (((slab * 2) + m_12) - 1);
            }
            condval_186 = condval_187;
          }
          int condval_183;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_184 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_186 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_188;
            if ((((slab * 2) + m_12) == 0)) {
              condval_188 = 0;
            } else {
              int condval_189;
              if ((((slab * 2) + m_12) == 1)) {
                condval_189 = 3;
              } else {
                condval_189 = (((slab * 2) + m_12) - 1);
              }
              condval_188 = condval_189;
            }
            int condval_190;
            if ((((slab * 2) + m_12) == 0)) {
              condval_190 = 0;
            } else {
              int condval_191;
              if ((((slab * 2) + m_12) == 1)) {
                condval_191 = 3;
              } else {
                condval_191 = (((slab * 2) + m_12) - 1);
              }
              condval_190 = condval_191;
            }
            condval_183 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_188 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_190 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_192;
            if ((((slab * 2) + m_12) == 0)) {
              condval_192 = 0;
            } else {
              int condval_193;
              if ((((slab * 2) + m_12) == 1)) {
                condval_193 = 3;
              } else {
                condval_193 = (((slab * 2) + m_12) - 1);
              }
              condval_192 = condval_193;
            }
            int condval_194;
            if ((((slab * 2) + m_12) == 0)) {
              condval_194 = 0;
            } else {
              int condval_195;
              if ((((slab * 2) + m_12) == 1)) {
                condval_195 = 3;
              } else {
                condval_195 = (((slab * 2) + m_12) - 1);
              }
              condval_194 = condval_195;
            }
            condval_183 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_192 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_194 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_196;
          if ((((slab * 2) + m_12) == 0)) {
            condval_196 = 0;
          } else {
            int condval_197;
            if ((((slab * 2) + m_12) == 1)) {
              condval_197 = 3;
            } else {
              condval_197 = (((slab * 2) + m_12) - 1);
            }
            condval_196 = condval_197;
          }
          int condval_198;
          if ((((slab * 2) + m_12) == 0)) {
            condval_198 = 0;
          } else {
            int condval_199;
            if ((((slab * 2) + m_12) == 1)) {
              condval_199 = 3;
            } else {
              condval_199 = (((slab * 2) + m_12) - 1);
            }
            condval_198 = condval_199;
          }
          int condval_200;
          if ((((slab * 2) + m_12) == 0)) {
            condval_200 = 0;
          } else {
            int condval_201;
            if ((((slab * 2) + m_12) == 1)) {
              condval_201 = 3;
            } else {
              condval_201 = (((slab * 2) + m_12) - 1);
            }
            condval_200 = condval_201;
          }
          int condval_203;
          if ((((slab * 2) + m_12) == 0)) {
            condval_203 = 0;
          } else {
            int condval_204;
            if ((((slab * 2) + m_12) == 1)) {
              condval_204 = 3;
            } else {
              condval_204 = (((slab * 2) + m_12) - 1);
            }
            condval_203 = condval_204;
          }
          int condval_205;
          if ((((slab * 2) + m_12) == 0)) {
            condval_205 = 0;
          } else {
            int condval_206;
            if ((((slab * 2) + m_12) == 1)) {
              condval_206 = 3;
            } else {
              condval_206 = (((slab * 2) + m_12) - 1);
            }
            condval_205 = condval_206;
          }
          int condval_202;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_203 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_205 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_207;
            if ((((slab * 2) + m_12) == 0)) {
              condval_207 = 0;
            } else {
              int condval_208;
              if ((((slab * 2) + m_12) == 1)) {
                condval_208 = 3;
              } else {
                condval_208 = (((slab * 2) + m_12) - 1);
              }
              condval_207 = condval_208;
            }
            int condval_209;
            if ((((slab * 2) + m_12) == 0)) {
              condval_209 = 0;
            } else {
              int condval_210;
              if ((((slab * 2) + m_12) == 1)) {
                condval_210 = 3;
              } else {
                condval_210 = (((slab * 2) + m_12) - 1);
              }
              condval_209 = condval_210;
            }
            condval_202 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_207 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_209 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_211;
            if ((((slab * 2) + m_12) == 0)) {
              condval_211 = 0;
            } else {
              int condval_212;
              if ((((slab * 2) + m_12) == 1)) {
                condval_212 = 3;
              } else {
                condval_212 = (((slab * 2) + m_12) - 1);
              }
              condval_211 = condval_212;
            }
            int condval_213;
            if ((((slab * 2) + m_12) == 0)) {
              condval_213 = 0;
            } else {
              int condval_214;
              if ((((slab * 2) + m_12) == 1)) {
                condval_214 = 3;
              } else {
                condval_214 = (((slab * 2) + m_12) - 1);
              }
              condval_213 = condval_214;
            }
            condval_202 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_211 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_213 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_215;
          if ((((slab * 2) + m_12) == 0)) {
            condval_215 = 0;
          } else {
            int condval_216;
            if ((((slab * 2) + m_12) == 1)) {
              condval_216 = 3;
            } else {
              condval_216 = (((slab * 2) + m_12) - 1);
            }
            condval_215 = condval_216;
          }
          int condval_217;
          if ((((slab * 2) + m_12) == 0)) {
            condval_217 = 0;
          } else {
            int condval_218;
            if ((((slab * 2) + m_12) == 1)) {
              condval_218 = 3;
            } else {
              condval_218 = (((slab * 2) + m_12) - 1);
            }
            condval_217 = condval_218;
          }
          int condval_219;
          if ((((slab * 2) + m_12) == 0)) {
            condval_219 = 0;
          } else {
            int condval_220;
            if ((((slab * 2) + m_12) == 1)) {
              condval_220 = 3;
            } else {
              condval_220 = (((slab * 2) + m_12) - 1);
            }
            condval_219 = condval_220;
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
          int condval_221;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_222 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_224 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_226;
            if ((((slab * 2) + m_12) == 0)) {
              condval_226 = 0;
            } else {
              int condval_227;
              if ((((slab * 2) + m_12) == 1)) {
                condval_227 = 3;
              } else {
                condval_227 = (((slab * 2) + m_12) - 1);
              }
              condval_226 = condval_227;
            }
            int condval_228;
            if ((((slab * 2) + m_12) == 0)) {
              condval_228 = 0;
            } else {
              int condval_229;
              if ((((slab * 2) + m_12) == 1)) {
                condval_229 = 3;
              } else {
                condval_229 = (((slab * 2) + m_12) - 1);
              }
              condval_228 = condval_229;
            }
            condval_221 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_226 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_228 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_230;
            if ((((slab * 2) + m_12) == 0)) {
              condval_230 = 0;
            } else {
              int condval_231;
              if ((((slab * 2) + m_12) == 1)) {
                condval_231 = 3;
              } else {
                condval_231 = (((slab * 2) + m_12) - 1);
              }
              condval_230 = condval_231;
            }
            int condval_232;
            if ((((slab * 2) + m_12) == 0)) {
              condval_232 = 0;
            } else {
              int condval_233;
              if ((((slab * 2) + m_12) == 1)) {
                condval_233 = 3;
              } else {
                condval_233 = (((slab * 2) + m_12) - 1);
              }
              condval_232 = condval_233;
            }
            condval_221 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_230 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_232 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_234;
          if ((((slab * 2) + m_12) == 0)) {
            condval_234 = 0;
          } else {
            int condval_235;
            if ((((slab * 2) + m_12) == 1)) {
              condval_235 = 3;
            } else {
              condval_235 = (((slab * 2) + m_12) - 1);
            }
            condval_234 = condval_235;
          }
          int condval_236;
          if ((((slab * 2) + m_12) == 0)) {
            condval_236 = 0;
          } else {
            int condval_237;
            if ((((slab * 2) + m_12) == 1)) {
              condval_237 = 3;
            } else {
              condval_237 = (((slab * 2) + m_12) - 1);
            }
            condval_236 = condval_237;
          }
          int condval_238;
          if ((((slab * 2) + m_12) == 0)) {
            condval_238 = 0;
          } else {
            int condval_239;
            if ((((slab * 2) + m_12) == 1)) {
              condval_239 = 3;
            } else {
              condval_239 = (((slab * 2) + m_12) - 1);
            }
            condval_238 = condval_239;
          }
          int condval_241;
          if ((((slab * 2) + m_12) == 0)) {
            condval_241 = 0;
          } else {
            int condval_242;
            if ((((slab * 2) + m_12) == 1)) {
              condval_242 = 3;
            } else {
              condval_242 = (((slab * 2) + m_12) - 1);
            }
            condval_241 = condval_242;
          }
          int condval_243;
          if ((((slab * 2) + m_12) == 0)) {
            condval_243 = 0;
          } else {
            int condval_244;
            if ((((slab * 2) + m_12) == 1)) {
              condval_244 = 3;
            } else {
              condval_244 = (((slab * 2) + m_12) - 1);
            }
            condval_243 = condval_244;
          }
          int condval_240;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_241 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_243 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_245;
            if ((((slab * 2) + m_12) == 0)) {
              condval_245 = 0;
            } else {
              int condval_246;
              if ((((slab * 2) + m_12) == 1)) {
                condval_246 = 3;
              } else {
                condval_246 = (((slab * 2) + m_12) - 1);
              }
              condval_245 = condval_246;
            }
            int condval_247;
            if ((((slab * 2) + m_12) == 0)) {
              condval_247 = 0;
            } else {
              int condval_248;
              if ((((slab * 2) + m_12) == 1)) {
                condval_248 = 3;
              } else {
                condval_248 = (((slab * 2) + m_12) - 1);
              }
              condval_247 = condval_248;
            }
            condval_240 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_245 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_247 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_249;
            if ((((slab * 2) + m_12) == 0)) {
              condval_249 = 0;
            } else {
              int condval_250;
              if ((((slab * 2) + m_12) == 1)) {
                condval_250 = 3;
              } else {
                condval_250 = (((slab * 2) + m_12) - 1);
              }
              condval_249 = condval_250;
            }
            int condval_251;
            if ((((slab * 2) + m_12) == 0)) {
              condval_251 = 0;
            } else {
              int condval_252;
              if ((((slab * 2) + m_12) == 1)) {
                condval_252 = 3;
              } else {
                condval_252 = (((slab * 2) + m_12) - 1);
              }
              condval_251 = condval_252;
            }
            condval_240 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_249 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_251 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
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
          int condval_260;
          if ((((slab * 2) + m_12) == 0)) {
            condval_260 = 0;
          } else {
            int condval_261;
            if ((((slab * 2) + m_12) == 1)) {
              condval_261 = 3;
            } else {
              condval_261 = (((slab * 2) + m_12) - 1);
            }
            condval_260 = condval_261;
          }
          int condval_262;
          if ((((slab * 2) + m_12) == 0)) {
            condval_262 = 0;
          } else {
            int condval_263;
            if ((((slab * 2) + m_12) == 1)) {
              condval_263 = 3;
            } else {
              condval_263 = (((slab * 2) + m_12) - 1);
            }
            condval_262 = condval_263;
          }
          int condval_259;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_260 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_262 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_264;
            if ((((slab * 2) + m_12) == 0)) {
              condval_264 = 0;
            } else {
              int condval_265;
              if ((((slab * 2) + m_12) == 1)) {
                condval_265 = 3;
              } else {
                condval_265 = (((slab * 2) + m_12) - 1);
              }
              condval_264 = condval_265;
            }
            int condval_266;
            if ((((slab * 2) + m_12) == 0)) {
              condval_266 = 0;
            } else {
              int condval_267;
              if ((((slab * 2) + m_12) == 1)) {
                condval_267 = 3;
              } else {
                condval_267 = (((slab * 2) + m_12) - 1);
              }
              condval_266 = condval_267;
            }
            condval_259 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_264 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_266 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_268;
            if ((((slab * 2) + m_12) == 0)) {
              condval_268 = 0;
            } else {
              int condval_269;
              if ((((slab * 2) + m_12) == 1)) {
                condval_269 = 3;
              } else {
                condval_269 = (((slab * 2) + m_12) - 1);
              }
              condval_268 = condval_269;
            }
            int condval_270;
            if ((((slab * 2) + m_12) == 0)) {
              condval_270 = 0;
            } else {
              int condval_271;
              if ((((slab * 2) + m_12) == 1)) {
                condval_271 = 3;
              } else {
                condval_271 = (((slab * 2) + m_12) - 1);
              }
              condval_270 = condval_271;
            }
            condval_259 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_268 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_270 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_272;
          if ((((slab * 2) + m_12) == 0)) {
            condval_272 = 0;
          } else {
            int condval_273;
            if ((((slab * 2) + m_12) == 1)) {
              condval_273 = 3;
            } else {
              condval_273 = (((slab * 2) + m_12) - 1);
            }
            condval_272 = condval_273;
          }
          int condval_274;
          if ((((slab * 2) + m_12) == 0)) {
            condval_274 = 0;
          } else {
            int condval_275;
            if ((((slab * 2) + m_12) == 1)) {
              condval_275 = 3;
            } else {
              condval_275 = (((slab * 2) + m_12) - 1);
            }
            condval_274 = condval_275;
          }
          int condval_276;
          if ((((slab * 2) + m_12) == 0)) {
            condval_276 = 0;
          } else {
            int condval_277;
            if ((((slab * 2) + m_12) == 1)) {
              condval_277 = 3;
            } else {
              condval_277 = (((slab * 2) + m_12) - 1);
            }
            condval_276 = condval_277;
          }
          int condval_279;
          if ((((slab * 2) + m_12) == 0)) {
            condval_279 = 0;
          } else {
            int condval_280;
            if ((((slab * 2) + m_12) == 1)) {
              condval_280 = 3;
            } else {
              condval_280 = (((slab * 2) + m_12) - 1);
            }
            condval_279 = condval_280;
          }
          int condval_281;
          if ((((slab * 2) + m_12) == 0)) {
            condval_281 = 0;
          } else {
            int condval_282;
            if ((((slab * 2) + m_12) == 1)) {
              condval_282 = 3;
            } else {
              condval_282 = (((slab * 2) + m_12) - 1);
            }
            condval_281 = condval_282;
          }
          int condval_278;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_279 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_281 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_283;
            if ((((slab * 2) + m_12) == 0)) {
              condval_283 = 0;
            } else {
              int condval_284;
              if ((((slab * 2) + m_12) == 1)) {
                condval_284 = 3;
              } else {
                condval_284 = (((slab * 2) + m_12) - 1);
              }
              condval_283 = condval_284;
            }
            int condval_285;
            if ((((slab * 2) + m_12) == 0)) {
              condval_285 = 0;
            } else {
              int condval_286;
              if ((((slab * 2) + m_12) == 1)) {
                condval_286 = 3;
              } else {
                condval_286 = (((slab * 2) + m_12) - 1);
              }
              condval_285 = condval_286;
            }
            condval_278 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_283 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_285 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_287;
            if ((((slab * 2) + m_12) == 0)) {
              condval_287 = 0;
            } else {
              int condval_288;
              if ((((slab * 2) + m_12) == 1)) {
                condval_288 = 3;
              } else {
                condval_288 = (((slab * 2) + m_12) - 1);
              }
              condval_287 = condval_288;
            }
            int condval_289;
            if ((((slab * 2) + m_12) == 0)) {
              condval_289 = 0;
            } else {
              int condval_290;
              if ((((slab * 2) + m_12) == 1)) {
                condval_290 = 3;
              } else {
                condval_290 = (((slab * 2) + m_12) - 1);
              }
              condval_289 = condval_290;
            }
            condval_278 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_287 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_289 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_291;
          if ((((slab * 2) + m_12) == 0)) {
            condval_291 = 0;
          } else {
            int condval_292;
            if ((((slab * 2) + m_12) == 1)) {
              condval_292 = 3;
            } else {
              condval_292 = (((slab * 2) + m_12) - 1);
            }
            condval_291 = condval_292;
          }
          int condval_293;
          if ((((slab * 2) + m_12) == 0)) {
            condval_293 = 0;
          } else {
            int condval_294;
            if ((((slab * 2) + m_12) == 1)) {
              condval_294 = 3;
            } else {
              condval_294 = (((slab * 2) + m_12) - 1);
            }
            condval_293 = condval_294;
          }
          int condval_295;
          if ((((slab * 2) + m_12) == 0)) {
            condval_295 = 0;
          } else {
            int condval_296;
            if ((((slab * 2) + m_12) == 1)) {
              condval_296 = 3;
            } else {
              condval_296 = (((slab * 2) + m_12) - 1);
            }
            condval_295 = condval_296;
          }
          int condval_297;
          if ((((slab * 2) + m_12) == 0)) {
            condval_297 = 0;
          } else {
            int condval_298;
            if ((((slab * 2) + m_12) == 1)) {
              condval_298 = 3;
            } else {
              condval_298 = (((slab * 2) + m_12) - 1);
            }
            condval_297 = condval_298;
          }
          int condval_299;
          if ((((slab * 2) + m_12) == 0)) {
            condval_299 = 0;
          } else {
            int condval_300;
            if ((((slab * 2) + m_12) == 1)) {
              condval_300 = 3;
            } else {
              condval_300 = (((slab * 2) + m_12) - 1);
            }
            condval_299 = condval_300;
          }
          int condval_301;
          if ((((slab * 2) + m_12) == 0)) {
            condval_301 = 0;
          } else {
            int condval_302;
            if ((((slab * 2) + m_12) == 1)) {
              condval_302 = 3;
            } else {
              condval_302 = (((slab * 2) + m_12) - 1);
            }
            condval_301 = condval_302;
          }
          int condval_303;
          if ((((slab * 2) + m_12) == 0)) {
            condval_303 = 0;
          } else {
            int condval_304;
            if ((((slab * 2) + m_12) == 1)) {
              condval_304 = 3;
            } else {
              condval_304 = (((slab * 2) + m_12) - 1);
            }
            condval_303 = condval_304;
          }
          int condval_305;
          if ((((slab * 2) + m_12) == 0)) {
            condval_305 = 0;
          } else {
            int condval_306;
            if ((((slab * 2) + m_12) == 1)) {
              condval_306 = 3;
            } else {
              condval_306 = (((slab * 2) + m_12) - 1);
            }
            condval_305 = condval_306;
          }
          int condval_307;
          if ((((slab * 2) + m_12) == 0)) {
            condval_307 = 0;
          } else {
            int condval_308;
            if ((((slab * 2) + m_12) == 1)) {
              condval_308 = 3;
            } else {
              condval_308 = (((slab * 2) + m_12) - 1);
            }
            condval_307 = condval_308;
          }
          int condval_309;
          if ((((slab * 2) + m_12) == 0)) {
            condval_309 = 0;
          } else {
            int condval_310;
            if ((((slab * 2) + m_12) == 1)) {
              condval_310 = 3;
            } else {
              condval_310 = (((slab * 2) + m_12) - 1);
            }
            condval_309 = condval_310;
          }
          int condval_311;
          if ((((slab * 2) + m_12) == 0)) {
            condval_311 = 0;
          } else {
            int condval_312;
            if ((((slab * 2) + m_12) == 1)) {
              condval_312 = 3;
            } else {
              condval_312 = (((slab * 2) + m_12) - 1);
            }
            condval_311 = condval_312;
          }
          int condval_314;
          if ((((slab * 2) + m_12) == 0)) {
            condval_314 = 0;
          } else {
            int condval_315;
            if ((((slab * 2) + m_12) == 1)) {
              condval_315 = 3;
            } else {
              condval_315 = (((slab * 2) + m_12) - 1);
            }
            condval_314 = condval_315;
          }
          int condval_316;
          if ((((slab * 2) + m_12) == 0)) {
            condval_316 = 0;
          } else {
            int condval_317;
            if ((((slab * 2) + m_12) == 1)) {
              condval_317 = 3;
            } else {
              condval_317 = (((slab * 2) + m_12) - 1);
            }
            condval_316 = condval_317;
          }
          int condval_313;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_314 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_316 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_318;
            if ((((slab * 2) + m_12) == 0)) {
              condval_318 = 0;
            } else {
              int condval_319;
              if ((((slab * 2) + m_12) == 1)) {
                condval_319 = 3;
              } else {
                condval_319 = (((slab * 2) + m_12) - 1);
              }
              condval_318 = condval_319;
            }
            int condval_320;
            if ((((slab * 2) + m_12) == 0)) {
              condval_320 = 0;
            } else {
              int condval_321;
              if ((((slab * 2) + m_12) == 1)) {
                condval_321 = 3;
              } else {
                condval_321 = (((slab * 2) + m_12) - 1);
              }
              condval_320 = condval_321;
            }
            condval_313 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_318 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_320 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_322;
            if ((((slab * 2) + m_12) == 0)) {
              condval_322 = 0;
            } else {
              int condval_323;
              if ((((slab * 2) + m_12) == 1)) {
                condval_323 = 3;
              } else {
                condval_323 = (((slab * 2) + m_12) - 1);
              }
              condval_322 = condval_323;
            }
            int condval_324;
            if ((((slab * 2) + m_12) == 0)) {
              condval_324 = 0;
            } else {
              int condval_325;
              if ((((slab * 2) + m_12) == 1)) {
                condval_325 = 3;
              } else {
                condval_325 = (((slab * 2) + m_12) - 1);
              }
              condval_324 = condval_325;
            }
            condval_313 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_322 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_324 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_326;
          if ((((slab * 2) + m_12) == 0)) {
            condval_326 = 0;
          } else {
            int condval_327;
            if ((((slab * 2) + m_12) == 1)) {
              condval_327 = 3;
            } else {
              condval_327 = (((slab * 2) + m_12) - 1);
            }
            condval_326 = condval_327;
          }
          int condval_328;
          if ((((slab * 2) + m_12) == 0)) {
            condval_328 = 0;
          } else {
            int condval_329;
            if ((((slab * 2) + m_12) == 1)) {
              condval_329 = 3;
            } else {
              condval_329 = (((slab * 2) + m_12) - 1);
            }
            condval_328 = condval_329;
          }
          int condval_330;
          if ((((slab * 2) + m_12) == 0)) {
            condval_330 = 0;
          } else {
            int condval_331;
            if ((((slab * 2) + m_12) == 1)) {
              condval_331 = 3;
            } else {
              condval_331 = (((slab * 2) + m_12) - 1);
            }
            condval_330 = condval_331;
          }
          int condval_333;
          if ((((slab * 2) + m_12) == 0)) {
            condval_333 = 0;
          } else {
            int condval_334;
            if ((((slab * 2) + m_12) == 1)) {
              condval_334 = 3;
            } else {
              condval_334 = (((slab * 2) + m_12) - 1);
            }
            condval_333 = condval_334;
          }
          int condval_335;
          if ((((slab * 2) + m_12) == 0)) {
            condval_335 = 0;
          } else {
            int condval_336;
            if ((((slab * 2) + m_12) == 1)) {
              condval_336 = 3;
            } else {
              condval_336 = (((slab * 2) + m_12) - 1);
            }
            condval_335 = condval_336;
          }
          int condval_332;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_333 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_335 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_337;
            if ((((slab * 2) + m_12) == 0)) {
              condval_337 = 0;
            } else {
              int condval_338;
              if ((((slab * 2) + m_12) == 1)) {
                condval_338 = 3;
              } else {
                condval_338 = (((slab * 2) + m_12) - 1);
              }
              condval_337 = condval_338;
            }
            int condval_339;
            if ((((slab * 2) + m_12) == 0)) {
              condval_339 = 0;
            } else {
              int condval_340;
              if ((((slab * 2) + m_12) == 1)) {
                condval_340 = 3;
              } else {
                condval_340 = (((slab * 2) + m_12) - 1);
              }
              condval_339 = condval_340;
            }
            condval_332 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_337 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_339 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_341;
            if ((((slab * 2) + m_12) == 0)) {
              condval_341 = 0;
            } else {
              int condval_342;
              if ((((slab * 2) + m_12) == 1)) {
                condval_342 = 3;
              } else {
                condval_342 = (((slab * 2) + m_12) - 1);
              }
              condval_341 = condval_342;
            }
            int condval_343;
            if ((((slab * 2) + m_12) == 0)) {
              condval_343 = 0;
            } else {
              int condval_344;
              if ((((slab * 2) + m_12) == 1)) {
                condval_344 = 3;
              } else {
                condval_344 = (((slab * 2) + m_12) - 1);
              }
              condval_343 = condval_344;
            }
            condval_332 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_341 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_343 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_345;
          if ((((slab * 2) + m_12) == 0)) {
            condval_345 = 0;
          } else {
            int condval_346;
            if ((((slab * 2) + m_12) == 1)) {
              condval_346 = 3;
            } else {
              condval_346 = (((slab * 2) + m_12) - 1);
            }
            condval_345 = condval_346;
          }
          int condval_347;
          if ((((slab * 2) + m_12) == 0)) {
            condval_347 = 0;
          } else {
            int condval_348;
            if ((((slab * 2) + m_12) == 1)) {
              condval_348 = 3;
            } else {
              condval_348 = (((slab * 2) + m_12) - 1);
            }
            condval_347 = condval_348;
          }
          int condval_349;
          if ((((slab * 2) + m_12) == 0)) {
            condval_349 = 0;
          } else {
            int condval_350;
            if ((((slab * 2) + m_12) == 1)) {
              condval_350 = 3;
            } else {
              condval_350 = (((slab * 2) + m_12) - 1);
            }
            condval_349 = condval_350;
          }
          int condval_352;
          if ((((slab * 2) + m_12) == 0)) {
            condval_352 = 0;
          } else {
            int condval_353;
            if ((((slab * 2) + m_12) == 1)) {
              condval_353 = 3;
            } else {
              condval_353 = (((slab * 2) + m_12) - 1);
            }
            condval_352 = condval_353;
          }
          int condval_354;
          if ((((slab * 2) + m_12) == 0)) {
            condval_354 = 0;
          } else {
            int condval_355;
            if ((((slab * 2) + m_12) == 1)) {
              condval_355 = 3;
            } else {
              condval_355 = (((slab * 2) + m_12) - 1);
            }
            condval_354 = condval_355;
          }
          int condval_351;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_352 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_354 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_356;
            if ((((slab * 2) + m_12) == 0)) {
              condval_356 = 0;
            } else {
              int condval_357;
              if ((((slab * 2) + m_12) == 1)) {
                condval_357 = 3;
              } else {
                condval_357 = (((slab * 2) + m_12) - 1);
              }
              condval_356 = condval_357;
            }
            int condval_358;
            if ((((slab * 2) + m_12) == 0)) {
              condval_358 = 0;
            } else {
              int condval_359;
              if ((((slab * 2) + m_12) == 1)) {
                condval_359 = 3;
              } else {
                condval_359 = (((slab * 2) + m_12) - 1);
              }
              condval_358 = condval_359;
            }
            condval_351 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_356 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_358 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_360;
            if ((((slab * 2) + m_12) == 0)) {
              condval_360 = 0;
            } else {
              int condval_361;
              if ((((slab * 2) + m_12) == 1)) {
                condval_361 = 3;
              } else {
                condval_361 = (((slab * 2) + m_12) - 1);
              }
              condval_360 = condval_361;
            }
            int condval_362;
            if ((((slab * 2) + m_12) == 0)) {
              condval_362 = 0;
            } else {
              int condval_363;
              if ((((slab * 2) + m_12) == 1)) {
                condval_363 = 3;
              } else {
                condval_363 = (((slab * 2) + m_12) - 1);
              }
              condval_362 = condval_363;
            }
            condval_351 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_360 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_362 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_364;
          if ((((slab * 2) + m_12) == 0)) {
            condval_364 = 0;
          } else {
            int condval_365;
            if ((((slab * 2) + m_12) == 1)) {
              condval_365 = 3;
            } else {
              condval_365 = (((slab * 2) + m_12) - 1);
            }
            condval_364 = condval_365;
          }
          int condval_366;
          if ((((slab * 2) + m_12) == 0)) {
            condval_366 = 0;
          } else {
            int condval_367;
            if ((((slab * 2) + m_12) == 1)) {
              condval_367 = 3;
            } else {
              condval_367 = (((slab * 2) + m_12) - 1);
            }
            condval_366 = condval_367;
          }
          int condval_368;
          if ((((slab * 2) + m_12) == 0)) {
            condval_368 = 0;
          } else {
            int condval_369;
            if ((((slab * 2) + m_12) == 1)) {
              condval_369 = 3;
            } else {
              condval_369 = (((slab * 2) + m_12) - 1);
            }
            condval_368 = condval_369;
          }
          int condval_371;
          if ((((slab * 2) + m_12) == 0)) {
            condval_371 = 0;
          } else {
            int condval_372;
            if ((((slab * 2) + m_12) == 1)) {
              condval_372 = 3;
            } else {
              condval_372 = (((slab * 2) + m_12) - 1);
            }
            condval_371 = condval_372;
          }
          int condval_373;
          if ((((slab * 2) + m_12) == 0)) {
            condval_373 = 0;
          } else {
            int condval_374;
            if ((((slab * 2) + m_12) == 1)) {
              condval_374 = 3;
            } else {
              condval_374 = (((slab * 2) + m_12) - 1);
            }
            condval_373 = condval_374;
          }
          int condval_370;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_371 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_373 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_375;
            if ((((slab * 2) + m_12) == 0)) {
              condval_375 = 0;
            } else {
              int condval_376;
              if ((((slab * 2) + m_12) == 1)) {
                condval_376 = 3;
              } else {
                condval_376 = (((slab * 2) + m_12) - 1);
              }
              condval_375 = condval_376;
            }
            int condval_377;
            if ((((slab * 2) + m_12) == 0)) {
              condval_377 = 0;
            } else {
              int condval_378;
              if ((((slab * 2) + m_12) == 1)) {
                condval_378 = 3;
              } else {
                condval_378 = (((slab * 2) + m_12) - 1);
              }
              condval_377 = condval_378;
            }
            condval_370 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_375 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_377 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_379;
            if ((((slab * 2) + m_12) == 0)) {
              condval_379 = 0;
            } else {
              int condval_380;
              if ((((slab * 2) + m_12) == 1)) {
                condval_380 = 3;
              } else {
                condval_380 = (((slab * 2) + m_12) - 1);
              }
              condval_379 = condval_380;
            }
            int condval_381;
            if ((((slab * 2) + m_12) == 0)) {
              condval_381 = 0;
            } else {
              int condval_382;
              if ((((slab * 2) + m_12) == 1)) {
                condval_382 = 3;
              } else {
                condval_382 = (((slab * 2) + m_12) - 1);
              }
              condval_381 = condval_382;
            }
            condval_370 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_379 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_381 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_383;
          if ((((slab * 2) + m_12) == 0)) {
            condval_383 = 0;
          } else {
            int condval_384;
            if ((((slab * 2) + m_12) == 1)) {
              condval_384 = 3;
            } else {
              condval_384 = (((slab * 2) + m_12) - 1);
            }
            condval_383 = condval_384;
          }
          int condval_385;
          if ((((slab * 2) + m_12) == 0)) {
            condval_385 = 0;
          } else {
            int condval_386;
            if ((((slab * 2) + m_12) == 1)) {
              condval_386 = 3;
            } else {
              condval_386 = (((slab * 2) + m_12) - 1);
            }
            condval_385 = condval_386;
          }
          int condval_387;
          if ((((slab * 2) + m_12) == 0)) {
            condval_387 = 0;
          } else {
            int condval_388;
            if ((((slab * 2) + m_12) == 1)) {
              condval_388 = 3;
            } else {
              condval_388 = (((slab * 2) + m_12) - 1);
            }
            condval_387 = condval_388;
          }
          int condval_390;
          if ((((slab * 2) + m_12) == 0)) {
            condval_390 = 0;
          } else {
            int condval_391;
            if ((((slab * 2) + m_12) == 1)) {
              condval_391 = 3;
            } else {
              condval_391 = (((slab * 2) + m_12) - 1);
            }
            condval_390 = condval_391;
          }
          int condval_392;
          if ((((slab * 2) + m_12) == 0)) {
            condval_392 = 0;
          } else {
            int condval_393;
            if ((((slab * 2) + m_12) == 1)) {
              condval_393 = 3;
            } else {
              condval_393 = (((slab * 2) + m_12) - 1);
            }
            condval_392 = condval_393;
          }
          int condval_389;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_390 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_392 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_394;
            if ((((slab * 2) + m_12) == 0)) {
              condval_394 = 0;
            } else {
              int condval_395;
              if ((((slab * 2) + m_12) == 1)) {
                condval_395 = 3;
              } else {
                condval_395 = (((slab * 2) + m_12) - 1);
              }
              condval_394 = condval_395;
            }
            int condval_396;
            if ((((slab * 2) + m_12) == 0)) {
              condval_396 = 0;
            } else {
              int condval_397;
              if ((((slab * 2) + m_12) == 1)) {
                condval_397 = 3;
              } else {
                condval_397 = (((slab * 2) + m_12) - 1);
              }
              condval_396 = condval_397;
            }
            condval_389 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_394 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_396 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_398;
            if ((((slab * 2) + m_12) == 0)) {
              condval_398 = 0;
            } else {
              int condval_399;
              if ((((slab * 2) + m_12) == 1)) {
                condval_399 = 3;
              } else {
                condval_399 = (((slab * 2) + m_12) - 1);
              }
              condval_398 = condval_399;
            }
            int condval_400;
            if ((((slab * 2) + m_12) == 0)) {
              condval_400 = 0;
            } else {
              int condval_401;
              if ((((slab * 2) + m_12) == 1)) {
                condval_401 = 3;
              } else {
                condval_401 = (((slab * 2) + m_12) - 1);
              }
              condval_400 = condval_401;
            }
            condval_389 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_398 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_400 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_402;
          if ((((slab * 2) + m_12) == 0)) {
            condval_402 = 0;
          } else {
            int condval_403;
            if ((((slab * 2) + m_12) == 1)) {
              condval_403 = 3;
            } else {
              condval_403 = (((slab * 2) + m_12) - 1);
            }
            condval_402 = condval_403;
          }
          int condval_404;
          if ((((slab * 2) + m_12) == 0)) {
            condval_404 = 0;
          } else {
            int condval_405;
            if ((((slab * 2) + m_12) == 1)) {
              condval_405 = 3;
            } else {
              condval_405 = (((slab * 2) + m_12) - 1);
            }
            condval_404 = condval_405;
          }
          int condval_406;
          if ((((slab * 2) + m_12) == 0)) {
            condval_406 = 0;
          } else {
            int condval_407;
            if ((((slab * 2) + m_12) == 1)) {
              condval_407 = 3;
            } else {
              condval_407 = (((slab * 2) + m_12) - 1);
            }
            condval_406 = condval_407;
          }
          int condval_409;
          if ((((slab * 2) + m_12) == 0)) {
            condval_409 = 0;
          } else {
            int condval_410;
            if ((((slab * 2) + m_12) == 1)) {
              condval_410 = 3;
            } else {
              condval_410 = (((slab * 2) + m_12) - 1);
            }
            condval_409 = condval_410;
          }
          int condval_411;
          if ((((slab * 2) + m_12) == 0)) {
            condval_411 = 0;
          } else {
            int condval_412;
            if ((((slab * 2) + m_12) == 1)) {
              condval_412 = 3;
            } else {
              condval_412 = (((slab * 2) + m_12) - 1);
            }
            condval_411 = condval_412;
          }
          int condval_408;
          if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_409 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_411 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int condval_413;
            if ((((slab * 2) + m_12) == 0)) {
              condval_413 = 0;
            } else {
              int condval_414;
              if ((((slab * 2) + m_12) == 1)) {
                condval_414 = 3;
              } else {
                condval_414 = (((slab * 2) + m_12) - 1);
              }
              condval_413 = condval_414;
            }
            int condval_415;
            if ((((slab * 2) + m_12) == 0)) {
              condval_415 = 0;
            } else {
              int condval_416;
              if ((((slab * 2) + m_12) == 1)) {
                condval_416 = 3;
              } else {
                condval_416 = (((slab * 2) + m_12) - 1);
              }
              condval_415 = condval_416;
            }
            condval_408 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_413 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_415 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int condval_417;
            if ((((slab * 2) + m_12) == 0)) {
              condval_417 = 0;
            } else {
              int condval_418;
              if ((((slab * 2) + m_12) == 1)) {
                condval_418 = 3;
              } else {
                condval_418 = (((slab * 2) + m_12) - 1);
              }
              condval_417 = condval_418;
            }
            int condval_419;
            if ((((slab * 2) + m_12) == 0)) {
              condval_419 = 0;
            } else {
              int condval_420;
              if ((((slab * 2) + m_12) == 1)) {
                condval_420 = 3;
              } else {
                condval_420 = (((slab * 2) + m_12) - 1);
              }
              condval_419 = condval_420;
            }
            condval_408 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_417 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_419 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_421;
          if ((((slab * 2) + m_12) == 0)) {
            condval_421 = 0;
          } else {
            int condval_422;
            if ((((slab * 2) + m_12) == 1)) {
              condval_422 = 3;
            } else {
              condval_422 = (((slab * 2) + m_12) - 1);
            }
            condval_421 = condval_422;
          }
          int condval_423;
          if ((((slab * 2) + m_12) == 0)) {
            condval_423 = 0;
          } else {
            int condval_424;
            if ((((slab * 2) + m_12) == 1)) {
              condval_424 = 3;
            } else {
              condval_424 = (((slab * 2) + m_12) - 1);
            }
            condval_423 = condval_424;
          }
          int condval_425;
          if ((((slab * 2) + m_12) == 0)) {
            condval_425 = 0;
          } else {
            int condval_426;
            if ((((slab * 2) + m_12) == 1)) {
              condval_426 = 3;
            } else {
              condval_426 = (((slab * 2) + m_12) - 1);
            }
            condval_425 = condval_426;
          }
          condval_41 = (((((((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_50 & 3) * 16) + (((((condval_63 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_65 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_67 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_69 & 3) * 16) + (((((condval_82 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_84 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_86 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_88 & 3) * 16) + (((((condval_101 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_103 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_105 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_107 & 3) * 16) + (((((condval_120 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_122 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_124 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_126 & 3) * 16) + (((((condval_139 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_141 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_143 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_145 & 3) * 16) + (((((condval_158 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_160 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_162 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + ((((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_164 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_166 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_168 * 2)) + ((((condval_181 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) * 40960)) + (((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_183 & 3) * 16) + (((((condval_196 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_198 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_200 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_202 & 3) * 16) + (((((condval_215 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_217 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_219 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_221 & 3) * 16) + (((((condval_234 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_236 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_238 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_240 & 3) * 16) + (((((condval_253 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_255 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_257 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_259 & 3) * 16) + (((((condval_272 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_274 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_276 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_278 & 3) * 16) + (((((condval_291 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_293 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_295 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_297 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_299 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_301 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_303 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_305 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_307 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_309 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_311 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_313 & 3) * 16) + (((((condval_326 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_328 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_330 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_332 & 3) * 16) + (((((condval_345 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_347 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_349 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_351 & 3) * 16) + (((((condval_364 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_366 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_368 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_370 & 3) * 16) + (((((condval_383 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_385 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_387 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_389 & 3) * 16) + (((((condval_402 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_404 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_406 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_408 & 3) * 16) + (((((condval_421 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_423 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_425 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_41 = -1;
        }
        if ((((int)threadIdx.x) < 16) & (0 <= condval_41)) {
          int condval_428;
          if ((((slab * 2) + m_12) == 0)) {
            condval_428 = 0;
          } else {
            int condval_429;
            if ((((slab * 2) + m_12) == 1)) {
              condval_429 = 3;
            } else {
              condval_429 = (((slab * 2) + m_12) - 1);
            }
            condval_428 = condval_429;
          }
          int condval_430;
          if ((((slab * 2) + m_12) == 0)) {
            condval_430 = 0;
          } else {
            int condval_431;
            if ((((slab * 2) + m_12) == 1)) {
              condval_431 = 3;
            } else {
              condval_431 = (((slab * 2) + m_12) - 1);
            }
            condval_430 = condval_431;
          }
          int condval_432;
          if ((((slab * 2) + m_12) == 0)) {
            condval_432 = 0;
          } else {
            int condval_433;
            if ((((slab * 2) + m_12) == 1)) {
              condval_433 = 3;
            } else {
              condval_433 = (((slab * 2) + m_12) - 1);
            }
            condval_432 = condval_433;
          }
          int condval_434;
          if ((((slab * 2) + m_12) == 0)) {
            condval_434 = 0;
          } else {
            int condval_435;
            if ((((slab * 2) + m_12) == 1)) {
              condval_435 = 3;
            } else {
              condval_435 = (((slab * 2) + m_12) - 1);
            }
            condval_434 = condval_435;
          }
          int condval_427;
          if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_428 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_430 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_432 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_434 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
            int condval_437;
            if ((((slab * 2) + m_12) == 0)) {
              condval_437 = 0;
            } else {
              int condval_438;
              if ((((slab * 2) + m_12) == 1)) {
                condval_438 = 3;
              } else {
                condval_438 = (((slab * 2) + m_12) - 1);
              }
              condval_437 = condval_438;
            }
            int condval_439;
            if ((((slab * 2) + m_12) == 0)) {
              condval_439 = 0;
            } else {
              int condval_440;
              if ((((slab * 2) + m_12) == 1)) {
                condval_440 = 3;
              } else {
                condval_440 = (((slab * 2) + m_12) - 1);
              }
              condval_439 = condval_440;
            }
            int condval_436;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_437 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_439 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_441;
              if ((((slab * 2) + m_12) == 0)) {
                condval_441 = 0;
              } else {
                int condval_442;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_442 = 3;
                } else {
                  condval_442 = (((slab * 2) + m_12) - 1);
                }
                condval_441 = condval_442;
              }
              int condval_443;
              if ((((slab * 2) + m_12) == 0)) {
                condval_443 = 0;
              } else {
                int condval_444;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_444 = 3;
                } else {
                  condval_444 = (((slab * 2) + m_12) - 1);
                }
                condval_443 = condval_444;
              }
              condval_436 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_441 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_443 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_445;
              if ((((slab * 2) + m_12) == 0)) {
                condval_445 = 0;
              } else {
                int condval_446;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_446 = 3;
                } else {
                  condval_446 = (((slab * 2) + m_12) - 1);
                }
                condval_445 = condval_446;
              }
              int condval_447;
              if ((((slab * 2) + m_12) == 0)) {
                condval_447 = 0;
              } else {
                int condval_448;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_448 = 3;
                } else {
                  condval_448 = (((slab * 2) + m_12) - 1);
                }
                condval_447 = condval_448;
              }
              condval_436 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_445 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_447 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_449;
            if ((((slab * 2) + m_12) == 0)) {
              condval_449 = 0;
            } else {
              int condval_450;
              if ((((slab * 2) + m_12) == 1)) {
                condval_450 = 3;
              } else {
                condval_450 = (((slab * 2) + m_12) - 1);
              }
              condval_449 = condval_450;
            }
            int condval_451;
            if ((((slab * 2) + m_12) == 0)) {
              condval_451 = 0;
            } else {
              int condval_452;
              if ((((slab * 2) + m_12) == 1)) {
                condval_452 = 3;
              } else {
                condval_452 = (((slab * 2) + m_12) - 1);
              }
              condval_451 = condval_452;
            }
            int condval_453;
            if ((((slab * 2) + m_12) == 0)) {
              condval_453 = 0;
            } else {
              int condval_454;
              if ((((slab * 2) + m_12) == 1)) {
                condval_454 = 3;
              } else {
                condval_454 = (((slab * 2) + m_12) - 1);
              }
              condval_453 = condval_454;
            }
            int condval_456;
            if ((((slab * 2) + m_12) == 0)) {
              condval_456 = 0;
            } else {
              int condval_457;
              if ((((slab * 2) + m_12) == 1)) {
                condval_457 = 3;
              } else {
                condval_457 = (((slab * 2) + m_12) - 1);
              }
              condval_456 = condval_457;
            }
            int condval_458;
            if ((((slab * 2) + m_12) == 0)) {
              condval_458 = 0;
            } else {
              int condval_459;
              if ((((slab * 2) + m_12) == 1)) {
                condval_459 = 3;
              } else {
                condval_459 = (((slab * 2) + m_12) - 1);
              }
              condval_458 = condval_459;
            }
            int condval_455;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_456 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_458 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_460;
              if ((((slab * 2) + m_12) == 0)) {
                condval_460 = 0;
              } else {
                int condval_461;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_461 = 3;
                } else {
                  condval_461 = (((slab * 2) + m_12) - 1);
                }
                condval_460 = condval_461;
              }
              int condval_462;
              if ((((slab * 2) + m_12) == 0)) {
                condval_462 = 0;
              } else {
                int condval_463;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_463 = 3;
                } else {
                  condval_463 = (((slab * 2) + m_12) - 1);
                }
                condval_462 = condval_463;
              }
              condval_455 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_460 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_462 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_464;
              if ((((slab * 2) + m_12) == 0)) {
                condval_464 = 0;
              } else {
                int condval_465;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_465 = 3;
                } else {
                  condval_465 = (((slab * 2) + m_12) - 1);
                }
                condval_464 = condval_465;
              }
              int condval_466;
              if ((((slab * 2) + m_12) == 0)) {
                condval_466 = 0;
              } else {
                int condval_467;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_467 = 3;
                } else {
                  condval_467 = (((slab * 2) + m_12) - 1);
                }
                condval_466 = condval_467;
              }
              condval_455 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_464 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_466 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_468;
            if ((((slab * 2) + m_12) == 0)) {
              condval_468 = 0;
            } else {
              int condval_469;
              if ((((slab * 2) + m_12) == 1)) {
                condval_469 = 3;
              } else {
                condval_469 = (((slab * 2) + m_12) - 1);
              }
              condval_468 = condval_469;
            }
            int condval_470;
            if ((((slab * 2) + m_12) == 0)) {
              condval_470 = 0;
            } else {
              int condval_471;
              if ((((slab * 2) + m_12) == 1)) {
                condval_471 = 3;
              } else {
                condval_471 = (((slab * 2) + m_12) - 1);
              }
              condval_470 = condval_471;
            }
            int condval_472;
            if ((((slab * 2) + m_12) == 0)) {
              condval_472 = 0;
            } else {
              int condval_473;
              if ((((slab * 2) + m_12) == 1)) {
                condval_473 = 3;
              } else {
                condval_473 = (((slab * 2) + m_12) - 1);
              }
              condval_472 = condval_473;
            }
            int condval_475;
            if ((((slab * 2) + m_12) == 0)) {
              condval_475 = 0;
            } else {
              int condval_476;
              if ((((slab * 2) + m_12) == 1)) {
                condval_476 = 3;
              } else {
                condval_476 = (((slab * 2) + m_12) - 1);
              }
              condval_475 = condval_476;
            }
            int condval_477;
            if ((((slab * 2) + m_12) == 0)) {
              condval_477 = 0;
            } else {
              int condval_478;
              if ((((slab * 2) + m_12) == 1)) {
                condval_478 = 3;
              } else {
                condval_478 = (((slab * 2) + m_12) - 1);
              }
              condval_477 = condval_478;
            }
            int condval_474;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_475 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_477 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_479;
              if ((((slab * 2) + m_12) == 0)) {
                condval_479 = 0;
              } else {
                int condval_480;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_480 = 3;
                } else {
                  condval_480 = (((slab * 2) + m_12) - 1);
                }
                condval_479 = condval_480;
              }
              int condval_481;
              if ((((slab * 2) + m_12) == 0)) {
                condval_481 = 0;
              } else {
                int condval_482;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_482 = 3;
                } else {
                  condval_482 = (((slab * 2) + m_12) - 1);
                }
                condval_481 = condval_482;
              }
              condval_474 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_479 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_481 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_483;
              if ((((slab * 2) + m_12) == 0)) {
                condval_483 = 0;
              } else {
                int condval_484;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_484 = 3;
                } else {
                  condval_484 = (((slab * 2) + m_12) - 1);
                }
                condval_483 = condval_484;
              }
              int condval_485;
              if ((((slab * 2) + m_12) == 0)) {
                condval_485 = 0;
              } else {
                int condval_486;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_486 = 3;
                } else {
                  condval_486 = (((slab * 2) + m_12) - 1);
                }
                condval_485 = condval_486;
              }
              condval_474 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_483 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_485 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_487;
            if ((((slab * 2) + m_12) == 0)) {
              condval_487 = 0;
            } else {
              int condval_488;
              if ((((slab * 2) + m_12) == 1)) {
                condval_488 = 3;
              } else {
                condval_488 = (((slab * 2) + m_12) - 1);
              }
              condval_487 = condval_488;
            }
            int condval_489;
            if ((((slab * 2) + m_12) == 0)) {
              condval_489 = 0;
            } else {
              int condval_490;
              if ((((slab * 2) + m_12) == 1)) {
                condval_490 = 3;
              } else {
                condval_490 = (((slab * 2) + m_12) - 1);
              }
              condval_489 = condval_490;
            }
            int condval_491;
            if ((((slab * 2) + m_12) == 0)) {
              condval_491 = 0;
            } else {
              int condval_492;
              if ((((slab * 2) + m_12) == 1)) {
                condval_492 = 3;
              } else {
                condval_492 = (((slab * 2) + m_12) - 1);
              }
              condval_491 = condval_492;
            }
            int condval_494;
            if ((((slab * 2) + m_12) == 0)) {
              condval_494 = 0;
            } else {
              int condval_495;
              if ((((slab * 2) + m_12) == 1)) {
                condval_495 = 3;
              } else {
                condval_495 = (((slab * 2) + m_12) - 1);
              }
              condval_494 = condval_495;
            }
            int condval_496;
            if ((((slab * 2) + m_12) == 0)) {
              condval_496 = 0;
            } else {
              int condval_497;
              if ((((slab * 2) + m_12) == 1)) {
                condval_497 = 3;
              } else {
                condval_497 = (((slab * 2) + m_12) - 1);
              }
              condval_496 = condval_497;
            }
            int condval_493;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_494 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_496 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_498;
              if ((((slab * 2) + m_12) == 0)) {
                condval_498 = 0;
              } else {
                int condval_499;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_499 = 3;
                } else {
                  condval_499 = (((slab * 2) + m_12) - 1);
                }
                condval_498 = condval_499;
              }
              int condval_500;
              if ((((slab * 2) + m_12) == 0)) {
                condval_500 = 0;
              } else {
                int condval_501;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_501 = 3;
                } else {
                  condval_501 = (((slab * 2) + m_12) - 1);
                }
                condval_500 = condval_501;
              }
              condval_493 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_498 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_500 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_502;
              if ((((slab * 2) + m_12) == 0)) {
                condval_502 = 0;
              } else {
                int condval_503;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_503 = 3;
                } else {
                  condval_503 = (((slab * 2) + m_12) - 1);
                }
                condval_502 = condval_503;
              }
              int condval_504;
              if ((((slab * 2) + m_12) == 0)) {
                condval_504 = 0;
              } else {
                int condval_505;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_505 = 3;
                } else {
                  condval_505 = (((slab * 2) + m_12) - 1);
                }
                condval_504 = condval_505;
              }
              condval_493 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_502 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_504 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_506;
            if ((((slab * 2) + m_12) == 0)) {
              condval_506 = 0;
            } else {
              int condval_507;
              if ((((slab * 2) + m_12) == 1)) {
                condval_507 = 3;
              } else {
                condval_507 = (((slab * 2) + m_12) - 1);
              }
              condval_506 = condval_507;
            }
            int condval_508;
            if ((((slab * 2) + m_12) == 0)) {
              condval_508 = 0;
            } else {
              int condval_509;
              if ((((slab * 2) + m_12) == 1)) {
                condval_509 = 3;
              } else {
                condval_509 = (((slab * 2) + m_12) - 1);
              }
              condval_508 = condval_509;
            }
            int condval_510;
            if ((((slab * 2) + m_12) == 0)) {
              condval_510 = 0;
            } else {
              int condval_511;
              if ((((slab * 2) + m_12) == 1)) {
                condval_511 = 3;
              } else {
                condval_511 = (((slab * 2) + m_12) - 1);
              }
              condval_510 = condval_511;
            }
            int condval_513;
            if ((((slab * 2) + m_12) == 0)) {
              condval_513 = 0;
            } else {
              int condval_514;
              if ((((slab * 2) + m_12) == 1)) {
                condval_514 = 3;
              } else {
                condval_514 = (((slab * 2) + m_12) - 1);
              }
              condval_513 = condval_514;
            }
            int condval_515;
            if ((((slab * 2) + m_12) == 0)) {
              condval_515 = 0;
            } else {
              int condval_516;
              if ((((slab * 2) + m_12) == 1)) {
                condval_516 = 3;
              } else {
                condval_516 = (((slab * 2) + m_12) - 1);
              }
              condval_515 = condval_516;
            }
            int condval_512;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_513 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_515 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_517;
              if ((((slab * 2) + m_12) == 0)) {
                condval_517 = 0;
              } else {
                int condval_518;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_518 = 3;
                } else {
                  condval_518 = (((slab * 2) + m_12) - 1);
                }
                condval_517 = condval_518;
              }
              int condval_519;
              if ((((slab * 2) + m_12) == 0)) {
                condval_519 = 0;
              } else {
                int condval_520;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_520 = 3;
                } else {
                  condval_520 = (((slab * 2) + m_12) - 1);
                }
                condval_519 = condval_520;
              }
              condval_512 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_517 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_519 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_521;
              if ((((slab * 2) + m_12) == 0)) {
                condval_521 = 0;
              } else {
                int condval_522;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_522 = 3;
                } else {
                  condval_522 = (((slab * 2) + m_12) - 1);
                }
                condval_521 = condval_522;
              }
              int condval_523;
              if ((((slab * 2) + m_12) == 0)) {
                condval_523 = 0;
              } else {
                int condval_524;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_524 = 3;
                } else {
                  condval_524 = (((slab * 2) + m_12) - 1);
                }
                condval_523 = condval_524;
              }
              condval_512 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_521 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_523 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_525;
            if ((((slab * 2) + m_12) == 0)) {
              condval_525 = 0;
            } else {
              int condval_526;
              if ((((slab * 2) + m_12) == 1)) {
                condval_526 = 3;
              } else {
                condval_526 = (((slab * 2) + m_12) - 1);
              }
              condval_525 = condval_526;
            }
            int condval_527;
            if ((((slab * 2) + m_12) == 0)) {
              condval_527 = 0;
            } else {
              int condval_528;
              if ((((slab * 2) + m_12) == 1)) {
                condval_528 = 3;
              } else {
                condval_528 = (((slab * 2) + m_12) - 1);
              }
              condval_527 = condval_528;
            }
            int condval_529;
            if ((((slab * 2) + m_12) == 0)) {
              condval_529 = 0;
            } else {
              int condval_530;
              if ((((slab * 2) + m_12) == 1)) {
                condval_530 = 3;
              } else {
                condval_530 = (((slab * 2) + m_12) - 1);
              }
              condval_529 = condval_530;
            }
            int condval_532;
            if ((((slab * 2) + m_12) == 0)) {
              condval_532 = 0;
            } else {
              int condval_533;
              if ((((slab * 2) + m_12) == 1)) {
                condval_533 = 3;
              } else {
                condval_533 = (((slab * 2) + m_12) - 1);
              }
              condval_532 = condval_533;
            }
            int condval_534;
            if ((((slab * 2) + m_12) == 0)) {
              condval_534 = 0;
            } else {
              int condval_535;
              if ((((slab * 2) + m_12) == 1)) {
                condval_535 = 3;
              } else {
                condval_535 = (((slab * 2) + m_12) - 1);
              }
              condval_534 = condval_535;
            }
            int condval_531;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_532 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_534 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_536;
              if ((((slab * 2) + m_12) == 0)) {
                condval_536 = 0;
              } else {
                int condval_537;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_537 = 3;
                } else {
                  condval_537 = (((slab * 2) + m_12) - 1);
                }
                condval_536 = condval_537;
              }
              int condval_538;
              if ((((slab * 2) + m_12) == 0)) {
                condval_538 = 0;
              } else {
                int condval_539;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_539 = 3;
                } else {
                  condval_539 = (((slab * 2) + m_12) - 1);
                }
                condval_538 = condval_539;
              }
              condval_531 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_536 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_538 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_540;
              if ((((slab * 2) + m_12) == 0)) {
                condval_540 = 0;
              } else {
                int condval_541;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_541 = 3;
                } else {
                  condval_541 = (((slab * 2) + m_12) - 1);
                }
                condval_540 = condval_541;
              }
              int condval_542;
              if ((((slab * 2) + m_12) == 0)) {
                condval_542 = 0;
              } else {
                int condval_543;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_543 = 3;
                } else {
                  condval_543 = (((slab * 2) + m_12) - 1);
                }
                condval_542 = condval_543;
              }
              condval_531 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_540 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_542 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_544;
            if ((((slab * 2) + m_12) == 0)) {
              condval_544 = 0;
            } else {
              int condval_545;
              if ((((slab * 2) + m_12) == 1)) {
                condval_545 = 3;
              } else {
                condval_545 = (((slab * 2) + m_12) - 1);
              }
              condval_544 = condval_545;
            }
            int condval_546;
            if ((((slab * 2) + m_12) == 0)) {
              condval_546 = 0;
            } else {
              int condval_547;
              if ((((slab * 2) + m_12) == 1)) {
                condval_547 = 3;
              } else {
                condval_547 = (((slab * 2) + m_12) - 1);
              }
              condval_546 = condval_547;
            }
            int condval_548;
            if ((((slab * 2) + m_12) == 0)) {
              condval_548 = 0;
            } else {
              int condval_549;
              if ((((slab * 2) + m_12) == 1)) {
                condval_549 = 3;
              } else {
                condval_549 = (((slab * 2) + m_12) - 1);
              }
              condval_548 = condval_549;
            }
            int condval_550;
            if ((((slab * 2) + m_12) == 0)) {
              condval_550 = 0;
            } else {
              int condval_551;
              if ((((slab * 2) + m_12) == 1)) {
                condval_551 = 3;
              } else {
                condval_551 = (((slab * 2) + m_12) - 1);
              }
              condval_550 = condval_551;
            }
            int condval_552;
            if ((((slab * 2) + m_12) == 0)) {
              condval_552 = 0;
            } else {
              int condval_553;
              if ((((slab * 2) + m_12) == 1)) {
                condval_553 = 3;
              } else {
                condval_553 = (((slab * 2) + m_12) - 1);
              }
              condval_552 = condval_553;
            }
            int condval_555;
            if ((((slab * 2) + m_12) == 0)) {
              condval_555 = 0;
            } else {
              int condval_556;
              if ((((slab * 2) + m_12) == 1)) {
                condval_556 = 3;
              } else {
                condval_556 = (((slab * 2) + m_12) - 1);
              }
              condval_555 = condval_556;
            }
            int condval_557;
            if ((((slab * 2) + m_12) == 0)) {
              condval_557 = 0;
            } else {
              int condval_558;
              if ((((slab * 2) + m_12) == 1)) {
                condval_558 = 3;
              } else {
                condval_558 = (((slab * 2) + m_12) - 1);
              }
              condval_557 = condval_558;
            }
            int condval_554;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_555 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_557 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_559;
              if ((((slab * 2) + m_12) == 0)) {
                condval_559 = 0;
              } else {
                int condval_560;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_560 = 3;
                } else {
                  condval_560 = (((slab * 2) + m_12) - 1);
                }
                condval_559 = condval_560;
              }
              int condval_561;
              if ((((slab * 2) + m_12) == 0)) {
                condval_561 = 0;
              } else {
                int condval_562;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_562 = 3;
                } else {
                  condval_562 = (((slab * 2) + m_12) - 1);
                }
                condval_561 = condval_562;
              }
              condval_554 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_559 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_561 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_563;
              if ((((slab * 2) + m_12) == 0)) {
                condval_563 = 0;
              } else {
                int condval_564;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_564 = 3;
                } else {
                  condval_564 = (((slab * 2) + m_12) - 1);
                }
                condval_563 = condval_564;
              }
              int condval_565;
              if ((((slab * 2) + m_12) == 0)) {
                condval_565 = 0;
              } else {
                int condval_566;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_566 = 3;
                } else {
                  condval_566 = (((slab * 2) + m_12) - 1);
                }
                condval_565 = condval_566;
              }
              condval_554 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_563 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_565 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_567;
            if ((((slab * 2) + m_12) == 0)) {
              condval_567 = 0;
            } else {
              int condval_568;
              if ((((slab * 2) + m_12) == 1)) {
                condval_568 = 3;
              } else {
                condval_568 = (((slab * 2) + m_12) - 1);
              }
              condval_567 = condval_568;
            }
            int condval_570;
            if ((((slab * 2) + m_12) == 0)) {
              condval_570 = 0;
            } else {
              int condval_571;
              if ((((slab * 2) + m_12) == 1)) {
                condval_571 = 3;
              } else {
                condval_571 = (((slab * 2) + m_12) - 1);
              }
              condval_570 = condval_571;
            }
            int condval_572;
            if ((((slab * 2) + m_12) == 0)) {
              condval_572 = 0;
            } else {
              int condval_573;
              if ((((slab * 2) + m_12) == 1)) {
                condval_573 = 3;
              } else {
                condval_573 = (((slab * 2) + m_12) - 1);
              }
              condval_572 = condval_573;
            }
            int condval_569;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_570 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_572 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_574;
              if ((((slab * 2) + m_12) == 0)) {
                condval_574 = 0;
              } else {
                int condval_575;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_575 = 3;
                } else {
                  condval_575 = (((slab * 2) + m_12) - 1);
                }
                condval_574 = condval_575;
              }
              int condval_576;
              if ((((slab * 2) + m_12) == 0)) {
                condval_576 = 0;
              } else {
                int condval_577;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_577 = 3;
                } else {
                  condval_577 = (((slab * 2) + m_12) - 1);
                }
                condval_576 = condval_577;
              }
              condval_569 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_574 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_576 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_578;
              if ((((slab * 2) + m_12) == 0)) {
                condval_578 = 0;
              } else {
                int condval_579;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_579 = 3;
                } else {
                  condval_579 = (((slab * 2) + m_12) - 1);
                }
                condval_578 = condval_579;
              }
              int condval_580;
              if ((((slab * 2) + m_12) == 0)) {
                condval_580 = 0;
              } else {
                int condval_581;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_581 = 3;
                } else {
                  condval_581 = (((slab * 2) + m_12) - 1);
                }
                condval_580 = condval_581;
              }
              condval_569 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_578 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_580 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_582;
            if ((((slab * 2) + m_12) == 0)) {
              condval_582 = 0;
            } else {
              int condval_583;
              if ((((slab * 2) + m_12) == 1)) {
                condval_583 = 3;
              } else {
                condval_583 = (((slab * 2) + m_12) - 1);
              }
              condval_582 = condval_583;
            }
            int condval_584;
            if ((((slab * 2) + m_12) == 0)) {
              condval_584 = 0;
            } else {
              int condval_585;
              if ((((slab * 2) + m_12) == 1)) {
                condval_585 = 3;
              } else {
                condval_585 = (((slab * 2) + m_12) - 1);
              }
              condval_584 = condval_585;
            }
            int condval_586;
            if ((((slab * 2) + m_12) == 0)) {
              condval_586 = 0;
            } else {
              int condval_587;
              if ((((slab * 2) + m_12) == 1)) {
                condval_587 = 3;
              } else {
                condval_587 = (((slab * 2) + m_12) - 1);
              }
              condval_586 = condval_587;
            }
            int condval_589;
            if ((((slab * 2) + m_12) == 0)) {
              condval_589 = 0;
            } else {
              int condval_590;
              if ((((slab * 2) + m_12) == 1)) {
                condval_590 = 3;
              } else {
                condval_590 = (((slab * 2) + m_12) - 1);
              }
              condval_589 = condval_590;
            }
            int condval_591;
            if ((((slab * 2) + m_12) == 0)) {
              condval_591 = 0;
            } else {
              int condval_592;
              if ((((slab * 2) + m_12) == 1)) {
                condval_592 = 3;
              } else {
                condval_592 = (((slab * 2) + m_12) - 1);
              }
              condval_591 = condval_592;
            }
            int condval_588;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_589 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_591 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_593;
              if ((((slab * 2) + m_12) == 0)) {
                condval_593 = 0;
              } else {
                int condval_594;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_594 = 3;
                } else {
                  condval_594 = (((slab * 2) + m_12) - 1);
                }
                condval_593 = condval_594;
              }
              int condval_595;
              if ((((slab * 2) + m_12) == 0)) {
                condval_595 = 0;
              } else {
                int condval_596;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_596 = 3;
                } else {
                  condval_596 = (((slab * 2) + m_12) - 1);
                }
                condval_595 = condval_596;
              }
              condval_588 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_593 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_595 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_597;
              if ((((slab * 2) + m_12) == 0)) {
                condval_597 = 0;
              } else {
                int condval_598;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_598 = 3;
                } else {
                  condval_598 = (((slab * 2) + m_12) - 1);
                }
                condval_597 = condval_598;
              }
              int condval_599;
              if ((((slab * 2) + m_12) == 0)) {
                condval_599 = 0;
              } else {
                int condval_600;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_600 = 3;
                } else {
                  condval_600 = (((slab * 2) + m_12) - 1);
                }
                condval_599 = condval_600;
              }
              condval_588 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_597 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_599 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_601;
            if ((((slab * 2) + m_12) == 0)) {
              condval_601 = 0;
            } else {
              int condval_602;
              if ((((slab * 2) + m_12) == 1)) {
                condval_602 = 3;
              } else {
                condval_602 = (((slab * 2) + m_12) - 1);
              }
              condval_601 = condval_602;
            }
            int condval_603;
            if ((((slab * 2) + m_12) == 0)) {
              condval_603 = 0;
            } else {
              int condval_604;
              if ((((slab * 2) + m_12) == 1)) {
                condval_604 = 3;
              } else {
                condval_604 = (((slab * 2) + m_12) - 1);
              }
              condval_603 = condval_604;
            }
            int condval_605;
            if ((((slab * 2) + m_12) == 0)) {
              condval_605 = 0;
            } else {
              int condval_606;
              if ((((slab * 2) + m_12) == 1)) {
                condval_606 = 3;
              } else {
                condval_606 = (((slab * 2) + m_12) - 1);
              }
              condval_605 = condval_606;
            }
            int condval_608;
            if ((((slab * 2) + m_12) == 0)) {
              condval_608 = 0;
            } else {
              int condval_609;
              if ((((slab * 2) + m_12) == 1)) {
                condval_609 = 3;
              } else {
                condval_609 = (((slab * 2) + m_12) - 1);
              }
              condval_608 = condval_609;
            }
            int condval_610;
            if ((((slab * 2) + m_12) == 0)) {
              condval_610 = 0;
            } else {
              int condval_611;
              if ((((slab * 2) + m_12) == 1)) {
                condval_611 = 3;
              } else {
                condval_611 = (((slab * 2) + m_12) - 1);
              }
              condval_610 = condval_611;
            }
            int condval_607;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_608 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_610 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_612;
              if ((((slab * 2) + m_12) == 0)) {
                condval_612 = 0;
              } else {
                int condval_613;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_613 = 3;
                } else {
                  condval_613 = (((slab * 2) + m_12) - 1);
                }
                condval_612 = condval_613;
              }
              int condval_614;
              if ((((slab * 2) + m_12) == 0)) {
                condval_614 = 0;
              } else {
                int condval_615;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_615 = 3;
                } else {
                  condval_615 = (((slab * 2) + m_12) - 1);
                }
                condval_614 = condval_615;
              }
              condval_607 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_612 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_614 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_616;
              if ((((slab * 2) + m_12) == 0)) {
                condval_616 = 0;
              } else {
                int condval_617;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_617 = 3;
                } else {
                  condval_617 = (((slab * 2) + m_12) - 1);
                }
                condval_616 = condval_617;
              }
              int condval_618;
              if ((((slab * 2) + m_12) == 0)) {
                condval_618 = 0;
              } else {
                int condval_619;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_619 = 3;
                } else {
                  condval_619 = (((slab * 2) + m_12) - 1);
                }
                condval_618 = condval_619;
              }
              condval_607 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_616 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_618 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_620;
            if ((((slab * 2) + m_12) == 0)) {
              condval_620 = 0;
            } else {
              int condval_621;
              if ((((slab * 2) + m_12) == 1)) {
                condval_621 = 3;
              } else {
                condval_621 = (((slab * 2) + m_12) - 1);
              }
              condval_620 = condval_621;
            }
            int condval_622;
            if ((((slab * 2) + m_12) == 0)) {
              condval_622 = 0;
            } else {
              int condval_623;
              if ((((slab * 2) + m_12) == 1)) {
                condval_623 = 3;
              } else {
                condval_623 = (((slab * 2) + m_12) - 1);
              }
              condval_622 = condval_623;
            }
            int condval_624;
            if ((((slab * 2) + m_12) == 0)) {
              condval_624 = 0;
            } else {
              int condval_625;
              if ((((slab * 2) + m_12) == 1)) {
                condval_625 = 3;
              } else {
                condval_625 = (((slab * 2) + m_12) - 1);
              }
              condval_624 = condval_625;
            }
            int condval_627;
            if ((((slab * 2) + m_12) == 0)) {
              condval_627 = 0;
            } else {
              int condval_628;
              if ((((slab * 2) + m_12) == 1)) {
                condval_628 = 3;
              } else {
                condval_628 = (((slab * 2) + m_12) - 1);
              }
              condval_627 = condval_628;
            }
            int condval_629;
            if ((((slab * 2) + m_12) == 0)) {
              condval_629 = 0;
            } else {
              int condval_630;
              if ((((slab * 2) + m_12) == 1)) {
                condval_630 = 3;
              } else {
                condval_630 = (((slab * 2) + m_12) - 1);
              }
              condval_629 = condval_630;
            }
            int condval_626;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_627 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_629 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_631;
              if ((((slab * 2) + m_12) == 0)) {
                condval_631 = 0;
              } else {
                int condval_632;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_632 = 3;
                } else {
                  condval_632 = (((slab * 2) + m_12) - 1);
                }
                condval_631 = condval_632;
              }
              int condval_633;
              if ((((slab * 2) + m_12) == 0)) {
                condval_633 = 0;
              } else {
                int condval_634;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_634 = 3;
                } else {
                  condval_634 = (((slab * 2) + m_12) - 1);
                }
                condval_633 = condval_634;
              }
              condval_626 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_631 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_633 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_635;
              if ((((slab * 2) + m_12) == 0)) {
                condval_635 = 0;
              } else {
                int condval_636;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_636 = 3;
                } else {
                  condval_636 = (((slab * 2) + m_12) - 1);
                }
                condval_635 = condval_636;
              }
              int condval_637;
              if ((((slab * 2) + m_12) == 0)) {
                condval_637 = 0;
              } else {
                int condval_638;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_638 = 3;
                } else {
                  condval_638 = (((slab * 2) + m_12) - 1);
                }
                condval_637 = condval_638;
              }
              condval_626 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_635 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_637 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_639;
            if ((((slab * 2) + m_12) == 0)) {
              condval_639 = 0;
            } else {
              int condval_640;
              if ((((slab * 2) + m_12) == 1)) {
                condval_640 = 3;
              } else {
                condval_640 = (((slab * 2) + m_12) - 1);
              }
              condval_639 = condval_640;
            }
            int condval_641;
            if ((((slab * 2) + m_12) == 0)) {
              condval_641 = 0;
            } else {
              int condval_642;
              if ((((slab * 2) + m_12) == 1)) {
                condval_642 = 3;
              } else {
                condval_642 = (((slab * 2) + m_12) - 1);
              }
              condval_641 = condval_642;
            }
            int condval_643;
            if ((((slab * 2) + m_12) == 0)) {
              condval_643 = 0;
            } else {
              int condval_644;
              if ((((slab * 2) + m_12) == 1)) {
                condval_644 = 3;
              } else {
                condval_644 = (((slab * 2) + m_12) - 1);
              }
              condval_643 = condval_644;
            }
            int condval_646;
            if ((((slab * 2) + m_12) == 0)) {
              condval_646 = 0;
            } else {
              int condval_647;
              if ((((slab * 2) + m_12) == 1)) {
                condval_647 = 3;
              } else {
                condval_647 = (((slab * 2) + m_12) - 1);
              }
              condval_646 = condval_647;
            }
            int condval_648;
            if ((((slab * 2) + m_12) == 0)) {
              condval_648 = 0;
            } else {
              int condval_649;
              if ((((slab * 2) + m_12) == 1)) {
                condval_649 = 3;
              } else {
                condval_649 = (((slab * 2) + m_12) - 1);
              }
              condval_648 = condval_649;
            }
            int condval_645;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_646 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_648 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_650;
              if ((((slab * 2) + m_12) == 0)) {
                condval_650 = 0;
              } else {
                int condval_651;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_651 = 3;
                } else {
                  condval_651 = (((slab * 2) + m_12) - 1);
                }
                condval_650 = condval_651;
              }
              int condval_652;
              if ((((slab * 2) + m_12) == 0)) {
                condval_652 = 0;
              } else {
                int condval_653;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_653 = 3;
                } else {
                  condval_653 = (((slab * 2) + m_12) - 1);
                }
                condval_652 = condval_653;
              }
              condval_645 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_650 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_652 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_654;
              if ((((slab * 2) + m_12) == 0)) {
                condval_654 = 0;
              } else {
                int condval_655;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_655 = 3;
                } else {
                  condval_655 = (((slab * 2) + m_12) - 1);
                }
                condval_654 = condval_655;
              }
              int condval_656;
              if ((((slab * 2) + m_12) == 0)) {
                condval_656 = 0;
              } else {
                int condval_657;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_657 = 3;
                } else {
                  condval_657 = (((slab * 2) + m_12) - 1);
                }
                condval_656 = condval_657;
              }
              condval_645 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_654 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_656 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_658;
            if ((((slab * 2) + m_12) == 0)) {
              condval_658 = 0;
            } else {
              int condval_659;
              if ((((slab * 2) + m_12) == 1)) {
                condval_659 = 3;
              } else {
                condval_659 = (((slab * 2) + m_12) - 1);
              }
              condval_658 = condval_659;
            }
            int condval_660;
            if ((((slab * 2) + m_12) == 0)) {
              condval_660 = 0;
            } else {
              int condval_661;
              if ((((slab * 2) + m_12) == 1)) {
                condval_661 = 3;
              } else {
                condval_661 = (((slab * 2) + m_12) - 1);
              }
              condval_660 = condval_661;
            }
            int condval_662;
            if ((((slab * 2) + m_12) == 0)) {
              condval_662 = 0;
            } else {
              int condval_663;
              if ((((slab * 2) + m_12) == 1)) {
                condval_663 = 3;
              } else {
                condval_663 = (((slab * 2) + m_12) - 1);
              }
              condval_662 = condval_663;
            }
            int condval_665;
            if ((((slab * 2) + m_12) == 0)) {
              condval_665 = 0;
            } else {
              int condval_666;
              if ((((slab * 2) + m_12) == 1)) {
                condval_666 = 3;
              } else {
                condval_666 = (((slab * 2) + m_12) - 1);
              }
              condval_665 = condval_666;
            }
            int condval_667;
            if ((((slab * 2) + m_12) == 0)) {
              condval_667 = 0;
            } else {
              int condval_668;
              if ((((slab * 2) + m_12) == 1)) {
                condval_668 = 3;
              } else {
                condval_668 = (((slab * 2) + m_12) - 1);
              }
              condval_667 = condval_668;
            }
            int condval_664;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_665 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_667 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_669;
              if ((((slab * 2) + m_12) == 0)) {
                condval_669 = 0;
              } else {
                int condval_670;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_670 = 3;
                } else {
                  condval_670 = (((slab * 2) + m_12) - 1);
                }
                condval_669 = condval_670;
              }
              int condval_671;
              if ((((slab * 2) + m_12) == 0)) {
                condval_671 = 0;
              } else {
                int condval_672;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_672 = 3;
                } else {
                  condval_672 = (((slab * 2) + m_12) - 1);
                }
                condval_671 = condval_672;
              }
              condval_664 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_669 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_671 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_673;
              if ((((slab * 2) + m_12) == 0)) {
                condval_673 = 0;
              } else {
                int condval_674;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_674 = 3;
                } else {
                  condval_674 = (((slab * 2) + m_12) - 1);
                }
                condval_673 = condval_674;
              }
              int condval_675;
              if ((((slab * 2) + m_12) == 0)) {
                condval_675 = 0;
              } else {
                int condval_676;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_676 = 3;
                } else {
                  condval_676 = (((slab * 2) + m_12) - 1);
                }
                condval_675 = condval_676;
              }
              condval_664 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_673 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_675 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_677;
            if ((((slab * 2) + m_12) == 0)) {
              condval_677 = 0;
            } else {
              int condval_678;
              if ((((slab * 2) + m_12) == 1)) {
                condval_678 = 3;
              } else {
                condval_678 = (((slab * 2) + m_12) - 1);
              }
              condval_677 = condval_678;
            }
            int condval_679;
            if ((((slab * 2) + m_12) == 0)) {
              condval_679 = 0;
            } else {
              int condval_680;
              if ((((slab * 2) + m_12) == 1)) {
                condval_680 = 3;
              } else {
                condval_680 = (((slab * 2) + m_12) - 1);
              }
              condval_679 = condval_680;
            }
            int condval_681;
            if ((((slab * 2) + m_12) == 0)) {
              condval_681 = 0;
            } else {
              int condval_682;
              if ((((slab * 2) + m_12) == 1)) {
                condval_682 = 3;
              } else {
                condval_682 = (((slab * 2) + m_12) - 1);
              }
              condval_681 = condval_682;
            }
            int condval_683;
            if ((((slab * 2) + m_12) == 0)) {
              condval_683 = 0;
            } else {
              int condval_684;
              if ((((slab * 2) + m_12) == 1)) {
                condval_684 = 3;
              } else {
                condval_684 = (((slab * 2) + m_12) - 1);
              }
              condval_683 = condval_684;
            }
            int condval_685;
            if ((((slab * 2) + m_12) == 0)) {
              condval_685 = 0;
            } else {
              int condval_686;
              if ((((slab * 2) + m_12) == 1)) {
                condval_686 = 3;
              } else {
                condval_686 = (((slab * 2) + m_12) - 1);
              }
              condval_685 = condval_686;
            }
            int condval_687;
            if ((((slab * 2) + m_12) == 0)) {
              condval_687 = 0;
            } else {
              int condval_688;
              if ((((slab * 2) + m_12) == 1)) {
                condval_688 = 3;
              } else {
                condval_688 = (((slab * 2) + m_12) - 1);
              }
              condval_687 = condval_688;
            }
            int condval_689;
            if ((((slab * 2) + m_12) == 0)) {
              condval_689 = 0;
            } else {
              int condval_690;
              if ((((slab * 2) + m_12) == 1)) {
                condval_690 = 3;
              } else {
                condval_690 = (((slab * 2) + m_12) - 1);
              }
              condval_689 = condval_690;
            }
            int condval_691;
            if ((((slab * 2) + m_12) == 0)) {
              condval_691 = 0;
            } else {
              int condval_692;
              if ((((slab * 2) + m_12) == 1)) {
                condval_692 = 3;
              } else {
                condval_692 = (((slab * 2) + m_12) - 1);
              }
              condval_691 = condval_692;
            }
            int condval_693;
            if ((((slab * 2) + m_12) == 0)) {
              condval_693 = 0;
            } else {
              int condval_694;
              if ((((slab * 2) + m_12) == 1)) {
                condval_694 = 3;
              } else {
                condval_694 = (((slab * 2) + m_12) - 1);
              }
              condval_693 = condval_694;
            }
            int condval_695;
            if ((((slab * 2) + m_12) == 0)) {
              condval_695 = 0;
            } else {
              int condval_696;
              if ((((slab * 2) + m_12) == 1)) {
                condval_696 = 3;
              } else {
                condval_696 = (((slab * 2) + m_12) - 1);
              }
              condval_695 = condval_696;
            }
            int condval_697;
            if ((((slab * 2) + m_12) == 0)) {
              condval_697 = 0;
            } else {
              int condval_698;
              if ((((slab * 2) + m_12) == 1)) {
                condval_698 = 3;
              } else {
                condval_698 = (((slab * 2) + m_12) - 1);
              }
              condval_697 = condval_698;
            }
            int condval_700;
            if ((((slab * 2) + m_12) == 0)) {
              condval_700 = 0;
            } else {
              int condval_701;
              if ((((slab * 2) + m_12) == 1)) {
                condval_701 = 3;
              } else {
                condval_701 = (((slab * 2) + m_12) - 1);
              }
              condval_700 = condval_701;
            }
            int condval_702;
            if ((((slab * 2) + m_12) == 0)) {
              condval_702 = 0;
            } else {
              int condval_703;
              if ((((slab * 2) + m_12) == 1)) {
                condval_703 = 3;
              } else {
                condval_703 = (((slab * 2) + m_12) - 1);
              }
              condval_702 = condval_703;
            }
            int condval_699;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_700 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_702 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_704;
              if ((((slab * 2) + m_12) == 0)) {
                condval_704 = 0;
              } else {
                int condval_705;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_705 = 3;
                } else {
                  condval_705 = (((slab * 2) + m_12) - 1);
                }
                condval_704 = condval_705;
              }
              int condval_706;
              if ((((slab * 2) + m_12) == 0)) {
                condval_706 = 0;
              } else {
                int condval_707;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_707 = 3;
                } else {
                  condval_707 = (((slab * 2) + m_12) - 1);
                }
                condval_706 = condval_707;
              }
              condval_699 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_704 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_706 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_708;
              if ((((slab * 2) + m_12) == 0)) {
                condval_708 = 0;
              } else {
                int condval_709;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_709 = 3;
                } else {
                  condval_709 = (((slab * 2) + m_12) - 1);
                }
                condval_708 = condval_709;
              }
              int condval_710;
              if ((((slab * 2) + m_12) == 0)) {
                condval_710 = 0;
              } else {
                int condval_711;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_711 = 3;
                } else {
                  condval_711 = (((slab * 2) + m_12) - 1);
                }
                condval_710 = condval_711;
              }
              condval_699 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_708 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_710 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_712;
            if ((((slab * 2) + m_12) == 0)) {
              condval_712 = 0;
            } else {
              int condval_713;
              if ((((slab * 2) + m_12) == 1)) {
                condval_713 = 3;
              } else {
                condval_713 = (((slab * 2) + m_12) - 1);
              }
              condval_712 = condval_713;
            }
            int condval_714;
            if ((((slab * 2) + m_12) == 0)) {
              condval_714 = 0;
            } else {
              int condval_715;
              if ((((slab * 2) + m_12) == 1)) {
                condval_715 = 3;
              } else {
                condval_715 = (((slab * 2) + m_12) - 1);
              }
              condval_714 = condval_715;
            }
            int condval_716;
            if ((((slab * 2) + m_12) == 0)) {
              condval_716 = 0;
            } else {
              int condval_717;
              if ((((slab * 2) + m_12) == 1)) {
                condval_717 = 3;
              } else {
                condval_717 = (((slab * 2) + m_12) - 1);
              }
              condval_716 = condval_717;
            }
            int condval_719;
            if ((((slab * 2) + m_12) == 0)) {
              condval_719 = 0;
            } else {
              int condval_720;
              if ((((slab * 2) + m_12) == 1)) {
                condval_720 = 3;
              } else {
                condval_720 = (((slab * 2) + m_12) - 1);
              }
              condval_719 = condval_720;
            }
            int condval_721;
            if ((((slab * 2) + m_12) == 0)) {
              condval_721 = 0;
            } else {
              int condval_722;
              if ((((slab * 2) + m_12) == 1)) {
                condval_722 = 3;
              } else {
                condval_722 = (((slab * 2) + m_12) - 1);
              }
              condval_721 = condval_722;
            }
            int condval_718;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_719 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_721 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_723;
              if ((((slab * 2) + m_12) == 0)) {
                condval_723 = 0;
              } else {
                int condval_724;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_724 = 3;
                } else {
                  condval_724 = (((slab * 2) + m_12) - 1);
                }
                condval_723 = condval_724;
              }
              int condval_725;
              if ((((slab * 2) + m_12) == 0)) {
                condval_725 = 0;
              } else {
                int condval_726;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_726 = 3;
                } else {
                  condval_726 = (((slab * 2) + m_12) - 1);
                }
                condval_725 = condval_726;
              }
              condval_718 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_723 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_725 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_727;
              if ((((slab * 2) + m_12) == 0)) {
                condval_727 = 0;
              } else {
                int condval_728;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_728 = 3;
                } else {
                  condval_728 = (((slab * 2) + m_12) - 1);
                }
                condval_727 = condval_728;
              }
              int condval_729;
              if ((((slab * 2) + m_12) == 0)) {
                condval_729 = 0;
              } else {
                int condval_730;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_730 = 3;
                } else {
                  condval_730 = (((slab * 2) + m_12) - 1);
                }
                condval_729 = condval_730;
              }
              condval_718 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_727 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_729 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_731;
            if ((((slab * 2) + m_12) == 0)) {
              condval_731 = 0;
            } else {
              int condval_732;
              if ((((slab * 2) + m_12) == 1)) {
                condval_732 = 3;
              } else {
                condval_732 = (((slab * 2) + m_12) - 1);
              }
              condval_731 = condval_732;
            }
            int condval_733;
            if ((((slab * 2) + m_12) == 0)) {
              condval_733 = 0;
            } else {
              int condval_734;
              if ((((slab * 2) + m_12) == 1)) {
                condval_734 = 3;
              } else {
                condval_734 = (((slab * 2) + m_12) - 1);
              }
              condval_733 = condval_734;
            }
            int condval_735;
            if ((((slab * 2) + m_12) == 0)) {
              condval_735 = 0;
            } else {
              int condval_736;
              if ((((slab * 2) + m_12) == 1)) {
                condval_736 = 3;
              } else {
                condval_736 = (((slab * 2) + m_12) - 1);
              }
              condval_735 = condval_736;
            }
            int condval_738;
            if ((((slab * 2) + m_12) == 0)) {
              condval_738 = 0;
            } else {
              int condval_739;
              if ((((slab * 2) + m_12) == 1)) {
                condval_739 = 3;
              } else {
                condval_739 = (((slab * 2) + m_12) - 1);
              }
              condval_738 = condval_739;
            }
            int condval_740;
            if ((((slab * 2) + m_12) == 0)) {
              condval_740 = 0;
            } else {
              int condval_741;
              if ((((slab * 2) + m_12) == 1)) {
                condval_741 = 3;
              } else {
                condval_741 = (((slab * 2) + m_12) - 1);
              }
              condval_740 = condval_741;
            }
            int condval_737;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_738 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_740 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_742;
              if ((((slab * 2) + m_12) == 0)) {
                condval_742 = 0;
              } else {
                int condval_743;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_743 = 3;
                } else {
                  condval_743 = (((slab * 2) + m_12) - 1);
                }
                condval_742 = condval_743;
              }
              int condval_744;
              if ((((slab * 2) + m_12) == 0)) {
                condval_744 = 0;
              } else {
                int condval_745;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_745 = 3;
                } else {
                  condval_745 = (((slab * 2) + m_12) - 1);
                }
                condval_744 = condval_745;
              }
              condval_737 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_742 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_744 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_746;
              if ((((slab * 2) + m_12) == 0)) {
                condval_746 = 0;
              } else {
                int condval_747;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_747 = 3;
                } else {
                  condval_747 = (((slab * 2) + m_12) - 1);
                }
                condval_746 = condval_747;
              }
              int condval_748;
              if ((((slab * 2) + m_12) == 0)) {
                condval_748 = 0;
              } else {
                int condval_749;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_749 = 3;
                } else {
                  condval_749 = (((slab * 2) + m_12) - 1);
                }
                condval_748 = condval_749;
              }
              condval_737 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_746 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_748 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_750;
            if ((((slab * 2) + m_12) == 0)) {
              condval_750 = 0;
            } else {
              int condval_751;
              if ((((slab * 2) + m_12) == 1)) {
                condval_751 = 3;
              } else {
                condval_751 = (((slab * 2) + m_12) - 1);
              }
              condval_750 = condval_751;
            }
            int condval_752;
            if ((((slab * 2) + m_12) == 0)) {
              condval_752 = 0;
            } else {
              int condval_753;
              if ((((slab * 2) + m_12) == 1)) {
                condval_753 = 3;
              } else {
                condval_753 = (((slab * 2) + m_12) - 1);
              }
              condval_752 = condval_753;
            }
            int condval_754;
            if ((((slab * 2) + m_12) == 0)) {
              condval_754 = 0;
            } else {
              int condval_755;
              if ((((slab * 2) + m_12) == 1)) {
                condval_755 = 3;
              } else {
                condval_755 = (((slab * 2) + m_12) - 1);
              }
              condval_754 = condval_755;
            }
            int condval_757;
            if ((((slab * 2) + m_12) == 0)) {
              condval_757 = 0;
            } else {
              int condval_758;
              if ((((slab * 2) + m_12) == 1)) {
                condval_758 = 3;
              } else {
                condval_758 = (((slab * 2) + m_12) - 1);
              }
              condval_757 = condval_758;
            }
            int condval_759;
            if ((((slab * 2) + m_12) == 0)) {
              condval_759 = 0;
            } else {
              int condval_760;
              if ((((slab * 2) + m_12) == 1)) {
                condval_760 = 3;
              } else {
                condval_760 = (((slab * 2) + m_12) - 1);
              }
              condval_759 = condval_760;
            }
            int condval_756;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_757 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_759 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_761;
              if ((((slab * 2) + m_12) == 0)) {
                condval_761 = 0;
              } else {
                int condval_762;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_762 = 3;
                } else {
                  condval_762 = (((slab * 2) + m_12) - 1);
                }
                condval_761 = condval_762;
              }
              int condval_763;
              if ((((slab * 2) + m_12) == 0)) {
                condval_763 = 0;
              } else {
                int condval_764;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_764 = 3;
                } else {
                  condval_764 = (((slab * 2) + m_12) - 1);
                }
                condval_763 = condval_764;
              }
              condval_756 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_761 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_763 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_765;
              if ((((slab * 2) + m_12) == 0)) {
                condval_765 = 0;
              } else {
                int condval_766;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_766 = 3;
                } else {
                  condval_766 = (((slab * 2) + m_12) - 1);
                }
                condval_765 = condval_766;
              }
              int condval_767;
              if ((((slab * 2) + m_12) == 0)) {
                condval_767 = 0;
              } else {
                int condval_768;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_768 = 3;
                } else {
                  condval_768 = (((slab * 2) + m_12) - 1);
                }
                condval_767 = condval_768;
              }
              condval_756 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_765 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_767 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_769;
            if ((((slab * 2) + m_12) == 0)) {
              condval_769 = 0;
            } else {
              int condval_770;
              if ((((slab * 2) + m_12) == 1)) {
                condval_770 = 3;
              } else {
                condval_770 = (((slab * 2) + m_12) - 1);
              }
              condval_769 = condval_770;
            }
            int condval_771;
            if ((((slab * 2) + m_12) == 0)) {
              condval_771 = 0;
            } else {
              int condval_772;
              if ((((slab * 2) + m_12) == 1)) {
                condval_772 = 3;
              } else {
                condval_772 = (((slab * 2) + m_12) - 1);
              }
              condval_771 = condval_772;
            }
            int condval_773;
            if ((((slab * 2) + m_12) == 0)) {
              condval_773 = 0;
            } else {
              int condval_774;
              if ((((slab * 2) + m_12) == 1)) {
                condval_774 = 3;
              } else {
                condval_774 = (((slab * 2) + m_12) - 1);
              }
              condval_773 = condval_774;
            }
            int condval_776;
            if ((((slab * 2) + m_12) == 0)) {
              condval_776 = 0;
            } else {
              int condval_777;
              if ((((slab * 2) + m_12) == 1)) {
                condval_777 = 3;
              } else {
                condval_777 = (((slab * 2) + m_12) - 1);
              }
              condval_776 = condval_777;
            }
            int condval_778;
            if ((((slab * 2) + m_12) == 0)) {
              condval_778 = 0;
            } else {
              int condval_779;
              if ((((slab * 2) + m_12) == 1)) {
                condval_779 = 3;
              } else {
                condval_779 = (((slab * 2) + m_12) - 1);
              }
              condval_778 = condval_779;
            }
            int condval_775;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_776 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_778 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_780;
              if ((((slab * 2) + m_12) == 0)) {
                condval_780 = 0;
              } else {
                int condval_781;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_781 = 3;
                } else {
                  condval_781 = (((slab * 2) + m_12) - 1);
                }
                condval_780 = condval_781;
              }
              int condval_782;
              if ((((slab * 2) + m_12) == 0)) {
                condval_782 = 0;
              } else {
                int condval_783;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_783 = 3;
                } else {
                  condval_783 = (((slab * 2) + m_12) - 1);
                }
                condval_782 = condval_783;
              }
              condval_775 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_780 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_782 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_784;
              if ((((slab * 2) + m_12) == 0)) {
                condval_784 = 0;
              } else {
                int condval_785;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_785 = 3;
                } else {
                  condval_785 = (((slab * 2) + m_12) - 1);
                }
                condval_784 = condval_785;
              }
              int condval_786;
              if ((((slab * 2) + m_12) == 0)) {
                condval_786 = 0;
              } else {
                int condval_787;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_787 = 3;
                } else {
                  condval_787 = (((slab * 2) + m_12) - 1);
                }
                condval_786 = condval_787;
              }
              condval_775 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_784 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_786 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_788;
            if ((((slab * 2) + m_12) == 0)) {
              condval_788 = 0;
            } else {
              int condval_789;
              if ((((slab * 2) + m_12) == 1)) {
                condval_789 = 3;
              } else {
                condval_789 = (((slab * 2) + m_12) - 1);
              }
              condval_788 = condval_789;
            }
            int condval_790;
            if ((((slab * 2) + m_12) == 0)) {
              condval_790 = 0;
            } else {
              int condval_791;
              if ((((slab * 2) + m_12) == 1)) {
                condval_791 = 3;
              } else {
                condval_791 = (((slab * 2) + m_12) - 1);
              }
              condval_790 = condval_791;
            }
            int condval_792;
            if ((((slab * 2) + m_12) == 0)) {
              condval_792 = 0;
            } else {
              int condval_793;
              if ((((slab * 2) + m_12) == 1)) {
                condval_793 = 3;
              } else {
                condval_793 = (((slab * 2) + m_12) - 1);
              }
              condval_792 = condval_793;
            }
            int condval_795;
            if ((((slab * 2) + m_12) == 0)) {
              condval_795 = 0;
            } else {
              int condval_796;
              if ((((slab * 2) + m_12) == 1)) {
                condval_796 = 3;
              } else {
                condval_796 = (((slab * 2) + m_12) - 1);
              }
              condval_795 = condval_796;
            }
            int condval_797;
            if ((((slab * 2) + m_12) == 0)) {
              condval_797 = 0;
            } else {
              int condval_798;
              if ((((slab * 2) + m_12) == 1)) {
                condval_798 = 3;
              } else {
                condval_798 = (((slab * 2) + m_12) - 1);
              }
              condval_797 = condval_798;
            }
            int condval_794;
            if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_795 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_797 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
              int condval_799;
              if ((((slab * 2) + m_12) == 0)) {
                condval_799 = 0;
              } else {
                int condval_800;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_800 = 3;
                } else {
                  condval_800 = (((slab * 2) + m_12) - 1);
                }
                condval_799 = condval_800;
              }
              int condval_801;
              if ((((slab * 2) + m_12) == 0)) {
                condval_801 = 0;
              } else {
                int condval_802;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_802 = 3;
                } else {
                  condval_802 = (((slab * 2) + m_12) - 1);
                }
                condval_801 = condval_802;
              }
              condval_794 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_799 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_801 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
            } else {
              int condval_803;
              if ((((slab * 2) + m_12) == 0)) {
                condval_803 = 0;
              } else {
                int condval_804;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_804 = 3;
                } else {
                  condval_804 = (((slab * 2) + m_12) - 1);
                }
                condval_803 = condval_804;
              }
              int condval_805;
              if ((((slab * 2) + m_12) == 0)) {
                condval_805 = 0;
              } else {
                int condval_806;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_806 = 3;
                } else {
                  condval_806 = (((slab * 2) + m_12) - 1);
                }
                condval_805 = condval_806;
              }
              condval_794 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_803 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_805 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
            }
            int condval_807;
            if ((((slab * 2) + m_12) == 0)) {
              condval_807 = 0;
            } else {
              int condval_808;
              if ((((slab * 2) + m_12) == 1)) {
                condval_808 = 3;
              } else {
                condval_808 = (((slab * 2) + m_12) - 1);
              }
              condval_807 = condval_808;
            }
            int condval_809;
            if ((((slab * 2) + m_12) == 0)) {
              condval_809 = 0;
            } else {
              int condval_810;
              if ((((slab * 2) + m_12) == 1)) {
                condval_810 = 3;
              } else {
                condval_810 = (((slab * 2) + m_12) - 1);
              }
              condval_809 = condval_810;
            }
            int condval_811;
            if ((((slab * 2) + m_12) == 0)) {
              condval_811 = 0;
            } else {
              int condval_812;
              if ((((slab * 2) + m_12) == 1)) {
                condval_812 = 3;
              } else {
                condval_812 = (((slab * 2) + m_12) - 1);
              }
              condval_811 = condval_812;
            }
            condval_427 = (((((((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_436 & 3) * 16) + (((((condval_449 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_451 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_453 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_455 & 3) * 16) + (((((condval_468 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_470 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_472 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_474 & 3) * 16) + (((((condval_487 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_489 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_491 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_493 & 3) * 16) + (((((condval_506 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_508 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_510 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_512 & 3) * 16) + (((((condval_525 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_527 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_529 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_531 & 3) * 16) + (((((condval_544 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_546 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_548 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + ((((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_550 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_552 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_554 * 2)) + ((((condval_567 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) * 40960)) + (((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_569 & 3) * 16) + (((((condval_582 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_584 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_586 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_588 & 3) * 16) + (((((condval_601 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_603 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_605 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_607 & 3) * 16) + (((((condval_620 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_622 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_624 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_626 & 3) * 16) + (((((condval_639 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_641 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_643 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_645 & 3) * 16) + (((((condval_658 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_660 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_662 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_664 & 3) * 16) + (((((condval_677 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_679 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_681 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_683 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_685 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_687 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_689 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_691 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_693 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_695 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_697 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_699 & 3) * 16) + (((((condval_712 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_714 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_716 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_718 & 3) * 16) + (((((condval_731 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_733 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_735 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_737 & 3) * 16) + (((((condval_750 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_752 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_754 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_756 & 3) * 16) + (((((condval_769 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_771 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_773 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_775 & 3) * 16) + (((((condval_788 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_790 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_792 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_794 & 3) * 16) + (((((condval_807 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_809 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_811 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
          } else {
            condval_427 = -1;
          }
          if (condval_427 < 2850208) {
            int condval_814;
            if ((((slab * 2) + m_12) == 0)) {
              condval_814 = 0;
            } else {
              int condval_815;
              if ((((slab * 2) + m_12) == 1)) {
                condval_815 = 3;
              } else {
                condval_815 = (((slab * 2) + m_12) - 1);
              }
              condval_814 = condval_815;
            }
            int condval_816;
            if ((((slab * 2) + m_12) == 0)) {
              condval_816 = 0;
            } else {
              int condval_817;
              if ((((slab * 2) + m_12) == 1)) {
                condval_817 = 3;
              } else {
                condval_817 = (((slab * 2) + m_12) - 1);
              }
              condval_816 = condval_817;
            }
            int condval_818;
            if ((((slab * 2) + m_12) == 0)) {
              condval_818 = 0;
            } else {
              int condval_819;
              if ((((slab * 2) + m_12) == 1)) {
                condval_819 = 3;
              } else {
                condval_819 = (((slab * 2) + m_12) - 1);
              }
              condval_818 = condval_819;
            }
            int condval_820;
            if ((((slab * 2) + m_12) == 0)) {
              condval_820 = 0;
            } else {
              int condval_821;
              if ((((slab * 2) + m_12) == 1)) {
                condval_821 = 3;
              } else {
                condval_821 = (((slab * 2) + m_12) - 1);
              }
              condval_820 = condval_821;
            }
            int condval_813;
            if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_814 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_816 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_818 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_820 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
              int condval_823;
              if ((((slab * 2) + m_12) == 0)) {
                condval_823 = 0;
              } else {
                int condval_824;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_824 = 3;
                } else {
                  condval_824 = (((slab * 2) + m_12) - 1);
                }
                condval_823 = condval_824;
              }
              int condval_825;
              if ((((slab * 2) + m_12) == 0)) {
                condval_825 = 0;
              } else {
                int condval_826;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_826 = 3;
                } else {
                  condval_826 = (((slab * 2) + m_12) - 1);
                }
                condval_825 = condval_826;
              }
              int condval_822;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_823 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_825 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_827;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_827 = 0;
                } else {
                  int condval_828;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_828 = 3;
                  } else {
                    condval_828 = (((slab * 2) + m_12) - 1);
                  }
                  condval_827 = condval_828;
                }
                int condval_829;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_829 = 0;
                } else {
                  int condval_830;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_830 = 3;
                  } else {
                    condval_830 = (((slab * 2) + m_12) - 1);
                  }
                  condval_829 = condval_830;
                }
                condval_822 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_827 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_829 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_831;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_831 = 0;
                } else {
                  int condval_832;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_832 = 3;
                  } else {
                    condval_832 = (((slab * 2) + m_12) - 1);
                  }
                  condval_831 = condval_832;
                }
                int condval_833;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_833 = 0;
                } else {
                  int condval_834;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_834 = 3;
                  } else {
                    condval_834 = (((slab * 2) + m_12) - 1);
                  }
                  condval_833 = condval_834;
                }
                condval_822 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_831 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_833 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_835;
              if ((((slab * 2) + m_12) == 0)) {
                condval_835 = 0;
              } else {
                int condval_836;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_836 = 3;
                } else {
                  condval_836 = (((slab * 2) + m_12) - 1);
                }
                condval_835 = condval_836;
              }
              int condval_837;
              if ((((slab * 2) + m_12) == 0)) {
                condval_837 = 0;
              } else {
                int condval_838;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_838 = 3;
                } else {
                  condval_838 = (((slab * 2) + m_12) - 1);
                }
                condval_837 = condval_838;
              }
              int condval_839;
              if ((((slab * 2) + m_12) == 0)) {
                condval_839 = 0;
              } else {
                int condval_840;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_840 = 3;
                } else {
                  condval_840 = (((slab * 2) + m_12) - 1);
                }
                condval_839 = condval_840;
              }
              int condval_842;
              if ((((slab * 2) + m_12) == 0)) {
                condval_842 = 0;
              } else {
                int condval_843;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_843 = 3;
                } else {
                  condval_843 = (((slab * 2) + m_12) - 1);
                }
                condval_842 = condval_843;
              }
              int condval_844;
              if ((((slab * 2) + m_12) == 0)) {
                condval_844 = 0;
              } else {
                int condval_845;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_845 = 3;
                } else {
                  condval_845 = (((slab * 2) + m_12) - 1);
                }
                condval_844 = condval_845;
              }
              int condval_841;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_842 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_844 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_846;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_846 = 0;
                } else {
                  int condval_847;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_847 = 3;
                  } else {
                    condval_847 = (((slab * 2) + m_12) - 1);
                  }
                  condval_846 = condval_847;
                }
                int condval_848;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_848 = 0;
                } else {
                  int condval_849;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_849 = 3;
                  } else {
                    condval_849 = (((slab * 2) + m_12) - 1);
                  }
                  condval_848 = condval_849;
                }
                condval_841 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_846 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_848 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_850;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_850 = 0;
                } else {
                  int condval_851;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_851 = 3;
                  } else {
                    condval_851 = (((slab * 2) + m_12) - 1);
                  }
                  condval_850 = condval_851;
                }
                int condval_852;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_852 = 0;
                } else {
                  int condval_853;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_853 = 3;
                  } else {
                    condval_853 = (((slab * 2) + m_12) - 1);
                  }
                  condval_852 = condval_853;
                }
                condval_841 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_850 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_852 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_854;
              if ((((slab * 2) + m_12) == 0)) {
                condval_854 = 0;
              } else {
                int condval_855;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_855 = 3;
                } else {
                  condval_855 = (((slab * 2) + m_12) - 1);
                }
                condval_854 = condval_855;
              }
              int condval_856;
              if ((((slab * 2) + m_12) == 0)) {
                condval_856 = 0;
              } else {
                int condval_857;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_857 = 3;
                } else {
                  condval_857 = (((slab * 2) + m_12) - 1);
                }
                condval_856 = condval_857;
              }
              int condval_858;
              if ((((slab * 2) + m_12) == 0)) {
                condval_858 = 0;
              } else {
                int condval_859;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_859 = 3;
                } else {
                  condval_859 = (((slab * 2) + m_12) - 1);
                }
                condval_858 = condval_859;
              }
              int condval_861;
              if ((((slab * 2) + m_12) == 0)) {
                condval_861 = 0;
              } else {
                int condval_862;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_862 = 3;
                } else {
                  condval_862 = (((slab * 2) + m_12) - 1);
                }
                condval_861 = condval_862;
              }
              int condval_863;
              if ((((slab * 2) + m_12) == 0)) {
                condval_863 = 0;
              } else {
                int condval_864;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_864 = 3;
                } else {
                  condval_864 = (((slab * 2) + m_12) - 1);
                }
                condval_863 = condval_864;
              }
              int condval_860;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_861 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_863 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_865;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_865 = 0;
                } else {
                  int condval_866;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_866 = 3;
                  } else {
                    condval_866 = (((slab * 2) + m_12) - 1);
                  }
                  condval_865 = condval_866;
                }
                int condval_867;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_867 = 0;
                } else {
                  int condval_868;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_868 = 3;
                  } else {
                    condval_868 = (((slab * 2) + m_12) - 1);
                  }
                  condval_867 = condval_868;
                }
                condval_860 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_865 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_867 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_869;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_869 = 0;
                } else {
                  int condval_870;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_870 = 3;
                  } else {
                    condval_870 = (((slab * 2) + m_12) - 1);
                  }
                  condval_869 = condval_870;
                }
                int condval_871;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_871 = 0;
                } else {
                  int condval_872;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_872 = 3;
                  } else {
                    condval_872 = (((slab * 2) + m_12) - 1);
                  }
                  condval_871 = condval_872;
                }
                condval_860 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_869 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_871 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_873;
              if ((((slab * 2) + m_12) == 0)) {
                condval_873 = 0;
              } else {
                int condval_874;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_874 = 3;
                } else {
                  condval_874 = (((slab * 2) + m_12) - 1);
                }
                condval_873 = condval_874;
              }
              int condval_875;
              if ((((slab * 2) + m_12) == 0)) {
                condval_875 = 0;
              } else {
                int condval_876;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_876 = 3;
                } else {
                  condval_876 = (((slab * 2) + m_12) - 1);
                }
                condval_875 = condval_876;
              }
              int condval_877;
              if ((((slab * 2) + m_12) == 0)) {
                condval_877 = 0;
              } else {
                int condval_878;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_878 = 3;
                } else {
                  condval_878 = (((slab * 2) + m_12) - 1);
                }
                condval_877 = condval_878;
              }
              int condval_880;
              if ((((slab * 2) + m_12) == 0)) {
                condval_880 = 0;
              } else {
                int condval_881;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_881 = 3;
                } else {
                  condval_881 = (((slab * 2) + m_12) - 1);
                }
                condval_880 = condval_881;
              }
              int condval_882;
              if ((((slab * 2) + m_12) == 0)) {
                condval_882 = 0;
              } else {
                int condval_883;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_883 = 3;
                } else {
                  condval_883 = (((slab * 2) + m_12) - 1);
                }
                condval_882 = condval_883;
              }
              int condval_879;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_880 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_882 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_884;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_884 = 0;
                } else {
                  int condval_885;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_885 = 3;
                  } else {
                    condval_885 = (((slab * 2) + m_12) - 1);
                  }
                  condval_884 = condval_885;
                }
                int condval_886;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_886 = 0;
                } else {
                  int condval_887;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_887 = 3;
                  } else {
                    condval_887 = (((slab * 2) + m_12) - 1);
                  }
                  condval_886 = condval_887;
                }
                condval_879 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_884 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_886 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_888;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_888 = 0;
                } else {
                  int condval_889;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_889 = 3;
                  } else {
                    condval_889 = (((slab * 2) + m_12) - 1);
                  }
                  condval_888 = condval_889;
                }
                int condval_890;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_890 = 0;
                } else {
                  int condval_891;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_891 = 3;
                  } else {
                    condval_891 = (((slab * 2) + m_12) - 1);
                  }
                  condval_890 = condval_891;
                }
                condval_879 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_888 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_890 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_892;
              if ((((slab * 2) + m_12) == 0)) {
                condval_892 = 0;
              } else {
                int condval_893;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_893 = 3;
                } else {
                  condval_893 = (((slab * 2) + m_12) - 1);
                }
                condval_892 = condval_893;
              }
              int condval_894;
              if ((((slab * 2) + m_12) == 0)) {
                condval_894 = 0;
              } else {
                int condval_895;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_895 = 3;
                } else {
                  condval_895 = (((slab * 2) + m_12) - 1);
                }
                condval_894 = condval_895;
              }
              int condval_896;
              if ((((slab * 2) + m_12) == 0)) {
                condval_896 = 0;
              } else {
                int condval_897;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_897 = 3;
                } else {
                  condval_897 = (((slab * 2) + m_12) - 1);
                }
                condval_896 = condval_897;
              }
              int condval_899;
              if ((((slab * 2) + m_12) == 0)) {
                condval_899 = 0;
              } else {
                int condval_900;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_900 = 3;
                } else {
                  condval_900 = (((slab * 2) + m_12) - 1);
                }
                condval_899 = condval_900;
              }
              int condval_901;
              if ((((slab * 2) + m_12) == 0)) {
                condval_901 = 0;
              } else {
                int condval_902;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_902 = 3;
                } else {
                  condval_902 = (((slab * 2) + m_12) - 1);
                }
                condval_901 = condval_902;
              }
              int condval_898;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_899 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_901 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_903;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_903 = 0;
                } else {
                  int condval_904;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_904 = 3;
                  } else {
                    condval_904 = (((slab * 2) + m_12) - 1);
                  }
                  condval_903 = condval_904;
                }
                int condval_905;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_905 = 0;
                } else {
                  int condval_906;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_906 = 3;
                  } else {
                    condval_906 = (((slab * 2) + m_12) - 1);
                  }
                  condval_905 = condval_906;
                }
                condval_898 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_903 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_905 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_907;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_907 = 0;
                } else {
                  int condval_908;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_908 = 3;
                  } else {
                    condval_908 = (((slab * 2) + m_12) - 1);
                  }
                  condval_907 = condval_908;
                }
                int condval_909;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_909 = 0;
                } else {
                  int condval_910;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_910 = 3;
                  } else {
                    condval_910 = (((slab * 2) + m_12) - 1);
                  }
                  condval_909 = condval_910;
                }
                condval_898 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_907 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_909 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_911;
              if ((((slab * 2) + m_12) == 0)) {
                condval_911 = 0;
              } else {
                int condval_912;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_912 = 3;
                } else {
                  condval_912 = (((slab * 2) + m_12) - 1);
                }
                condval_911 = condval_912;
              }
              int condval_913;
              if ((((slab * 2) + m_12) == 0)) {
                condval_913 = 0;
              } else {
                int condval_914;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_914 = 3;
                } else {
                  condval_914 = (((slab * 2) + m_12) - 1);
                }
                condval_913 = condval_914;
              }
              int condval_915;
              if ((((slab * 2) + m_12) == 0)) {
                condval_915 = 0;
              } else {
                int condval_916;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_916 = 3;
                } else {
                  condval_916 = (((slab * 2) + m_12) - 1);
                }
                condval_915 = condval_916;
              }
              int condval_918;
              if ((((slab * 2) + m_12) == 0)) {
                condval_918 = 0;
              } else {
                int condval_919;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_919 = 3;
                } else {
                  condval_919 = (((slab * 2) + m_12) - 1);
                }
                condval_918 = condval_919;
              }
              int condval_920;
              if ((((slab * 2) + m_12) == 0)) {
                condval_920 = 0;
              } else {
                int condval_921;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_921 = 3;
                } else {
                  condval_921 = (((slab * 2) + m_12) - 1);
                }
                condval_920 = condval_921;
              }
              int condval_917;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_918 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_920 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_922;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_922 = 0;
                } else {
                  int condval_923;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_923 = 3;
                  } else {
                    condval_923 = (((slab * 2) + m_12) - 1);
                  }
                  condval_922 = condval_923;
                }
                int condval_924;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_924 = 0;
                } else {
                  int condval_925;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_925 = 3;
                  } else {
                    condval_925 = (((slab * 2) + m_12) - 1);
                  }
                  condval_924 = condval_925;
                }
                condval_917 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_922 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_924 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_926;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_926 = 0;
                } else {
                  int condval_927;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_927 = 3;
                  } else {
                    condval_927 = (((slab * 2) + m_12) - 1);
                  }
                  condval_926 = condval_927;
                }
                int condval_928;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_928 = 0;
                } else {
                  int condval_929;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_929 = 3;
                  } else {
                    condval_929 = (((slab * 2) + m_12) - 1);
                  }
                  condval_928 = condval_929;
                }
                condval_917 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_926 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_928 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_930;
              if ((((slab * 2) + m_12) == 0)) {
                condval_930 = 0;
              } else {
                int condval_931;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_931 = 3;
                } else {
                  condval_931 = (((slab * 2) + m_12) - 1);
                }
                condval_930 = condval_931;
              }
              int condval_932;
              if ((((slab * 2) + m_12) == 0)) {
                condval_932 = 0;
              } else {
                int condval_933;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_933 = 3;
                } else {
                  condval_933 = (((slab * 2) + m_12) - 1);
                }
                condval_932 = condval_933;
              }
              int condval_934;
              if ((((slab * 2) + m_12) == 0)) {
                condval_934 = 0;
              } else {
                int condval_935;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_935 = 3;
                } else {
                  condval_935 = (((slab * 2) + m_12) - 1);
                }
                condval_934 = condval_935;
              }
              int condval_936;
              if ((((slab * 2) + m_12) == 0)) {
                condval_936 = 0;
              } else {
                int condval_937;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_937 = 3;
                } else {
                  condval_937 = (((slab * 2) + m_12) - 1);
                }
                condval_936 = condval_937;
              }
              int condval_938;
              if ((((slab * 2) + m_12) == 0)) {
                condval_938 = 0;
              } else {
                int condval_939;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_939 = 3;
                } else {
                  condval_939 = (((slab * 2) + m_12) - 1);
                }
                condval_938 = condval_939;
              }
              int condval_941;
              if ((((slab * 2) + m_12) == 0)) {
                condval_941 = 0;
              } else {
                int condval_942;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_942 = 3;
                } else {
                  condval_942 = (((slab * 2) + m_12) - 1);
                }
                condval_941 = condval_942;
              }
              int condval_943;
              if ((((slab * 2) + m_12) == 0)) {
                condval_943 = 0;
              } else {
                int condval_944;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_944 = 3;
                } else {
                  condval_944 = (((slab * 2) + m_12) - 1);
                }
                condval_943 = condval_944;
              }
              int condval_940;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_941 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_943 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_945;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_945 = 0;
                } else {
                  int condval_946;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_946 = 3;
                  } else {
                    condval_946 = (((slab * 2) + m_12) - 1);
                  }
                  condval_945 = condval_946;
                }
                int condval_947;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_947 = 0;
                } else {
                  int condval_948;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_948 = 3;
                  } else {
                    condval_948 = (((slab * 2) + m_12) - 1);
                  }
                  condval_947 = condval_948;
                }
                condval_940 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_945 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_947 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_949;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_949 = 0;
                } else {
                  int condval_950;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_950 = 3;
                  } else {
                    condval_950 = (((slab * 2) + m_12) - 1);
                  }
                  condval_949 = condval_950;
                }
                int condval_951;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_951 = 0;
                } else {
                  int condval_952;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_952 = 3;
                  } else {
                    condval_952 = (((slab * 2) + m_12) - 1);
                  }
                  condval_951 = condval_952;
                }
                condval_940 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_949 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_951 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_953;
              if ((((slab * 2) + m_12) == 0)) {
                condval_953 = 0;
              } else {
                int condval_954;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_954 = 3;
                } else {
                  condval_954 = (((slab * 2) + m_12) - 1);
                }
                condval_953 = condval_954;
              }
              int condval_956;
              if ((((slab * 2) + m_12) == 0)) {
                condval_956 = 0;
              } else {
                int condval_957;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_957 = 3;
                } else {
                  condval_957 = (((slab * 2) + m_12) - 1);
                }
                condval_956 = condval_957;
              }
              int condval_958;
              if ((((slab * 2) + m_12) == 0)) {
                condval_958 = 0;
              } else {
                int condval_959;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_959 = 3;
                } else {
                  condval_959 = (((slab * 2) + m_12) - 1);
                }
                condval_958 = condval_959;
              }
              int condval_955;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_956 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_958 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_960;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_960 = 0;
                } else {
                  int condval_961;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_961 = 3;
                  } else {
                    condval_961 = (((slab * 2) + m_12) - 1);
                  }
                  condval_960 = condval_961;
                }
                int condval_962;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_962 = 0;
                } else {
                  int condval_963;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_963 = 3;
                  } else {
                    condval_963 = (((slab * 2) + m_12) - 1);
                  }
                  condval_962 = condval_963;
                }
                condval_955 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_960 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_962 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_964;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_964 = 0;
                } else {
                  int condval_965;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_965 = 3;
                  } else {
                    condval_965 = (((slab * 2) + m_12) - 1);
                  }
                  condval_964 = condval_965;
                }
                int condval_966;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_966 = 0;
                } else {
                  int condval_967;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_967 = 3;
                  } else {
                    condval_967 = (((slab * 2) + m_12) - 1);
                  }
                  condval_966 = condval_967;
                }
                condval_955 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_964 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_966 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_968;
              if ((((slab * 2) + m_12) == 0)) {
                condval_968 = 0;
              } else {
                int condval_969;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_969 = 3;
                } else {
                  condval_969 = (((slab * 2) + m_12) - 1);
                }
                condval_968 = condval_969;
              }
              int condval_970;
              if ((((slab * 2) + m_12) == 0)) {
                condval_970 = 0;
              } else {
                int condval_971;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_971 = 3;
                } else {
                  condval_971 = (((slab * 2) + m_12) - 1);
                }
                condval_970 = condval_971;
              }
              int condval_972;
              if ((((slab * 2) + m_12) == 0)) {
                condval_972 = 0;
              } else {
                int condval_973;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_973 = 3;
                } else {
                  condval_973 = (((slab * 2) + m_12) - 1);
                }
                condval_972 = condval_973;
              }
              int condval_975;
              if ((((slab * 2) + m_12) == 0)) {
                condval_975 = 0;
              } else {
                int condval_976;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_976 = 3;
                } else {
                  condval_976 = (((slab * 2) + m_12) - 1);
                }
                condval_975 = condval_976;
              }
              int condval_977;
              if ((((slab * 2) + m_12) == 0)) {
                condval_977 = 0;
              } else {
                int condval_978;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_978 = 3;
                } else {
                  condval_978 = (((slab * 2) + m_12) - 1);
                }
                condval_977 = condval_978;
              }
              int condval_974;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_975 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_977 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_979;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_979 = 0;
                } else {
                  int condval_980;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_980 = 3;
                  } else {
                    condval_980 = (((slab * 2) + m_12) - 1);
                  }
                  condval_979 = condval_980;
                }
                int condval_981;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_981 = 0;
                } else {
                  int condval_982;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_982 = 3;
                  } else {
                    condval_982 = (((slab * 2) + m_12) - 1);
                  }
                  condval_981 = condval_982;
                }
                condval_974 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_979 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_981 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_983;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_983 = 0;
                } else {
                  int condval_984;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_984 = 3;
                  } else {
                    condval_984 = (((slab * 2) + m_12) - 1);
                  }
                  condval_983 = condval_984;
                }
                int condval_985;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_985 = 0;
                } else {
                  int condval_986;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_986 = 3;
                  } else {
                    condval_986 = (((slab * 2) + m_12) - 1);
                  }
                  condval_985 = condval_986;
                }
                condval_974 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_983 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_985 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_987;
              if ((((slab * 2) + m_12) == 0)) {
                condval_987 = 0;
              } else {
                int condval_988;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_988 = 3;
                } else {
                  condval_988 = (((slab * 2) + m_12) - 1);
                }
                condval_987 = condval_988;
              }
              int condval_989;
              if ((((slab * 2) + m_12) == 0)) {
                condval_989 = 0;
              } else {
                int condval_990;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_990 = 3;
                } else {
                  condval_990 = (((slab * 2) + m_12) - 1);
                }
                condval_989 = condval_990;
              }
              int condval_991;
              if ((((slab * 2) + m_12) == 0)) {
                condval_991 = 0;
              } else {
                int condval_992;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_992 = 3;
                } else {
                  condval_992 = (((slab * 2) + m_12) - 1);
                }
                condval_991 = condval_992;
              }
              int condval_994;
              if ((((slab * 2) + m_12) == 0)) {
                condval_994 = 0;
              } else {
                int condval_995;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_995 = 3;
                } else {
                  condval_995 = (((slab * 2) + m_12) - 1);
                }
                condval_994 = condval_995;
              }
              int condval_996;
              if ((((slab * 2) + m_12) == 0)) {
                condval_996 = 0;
              } else {
                int condval_997;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_997 = 3;
                } else {
                  condval_997 = (((slab * 2) + m_12) - 1);
                }
                condval_996 = condval_997;
              }
              int condval_993;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_994 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_996 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_998;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_998 = 0;
                } else {
                  int condval_999;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_999 = 3;
                  } else {
                    condval_999 = (((slab * 2) + m_12) - 1);
                  }
                  condval_998 = condval_999;
                }
                int condval_1000;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1000 = 0;
                } else {
                  int condval_1001;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1001 = 3;
                  } else {
                    condval_1001 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1000 = condval_1001;
                }
                condval_993 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_998 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1000 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1002;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1002 = 0;
                } else {
                  int condval_1003;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1003 = 3;
                  } else {
                    condval_1003 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1002 = condval_1003;
                }
                int condval_1004;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1004 = 0;
                } else {
                  int condval_1005;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1005 = 3;
                  } else {
                    condval_1005 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1004 = condval_1005;
                }
                condval_993 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1002 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1004 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1006;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1006 = 0;
              } else {
                int condval_1007;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1007 = 3;
                } else {
                  condval_1007 = (((slab * 2) + m_12) - 1);
                }
                condval_1006 = condval_1007;
              }
              int condval_1008;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1008 = 0;
              } else {
                int condval_1009;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1009 = 3;
                } else {
                  condval_1009 = (((slab * 2) + m_12) - 1);
                }
                condval_1008 = condval_1009;
              }
              int condval_1010;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1010 = 0;
              } else {
                int condval_1011;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1011 = 3;
                } else {
                  condval_1011 = (((slab * 2) + m_12) - 1);
                }
                condval_1010 = condval_1011;
              }
              int condval_1013;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1013 = 0;
              } else {
                int condval_1014;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1014 = 3;
                } else {
                  condval_1014 = (((slab * 2) + m_12) - 1);
                }
                condval_1013 = condval_1014;
              }
              int condval_1015;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1015 = 0;
              } else {
                int condval_1016;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1016 = 3;
                } else {
                  condval_1016 = (((slab * 2) + m_12) - 1);
                }
                condval_1015 = condval_1016;
              }
              int condval_1012;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1013 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1015 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1017;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1017 = 0;
                } else {
                  int condval_1018;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1018 = 3;
                  } else {
                    condval_1018 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1017 = condval_1018;
                }
                int condval_1019;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1019 = 0;
                } else {
                  int condval_1020;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1020 = 3;
                  } else {
                    condval_1020 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1019 = condval_1020;
                }
                condval_1012 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1017 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1019 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1021;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1021 = 0;
                } else {
                  int condval_1022;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1022 = 3;
                  } else {
                    condval_1022 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1021 = condval_1022;
                }
                int condval_1023;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1023 = 0;
                } else {
                  int condval_1024;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1024 = 3;
                  } else {
                    condval_1024 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1023 = condval_1024;
                }
                condval_1012 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1021 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1023 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1025;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1025 = 0;
              } else {
                int condval_1026;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1026 = 3;
                } else {
                  condval_1026 = (((slab * 2) + m_12) - 1);
                }
                condval_1025 = condval_1026;
              }
              int condval_1027;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1027 = 0;
              } else {
                int condval_1028;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1028 = 3;
                } else {
                  condval_1028 = (((slab * 2) + m_12) - 1);
                }
                condval_1027 = condval_1028;
              }
              int condval_1029;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1029 = 0;
              } else {
                int condval_1030;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1030 = 3;
                } else {
                  condval_1030 = (((slab * 2) + m_12) - 1);
                }
                condval_1029 = condval_1030;
              }
              int condval_1032;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1032 = 0;
              } else {
                int condval_1033;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1033 = 3;
                } else {
                  condval_1033 = (((slab * 2) + m_12) - 1);
                }
                condval_1032 = condval_1033;
              }
              int condval_1034;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1034 = 0;
              } else {
                int condval_1035;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1035 = 3;
                } else {
                  condval_1035 = (((slab * 2) + m_12) - 1);
                }
                condval_1034 = condval_1035;
              }
              int condval_1031;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1032 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1034 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1036;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1036 = 0;
                } else {
                  int condval_1037;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1037 = 3;
                  } else {
                    condval_1037 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1036 = condval_1037;
                }
                int condval_1038;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1038 = 0;
                } else {
                  int condval_1039;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1039 = 3;
                  } else {
                    condval_1039 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1038 = condval_1039;
                }
                condval_1031 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1036 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1038 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1040;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1040 = 0;
                } else {
                  int condval_1041;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1041 = 3;
                  } else {
                    condval_1041 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1040 = condval_1041;
                }
                int condval_1042;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1042 = 0;
                } else {
                  int condval_1043;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1043 = 3;
                  } else {
                    condval_1043 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1042 = condval_1043;
                }
                condval_1031 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1040 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1042 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1044;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1044 = 0;
              } else {
                int condval_1045;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1045 = 3;
                } else {
                  condval_1045 = (((slab * 2) + m_12) - 1);
                }
                condval_1044 = condval_1045;
              }
              int condval_1046;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1046 = 0;
              } else {
                int condval_1047;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1047 = 3;
                } else {
                  condval_1047 = (((slab * 2) + m_12) - 1);
                }
                condval_1046 = condval_1047;
              }
              int condval_1048;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1048 = 0;
              } else {
                int condval_1049;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1049 = 3;
                } else {
                  condval_1049 = (((slab * 2) + m_12) - 1);
                }
                condval_1048 = condval_1049;
              }
              int condval_1051;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1051 = 0;
              } else {
                int condval_1052;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1052 = 3;
                } else {
                  condval_1052 = (((slab * 2) + m_12) - 1);
                }
                condval_1051 = condval_1052;
              }
              int condval_1053;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1053 = 0;
              } else {
                int condval_1054;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1054 = 3;
                } else {
                  condval_1054 = (((slab * 2) + m_12) - 1);
                }
                condval_1053 = condval_1054;
              }
              int condval_1050;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1051 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1053 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1055;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1055 = 0;
                } else {
                  int condval_1056;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1056 = 3;
                  } else {
                    condval_1056 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1055 = condval_1056;
                }
                int condval_1057;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1057 = 0;
                } else {
                  int condval_1058;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1058 = 3;
                  } else {
                    condval_1058 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1057 = condval_1058;
                }
                condval_1050 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1055 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1057 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1059;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1059 = 0;
                } else {
                  int condval_1060;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1060 = 3;
                  } else {
                    condval_1060 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1059 = condval_1060;
                }
                int condval_1061;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1061 = 0;
                } else {
                  int condval_1062;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1062 = 3;
                  } else {
                    condval_1062 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1061 = condval_1062;
                }
                condval_1050 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1059 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1061 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1063;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1063 = 0;
              } else {
                int condval_1064;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1064 = 3;
                } else {
                  condval_1064 = (((slab * 2) + m_12) - 1);
                }
                condval_1063 = condval_1064;
              }
              int condval_1065;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1065 = 0;
              } else {
                int condval_1066;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1066 = 3;
                } else {
                  condval_1066 = (((slab * 2) + m_12) - 1);
                }
                condval_1065 = condval_1066;
              }
              int condval_1067;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1067 = 0;
              } else {
                int condval_1068;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1068 = 3;
                } else {
                  condval_1068 = (((slab * 2) + m_12) - 1);
                }
                condval_1067 = condval_1068;
              }
              int condval_1069;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1069 = 0;
              } else {
                int condval_1070;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1070 = 3;
                } else {
                  condval_1070 = (((slab * 2) + m_12) - 1);
                }
                condval_1069 = condval_1070;
              }
              int condval_1071;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1071 = 0;
              } else {
                int condval_1072;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1072 = 3;
                } else {
                  condval_1072 = (((slab * 2) + m_12) - 1);
                }
                condval_1071 = condval_1072;
              }
              int condval_1073;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1073 = 0;
              } else {
                int condval_1074;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1074 = 3;
                } else {
                  condval_1074 = (((slab * 2) + m_12) - 1);
                }
                condval_1073 = condval_1074;
              }
              int condval_1075;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1075 = 0;
              } else {
                int condval_1076;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1076 = 3;
                } else {
                  condval_1076 = (((slab * 2) + m_12) - 1);
                }
                condval_1075 = condval_1076;
              }
              int condval_1077;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1077 = 0;
              } else {
                int condval_1078;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1078 = 3;
                } else {
                  condval_1078 = (((slab * 2) + m_12) - 1);
                }
                condval_1077 = condval_1078;
              }
              int condval_1079;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1079 = 0;
              } else {
                int condval_1080;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1080 = 3;
                } else {
                  condval_1080 = (((slab * 2) + m_12) - 1);
                }
                condval_1079 = condval_1080;
              }
              int condval_1081;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1081 = 0;
              } else {
                int condval_1082;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1082 = 3;
                } else {
                  condval_1082 = (((slab * 2) + m_12) - 1);
                }
                condval_1081 = condval_1082;
              }
              int condval_1083;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1083 = 0;
              } else {
                int condval_1084;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1084 = 3;
                } else {
                  condval_1084 = (((slab * 2) + m_12) - 1);
                }
                condval_1083 = condval_1084;
              }
              int condval_1086;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1086 = 0;
              } else {
                int condval_1087;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1087 = 3;
                } else {
                  condval_1087 = (((slab * 2) + m_12) - 1);
                }
                condval_1086 = condval_1087;
              }
              int condval_1088;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1088 = 0;
              } else {
                int condval_1089;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1089 = 3;
                } else {
                  condval_1089 = (((slab * 2) + m_12) - 1);
                }
                condval_1088 = condval_1089;
              }
              int condval_1085;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1086 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1088 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1090;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1090 = 0;
                } else {
                  int condval_1091;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1091 = 3;
                  } else {
                    condval_1091 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1090 = condval_1091;
                }
                int condval_1092;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1092 = 0;
                } else {
                  int condval_1093;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1093 = 3;
                  } else {
                    condval_1093 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1092 = condval_1093;
                }
                condval_1085 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1090 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1092 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1094;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1094 = 0;
                } else {
                  int condval_1095;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1095 = 3;
                  } else {
                    condval_1095 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1094 = condval_1095;
                }
                int condval_1096;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1096 = 0;
                } else {
                  int condval_1097;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1097 = 3;
                  } else {
                    condval_1097 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1096 = condval_1097;
                }
                condval_1085 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1094 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1096 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1098;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1098 = 0;
              } else {
                int condval_1099;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1099 = 3;
                } else {
                  condval_1099 = (((slab * 2) + m_12) - 1);
                }
                condval_1098 = condval_1099;
              }
              int condval_1100;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1100 = 0;
              } else {
                int condval_1101;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1101 = 3;
                } else {
                  condval_1101 = (((slab * 2) + m_12) - 1);
                }
                condval_1100 = condval_1101;
              }
              int condval_1102;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1102 = 0;
              } else {
                int condval_1103;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1103 = 3;
                } else {
                  condval_1103 = (((slab * 2) + m_12) - 1);
                }
                condval_1102 = condval_1103;
              }
              int condval_1105;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1105 = 0;
              } else {
                int condval_1106;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1106 = 3;
                } else {
                  condval_1106 = (((slab * 2) + m_12) - 1);
                }
                condval_1105 = condval_1106;
              }
              int condval_1107;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1107 = 0;
              } else {
                int condval_1108;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1108 = 3;
                } else {
                  condval_1108 = (((slab * 2) + m_12) - 1);
                }
                condval_1107 = condval_1108;
              }
              int condval_1104;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1105 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1107 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1109;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1109 = 0;
                } else {
                  int condval_1110;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1110 = 3;
                  } else {
                    condval_1110 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1109 = condval_1110;
                }
                int condval_1111;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1111 = 0;
                } else {
                  int condval_1112;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1112 = 3;
                  } else {
                    condval_1112 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1111 = condval_1112;
                }
                condval_1104 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1109 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1111 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1113;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1113 = 0;
                } else {
                  int condval_1114;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1114 = 3;
                  } else {
                    condval_1114 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1113 = condval_1114;
                }
                int condval_1115;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1115 = 0;
                } else {
                  int condval_1116;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1116 = 3;
                  } else {
                    condval_1116 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1115 = condval_1116;
                }
                condval_1104 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1113 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1115 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1117;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1117 = 0;
              } else {
                int condval_1118;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1118 = 3;
                } else {
                  condval_1118 = (((slab * 2) + m_12) - 1);
                }
                condval_1117 = condval_1118;
              }
              int condval_1119;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1119 = 0;
              } else {
                int condval_1120;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1120 = 3;
                } else {
                  condval_1120 = (((slab * 2) + m_12) - 1);
                }
                condval_1119 = condval_1120;
              }
              int condval_1121;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1121 = 0;
              } else {
                int condval_1122;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1122 = 3;
                } else {
                  condval_1122 = (((slab * 2) + m_12) - 1);
                }
                condval_1121 = condval_1122;
              }
              int condval_1124;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1124 = 0;
              } else {
                int condval_1125;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1125 = 3;
                } else {
                  condval_1125 = (((slab * 2) + m_12) - 1);
                }
                condval_1124 = condval_1125;
              }
              int condval_1126;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1126 = 0;
              } else {
                int condval_1127;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1127 = 3;
                } else {
                  condval_1127 = (((slab * 2) + m_12) - 1);
                }
                condval_1126 = condval_1127;
              }
              int condval_1123;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1124 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1126 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1128;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1128 = 0;
                } else {
                  int condval_1129;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1129 = 3;
                  } else {
                    condval_1129 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1128 = condval_1129;
                }
                int condval_1130;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1130 = 0;
                } else {
                  int condval_1131;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1131 = 3;
                  } else {
                    condval_1131 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1130 = condval_1131;
                }
                condval_1123 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1128 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1130 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1132;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1132 = 0;
                } else {
                  int condval_1133;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1133 = 3;
                  } else {
                    condval_1133 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1132 = condval_1133;
                }
                int condval_1134;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1134 = 0;
                } else {
                  int condval_1135;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1135 = 3;
                  } else {
                    condval_1135 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1134 = condval_1135;
                }
                condval_1123 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1132 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1134 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1136;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1136 = 0;
              } else {
                int condval_1137;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1137 = 3;
                } else {
                  condval_1137 = (((slab * 2) + m_12) - 1);
                }
                condval_1136 = condval_1137;
              }
              int condval_1138;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1138 = 0;
              } else {
                int condval_1139;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1139 = 3;
                } else {
                  condval_1139 = (((slab * 2) + m_12) - 1);
                }
                condval_1138 = condval_1139;
              }
              int condval_1140;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1140 = 0;
              } else {
                int condval_1141;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1141 = 3;
                } else {
                  condval_1141 = (((slab * 2) + m_12) - 1);
                }
                condval_1140 = condval_1141;
              }
              int condval_1143;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1143 = 0;
              } else {
                int condval_1144;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1144 = 3;
                } else {
                  condval_1144 = (((slab * 2) + m_12) - 1);
                }
                condval_1143 = condval_1144;
              }
              int condval_1145;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1145 = 0;
              } else {
                int condval_1146;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1146 = 3;
                } else {
                  condval_1146 = (((slab * 2) + m_12) - 1);
                }
                condval_1145 = condval_1146;
              }
              int condval_1142;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1143 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1145 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1147;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1147 = 0;
                } else {
                  int condval_1148;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1148 = 3;
                  } else {
                    condval_1148 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1147 = condval_1148;
                }
                int condval_1149;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1149 = 0;
                } else {
                  int condval_1150;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1150 = 3;
                  } else {
                    condval_1150 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1149 = condval_1150;
                }
                condval_1142 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1147 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1149 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1151;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1151 = 0;
                } else {
                  int condval_1152;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1152 = 3;
                  } else {
                    condval_1152 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1151 = condval_1152;
                }
                int condval_1153;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1153 = 0;
                } else {
                  int condval_1154;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1154 = 3;
                  } else {
                    condval_1154 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1153 = condval_1154;
                }
                condval_1142 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1151 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1153 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1155;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1155 = 0;
              } else {
                int condval_1156;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1156 = 3;
                } else {
                  condval_1156 = (((slab * 2) + m_12) - 1);
                }
                condval_1155 = condval_1156;
              }
              int condval_1157;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1157 = 0;
              } else {
                int condval_1158;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1158 = 3;
                } else {
                  condval_1158 = (((slab * 2) + m_12) - 1);
                }
                condval_1157 = condval_1158;
              }
              int condval_1159;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1159 = 0;
              } else {
                int condval_1160;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1160 = 3;
                } else {
                  condval_1160 = (((slab * 2) + m_12) - 1);
                }
                condval_1159 = condval_1160;
              }
              int condval_1162;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1162 = 0;
              } else {
                int condval_1163;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1163 = 3;
                } else {
                  condval_1163 = (((slab * 2) + m_12) - 1);
                }
                condval_1162 = condval_1163;
              }
              int condval_1164;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1164 = 0;
              } else {
                int condval_1165;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1165 = 3;
                } else {
                  condval_1165 = (((slab * 2) + m_12) - 1);
                }
                condval_1164 = condval_1165;
              }
              int condval_1161;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1162 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1164 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1166;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1166 = 0;
                } else {
                  int condval_1167;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1167 = 3;
                  } else {
                    condval_1167 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1166 = condval_1167;
                }
                int condval_1168;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1168 = 0;
                } else {
                  int condval_1169;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1169 = 3;
                  } else {
                    condval_1169 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1168 = condval_1169;
                }
                condval_1161 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1166 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1168 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1170;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1170 = 0;
                } else {
                  int condval_1171;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1171 = 3;
                  } else {
                    condval_1171 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1170 = condval_1171;
                }
                int condval_1172;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1172 = 0;
                } else {
                  int condval_1173;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1173 = 3;
                  } else {
                    condval_1173 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1172 = condval_1173;
                }
                condval_1161 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1170 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1172 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1174;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1174 = 0;
              } else {
                int condval_1175;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1175 = 3;
                } else {
                  condval_1175 = (((slab * 2) + m_12) - 1);
                }
                condval_1174 = condval_1175;
              }
              int condval_1176;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1176 = 0;
              } else {
                int condval_1177;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1177 = 3;
                } else {
                  condval_1177 = (((slab * 2) + m_12) - 1);
                }
                condval_1176 = condval_1177;
              }
              int condval_1178;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1178 = 0;
              } else {
                int condval_1179;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1179 = 3;
                } else {
                  condval_1179 = (((slab * 2) + m_12) - 1);
                }
                condval_1178 = condval_1179;
              }
              int condval_1181;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1181 = 0;
              } else {
                int condval_1182;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1182 = 3;
                } else {
                  condval_1182 = (((slab * 2) + m_12) - 1);
                }
                condval_1181 = condval_1182;
              }
              int condval_1183;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1183 = 0;
              } else {
                int condval_1184;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1184 = 3;
                } else {
                  condval_1184 = (((slab * 2) + m_12) - 1);
                }
                condval_1183 = condval_1184;
              }
              int condval_1180;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1181 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1183 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1185;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1185 = 0;
                } else {
                  int condval_1186;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1186 = 3;
                  } else {
                    condval_1186 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1185 = condval_1186;
                }
                int condval_1187;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1187 = 0;
                } else {
                  int condval_1188;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1188 = 3;
                  } else {
                    condval_1188 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1187 = condval_1188;
                }
                condval_1180 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1185 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1187 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1189;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1189 = 0;
                } else {
                  int condval_1190;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1190 = 3;
                  } else {
                    condval_1190 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1189 = condval_1190;
                }
                int condval_1191;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1191 = 0;
                } else {
                  int condval_1192;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1192 = 3;
                  } else {
                    condval_1192 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1191 = condval_1192;
                }
                condval_1180 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1189 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1191 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1193;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1193 = 0;
              } else {
                int condval_1194;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1194 = 3;
                } else {
                  condval_1194 = (((slab * 2) + m_12) - 1);
                }
                condval_1193 = condval_1194;
              }
              int condval_1195;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1195 = 0;
              } else {
                int condval_1196;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1196 = 3;
                } else {
                  condval_1196 = (((slab * 2) + m_12) - 1);
                }
                condval_1195 = condval_1196;
              }
              int condval_1197;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1197 = 0;
              } else {
                int condval_1198;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1198 = 3;
                } else {
                  condval_1198 = (((slab * 2) + m_12) - 1);
                }
                condval_1197 = condval_1198;
              }
              condval_813 = (((((((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_822 & 3) * 16) + (((((condval_835 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_837 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_839 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_841 & 3) * 16) + (((((condval_854 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_856 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_858 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_860 & 3) * 16) + (((((condval_873 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_875 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_877 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_879 & 3) * 16) + (((((condval_892 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_894 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_896 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_898 & 3) * 16) + (((((condval_911 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_913 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_915 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_917 & 3) * 16) + (((((condval_930 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_932 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_934 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + ((((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_936 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_938 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_940 * 2)) + ((((condval_953 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) * 40960)) + (((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_955 & 3) * 16) + (((((condval_968 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_970 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_972 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_974 & 3) * 16) + (((((condval_987 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_989 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_991 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_993 & 3) * 16) + (((((condval_1006 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1008 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1010 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_1012 & 3) * 16) + (((((condval_1025 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1027 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1029 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_1031 & 3) * 16) + (((((condval_1044 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1046 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1048 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_1050 & 3) * 16) + (((((condval_1063 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1065 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1067 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1069 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1071 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_1073 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1075 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1077 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1079 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_1081 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1083 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_1085 & 3) * 16) + (((((condval_1098 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1100 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1102 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_1104 & 3) * 16) + (((((condval_1117 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1119 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1121 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_1123 & 3) * 16) + (((((condval_1136 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1138 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1140 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_1142 & 3) * 16) + (((((condval_1155 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1157 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1159 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_1161 & 3) * 16) + (((((condval_1174 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1176 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1178 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_1180 & 3) * 16) + (((((condval_1193 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1195 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1197 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
            } else {
              condval_813 = -1;
            }
            int condval_1200;
            if ((((slab * 2) + m_12) == 0)) {
              condval_1200 = 0;
            } else {
              int condval_1201;
              if ((((slab * 2) + m_12) == 1)) {
                condval_1201 = 3;
              } else {
                condval_1201 = (((slab * 2) + m_12) - 1);
              }
              condval_1200 = condval_1201;
            }
            int condval_1202;
            if ((((slab * 2) + m_12) == 0)) {
              condval_1202 = 0;
            } else {
              int condval_1203;
              if ((((slab * 2) + m_12) == 1)) {
                condval_1203 = 3;
              } else {
                condval_1203 = (((slab * 2) + m_12) - 1);
              }
              condval_1202 = condval_1203;
            }
            int condval_1204;
            if ((((slab * 2) + m_12) == 0)) {
              condval_1204 = 0;
            } else {
              int condval_1205;
              if ((((slab * 2) + m_12) == 1)) {
                condval_1205 = 3;
              } else {
                condval_1205 = (((slab * 2) + m_12) - 1);
              }
              condval_1204 = condval_1205;
            }
            int condval_1206;
            if ((((slab * 2) + m_12) == 0)) {
              condval_1206 = 0;
            } else {
              int condval_1207;
              if ((((slab * 2) + m_12) == 1)) {
                condval_1207 = 3;
              } else {
                condval_1207 = (((slab * 2) + m_12) - 1);
              }
              condval_1206 = condval_1207;
            }
            int condval_1199;
            if (((((1 <= (((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1200 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1202 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & ((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1204 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1206 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) < 65)) & (bool)1) & (bool)1)) {
              int condval_1209;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1209 = 0;
              } else {
                int condval_1210;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1210 = 3;
                } else {
                  condval_1210 = (((slab * 2) + m_12) - 1);
                }
                condval_1209 = condval_1210;
              }
              int condval_1211;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1211 = 0;
              } else {
                int condval_1212;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1212 = 3;
                } else {
                  condval_1212 = (((slab * 2) + m_12) - 1);
                }
                condval_1211 = condval_1212;
              }
              int condval_1208;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1209 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1211 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1213;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1213 = 0;
                } else {
                  int condval_1214;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1214 = 3;
                  } else {
                    condval_1214 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1213 = condval_1214;
                }
                int condval_1215;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1215 = 0;
                } else {
                  int condval_1216;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1216 = 3;
                  } else {
                    condval_1216 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1215 = condval_1216;
                }
                condval_1208 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1213 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1215 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1217;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1217 = 0;
                } else {
                  int condval_1218;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1218 = 3;
                  } else {
                    condval_1218 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1217 = condval_1218;
                }
                int condval_1219;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1219 = 0;
                } else {
                  int condval_1220;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1220 = 3;
                  } else {
                    condval_1220 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1219 = condval_1220;
                }
                condval_1208 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1217 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1219 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1221;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1221 = 0;
              } else {
                int condval_1222;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1222 = 3;
                } else {
                  condval_1222 = (((slab * 2) + m_12) - 1);
                }
                condval_1221 = condval_1222;
              }
              int condval_1223;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1223 = 0;
              } else {
                int condval_1224;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1224 = 3;
                } else {
                  condval_1224 = (((slab * 2) + m_12) - 1);
                }
                condval_1223 = condval_1224;
              }
              int condval_1225;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1225 = 0;
              } else {
                int condval_1226;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1226 = 3;
                } else {
                  condval_1226 = (((slab * 2) + m_12) - 1);
                }
                condval_1225 = condval_1226;
              }
              int condval_1228;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1228 = 0;
              } else {
                int condval_1229;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1229 = 3;
                } else {
                  condval_1229 = (((slab * 2) + m_12) - 1);
                }
                condval_1228 = condval_1229;
              }
              int condval_1230;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1230 = 0;
              } else {
                int condval_1231;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1231 = 3;
                } else {
                  condval_1231 = (((slab * 2) + m_12) - 1);
                }
                condval_1230 = condval_1231;
              }
              int condval_1227;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1228 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1230 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1232;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1232 = 0;
                } else {
                  int condval_1233;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1233 = 3;
                  } else {
                    condval_1233 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1232 = condval_1233;
                }
                int condval_1234;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1234 = 0;
                } else {
                  int condval_1235;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1235 = 3;
                  } else {
                    condval_1235 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1234 = condval_1235;
                }
                condval_1227 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1232 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1234 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1236;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1236 = 0;
                } else {
                  int condval_1237;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1237 = 3;
                  } else {
                    condval_1237 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1236 = condval_1237;
                }
                int condval_1238;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1238 = 0;
                } else {
                  int condval_1239;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1239 = 3;
                  } else {
                    condval_1239 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1238 = condval_1239;
                }
                condval_1227 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1236 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1238 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1240;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1240 = 0;
              } else {
                int condval_1241;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1241 = 3;
                } else {
                  condval_1241 = (((slab * 2) + m_12) - 1);
                }
                condval_1240 = condval_1241;
              }
              int condval_1242;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1242 = 0;
              } else {
                int condval_1243;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1243 = 3;
                } else {
                  condval_1243 = (((slab * 2) + m_12) - 1);
                }
                condval_1242 = condval_1243;
              }
              int condval_1244;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1244 = 0;
              } else {
                int condval_1245;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1245 = 3;
                } else {
                  condval_1245 = (((slab * 2) + m_12) - 1);
                }
                condval_1244 = condval_1245;
              }
              int condval_1247;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1247 = 0;
              } else {
                int condval_1248;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1248 = 3;
                } else {
                  condval_1248 = (((slab * 2) + m_12) - 1);
                }
                condval_1247 = condval_1248;
              }
              int condval_1249;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1249 = 0;
              } else {
                int condval_1250;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1250 = 3;
                } else {
                  condval_1250 = (((slab * 2) + m_12) - 1);
                }
                condval_1249 = condval_1250;
              }
              int condval_1246;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1247 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1249 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1251;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1251 = 0;
                } else {
                  int condval_1252;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1252 = 3;
                  } else {
                    condval_1252 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1251 = condval_1252;
                }
                int condval_1253;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1253 = 0;
                } else {
                  int condval_1254;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1254 = 3;
                  } else {
                    condval_1254 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1253 = condval_1254;
                }
                condval_1246 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1251 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1253 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1255;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1255 = 0;
                } else {
                  int condval_1256;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1256 = 3;
                  } else {
                    condval_1256 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1255 = condval_1256;
                }
                int condval_1257;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1257 = 0;
                } else {
                  int condval_1258;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1258 = 3;
                  } else {
                    condval_1258 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1257 = condval_1258;
                }
                condval_1246 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1255 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1257 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1259;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1259 = 0;
              } else {
                int condval_1260;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1260 = 3;
                } else {
                  condval_1260 = (((slab * 2) + m_12) - 1);
                }
                condval_1259 = condval_1260;
              }
              int condval_1261;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1261 = 0;
              } else {
                int condval_1262;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1262 = 3;
                } else {
                  condval_1262 = (((slab * 2) + m_12) - 1);
                }
                condval_1261 = condval_1262;
              }
              int condval_1263;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1263 = 0;
              } else {
                int condval_1264;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1264 = 3;
                } else {
                  condval_1264 = (((slab * 2) + m_12) - 1);
                }
                condval_1263 = condval_1264;
              }
              int condval_1266;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1266 = 0;
              } else {
                int condval_1267;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1267 = 3;
                } else {
                  condval_1267 = (((slab * 2) + m_12) - 1);
                }
                condval_1266 = condval_1267;
              }
              int condval_1268;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1268 = 0;
              } else {
                int condval_1269;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1269 = 3;
                } else {
                  condval_1269 = (((slab * 2) + m_12) - 1);
                }
                condval_1268 = condval_1269;
              }
              int condval_1265;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1266 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1268 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1270;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1270 = 0;
                } else {
                  int condval_1271;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1271 = 3;
                  } else {
                    condval_1271 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1270 = condval_1271;
                }
                int condval_1272;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1272 = 0;
                } else {
                  int condval_1273;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1273 = 3;
                  } else {
                    condval_1273 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1272 = condval_1273;
                }
                condval_1265 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1270 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1272 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1274;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1274 = 0;
                } else {
                  int condval_1275;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1275 = 3;
                  } else {
                    condval_1275 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1274 = condval_1275;
                }
                int condval_1276;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1276 = 0;
                } else {
                  int condval_1277;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1277 = 3;
                  } else {
                    condval_1277 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1276 = condval_1277;
                }
                condval_1265 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1274 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1276 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1278;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1278 = 0;
              } else {
                int condval_1279;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1279 = 3;
                } else {
                  condval_1279 = (((slab * 2) + m_12) - 1);
                }
                condval_1278 = condval_1279;
              }
              int condval_1280;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1280 = 0;
              } else {
                int condval_1281;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1281 = 3;
                } else {
                  condval_1281 = (((slab * 2) + m_12) - 1);
                }
                condval_1280 = condval_1281;
              }
              int condval_1282;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1282 = 0;
              } else {
                int condval_1283;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1283 = 3;
                } else {
                  condval_1283 = (((slab * 2) + m_12) - 1);
                }
                condval_1282 = condval_1283;
              }
              int condval_1285;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1285 = 0;
              } else {
                int condval_1286;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1286 = 3;
                } else {
                  condval_1286 = (((slab * 2) + m_12) - 1);
                }
                condval_1285 = condval_1286;
              }
              int condval_1287;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1287 = 0;
              } else {
                int condval_1288;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1288 = 3;
                } else {
                  condval_1288 = (((slab * 2) + m_12) - 1);
                }
                condval_1287 = condval_1288;
              }
              int condval_1284;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1285 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1287 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1289;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1289 = 0;
                } else {
                  int condval_1290;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1290 = 3;
                  } else {
                    condval_1290 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1289 = condval_1290;
                }
                int condval_1291;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1291 = 0;
                } else {
                  int condval_1292;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1292 = 3;
                  } else {
                    condval_1292 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1291 = condval_1292;
                }
                condval_1284 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1289 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1291 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1293;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1293 = 0;
                } else {
                  int condval_1294;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1294 = 3;
                  } else {
                    condval_1294 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1293 = condval_1294;
                }
                int condval_1295;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1295 = 0;
                } else {
                  int condval_1296;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1296 = 3;
                  } else {
                    condval_1296 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1295 = condval_1296;
                }
                condval_1284 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1293 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1295 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1297;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1297 = 0;
              } else {
                int condval_1298;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1298 = 3;
                } else {
                  condval_1298 = (((slab * 2) + m_12) - 1);
                }
                condval_1297 = condval_1298;
              }
              int condval_1299;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1299 = 0;
              } else {
                int condval_1300;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1300 = 3;
                } else {
                  condval_1300 = (((slab * 2) + m_12) - 1);
                }
                condval_1299 = condval_1300;
              }
              int condval_1301;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1301 = 0;
              } else {
                int condval_1302;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1302 = 3;
                } else {
                  condval_1302 = (((slab * 2) + m_12) - 1);
                }
                condval_1301 = condval_1302;
              }
              int condval_1304;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1304 = 0;
              } else {
                int condval_1305;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1305 = 3;
                } else {
                  condval_1305 = (((slab * 2) + m_12) - 1);
                }
                condval_1304 = condval_1305;
              }
              int condval_1306;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1306 = 0;
              } else {
                int condval_1307;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1307 = 3;
                } else {
                  condval_1307 = (((slab * 2) + m_12) - 1);
                }
                condval_1306 = condval_1307;
              }
              int condval_1303;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1304 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1306 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1308;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1308 = 0;
                } else {
                  int condval_1309;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1309 = 3;
                  } else {
                    condval_1309 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1308 = condval_1309;
                }
                int condval_1310;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1310 = 0;
                } else {
                  int condval_1311;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1311 = 3;
                  } else {
                    condval_1311 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1310 = condval_1311;
                }
                condval_1303 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1308 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1310 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1312;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1312 = 0;
                } else {
                  int condval_1313;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1313 = 3;
                  } else {
                    condval_1313 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1312 = condval_1313;
                }
                int condval_1314;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1314 = 0;
                } else {
                  int condval_1315;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1315 = 3;
                  } else {
                    condval_1315 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1314 = condval_1315;
                }
                condval_1303 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1312 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1314 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1316;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1316 = 0;
              } else {
                int condval_1317;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1317 = 3;
                } else {
                  condval_1317 = (((slab * 2) + m_12) - 1);
                }
                condval_1316 = condval_1317;
              }
              int condval_1318;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1318 = 0;
              } else {
                int condval_1319;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1319 = 3;
                } else {
                  condval_1319 = (((slab * 2) + m_12) - 1);
                }
                condval_1318 = condval_1319;
              }
              int condval_1320;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1320 = 0;
              } else {
                int condval_1321;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1321 = 3;
                } else {
                  condval_1321 = (((slab * 2) + m_12) - 1);
                }
                condval_1320 = condval_1321;
              }
              int condval_1322;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1322 = 0;
              } else {
                int condval_1323;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1323 = 3;
                } else {
                  condval_1323 = (((slab * 2) + m_12) - 1);
                }
                condval_1322 = condval_1323;
              }
              int condval_1324;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1324 = 0;
              } else {
                int condval_1325;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1325 = 3;
                } else {
                  condval_1325 = (((slab * 2) + m_12) - 1);
                }
                condval_1324 = condval_1325;
              }
              int condval_1327;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1327 = 0;
              } else {
                int condval_1328;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1328 = 3;
                } else {
                  condval_1328 = (((slab * 2) + m_12) - 1);
                }
                condval_1327 = condval_1328;
              }
              int condval_1329;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1329 = 0;
              } else {
                int condval_1330;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1330 = 3;
                } else {
                  condval_1330 = (((slab * 2) + m_12) - 1);
                }
                condval_1329 = condval_1330;
              }
              int condval_1326;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1327 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1329 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1331;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1331 = 0;
                } else {
                  int condval_1332;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1332 = 3;
                  } else {
                    condval_1332 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1331 = condval_1332;
                }
                int condval_1333;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1333 = 0;
                } else {
                  int condval_1334;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1334 = 3;
                  } else {
                    condval_1334 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1333 = condval_1334;
                }
                condval_1326 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1331 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1333 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1335;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1335 = 0;
                } else {
                  int condval_1336;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1336 = 3;
                  } else {
                    condval_1336 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1335 = condval_1336;
                }
                int condval_1337;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1337 = 0;
                } else {
                  int condval_1338;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1338 = 3;
                  } else {
                    condval_1338 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1337 = condval_1338;
                }
                condval_1326 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1335 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1337 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1339;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1339 = 0;
              } else {
                int condval_1340;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1340 = 3;
                } else {
                  condval_1340 = (((slab * 2) + m_12) - 1);
                }
                condval_1339 = condval_1340;
              }
              int condval_1342;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1342 = 0;
              } else {
                int condval_1343;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1343 = 3;
                } else {
                  condval_1343 = (((slab * 2) + m_12) - 1);
                }
                condval_1342 = condval_1343;
              }
              int condval_1344;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1344 = 0;
              } else {
                int condval_1345;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1345 = 3;
                } else {
                  condval_1345 = (((slab * 2) + m_12) - 1);
                }
                condval_1344 = condval_1345;
              }
              int condval_1341;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1342 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1344 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1346;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1346 = 0;
                } else {
                  int condval_1347;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1347 = 3;
                  } else {
                    condval_1347 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1346 = condval_1347;
                }
                int condval_1348;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1348 = 0;
                } else {
                  int condval_1349;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1349 = 3;
                  } else {
                    condval_1349 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1348 = condval_1349;
                }
                condval_1341 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1346 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1348 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1350;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1350 = 0;
                } else {
                  int condval_1351;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1351 = 3;
                  } else {
                    condval_1351 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1350 = condval_1351;
                }
                int condval_1352;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1352 = 0;
                } else {
                  int condval_1353;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1353 = 3;
                  } else {
                    condval_1353 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1352 = condval_1353;
                }
                condval_1341 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1350 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1352 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1354;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1354 = 0;
              } else {
                int condval_1355;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1355 = 3;
                } else {
                  condval_1355 = (((slab * 2) + m_12) - 1);
                }
                condval_1354 = condval_1355;
              }
              int condval_1356;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1356 = 0;
              } else {
                int condval_1357;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1357 = 3;
                } else {
                  condval_1357 = (((slab * 2) + m_12) - 1);
                }
                condval_1356 = condval_1357;
              }
              int condval_1358;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1358 = 0;
              } else {
                int condval_1359;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1359 = 3;
                } else {
                  condval_1359 = (((slab * 2) + m_12) - 1);
                }
                condval_1358 = condval_1359;
              }
              int condval_1361;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1361 = 0;
              } else {
                int condval_1362;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1362 = 3;
                } else {
                  condval_1362 = (((slab * 2) + m_12) - 1);
                }
                condval_1361 = condval_1362;
              }
              int condval_1363;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1363 = 0;
              } else {
                int condval_1364;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1364 = 3;
                } else {
                  condval_1364 = (((slab * 2) + m_12) - 1);
                }
                condval_1363 = condval_1364;
              }
              int condval_1360;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1361 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1363 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1365;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1365 = 0;
                } else {
                  int condval_1366;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1366 = 3;
                  } else {
                    condval_1366 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1365 = condval_1366;
                }
                int condval_1367;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1367 = 0;
                } else {
                  int condval_1368;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1368 = 3;
                  } else {
                    condval_1368 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1367 = condval_1368;
                }
                condval_1360 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1365 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1367 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1369;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1369 = 0;
                } else {
                  int condval_1370;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1370 = 3;
                  } else {
                    condval_1370 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1369 = condval_1370;
                }
                int condval_1371;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1371 = 0;
                } else {
                  int condval_1372;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1372 = 3;
                  } else {
                    condval_1372 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1371 = condval_1372;
                }
                condval_1360 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1369 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1371 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1373;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1373 = 0;
              } else {
                int condval_1374;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1374 = 3;
                } else {
                  condval_1374 = (((slab * 2) + m_12) - 1);
                }
                condval_1373 = condval_1374;
              }
              int condval_1375;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1375 = 0;
              } else {
                int condval_1376;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1376 = 3;
                } else {
                  condval_1376 = (((slab * 2) + m_12) - 1);
                }
                condval_1375 = condval_1376;
              }
              int condval_1377;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1377 = 0;
              } else {
                int condval_1378;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1378 = 3;
                } else {
                  condval_1378 = (((slab * 2) + m_12) - 1);
                }
                condval_1377 = condval_1378;
              }
              int condval_1380;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1380 = 0;
              } else {
                int condval_1381;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1381 = 3;
                } else {
                  condval_1381 = (((slab * 2) + m_12) - 1);
                }
                condval_1380 = condval_1381;
              }
              int condval_1382;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1382 = 0;
              } else {
                int condval_1383;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1383 = 3;
                } else {
                  condval_1383 = (((slab * 2) + m_12) - 1);
                }
                condval_1382 = condval_1383;
              }
              int condval_1379;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1380 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1382 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1384;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1384 = 0;
                } else {
                  int condval_1385;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1385 = 3;
                  } else {
                    condval_1385 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1384 = condval_1385;
                }
                int condval_1386;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1386 = 0;
                } else {
                  int condval_1387;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1387 = 3;
                  } else {
                    condval_1387 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1386 = condval_1387;
                }
                condval_1379 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1384 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1386 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1388;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1388 = 0;
                } else {
                  int condval_1389;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1389 = 3;
                  } else {
                    condval_1389 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1388 = condval_1389;
                }
                int condval_1390;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1390 = 0;
                } else {
                  int condval_1391;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1391 = 3;
                  } else {
                    condval_1391 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1390 = condval_1391;
                }
                condval_1379 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1388 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1390 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1392;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1392 = 0;
              } else {
                int condval_1393;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1393 = 3;
                } else {
                  condval_1393 = (((slab * 2) + m_12) - 1);
                }
                condval_1392 = condval_1393;
              }
              int condval_1394;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1394 = 0;
              } else {
                int condval_1395;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1395 = 3;
                } else {
                  condval_1395 = (((slab * 2) + m_12) - 1);
                }
                condval_1394 = condval_1395;
              }
              int condval_1396;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1396 = 0;
              } else {
                int condval_1397;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1397 = 3;
                } else {
                  condval_1397 = (((slab * 2) + m_12) - 1);
                }
                condval_1396 = condval_1397;
              }
              int condval_1399;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1399 = 0;
              } else {
                int condval_1400;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1400 = 3;
                } else {
                  condval_1400 = (((slab * 2) + m_12) - 1);
                }
                condval_1399 = condval_1400;
              }
              int condval_1401;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1401 = 0;
              } else {
                int condval_1402;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1402 = 3;
                } else {
                  condval_1402 = (((slab * 2) + m_12) - 1);
                }
                condval_1401 = condval_1402;
              }
              int condval_1398;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1399 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1401 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1403;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1403 = 0;
                } else {
                  int condval_1404;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1404 = 3;
                  } else {
                    condval_1404 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1403 = condval_1404;
                }
                int condval_1405;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1405 = 0;
                } else {
                  int condval_1406;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1406 = 3;
                  } else {
                    condval_1406 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1405 = condval_1406;
                }
                condval_1398 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1403 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1405 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1407;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1407 = 0;
                } else {
                  int condval_1408;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1408 = 3;
                  } else {
                    condval_1408 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1407 = condval_1408;
                }
                int condval_1409;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1409 = 0;
                } else {
                  int condval_1410;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1410 = 3;
                  } else {
                    condval_1410 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1409 = condval_1410;
                }
                condval_1398 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1407 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1409 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1411;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1411 = 0;
              } else {
                int condval_1412;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1412 = 3;
                } else {
                  condval_1412 = (((slab * 2) + m_12) - 1);
                }
                condval_1411 = condval_1412;
              }
              int condval_1413;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1413 = 0;
              } else {
                int condval_1414;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1414 = 3;
                } else {
                  condval_1414 = (((slab * 2) + m_12) - 1);
                }
                condval_1413 = condval_1414;
              }
              int condval_1415;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1415 = 0;
              } else {
                int condval_1416;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1416 = 3;
                } else {
                  condval_1416 = (((slab * 2) + m_12) - 1);
                }
                condval_1415 = condval_1416;
              }
              int condval_1418;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1418 = 0;
              } else {
                int condval_1419;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1419 = 3;
                } else {
                  condval_1419 = (((slab * 2) + m_12) - 1);
                }
                condval_1418 = condval_1419;
              }
              int condval_1420;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1420 = 0;
              } else {
                int condval_1421;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1421 = 3;
                } else {
                  condval_1421 = (((slab * 2) + m_12) - 1);
                }
                condval_1420 = condval_1421;
              }
              int condval_1417;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1418 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1420 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1422;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1422 = 0;
                } else {
                  int condval_1423;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1423 = 3;
                  } else {
                    condval_1423 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1422 = condval_1423;
                }
                int condval_1424;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1424 = 0;
                } else {
                  int condval_1425;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1425 = 3;
                  } else {
                    condval_1425 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1424 = condval_1425;
                }
                condval_1417 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1422 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1424 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1426;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1426 = 0;
                } else {
                  int condval_1427;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1427 = 3;
                  } else {
                    condval_1427 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1426 = condval_1427;
                }
                int condval_1428;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1428 = 0;
                } else {
                  int condval_1429;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1429 = 3;
                  } else {
                    condval_1429 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1428 = condval_1429;
                }
                condval_1417 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1426 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1428 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1430;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1430 = 0;
              } else {
                int condval_1431;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1431 = 3;
                } else {
                  condval_1431 = (((slab * 2) + m_12) - 1);
                }
                condval_1430 = condval_1431;
              }
              int condval_1432;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1432 = 0;
              } else {
                int condval_1433;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1433 = 3;
                } else {
                  condval_1433 = (((slab * 2) + m_12) - 1);
                }
                condval_1432 = condval_1433;
              }
              int condval_1434;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1434 = 0;
              } else {
                int condval_1435;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1435 = 3;
                } else {
                  condval_1435 = (((slab * 2) + m_12) - 1);
                }
                condval_1434 = condval_1435;
              }
              int condval_1437;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1437 = 0;
              } else {
                int condval_1438;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1438 = 3;
                } else {
                  condval_1438 = (((slab * 2) + m_12) - 1);
                }
                condval_1437 = condval_1438;
              }
              int condval_1439;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1439 = 0;
              } else {
                int condval_1440;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1440 = 3;
                } else {
                  condval_1440 = (((slab * 2) + m_12) - 1);
                }
                condval_1439 = condval_1440;
              }
              int condval_1436;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1437 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1439 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1441;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1441 = 0;
                } else {
                  int condval_1442;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1442 = 3;
                  } else {
                    condval_1442 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1441 = condval_1442;
                }
                int condval_1443;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1443 = 0;
                } else {
                  int condval_1444;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1444 = 3;
                  } else {
                    condval_1444 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1443 = condval_1444;
                }
                condval_1436 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1441 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1443 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1445;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1445 = 0;
                } else {
                  int condval_1446;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1446 = 3;
                  } else {
                    condval_1446 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1445 = condval_1446;
                }
                int condval_1447;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1447 = 0;
                } else {
                  int condval_1448;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1448 = 3;
                  } else {
                    condval_1448 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1447 = condval_1448;
                }
                condval_1436 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1445 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1447 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1449;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1449 = 0;
              } else {
                int condval_1450;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1450 = 3;
                } else {
                  condval_1450 = (((slab * 2) + m_12) - 1);
                }
                condval_1449 = condval_1450;
              }
              int condval_1451;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1451 = 0;
              } else {
                int condval_1452;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1452 = 3;
                } else {
                  condval_1452 = (((slab * 2) + m_12) - 1);
                }
                condval_1451 = condval_1452;
              }
              int condval_1453;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1453 = 0;
              } else {
                int condval_1454;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1454 = 3;
                } else {
                  condval_1454 = (((slab * 2) + m_12) - 1);
                }
                condval_1453 = condval_1454;
              }
              int condval_1455;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1455 = 0;
              } else {
                int condval_1456;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1456 = 3;
                } else {
                  condval_1456 = (((slab * 2) + m_12) - 1);
                }
                condval_1455 = condval_1456;
              }
              int condval_1457;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1457 = 0;
              } else {
                int condval_1458;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1458 = 3;
                } else {
                  condval_1458 = (((slab * 2) + m_12) - 1);
                }
                condval_1457 = condval_1458;
              }
              int condval_1459;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1459 = 0;
              } else {
                int condval_1460;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1460 = 3;
                } else {
                  condval_1460 = (((slab * 2) + m_12) - 1);
                }
                condval_1459 = condval_1460;
              }
              int condval_1461;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1461 = 0;
              } else {
                int condval_1462;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1462 = 3;
                } else {
                  condval_1462 = (((slab * 2) + m_12) - 1);
                }
                condval_1461 = condval_1462;
              }
              int condval_1463;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1463 = 0;
              } else {
                int condval_1464;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1464 = 3;
                } else {
                  condval_1464 = (((slab * 2) + m_12) - 1);
                }
                condval_1463 = condval_1464;
              }
              int condval_1465;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1465 = 0;
              } else {
                int condval_1466;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1466 = 3;
                } else {
                  condval_1466 = (((slab * 2) + m_12) - 1);
                }
                condval_1465 = condval_1466;
              }
              int condval_1467;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1467 = 0;
              } else {
                int condval_1468;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1468 = 3;
                } else {
                  condval_1468 = (((slab * 2) + m_12) - 1);
                }
                condval_1467 = condval_1468;
              }
              int condval_1469;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1469 = 0;
              } else {
                int condval_1470;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1470 = 3;
                } else {
                  condval_1470 = (((slab * 2) + m_12) - 1);
                }
                condval_1469 = condval_1470;
              }
              int condval_1472;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1472 = 0;
              } else {
                int condval_1473;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1473 = 3;
                } else {
                  condval_1473 = (((slab * 2) + m_12) - 1);
                }
                condval_1472 = condval_1473;
              }
              int condval_1474;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1474 = 0;
              } else {
                int condval_1475;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1475 = 3;
                } else {
                  condval_1475 = (((slab * 2) + m_12) - 1);
                }
                condval_1474 = condval_1475;
              }
              int condval_1471;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1472 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1474 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1476;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1476 = 0;
                } else {
                  int condval_1477;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1477 = 3;
                  } else {
                    condval_1477 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1476 = condval_1477;
                }
                int condval_1478;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1478 = 0;
                } else {
                  int condval_1479;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1479 = 3;
                  } else {
                    condval_1479 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1478 = condval_1479;
                }
                condval_1471 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1476 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1478 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1480;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1480 = 0;
                } else {
                  int condval_1481;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1481 = 3;
                  } else {
                    condval_1481 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1480 = condval_1481;
                }
                int condval_1482;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1482 = 0;
                } else {
                  int condval_1483;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1483 = 3;
                  } else {
                    condval_1483 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1482 = condval_1483;
                }
                condval_1471 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1480 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1482 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1484;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1484 = 0;
              } else {
                int condval_1485;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1485 = 3;
                } else {
                  condval_1485 = (((slab * 2) + m_12) - 1);
                }
                condval_1484 = condval_1485;
              }
              int condval_1486;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1486 = 0;
              } else {
                int condval_1487;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1487 = 3;
                } else {
                  condval_1487 = (((slab * 2) + m_12) - 1);
                }
                condval_1486 = condval_1487;
              }
              int condval_1488;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1488 = 0;
              } else {
                int condval_1489;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1489 = 3;
                } else {
                  condval_1489 = (((slab * 2) + m_12) - 1);
                }
                condval_1488 = condval_1489;
              }
              int condval_1491;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1491 = 0;
              } else {
                int condval_1492;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1492 = 3;
                } else {
                  condval_1492 = (((slab * 2) + m_12) - 1);
                }
                condval_1491 = condval_1492;
              }
              int condval_1493;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1493 = 0;
              } else {
                int condval_1494;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1494 = 3;
                } else {
                  condval_1494 = (((slab * 2) + m_12) - 1);
                }
                condval_1493 = condval_1494;
              }
              int condval_1490;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1491 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1493 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1495;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1495 = 0;
                } else {
                  int condval_1496;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1496 = 3;
                  } else {
                    condval_1496 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1495 = condval_1496;
                }
                int condval_1497;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1497 = 0;
                } else {
                  int condval_1498;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1498 = 3;
                  } else {
                    condval_1498 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1497 = condval_1498;
                }
                condval_1490 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1495 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1497 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1499;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1499 = 0;
                } else {
                  int condval_1500;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1500 = 3;
                  } else {
                    condval_1500 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1499 = condval_1500;
                }
                int condval_1501;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1501 = 0;
                } else {
                  int condval_1502;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1502 = 3;
                  } else {
                    condval_1502 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1501 = condval_1502;
                }
                condval_1490 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1499 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1501 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1503;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1503 = 0;
              } else {
                int condval_1504;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1504 = 3;
                } else {
                  condval_1504 = (((slab * 2) + m_12) - 1);
                }
                condval_1503 = condval_1504;
              }
              int condval_1505;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1505 = 0;
              } else {
                int condval_1506;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1506 = 3;
                } else {
                  condval_1506 = (((slab * 2) + m_12) - 1);
                }
                condval_1505 = condval_1506;
              }
              int condval_1507;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1507 = 0;
              } else {
                int condval_1508;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1508 = 3;
                } else {
                  condval_1508 = (((slab * 2) + m_12) - 1);
                }
                condval_1507 = condval_1508;
              }
              int condval_1510;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1510 = 0;
              } else {
                int condval_1511;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1511 = 3;
                } else {
                  condval_1511 = (((slab * 2) + m_12) - 1);
                }
                condval_1510 = condval_1511;
              }
              int condval_1512;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1512 = 0;
              } else {
                int condval_1513;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1513 = 3;
                } else {
                  condval_1513 = (((slab * 2) + m_12) - 1);
                }
                condval_1512 = condval_1513;
              }
              int condval_1509;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1510 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1512 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1514;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1514 = 0;
                } else {
                  int condval_1515;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1515 = 3;
                  } else {
                    condval_1515 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1514 = condval_1515;
                }
                int condval_1516;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1516 = 0;
                } else {
                  int condval_1517;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1517 = 3;
                  } else {
                    condval_1517 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1516 = condval_1517;
                }
                condval_1509 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1514 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1516 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1518;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1518 = 0;
                } else {
                  int condval_1519;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1519 = 3;
                  } else {
                    condval_1519 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1518 = condval_1519;
                }
                int condval_1520;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1520 = 0;
                } else {
                  int condval_1521;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1521 = 3;
                  } else {
                    condval_1521 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1520 = condval_1521;
                }
                condval_1509 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1518 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1520 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1522;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1522 = 0;
              } else {
                int condval_1523;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1523 = 3;
                } else {
                  condval_1523 = (((slab * 2) + m_12) - 1);
                }
                condval_1522 = condval_1523;
              }
              int condval_1524;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1524 = 0;
              } else {
                int condval_1525;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1525 = 3;
                } else {
                  condval_1525 = (((slab * 2) + m_12) - 1);
                }
                condval_1524 = condval_1525;
              }
              int condval_1526;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1526 = 0;
              } else {
                int condval_1527;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1527 = 3;
                } else {
                  condval_1527 = (((slab * 2) + m_12) - 1);
                }
                condval_1526 = condval_1527;
              }
              int condval_1529;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1529 = 0;
              } else {
                int condval_1530;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1530 = 3;
                } else {
                  condval_1530 = (((slab * 2) + m_12) - 1);
                }
                condval_1529 = condval_1530;
              }
              int condval_1531;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1531 = 0;
              } else {
                int condval_1532;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1532 = 3;
                } else {
                  condval_1532 = (((slab * 2) + m_12) - 1);
                }
                condval_1531 = condval_1532;
              }
              int condval_1528;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1529 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1531 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1533;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1533 = 0;
                } else {
                  int condval_1534;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1534 = 3;
                  } else {
                    condval_1534 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1533 = condval_1534;
                }
                int condval_1535;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1535 = 0;
                } else {
                  int condval_1536;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1536 = 3;
                  } else {
                    condval_1536 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1535 = condval_1536;
                }
                condval_1528 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1533 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1535 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1537;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1537 = 0;
                } else {
                  int condval_1538;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1538 = 3;
                  } else {
                    condval_1538 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1537 = condval_1538;
                }
                int condval_1539;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1539 = 0;
                } else {
                  int condval_1540;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1540 = 3;
                  } else {
                    condval_1540 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1539 = condval_1540;
                }
                condval_1528 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1537 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1539 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1541;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1541 = 0;
              } else {
                int condval_1542;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1542 = 3;
                } else {
                  condval_1542 = (((slab * 2) + m_12) - 1);
                }
                condval_1541 = condval_1542;
              }
              int condval_1543;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1543 = 0;
              } else {
                int condval_1544;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1544 = 3;
                } else {
                  condval_1544 = (((slab * 2) + m_12) - 1);
                }
                condval_1543 = condval_1544;
              }
              int condval_1545;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1545 = 0;
              } else {
                int condval_1546;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1546 = 3;
                } else {
                  condval_1546 = (((slab * 2) + m_12) - 1);
                }
                condval_1545 = condval_1546;
              }
              int condval_1548;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1548 = 0;
              } else {
                int condval_1549;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1549 = 3;
                } else {
                  condval_1549 = (((slab * 2) + m_12) - 1);
                }
                condval_1548 = condval_1549;
              }
              int condval_1550;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1550 = 0;
              } else {
                int condval_1551;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1551 = 3;
                } else {
                  condval_1551 = (((slab * 2) + m_12) - 1);
                }
                condval_1550 = condval_1551;
              }
              int condval_1547;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1548 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1550 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1552;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1552 = 0;
                } else {
                  int condval_1553;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1553 = 3;
                  } else {
                    condval_1553 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1552 = condval_1553;
                }
                int condval_1554;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1554 = 0;
                } else {
                  int condval_1555;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1555 = 3;
                  } else {
                    condval_1555 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1554 = condval_1555;
                }
                condval_1547 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1552 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1554 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1556;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1556 = 0;
                } else {
                  int condval_1557;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1557 = 3;
                  } else {
                    condval_1557 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1556 = condval_1557;
                }
                int condval_1558;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1558 = 0;
                } else {
                  int condval_1559;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1559 = 3;
                  } else {
                    condval_1559 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1558 = condval_1559;
                }
                condval_1547 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1556 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1558 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1560;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1560 = 0;
              } else {
                int condval_1561;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1561 = 3;
                } else {
                  condval_1561 = (((slab * 2) + m_12) - 1);
                }
                condval_1560 = condval_1561;
              }
              int condval_1562;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1562 = 0;
              } else {
                int condval_1563;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1563 = 3;
                } else {
                  condval_1563 = (((slab * 2) + m_12) - 1);
                }
                condval_1562 = condval_1563;
              }
              int condval_1564;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1564 = 0;
              } else {
                int condval_1565;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1565 = 3;
                } else {
                  condval_1565 = (((slab * 2) + m_12) - 1);
                }
                condval_1564 = condval_1565;
              }
              int condval_1567;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1567 = 0;
              } else {
                int condval_1568;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1568 = 3;
                } else {
                  condval_1568 = (((slab * 2) + m_12) - 1);
                }
                condval_1567 = condval_1568;
              }
              int condval_1569;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1569 = 0;
              } else {
                int condval_1570;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1570 = 3;
                } else {
                  condval_1570 = (((slab * 2) + m_12) - 1);
                }
                condval_1569 = condval_1570;
              }
              int condval_1566;
              if ((0 < (((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1567 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1569 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
                int condval_1571;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1571 = 0;
                } else {
                  int condval_1572;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1572 = 3;
                  } else {
                    condval_1572 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1571 = condval_1572;
                }
                int condval_1573;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1573 = 0;
                } else {
                  int condval_1574;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1574 = 3;
                  } else {
                    condval_1574 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1573 = condval_1574;
                }
                condval_1566 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1571 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1573 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
              } else {
                int condval_1575;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1575 = 0;
                } else {
                  int condval_1576;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1576 = 3;
                  } else {
                    condval_1576 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1575 = condval_1576;
                }
                int condval_1577;
                if ((((slab * 2) + m_12) == 0)) {
                  condval_1577 = 0;
                } else {
                  int condval_1578;
                  if ((((slab * 2) + m_12) == 1)) {
                    condval_1578 = 3;
                  } else {
                    condval_1578 = (((slab * 2) + m_12) - 1);
                  }
                  condval_1577 = condval_1578;
                }
                condval_1566 = (((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1575 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1577 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
              }
              int condval_1579;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1579 = 0;
              } else {
                int condval_1580;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1580 = 3;
                } else {
                  condval_1580 = (((slab * 2) + m_12) - 1);
                }
                condval_1579 = condval_1580;
              }
              int condval_1581;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1581 = 0;
              } else {
                int condval_1582;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1582 = 3;
                } else {
                  condval_1582 = (((slab * 2) + m_12) - 1);
                }
                condval_1581 = condval_1582;
              }
              int condval_1583;
              if ((((slab * 2) + m_12) == 0)) {
                condval_1583 = 0;
              } else {
                int condval_1584;
                if ((((slab * 2) + m_12) == 1)) {
                  condval_1584 = 3;
                } else {
                  condval_1584 = (((slab * 2) + m_12) - 1);
                }
                condval_1583 = condval_1584;
              }
              condval_1199 = (((((((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_1208 & 3) * 16) + (((((condval_1221 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1223 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1225 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_1227 & 3) * 16) + (((((condval_1240 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1242 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1244 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_1246 & 3) * 16) + (((((condval_1259 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1261 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1263 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_1265 & 3) * 16) + (((((condval_1278 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1280 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1282 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_1284 & 3) * 16) + (((((condval_1297 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1299 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1301 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_1303 & 3) * 16) + (((((condval_1316 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1318 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1320 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + ((((((((((((int)blockIdx.x) / 40) * 2) + ((int)(((((condval_1322 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_1324 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_1326 * 2)) + ((((condval_1339 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) * 40960)) + (((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_1341 & 3) * 16) + (((((condval_1354 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1356 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1358 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_1360 & 3) * 16) + (((((condval_1373 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1375 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1377 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_1379 & 3) * 16) + (((((condval_1392 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1394 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1396 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_1398 & 3) * 16) + (((((condval_1411 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1413 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1415 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_1417 & 3) * 16) + (((((condval_1430 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1432 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1434 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_1436 & 3) * 16) + (((((condval_1449 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1451 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1453 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1455 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1457 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_1459 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1461 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + ((((((((((((int)blockIdx.x) % 40) * 2) + ((int)(((((condval_1463 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_1465 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_1467 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1469 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_1471 & 3) * 16) + (((((condval_1484 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1486 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1488 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_1490 & 3) * 16) + (((((condval_1503 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1505 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1507 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_1509 & 3) * 16) + (((((condval_1522 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1524 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1526 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_1528 & 3) * 16) + (((((condval_1541 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1543 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1545 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_1547 & 3) * 16) + (((((condval_1560 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1562 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1564 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_1566 & 3) * 16) + (((((condval_1579 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_1581 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_1583 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
            } else {
              condval_1199 = -1;
            }
            nr_tl_shallow_joint::st128((&(X[(condval_1199 + 57507840)])), q[0], q[1], q[2], q[3]);
          }
        }
      }
    }
  }
  if (((int)threadIdx.x) == 0) {
    nr_tl_shallow_joint::st32((&(X[((((int)blockIdx.x) * 4) + 60352768)])), (uint)0);
  }
}

