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
    if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
      condval_1 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
    } else {
      condval_1 = -1;
    }
    int condval;
    if ((0 <= condval_1)) {
      int condval_2;
      if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
        condval_2 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
      } else {
        condval_2 = -1;
      }
      condval = (condval_2 + 52264960);
    } else {
      condval = -1;
    }
    if (0 <= condval) {
      int condval_4;
      if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
        condval_4 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
      } else {
        condval_4 = -1;
      }
      int condval_3;
      if ((0 <= condval_4)) {
        int condval_5;
        if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
          condval_5 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
        } else {
          condval_5 = -1;
        }
        condval_3 = (condval_5 + 52264960);
      } else {
        condval_3 = -1;
      }
      if (condval_3 < 60358048) {
        int condval_7;
        if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
          condval_7 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
        } else {
          condval_7 = -1;
        }
        int condval_6;
        if ((0 <= condval_7)) {
          int condval_8;
          if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
            condval_8 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
          } else {
            condval_8 = -1;
          }
          condval_6 = (condval_8 + 52264960);
        } else {
          condval_6 = -1;
        }
        int condval_10;
        if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
          condval_10 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
        } else {
          condval_10 = -1;
        }
        int condval_9;
        if ((0 <= condval_10)) {
          int condval_11;
          if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
            condval_11 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
          } else {
            condval_11 = -1;
          }
          condval_9 = (condval_11 + 52264960);
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
  ushort v_ = (ushort)18534;
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
          ushort v__1 = (ushort)18534;
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
      int condval_41;
      if ((((slab * 2) + m_12) == 0)) {
        condval_41 = 0;
      } else {
        int condval_42;
        if ((((slab * 2) + m_12) == 1)) {
          condval_42 = 3;
        } else {
          condval_42 = (((slab * 2) + m_12) - 1);
        }
        condval_41 = condval_42;
      }
      int condval_43;
      if ((((slab * 2) + m_12) == 0)) {
        condval_43 = 0;
      } else {
        int condval_44;
        if ((((slab * 2) + m_12) == 1)) {
          condval_44 = 3;
        } else {
          condval_44 = (((slab * 2) + m_12) - 1);
        }
        condval_43 = condval_44;
      }
      int condval_45;
      if ((((slab * 2) + m_12) == 0)) {
        condval_45 = 0;
      } else {
        int condval_46;
        if ((((slab * 2) + m_12) == 1)) {
          condval_46 = 3;
        } else {
          condval_46 = (((slab * 2) + m_12) - 1);
        }
        condval_45 = condval_46;
      }
      int condval_47;
      if ((((slab * 2) + m_12) == 0)) {
        condval_47 = 0;
      } else {
        int condval_48;
        if ((((slab * 2) + m_12) == 1)) {
          condval_48 = 3;
        } else {
          condval_48 = (((slab * 2) + m_12) - 1);
        }
        condval_47 = condval_48;
      }
      int condval_40;
      if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_41 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_43 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_45 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_47 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
        int condval_49;
        if ((((slab * 2) + m_12) == 0)) {
          condval_49 = 0;
        } else {
          int condval_50;
          if ((((slab * 2) + m_12) == 1)) {
            condval_50 = 3;
          } else {
            condval_50 = (((slab * 2) + m_12) - 1);
          }
          condval_49 = condval_50;
        }
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
        condval_40 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((condval_49 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_51 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((condval_53 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_55 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_57 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_59 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
      } else {
        condval_40 = -1;
      }
      if (0 <= condval_40) {
        int condval_62;
        if ((((slab * 2) + m_12) == 0)) {
          condval_62 = 0;
        } else {
          int condval_63;
          if ((((slab * 2) + m_12) == 1)) {
            condval_63 = 3;
          } else {
            condval_63 = (((slab * 2) + m_12) - 1);
          }
          condval_62 = condval_63;
        }
        int condval_64;
        if ((((slab * 2) + m_12) == 0)) {
          condval_64 = 0;
        } else {
          int condval_65;
          if ((((slab * 2) + m_12) == 1)) {
            condval_65 = 3;
          } else {
            condval_65 = (((slab * 2) + m_12) - 1);
          }
          condval_64 = condval_65;
        }
        int condval_66;
        if ((((slab * 2) + m_12) == 0)) {
          condval_66 = 0;
        } else {
          int condval_67;
          if ((((slab * 2) + m_12) == 1)) {
            condval_67 = 3;
          } else {
            condval_67 = (((slab * 2) + m_12) - 1);
          }
          condval_66 = condval_67;
        }
        int condval_68;
        if ((((slab * 2) + m_12) == 0)) {
          condval_68 = 0;
        } else {
          int condval_69;
          if ((((slab * 2) + m_12) == 1)) {
            condval_69 = 3;
          } else {
            condval_69 = (((slab * 2) + m_12) - 1);
          }
          condval_68 = condval_69;
        }
        int condval_61;
        if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_62 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_64 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_66 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_68 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
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
          condval_61 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((condval_70 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_72 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((condval_74 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_76 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_78 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_80 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
        } else {
          condval_61 = -1;
        }
        if (condval_61 < 5471648) {
          int condval_83;
          if ((((slab * 2) + m_12) == 0)) {
            condval_83 = 0;
          } else {
            int condval_84;
            if ((((slab * 2) + m_12) == 1)) {
              condval_84 = 3;
            } else {
              condval_84 = (((slab * 2) + m_12) - 1);
            }
            condval_83 = condval_84;
          }
          int condval_85;
          if ((((slab * 2) + m_12) == 0)) {
            condval_85 = 0;
          } else {
            int condval_86;
            if ((((slab * 2) + m_12) == 1)) {
              condval_86 = 3;
            } else {
              condval_86 = (((slab * 2) + m_12) - 1);
            }
            condval_85 = condval_86;
          }
          int condval_87;
          if ((((slab * 2) + m_12) == 0)) {
            condval_87 = 0;
          } else {
            int condval_88;
            if ((((slab * 2) + m_12) == 1)) {
              condval_88 = 3;
            } else {
              condval_88 = (((slab * 2) + m_12) - 1);
            }
            condval_87 = condval_88;
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
          int condval_82;
          if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_83 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_85 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_87 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_89 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
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
            condval_82 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((condval_91 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_93 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((condval_95 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_97 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_99 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_101 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
          } else {
            condval_82 = -1;
          }
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
          int condval_103;
          if (((((bool)1 & (bool)1) & (1 <= (((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_104 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_106 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & ((((((int)blockIdx.x) % 41) * 2) + ((int)(((((condval_108 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_110 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < 81))) {
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
            condval_103 = ((((((((((((int)blockIdx.x) / 41) * 81920) + (((int)(((((condval_112 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_114 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))) * 40960)) + ((((int)blockIdx.x) % 41) * 1024)) + (((int)(((((condval_116 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_118 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_120 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_122 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
          } else {
            condval_103 = -1;
          }
          nr_tl_shallow_joint::st128((&(X[(condval_103 + 54886400)])), (nr_tl_shallow::e4pair(out[(m_12 * 8)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 2)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 1)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 3)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 4)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 6)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_12 * 8) + 5)]) | (nr_tl_shallow::e4pair(out[((m_12 * 8) + 7)]) << (uint)16)));
        }
      }
    }
  }
  if (((int)threadIdx.x) == 0) {
    nr_tl_shallow_joint::st32((&(X[((((int)blockIdx.x) * 4) + 60347392)])), (uint)0);
  }
}

