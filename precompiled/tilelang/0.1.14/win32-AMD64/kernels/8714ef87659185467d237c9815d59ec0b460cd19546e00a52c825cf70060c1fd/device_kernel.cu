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
    int rmod = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_1 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_1 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_2 = (((int)blockIdx.x) % (W0 >> 4));
    int rmod_3 = (((int)blockIdx.x) % (W0 >> 4));
    int condval;
    if (((((1 <= ((((((0 <= (W0 >> 4)) && (0 <= rmod)) || (((W0 >> 4) < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_1)) || (((W0 >> 4) < 0) && (rmod_1 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) <= (H0 >> 3))) & (0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod_2)) || (((W0 >> 4) < 0) && (rmod_2 <= 0))) ? rmod_2 : (rmod_2 + (W0 >> 4))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_3)) || (((W0 >> 4) < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < (W0 >> 3)))) {
      int rmod_4 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_5 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_2 = (((int)blockIdx.x) / (W0 >> 4));
      condval = (((((((((((0 <= (W0 >> 4)) && (0 <= rmod_4)) || (((W0 >> 4) < 0) && (rmod_4 <= 0))) ? rmod_4 : (rmod_4 + (W0 >> 4))) * 1024) + (((((((((0 <= (W0 >> 4)) && (0 <= rmod_5)) || (((W0 >> 4) < 0) && (rmod_5 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) - 1) * (W0 >> 3)) * 512)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
    } else {
      condval = -1;
    }
    int rmod_6 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_3 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_7 = (((int)blockIdx.x) % (W0 >> 4));
    int rdiv_4 = (((int)blockIdx.x) / (W0 >> 4));
    int rmod_8 = (((int)blockIdx.x) % (W0 >> 4));
    int rmod_9 = (((int)blockIdx.x) % (W0 >> 4));
    int condval_1;
    if (((((1 <= ((((((0 <= (W0 >> 4)) && (0 <= rmod_6)) || (((W0 >> 4) < 0) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_7)) || (((W0 >> 4) < 0) && (rmod_7 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) <= (H0 >> 3))) & (0 <= ((((0 <= (W0 >> 4)) && (0 <= rmod_8)) || (((W0 >> 4) < 0) && (rmod_8 <= 0))) ? rmod_8 : (rmod_8 + (W0 >> 4))))) & (((((((0 <= (W0 >> 4)) && (0 <= rmod_9)) || (((W0 >> 4) < 0) && (rmod_9 <= 0))) ? rmod_9 : (rmod_9 + (W0 >> 4))) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) < (W0 >> 3)))) {
      int rmod_10 = (((int)blockIdx.x) % (W0 >> 4));
      int rmod_11 = (((int)blockIdx.x) % (W0 >> 4));
      int rdiv_5 = (((int)blockIdx.x) / (W0 >> 4));
      condval_1 = (((((((((((0 <= (W0 >> 4)) && (0 <= rmod_10)) || (((W0 >> 4) < 0) && (rmod_10 <= 0))) ? rmod_10 : (rmod_10 + (W0 >> 4))) * 1024) + (((((((((0 <= (W0 >> 4)) && (0 <= rmod_11)) || (((W0 >> 4) < 0) && (rmod_11 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) - 1) * (W0 >> 3)) * 512)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3));
    } else {
      condval_1 = -1;
    }
    int base = max(condval, condval_1);
    int condval_2;
    if ((0 <= base)) {
      condval_2 = (in_offset + base);
    } else {
      condval_2 = -1;
    }
    int condval_3;
    if ((0 <= base)) {
      condval_3 = (in_offset + base);
    } else {
      condval_3 = -1;
    }
    int base_1 = max(condval_2, condval_3);
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
          half_t condval_4;
          if (((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) < sizes_4)) {
            condval_4 = G0[(((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2))];
          } else {
            condval_4 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          half_t condval_5;
          if ((((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_4)) {
            condval_5 = G0[((((p * 16) + (j_3 * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
          } else {
            condval_5 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          ff[((((m * 8) + (p * 4)) + (j_3 * 2)) + i)] = nr_tl_shallow::mul(raw_1, nr_tl_shallow::pack(condval_4, condval_5));
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
            int64_t condval_6;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_6 = (int64_t)0;
            } else {
              int64_t condval_7;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_7 = (int64_t)3;
              } else {
                condval_7 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_6 = condval_7;
            }
            int64_t condval_8;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_8 = (int64_t)0;
            } else {
              int64_t condval_9;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_9 = (int64_t)3;
              } else {
                condval_9 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_8 = condval_9;
            }
            int64_t condval_10;
            if ((((int64_t)m_6) == (int64_t)0)) {
              condval_10 = (int64_t)0;
            } else {
              int64_t condval_11;
              if ((((int64_t)m_6) == (int64_t)1)) {
                condval_11 = (int64_t)3;
              } else {
                condval_11 = (((int64_t)m_6) - (int64_t)1);
              }
              condval_10 = condval_11;
            }
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
            qa[(((m_6 * 4) + (p_4 * 2)) + i_3)] = (nr_tl_shallow::e4pair(z[(((condval_8 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_12 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
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
            kb[(((m_6 * 4) + (i_3 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_16 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3))]) | (nr_tl_shallow::e4pair(z[((((condval_20 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_3)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          int64_t condval_22;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_22 = (int64_t)0;
          } else {
            condval_22 = (int64_t)1;
          }
          int64_t condval_23;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_23 = (int64_t)0;
          } else {
            condval_23 = (int64_t)1;
          }
          int64_t condval_24;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_24 = (int64_t)0;
          } else {
            condval_24 = (int64_t)1;
          }
          int64_t condval_25;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_25 = (int64_t)0;
          } else {
            condval_25 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_3 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_23 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_25 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_26;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_26 = (int64_t)3;
          } else {
            condval_26 = (int64_t)2;
          }
          int64_t condval_27;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_27 = (int64_t)3;
          } else {
            condval_27 = (int64_t)2;
          }
          int64_t condval_28;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_28 = (int64_t)3;
          } else {
            condval_28 = (int64_t)2;
          }
          int64_t condval_29;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_29 = (int64_t)3;
          } else {
            condval_29 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_3 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_27 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_29 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
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
      int64_t condval_30;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_30 = (int64_t)0;
      } else {
        int64_t condval_31;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_31 = (int64_t)3;
        } else {
          condval_31 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_30 = condval_31;
      }
      int64_t condval_32;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)0)) {
        condval_32 = (int64_t)0;
      } else {
        int64_t condval_33;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) == (int64_t)1)) {
          condval_33 = (int64_t)3;
        } else {
          condval_33 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_10) >> (int64_t)3)) - (int64_t)1);
        }
        condval_32 = condval_33;
      }
      half_t condval_34;
      if ((((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_5)) {
        condval_34 = G1[((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
      } else {
        condval_34 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      half_t condval_35;
      if (((((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_5)) {
        condval_35 = G1[(((((j_10 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
      } else {
        condval_35 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      out[j_10] = nr_tl_shallow::mul(ff[((condval_32 * (int64_t)8) + (((int64_t)j_10) & (int64_t)7))], nr_tl_shallow::pack(condval_34, condval_35));
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
      #pragma unroll
      for (int p_9 = 0; p_9 < 2; ++p_9) {
        uint lo = (nr_tl_shallow::e4pair(out[((m_12 * 8) + (p_9 * 4))]) | (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 2)]) << (uint)16));
        uint hi = (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 1)]) | (nr_tl_shallow::e4pair(out[(((m_12 * 8) + (p_9 * 4)) + 3)]) << (uint)16));
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint l = nr_tl_shallow::shfl(lo, (((((int)threadIdx.x) & 7) * 4) + k));
          uint h_2 = nr_tl_shallow::shfl(hi, (((((int)threadIdx.x) & 7) * 4) + k));
          uint condval_36;
          if ((((int)threadIdx.x) < 8)) {
            condval_36 = l;
          } else {
            condval_36 = h_2;
          }
          q[k] = condval_36;
        }
        int rmod_12 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_6 = (((int)blockIdx.x) / (W >> 3));
        int condval_38;
        if ((((slab * 2) + m_12) == 0)) {
          condval_38 = 0;
        } else {
          int condval_39;
          if ((((slab * 2) + m_12) == 1)) {
            condval_39 = 3;
          } else {
            condval_39 = (((slab * 2) + m_12) - 1);
          }
          condval_38 = condval_39;
        }
        int condval_40;
        if ((((slab * 2) + m_12) == 0)) {
          condval_40 = 0;
        } else {
          int condval_41;
          if ((((slab * 2) + m_12) == 1)) {
            condval_41 = 3;
          } else {
            condval_41 = (((slab * 2) + m_12) - 1);
          }
          condval_40 = condval_41;
        }
        int rmod_13 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_7 = (((int)blockIdx.x) / (W >> 3));
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
        int rmod_14 = (((int)blockIdx.x) % (W >> 3));
        int rmod_15 = (((int)blockIdx.x) % (W >> 3));
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
        int condval_37;
        if (((((1 <= ((((((0 <= (W >> 3)) && (0 <= rmod_12)) || (((W >> 3) < 0) && (rmod_12 <= 0))) ? rdiv_6 : (rdiv_6 - 1)) * 2) + ((int)(((((condval_38 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_40 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_13)) || (((W >> 3) < 0) && (rmod_13 <= 0))) ? rdiv_7 : (rdiv_7 - 1)) * 2) + ((int)(((((condval_42 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_44 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) <= (H >> 2))) & (0 <= ((((0 <= (W >> 3)) && (0 <= rmod_14)) || (((W >> 3) < 0) && (rmod_14 <= 0))) ? rmod_14 : (rmod_14 + (W >> 3))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_15)) || (((W >> 3) < 0) && (rmod_15 <= 0))) ? rmod_15 : (rmod_15 + (W >> 3))) * 2) + ((int)(((((condval_46 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_48 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) < (W >> 2)))) {
          int rmod_16 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_8 = (((int)blockIdx.x) / (W >> 3));
          int condval_50;
          if ((((slab * 2) + m_12) == 0)) {
            condval_50 = 0;
          } else {
            int condval_51;
            if ((((slab * 2) + m_12) == 1)) {
              condval_51 = 3;
            } else {
              condval_51 = (((slab * 2) + m_12) - 1);
            }
            condval_50 = condval_51;
          }
          int condval_52;
          if ((((slab * 2) + m_12) == 0)) {
            condval_52 = 0;
          } else {
            int condval_53;
            if ((((slab * 2) + m_12) == 1)) {
              condval_53 = 3;
            } else {
              condval_53 = (((slab * 2) + m_12) - 1);
            }
            condval_52 = condval_53;
          }
          int rmod_17 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_9 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_54;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_17)) || (((W >> 3) < 0) && (rmod_17 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * 2) + ((int)(((((condval_55 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_57 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_18 = (((int)blockIdx.x) % (W >> 3));
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
            condval_54 = ((((((((0 <= (W >> 3)) && (0 <= rmod_18)) || (((W >> 3) < 0) && (rmod_18 <= 0))) ? rmod_18 : (rmod_18 + (W >> 3))) * 2) + ((int)(((((condval_59 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_61 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_19 = (((int)blockIdx.x) % (W >> 3));
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
            condval_54 = ((((((((0 <= (W >> 3)) && (0 <= rmod_19)) || (((W >> 3) < 0) && (rmod_19 <= 0))) ? rmod_19 : (rmod_19 + (W >> 3))) * 2) + ((int)(((((condval_63 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_65 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_20 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_10 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_20)) || (((W >> 3) < 0) && (rmod_20 <= 0))) ? rdiv_10 : (rdiv_10 - 1)) * 2) + ((int)(((((condval_70 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_72 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_21 = (((int)blockIdx.x) % (W >> 3));
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
            condval_69 = ((((((((0 <= (W >> 3)) && (0 <= rmod_21)) || (((W >> 3) < 0) && (rmod_21 <= 0))) ? rmod_21 : (rmod_21 + (W >> 3))) * 2) + ((int)(((((condval_74 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_76 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_22 = (((int)blockIdx.x) % (W >> 3));
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
            condval_69 = ((((((((0 <= (W >> 3)) && (0 <= rmod_22)) || (((W >> 3) < 0) && (rmod_22 <= 0))) ? rmod_22 : (rmod_22 + (W >> 3))) * 2) + ((int)(((((condval_78 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_80 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_23 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_11 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_23)) || (((W >> 3) < 0) && (rmod_23 <= 0))) ? rdiv_11 : (rdiv_11 - 1)) * 2) + ((int)(((((condval_89 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_91 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_24 = (((int)blockIdx.x) % (W >> 3));
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
            condval_88 = ((((((((0 <= (W >> 3)) && (0 <= rmod_24)) || (((W >> 3) < 0) && (rmod_24 <= 0))) ? rmod_24 : (rmod_24 + (W >> 3))) * 2) + ((int)(((((condval_93 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_95 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_25 = (((int)blockIdx.x) % (W >> 3));
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
            condval_88 = ((((((((0 <= (W >> 3)) && (0 <= rmod_25)) || (((W >> 3) < 0) && (rmod_25 <= 0))) ? rmod_25 : (rmod_25 + (W >> 3))) * 2) + ((int)(((((condval_97 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_99 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_26 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_12 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_26)) || (((W >> 3) < 0) && (rmod_26 <= 0))) ? rdiv_12 : (rdiv_12 - 1)) * 2) + ((int)(((((condval_108 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_110 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_27 = (((int)blockIdx.x) % (W >> 3));
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
            condval_107 = ((((((((0 <= (W >> 3)) && (0 <= rmod_27)) || (((W >> 3) < 0) && (rmod_27 <= 0))) ? rmod_27 : (rmod_27 + (W >> 3))) * 2) + ((int)(((((condval_112 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_114 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_28 = (((int)blockIdx.x) % (W >> 3));
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
            condval_107 = ((((((((0 <= (W >> 3)) && (0 <= rmod_28)) || (((W >> 3) < 0) && (rmod_28 <= 0))) ? rmod_28 : (rmod_28 + (W >> 3))) * 2) + ((int)(((((condval_116 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_118 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_29 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_13 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_29)) || (((W >> 3) < 0) && (rmod_29 <= 0))) ? rdiv_13 : (rdiv_13 - 1)) * 2) + ((int)(((((condval_127 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_129 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_30 = (((int)blockIdx.x) % (W >> 3));
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
            condval_126 = ((((((((0 <= (W >> 3)) && (0 <= rmod_30)) || (((W >> 3) < 0) && (rmod_30 <= 0))) ? rmod_30 : (rmod_30 + (W >> 3))) * 2) + ((int)(((((condval_131 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_133 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_31 = (((int)blockIdx.x) % (W >> 3));
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
            condval_126 = ((((((((0 <= (W >> 3)) && (0 <= rmod_31)) || (((W >> 3) < 0) && (rmod_31 <= 0))) ? rmod_31 : (rmod_31 + (W >> 3))) * 2) + ((int)(((((condval_135 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_137 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_32 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_14 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_32)) || (((W >> 3) < 0) && (rmod_32 <= 0))) ? rdiv_14 : (rdiv_14 - 1)) * 2) + ((int)(((((condval_146 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_148 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_33 = (((int)blockIdx.x) % (W >> 3));
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
            condval_145 = ((((((((0 <= (W >> 3)) && (0 <= rmod_33)) || (((W >> 3) < 0) && (rmod_33 <= 0))) ? rmod_33 : (rmod_33 + (W >> 3))) * 2) + ((int)(((((condval_150 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_152 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_34 = (((int)blockIdx.x) % (W >> 3));
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
            condval_145 = ((((((((0 <= (W >> 3)) && (0 <= rmod_34)) || (((W >> 3) < 0) && (rmod_34 <= 0))) ? rmod_34 : (rmod_34 + (W >> 3))) * 2) + ((int)(((((condval_154 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_156 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_35 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_15 = (((int)blockIdx.x) / (W >> 3));
          int condval_165;
          if ((((slab * 2) + m_12) == 0)) {
            condval_165 = 0;
          } else {
            int condval_166;
            if ((((slab * 2) + m_12) == 1)) {
              condval_166 = 3;
            } else {
              condval_166 = (((slab * 2) + m_12) - 1);
            }
            condval_165 = condval_166;
          }
          int condval_167;
          if ((((slab * 2) + m_12) == 0)) {
            condval_167 = 0;
          } else {
            int condval_168;
            if ((((slab * 2) + m_12) == 1)) {
              condval_168 = 3;
            } else {
              condval_168 = (((slab * 2) + m_12) - 1);
            }
            condval_167 = condval_168;
          }
          int condval_164;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_35)) || (((W >> 3) < 0) && (rmod_35 <= 0))) ? rdiv_15 : (rdiv_15 - 1)) * 2) + ((int)(((((condval_165 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_167 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_36 = (((int)blockIdx.x) % (W >> 3));
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
            condval_164 = ((((((((0 <= (W >> 3)) && (0 <= rmod_36)) || (((W >> 3) < 0) && (rmod_36 <= 0))) ? rmod_36 : (rmod_36 + (W >> 3))) * 2) + ((int)(((((condval_169 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_171 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_37 = (((int)blockIdx.x) % (W >> 3));
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
            condval_164 = ((((((((0 <= (W >> 3)) && (0 <= rmod_37)) || (((W >> 3) < 0) && (rmod_37 <= 0))) ? rmod_37 : (rmod_37 + (W >> 3))) * 2) + ((int)(((((condval_173 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_175 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
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
          int rmod_38 = (((int)blockIdx.x) % (W >> 3));
          int condval_183;
          if ((((slab * 2) + m_12) == 0)) {
            condval_183 = 0;
          } else {
            int condval_184;
            if ((((slab * 2) + m_12) == 1)) {
              condval_184 = 3;
            } else {
              condval_184 = (((slab * 2) + m_12) - 1);
            }
            condval_183 = condval_184;
          }
          int condval_185;
          if ((((slab * 2) + m_12) == 0)) {
            condval_185 = 0;
          } else {
            int condval_186;
            if ((((slab * 2) + m_12) == 1)) {
              condval_186 = 3;
            } else {
              condval_186 = (((slab * 2) + m_12) - 1);
            }
            condval_185 = condval_186;
          }
          int condval_187;
          if ((((slab * 2) + m_12) == 0)) {
            condval_187 = 0;
          } else {
            int condval_188;
            if ((((slab * 2) + m_12) == 1)) {
              condval_188 = 3;
            } else {
              condval_188 = (((slab * 2) + m_12) - 1);
            }
            condval_187 = condval_188;
          }
          int condval_189;
          if ((((slab * 2) + m_12) == 0)) {
            condval_189 = 0;
          } else {
            int condval_190;
            if ((((slab * 2) + m_12) == 1)) {
              condval_190 = 3;
            } else {
              condval_190 = (((slab * 2) + m_12) - 1);
            }
            condval_189 = condval_190;
          }
          int rmod_39 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_16 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_191;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_39)) || (((W >> 3) < 0) && (rmod_39 <= 0))) ? rdiv_16 : (rdiv_16 - 1)) * 2) + ((int)(((((condval_192 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_194 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_40 = (((int)blockIdx.x) % (W >> 3));
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
            condval_191 = ((((((((0 <= (W >> 3)) && (0 <= rmod_40)) || (((W >> 3) < 0) && (rmod_40 <= 0))) ? rmod_40 : (rmod_40 + (W >> 3))) * 2) + ((int)(((((condval_196 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_198 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_41 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_202;
            if ((((slab * 2) + m_12) == 0)) {
              condval_202 = 0;
            } else {
              int condval_203;
              if ((((slab * 2) + m_12) == 1)) {
                condval_203 = 3;
              } else {
                condval_203 = (((slab * 2) + m_12) - 1);
              }
              condval_202 = condval_203;
            }
            condval_191 = ((((((((0 <= (W >> 3)) && (0 <= rmod_41)) || (((W >> 3) < 0) && (rmod_41 <= 0))) ? rmod_41 : (rmod_41 + (W >> 3))) * 2) + ((int)(((((condval_200 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_202 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_204;
          if ((((slab * 2) + m_12) == 0)) {
            condval_204 = 0;
          } else {
            int condval_205;
            if ((((slab * 2) + m_12) == 1)) {
              condval_205 = 3;
            } else {
              condval_205 = (((slab * 2) + m_12) - 1);
            }
            condval_204 = condval_205;
          }
          int condval_206;
          if ((((slab * 2) + m_12) == 0)) {
            condval_206 = 0;
          } else {
            int condval_207;
            if ((((slab * 2) + m_12) == 1)) {
              condval_207 = 3;
            } else {
              condval_207 = (((slab * 2) + m_12) - 1);
            }
            condval_206 = condval_207;
          }
          int condval_208;
          if ((((slab * 2) + m_12) == 0)) {
            condval_208 = 0;
          } else {
            int condval_209;
            if ((((slab * 2) + m_12) == 1)) {
              condval_209 = 3;
            } else {
              condval_209 = (((slab * 2) + m_12) - 1);
            }
            condval_208 = condval_209;
          }
          int rmod_42 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_17 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_210;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_42)) || (((W >> 3) < 0) && (rmod_42 <= 0))) ? rdiv_17 : (rdiv_17 - 1)) * 2) + ((int)(((((condval_211 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_213 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_43 = (((int)blockIdx.x) % (W >> 3));
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
            condval_210 = ((((((((0 <= (W >> 3)) && (0 <= rmod_43)) || (((W >> 3) < 0) && (rmod_43 <= 0))) ? rmod_43 : (rmod_43 + (W >> 3))) * 2) + ((int)(((((condval_215 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_217 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_44 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_221;
            if ((((slab * 2) + m_12) == 0)) {
              condval_221 = 0;
            } else {
              int condval_222;
              if ((((slab * 2) + m_12) == 1)) {
                condval_222 = 3;
              } else {
                condval_222 = (((slab * 2) + m_12) - 1);
              }
              condval_221 = condval_222;
            }
            condval_210 = ((((((((0 <= (W >> 3)) && (0 <= rmod_44)) || (((W >> 3) < 0) && (rmod_44 <= 0))) ? rmod_44 : (rmod_44 + (W >> 3))) * 2) + ((int)(((((condval_219 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_221 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_223;
          if ((((slab * 2) + m_12) == 0)) {
            condval_223 = 0;
          } else {
            int condval_224;
            if ((((slab * 2) + m_12) == 1)) {
              condval_224 = 3;
            } else {
              condval_224 = (((slab * 2) + m_12) - 1);
            }
            condval_223 = condval_224;
          }
          int condval_225;
          if ((((slab * 2) + m_12) == 0)) {
            condval_225 = 0;
          } else {
            int condval_226;
            if ((((slab * 2) + m_12) == 1)) {
              condval_226 = 3;
            } else {
              condval_226 = (((slab * 2) + m_12) - 1);
            }
            condval_225 = condval_226;
          }
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
          int rmod_45 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_18 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_229;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_45)) || (((W >> 3) < 0) && (rmod_45 <= 0))) ? rdiv_18 : (rdiv_18 - 1)) * 2) + ((int)(((((condval_230 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_232 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_46 = (((int)blockIdx.x) % (W >> 3));
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
            condval_229 = ((((((((0 <= (W >> 3)) && (0 <= rmod_46)) || (((W >> 3) < 0) && (rmod_46 <= 0))) ? rmod_46 : (rmod_46 + (W >> 3))) * 2) + ((int)(((((condval_234 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_236 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_47 = (((int)blockIdx.x) % (W >> 3));
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
            condval_229 = ((((((((0 <= (W >> 3)) && (0 <= rmod_47)) || (((W >> 3) < 0) && (rmod_47 <= 0))) ? rmod_47 : (rmod_47 + (W >> 3))) * 2) + ((int)(((((condval_238 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_240 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_48 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_19 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_248;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_48)) || (((W >> 3) < 0) && (rmod_48 <= 0))) ? rdiv_19 : (rdiv_19 - 1)) * 2) + ((int)(((((condval_249 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_251 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_49 = (((int)blockIdx.x) % (W >> 3));
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
            condval_248 = ((((((((0 <= (W >> 3)) && (0 <= rmod_49)) || (((W >> 3) < 0) && (rmod_49 <= 0))) ? rmod_49 : (rmod_49 + (W >> 3))) * 2) + ((int)(((((condval_253 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_255 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_50 = (((int)blockIdx.x) % (W >> 3));
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
            condval_248 = ((((((((0 <= (W >> 3)) && (0 <= rmod_50)) || (((W >> 3) < 0) && (rmod_50 <= 0))) ? rmod_50 : (rmod_50 + (W >> 3))) * 2) + ((int)(((((condval_257 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_259 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_265;
          if ((((slab * 2) + m_12) == 0)) {
            condval_265 = 0;
          } else {
            int condval_266;
            if ((((slab * 2) + m_12) == 1)) {
              condval_266 = 3;
            } else {
              condval_266 = (((slab * 2) + m_12) - 1);
            }
            condval_265 = condval_266;
          }
          int rmod_51 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_20 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_267;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_51)) || (((W >> 3) < 0) && (rmod_51 <= 0))) ? rdiv_20 : (rdiv_20 - 1)) * 2) + ((int)(((((condval_268 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_270 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_52 = (((int)blockIdx.x) % (W >> 3));
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
            condval_267 = ((((((((0 <= (W >> 3)) && (0 <= rmod_52)) || (((W >> 3) < 0) && (rmod_52 <= 0))) ? rmod_52 : (rmod_52 + (W >> 3))) * 2) + ((int)(((((condval_272 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_274 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_53 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_278;
            if ((((slab * 2) + m_12) == 0)) {
              condval_278 = 0;
            } else {
              int condval_279;
              if ((((slab * 2) + m_12) == 1)) {
                condval_279 = 3;
              } else {
                condval_279 = (((slab * 2) + m_12) - 1);
              }
              condval_278 = condval_279;
            }
            condval_267 = ((((((((0 <= (W >> 3)) && (0 <= rmod_53)) || (((W >> 3) < 0) && (rmod_53 <= 0))) ? rmod_53 : (rmod_53 + (W >> 3))) * 2) + ((int)(((((condval_276 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_278 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_280;
          if ((((slab * 2) + m_12) == 0)) {
            condval_280 = 0;
          } else {
            int condval_281;
            if ((((slab * 2) + m_12) == 1)) {
              condval_281 = 3;
            } else {
              condval_281 = (((slab * 2) + m_12) - 1);
            }
            condval_280 = condval_281;
          }
          int condval_282;
          if ((((slab * 2) + m_12) == 0)) {
            condval_282 = 0;
          } else {
            int condval_283;
            if ((((slab * 2) + m_12) == 1)) {
              condval_283 = 3;
            } else {
              condval_283 = (((slab * 2) + m_12) - 1);
            }
            condval_282 = condval_283;
          }
          int condval_284;
          if ((((slab * 2) + m_12) == 0)) {
            condval_284 = 0;
          } else {
            int condval_285;
            if ((((slab * 2) + m_12) == 1)) {
              condval_285 = 3;
            } else {
              condval_285 = (((slab * 2) + m_12) - 1);
            }
            condval_284 = condval_285;
          }
          int rmod_54 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_21 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_286;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_54)) || (((W >> 3) < 0) && (rmod_54 <= 0))) ? rdiv_21 : (rdiv_21 - 1)) * 2) + ((int)(((((condval_287 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_289 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_55 = (((int)blockIdx.x) % (W >> 3));
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
            condval_286 = ((((((((0 <= (W >> 3)) && (0 <= rmod_55)) || (((W >> 3) < 0) && (rmod_55 <= 0))) ? rmod_55 : (rmod_55 + (W >> 3))) * 2) + ((int)(((((condval_291 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_293 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_56 = (((int)blockIdx.x) % (W >> 3));
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
            condval_286 = ((((((((0 <= (W >> 3)) && (0 <= rmod_56)) || (((W >> 3) < 0) && (rmod_56 <= 0))) ? rmod_56 : (rmod_56 + (W >> 3))) * 2) + ((int)(((((condval_295 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_297 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_57 = (((int)blockIdx.x) % (W >> 3));
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
          int rmod_58 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_22 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_58)) || (((W >> 3) < 0) && (rmod_58 <= 0))) ? rdiv_22 : (rdiv_22 - 1)) * 2) + ((int)(((((condval_314 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_316 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_59 = (((int)blockIdx.x) % (W >> 3));
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
            condval_313 = ((((((((0 <= (W >> 3)) && (0 <= rmod_59)) || (((W >> 3) < 0) && (rmod_59 <= 0))) ? rmod_59 : (rmod_59 + (W >> 3))) * 2) + ((int)(((((condval_318 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_320 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_60 = (((int)blockIdx.x) % (W >> 3));
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
            condval_313 = ((((((((0 <= (W >> 3)) && (0 <= rmod_60)) || (((W >> 3) < 0) && (rmod_60 <= 0))) ? rmod_60 : (rmod_60 + (W >> 3))) * 2) + ((int)(((((condval_322 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_324 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_61 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_23 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_61)) || (((W >> 3) < 0) && (rmod_61 <= 0))) ? rdiv_23 : (rdiv_23 - 1)) * 2) + ((int)(((((condval_333 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_335 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_62 = (((int)blockIdx.x) % (W >> 3));
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
            condval_332 = ((((((((0 <= (W >> 3)) && (0 <= rmod_62)) || (((W >> 3) < 0) && (rmod_62 <= 0))) ? rmod_62 : (rmod_62 + (W >> 3))) * 2) + ((int)(((((condval_337 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_339 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_63 = (((int)blockIdx.x) % (W >> 3));
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
            condval_332 = ((((((((0 <= (W >> 3)) && (0 <= rmod_63)) || (((W >> 3) < 0) && (rmod_63 <= 0))) ? rmod_63 : (rmod_63 + (W >> 3))) * 2) + ((int)(((((condval_341 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_343 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_64 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_24 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_64)) || (((W >> 3) < 0) && (rmod_64 <= 0))) ? rdiv_24 : (rdiv_24 - 1)) * 2) + ((int)(((((condval_352 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_354 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_65 = (((int)blockIdx.x) % (W >> 3));
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
            condval_351 = ((((((((0 <= (W >> 3)) && (0 <= rmod_65)) || (((W >> 3) < 0) && (rmod_65 <= 0))) ? rmod_65 : (rmod_65 + (W >> 3))) * 2) + ((int)(((((condval_356 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_358 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_66 = (((int)blockIdx.x) % (W >> 3));
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
            condval_351 = ((((((((0 <= (W >> 3)) && (0 <= rmod_66)) || (((W >> 3) < 0) && (rmod_66 <= 0))) ? rmod_66 : (rmod_66 + (W >> 3))) * 2) + ((int)(((((condval_360 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_362 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_67 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_25 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_67)) || (((W >> 3) < 0) && (rmod_67 <= 0))) ? rdiv_25 : (rdiv_25 - 1)) * 2) + ((int)(((((condval_371 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_373 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_68 = (((int)blockIdx.x) % (W >> 3));
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
            condval_370 = ((((((((0 <= (W >> 3)) && (0 <= rmod_68)) || (((W >> 3) < 0) && (rmod_68 <= 0))) ? rmod_68 : (rmod_68 + (W >> 3))) * 2) + ((int)(((((condval_375 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_377 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_69 = (((int)blockIdx.x) % (W >> 3));
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
            condval_370 = ((((((((0 <= (W >> 3)) && (0 <= rmod_69)) || (((W >> 3) < 0) && (rmod_69 <= 0))) ? rmod_69 : (rmod_69 + (W >> 3))) * 2) + ((int)(((((condval_379 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_381 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_70 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_26 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_70)) || (((W >> 3) < 0) && (rmod_70 <= 0))) ? rdiv_26 : (rdiv_26 - 1)) * 2) + ((int)(((((condval_390 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_392 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_71 = (((int)blockIdx.x) % (W >> 3));
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
            condval_389 = ((((((((0 <= (W >> 3)) && (0 <= rmod_71)) || (((W >> 3) < 0) && (rmod_71 <= 0))) ? rmod_71 : (rmod_71 + (W >> 3))) * 2) + ((int)(((((condval_394 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_396 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_72 = (((int)blockIdx.x) % (W >> 3));
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
            condval_389 = ((((((((0 <= (W >> 3)) && (0 <= rmod_72)) || (((W >> 3) < 0) && (rmod_72 <= 0))) ? rmod_72 : (rmod_72 + (W >> 3))) * 2) + ((int)(((((condval_398 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_400 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_73 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_27 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_73)) || (((W >> 3) < 0) && (rmod_73 <= 0))) ? rdiv_27 : (rdiv_27 - 1)) * 2) + ((int)(((((condval_409 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_411 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_74 = (((int)blockIdx.x) % (W >> 3));
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
            condval_408 = ((((((((0 <= (W >> 3)) && (0 <= rmod_74)) || (((W >> 3) < 0) && (rmod_74 <= 0))) ? rmod_74 : (rmod_74 + (W >> 3))) * 2) + ((int)(((((condval_413 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_415 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_75 = (((int)blockIdx.x) % (W >> 3));
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
            condval_408 = ((((((((0 <= (W >> 3)) && (0 <= rmod_75)) || (((W >> 3) < 0) && (rmod_75 <= 0))) ? rmod_75 : (rmod_75 + (W >> 3))) * 2) + ((int)(((((condval_417 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_419 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          condval_37 = (((((((((((((((((((0 <= (W >> 3)) && (0 <= rmod_16)) || (((W >> 3) < 0) && (rmod_16 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * 2) + ((int)(((((condval_50 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_52 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_54 * 2)) + ((((condval_67 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) + ((H >> 3) * (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_69 & 3) * 16) + (((((condval_82 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_84 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_86 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_88 & 3) * 16) + (((((condval_101 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_103 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_105 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_107 & 3) * 16) + (((((condval_120 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_122 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_124 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_126 & 3) * 16) + (((((condval_139 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_141 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_143 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_145 & 3) * 16) + (((((condval_158 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_160 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_162 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_164 & 3) * 16) + (((((condval_177 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_179 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_181 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10))) * (W >> 2)) * 512) + ((((((((((((0 <= (W >> 3)) && (0 <= rmod_38)) || (((W >> 3) < 0) && (rmod_38 <= 0))) ? rmod_38 : (rmod_38 + (W >> 3))) * 2) + ((int)(((((condval_183 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_185 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_187 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_189 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + (((W >> 5) * ((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_191 & 3) * 16) + (((((condval_204 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_206 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_208 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_210 & 3) * 16) + (((((condval_223 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_225 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_227 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_229 & 3) * 16) + (((((condval_242 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_244 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_246 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_248 & 3) * 16) + (((((condval_261 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_263 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_265 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_267 & 3) * 16) + (((((condval_280 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_282 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_284 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_286 & 3) * 16) + (((((condval_299 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_301 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_303 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + (((((((((((((0 <= (W >> 3)) && (0 <= rmod_57)) || (((W >> 3) < 0) && (rmod_57 <= 0))) ? rmod_57 : (rmod_57 + (W >> 3))) * 2) + ((int)(((((condval_305 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_307 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_309 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_311 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_313 & 3) * 16) + (((((condval_326 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_328 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_330 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_332 & 3) * 16) + (((((condval_345 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_347 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_349 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_351 & 3) * 16) + (((((condval_364 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_366 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_368 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_370 & 3) * 16) + (((((condval_383 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_385 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_387 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_389 & 3) * 16) + (((((condval_402 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_404 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_406 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_408 & 3) * 16) + (((((condval_421 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_423 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_425 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_37 = -1;
        }
        int rmod_76 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_28 = (((int)blockIdx.x) / (W >> 3));
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
        int rmod_77 = (((int)blockIdx.x) % (W >> 3));
        int rdiv_29 = (((int)blockIdx.x) / (W >> 3));
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
        int rmod_78 = (((int)blockIdx.x) % (W >> 3));
        int rmod_79 = (((int)blockIdx.x) % (W >> 3));
        int condval_436;
        if ((((slab * 2) + m_12) == 0)) {
          condval_436 = 0;
        } else {
          int condval_437;
          if ((((slab * 2) + m_12) == 1)) {
            condval_437 = 3;
          } else {
            condval_437 = (((slab * 2) + m_12) - 1);
          }
          condval_436 = condval_437;
        }
        int condval_438;
        if ((((slab * 2) + m_12) == 0)) {
          condval_438 = 0;
        } else {
          int condval_439;
          if ((((slab * 2) + m_12) == 1)) {
            condval_439 = 3;
          } else {
            condval_439 = (((slab * 2) + m_12) - 1);
          }
          condval_438 = condval_439;
        }
        int condval_427;
        if (((((1 <= ((((((0 <= (W >> 3)) && (0 <= rmod_76)) || (((W >> 3) < 0) && (rmod_76 <= 0))) ? rdiv_28 : (rdiv_28 - 1)) * 2) + ((int)(((((condval_428 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_430 * 16) + ((int)threadIdx.x)) >> 4) == 2))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_77)) || (((W >> 3) < 0) && (rmod_77 <= 0))) ? rdiv_29 : (rdiv_29 - 1)) * 2) + ((int)(((((condval_432 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_434 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) <= (H >> 2))) & (0 <= ((((0 <= (W >> 3)) && (0 <= rmod_78)) || (((W >> 3) < 0) && (rmod_78 <= 0))) ? rmod_78 : (rmod_78 + (W >> 3))))) & (((((((0 <= (W >> 3)) && (0 <= rmod_79)) || (((W >> 3) < 0) && (rmod_79 <= 0))) ? rmod_79 : (rmod_79 + (W >> 3))) * 2) + ((int)(((((condval_436 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_438 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) < (W >> 2)))) {
          int rmod_80 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_30 = (((int)blockIdx.x) / (W >> 3));
          int condval_440;
          if ((((slab * 2) + m_12) == 0)) {
            condval_440 = 0;
          } else {
            int condval_441;
            if ((((slab * 2) + m_12) == 1)) {
              condval_441 = 3;
            } else {
              condval_441 = (((slab * 2) + m_12) - 1);
            }
            condval_440 = condval_441;
          }
          int condval_442;
          if ((((slab * 2) + m_12) == 0)) {
            condval_442 = 0;
          } else {
            int condval_443;
            if ((((slab * 2) + m_12) == 1)) {
              condval_443 = 3;
            } else {
              condval_443 = (((slab * 2) + m_12) - 1);
            }
            condval_442 = condval_443;
          }
          int rmod_81 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_31 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_444;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_81)) || (((W >> 3) < 0) && (rmod_81 <= 0))) ? rdiv_31 : (rdiv_31 - 1)) * 2) + ((int)(((((condval_445 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_447 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_82 = (((int)blockIdx.x) % (W >> 3));
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
            condval_444 = ((((((((0 <= (W >> 3)) && (0 <= rmod_82)) || (((W >> 3) < 0) && (rmod_82 <= 0))) ? rmod_82 : (rmod_82 + (W >> 3))) * 2) + ((int)(((((condval_449 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_451 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_83 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_455;
            if ((((slab * 2) + m_12) == 0)) {
              condval_455 = 0;
            } else {
              int condval_456;
              if ((((slab * 2) + m_12) == 1)) {
                condval_456 = 3;
              } else {
                condval_456 = (((slab * 2) + m_12) - 1);
              }
              condval_455 = condval_456;
            }
            condval_444 = ((((((((0 <= (W >> 3)) && (0 <= rmod_83)) || (((W >> 3) < 0) && (rmod_83 <= 0))) ? rmod_83 : (rmod_83 + (W >> 3))) * 2) + ((int)(((((condval_453 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_455 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_457;
          if ((((slab * 2) + m_12) == 0)) {
            condval_457 = 0;
          } else {
            int condval_458;
            if ((((slab * 2) + m_12) == 1)) {
              condval_458 = 3;
            } else {
              condval_458 = (((slab * 2) + m_12) - 1);
            }
            condval_457 = condval_458;
          }
          int rmod_84 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_32 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_459;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_84)) || (((W >> 3) < 0) && (rmod_84 <= 0))) ? rdiv_32 : (rdiv_32 - 1)) * 2) + ((int)(((((condval_460 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_462 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_85 = (((int)blockIdx.x) % (W >> 3));
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
            condval_459 = ((((((((0 <= (W >> 3)) && (0 <= rmod_85)) || (((W >> 3) < 0) && (rmod_85 <= 0))) ? rmod_85 : (rmod_85 + (W >> 3))) * 2) + ((int)(((((condval_464 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_466 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_86 = (((int)blockIdx.x) % (W >> 3));
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
            condval_459 = ((((((((0 <= (W >> 3)) && (0 <= rmod_86)) || (((W >> 3) < 0) && (rmod_86 <= 0))) ? rmod_86 : (rmod_86 + (W >> 3))) * 2) + ((int)(((((condval_468 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_470 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_474;
          if ((((slab * 2) + m_12) == 0)) {
            condval_474 = 0;
          } else {
            int condval_475;
            if ((((slab * 2) + m_12) == 1)) {
              condval_475 = 3;
            } else {
              condval_475 = (((slab * 2) + m_12) - 1);
            }
            condval_474 = condval_475;
          }
          int condval_476;
          if ((((slab * 2) + m_12) == 0)) {
            condval_476 = 0;
          } else {
            int condval_477;
            if ((((slab * 2) + m_12) == 1)) {
              condval_477 = 3;
            } else {
              condval_477 = (((slab * 2) + m_12) - 1);
            }
            condval_476 = condval_477;
          }
          int rmod_87 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_33 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_478;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_87)) || (((W >> 3) < 0) && (rmod_87 <= 0))) ? rdiv_33 : (rdiv_33 - 1)) * 2) + ((int)(((((condval_479 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_481 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_88 = (((int)blockIdx.x) % (W >> 3));
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
            condval_478 = ((((((((0 <= (W >> 3)) && (0 <= rmod_88)) || (((W >> 3) < 0) && (rmod_88 <= 0))) ? rmod_88 : (rmod_88 + (W >> 3))) * 2) + ((int)(((((condval_483 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_485 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_89 = (((int)blockIdx.x) % (W >> 3));
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
            condval_478 = ((((((((0 <= (W >> 3)) && (0 <= rmod_89)) || (((W >> 3) < 0) && (rmod_89 <= 0))) ? rmod_89 : (rmod_89 + (W >> 3))) * 2) + ((int)(((((condval_487 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_489 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_493;
          if ((((slab * 2) + m_12) == 0)) {
            condval_493 = 0;
          } else {
            int condval_494;
            if ((((slab * 2) + m_12) == 1)) {
              condval_494 = 3;
            } else {
              condval_494 = (((slab * 2) + m_12) - 1);
            }
            condval_493 = condval_494;
          }
          int condval_495;
          if ((((slab * 2) + m_12) == 0)) {
            condval_495 = 0;
          } else {
            int condval_496;
            if ((((slab * 2) + m_12) == 1)) {
              condval_496 = 3;
            } else {
              condval_496 = (((slab * 2) + m_12) - 1);
            }
            condval_495 = condval_496;
          }
          int rmod_90 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_34 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_497;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_90)) || (((W >> 3) < 0) && (rmod_90 <= 0))) ? rdiv_34 : (rdiv_34 - 1)) * 2) + ((int)(((((condval_498 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_500 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_91 = (((int)blockIdx.x) % (W >> 3));
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
            condval_497 = ((((((((0 <= (W >> 3)) && (0 <= rmod_91)) || (((W >> 3) < 0) && (rmod_91 <= 0))) ? rmod_91 : (rmod_91 + (W >> 3))) * 2) + ((int)(((((condval_502 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_504 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_92 = (((int)blockIdx.x) % (W >> 3));
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
            condval_497 = ((((((((0 <= (W >> 3)) && (0 <= rmod_92)) || (((W >> 3) < 0) && (rmod_92 <= 0))) ? rmod_92 : (rmod_92 + (W >> 3))) * 2) + ((int)(((((condval_506 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_508 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_512;
          if ((((slab * 2) + m_12) == 0)) {
            condval_512 = 0;
          } else {
            int condval_513;
            if ((((slab * 2) + m_12) == 1)) {
              condval_513 = 3;
            } else {
              condval_513 = (((slab * 2) + m_12) - 1);
            }
            condval_512 = condval_513;
          }
          int condval_514;
          if ((((slab * 2) + m_12) == 0)) {
            condval_514 = 0;
          } else {
            int condval_515;
            if ((((slab * 2) + m_12) == 1)) {
              condval_515 = 3;
            } else {
              condval_515 = (((slab * 2) + m_12) - 1);
            }
            condval_514 = condval_515;
          }
          int rmod_93 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_35 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_516;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_93)) || (((W >> 3) < 0) && (rmod_93 <= 0))) ? rdiv_35 : (rdiv_35 - 1)) * 2) + ((int)(((((condval_517 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_519 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_94 = (((int)blockIdx.x) % (W >> 3));
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
            condval_516 = ((((((((0 <= (W >> 3)) && (0 <= rmod_94)) || (((W >> 3) < 0) && (rmod_94 <= 0))) ? rmod_94 : (rmod_94 + (W >> 3))) * 2) + ((int)(((((condval_521 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_523 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_95 = (((int)blockIdx.x) % (W >> 3));
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
            condval_516 = ((((((((0 <= (W >> 3)) && (0 <= rmod_95)) || (((W >> 3) < 0) && (rmod_95 <= 0))) ? rmod_95 : (rmod_95 + (W >> 3))) * 2) + ((int)(((((condval_525 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_527 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_531;
          if ((((slab * 2) + m_12) == 0)) {
            condval_531 = 0;
          } else {
            int condval_532;
            if ((((slab * 2) + m_12) == 1)) {
              condval_532 = 3;
            } else {
              condval_532 = (((slab * 2) + m_12) - 1);
            }
            condval_531 = condval_532;
          }
          int condval_533;
          if ((((slab * 2) + m_12) == 0)) {
            condval_533 = 0;
          } else {
            int condval_534;
            if ((((slab * 2) + m_12) == 1)) {
              condval_534 = 3;
            } else {
              condval_534 = (((slab * 2) + m_12) - 1);
            }
            condval_533 = condval_534;
          }
          int rmod_96 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_36 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_535;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_96)) || (((W >> 3) < 0) && (rmod_96 <= 0))) ? rdiv_36 : (rdiv_36 - 1)) * 2) + ((int)(((((condval_536 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_538 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_97 = (((int)blockIdx.x) % (W >> 3));
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
            condval_535 = ((((((((0 <= (W >> 3)) && (0 <= rmod_97)) || (((W >> 3) < 0) && (rmod_97 <= 0))) ? rmod_97 : (rmod_97 + (W >> 3))) * 2) + ((int)(((((condval_540 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_542 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_98 = (((int)blockIdx.x) % (W >> 3));
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
            condval_535 = ((((((((0 <= (W >> 3)) && (0 <= rmod_98)) || (((W >> 3) < 0) && (rmod_98 <= 0))) ? rmod_98 : (rmod_98 + (W >> 3))) * 2) + ((int)(((((condval_544 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_546 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_99 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_37 = (((int)blockIdx.x) / (W >> 3));
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
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_99)) || (((W >> 3) < 0) && (rmod_99 <= 0))) ? rdiv_37 : (rdiv_37 - 1)) * 2) + ((int)(((((condval_555 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_557 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_100 = (((int)blockIdx.x) % (W >> 3));
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
            condval_554 = ((((((((0 <= (W >> 3)) && (0 <= rmod_100)) || (((W >> 3) < 0) && (rmod_100 <= 0))) ? rmod_100 : (rmod_100 + (W >> 3))) * 2) + ((int)(((((condval_559 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_561 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_101 = (((int)blockIdx.x) % (W >> 3));
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
            condval_554 = ((((((((0 <= (W >> 3)) && (0 <= rmod_101)) || (((W >> 3) < 0) && (rmod_101 <= 0))) ? rmod_101 : (rmod_101 + (W >> 3))) * 2) + ((int)(((((condval_563 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_565 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_569;
          if ((((slab * 2) + m_12) == 0)) {
            condval_569 = 0;
          } else {
            int condval_570;
            if ((((slab * 2) + m_12) == 1)) {
              condval_570 = 3;
            } else {
              condval_570 = (((slab * 2) + m_12) - 1);
            }
            condval_569 = condval_570;
          }
          int condval_571;
          if ((((slab * 2) + m_12) == 0)) {
            condval_571 = 0;
          } else {
            int condval_572;
            if ((((slab * 2) + m_12) == 1)) {
              condval_572 = 3;
            } else {
              condval_572 = (((slab * 2) + m_12) - 1);
            }
            condval_571 = condval_572;
          }
          int rmod_102 = (((int)blockIdx.x) % (W >> 3));
          int condval_573;
          if ((((slab * 2) + m_12) == 0)) {
            condval_573 = 0;
          } else {
            int condval_574;
            if ((((slab * 2) + m_12) == 1)) {
              condval_574 = 3;
            } else {
              condval_574 = (((slab * 2) + m_12) - 1);
            }
            condval_573 = condval_574;
          }
          int condval_575;
          if ((((slab * 2) + m_12) == 0)) {
            condval_575 = 0;
          } else {
            int condval_576;
            if ((((slab * 2) + m_12) == 1)) {
              condval_576 = 3;
            } else {
              condval_576 = (((slab * 2) + m_12) - 1);
            }
            condval_575 = condval_576;
          }
          int condval_577;
          if ((((slab * 2) + m_12) == 0)) {
            condval_577 = 0;
          } else {
            int condval_578;
            if ((((slab * 2) + m_12) == 1)) {
              condval_578 = 3;
            } else {
              condval_578 = (((slab * 2) + m_12) - 1);
            }
            condval_577 = condval_578;
          }
          int condval_579;
          if ((((slab * 2) + m_12) == 0)) {
            condval_579 = 0;
          } else {
            int condval_580;
            if ((((slab * 2) + m_12) == 1)) {
              condval_580 = 3;
            } else {
              condval_580 = (((slab * 2) + m_12) - 1);
            }
            condval_579 = condval_580;
          }
          int rmod_103 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_38 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_581;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_103)) || (((W >> 3) < 0) && (rmod_103 <= 0))) ? rdiv_38 : (rdiv_38 - 1)) * 2) + ((int)(((((condval_582 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_584 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_104 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_588;
            if ((((slab * 2) + m_12) == 0)) {
              condval_588 = 0;
            } else {
              int condval_589;
              if ((((slab * 2) + m_12) == 1)) {
                condval_589 = 3;
              } else {
                condval_589 = (((slab * 2) + m_12) - 1);
              }
              condval_588 = condval_589;
            }
            condval_581 = ((((((((0 <= (W >> 3)) && (0 <= rmod_104)) || (((W >> 3) < 0) && (rmod_104 <= 0))) ? rmod_104 : (rmod_104 + (W >> 3))) * 2) + ((int)(((((condval_586 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_588 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_105 = (((int)blockIdx.x) % (W >> 3));
            int condval_590;
            if ((((slab * 2) + m_12) == 0)) {
              condval_590 = 0;
            } else {
              int condval_591;
              if ((((slab * 2) + m_12) == 1)) {
                condval_591 = 3;
              } else {
                condval_591 = (((slab * 2) + m_12) - 1);
              }
              condval_590 = condval_591;
            }
            int condval_592;
            if ((((slab * 2) + m_12) == 0)) {
              condval_592 = 0;
            } else {
              int condval_593;
              if ((((slab * 2) + m_12) == 1)) {
                condval_593 = 3;
              } else {
                condval_593 = (((slab * 2) + m_12) - 1);
              }
              condval_592 = condval_593;
            }
            condval_581 = ((((((((0 <= (W >> 3)) && (0 <= rmod_105)) || (((W >> 3) < 0) && (rmod_105 <= 0))) ? rmod_105 : (rmod_105 + (W >> 3))) * 2) + ((int)(((((condval_590 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_592 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_594;
          if ((((slab * 2) + m_12) == 0)) {
            condval_594 = 0;
          } else {
            int condval_595;
            if ((((slab * 2) + m_12) == 1)) {
              condval_595 = 3;
            } else {
              condval_595 = (((slab * 2) + m_12) - 1);
            }
            condval_594 = condval_595;
          }
          int condval_596;
          if ((((slab * 2) + m_12) == 0)) {
            condval_596 = 0;
          } else {
            int condval_597;
            if ((((slab * 2) + m_12) == 1)) {
              condval_597 = 3;
            } else {
              condval_597 = (((slab * 2) + m_12) - 1);
            }
            condval_596 = condval_597;
          }
          int condval_598;
          if ((((slab * 2) + m_12) == 0)) {
            condval_598 = 0;
          } else {
            int condval_599;
            if ((((slab * 2) + m_12) == 1)) {
              condval_599 = 3;
            } else {
              condval_599 = (((slab * 2) + m_12) - 1);
            }
            condval_598 = condval_599;
          }
          int rmod_106 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_39 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_600;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_106)) || (((W >> 3) < 0) && (rmod_106 <= 0))) ? rdiv_39 : (rdiv_39 - 1)) * 2) + ((int)(((((condval_601 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_603 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_107 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_607;
            if ((((slab * 2) + m_12) == 0)) {
              condval_607 = 0;
            } else {
              int condval_608;
              if ((((slab * 2) + m_12) == 1)) {
                condval_608 = 3;
              } else {
                condval_608 = (((slab * 2) + m_12) - 1);
              }
              condval_607 = condval_608;
            }
            condval_600 = ((((((((0 <= (W >> 3)) && (0 <= rmod_107)) || (((W >> 3) < 0) && (rmod_107 <= 0))) ? rmod_107 : (rmod_107 + (W >> 3))) * 2) + ((int)(((((condval_605 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_607 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_108 = (((int)blockIdx.x) % (W >> 3));
            int condval_609;
            if ((((slab * 2) + m_12) == 0)) {
              condval_609 = 0;
            } else {
              int condval_610;
              if ((((slab * 2) + m_12) == 1)) {
                condval_610 = 3;
              } else {
                condval_610 = (((slab * 2) + m_12) - 1);
              }
              condval_609 = condval_610;
            }
            int condval_611;
            if ((((slab * 2) + m_12) == 0)) {
              condval_611 = 0;
            } else {
              int condval_612;
              if ((((slab * 2) + m_12) == 1)) {
                condval_612 = 3;
              } else {
                condval_612 = (((slab * 2) + m_12) - 1);
              }
              condval_611 = condval_612;
            }
            condval_600 = ((((((((0 <= (W >> 3)) && (0 <= rmod_108)) || (((W >> 3) < 0) && (rmod_108 <= 0))) ? rmod_108 : (rmod_108 + (W >> 3))) * 2) + ((int)(((((condval_609 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_611 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_613;
          if ((((slab * 2) + m_12) == 0)) {
            condval_613 = 0;
          } else {
            int condval_614;
            if ((((slab * 2) + m_12) == 1)) {
              condval_614 = 3;
            } else {
              condval_614 = (((slab * 2) + m_12) - 1);
            }
            condval_613 = condval_614;
          }
          int condval_615;
          if ((((slab * 2) + m_12) == 0)) {
            condval_615 = 0;
          } else {
            int condval_616;
            if ((((slab * 2) + m_12) == 1)) {
              condval_616 = 3;
            } else {
              condval_616 = (((slab * 2) + m_12) - 1);
            }
            condval_615 = condval_616;
          }
          int condval_617;
          if ((((slab * 2) + m_12) == 0)) {
            condval_617 = 0;
          } else {
            int condval_618;
            if ((((slab * 2) + m_12) == 1)) {
              condval_618 = 3;
            } else {
              condval_618 = (((slab * 2) + m_12) - 1);
            }
            condval_617 = condval_618;
          }
          int rmod_109 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_40 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_619;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_109)) || (((W >> 3) < 0) && (rmod_109 <= 0))) ? rdiv_40 : (rdiv_40 - 1)) * 2) + ((int)(((((condval_620 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_622 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_110 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_626;
            if ((((slab * 2) + m_12) == 0)) {
              condval_626 = 0;
            } else {
              int condval_627;
              if ((((slab * 2) + m_12) == 1)) {
                condval_627 = 3;
              } else {
                condval_627 = (((slab * 2) + m_12) - 1);
              }
              condval_626 = condval_627;
            }
            condval_619 = ((((((((0 <= (W >> 3)) && (0 <= rmod_110)) || (((W >> 3) < 0) && (rmod_110 <= 0))) ? rmod_110 : (rmod_110 + (W >> 3))) * 2) + ((int)(((((condval_624 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_626 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_111 = (((int)blockIdx.x) % (W >> 3));
            int condval_628;
            if ((((slab * 2) + m_12) == 0)) {
              condval_628 = 0;
            } else {
              int condval_629;
              if ((((slab * 2) + m_12) == 1)) {
                condval_629 = 3;
              } else {
                condval_629 = (((slab * 2) + m_12) - 1);
              }
              condval_628 = condval_629;
            }
            int condval_630;
            if ((((slab * 2) + m_12) == 0)) {
              condval_630 = 0;
            } else {
              int condval_631;
              if ((((slab * 2) + m_12) == 1)) {
                condval_631 = 3;
              } else {
                condval_631 = (((slab * 2) + m_12) - 1);
              }
              condval_630 = condval_631;
            }
            condval_619 = ((((((((0 <= (W >> 3)) && (0 <= rmod_111)) || (((W >> 3) < 0) && (rmod_111 <= 0))) ? rmod_111 : (rmod_111 + (W >> 3))) * 2) + ((int)(((((condval_628 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_630 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_632;
          if ((((slab * 2) + m_12) == 0)) {
            condval_632 = 0;
          } else {
            int condval_633;
            if ((((slab * 2) + m_12) == 1)) {
              condval_633 = 3;
            } else {
              condval_633 = (((slab * 2) + m_12) - 1);
            }
            condval_632 = condval_633;
          }
          int condval_634;
          if ((((slab * 2) + m_12) == 0)) {
            condval_634 = 0;
          } else {
            int condval_635;
            if ((((slab * 2) + m_12) == 1)) {
              condval_635 = 3;
            } else {
              condval_635 = (((slab * 2) + m_12) - 1);
            }
            condval_634 = condval_635;
          }
          int condval_636;
          if ((((slab * 2) + m_12) == 0)) {
            condval_636 = 0;
          } else {
            int condval_637;
            if ((((slab * 2) + m_12) == 1)) {
              condval_637 = 3;
            } else {
              condval_637 = (((slab * 2) + m_12) - 1);
            }
            condval_636 = condval_637;
          }
          int rmod_112 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_41 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_638;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_112)) || (((W >> 3) < 0) && (rmod_112 <= 0))) ? rdiv_41 : (rdiv_41 - 1)) * 2) + ((int)(((((condval_639 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_641 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_113 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_645;
            if ((((slab * 2) + m_12) == 0)) {
              condval_645 = 0;
            } else {
              int condval_646;
              if ((((slab * 2) + m_12) == 1)) {
                condval_646 = 3;
              } else {
                condval_646 = (((slab * 2) + m_12) - 1);
              }
              condval_645 = condval_646;
            }
            condval_638 = ((((((((0 <= (W >> 3)) && (0 <= rmod_113)) || (((W >> 3) < 0) && (rmod_113 <= 0))) ? rmod_113 : (rmod_113 + (W >> 3))) * 2) + ((int)(((((condval_643 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_645 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_114 = (((int)blockIdx.x) % (W >> 3));
            int condval_647;
            if ((((slab * 2) + m_12) == 0)) {
              condval_647 = 0;
            } else {
              int condval_648;
              if ((((slab * 2) + m_12) == 1)) {
                condval_648 = 3;
              } else {
                condval_648 = (((slab * 2) + m_12) - 1);
              }
              condval_647 = condval_648;
            }
            int condval_649;
            if ((((slab * 2) + m_12) == 0)) {
              condval_649 = 0;
            } else {
              int condval_650;
              if ((((slab * 2) + m_12) == 1)) {
                condval_650 = 3;
              } else {
                condval_650 = (((slab * 2) + m_12) - 1);
              }
              condval_649 = condval_650;
            }
            condval_638 = ((((((((0 <= (W >> 3)) && (0 <= rmod_114)) || (((W >> 3) < 0) && (rmod_114 <= 0))) ? rmod_114 : (rmod_114 + (W >> 3))) * 2) + ((int)(((((condval_647 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_649 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_651;
          if ((((slab * 2) + m_12) == 0)) {
            condval_651 = 0;
          } else {
            int condval_652;
            if ((((slab * 2) + m_12) == 1)) {
              condval_652 = 3;
            } else {
              condval_652 = (((slab * 2) + m_12) - 1);
            }
            condval_651 = condval_652;
          }
          int condval_653;
          if ((((slab * 2) + m_12) == 0)) {
            condval_653 = 0;
          } else {
            int condval_654;
            if ((((slab * 2) + m_12) == 1)) {
              condval_654 = 3;
            } else {
              condval_654 = (((slab * 2) + m_12) - 1);
            }
            condval_653 = condval_654;
          }
          int condval_655;
          if ((((slab * 2) + m_12) == 0)) {
            condval_655 = 0;
          } else {
            int condval_656;
            if ((((slab * 2) + m_12) == 1)) {
              condval_656 = 3;
            } else {
              condval_656 = (((slab * 2) + m_12) - 1);
            }
            condval_655 = condval_656;
          }
          int rmod_115 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_42 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_657;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_115)) || (((W >> 3) < 0) && (rmod_115 <= 0))) ? rdiv_42 : (rdiv_42 - 1)) * 2) + ((int)(((((condval_658 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_660 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_116 = (((int)blockIdx.x) % (W >> 3));
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
            int condval_664;
            if ((((slab * 2) + m_12) == 0)) {
              condval_664 = 0;
            } else {
              int condval_665;
              if ((((slab * 2) + m_12) == 1)) {
                condval_665 = 3;
              } else {
                condval_665 = (((slab * 2) + m_12) - 1);
              }
              condval_664 = condval_665;
            }
            condval_657 = ((((((((0 <= (W >> 3)) && (0 <= rmod_116)) || (((W >> 3) < 0) && (rmod_116 <= 0))) ? rmod_116 : (rmod_116 + (W >> 3))) * 2) + ((int)(((((condval_662 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_664 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_117 = (((int)blockIdx.x) % (W >> 3));
            int condval_666;
            if ((((slab * 2) + m_12) == 0)) {
              condval_666 = 0;
            } else {
              int condval_667;
              if ((((slab * 2) + m_12) == 1)) {
                condval_667 = 3;
              } else {
                condval_667 = (((slab * 2) + m_12) - 1);
              }
              condval_666 = condval_667;
            }
            int condval_668;
            if ((((slab * 2) + m_12) == 0)) {
              condval_668 = 0;
            } else {
              int condval_669;
              if ((((slab * 2) + m_12) == 1)) {
                condval_669 = 3;
              } else {
                condval_669 = (((slab * 2) + m_12) - 1);
              }
              condval_668 = condval_669;
            }
            condval_657 = ((((((((0 <= (W >> 3)) && (0 <= rmod_117)) || (((W >> 3) < 0) && (rmod_117 <= 0))) ? rmod_117 : (rmod_117 + (W >> 3))) * 2) + ((int)(((((condval_666 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_668 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
          }
          int condval_670;
          if ((((slab * 2) + m_12) == 0)) {
            condval_670 = 0;
          } else {
            int condval_671;
            if ((((slab * 2) + m_12) == 1)) {
              condval_671 = 3;
            } else {
              condval_671 = (((slab * 2) + m_12) - 1);
            }
            condval_670 = condval_671;
          }
          int condval_672;
          if ((((slab * 2) + m_12) == 0)) {
            condval_672 = 0;
          } else {
            int condval_673;
            if ((((slab * 2) + m_12) == 1)) {
              condval_673 = 3;
            } else {
              condval_673 = (((slab * 2) + m_12) - 1);
            }
            condval_672 = condval_673;
          }
          int condval_674;
          if ((((slab * 2) + m_12) == 0)) {
            condval_674 = 0;
          } else {
            int condval_675;
            if ((((slab * 2) + m_12) == 1)) {
              condval_675 = 3;
            } else {
              condval_675 = (((slab * 2) + m_12) - 1);
            }
            condval_674 = condval_675;
          }
          int rmod_118 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_43 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_676;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_118)) || (((W >> 3) < 0) && (rmod_118 <= 0))) ? rdiv_43 : (rdiv_43 - 1)) * 2) + ((int)(((((condval_677 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_679 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_119 = (((int)blockIdx.x) % (W >> 3));
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
            condval_676 = ((((((((0 <= (W >> 3)) && (0 <= rmod_119)) || (((W >> 3) < 0) && (rmod_119 <= 0))) ? rmod_119 : (rmod_119 + (W >> 3))) * 2) + ((int)(((((condval_681 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_683 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_120 = (((int)blockIdx.x) % (W >> 3));
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
            condval_676 = ((((((((0 <= (W >> 3)) && (0 <= rmod_120)) || (((W >> 3) < 0) && (rmod_120 <= 0))) ? rmod_120 : (rmod_120 + (W >> 3))) * 2) + ((int)(((((condval_685 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_687 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int rmod_121 = (((int)blockIdx.x) % (W >> 3));
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
          int condval_699;
          if ((((slab * 2) + m_12) == 0)) {
            condval_699 = 0;
          } else {
            int condval_700;
            if ((((slab * 2) + m_12) == 1)) {
              condval_700 = 3;
            } else {
              condval_700 = (((slab * 2) + m_12) - 1);
            }
            condval_699 = condval_700;
          }
          int condval_701;
          if ((((slab * 2) + m_12) == 0)) {
            condval_701 = 0;
          } else {
            int condval_702;
            if ((((slab * 2) + m_12) == 1)) {
              condval_702 = 3;
            } else {
              condval_702 = (((slab * 2) + m_12) - 1);
            }
            condval_701 = condval_702;
          }
          int rmod_122 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_44 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_703;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_122)) || (((W >> 3) < 0) && (rmod_122 <= 0))) ? rdiv_44 : (rdiv_44 - 1)) * 2) + ((int)(((((condval_704 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_706 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_123 = (((int)blockIdx.x) % (W >> 3));
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
            condval_703 = ((((((((0 <= (W >> 3)) && (0 <= rmod_123)) || (((W >> 3) < 0) && (rmod_123 <= 0))) ? rmod_123 : (rmod_123 + (W >> 3))) * 2) + ((int)(((((condval_708 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_710 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_124 = (((int)blockIdx.x) % (W >> 3));
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
            condval_703 = ((((((((0 <= (W >> 3)) && (0 <= rmod_124)) || (((W >> 3) < 0) && (rmod_124 <= 0))) ? rmod_124 : (rmod_124 + (W >> 3))) * 2) + ((int)(((((condval_712 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_714 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_718;
          if ((((slab * 2) + m_12) == 0)) {
            condval_718 = 0;
          } else {
            int condval_719;
            if ((((slab * 2) + m_12) == 1)) {
              condval_719 = 3;
            } else {
              condval_719 = (((slab * 2) + m_12) - 1);
            }
            condval_718 = condval_719;
          }
          int condval_720;
          if ((((slab * 2) + m_12) == 0)) {
            condval_720 = 0;
          } else {
            int condval_721;
            if ((((slab * 2) + m_12) == 1)) {
              condval_721 = 3;
            } else {
              condval_721 = (((slab * 2) + m_12) - 1);
            }
            condval_720 = condval_721;
          }
          int rmod_125 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_45 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_722;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_125)) || (((W >> 3) < 0) && (rmod_125 <= 0))) ? rdiv_45 : (rdiv_45 - 1)) * 2) + ((int)(((((condval_723 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_725 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_126 = (((int)blockIdx.x) % (W >> 3));
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
            condval_722 = ((((((((0 <= (W >> 3)) && (0 <= rmod_126)) || (((W >> 3) < 0) && (rmod_126 <= 0))) ? rmod_126 : (rmod_126 + (W >> 3))) * 2) + ((int)(((((condval_727 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_729 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_127 = (((int)blockIdx.x) % (W >> 3));
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
            condval_722 = ((((((((0 <= (W >> 3)) && (0 <= rmod_127)) || (((W >> 3) < 0) && (rmod_127 <= 0))) ? rmod_127 : (rmod_127 + (W >> 3))) * 2) + ((int)(((((condval_731 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_733 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_737;
          if ((((slab * 2) + m_12) == 0)) {
            condval_737 = 0;
          } else {
            int condval_738;
            if ((((slab * 2) + m_12) == 1)) {
              condval_738 = 3;
            } else {
              condval_738 = (((slab * 2) + m_12) - 1);
            }
            condval_737 = condval_738;
          }
          int condval_739;
          if ((((slab * 2) + m_12) == 0)) {
            condval_739 = 0;
          } else {
            int condval_740;
            if ((((slab * 2) + m_12) == 1)) {
              condval_740 = 3;
            } else {
              condval_740 = (((slab * 2) + m_12) - 1);
            }
            condval_739 = condval_740;
          }
          int rmod_128 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_46 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_741;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_128)) || (((W >> 3) < 0) && (rmod_128 <= 0))) ? rdiv_46 : (rdiv_46 - 1)) * 2) + ((int)(((((condval_742 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_744 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_129 = (((int)blockIdx.x) % (W >> 3));
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
            condval_741 = ((((((((0 <= (W >> 3)) && (0 <= rmod_129)) || (((W >> 3) < 0) && (rmod_129 <= 0))) ? rmod_129 : (rmod_129 + (W >> 3))) * 2) + ((int)(((((condval_746 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_748 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_130 = (((int)blockIdx.x) % (W >> 3));
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
            condval_741 = ((((((((0 <= (W >> 3)) && (0 <= rmod_130)) || (((W >> 3) < 0) && (rmod_130 <= 0))) ? rmod_130 : (rmod_130 + (W >> 3))) * 2) + ((int)(((((condval_750 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_752 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_756;
          if ((((slab * 2) + m_12) == 0)) {
            condval_756 = 0;
          } else {
            int condval_757;
            if ((((slab * 2) + m_12) == 1)) {
              condval_757 = 3;
            } else {
              condval_757 = (((slab * 2) + m_12) - 1);
            }
            condval_756 = condval_757;
          }
          int condval_758;
          if ((((slab * 2) + m_12) == 0)) {
            condval_758 = 0;
          } else {
            int condval_759;
            if ((((slab * 2) + m_12) == 1)) {
              condval_759 = 3;
            } else {
              condval_759 = (((slab * 2) + m_12) - 1);
            }
            condval_758 = condval_759;
          }
          int rmod_131 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_47 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_760;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_131)) || (((W >> 3) < 0) && (rmod_131 <= 0))) ? rdiv_47 : (rdiv_47 - 1)) * 2) + ((int)(((((condval_761 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_763 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_132 = (((int)blockIdx.x) % (W >> 3));
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
            condval_760 = ((((((((0 <= (W >> 3)) && (0 <= rmod_132)) || (((W >> 3) < 0) && (rmod_132 <= 0))) ? rmod_132 : (rmod_132 + (W >> 3))) * 2) + ((int)(((((condval_765 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_767 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_133 = (((int)blockIdx.x) % (W >> 3));
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
            condval_760 = ((((((((0 <= (W >> 3)) && (0 <= rmod_133)) || (((W >> 3) < 0) && (rmod_133 <= 0))) ? rmod_133 : (rmod_133 + (W >> 3))) * 2) + ((int)(((((condval_769 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_771 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_775;
          if ((((slab * 2) + m_12) == 0)) {
            condval_775 = 0;
          } else {
            int condval_776;
            if ((((slab * 2) + m_12) == 1)) {
              condval_776 = 3;
            } else {
              condval_776 = (((slab * 2) + m_12) - 1);
            }
            condval_775 = condval_776;
          }
          int condval_777;
          if ((((slab * 2) + m_12) == 0)) {
            condval_777 = 0;
          } else {
            int condval_778;
            if ((((slab * 2) + m_12) == 1)) {
              condval_778 = 3;
            } else {
              condval_778 = (((slab * 2) + m_12) - 1);
            }
            condval_777 = condval_778;
          }
          int rmod_134 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_48 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_779;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_134)) || (((W >> 3) < 0) && (rmod_134 <= 0))) ? rdiv_48 : (rdiv_48 - 1)) * 2) + ((int)(((((condval_780 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_782 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_135 = (((int)blockIdx.x) % (W >> 3));
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
            condval_779 = ((((((((0 <= (W >> 3)) && (0 <= rmod_135)) || (((W >> 3) < 0) && (rmod_135 <= 0))) ? rmod_135 : (rmod_135 + (W >> 3))) * 2) + ((int)(((((condval_784 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_786 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_136 = (((int)blockIdx.x) % (W >> 3));
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
            condval_779 = ((((((((0 <= (W >> 3)) && (0 <= rmod_136)) || (((W >> 3) < 0) && (rmod_136 <= 0))) ? rmod_136 : (rmod_136 + (W >> 3))) * 2) + ((int)(((((condval_788 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_790 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_794;
          if ((((slab * 2) + m_12) == 0)) {
            condval_794 = 0;
          } else {
            int condval_795;
            if ((((slab * 2) + m_12) == 1)) {
              condval_795 = 3;
            } else {
              condval_795 = (((slab * 2) + m_12) - 1);
            }
            condval_794 = condval_795;
          }
          int condval_796;
          if ((((slab * 2) + m_12) == 0)) {
            condval_796 = 0;
          } else {
            int condval_797;
            if ((((slab * 2) + m_12) == 1)) {
              condval_797 = 3;
            } else {
              condval_797 = (((slab * 2) + m_12) - 1);
            }
            condval_796 = condval_797;
          }
          int rmod_137 = (((int)blockIdx.x) % (W >> 3));
          int rdiv_49 = (((int)blockIdx.x) / (W >> 3));
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
          int condval_798;
          if ((0 < ((((((((0 <= (W >> 3)) && (0 <= rmod_137)) || (((W >> 3) < 0) && (rmod_137 <= 0))) ? rdiv_49 : (rdiv_49 - 1)) * 2) + ((int)(((((condval_799 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_801 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) & 1))) {
            int rmod_138 = (((int)blockIdx.x) % (W >> 3));
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
            condval_798 = ((((((((0 <= (W >> 3)) && (0 <= rmod_138)) || (((W >> 3) < 0) && (rmod_138 <= 0))) ? rmod_138 : (rmod_138 + (W >> 3))) * 2) + ((int)(((((condval_803 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_805 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) + 1);
          } else {
            int rmod_139 = (((int)blockIdx.x) % (W >> 3));
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
            condval_798 = ((((((((0 <= (W >> 3)) && (0 <= rmod_139)) || (((W >> 3) < 0) && (rmod_139 <= 0))) ? rmod_139 : (rmod_139 + (W >> 3))) * 2) + ((int)(((((condval_807 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_809 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) & 1) * 3);
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
          int condval_813;
          if ((((slab * 2) + m_12) == 0)) {
            condval_813 = 0;
          } else {
            int condval_814;
            if ((((slab * 2) + m_12) == 1)) {
              condval_814 = 3;
            } else {
              condval_814 = (((slab * 2) + m_12) - 1);
            }
            condval_813 = condval_814;
          }
          int condval_815;
          if ((((slab * 2) + m_12) == 0)) {
            condval_815 = 0;
          } else {
            int condval_816;
            if ((((slab * 2) + m_12) == 1)) {
              condval_816 = 3;
            } else {
              condval_816 = (((slab * 2) + m_12) - 1);
            }
            condval_815 = condval_816;
          }
          condval_427 = (((((((((((((((((((0 <= (W >> 3)) && (0 <= rmod_80)) || (((W >> 3) < 0) && (rmod_80 <= 0))) ? rdiv_30 : (rdiv_30 - 1)) * 2) + ((int)(((((condval_440 * 16) + ((int)threadIdx.x)) >> 4) == 1) | ((((condval_442 * 16) + ((int)threadIdx.x)) >> 4) == 2)))) - 1) >> 1) * 8) + (condval_444 * 2)) + ((((condval_457 * 16) + ((int)threadIdx.x)) >> 2) & 1)) >> 3) + ((H >> 3) * (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_459 & 3) * 16) + (((((condval_472 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_474 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_476 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_478 & 3) * 16) + (((((condval_491 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_493 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_495 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_497 & 3) * 16) + (((((condval_510 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_512 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_514 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_516 & 3) * 16) + (((((condval_529 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_531 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_533 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_535 & 3) * 16) + (((((condval_548 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_550 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_552 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_554 & 3) * 16) + (((((condval_567 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_569 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_571 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 10))) * (W >> 2)) * 512) + ((((((((((((0 <= (W >> 3)) && (0 <= rmod_102)) || (((W >> 3) < 0) && (rmod_102 <= 0))) ? rmod_102 : (rmod_102 + (W >> 3))) * 2) + ((int)(((((condval_573 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_575 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_577 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_579 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 5) * 512)) + (((W >> 5) * ((((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_581 & 3) * 16) + (((((condval_594 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_596 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_598 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_600 & 3) * 16) + (((((condval_613 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_615 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_617 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_619 & 3) * 16) + (((((condval_632 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_634 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_636 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_638 & 3) * 16) + (((((condval_651 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_653 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_655 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_657 & 3) * 16) + (((((condval_670 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_672 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_674 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_676 & 3) * 16) + (((((condval_689 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_691 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_693 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + (((((((((((((0 <= (W >> 3)) && (0 <= rmod_121)) || (((W >> 3) < 0) && (rmod_121 <= 0))) ? rmod_121 : (rmod_121 + (W >> 3))) * 2) + ((int)(((((condval_695 * 16) + ((int)threadIdx.x)) >> 4) == 2) | ((((condval_697 * 16) + ((int)threadIdx.x)) >> 4) == 3)))) >> 1) * 8) + ((((condval_699 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_701 * 16) + ((int)threadIdx.x)) >> 3) & 1)) >> 3) & 3) * 128)) + (((((((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) >> 4) * 1024) + (((((((((((((condval_703 & 3) * 16) + (((((condval_716 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_718 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_720 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 1) << 4) ^ (((((((condval_722 & 3) * 16) + (((((condval_735 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_737 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_739 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 6) >> 1)) ^ ((((((condval_741 & 3) * 16) + (((((condval_754 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_756 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_758 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 8)) ^ (((((((condval_760 & 3) * 16) + (((((condval_773 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_775 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_777 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 16) << 1)) ^ ((((((condval_779 & 3) * 16) + (((((condval_792 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_794 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_796 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32)) ^ (((((((condval_798 & 3) * 16) + (((((condval_811 * 16) + ((int)threadIdx.x)) >> 2) & 1) * 8)) + ((((condval_813 * 16) + ((int)threadIdx.x)) & 3) << 1)) + ((((condval_815 * 16) + ((int)threadIdx.x)) >> 3) & 1)) & 32) >> 3)) * 16)) + (((((p_9 * 4) & 3) | (((p_9 * 4) & 4) << 2)) | (((p_9 * 4) & 24) >> 1)) & 15)) & 127));
        } else {
          condval_427 = -1;
        }
        int dest = max(condval_37, condval_427);
        if ((((int)threadIdx.x) < 16) & (0 <= dest)) {
          if (0 <= (out_offset + dest)) {
            if ((out_offset + dest) < sizes_6) {
              nr_tl_shallow_joint::st128((&(X[(((int64_t)out_offset) + ((int64_t)dest))])), q[0], q[1], q[2], q[3]);
            }
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

