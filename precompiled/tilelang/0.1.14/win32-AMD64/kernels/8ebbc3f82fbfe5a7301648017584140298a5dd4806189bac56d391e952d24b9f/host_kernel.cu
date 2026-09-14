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
    int rmod = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rdiv = (((int)blockIdx.x) / ((W0 >> 4) + 1));
    int rmod_1 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rdiv_1 = (((int)blockIdx.x) / ((W0 >> 4) + 1));
    int rmod_2 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rmod_3 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int condval;
    if (((((0 <= ((((-1 <= (W0 >> 4)) && (0 <= rmod)) || (((W0 >> 4) < -1) && (rmod <= 0))) ? rdiv : (rdiv - 1))) & (((((((-1 <= (W0 >> 4)) && (0 <= rmod_1)) || (((W0 >> 4) < -1) && (rmod_1 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H0 >> 3))) & (1 <= ((((((-1 <= (W0 >> 4)) && (0 <= rmod_2)) || (((W0 >> 4) < -1) && (rmod_2 <= 0))) ? rmod_2 : ((rmod_2 + (W0 >> 4)) + 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & (((((((-1 <= (W0 >> 4)) && (0 <= rmod_3)) || (((W0 >> 4) < -1) && (rmod_3 <= 0))) ? rmod_3 : ((rmod_3 + (W0 >> 4)) + 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) <= (W0 >> 3)))) {
      int rmod_4 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
      int rmod_5 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
      int rdiv_2 = (((int)blockIdx.x) / ((W0 >> 4) + 1));
      condval = ((((((((((((-1 <= (W0 >> 4)) && (0 <= rmod_4)) || (((W0 >> 4) < -1) && (rmod_4 <= 0))) ? rmod_4 : ((rmod_4 + (W0 >> 4)) + 1)) * 1024) + ((((((((-1 <= (W0 >> 4)) && (0 <= rmod_5)) || (((W0 >> 4) < -1) && (rmod_5 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W0 >> 3)) * 512)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
    } else {
      condval = -1;
    }
    int rmod_6 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rdiv_3 = (((int)blockIdx.x) / ((W0 >> 4) + 1));
    int rmod_7 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rdiv_4 = (((int)blockIdx.x) / ((W0 >> 4) + 1));
    int rmod_8 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int rmod_9 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
    int condval_1;
    if (((((0 <= ((((-1 <= (W0 >> 4)) && (0 <= rmod_6)) || (((W0 >> 4) < -1) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1))) & (((((((-1 <= (W0 >> 4)) && (0 <= rmod_7)) || (((W0 >> 4) < -1) && (rmod_7 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H0 >> 3))) & (1 <= ((((((-1 <= (W0 >> 4)) && (0 <= rmod_8)) || (((W0 >> 4) < -1) && (rmod_8 <= 0))) ? rmod_8 : ((rmod_8 + (W0 >> 4)) + 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & (((((((-1 <= (W0 >> 4)) && (0 <= rmod_9)) || (((W0 >> 4) < -1) && (rmod_9 <= 0))) ? rmod_9 : ((rmod_9 + (W0 >> 4)) + 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) <= (W0 >> 3)))) {
      int rmod_10 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
      int rmod_11 = (((int)blockIdx.x) % ((W0 >> 4) + 1));
      int rdiv_5 = (((int)blockIdx.x) / ((W0 >> 4) + 1));
      condval_1 = ((((((((((((-1 <= (W0 >> 4)) && (0 <= rmod_10)) || (((W0 >> 4) < -1) && (rmod_10 <= 0))) ? rmod_10 : ((rmod_10 + (W0 >> 4)) + 1)) * 1024) + ((((((((-1 <= (W0 >> 4)) && (0 <= rmod_11)) || (((W0 >> 4) < -1) && (rmod_11 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * 2) + ((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W0 >> 3)) * 512)) + (((int)(((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((m * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((m * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((m * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
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
  ushort v_ = (ushort)14904;
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
          ushort v__1 = (ushort)14904;
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
      int rmod_12 = (((int)blockIdx.x) % ((W >> 3) + 1));
      int rdiv_6 = (((int)blockIdx.x) / ((W >> 3) + 1));
      int rmod_13 = (((int)blockIdx.x) % ((W >> 3) + 1));
      int rdiv_7 = (((int)blockIdx.x) / ((W >> 3) + 1));
      int condval_37;
      if ((((slab * 2) + m_12) == 0)) {
        condval_37 = 0;
      } else {
        int condval_38;
        if ((((slab * 2) + m_12) == 1)) {
          condval_38 = 3;
        } else {
          condval_38 = (((slab * 2) + m_12) - 1);
        }
        condval_37 = condval_38;
      }
      int condval_39;
      if ((((slab * 2) + m_12) == 0)) {
        condval_39 = 0;
      } else {
        int condval_40;
        if ((((slab * 2) + m_12) == 1)) {
          condval_40 = 3;
        } else {
          condval_40 = (((slab * 2) + m_12) - 1);
        }
        condval_39 = condval_40;
      }
      int rmod_14 = (((int)blockIdx.x) % ((W >> 3) + 1));
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
      int rmod_15 = (((int)blockIdx.x) % ((W >> 3) + 1));
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
      int condval_36;
      if (((((0 <= ((((-1 <= (W >> 3)) && (0 <= rmod_12)) || (((W >> 3) < -1) && (rmod_12 <= 0))) ? rdiv_6 : (rdiv_6 - 1))) & (((((((-1 <= (W >> 3)) && (0 <= rmod_13)) || (((W >> 3) < -1) && (rmod_13 <= 0))) ? rdiv_7 : (rdiv_7 - 1)) * 2) + ((int)(((((condval_37 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_39 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H >> 2))) & (1 <= ((((((-1 <= (W >> 3)) && (0 <= rmod_14)) || (((W >> 3) < -1) && (rmod_14 <= 0))) ? rmod_14 : ((rmod_14 + (W >> 3)) + 1)) * 2) + ((int)(((((condval_41 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_43 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & (((((((-1 <= (W >> 3)) && (0 <= rmod_15)) || (((W >> 3) < -1) && (rmod_15 <= 0))) ? rmod_15 : ((rmod_15 + (W >> 3)) + 1)) * 2) + ((int)(((((condval_45 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_47 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) <= (W >> 2)))) {
        int rmod_16 = (((int)blockIdx.x) % ((W >> 3) + 1));
        int rmod_17 = (((int)blockIdx.x) % ((W >> 3) + 1));
        int rdiv_8 = (((int)blockIdx.x) / ((W >> 3) + 1));
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
        condval_36 = ((((((((((((-1 <= (W >> 3)) && (0 <= rmod_16)) || (((W >> 3) < -1) && (rmod_16 <= 0))) ? rmod_16 : ((rmod_16 + (W >> 3)) + 1)) * 1024) + ((((((((-1 <= (W >> 3)) && (0 <= rmod_17)) || (((W >> 3) < -1) && (rmod_17 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * 2) + ((int)(((((condval_49 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_51 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W >> 2)) * 512)) + (((int)(((((condval_53 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_55 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_57 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_59 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
      } else {
        condval_36 = -1;
      }
      int rmod_18 = (((int)blockIdx.x) % ((W >> 3) + 1));
      int rdiv_9 = (((int)blockIdx.x) / ((W >> 3) + 1));
      int rmod_19 = (((int)blockIdx.x) % ((W >> 3) + 1));
      int rdiv_10 = (((int)blockIdx.x) / ((W >> 3) + 1));
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
      int rmod_20 = (((int)blockIdx.x) % ((W >> 3) + 1));
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
      int rmod_21 = (((int)blockIdx.x) % ((W >> 3) + 1));
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
      int condval_61;
      if (((((0 <= ((((-1 <= (W >> 3)) && (0 <= rmod_18)) || (((W >> 3) < -1) && (rmod_18 <= 0))) ? rdiv_9 : (rdiv_9 - 1))) & (((((((-1 <= (W >> 3)) && (0 <= rmod_19)) || (((W >> 3) < -1) && (rmod_19 <= 0))) ? rdiv_10 : (rdiv_10 - 1)) * 2) + ((int)(((((condval_62 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_64 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) < (H >> 2))) & (1 <= ((((((-1 <= (W >> 3)) && (0 <= rmod_20)) || (((W >> 3) < -1) && (rmod_20 <= 0))) ? rmod_20 : ((rmod_20 + (W >> 3)) + 1)) * 2) + ((int)(((((condval_66 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_68 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))))) & (((((((-1 <= (W >> 3)) && (0 <= rmod_21)) || (((W >> 3) < -1) && (rmod_21 <= 0))) ? rmod_21 : ((rmod_21 + (W >> 3)) + 1)) * 2) + ((int)(((((condval_70 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_72 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3)))) <= (W >> 2)))) {
        int rmod_22 = (((int)blockIdx.x) % ((W >> 3) + 1));
        int rmod_23 = (((int)blockIdx.x) % ((W >> 3) + 1));
        int rdiv_11 = (((int)blockIdx.x) / ((W >> 3) + 1));
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
        condval_61 = ((((((((((((-1 <= (W >> 3)) && (0 <= rmod_22)) || (((W >> 3) < -1) && (rmod_22 <= 0))) ? rmod_22 : ((rmod_22 + (W >> 3)) + 1)) * 1024) + ((((((((-1 <= (W >> 3)) && (0 <= rmod_23)) || (((W >> 3) < -1) && (rmod_23 <= 0))) ? rdiv_11 : (rdiv_11 - 1)) * 2) + ((int)(((((condval_74 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 1) | ((((condval_76 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2)))) * (W >> 2)) * 512)) + (((int)(((((condval_78 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 2) | ((((condval_80 * 16) + (((int)threadIdx.x) >> 2)) >> 4) == 3))) * 512)) + ((((condval_82 * 16) + (((int)threadIdx.x) >> 2)) & 7) * 64)) + ((((((int)threadIdx.x) & 3) * 8) >> 2) * 8)) + (((((condval_84 * 16) + (((int)threadIdx.x) >> 2)) >> 3) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) - 512);
      } else {
        condval_61 = -1;
      }
      int dest = max(condval_36, condval_61);
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

