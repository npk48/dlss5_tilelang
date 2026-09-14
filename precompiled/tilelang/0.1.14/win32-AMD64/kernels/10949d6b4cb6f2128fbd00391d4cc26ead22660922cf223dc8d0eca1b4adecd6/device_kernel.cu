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
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ Adapter, const half_t* __restrict__ G0, const half_t* __restrict__ G1, const half_t* __restrict__ Gate, float* __restrict__ Head, const half_t* __restrict__ Readout, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y, int H0, int W0, int counter, int gx, int gy, int sizes_0, int sizes_1, int sizes_13, int sizes_14, int sizes_15, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6, int sizes_7, int sizes_9);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const half_t* __restrict__ Adapter, const half_t* __restrict__ G0, const half_t* __restrict__ G1, const half_t* __restrict__ Gate, float* __restrict__ Head, const half_t* __restrict__ Readout, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y, int H0, int W0, int counter, int gx, int gy, int sizes_0, int sizes_1, int sizes_13, int sizes_14, int sizes_15, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6, int sizes_7, int sizes_9) {
  uint raw[32];
  uint pool_a[4];
  uint sk[4];
  uint mp[4];
  uint aa[16];
  uint ff[32];
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
  uint z_1[4];
  uint b_4[2];
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
    bool valid = ((((1 <= ((((int)blockIdx.x) * 2) + ((int)((m == 2) | (m == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((m == 2) | (m == 3))) * 4)) + ((((int)threadIdx.x) >> 2) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((m == 1) | (m == 2))) * 4)) + ((((int)threadIdx.x) >> 2) >> 2)) < (H0 + 4)));
    int sa = max((((((((((int)blockIdx.x) * 1024) + (((((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))) - 1) * (W0 >> 2)) * 512)) + (((int)((m == 2) | (m == 3))) * 512)) + (((((int)threadIdx.x) >> 2) >> 2) * 256)) + (((((int)threadIdx.x) >> 2) & 3) * 64)) + ((((int)threadIdx.x) & 3) * 16)) - 512), (((((((((int)blockIdx.x) * 1024) + (((((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))) - 1) * (W0 >> 2)) * 512)) + (((int)((m == 2) | (m == 3))) * 512)) + (((((int)threadIdx.x) >> 2) >> 2) * 256)) + (((((int)threadIdx.x) >> 2) & 3) * 64)) + ((((int)threadIdx.x) & 3) * 16)) - 512));
    int condval;
    if (((((1 <= ((((int)blockIdx.x) * 2) + ((int)((m == 2) | (m == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((m == 2) | (m == 3))) * 4)) + ((((int)threadIdx.x) >> 2) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((m == 1) | (m == 2))) * 4)) + ((((int)threadIdx.x) >> 2) >> 2)) < (H0 + 4)))) {
      condval = sa;
    } else {
      condval = -1;
    }
    int condval_1;
    if (((((1 <= ((((int)blockIdx.x) * 2) + ((int)((m == 2) | (m == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((m == 2) | (m == 3))) * 4)) + ((((int)threadIdx.x) >> 2) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((m == 1) | (m == 2))) * 4)) + ((((int)threadIdx.x) >> 2) >> 2)) < (H0 + 4)))) {
      condval_1 = sa;
    } else {
      condval_1 = -1;
    }
    int base = max(condval, condval_1);
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      sk[j_2] = (uint)0;
    }
    if (0 <= base) {
      if (base < sizes_7) {
        nr_tl_shallow_joint::ld128((&(sk[0])), (&(Y[((int64_t)base)])));
      }
    }
    #pragma unroll
    for (int j_3 = 0; j_3 < 4; ++j_3) {
      mp[j_3] = (uint)0;
    }
    if (((((1 <= ((((int)blockIdx.x) * 2) + ((int)((m == 2) | (m == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((m == 2) | (m == 3))) * 4)) + ((((int)threadIdx.x) >> 2) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((m == 1) | (m == 2))) * 4)) + ((((int)threadIdx.x) >> 2) >> 2)) < (H0 + 4))) & ((((int)threadIdx.x) & 20) == 0)) {
      int ma = max(((((((((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) >> 3) + ((H0 >> 4) * (((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) >> 10))) * (W0 >> 3)) * 512) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) >> 5) * 512)) + (((W0 >> 6) * ((((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) >> 7) & 7)) * 512)) + (((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) >> 3) & 3) * 128)) + (((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) & 127)), ((((((((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) >> 3) + ((H0 >> 4) * (((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) >> 10))) * (W0 >> 3)) * 512) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) >> 5) * 512)) + (((W0 >> 6) * ((((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) >> 7) & 7)) * 512)) + (((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) >> 3) & 3) * 128)) + (((((((int)threadIdx.x) & 1) * 1024) + ((((((((int)blockIdx.y) * 4) + (((int)((m == 1) | (m == 2))) * 2)) + ((((int)threadIdx.x) >> 1) & 1)) - 2) & 7) * 128)) + ((((((((int)blockIdx.x) * 4) + (((int)((m == 2) | (m == 3))) * 2)) + ((((int)threadIdx.x) >> 3) & 1)) - 2) & 7) * 16)) & 127)));
      if (0 <= ma) {
        if (ma < sizes_6) {
          nr_tl_shallow_joint::ld128((&(mp[0])), (&(X[((int64_t)ma)])));
        }
      }
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint s0 = nr_tl_shallow::shfl(mp[0], (((i * 2) + (((int)threadIdx.x) & 8)) + p));
        uint s1 = nr_tl_shallow::shfl(mp[1], (((i * 2) + (((int)threadIdx.x) & 8)) + p));
        uint s2 = nr_tl_shallow::shfl(mp[2], (((i * 2) + (((int)threadIdx.x) & 8)) + p));
        uint s3 = nr_tl_shallow::shfl(mp[3], (((i * 2) + (((int)threadIdx.x) & 8)) + p));
        uint condval_2;
        if (((((int)threadIdx.x) % 4) == 0)) {
          condval_2 = s0;
        } else {
          uint condval_3;
          if (((((int)threadIdx.x) & 3) == 1)) {
            condval_3 = s1;
          } else {
            uint condval_4;
            if (((((int)threadIdx.x) & 3) == 2)) {
              condval_4 = s2;
            } else {
              condval_4 = s3;
            }
            condval_3 = condval_4;
          }
          condval_2 = condval_3;
        }
        uint mv = condval_2;
        if ((((1 <= ((((int)blockIdx.x) * 2) + ((int)((m == 2) | (m == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((m == 2) | (m == 3))) * 4)) + ((((int)threadIdx.x) >> 2) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((m == 1) | (m == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((m == 1) | (m == 2))) * 4)) + ((((int)threadIdx.x) >> 2) >> 2)) < (H0 + 4))) {
          #pragma unroll
          for (int j_4 = 0; j_4 < 2; ++j_4) {
            uint sv = sk[((p * 2) + i)];
            half_t condval_5;
            if ((((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) < sizes_9)) {
              condval_5 = Adapter[((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2))];
            } else {
              condval_5 = half_t(0x0p+0f/*0.000000e+00*/);
            }
            uint condval_6;
            if (((((int)threadIdx.x) % 4) == 0)) {
              condval_6 = s0;
            } else {
              uint condval_7;
              if (((((int)threadIdx.x) & 3) == 1)) {
                condval_7 = s1;
              } else {
                uint condval_8;
                if (((((int)threadIdx.x) & 3) == 2)) {
                  condval_8 = s2;
                } else {
                  condval_8 = s3;
                }
                condval_7 = condval_8;
              }
              condval_6 = condval_7;
            }
            half_t condval_9;
            if ((((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) < sizes_13)) {
              condval_9 = Gate[((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2))];
            } else {
              condval_9 = half_t(0x0p+0f/*0.000000e+00*/);
            }
            half_t condval_10;
            if (((((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) + 1) < sizes_9)) {
              condval_10 = Adapter[(((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) + 1)];
            } else {
              condval_10 = half_t(0x0p+0f/*0.000000e+00*/);
            }
            uint condval_11;
            if (((((int)threadIdx.x) % 4) == 0)) {
              condval_11 = s0;
            } else {
              uint condval_12;
              if (((((int)threadIdx.x) & 3) == 1)) {
                condval_12 = s1;
              } else {
                uint condval_13;
                if (((((int)threadIdx.x) & 3) == 2)) {
                  condval_13 = s2;
                } else {
                  condval_13 = s3;
                }
                condval_12 = condval_13;
              }
              condval_11 = condval_12;
            }
            half_t condval_14;
            if (((((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) + 1) < sizes_13)) {
              condval_14 = Gate[(((((((int)threadIdx.x) & 3) * 8) + (p * 4)) + (j_4 * 2)) + 1)];
            } else {
              condval_14 = half_t(0x0p+0f/*0.000000e+00*/);
            }
            raw[((((m * 8) + (p * 4)) + (j_4 * 2)) + i)] = nr_tl_shallow::pack(((half_t)nr_tl_shallow::hfma(((half_t)nr_tl_shallow::une4((sv >> ((uint)(j_4 * 16))))), condval_5, ((half_t)nr_tl_shallow::hmul(((half_t)nr_tl_shallow::une4((condval_6 >> ((uint)(j_4 * 16))))), condval_9)))), ((half_t)nr_tl_shallow::hfma(((half_t)nr_tl_shallow::une4((sv >> ((uint)((j_4 * 16) + 8))))), condval_10, ((half_t)nr_tl_shallow::hmul(((half_t)nr_tl_shallow::une4((condval_11 >> ((uint)((j_4 * 16) + 8))))), condval_14)))));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int m_1 = 0; m_1 < 4; ++m_1) {
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; ++p_1) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        aa[(((m_1 * 4) + (p_1 * 2)) + i_1)] = (nr_tl_shallow::e4pair(raw[(((m_1 * 8) + (p_1 * 4)) + i_1)]) | (nr_tl_shallow::e4pair(raw[((((m_1 * 8) + (p_1 * 4)) + i_1) + 2)]) << (uint)16));
      }
    }
  }
  #pragma unroll
  for (int j_5 = 0; j_5 < 32; ++j_5) {
    half_t condval_15;
    if ((((((j_5 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_4)) {
      condval_15 = G0[((((j_5 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
    } else {
      condval_15 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_16;
    if (((((((j_5 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_4)) {
      condval_16 = G0[(((((j_5 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
    } else {
      condval_16 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    ff[j_5] = nr_tl_shallow::mul(raw[j_5], nr_tl_shallow::pack(condval_15, condval_16));
  }
  ushort v_ = (ushort)18680;
  half_t scale = (*(half_t *)(&(v_)));
  for (int part = 0; part < 4; ++part) {
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      #pragma unroll
      for (int j_6 = 0; j_6 < 16; ++j_6) {
        hidden[j_6] = (uint)0;
      }
      if ((((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4)) < sizes_0) {
        nr_tl_shallow_joint::ld128((&(b[0])), (&(Wt0[(((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_2 = 0; m_2 < 4; ++m_2) {
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          nr_tl_shallow::mma8((&(hidden[((m_2 * 4) + (n * 2))])), aa[(m_2 * 4)], aa[((m_2 * 4) + 1)], aa[((m_2 * 4) + 2)], aa[((m_2 * 4) + 3)], b[(n * 2)], b[((n * 2) + 1)]);
        }
      }
      #pragma unroll
      for (int m_3 = 0; m_3 < 4; ++m_3) {
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          ha_[(((m_3 * 4) + (pair * 2)) + i_2)] = (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[((m_3 * 4) + i_2)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_3 * 4) + i_2)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_3 * 4) + i_2)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) | (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[(((m_3 * 4) + i_2) + 2)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_3 * 4) + i_2) + 2)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_3 * 4) + i_2) + 2)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; ++p_2) {
      if ((((part * 256) + (p_2 * 128)) + (((int)threadIdx.x) * 4)) < sizes_1) {
        nr_tl_shallow_joint::ld128((&(b_1[0])), (&(Wt1[(((part * 256) + (p_2 * 128)) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_4 = 0; m_4 < 4; ++m_4) {
        #pragma unroll
        for (int n_1 = 0; n_1 < 2; ++n_1) {
          nr_tl_shallow::mma8((&(ff[(((m_4 * 8) + (p_2 * 4)) + (n_1 * 2))])), ha_[(m_4 * 4)], ha_[((m_4 * 4) + 1)], ha_[((m_4 * 4) + 2)], ha_[((m_4 * 4) + 3)], b_1[(n_1 * 2)], b_1[((n_1 * 2) + 1)]);
        }
      }
    }
  }
  #pragma unroll
  for (int m_5 = 0; m_5 < 4; ++m_5) {
    #pragma unroll
    for (int p_3 = 0; p_3 < 2; ++p_3) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        aa[(((m_5 * 4) + (p_3 * 2)) + i_3)] = (nr_tl_shallow::e4pair(ff[(((m_5 * 8) + (p_3 * 4)) + i_3)]) | (nr_tl_shallow::e4pair(ff[((((m_5 * 8) + (p_3 * 4)) + i_3) + 2)]) << (uint)16));
      }
    }
  }
  #pragma unroll
  for (int component = 0; component < 3; ++component) {
    #pragma unroll
    for (int j_7 = 0; j_7 < 32; ++j_7) {
      z[j_7] = (uint)0;
    }
    #pragma unroll
    for (int p_4 = 0; p_4 < 2; ++p_4) {
      if ((((component * 256) + (p_4 * 128)) + (((int)threadIdx.x) * 4)) < sizes_2) {
        nr_tl_shallow_joint::ld128((&(b_2[0])), (&(Wq[(((component * 256) + (p_4 * 128)) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_6 = 0; m_6 < 4; ++m_6) {
        #pragma unroll
        for (int n_2 = 0; n_2 < 2; ++n_2) {
          nr_tl_shallow::mma8((&(z[(((m_6 * 8) + (p_4 * 4)) + (n_2 * 2))])), aa[(m_6 * 4)], aa[((m_6 * 4) + 1)], aa[((m_6 * 4) + 2)], aa[((m_6 * 4) + 3)], b_2[(n_2 * 2)], b_2[((n_2 * 2) + 1)]);
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
      for (int j_8 = 0; j_8 < 32; ++j_8) {
        z[j_8] = nr_tl_shallow::mul(z[j_8], inv[(((j_8 >> 3) * 2) + (j_8 & 1))]);
        if (component == 0) {
          ushort v__1 = (ushort)18680;
          z[j_8] = nr_tl_shallow::mul(z[j_8], nr_tl_shallow::pack((*(half_t *)(&(v__1))), (*(half_t *)(&(v__1)))));
        }
      }
    }
    #pragma unroll
    for (int m_7 = 0; m_7 < 4; ++m_7) {
      #pragma unroll
      for (int p_5 = 0; p_5 < 2; ++p_5) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          if (component == 0) {
            int64_t condval_17;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_17 = (int64_t)0;
            } else {
              int64_t condval_18;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_18 = (int64_t)3;
              } else {
                condval_18 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_17 = condval_18;
            }
            int64_t condval_19;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_19 = (int64_t)0;
            } else {
              int64_t condval_20;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_20 = (int64_t)3;
              } else {
                condval_20 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_19 = condval_20;
            }
            int64_t condval_21;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_21 = (int64_t)0;
            } else {
              int64_t condval_22;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_22 = (int64_t)3;
              } else {
                condval_22 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_21 = condval_22;
            }
            int64_t condval_23;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_23 = (int64_t)0;
            } else {
              int64_t condval_24;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_24 = (int64_t)3;
              } else {
                condval_24 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_23 = condval_24;
            }
            qa[(((m_7 * 4) + (p_5 * 2)) + i_4)] = (nr_tl_shallow::e4pair(z[(((condval_19 * (int64_t)8) + (((int64_t)p_5) * (int64_t)4)) + ((int64_t)i_4))]) | (nr_tl_shallow::e4pair(z[((((condval_23 * (int64_t)8) + (((int64_t)p_5) * (int64_t)4)) + ((int64_t)i_4)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_25;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_25 = (int64_t)0;
            } else {
              int64_t condval_26;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_26 = (int64_t)3;
              } else {
                condval_26 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_25 = condval_26;
            }
            int64_t condval_27;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_27 = (int64_t)0;
            } else {
              int64_t condval_28;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_28 = (int64_t)3;
              } else {
                condval_28 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_27 = condval_28;
            }
            int64_t condval_29;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_29 = (int64_t)0;
            } else {
              int64_t condval_30;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_30 = (int64_t)3;
              } else {
                condval_30 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_29 = condval_30;
            }
            int64_t condval_31;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_31 = (int64_t)0;
            } else {
              int64_t condval_32;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_32 = (int64_t)3;
              } else {
                condval_32 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_31 = condval_32;
            }
            kb[(((m_7 * 4) + (i_4 * 2)) + p_5)] = (nr_tl_shallow::e4pair(z[(((condval_27 * (int64_t)8) + (((int64_t)p_5) * (int64_t)4)) + ((int64_t)i_4))]) | (nr_tl_shallow::e4pair(z[((((condval_31 * (int64_t)8) + (((int64_t)p_5) * (int64_t)4)) + ((int64_t)i_4)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          int64_t condval_33;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_33 = (int64_t)0;
          } else {
            condval_33 = (int64_t)1;
          }
          int64_t condval_34;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_34 = (int64_t)0;
          } else {
            condval_34 = (int64_t)1;
          }
          int64_t condval_35;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_35 = (int64_t)0;
          } else {
            condval_35 = (int64_t)1;
          }
          int64_t condval_36;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_36 = (int64_t)0;
          } else {
            condval_36 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_3 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_34 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_36 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_37;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_37 = (int64_t)3;
          } else {
            condval_37 = (int64_t)2;
          }
          int64_t condval_38;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_38 = (int64_t)3;
          } else {
            condval_38 = (int64_t)2;
          }
          int64_t condval_39;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_39 = (int64_t)3;
          } else {
            condval_39 = (int64_t)2;
          }
          int64_t condval_40;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_40 = (int64_t)3;
          } else {
            condval_40 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_3 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_38 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_40 * (int64_t)8) + (((int64_t)n_3) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
        }
      }
    }
  }
  #pragma unroll
  for (int slab = 0; slab < 2; ++slab) {
    #pragma unroll
    for (int m_8 = 0; m_8 < 2; ++m_8) {
      #pragma unroll
      for (int p_6 = 0; p_6 < 4; ++p_6) {
        if ((((((slab * 1024) + (m_8 * 512)) + (p_6 * 128)) + (((int)threadIdx.x) * 4)) + 768) < sizes_2) {
          nr_tl_shallow_joint::ld128((&(seed[0])), (&(Wq[(((((slab * 1024) + (m_8 * 512)) + (p_6 * 128)) + (((int)threadIdx.x) * 4)) + 768)])));
        }
        #pragma unroll
        for (int n_4 = 0; n_4 < 2; ++n_4) {
          logits[(((m_8 * 16) + (p_6 * 4)) + (n_4 * 2))] = seed[(n_4 * 2)];
          logits[((((m_8 * 16) + (p_6 * 4)) + (n_4 * 2)) + 1)] = seed[((n_4 * 2) + 1)];
          nr_tl_shallow::mma8((&(logits[(((m_8 * 16) + (p_6 * 4)) + (n_4 * 2))])), qa[((slab * 8) + (m_8 * 4))], qa[(((slab * 8) + (m_8 * 4)) + 1)], qa[(((slab * 8) + (m_8 * 4)) + 2)], qa[(((slab * 8) + (m_8 * 4)) + 3)], kb[((p_6 * 4) + (n_4 * 2))], kb[(((p_6 * 4) + (n_4 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int j_9 = 0; j_9 < 32; ++j_9) {
      logits[j_9] = (((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_9], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) & (uint)65535) << (uint)5) + (uint)32768) & (uint)65535) | ((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_9], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) >> (uint)16) << (uint)5) + (uint)32768) << (uint)16));
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
    for (int j_10 = 0; j_10 < 32; ++j_10) {
      logits[j_10] = nr_tl_shallow::mul(logits[j_10], inv[(((j_10 >> 4) * 2) + (j_10 & 1))]);
    }
    #pragma unroll
    for (int part_2 = 0; part_2 < 2; ++part_2) {
      #pragma unroll
      for (int m_9 = 0; m_9 < 2; ++m_9) {
        #pragma unroll
        for (int p_7 = 0; p_7 < 2; ++p_7) {
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            pa[((((part_2 * 8) + (m_9 * 4)) + (p_7 * 2)) + i_5)] = (nr_tl_shallow::e4pair(logits[((((m_9 * 16) + (part_2 * 8)) + (p_7 * 4)) + i_5)]) | (nr_tl_shallow::e4pair(logits[(((((m_9 * 16) + (part_2 * 8)) + (p_7 * 4)) + i_5) + 2)]) << (uint)16));
          }
        }
      }
    }
    #pragma unroll
    for (int j_11 = 0; j_11 < 16; ++j_11) {
      attended[j_11] = (uint)0;
    }
    #pragma unroll
    for (int part_3 = 0; part_3 < 2; ++part_3) {
      #pragma unroll
      for (int n_5 = 0; n_5 < 4; ++n_5) {
        #pragma unroll
        for (int m_10 = 0; m_10 < 2; ++m_10) {
          nr_tl_shallow::mma8((&(attended[((m_10 * 8) + (n_5 * 2))])), pa[((part_3 * 8) + (m_10 * 4))], pa[(((part_3 * 8) + (m_10 * 4)) + 1)], pa[(((part_3 * 8) + (m_10 * 4)) + 2)], pa[(((part_3 * 8) + (m_10 * 4)) + 3)], vb[((part_3 * 8) + (n_5 * 2))], vb[(((part_3 * 8) + (n_5 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int m_11 = 0; m_11 < 2; ++m_11) {
      #pragma unroll
      for (int p_8 = 0; p_8 < 2; ++p_8) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          aa[(((m_11 * 4) + (p_8 * 2)) + i_6)] = (nr_tl_shallow::e4pair(attended[(((m_11 * 8) + (p_8 * 4)) + i_6)]) | (nr_tl_shallow::e4pair(attended[((((m_11 * 8) + (p_8 * 4)) + i_6) + 2)]) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int j_12 = 0; j_12 < 16; ++j_12) {
      int64_t condval_41;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) == (int64_t)0)) {
        condval_41 = (int64_t)0;
      } else {
        int64_t condval_42;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) == (int64_t)1)) {
          condval_42 = (int64_t)3;
        } else {
          condval_42 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) - (int64_t)1);
        }
        condval_41 = condval_42;
      }
      int64_t condval_43;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) == (int64_t)0)) {
        condval_43 = (int64_t)0;
      } else {
        int64_t condval_44;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) == (int64_t)1)) {
          condval_44 = (int64_t)3;
        } else {
          condval_44 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_12) >> (int64_t)3)) - (int64_t)1);
        }
        condval_43 = condval_44;
      }
      half_t condval_45;
      if ((((((j_12 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_5)) {
        condval_45 = G1[((((j_12 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
      } else {
        condval_45 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      half_t condval_46;
      if (((((((j_12 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_5)) {
        condval_46 = G1[(((((j_12 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
      } else {
        condval_46 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      out[j_12] = nr_tl_shallow::mul(ff[((condval_43 * (int64_t)8) + (((int64_t)j_12) & (int64_t)7))], nr_tl_shallow::pack(condval_45, condval_46));
    }
    #pragma unroll
    for (int p_9 = 0; p_9 < 2; ++p_9) {
      if (((p_9 * 128) + (((int)threadIdx.x) * 4)) < sizes_3) {
        nr_tl_shallow_joint::ld128((&(b_3[0])), (&(Wp[((p_9 * 128) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_12 = 0; m_12 < 2; ++m_12) {
        #pragma unroll
        for (int n_6 = 0; n_6 < 2; ++n_6) {
          nr_tl_shallow::mma8((&(out[(((m_12 * 8) + (p_9 * 4)) + (n_6 * 2))])), aa[(m_12 * 4)], aa[((m_12 * 4) + 1)], aa[((m_12 * 4) + 2)], aa[((m_12 * 4) + 3)], b_3[(n_6 * 2)], b_3[((n_6 * 2) + 1)]);
        }
      }
    }
    int H = max((H0 >> 1), (H0 >> 1));
    int W = max((W0 >> 1), (W0 >> 1));
    #pragma unroll
    for (int j_13 = 0; j_13 < 4; ++j_13) {
      z_1[j_13] = (uint)0;
    }
    #pragma unroll
    for (int kp = 0; kp < 2; ++kp) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        half_t condval_47;
        if ((((((kp * 128) + (i_7 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((int)threadIdx.x) >> 2)) < sizes_14)) {
          condval_47 = Readout[((((kp * 128) + (i_7 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((int)threadIdx.x) >> 2))];
        } else {
          condval_47 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        half_t condval_48;
        if (((((((kp * 128) + (i_7 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((int)threadIdx.x) >> 2)) + 8) < sizes_14)) {
          condval_48 = Readout[(((((kp * 128) + (i_7 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((int)threadIdx.x) >> 2)) + 8)];
        } else {
          condval_48 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        b_4[i_7] = nr_tl_shallow::pack(condval_47, condval_48);
      }
      #pragma unroll
      for (int m_13 = 0; m_13 < 2; ++m_13) {
        nr_tl_shallow::mma16((&(z_1[(m_13 * 2)])), out[((m_13 * 8) + (kp * 4))], out[(((m_13 * 8) + (kp * 4)) + 1)], out[(((m_13 * 8) + (kp * 4)) + 2)], out[(((m_13 * 8) + (kp * 4)) + 3)], b_4[0], b_4[1]);
      }
    }
    #pragma unroll
    for (int j_14 = 0; j_14 < 4; ++j_14) {
      uint rg = nr_tl_shallow::shfl(z_1[j_14], ((((int)threadIdx.x) & 7) * 4));
      uint ba = nr_tl_shallow::shfl(z_1[j_14], (((((int)threadIdx.x) & 7) * 4) + 1));
      if ((((int)threadIdx.x) >> 3) == j_14) {
        q[0] = rg;
        q[1] = ba;
      }
    }
    int condval_49;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_49 = 0;
    } else {
      int condval_50;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_50 = 3;
      } else {
        condval_50 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_49 = condval_50;
    }
    int condval_51;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_51 = 0;
    } else {
      int condval_52;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_52 = 3;
      } else {
        condval_52 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_51 = condval_52;
    }
    int condval_53;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_53 = 0;
    } else {
      int condval_54;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_54 = 3;
      } else {
        condval_54 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_53 = condval_54;
    }
    int condval_55;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_55 = 0;
    } else {
      int condval_56;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_56 = 3;
      } else {
        condval_56 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_55 = condval_56;
    }
    int condval_57;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_57 = 0;
    } else {
      int condval_58;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_58 = 3;
      } else {
        condval_58 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_57 = condval_58;
    }
    int condval_59;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_59 = 0;
    } else {
      int condval_60;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_60 = 3;
      } else {
        condval_60 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_59 = condval_60;
    }
    int condval_61;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_61 = 0;
    } else {
      int condval_62;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_62 = 3;
      } else {
        condval_62 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_61 = condval_62;
    }
    int condval_63;
    if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
      condval_63 = 0;
    } else {
      int condval_64;
      if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
        condval_64 = 3;
      } else {
        condval_64 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
      }
      condval_63 = condval_64;
    }
    if ((((1 <= ((((int)blockIdx.x) * 2) + ((int)((condval_49 == 2) | (condval_51 == 3))))) & ((((((int)blockIdx.x) * 8) + (((int)((condval_53 == 2) | (condval_55 == 3))) * 4)) + ((((int)threadIdx.x) & 15) & 3)) < (W0 + 4))) & (1 <= ((((int)blockIdx.y) * 2) + ((int)((condval_57 == 1) | (condval_59 == 2)))))) & ((((((int)blockIdx.y) * 8) + (((int)((condval_61 == 1) | (condval_63 == 2))) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) < (H0 + 4))) {
      #pragma unroll
      for (int j_15 = 0; j_15 < 4; ++j_15) {
        float v = ((float)((half_t)nr_tl_shallow::unpack(q[(j_15 >> 1)], (j_15 & 1))));
        if (((fabsf(v) == CUDART_INF_F) && !(v != v)) | (v != v)) {
          nr_tl_shallow::status_or((&(Status[0])), 4);
        }
        int condval_65;
        if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
          condval_65 = 0;
        } else {
          int condval_66;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
            condval_66 = 3;
          } else {
            condval_66 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
          }
          condval_65 = condval_66;
        }
        int condval_67;
        if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
          condval_67 = 0;
        } else {
          int condval_68;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
            condval_68 = 3;
          } else {
            condval_68 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
          }
          condval_67 = condval_68;
        }
        int condval_69;
        if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
          condval_69 = 0;
        } else {
          int condval_70;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
            condval_70 = 3;
          } else {
            condval_70 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
          }
          condval_69 = condval_70;
        }
        int condval_71;
        if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
          condval_71 = 0;
        } else {
          int condval_72;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
            condval_72 = 3;
          } else {
            condval_72 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
          }
          condval_71 = condval_72;
        }
        if (1 <= (((((int)blockIdx.x) * 2) + (((((((((int)blockIdx.y) * 8) + (((int)((condval_65 == 1) | (condval_67 == 2))) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) - 4) * W0) + ((((int)threadIdx.x) & 15) & 3)) >> 2)) + ((int)((condval_69 == 2) | (condval_71 == 3))))) {
          int condval_73;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
            condval_73 = 0;
          } else {
            int condval_74;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
              condval_74 = 3;
            } else {
              condval_74 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
            }
            condval_73 = condval_74;
          }
          int condval_75;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
            condval_75 = 0;
          } else {
            int condval_76;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
              condval_76 = 3;
            } else {
              condval_76 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
            }
            condval_75 = condval_76;
          }
          int condval_77;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
            condval_77 = 0;
          } else {
            int condval_78;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
              condval_78 = 3;
            } else {
              condval_78 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
            }
            condval_77 = condval_78;
          }
          int condval_79;
          if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
            condval_79 = 0;
          } else {
            int condval_80;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
              condval_80 = 3;
            } else {
              condval_80 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
            }
            condval_79 = condval_80;
          }
          if ((((((((int)blockIdx.x) * 32) + (((int)((condval_73 == 2) | (condval_75 == 3))) * 16)) + ((((((((int)blockIdx.y) * 8) + (((int)((condval_77 == 1) | (condval_79 == 2))) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) - 4) * W0) * 4)) + (((((int)threadIdx.x) & 15) & 3) * 4)) + j_15) < (sizes_15 + 16)) {
            int condval_81;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_81 = 0;
            } else {
              int condval_82;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_82 = 3;
              } else {
                condval_82 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_81 = condval_82;
            }
            int condval_83;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_83 = 0;
            } else {
              int condval_84;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_84 = 3;
              } else {
                condval_84 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_83 = condval_84;
            }
            int condval_85;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_85 = 0;
            } else {
              int condval_86;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_86 = 3;
              } else {
                condval_86 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_85 = condval_86;
            }
            int condval_87;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_87 = 0;
            } else {
              int condval_88;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_88 = 3;
              } else {
                condval_88 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_87 = condval_88;
            }
            int condval_89;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_89 = 0;
            } else {
              int condval_90;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_90 = 3;
              } else {
                condval_90 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_89 = condval_90;
            }
            int condval_91;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_91 = 0;
            } else {
              int condval_92;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_92 = 3;
              } else {
                condval_92 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_91 = condval_92;
            }
            int condval_93;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_93 = 0;
            } else {
              int condval_94;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_94 = 3;
              } else {
                condval_94 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_93 = condval_94;
            }
            int condval_95;
            if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 0)) {
              condval_95 = 0;
            } else {
              int condval_96;
              if ((((slab * 2) + (((int)threadIdx.x) >> 4)) == 1)) {
                condval_96 = 3;
              } else {
                condval_96 = (((slab * 2) + (((int)threadIdx.x) >> 4)) - 1);
              }
              condval_95 = condval_96;
            }
            Head[((((((((int64_t)((int)blockIdx.x)) * (int64_t)32) + (((int64_t)((condval_89 == 2) | (condval_91 == 3))) * (int64_t)16)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)8) + (((int64_t)((condval_93 == 1) | (condval_95 == 2))) * (int64_t)4)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) >> (int64_t)2)) - (int64_t)4) * ((int64_t)W0)) * (int64_t)4)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)15) & (int64_t)3) * (int64_t)4)) + ((int64_t)j_15)) - (int64_t)16)] = v;
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

