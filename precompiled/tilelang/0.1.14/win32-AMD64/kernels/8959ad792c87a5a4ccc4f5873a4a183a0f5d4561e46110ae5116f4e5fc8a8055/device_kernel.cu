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

extern "C" __global__ void main_kernel(const half_t* __restrict__ Adapter, const float* __restrict__ Features, const half_t* __restrict__ G0, const half_t* __restrict__ G1, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const half_t* __restrict__ Adapter, const float* __restrict__ Features, const half_t* __restrict__ G0, const half_t* __restrict__ G1, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* packet = ((void*)((char*)buf_dyn_shmem + 0));
  void* rail = ((void*)((char*)buf_dyn_shmem + 2048));
  uint raw[32];
  uint pool_a[4];
  uint aa[16];
  uint bb[2];
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
  uint mp[4];
  #pragma unroll
  for (int j = 0; j < 32; ++j) {
    raw[j] = (uint)0;
  }
  #pragma unroll
  for (int j_1 = 0; j_1 < 4; ++j_1) {
    pool_a[j_1] = (uint)0;
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
      float v = Features[((((((c * 327680) + (((int)blockIdx.y) * 5120)) + (i * 2560)) + ((((int)threadIdx.x) >> 3) * 640)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) & 7))];
      half_t hv = ((half_t)v);
      if ((((fabsf(v) == CUDART_INF_F) && !(v != v)) | (v != v)) | ((fabsf(((float)((half_t)v))) == CUDART_INF_F) && !(((float)((half_t)v)) != ((float)((half_t)v))))) {
        nr_tl_shallow::status_or((&(Status[0])), 1);
      }
      ((half_t*)packet)[(((i * 512) + (((int)threadIdx.x) * 16)) + c)] = ((half_t)v);
    }
  }
  nr_tl_shallow::sync();
  __syncthreads();
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      aa[i_1] = nr_tl_shallow::pack(((half_t*)packet)[(((((((((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 3) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 12) << 1)) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 16) << 1)) ^ ((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 32)) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 32) >> 3)) * 16) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))], ((half_t*)packet)[((((((((((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 3) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 12) << 1)) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 16) << 1)) ^ ((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 32)) ^ (((((m * 16) + ((i_1 & 1) * 8)) + (((int)threadIdx.x) >> 2)) & 32) >> 3)) * 16) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)]);
    }
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        bb[i_2] = nr_tl_shallow::pack(Adapter[(((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2))], Adapter[((((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2)) + 32)]);
      }
      nr_tl_shallow::mma16((&(raw[((m * 8) + (n * 2))])), aa[0], aa[1], aa[2], aa[3], bb[0], bb[1]);
    }
  }
  #pragma unroll
  for (int m_1 = 0; m_1 < 4; ++m_1) {
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        aa[(((m_1 * 4) + (p * 2)) + i_3)] = (nr_tl_shallow::e4pair(raw[(((m_1 * 8) + (p * 4)) + i_3)]) | (nr_tl_shallow::e4pair(raw[((((m_1 * 8) + (p * 4)) + i_3) + 2)]) << (uint)16));
      }
    }
  }
  #pragma unroll
  for (int j_2 = 0; j_2 < 32; ++j_2) {
    ff[j_2] = nr_tl_shallow::mul(raw[j_2], nr_tl_shallow::pack(G0[((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))], G0[(((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
  }
  ushort v_ = (ushort)14540;
  half_t scale = (*(half_t *)(&(v_)));
  for (int part = 0; part < 4; ++part) {
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      #pragma unroll
      for (int j_3 = 0; j_3 < 16; ++j_3) {
        hidden[j_3] = (uint)0;
      }
      nr_tl_shallow_joint::ld128((&(b[0])), (&(Wt0[(((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_2 = 0; m_2 < 4; ++m_2) {
        #pragma unroll
        for (int n_1 = 0; n_1 < 2; ++n_1) {
          nr_tl_shallow::mma8((&(hidden[((m_2 * 4) + (n_1 * 2))])), aa[(m_2 * 4)], aa[((m_2 * 4) + 1)], aa[((m_2 * 4) + 2)], aa[((m_2 * 4) + 3)], b[(n_1 * 2)], b[((n_1 * 2) + 1)]);
        }
      }
      #pragma unroll
      for (int m_3 = 0; m_3 < 4; ++m_3) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          ha_[(((m_3 * 4) + (pair * 2)) + i_4)] = (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[((m_3 * 4) + i_4)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_3 * 4) + i_4)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[((m_3 * 4) + i_4)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) | (nr_tl_shallow::e4pair(nr_tl_shallow::mul(hidden[(((m_3 * 4) + i_4) + 2)], nr_tl_shallow::fma(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_3 * 4) + i_4) + 2)])), nr_tl_shallow::fma(nr_tl_shallow::abs(nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1p+2f/*4.000000e+00*/), half_t(0x1p+2f/*4.000000e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(-0x1p+2f/*-4.000000e+00*/), half_t(-0x1p+2f/*-4.000000e+00*/)), hidden[(((m_3 * 4) + i_4) + 2)]))), nr_tl_shallow::pack(half_t(-0x1.cap-5f/*-5.590820e-02*/), half_t(-0x1.cap-5f/*-5.590820e-02*/)), nr_tl_shallow::pack(half_t(0x1.cap-2f/*4.472656e-01*/), half_t(0x1.cap-2f/*4.472656e-01*/))), nr_tl_shallow::pack(half_t(0x1.cap-1f/*8.945312e-01*/), half_t(0x1.cap-1f/*8.945312e-01*/))))) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; ++p_1) {
      nr_tl_shallow_joint::ld128((&(b_1[0])), (&(Wt1[(((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_4 = 0; m_4 < 4; ++m_4) {
        #pragma unroll
        for (int n_2 = 0; n_2 < 2; ++n_2) {
          nr_tl_shallow::mma8((&(ff[(((m_4 * 8) + (p_1 * 4)) + (n_2 * 2))])), ha_[(m_4 * 4)], ha_[((m_4 * 4) + 1)], ha_[((m_4 * 4) + 2)], ha_[((m_4 * 4) + 3)], b_1[(n_2 * 2)], b_1[((n_2 * 2) + 1)]);
        }
      }
    }
  }
  #pragma unroll
  for (int m_5 = 0; m_5 < 4; ++m_5) {
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; ++p_2) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        aa[(((m_5 * 4) + (p_2 * 2)) + i_5)] = (nr_tl_shallow::e4pair(ff[(((m_5 * 8) + (p_2 * 4)) + i_5)]) | (nr_tl_shallow::e4pair(ff[((((m_5 * 8) + (p_2 * 4)) + i_5) + 2)]) << (uint)16));
      }
    }
  }
  #pragma unroll
  for (int component = 0; component < 3; ++component) {
    #pragma unroll
    for (int j_4 = 0; j_4 < 32; ++j_4) {
      z[j_4] = (uint)0;
    }
    #pragma unroll
    for (int p_3 = 0; p_3 < 2; ++p_3) {
      nr_tl_shallow_joint::ld128((&(b_2[0])), (&(Wq[(((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_6 = 0; m_6 < 4; ++m_6) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 2; ++n_3) {
          nr_tl_shallow::mma8((&(z[(((m_6 * 8) + (p_3 * 4)) + (n_3 * 2))])), aa[(m_6 * 4)], aa[((m_6 * 4) + 1)], aa[((m_6 * 4) + 2)], aa[((m_6 * 4) + 3)], b_2[(n_3 * 2)], b_2[((n_3 * 2) + 1)]);
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
      for (int j_5 = 0; j_5 < 32; ++j_5) {
        z[j_5] = nr_tl_shallow::mul(z[j_5], inv[(((j_5 >> 3) * 2) + (j_5 & 1))]);
        if (component == 0) {
          ushort v__1 = (ushort)14540;
          z[j_5] = nr_tl_shallow::mul(z[j_5], nr_tl_shallow::pack((*(half_t *)(&(v__1))), (*(half_t *)(&(v__1)))));
        }
      }
    }
    #pragma unroll
    for (int m_7 = 0; m_7 < 4; ++m_7) {
      #pragma unroll
      for (int p_4 = 0; p_4 < 2; ++p_4) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          if (component == 0) {
            int64_t condval;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval = (int64_t)0;
            } else {
              int64_t condval_1;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_1 = (int64_t)3;
              } else {
                condval_1 = (((int64_t)m_7) - (int64_t)1);
              }
              condval = condval_1;
            }
            int64_t condval_2;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_2 = (int64_t)0;
            } else {
              int64_t condval_3;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_3 = (int64_t)3;
              } else {
                condval_3 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_2 = condval_3;
            }
            int64_t condval_4;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_4 = (int64_t)0;
            } else {
              int64_t condval_5;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_5 = (int64_t)3;
              } else {
                condval_5 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_4 = condval_5;
            }
            int64_t condval_6;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_6 = (int64_t)0;
            } else {
              int64_t condval_7;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_7 = (int64_t)3;
              } else {
                condval_7 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_6 = condval_7;
            }
            qa[(((m_7 * 4) + (p_4 * 2)) + i_6)] = (nr_tl_shallow::e4pair(z[(((condval_2 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6))]) | (nr_tl_shallow::e4pair(z[((((condval_6 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_8;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_8 = (int64_t)0;
            } else {
              int64_t condval_9;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_9 = (int64_t)3;
              } else {
                condval_9 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_8 = condval_9;
            }
            int64_t condval_10;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_10 = (int64_t)0;
            } else {
              int64_t condval_11;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_11 = (int64_t)3;
              } else {
                condval_11 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_10 = condval_11;
            }
            int64_t condval_12;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_12 = (int64_t)0;
            } else {
              int64_t condval_13;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_13 = (int64_t)3;
              } else {
                condval_13 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_12 = condval_13;
            }
            int64_t condval_14;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_14 = (int64_t)0;
            } else {
              int64_t condval_15;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_15 = (int64_t)3;
              } else {
                condval_15 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_14 = condval_15;
            }
            kb[(((m_7 * 4) + (i_6 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_10 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6))]) | (nr_tl_shallow::e4pair(z[((((condval_14 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_4 = 0; n_4 < 4; ++n_4) {
          int64_t condval_16;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_16 = (int64_t)0;
          } else {
            condval_16 = (int64_t)1;
          }
          int64_t condval_17;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_17 = (int64_t)0;
          } else {
            condval_17 = (int64_t)1;
          }
          int64_t condval_18;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_18 = (int64_t)0;
          } else {
            condval_18 = (int64_t)1;
          }
          int64_t condval_19;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_19 = (int64_t)0;
          } else {
            condval_19 = (int64_t)1;
          }
          vb[((part_1 * 8) + (n_4 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_17 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_19 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_20;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_20 = (int64_t)3;
          } else {
            condval_20 = (int64_t)2;
          }
          int64_t condval_21;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_21 = (int64_t)3;
          } else {
            condval_21 = (int64_t)2;
          }
          int64_t condval_22;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_22 = (int64_t)3;
          } else {
            condval_22 = (int64_t)2;
          }
          int64_t condval_23;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_23 = (int64_t)3;
          } else {
            condval_23 = (int64_t)2;
          }
          vb[(((part_1 * 8) + (n_4 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_21 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_23 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
        }
      }
    }
  }
  #pragma unroll
  for (int slab = 0; slab < 2; ++slab) {
    #pragma unroll
    for (int m_8 = 0; m_8 < 2; ++m_8) {
      #pragma unroll
      for (int p_5 = 0; p_5 < 4; ++p_5) {
        nr_tl_shallow_joint::ld128((&(seed[0])), (&(Wq[(((((slab * 1024) + (m_8 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768)])));
        #pragma unroll
        for (int n_5 = 0; n_5 < 2; ++n_5) {
          logits[(((m_8 * 16) + (p_5 * 4)) + (n_5 * 2))] = seed[(n_5 * 2)];
          logits[((((m_8 * 16) + (p_5 * 4)) + (n_5 * 2)) + 1)] = seed[((n_5 * 2) + 1)];
          nr_tl_shallow::mma8((&(logits[(((m_8 * 16) + (p_5 * 4)) + (n_5 * 2))])), qa[((slab * 8) + (m_8 * 4))], qa[(((slab * 8) + (m_8 * 4)) + 1)], qa[(((slab * 8) + (m_8 * 4)) + 2)], qa[(((slab * 8) + (m_8 * 4)) + 3)], kb[((p_5 * 4) + (n_5 * 2))], kb[(((p_5 * 4) + (n_5 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int j_6 = 0; j_6 < 32; ++j_6) {
      logits[j_6] = (((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_6], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) & (uint)65535) << (uint)5) + (uint)32768) & (uint)65535) | ((((nr_tl_shallow::min(nr_tl_shallow::pack(half_t(0x1.91cp+0f/*1.569336e+00*/), half_t(0x1.91cp+0f/*1.569336e+00*/)), nr_tl_shallow::max(nr_tl_shallow::pack(half_t(0x1.08p+0f/*1.031250e+00*/), half_t(0x1.08p+0f/*1.031250e+00*/)), nr_tl_shallow::fma(logits[j_6], nr_tl_shallow::pack(half_t(0x1.7p-5f/*4.492188e-02*/), half_t(0x1.7p-5f/*4.492188e-02*/)), nr_tl_shallow::pack(half_t(0x1.4dp+0f/*1.300781e+00*/), half_t(0x1.4dp+0f/*1.300781e+00*/))))) >> (uint)16) << (uint)5) + (uint)32768) << (uint)16));
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
    for (int j_7 = 0; j_7 < 32; ++j_7) {
      logits[j_7] = nr_tl_shallow::mul(logits[j_7], inv[(((j_7 >> 4) * 2) + (j_7 & 1))]);
    }
    #pragma unroll
    for (int part_2 = 0; part_2 < 2; ++part_2) {
      #pragma unroll
      for (int m_9 = 0; m_9 < 2; ++m_9) {
        #pragma unroll
        for (int p_6 = 0; p_6 < 2; ++p_6) {
          #pragma unroll
          for (int i_7 = 0; i_7 < 2; ++i_7) {
            pa[((((part_2 * 8) + (m_9 * 4)) + (p_6 * 2)) + i_7)] = (nr_tl_shallow::e4pair(logits[((((m_9 * 16) + (part_2 * 8)) + (p_6 * 4)) + i_7)]) | (nr_tl_shallow::e4pair(logits[(((((m_9 * 16) + (part_2 * 8)) + (p_6 * 4)) + i_7) + 2)]) << (uint)16));
          }
        }
      }
    }
    #pragma unroll
    for (int j_8 = 0; j_8 < 16; ++j_8) {
      attended[j_8] = (uint)0;
    }
    #pragma unroll
    for (int part_3 = 0; part_3 < 2; ++part_3) {
      #pragma unroll
      for (int n_6 = 0; n_6 < 4; ++n_6) {
        #pragma unroll
        for (int m_10 = 0; m_10 < 2; ++m_10) {
          nr_tl_shallow::mma8((&(attended[((m_10 * 8) + (n_6 * 2))])), pa[((part_3 * 8) + (m_10 * 4))], pa[(((part_3 * 8) + (m_10 * 4)) + 1)], pa[(((part_3 * 8) + (m_10 * 4)) + 2)], pa[(((part_3 * 8) + (m_10 * 4)) + 3)], vb[((part_3 * 8) + (n_6 * 2))], vb[(((part_3 * 8) + (n_6 * 2)) + 1)]);
        }
      }
    }
    #pragma unroll
    for (int m_11 = 0; m_11 < 2; ++m_11) {
      #pragma unroll
      for (int p_7 = 0; p_7 < 2; ++p_7) {
        #pragma unroll
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          aa[(((m_11 * 4) + (p_7 * 2)) + i_8)] = (nr_tl_shallow::e4pair(attended[(((m_11 * 8) + (p_7 * 4)) + i_8)]) | (nr_tl_shallow::e4pair(attended[((((m_11 * 8) + (p_7 * 4)) + i_8) + 2)]) << (uint)16));
        }
      }
    }
    #pragma unroll
    for (int j_9 = 0; j_9 < 16; ++j_9) {
      int64_t condval_24;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)0)) {
        condval_24 = (int64_t)0;
      } else {
        int64_t condval_25;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)1)) {
          condval_25 = (int64_t)3;
        } else {
          condval_25 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) - (int64_t)1);
        }
        condval_24 = condval_25;
      }
      int64_t condval_26;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)0)) {
        condval_26 = (int64_t)0;
      } else {
        int64_t condval_27;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)1)) {
          condval_27 = (int64_t)3;
        } else {
          condval_27 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) - (int64_t)1);
        }
        condval_26 = condval_27;
      }
      out[j_9] = nr_tl_shallow::mul(ff[((condval_26 * (int64_t)8) + (((int64_t)j_9) & (int64_t)7))], nr_tl_shallow::pack(G1[((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))], G1[(((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)]));
    }
    #pragma unroll
    for (int p_8 = 0; p_8 < 2; ++p_8) {
      nr_tl_shallow_joint::ld128((&(b_3[0])), (&(Wp[((p_8 * 128) + (((int)threadIdx.x) * 4))])));
      #pragma unroll
      for (int m_12 = 0; m_12 < 2; ++m_12) {
        #pragma unroll
        for (int n_7 = 0; n_7 < 2; ++n_7) {
          nr_tl_shallow::mma8((&(out[(((m_12 * 8) + (p_8 * 4)) + (n_7 * 2))])), aa[(m_12 * 4)], aa[((m_12 * 4) + 1)], aa[((m_12 * 4) + 2)], aa[((m_12 * 4) + 3)], b_3[(n_7 * 2)], b_3[((n_7 * 2) + 1)]);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int m_13 = 0; m_13 < 2; ++m_13) {
      int condval_28;
      if ((((slab * 2) + m_13) == 0)) {
        condval_28 = 0;
      } else {
        int condval_29;
        if ((((slab * 2) + m_13) == 1)) {
          condval_29 = 3;
        } else {
          condval_29 = (((slab * 2) + m_13) - 1);
        }
        condval_28 = condval_29;
      }
      int condval_30;
      if ((((slab * 2) + m_13) == 0)) {
        condval_30 = 0;
      } else {
        int condval_31;
        if ((((slab * 2) + m_13) == 1)) {
          condval_31 = 3;
        } else {
          condval_31 = (((slab * 2) + m_13) - 1);
        }
        condval_30 = condval_31;
      }
      int condval_32;
      if ((((slab * 2) + m_13) == 0)) {
        condval_32 = 0;
      } else {
        int condval_33;
        if ((((slab * 2) + m_13) == 1)) {
          condval_33 = 3;
        } else {
          condval_33 = (((slab * 2) + m_13) - 1);
        }
        condval_32 = condval_33;
      }
      int condval_34;
      if ((((slab * 2) + m_13) == 0)) {
        condval_34 = 0;
      } else {
        int condval_35;
        if ((((slab * 2) + m_13) == 1)) {
          condval_35 = 3;
        } else {
          condval_35 = (((slab * 2) + m_13) - 1);
        }
        condval_34 = condval_35;
      }
      int condval_36;
      if ((((slab * 2) + m_13) == 0)) {
        condval_36 = 0;
      } else {
        int condval_37;
        if ((((slab * 2) + m_13) == 1)) {
          condval_37 = 3;
        } else {
          condval_37 = (((slab * 2) + m_13) - 1);
        }
        condval_36 = condval_37;
      }
      int condval_38;
      if ((((slab * 2) + m_13) == 0)) {
        condval_38 = 0;
      } else {
        int condval_39;
        if ((((slab * 2) + m_13) == 1)) {
          condval_39 = 3;
        } else {
          condval_39 = (((slab * 2) + m_13) - 1);
        }
        condval_38 = condval_39;
      }
      int condval_40;
      if ((((slab * 2) + m_13) == 0)) {
        condval_40 = 0;
      } else {
        int condval_41;
        if ((((slab * 2) + m_13) == 1)) {
          condval_41 = 3;
        } else {
          condval_41 = (((slab * 2) + m_13) - 1);
        }
        condval_40 = condval_41;
      }
      int condval_42;
      if ((((slab * 2) + m_13) == 0)) {
        condval_42 = 0;
      } else {
        int condval_43;
        if ((((slab * 2) + m_13) == 1)) {
          condval_43 = 3;
        } else {
          condval_43 = (((slab * 2) + m_13) - 1);
        }
        condval_42 = condval_43;
      }
      int condval_44;
      if ((((slab * 2) + m_13) == 0)) {
        condval_44 = 0;
      } else {
        int condval_45;
        if ((((slab * 2) + m_13) == 1)) {
          condval_45 = 3;
        } else {
          condval_45 = (((slab * 2) + m_13) - 1);
        }
        condval_44 = condval_45;
      }
      int condval_46;
      if ((((slab * 2) + m_13) == 0)) {
        condval_46 = 0;
      } else {
        int condval_47;
        if ((((slab * 2) + m_13) == 1)) {
          condval_47 = 3;
        } else {
          condval_47 = (((slab * 2) + m_13) - 1);
        }
        condval_46 = condval_47;
      }
      int condval_48;
      if ((((slab * 2) + m_13) == 0)) {
        condval_48 = 0;
      } else {
        int condval_49;
        if ((((slab * 2) + m_13) == 1)) {
          condval_49 = 3;
        } else {
          condval_49 = (((slab * 2) + m_13) - 1);
        }
        condval_48 = condval_49;
      }
      int condval_50;
      if ((((slab * 2) + m_13) == 0)) {
        condval_50 = 0;
      } else {
        int condval_51;
        if ((((slab * 2) + m_13) == 1)) {
          condval_51 = 3;
        } else {
          condval_51 = (((slab * 2) + m_13) - 1);
        }
        condval_50 = condval_51;
      }
      int condval_52;
      if ((((slab * 2) + m_13) == 0)) {
        condval_52 = 0;
      } else {
        int condval_53;
        if ((((slab * 2) + m_13) == 1)) {
          condval_53 = 3;
        } else {
          condval_53 = (((slab * 2) + m_13) - 1);
        }
        condval_52 = condval_53;
      }
      int condval_54;
      if ((((slab * 2) + m_13) == 0)) {
        condval_54 = 0;
      } else {
        int condval_55;
        if ((((slab * 2) + m_13) == 1)) {
          condval_55 = 3;
        } else {
          condval_55 = (((slab * 2) + m_13) - 1);
        }
        condval_54 = condval_55;
      }
      int condval_56;
      if ((((slab * 2) + m_13) == 0)) {
        condval_56 = 0;
      } else {
        int condval_57;
        if ((((slab * 2) + m_13) == 1)) {
          condval_57 = 3;
        } else {
          condval_57 = (((slab * 2) + m_13) - 1);
        }
        condval_56 = condval_57;
      }
      int condval_58;
      if ((((slab * 2) + m_13) == 0)) {
        condval_58 = 0;
      } else {
        int condval_59;
        if ((((slab * 2) + m_13) == 1)) {
          condval_59 = 3;
        } else {
          condval_59 = (((slab * 2) + m_13) - 1);
        }
        condval_58 = condval_59;
      }
      int condval_60;
      if ((((slab * 2) + m_13) == 0)) {
        condval_60 = 0;
      } else {
        int condval_61;
        if ((((slab * 2) + m_13) == 1)) {
          condval_61 = 3;
        } else {
          condval_61 = (((slab * 2) + m_13) - 1);
        }
        condval_60 = condval_61;
      }
      int condval_62;
      if ((((slab * 2) + m_13) == 0)) {
        condval_62 = 0;
      } else {
        int condval_63;
        if ((((slab * 2) + m_13) == 1)) {
          condval_63 = 3;
        } else {
          condval_63 = (((slab * 2) + m_13) - 1);
        }
        condval_62 = condval_63;
      }
      int condval_64;
      if ((((slab * 2) + m_13) == 0)) {
        condval_64 = 0;
      } else {
        int condval_65;
        if ((((slab * 2) + m_13) == 1)) {
          condval_65 = 3;
        } else {
          condval_65 = (((slab * 2) + m_13) - 1);
        }
        condval_64 = condval_65;
      }
      int condval_66;
      if ((((slab * 2) + m_13) == 0)) {
        condval_66 = 0;
      } else {
        int condval_67;
        if ((((slab * 2) + m_13) == 1)) {
          condval_67 = 3;
        } else {
          condval_67 = (((slab * 2) + m_13) - 1);
        }
        condval_66 = condval_67;
      }
      int condval_68;
      if ((((slab * 2) + m_13) == 0)) {
        condval_68 = 0;
      } else {
        int condval_69;
        if ((((slab * 2) + m_13) == 1)) {
          condval_69 = 3;
        } else {
          condval_69 = (((slab * 2) + m_13) - 1);
        }
        condval_68 = condval_69;
      }
      int condval_70;
      if ((((slab * 2) + m_13) == 0)) {
        condval_70 = 0;
      } else {
        int condval_71;
        if ((((slab * 2) + m_13) == 1)) {
          condval_71 = 3;
        } else {
          condval_71 = (((slab * 2) + m_13) - 1);
        }
        condval_70 = condval_71;
      }
      int condval_72;
      if ((((slab * 2) + m_13) == 0)) {
        condval_72 = 0;
      } else {
        int condval_73;
        if ((((slab * 2) + m_13) == 1)) {
          condval_73 = 3;
        } else {
          condval_73 = (((slab * 2) + m_13) - 1);
        }
        condval_72 = condval_73;
      }
      int condval_74;
      if ((((slab * 2) + m_13) == 0)) {
        condval_74 = 0;
      } else {
        int condval_75;
        if ((((slab * 2) + m_13) == 1)) {
          condval_75 = 3;
        } else {
          condval_75 = (((slab * 2) + m_13) - 1);
        }
        condval_74 = condval_75;
      }
      int condval_76;
      if ((((slab * 2) + m_13) == 0)) {
        condval_76 = 0;
      } else {
        int condval_77;
        if ((((slab * 2) + m_13) == 1)) {
          condval_77 = 3;
        } else {
          condval_77 = (((slab * 2) + m_13) - 1);
        }
        condval_76 = condval_77;
      }
      int condval_78;
      if ((((slab * 2) + m_13) == 0)) {
        condval_78 = 0;
      } else {
        int condval_79;
        if ((((slab * 2) + m_13) == 1)) {
          condval_79 = 3;
        } else {
          condval_79 = (((slab * 2) + m_13) - 1);
        }
        condval_78 = condval_79;
      }
      int condval_80;
      if ((((slab * 2) + m_13) == 0)) {
        condval_80 = 0;
      } else {
        int condval_81;
        if ((((slab * 2) + m_13) == 1)) {
          condval_81 = 3;
        } else {
          condval_81 = (((slab * 2) + m_13) - 1);
        }
        condval_80 = condval_81;
      }
      int condval_82;
      if ((((slab * 2) + m_13) == 0)) {
        condval_82 = 0;
      } else {
        int condval_83;
        if ((((slab * 2) + m_13) == 1)) {
          condval_83 = 3;
        } else {
          condval_83 = (((slab * 2) + m_13) - 1);
        }
        condval_82 = condval_83;
      }
      int condval_84;
      if ((((slab * 2) + m_13) == 0)) {
        condval_84 = 0;
      } else {
        int condval_85;
        if ((((slab * 2) + m_13) == 1)) {
          condval_85 = 3;
        } else {
          condval_85 = (((slab * 2) + m_13) - 1);
        }
        condval_84 = condval_85;
      }
      int condval_86;
      if ((((slab * 2) + m_13) == 0)) {
        condval_86 = 0;
      } else {
        int condval_87;
        if ((((slab * 2) + m_13) == 1)) {
          condval_87 = 3;
        } else {
          condval_87 = (((slab * 2) + m_13) - 1);
        }
        condval_86 = condval_87;
      }
      int condval_88;
      if ((((slab * 2) + m_13) == 0)) {
        condval_88 = 0;
      } else {
        int condval_89;
        if ((((slab * 2) + m_13) == 1)) {
          condval_89 = 3;
        } else {
          condval_89 = (((slab * 2) + m_13) - 1);
        }
        condval_88 = condval_89;
      }
      int condval_90;
      if ((((slab * 2) + m_13) == 0)) {
        condval_90 = 0;
      } else {
        int condval_91;
        if ((((slab * 2) + m_13) == 1)) {
          condval_91 = 3;
        } else {
          condval_91 = (((slab * 2) + m_13) - 1);
        }
        condval_90 = condval_91;
      }
      int condval_92;
      if ((((slab * 2) + m_13) == 0)) {
        condval_92 = 0;
      } else {
        int condval_93;
        if ((((slab * 2) + m_13) == 1)) {
          condval_93 = 3;
        } else {
          condval_93 = (((slab * 2) + m_13) - 1);
        }
        condval_92 = condval_93;
      }
      int condval_94;
      if ((((slab * 2) + m_13) == 0)) {
        condval_94 = 0;
      } else {
        int condval_95;
        if ((((slab * 2) + m_13) == 1)) {
          condval_95 = 3;
        } else {
          condval_95 = (((slab * 2) + m_13) - 1);
        }
        condval_94 = condval_95;
      }
      int condval_96;
      if ((((slab * 2) + m_13) == 0)) {
        condval_96 = 0;
      } else {
        int condval_97;
        if ((((slab * 2) + m_13) == 1)) {
          condval_97 = 3;
        } else {
          condval_97 = (((slab * 2) + m_13) - 1);
        }
        condval_96 = condval_97;
      }
      int condval_98;
      if ((((slab * 2) + m_13) == 0)) {
        condval_98 = 0;
      } else {
        int condval_99;
        if ((((slab * 2) + m_13) == 1)) {
          condval_99 = 3;
        } else {
          condval_99 = (((slab * 2) + m_13) - 1);
        }
        condval_98 = condval_99;
      }
      int condval_100;
      if ((((slab * 2) + m_13) == 0)) {
        condval_100 = 0;
      } else {
        int condval_101;
        if ((((slab * 2) + m_13) == 1)) {
          condval_101 = 3;
        } else {
          condval_101 = (((slab * 2) + m_13) - 1);
        }
        condval_100 = condval_101;
      }
      int condval_102;
      if ((((slab * 2) + m_13) == 0)) {
        condval_102 = 0;
      } else {
        int condval_103;
        if ((((slab * 2) + m_13) == 1)) {
          condval_103 = 3;
        } else {
          condval_103 = (((slab * 2) + m_13) - 1);
        }
        condval_102 = condval_103;
      }
      int condval_104;
      if ((((slab * 2) + m_13) == 0)) {
        condval_104 = 0;
      } else {
        int condval_105;
        if ((((slab * 2) + m_13) == 1)) {
          condval_105 = 3;
        } else {
          condval_105 = (((slab * 2) + m_13) - 1);
        }
        condval_104 = condval_105;
      }
      int condval_106;
      if ((((slab * 2) + m_13) == 0)) {
        condval_106 = 0;
      } else {
        int condval_107;
        if ((((slab * 2) + m_13) == 1)) {
          condval_107 = 3;
        } else {
          condval_107 = (((slab * 2) + m_13) - 1);
        }
        condval_106 = condval_107;
      }
      int condval_108;
      if ((((slab * 2) + m_13) == 0)) {
        condval_108 = 0;
      } else {
        int condval_109;
        if ((((slab * 2) + m_13) == 1)) {
          condval_109 = 3;
        } else {
          condval_109 = (((slab * 2) + m_13) - 1);
        }
        condval_108 = condval_109;
      }
      int condval_110;
      if ((((slab * 2) + m_13) == 0)) {
        condval_110 = 0;
      } else {
        int condval_111;
        if ((((slab * 2) + m_13) == 1)) {
          condval_111 = 3;
        } else {
          condval_111 = (((slab * 2) + m_13) - 1);
        }
        condval_110 = condval_111;
      }
      int condval_112;
      if ((((slab * 2) + m_13) == 0)) {
        condval_112 = 0;
      } else {
        int condval_113;
        if ((((slab * 2) + m_13) == 1)) {
          condval_113 = 3;
        } else {
          condval_113 = (((slab * 2) + m_13) - 1);
        }
        condval_112 = condval_113;
      }
      int condval_114;
      if ((((slab * 2) + m_13) == 0)) {
        condval_114 = 0;
      } else {
        int condval_115;
        if ((((slab * 2) + m_13) == 1)) {
          condval_115 = 3;
        } else {
          condval_115 = (((slab * 2) + m_13) - 1);
        }
        condval_114 = condval_115;
      }
      int condval_116;
      if ((((slab * 2) + m_13) == 0)) {
        condval_116 = 0;
      } else {
        int condval_117;
        if ((((slab * 2) + m_13) == 1)) {
          condval_117 = 3;
        } else {
          condval_117 = (((slab * 2) + m_13) - 1);
        }
        condval_116 = condval_117;
      }
      int condval_118;
      if ((((slab * 2) + m_13) == 0)) {
        condval_118 = 0;
      } else {
        int condval_119;
        if ((((slab * 2) + m_13) == 1)) {
          condval_119 = 3;
        } else {
          condval_119 = (((slab * 2) + m_13) - 1);
        }
        condval_118 = condval_119;
      }
      int condval_120;
      if ((((slab * 2) + m_13) == 0)) {
        condval_120 = 0;
      } else {
        int condval_121;
        if ((((slab * 2) + m_13) == 1)) {
          condval_121 = 3;
        } else {
          condval_121 = (((slab * 2) + m_13) - 1);
        }
        condval_120 = condval_121;
      }
      int condval_122;
      if ((((slab * 2) + m_13) == 0)) {
        condval_122 = 0;
      } else {
        int condval_123;
        if ((((slab * 2) + m_13) == 1)) {
          condval_123 = 3;
        } else {
          condval_123 = (((slab * 2) + m_13) - 1);
        }
        condval_122 = condval_123;
      }
      int condval_124;
      if ((((slab * 2) + m_13) == 0)) {
        condval_124 = 0;
      } else {
        int condval_125;
        if ((((slab * 2) + m_13) == 1)) {
          condval_125 = 3;
        } else {
          condval_125 = (((slab * 2) + m_13) - 1);
        }
        condval_124 = condval_125;
      }
      int condval_126;
      if ((((slab * 2) + m_13) == 0)) {
        condval_126 = 0;
      } else {
        int condval_127;
        if ((((slab * 2) + m_13) == 1)) {
          condval_127 = 3;
        } else {
          condval_127 = (((slab * 2) + m_13) - 1);
        }
        condval_126 = condval_127;
      }
      int condval_128;
      if ((((slab * 2) + m_13) == 0)) {
        condval_128 = 0;
      } else {
        int condval_129;
        if ((((slab * 2) + m_13) == 1)) {
          condval_129 = 3;
        } else {
          condval_129 = (((slab * 2) + m_13) - 1);
        }
        condval_128 = condval_129;
      }
      int condval_130;
      if ((((slab * 2) + m_13) == 0)) {
        condval_130 = 0;
      } else {
        int condval_131;
        if ((((slab * 2) + m_13) == 1)) {
          condval_131 = 3;
        } else {
          condval_131 = (((slab * 2) + m_13) - 1);
        }
        condval_130 = condval_131;
      }
      int condval_132;
      if ((((slab * 2) + m_13) == 0)) {
        condval_132 = 0;
      } else {
        int condval_133;
        if ((((slab * 2) + m_13) == 1)) {
          condval_133 = 3;
        } else {
          condval_133 = (((slab * 2) + m_13) - 1);
        }
        condval_132 = condval_133;
      }
      int condval_134;
      if ((((slab * 2) + m_13) == 0)) {
        condval_134 = 0;
      } else {
        int condval_135;
        if ((((slab * 2) + m_13) == 1)) {
          condval_135 = 3;
        } else {
          condval_135 = (((slab * 2) + m_13) - 1);
        }
        condval_134 = condval_135;
      }
      int condval_136;
      if ((((slab * 2) + m_13) == 0)) {
        condval_136 = 0;
      } else {
        int condval_137;
        if ((((slab * 2) + m_13) == 1)) {
          condval_137 = 3;
        } else {
          condval_137 = (((slab * 2) + m_13) - 1);
        }
        condval_136 = condval_137;
      }
      int condval_138;
      if ((((slab * 2) + m_13) == 0)) {
        condval_138 = 0;
      } else {
        int condval_139;
        if ((((slab * 2) + m_13) == 1)) {
          condval_139 = 3;
        } else {
          condval_139 = (((slab * 2) + m_13) - 1);
        }
        condval_138 = condval_139;
      }
      int condval_140;
      if ((((slab * 2) + m_13) == 0)) {
        condval_140 = 0;
      } else {
        int condval_141;
        if ((((slab * 2) + m_13) == 1)) {
          condval_141 = 3;
        } else {
          condval_141 = (((slab * 2) + m_13) - 1);
        }
        condval_140 = condval_141;
      }
      int condval_142;
      if ((((slab * 2) + m_13) == 0)) {
        condval_142 = 0;
      } else {
        int condval_143;
        if ((((slab * 2) + m_13) == 1)) {
          condval_143 = 3;
        } else {
          condval_143 = (((slab * 2) + m_13) - 1);
        }
        condval_142 = condval_143;
      }
      int condval_144;
      if ((((slab * 2) + m_13) == 0)) {
        condval_144 = 0;
      } else {
        int condval_145;
        if ((((slab * 2) + m_13) == 1)) {
          condval_145 = 3;
        } else {
          condval_145 = (((slab * 2) + m_13) - 1);
        }
        condval_144 = condval_145;
      }
      int condval_146;
      if ((((slab * 2) + m_13) == 0)) {
        condval_146 = 0;
      } else {
        int condval_147;
        if ((((slab * 2) + m_13) == 1)) {
          condval_147 = 3;
        } else {
          condval_147 = (((slab * 2) + m_13) - 1);
        }
        condval_146 = condval_147;
      }
      int condval_148;
      if ((((slab * 2) + m_13) == 0)) {
        condval_148 = 0;
      } else {
        int condval_149;
        if ((((slab * 2) + m_13) == 1)) {
          condval_149 = 3;
        } else {
          condval_149 = (((slab * 2) + m_13) - 1);
        }
        condval_148 = condval_149;
      }
      int condval_150;
      if ((((slab * 2) + m_13) == 0)) {
        condval_150 = 0;
      } else {
        int condval_151;
        if ((((slab * 2) + m_13) == 1)) {
          condval_151 = 3;
        } else {
          condval_151 = (((slab * 2) + m_13) - 1);
        }
        condval_150 = condval_151;
      }
      int condval_152;
      if ((((slab * 2) + m_13) == 0)) {
        condval_152 = 0;
      } else {
        int condval_153;
        if ((((slab * 2) + m_13) == 1)) {
          condval_153 = 3;
        } else {
          condval_153 = (((slab * 2) + m_13) - 1);
        }
        condval_152 = condval_153;
      }
      int condval_154;
      if ((((slab * 2) + m_13) == 0)) {
        condval_154 = 0;
      } else {
        int condval_155;
        if ((((slab * 2) + m_13) == 1)) {
          condval_155 = 3;
        } else {
          condval_155 = (((slab * 2) + m_13) - 1);
        }
        condval_154 = condval_155;
      }
      int condval_156;
      if ((((slab * 2) + m_13) == 0)) {
        condval_156 = 0;
      } else {
        int condval_157;
        if ((((slab * 2) + m_13) == 1)) {
          condval_157 = 3;
        } else {
          condval_157 = (((slab * 2) + m_13) - 1);
        }
        condval_156 = condval_157;
      }
      int condval_158;
      if ((((slab * 2) + m_13) == 0)) {
        condval_158 = 0;
      } else {
        int condval_159;
        if ((((slab * 2) + m_13) == 1)) {
          condval_159 = 3;
        } else {
          condval_159 = (((slab * 2) + m_13) - 1);
        }
        condval_158 = condval_159;
      }
      int condval_160;
      if ((((slab * 2) + m_13) == 0)) {
        condval_160 = 0;
      } else {
        int condval_161;
        if ((((slab * 2) + m_13) == 1)) {
          condval_161 = 3;
        } else {
          condval_161 = (((slab * 2) + m_13) - 1);
        }
        condval_160 = condval_161;
      }
      int condval_162;
      if ((((slab * 2) + m_13) == 0)) {
        condval_162 = 0;
      } else {
        int condval_163;
        if ((((slab * 2) + m_13) == 1)) {
          condval_163 = 3;
        } else {
          condval_163 = (((slab * 2) + m_13) - 1);
        }
        condval_162 = condval_163;
      }
      int condval_164;
      if ((((slab * 2) + m_13) == 0)) {
        condval_164 = 0;
      } else {
        int condval_165;
        if ((((slab * 2) + m_13) == 1)) {
          condval_165 = 3;
        } else {
          condval_165 = (((slab * 2) + m_13) - 1);
        }
        condval_164 = condval_165;
      }
      int condval_166;
      if ((((slab * 2) + m_13) == 0)) {
        condval_166 = 0;
      } else {
        int condval_167;
        if ((((slab * 2) + m_13) == 1)) {
          condval_167 = 3;
        } else {
          condval_167 = (((slab * 2) + m_13) - 1);
        }
        condval_166 = condval_167;
      }
      int condval_168;
      if ((((slab * 2) + m_13) == 0)) {
        condval_168 = 0;
      } else {
        int condval_169;
        if ((((slab * 2) + m_13) == 1)) {
          condval_169 = 3;
        } else {
          condval_169 = (((slab * 2) + m_13) - 1);
        }
        condval_168 = condval_169;
      }
      int condval_170;
      if ((((slab * 2) + m_13) == 0)) {
        condval_170 = 0;
      } else {
        int condval_171;
        if ((((slab * 2) + m_13) == 1)) {
          condval_171 = 3;
        } else {
          condval_171 = (((slab * 2) + m_13) - 1);
        }
        condval_170 = condval_171;
      }
      int condval_172;
      if ((((slab * 2) + m_13) == 0)) {
        condval_172 = 0;
      } else {
        int condval_173;
        if ((((slab * 2) + m_13) == 1)) {
          condval_173 = 3;
        } else {
          condval_173 = (((slab * 2) + m_13) - 1);
        }
        condval_172 = condval_173;
      }
      int condval_174;
      if ((((slab * 2) + m_13) == 0)) {
        condval_174 = 0;
      } else {
        int condval_175;
        if ((((slab * 2) + m_13) == 1)) {
          condval_175 = 3;
        } else {
          condval_175 = (((slab * 2) + m_13) - 1);
        }
        condval_174 = condval_175;
      }
      int condval_176;
      if ((((slab * 2) + m_13) == 0)) {
        condval_176 = 0;
      } else {
        int condval_177;
        if ((((slab * 2) + m_13) == 1)) {
          condval_177 = 3;
        } else {
          condval_177 = (((slab * 2) + m_13) - 1);
        }
        condval_176 = condval_177;
      }
      int condval_178;
      if ((((slab * 2) + m_13) == 0)) {
        condval_178 = 0;
      } else {
        int condval_179;
        if ((((slab * 2) + m_13) == 1)) {
          condval_179 = 3;
        } else {
          condval_179 = (((slab * 2) + m_13) - 1);
        }
        condval_178 = condval_179;
      }
      int condval_180;
      if ((((slab * 2) + m_13) == 0)) {
        condval_180 = 0;
      } else {
        int condval_181;
        if ((((slab * 2) + m_13) == 1)) {
          condval_181 = 3;
        } else {
          condval_181 = (((slab * 2) + m_13) - 1);
        }
        condval_180 = condval_181;
      }
      int condval_182;
      if ((((slab * 2) + m_13) == 0)) {
        condval_182 = 0;
      } else {
        int condval_183;
        if ((((slab * 2) + m_13) == 1)) {
          condval_183 = 3;
        } else {
          condval_183 = (((slab * 2) + m_13) - 1);
        }
        condval_182 = condval_183;
      }
      int condval_184;
      if ((((slab * 2) + m_13) == 0)) {
        condval_184 = 0;
      } else {
        int condval_185;
        if ((((slab * 2) + m_13) == 1)) {
          condval_185 = 3;
        } else {
          condval_185 = (((slab * 2) + m_13) - 1);
        }
        condval_184 = condval_185;
      }
      int condval_186;
      if ((((slab * 2) + m_13) == 0)) {
        condval_186 = 0;
      } else {
        int condval_187;
        if ((((slab * 2) + m_13) == 1)) {
          condval_187 = 3;
        } else {
          condval_187 = (((slab * 2) + m_13) - 1);
        }
        condval_186 = condval_187;
      }
      int condval_188;
      if ((((slab * 2) + m_13) == 0)) {
        condval_188 = 0;
      } else {
        int condval_189;
        if ((((slab * 2) + m_13) == 1)) {
          condval_189 = 3;
        } else {
          condval_189 = (((slab * 2) + m_13) - 1);
        }
        condval_188 = condval_189;
      }
      int condval_190;
      if ((((slab * 2) + m_13) == 0)) {
        condval_190 = 0;
      } else {
        int condval_191;
        if ((((slab * 2) + m_13) == 1)) {
          condval_191 = 3;
        } else {
          condval_191 = (((slab * 2) + m_13) - 1);
        }
        condval_190 = condval_191;
      }
      int condval_192;
      if ((((slab * 2) + m_13) == 0)) {
        condval_192 = 0;
      } else {
        int condval_193;
        if ((((slab * 2) + m_13) == 1)) {
          condval_193 = 3;
        } else {
          condval_193 = (((slab * 2) + m_13) - 1);
        }
        condval_192 = condval_193;
      }
      int condval_194;
      if ((((slab * 2) + m_13) == 0)) {
        condval_194 = 0;
      } else {
        int condval_195;
        if ((((slab * 2) + m_13) == 1)) {
          condval_195 = 3;
        } else {
          condval_195 = (((slab * 2) + m_13) - 1);
        }
        condval_194 = condval_195;
      }
      int condval_196;
      if ((((slab * 2) + m_13) == 0)) {
        condval_196 = 0;
      } else {
        int condval_197;
        if ((((slab * 2) + m_13) == 1)) {
          condval_197 = 3;
        } else {
          condval_197 = (((slab * 2) + m_13) - 1);
        }
        condval_196 = condval_197;
      }
      int condval_198;
      if ((((slab * 2) + m_13) == 0)) {
        condval_198 = 0;
      } else {
        int condval_199;
        if ((((slab * 2) + m_13) == 1)) {
          condval_199 = 3;
        } else {
          condval_199 = (((slab * 2) + m_13) - 1);
        }
        condval_198 = condval_199;
      }
      int condval_200;
      if ((((slab * 2) + m_13) == 0)) {
        condval_200 = 0;
      } else {
        int condval_201;
        if ((((slab * 2) + m_13) == 1)) {
          condval_201 = 3;
        } else {
          condval_201 = (((slab * 2) + m_13) - 1);
        }
        condval_200 = condval_201;
      }
      int condval_202;
      if ((((slab * 2) + m_13) == 0)) {
        condval_202 = 0;
      } else {
        int condval_203;
        if ((((slab * 2) + m_13) == 1)) {
          condval_203 = 3;
        } else {
          condval_203 = (((slab * 2) + m_13) - 1);
        }
        condval_202 = condval_203;
      }
      int condval_204;
      if ((((slab * 2) + m_13) == 0)) {
        condval_204 = 0;
      } else {
        int condval_205;
        if ((((slab * 2) + m_13) == 1)) {
          condval_205 = 3;
        } else {
          condval_205 = (((slab * 2) + m_13) - 1);
        }
        condval_204 = condval_205;
      }
      int condval_206;
      if ((((slab * 2) + m_13) == 0)) {
        condval_206 = 0;
      } else {
        int condval_207;
        if ((((slab * 2) + m_13) == 1)) {
          condval_207 = 3;
        } else {
          condval_207 = (((slab * 2) + m_13) - 1);
        }
        condval_206 = condval_207;
      }
      int condval_208;
      if ((((slab * 2) + m_13) == 0)) {
        condval_208 = 0;
      } else {
        int condval_209;
        if ((((slab * 2) + m_13) == 1)) {
          condval_209 = 3;
        } else {
          condval_209 = (((slab * 2) + m_13) - 1);
        }
        condval_208 = condval_209;
      }
      int condval_210;
      if ((((slab * 2) + m_13) == 0)) {
        condval_210 = 0;
      } else {
        int condval_211;
        if ((((slab * 2) + m_13) == 1)) {
          condval_211 = 3;
        } else {
          condval_211 = (((slab * 2) + m_13) - 1);
        }
        condval_210 = condval_211;
      }
      int condval_212;
      if ((((slab * 2) + m_13) == 0)) {
        condval_212 = 0;
      } else {
        int condval_213;
        if ((((slab * 2) + m_13) == 1)) {
          condval_213 = 3;
        } else {
          condval_213 = (((slab * 2) + m_13) - 1);
        }
        condval_212 = condval_213;
      }
      int condval_214;
      if ((((slab * 2) + m_13) == 0)) {
        condval_214 = 0;
      } else {
        int condval_215;
        if ((((slab * 2) + m_13) == 1)) {
          condval_215 = 3;
        } else {
          condval_215 = (((slab * 2) + m_13) - 1);
        }
        condval_214 = condval_215;
      }
      int condval_216;
      if ((((slab * 2) + m_13) == 0)) {
        condval_216 = 0;
      } else {
        int condval_217;
        if ((((slab * 2) + m_13) == 1)) {
          condval_217 = 3;
        } else {
          condval_217 = (((slab * 2) + m_13) - 1);
        }
        condval_216 = condval_217;
      }
      int condval_218;
      if ((((slab * 2) + m_13) == 0)) {
        condval_218 = 0;
      } else {
        int condval_219;
        if ((((slab * 2) + m_13) == 1)) {
          condval_219 = 3;
        } else {
          condval_219 = (((slab * 2) + m_13) - 1);
        }
        condval_218 = condval_219;
      }
      int condval_220;
      if ((((slab * 2) + m_13) == 0)) {
        condval_220 = 0;
      } else {
        int condval_221;
        if ((((slab * 2) + m_13) == 1)) {
          condval_221 = 3;
        } else {
          condval_221 = (((slab * 2) + m_13) - 1);
        }
        condval_220 = condval_221;
      }
      int condval_222;
      if ((((slab * 2) + m_13) == 0)) {
        condval_222 = 0;
      } else {
        int condval_223;
        if ((((slab * 2) + m_13) == 1)) {
          condval_223 = 3;
        } else {
          condval_223 = (((slab * 2) + m_13) - 1);
        }
        condval_222 = condval_223;
      }
      int condval_224;
      if ((((slab * 2) + m_13) == 0)) {
        condval_224 = 0;
      } else {
        int condval_225;
        if ((((slab * 2) + m_13) == 1)) {
          condval_225 = 3;
        } else {
          condval_225 = (((slab * 2) + m_13) - 1);
        }
        condval_224 = condval_225;
      }
      int condval_226;
      if ((((slab * 2) + m_13) == 0)) {
        condval_226 = 0;
      } else {
        int condval_227;
        if ((((slab * 2) + m_13) == 1)) {
          condval_227 = 3;
        } else {
          condval_227 = (((slab * 2) + m_13) - 1);
        }
        condval_226 = condval_227;
      }
      int condval_228;
      if ((((slab * 2) + m_13) == 0)) {
        condval_228 = 0;
      } else {
        int condval_229;
        if ((((slab * 2) + m_13) == 1)) {
          condval_229 = 3;
        } else {
          condval_229 = (((slab * 2) + m_13) - 1);
        }
        condval_228 = condval_229;
      }
      int condval_230;
      if ((((slab * 2) + m_13) == 0)) {
        condval_230 = 0;
      } else {
        int condval_231;
        if ((((slab * 2) + m_13) == 1)) {
          condval_231 = 3;
        } else {
          condval_231 = (((slab * 2) + m_13) - 1);
        }
        condval_230 = condval_231;
      }
      int condval_232;
      if ((((slab * 2) + m_13) == 0)) {
        condval_232 = 0;
      } else {
        int condval_233;
        if ((((slab * 2) + m_13) == 1)) {
          condval_233 = 3;
        } else {
          condval_233 = (((slab * 2) + m_13) - 1);
        }
        condval_232 = condval_233;
      }
      int condval_234;
      if ((((slab * 2) + m_13) == 0)) {
        condval_234 = 0;
      } else {
        int condval_235;
        if ((((slab * 2) + m_13) == 1)) {
          condval_235 = 3;
        } else {
          condval_235 = (((slab * 2) + m_13) - 1);
        }
        condval_234 = condval_235;
      }
      int condval_236;
      if ((((slab * 2) + m_13) == 0)) {
        condval_236 = 0;
      } else {
        int condval_237;
        if ((((slab * 2) + m_13) == 1)) {
          condval_237 = 3;
        } else {
          condval_237 = (((slab * 2) + m_13) - 1);
        }
        condval_236 = condval_237;
      }
      int condval_238;
      if ((((slab * 2) + m_13) == 0)) {
        condval_238 = 0;
      } else {
        int condval_239;
        if ((((slab * 2) + m_13) == 1)) {
          condval_239 = 3;
        } else {
          condval_239 = (((slab * 2) + m_13) - 1);
        }
        condval_238 = condval_239;
      }
      int condval_240;
      if ((((slab * 2) + m_13) == 0)) {
        condval_240 = 0;
      } else {
        int condval_241;
        if ((((slab * 2) + m_13) == 1)) {
          condval_241 = 3;
        } else {
          condval_241 = (((slab * 2) + m_13) - 1);
        }
        condval_240 = condval_241;
      }
      int condval_242;
      if ((((slab * 2) + m_13) == 0)) {
        condval_242 = 0;
      } else {
        int condval_243;
        if ((((slab * 2) + m_13) == 1)) {
          condval_243 = 3;
        } else {
          condval_243 = (((slab * 2) + m_13) - 1);
        }
        condval_242 = condval_243;
      }
      int condval_244;
      if ((((slab * 2) + m_13) == 0)) {
        condval_244 = 0;
      } else {
        int condval_245;
        if ((((slab * 2) + m_13) == 1)) {
          condval_245 = 3;
        } else {
          condval_245 = (((slab * 2) + m_13) - 1);
        }
        condval_244 = condval_245;
      }
      int condval_246;
      if ((((slab * 2) + m_13) == 0)) {
        condval_246 = 0;
      } else {
        int condval_247;
        if ((((slab * 2) + m_13) == 1)) {
          condval_247 = 3;
        } else {
          condval_247 = (((slab * 2) + m_13) - 1);
        }
        condval_246 = condval_247;
      }
      int condval_248;
      if ((((slab * 2) + m_13) == 0)) {
        condval_248 = 0;
      } else {
        int condval_249;
        if ((((slab * 2) + m_13) == 1)) {
          condval_249 = 3;
        } else {
          condval_249 = (((slab * 2) + m_13) - 1);
        }
        condval_248 = condval_249;
      }
      int condval_250;
      if ((((slab * 2) + m_13) == 0)) {
        condval_250 = 0;
      } else {
        int condval_251;
        if ((((slab * 2) + m_13) == 1)) {
          condval_251 = 3;
        } else {
          condval_251 = (((slab * 2) + m_13) - 1);
        }
        condval_250 = condval_251;
      }
      int condval_252;
      if ((((slab * 2) + m_13) == 0)) {
        condval_252 = 0;
      } else {
        int condval_253;
        if ((((slab * 2) + m_13) == 1)) {
          condval_253 = 3;
        } else {
          condval_253 = (((slab * 2) + m_13) - 1);
        }
        condval_252 = condval_253;
      }
      int condval_254;
      if ((((slab * 2) + m_13) == 0)) {
        condval_254 = 0;
      } else {
        int condval_255;
        if ((((slab * 2) + m_13) == 1)) {
          condval_255 = 3;
        } else {
          condval_255 = (((slab * 2) + m_13) - 1);
        }
        condval_254 = condval_255;
      }
      int condval_256;
      if ((((slab * 2) + m_13) == 0)) {
        condval_256 = 0;
      } else {
        int condval_257;
        if ((((slab * 2) + m_13) == 1)) {
          condval_257 = 3;
        } else {
          condval_257 = (((slab * 2) + m_13) - 1);
        }
        condval_256 = condval_257;
      }
      int condval_258;
      if ((((slab * 2) + m_13) == 0)) {
        condval_258 = 0;
      } else {
        int condval_259;
        if ((((slab * 2) + m_13) == 1)) {
          condval_259 = 3;
        } else {
          condval_259 = (((slab * 2) + m_13) - 1);
        }
        condval_258 = condval_259;
      }
      int condval_260;
      if ((((slab * 2) + m_13) == 0)) {
        condval_260 = 0;
      } else {
        int condval_261;
        if ((((slab * 2) + m_13) == 1)) {
          condval_261 = 3;
        } else {
          condval_261 = (((slab * 2) + m_13) - 1);
        }
        condval_260 = condval_261;
      }
      int condval_262;
      if ((((slab * 2) + m_13) == 0)) {
        condval_262 = 0;
      } else {
        int condval_263;
        if ((((slab * 2) + m_13) == 1)) {
          condval_263 = 3;
        } else {
          condval_263 = (((slab * 2) + m_13) - 1);
        }
        condval_262 = condval_263;
      }
      int condval_264;
      if ((((slab * 2) + m_13) == 0)) {
        condval_264 = 0;
      } else {
        int condval_265;
        if ((((slab * 2) + m_13) == 1)) {
          condval_265 = 3;
        } else {
          condval_265 = (((slab * 2) + m_13) - 1);
        }
        condval_264 = condval_265;
      }
      int condval_266;
      if ((((slab * 2) + m_13) == 0)) {
        condval_266 = 0;
      } else {
        int condval_267;
        if ((((slab * 2) + m_13) == 1)) {
          condval_267 = 3;
        } else {
          condval_267 = (((slab * 2) + m_13) - 1);
        }
        condval_266 = condval_267;
      }
      nr_tl_shallow_joint::st128((&(X[(((((((int)blockIdx.y) * 163840) + (((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_148 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_150 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_152 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_154 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_156 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_158 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_160 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_162 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_164 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_166 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_168 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_170 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 1) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_172 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_174 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_176 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_178 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_180 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_182 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_184 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_186 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_188 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_190 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_192 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_194 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2))) * 81920)) + (((int)blockIdx.x) * 1024)) + (((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_196 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_198 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_200 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_202 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_204 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_206 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_208 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_210 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_212 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_214 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_216 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_218 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_220 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_222 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_224 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_226 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_228 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_230 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_232 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_234 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_236 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_238 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_240 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_242 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 3))) * 512)) + (((((((((((int)blockIdx.y) * 8) + ((((((condval_244 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_246 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_248 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_250 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_252 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_254 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_256 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_258 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_260 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_262 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_264 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_266 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) & 511))])), (nr_tl_shallow::e4pair(out[(m_13 * 8)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 2)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 1)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 3)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 4)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 6)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 5)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 7)]) << (uint)16)));
      #pragma unroll
      for (int n_8 = 0; n_8 < 4; ++n_8) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          #pragma unroll
          for (int j_10 = 0; j_10 < 2; ++j_10) {
            int64_t condval_268;
            if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)0)) {
              condval_268 = (int64_t)0;
            } else {
              int64_t condval_269;
              if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)1)) {
                condval_269 = (int64_t)3;
              } else {
                condval_269 = (((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) - (int64_t)1);
              }
              condval_268 = condval_269;
            }
            int64_t condval_270;
            if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)0)) {
              condval_270 = (int64_t)0;
            } else {
              int64_t condval_271;
              if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)1)) {
                condval_271 = (int64_t)3;
              } else {
                condval_271 = (((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) - (int64_t)1);
              }
              condval_270 = condval_271;
            }
            ((half_t*)rail)[((((((condval_270 * (int64_t)512) + (((int64_t)i_9) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((int64_t)n_8) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((int64_t)j_10))] = ((half_t)nr_tl_shallow::unpack(out[(((m_13 * 8) + (n_8 * 2)) + i_9)], j_10));
          }
        }
      }
    }
  }
  nr_tl_shallow::sync();
  __syncthreads();
  #pragma unroll
  for (int j_11 = 0; j_11 < 4; ++j_11) {
    mp[j_11] = (uint)0;
    #pragma unroll
    for (int bit = 0; bit < 4; ++bit) {
      half_t avg = ((half_t)nr_tl_shallow::hmul(((half_t)nr_tl_shallow::hadd(((half_t)nr_tl_shallow::hadd(((half_t*)rail)[((((((((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) & 3) ^ (((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) & 4) << 2)) ^ (((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) & 4) << 3)) ^ (((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) & 56) >> 1)) * 32) + ((((int)threadIdx.x) & 1) * 16)) + ((bit >> 1) * 8)) + (j_11 * 2)) + (bit & 1))], ((half_t*)rail)[(((((((((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 1) & 3) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 1) & 4) << 2)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 1) & 4) << 3)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 1) & 56) >> 1)) * 32) + ((((int)threadIdx.x) & 1) * 16)) + ((bit >> 1) * 8)) + (j_11 * 2)) + (bit & 1))])), ((half_t)nr_tl_shallow::hadd(((half_t*)rail)[(((((((((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 8) & 3) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 8) & 4) << 2)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 8) & 4) << 3)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 8) & 56) >> 1)) * 32) + ((((int)threadIdx.x) & 1) * 16)) + ((bit >> 1) * 8)) + (j_11 * 2)) + (bit & 1))], ((half_t*)rail)[(((((((((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 9) & 3) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 9) & 4) << 2)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 9) & 4) << 3)) ^ ((((((((int)threadIdx.x) >> 3) * 16) + (((((int)threadIdx.x) & 7) >> 1) * 2)) + 9) & 56) >> 1)) * 32) + ((((int)threadIdx.x) & 1) * 16)) + ((bit >> 1) * 8)) + (j_11 * 2)) + (bit & 1))])))), half_t(0x1p-2f/*2.500000e-01*/)));
      mp[j_11] = (mp[j_11] | (nr_tl_shallow::e4(avg) << ((uint)(bit * 8))));
    }
  }
  nr_tl_shallow_joint::st128((&(Y[((((((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10) * 1310720) + (((((((int)blockIdx.y) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) >> 3)) >> 3) * 40960)) + ((((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7) * 5120)) + (((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 5) * 512)) + ((((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127))])), mp[0], mp[1], mp[2], mp[3]);
}

