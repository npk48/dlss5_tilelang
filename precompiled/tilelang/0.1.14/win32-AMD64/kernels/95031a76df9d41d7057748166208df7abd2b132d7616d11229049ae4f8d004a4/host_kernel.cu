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

extern "C" __global__ void main_kernel(const half_t* __restrict__ Adapter, const float* __restrict__ Features, const half_t* __restrict__ G0, const half_t* __restrict__ G1, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y, int H0, int W0, int counter, int gx, int gy, int out_offset, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6, int sizes_7, int sizes_8, int sizes_9);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const half_t* __restrict__ Adapter, const float* __restrict__ Features, const half_t* __restrict__ G0, const half_t* __restrict__ G1, int* __restrict__ Status, uint* __restrict__ Wp, uint* __restrict__ Wq, uint* __restrict__ Wt0, uint* __restrict__ Wt1, uchar* __restrict__ X, uchar* __restrict__ Y, int H0, int W0, int counter, int gx, int gy, int out_offset, int sizes_0, int sizes_1, int sizes_2, int sizes_3, int sizes_4, int sizes_5, int sizes_6, int sizes_7, int sizes_8, int sizes_9) {
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
      float condval;
      if (((0 <= (((((int)blockIdx.x) * 8) + (((((((int)blockIdx.y) * 8) + (i * 4)) + (((int)threadIdx.x) >> 3)) + (c * H0)) * W0)) + (((int)threadIdx.x) & 7))) && ((((((int)blockIdx.x) * 8) + (((((((int)blockIdx.y) * 8) + (i * 4)) + (((int)threadIdx.x) >> 3)) + (c * H0)) * W0)) + (((int)threadIdx.x) & 7)) < sizes_8))) {
        condval = Features[(((((int64_t)((int)blockIdx.x)) * (int64_t)8) + (((((((int64_t)((int)blockIdx.y)) * (int64_t)8) + (((int64_t)i) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) + (((int64_t)c) * ((int64_t)H0))) * ((int64_t)W0))) + (((int64_t)((int)threadIdx.x)) & (int64_t)7))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      float v = condval;
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
        half_t condval_1;
        if (((((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2)) < sizes_9)) {
          condval_1 = Adapter[(((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2))];
        } else {
          condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        half_t condval_2;
        if ((((((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2)) + 32) < sizes_9)) {
          condval_2 = Adapter[((((((i_2 * 256) + ((((int)threadIdx.x) & 3) * 64)) + ((((int)threadIdx.x) >> 3) * 8)) + (n * 2)) + ((((int)threadIdx.x) & 7) >> 2)) + 32)];
        } else {
          condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
        }
        bb[i_2] = nr_tl_shallow::pack(condval_1, condval_2);
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
    half_t condval_3;
    if ((((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_4)) {
      condval_3 = G0[((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
    } else {
      condval_3 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    half_t condval_4;
    if (((((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_4)) {
      condval_4 = G0[(((((j_2 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
    } else {
      condval_4 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    ff[j_2] = nr_tl_shallow::mul(raw[j_2], nr_tl_shallow::pack(condval_3, condval_4));
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
      if ((((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4)) < sizes_0) {
        nr_tl_shallow_joint::ld128((&(b[0])), (&(Wt0[(((part * 256) + (pair * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
      if ((((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4)) < sizes_1) {
        nr_tl_shallow_joint::ld128((&(b_1[0])), (&(Wt1[(((part * 256) + (p_1 * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
      if ((((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4)) < sizes_2) {
        nr_tl_shallow_joint::ld128((&(b_2[0])), (&(Wq[(((component * 256) + (p_3 * 128)) + (((int)threadIdx.x) * 4))])));
      }
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
            int64_t condval_5;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_5 = (int64_t)0;
            } else {
              int64_t condval_6;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_6 = (int64_t)3;
              } else {
                condval_6 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_5 = condval_6;
            }
            int64_t condval_7;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_7 = (int64_t)0;
            } else {
              int64_t condval_8;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_8 = (int64_t)3;
              } else {
                condval_8 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_7 = condval_8;
            }
            int64_t condval_9;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_9 = (int64_t)0;
            } else {
              int64_t condval_10;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_10 = (int64_t)3;
              } else {
                condval_10 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_9 = condval_10;
            }
            int64_t condval_11;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_11 = (int64_t)0;
            } else {
              int64_t condval_12;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_12 = (int64_t)3;
              } else {
                condval_12 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_11 = condval_12;
            }
            qa[(((m_7 * 4) + (p_4 * 2)) + i_6)] = (nr_tl_shallow::e4pair(z[(((condval_7 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6))]) | (nr_tl_shallow::e4pair(z[((((condval_11 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6)) + (int64_t)2)]) << (uint)16));
          }
          if (component == 1) {
            int64_t condval_13;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_13 = (int64_t)0;
            } else {
              int64_t condval_14;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_14 = (int64_t)3;
              } else {
                condval_14 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_13 = condval_14;
            }
            int64_t condval_15;
            if ((((int64_t)m_7) == (int64_t)0)) {
              condval_15 = (int64_t)0;
            } else {
              int64_t condval_16;
              if ((((int64_t)m_7) == (int64_t)1)) {
                condval_16 = (int64_t)3;
              } else {
                condval_16 = (((int64_t)m_7) - (int64_t)1);
              }
              condval_15 = condval_16;
            }
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
            kb[(((m_7 * 4) + (i_6 * 2)) + p_4)] = (nr_tl_shallow::e4pair(z[(((condval_15 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6))]) | (nr_tl_shallow::e4pair(z[((((condval_19 * (int64_t)8) + (((int64_t)p_4) * (int64_t)4)) + ((int64_t)i_6)) + (int64_t)2)]) << (uint)16));
          }
        }
      }
    }
    if (component == 2) {
      #pragma unroll
      for (int part_1 = 0; part_1 < 2; ++part_1) {
        #pragma unroll
        for (int n_4 = 0; n_4 < 4; ++n_4) {
          int64_t condval_21;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_21 = (int64_t)0;
          } else {
            condval_21 = (int64_t)1;
          }
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
          vb[((part_1 * 8) + (n_4 * 2))] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_22 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_24 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
          int64_t condval_25;
          if ((((int64_t)part_1) == (int64_t)0)) {
            condval_25 = (int64_t)3;
          } else {
            condval_25 = (int64_t)2;
          }
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
          vb[(((part_1 * 8) + (n_4 * 2)) + 1)] = (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[((condval_26 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2))])) | (nr_tl_shallow::e4pair(nr_tl_shallow::transpose(z[(((condval_28 * (int64_t)8) + (((int64_t)n_4) * (int64_t)2)) + (int64_t)1)])) << (uint)16));
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
        if ((((((slab * 1024) + (m_8 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768) < sizes_2) {
          nr_tl_shallow_joint::ld128((&(seed[0])), (&(Wq[(((((slab * 1024) + (m_8 * 512)) + (p_5 * 128)) + (((int)threadIdx.x) * 4)) + 768)])));
        }
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
      int64_t condval_29;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)0)) {
        condval_29 = (int64_t)0;
      } else {
        int64_t condval_30;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)1)) {
          condval_30 = (int64_t)3;
        } else {
          condval_30 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) - (int64_t)1);
        }
        condval_29 = condval_30;
      }
      int64_t condval_31;
      if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)0)) {
        condval_31 = (int64_t)0;
      } else {
        int64_t condval_32;
        if ((((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) == (int64_t)1)) {
          condval_32 = (int64_t)3;
        } else {
          condval_32 = (((((int64_t)slab) * (int64_t)2) + (((int64_t)j_9) >> (int64_t)3)) - (int64_t)1);
        }
        condval_31 = condval_32;
      }
      half_t condval_33;
      if ((((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) < sizes_5)) {
        condval_33 = G1[((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2))];
      } else {
        condval_33 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      half_t condval_34;
      if (((((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1) < sizes_5)) {
        condval_34 = G1[(((((j_9 & 7) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 1)];
      } else {
        condval_34 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      out[j_9] = nr_tl_shallow::mul(ff[((condval_31 * (int64_t)8) + (((int64_t)j_9) & (int64_t)7))], nr_tl_shallow::pack(condval_33, condval_34));
    }
    #pragma unroll
    for (int p_8 = 0; p_8 < 2; ++p_8) {
      if (((p_8 * 128) + (((int)threadIdx.x) * 4)) < sizes_3) {
        nr_tl_shallow_joint::ld128((&(b_3[0])), (&(Wp[((p_8 * 128) + (((int)threadIdx.x) * 4))])));
      }
      #pragma unroll
      for (int m_12 = 0; m_12 < 2; ++m_12) {
        #pragma unroll
        for (int n_7 = 0; n_7 < 2; ++n_7) {
          nr_tl_shallow::mma8((&(out[(((m_12 * 8) + (p_8 * 4)) + (n_7 * 2))])), aa[(m_12 * 4)], aa[((m_12 * 4) + 1)], aa[((m_12 * 4) + 2)], aa[((m_12 * 4) + 3)], b_3[(n_7 * 2)], b_3[((n_7 * 2) + 1)]);
        }
      }
    }
    int H = max((H0 >> 1), (H0 >> 1));
    int W = max((W0 >> 1), (W0 >> 1));
    __syncthreads();
    #pragma unroll
    for (int m_13 = 0; m_13 < 2; ++m_13) {
      int condval_35;
      if ((((slab * 2) + m_13) == 0)) {
        condval_35 = 0;
      } else {
        int condval_36;
        if ((((slab * 2) + m_13) == 1)) {
          condval_36 = 3;
        } else {
          condval_36 = (((slab * 2) + m_13) - 1);
        }
        condval_35 = condval_36;
      }
      int condval_37;
      if ((((slab * 2) + m_13) == 0)) {
        condval_37 = 0;
      } else {
        int condval_38;
        if ((((slab * 2) + m_13) == 1)) {
          condval_38 = 3;
        } else {
          condval_38 = (((slab * 2) + m_13) - 1);
        }
        condval_37 = condval_38;
      }
      int condval_39;
      if ((((slab * 2) + m_13) == 0)) {
        condval_39 = 0;
      } else {
        int condval_40;
        if ((((slab * 2) + m_13) == 1)) {
          condval_40 = 3;
        } else {
          condval_40 = (((slab * 2) + m_13) - 1);
        }
        condval_39 = condval_40;
      }
      int condval_41;
      if ((((slab * 2) + m_13) == 0)) {
        condval_41 = 0;
      } else {
        int condval_42;
        if ((((slab * 2) + m_13) == 1)) {
          condval_42 = 3;
        } else {
          condval_42 = (((slab * 2) + m_13) - 1);
        }
        condval_41 = condval_42;
      }
      int condval_43;
      if ((((slab * 2) + m_13) == 0)) {
        condval_43 = 0;
      } else {
        int condval_44;
        if ((((slab * 2) + m_13) == 1)) {
          condval_44 = 3;
        } else {
          condval_44 = (((slab * 2) + m_13) - 1);
        }
        condval_43 = condval_44;
      }
      int condval_45;
      if ((((slab * 2) + m_13) == 0)) {
        condval_45 = 0;
      } else {
        int condval_46;
        if ((((slab * 2) + m_13) == 1)) {
          condval_46 = 3;
        } else {
          condval_46 = (((slab * 2) + m_13) - 1);
        }
        condval_45 = condval_46;
      }
      int condval_47;
      if ((((slab * 2) + m_13) == 0)) {
        condval_47 = 0;
      } else {
        int condval_48;
        if ((((slab * 2) + m_13) == 1)) {
          condval_48 = 3;
        } else {
          condval_48 = (((slab * 2) + m_13) - 1);
        }
        condval_47 = condval_48;
      }
      int condval_49;
      if ((((slab * 2) + m_13) == 0)) {
        condval_49 = 0;
      } else {
        int condval_50;
        if ((((slab * 2) + m_13) == 1)) {
          condval_50 = 3;
        } else {
          condval_50 = (((slab * 2) + m_13) - 1);
        }
        condval_49 = condval_50;
      }
      int condval_51;
      if ((((slab * 2) + m_13) == 0)) {
        condval_51 = 0;
      } else {
        int condval_52;
        if ((((slab * 2) + m_13) == 1)) {
          condval_52 = 3;
        } else {
          condval_52 = (((slab * 2) + m_13) - 1);
        }
        condval_51 = condval_52;
      }
      int condval_53;
      if ((((slab * 2) + m_13) == 0)) {
        condval_53 = 0;
      } else {
        int condval_54;
        if ((((slab * 2) + m_13) == 1)) {
          condval_54 = 3;
        } else {
          condval_54 = (((slab * 2) + m_13) - 1);
        }
        condval_53 = condval_54;
      }
      int condval_55;
      if ((((slab * 2) + m_13) == 0)) {
        condval_55 = 0;
      } else {
        int condval_56;
        if ((((slab * 2) + m_13) == 1)) {
          condval_56 = 3;
        } else {
          condval_56 = (((slab * 2) + m_13) - 1);
        }
        condval_55 = condval_56;
      }
      int condval_57;
      if ((((slab * 2) + m_13) == 0)) {
        condval_57 = 0;
      } else {
        int condval_58;
        if ((((slab * 2) + m_13) == 1)) {
          condval_58 = 3;
        } else {
          condval_58 = (((slab * 2) + m_13) - 1);
        }
        condval_57 = condval_58;
      }
      int condval_59;
      if ((((slab * 2) + m_13) == 0)) {
        condval_59 = 0;
      } else {
        int condval_60;
        if ((((slab * 2) + m_13) == 1)) {
          condval_60 = 3;
        } else {
          condval_60 = (((slab * 2) + m_13) - 1);
        }
        condval_59 = condval_60;
      }
      int condval_61;
      if ((((slab * 2) + m_13) == 0)) {
        condval_61 = 0;
      } else {
        int condval_62;
        if ((((slab * 2) + m_13) == 1)) {
          condval_62 = 3;
        } else {
          condval_62 = (((slab * 2) + m_13) - 1);
        }
        condval_61 = condval_62;
      }
      int condval_63;
      if ((((slab * 2) + m_13) == 0)) {
        condval_63 = 0;
      } else {
        int condval_64;
        if ((((slab * 2) + m_13) == 1)) {
          condval_64 = 3;
        } else {
          condval_64 = (((slab * 2) + m_13) - 1);
        }
        condval_63 = condval_64;
      }
      int condval_65;
      if ((((slab * 2) + m_13) == 0)) {
        condval_65 = 0;
      } else {
        int condval_66;
        if ((((slab * 2) + m_13) == 1)) {
          condval_66 = 3;
        } else {
          condval_66 = (((slab * 2) + m_13) - 1);
        }
        condval_65 = condval_66;
      }
      int condval_67;
      if ((((slab * 2) + m_13) == 0)) {
        condval_67 = 0;
      } else {
        int condval_68;
        if ((((slab * 2) + m_13) == 1)) {
          condval_68 = 3;
        } else {
          condval_68 = (((slab * 2) + m_13) - 1);
        }
        condval_67 = condval_68;
      }
      int condval_69;
      if ((((slab * 2) + m_13) == 0)) {
        condval_69 = 0;
      } else {
        int condval_70;
        if ((((slab * 2) + m_13) == 1)) {
          condval_70 = 3;
        } else {
          condval_70 = (((slab * 2) + m_13) - 1);
        }
        condval_69 = condval_70;
      }
      int condval_71;
      if ((((slab * 2) + m_13) == 0)) {
        condval_71 = 0;
      } else {
        int condval_72;
        if ((((slab * 2) + m_13) == 1)) {
          condval_72 = 3;
        } else {
          condval_72 = (((slab * 2) + m_13) - 1);
        }
        condval_71 = condval_72;
      }
      int condval_73;
      if ((((slab * 2) + m_13) == 0)) {
        condval_73 = 0;
      } else {
        int condval_74;
        if ((((slab * 2) + m_13) == 1)) {
          condval_74 = 3;
        } else {
          condval_74 = (((slab * 2) + m_13) - 1);
        }
        condval_73 = condval_74;
      }
      int condval_75;
      if ((((slab * 2) + m_13) == 0)) {
        condval_75 = 0;
      } else {
        int condval_76;
        if ((((slab * 2) + m_13) == 1)) {
          condval_76 = 3;
        } else {
          condval_76 = (((slab * 2) + m_13) - 1);
        }
        condval_75 = condval_76;
      }
      int condval_77;
      if ((((slab * 2) + m_13) == 0)) {
        condval_77 = 0;
      } else {
        int condval_78;
        if ((((slab * 2) + m_13) == 1)) {
          condval_78 = 3;
        } else {
          condval_78 = (((slab * 2) + m_13) - 1);
        }
        condval_77 = condval_78;
      }
      int condval_79;
      if ((((slab * 2) + m_13) == 0)) {
        condval_79 = 0;
      } else {
        int condval_80;
        if ((((slab * 2) + m_13) == 1)) {
          condval_80 = 3;
        } else {
          condval_80 = (((slab * 2) + m_13) - 1);
        }
        condval_79 = condval_80;
      }
      int condval_81;
      if ((((slab * 2) + m_13) == 0)) {
        condval_81 = 0;
      } else {
        int condval_82;
        if ((((slab * 2) + m_13) == 1)) {
          condval_82 = 3;
        } else {
          condval_82 = (((slab * 2) + m_13) - 1);
        }
        condval_81 = condval_82;
      }
      int condval_83;
      if ((((slab * 2) + m_13) == 0)) {
        condval_83 = 0;
      } else {
        int condval_84;
        if ((((slab * 2) + m_13) == 1)) {
          condval_84 = 3;
        } else {
          condval_84 = (((slab * 2) + m_13) - 1);
        }
        condval_83 = condval_84;
      }
      int condval_85;
      if ((((slab * 2) + m_13) == 0)) {
        condval_85 = 0;
      } else {
        int condval_86;
        if ((((slab * 2) + m_13) == 1)) {
          condval_86 = 3;
        } else {
          condval_86 = (((slab * 2) + m_13) - 1);
        }
        condval_85 = condval_86;
      }
      int condval_87;
      if ((((slab * 2) + m_13) == 0)) {
        condval_87 = 0;
      } else {
        int condval_88;
        if ((((slab * 2) + m_13) == 1)) {
          condval_88 = 3;
        } else {
          condval_88 = (((slab * 2) + m_13) - 1);
        }
        condval_87 = condval_88;
      }
      int condval_89;
      if ((((slab * 2) + m_13) == 0)) {
        condval_89 = 0;
      } else {
        int condval_90;
        if ((((slab * 2) + m_13) == 1)) {
          condval_90 = 3;
        } else {
          condval_90 = (((slab * 2) + m_13) - 1);
        }
        condval_89 = condval_90;
      }
      int condval_91;
      if ((((slab * 2) + m_13) == 0)) {
        condval_91 = 0;
      } else {
        int condval_92;
        if ((((slab * 2) + m_13) == 1)) {
          condval_92 = 3;
        } else {
          condval_92 = (((slab * 2) + m_13) - 1);
        }
        condval_91 = condval_92;
      }
      int condval_93;
      if ((((slab * 2) + m_13) == 0)) {
        condval_93 = 0;
      } else {
        int condval_94;
        if ((((slab * 2) + m_13) == 1)) {
          condval_94 = 3;
        } else {
          condval_94 = (((slab * 2) + m_13) - 1);
        }
        condval_93 = condval_94;
      }
      int condval_95;
      if ((((slab * 2) + m_13) == 0)) {
        condval_95 = 0;
      } else {
        int condval_96;
        if ((((slab * 2) + m_13) == 1)) {
          condval_96 = 3;
        } else {
          condval_96 = (((slab * 2) + m_13) - 1);
        }
        condval_95 = condval_96;
      }
      int condval_97;
      if ((((slab * 2) + m_13) == 0)) {
        condval_97 = 0;
      } else {
        int condval_98;
        if ((((slab * 2) + m_13) == 1)) {
          condval_98 = 3;
        } else {
          condval_98 = (((slab * 2) + m_13) - 1);
        }
        condval_97 = condval_98;
      }
      int condval_99;
      if ((((slab * 2) + m_13) == 0)) {
        condval_99 = 0;
      } else {
        int condval_100;
        if ((((slab * 2) + m_13) == 1)) {
          condval_100 = 3;
        } else {
          condval_100 = (((slab * 2) + m_13) - 1);
        }
        condval_99 = condval_100;
      }
      int condval_101;
      if ((((slab * 2) + m_13) == 0)) {
        condval_101 = 0;
      } else {
        int condval_102;
        if ((((slab * 2) + m_13) == 1)) {
          condval_102 = 3;
        } else {
          condval_102 = (((slab * 2) + m_13) - 1);
        }
        condval_101 = condval_102;
      }
      int condval_103;
      if ((((slab * 2) + m_13) == 0)) {
        condval_103 = 0;
      } else {
        int condval_104;
        if ((((slab * 2) + m_13) == 1)) {
          condval_104 = 3;
        } else {
          condval_104 = (((slab * 2) + m_13) - 1);
        }
        condval_103 = condval_104;
      }
      int condval_105;
      if ((((slab * 2) + m_13) == 0)) {
        condval_105 = 0;
      } else {
        int condval_106;
        if ((((slab * 2) + m_13) == 1)) {
          condval_106 = 3;
        } else {
          condval_106 = (((slab * 2) + m_13) - 1);
        }
        condval_105 = condval_106;
      }
      int condval_107;
      if ((((slab * 2) + m_13) == 0)) {
        condval_107 = 0;
      } else {
        int condval_108;
        if ((((slab * 2) + m_13) == 1)) {
          condval_108 = 3;
        } else {
          condval_108 = (((slab * 2) + m_13) - 1);
        }
        condval_107 = condval_108;
      }
      int condval_109;
      if ((((slab * 2) + m_13) == 0)) {
        condval_109 = 0;
      } else {
        int condval_110;
        if ((((slab * 2) + m_13) == 1)) {
          condval_110 = 3;
        } else {
          condval_110 = (((slab * 2) + m_13) - 1);
        }
        condval_109 = condval_110;
      }
      int condval_111;
      if ((((slab * 2) + m_13) == 0)) {
        condval_111 = 0;
      } else {
        int condval_112;
        if ((((slab * 2) + m_13) == 1)) {
          condval_112 = 3;
        } else {
          condval_112 = (((slab * 2) + m_13) - 1);
        }
        condval_111 = condval_112;
      }
      int condval_113;
      if ((((slab * 2) + m_13) == 0)) {
        condval_113 = 0;
      } else {
        int condval_114;
        if ((((slab * 2) + m_13) == 1)) {
          condval_114 = 3;
        } else {
          condval_114 = (((slab * 2) + m_13) - 1);
        }
        condval_113 = condval_114;
      }
      int condval_115;
      if ((((slab * 2) + m_13) == 0)) {
        condval_115 = 0;
      } else {
        int condval_116;
        if ((((slab * 2) + m_13) == 1)) {
          condval_116 = 3;
        } else {
          condval_116 = (((slab * 2) + m_13) - 1);
        }
        condval_115 = condval_116;
      }
      int condval_117;
      if ((((slab * 2) + m_13) == 0)) {
        condval_117 = 0;
      } else {
        int condval_118;
        if ((((slab * 2) + m_13) == 1)) {
          condval_118 = 3;
        } else {
          condval_118 = (((slab * 2) + m_13) - 1);
        }
        condval_117 = condval_118;
      }
      int condval_119;
      if ((((slab * 2) + m_13) == 0)) {
        condval_119 = 0;
      } else {
        int condval_120;
        if ((((slab * 2) + m_13) == 1)) {
          condval_120 = 3;
        } else {
          condval_120 = (((slab * 2) + m_13) - 1);
        }
        condval_119 = condval_120;
      }
      int condval_121;
      if ((((slab * 2) + m_13) == 0)) {
        condval_121 = 0;
      } else {
        int condval_122;
        if ((((slab * 2) + m_13) == 1)) {
          condval_122 = 3;
        } else {
          condval_122 = (((slab * 2) + m_13) - 1);
        }
        condval_121 = condval_122;
      }
      int condval_123;
      if ((((slab * 2) + m_13) == 0)) {
        condval_123 = 0;
      } else {
        int condval_124;
        if ((((slab * 2) + m_13) == 1)) {
          condval_124 = 3;
        } else {
          condval_124 = (((slab * 2) + m_13) - 1);
        }
        condval_123 = condval_124;
      }
      int condval_125;
      if ((((slab * 2) + m_13) == 0)) {
        condval_125 = 0;
      } else {
        int condval_126;
        if ((((slab * 2) + m_13) == 1)) {
          condval_126 = 3;
        } else {
          condval_126 = (((slab * 2) + m_13) - 1);
        }
        condval_125 = condval_126;
      }
      int condval_127;
      if ((((slab * 2) + m_13) == 0)) {
        condval_127 = 0;
      } else {
        int condval_128;
        if ((((slab * 2) + m_13) == 1)) {
          condval_128 = 3;
        } else {
          condval_128 = (((slab * 2) + m_13) - 1);
        }
        condval_127 = condval_128;
      }
      int condval_129;
      if ((((slab * 2) + m_13) == 0)) {
        condval_129 = 0;
      } else {
        int condval_130;
        if ((((slab * 2) + m_13) == 1)) {
          condval_130 = 3;
        } else {
          condval_130 = (((slab * 2) + m_13) - 1);
        }
        condval_129 = condval_130;
      }
      int condval_131;
      if ((((slab * 2) + m_13) == 0)) {
        condval_131 = 0;
      } else {
        int condval_132;
        if ((((slab * 2) + m_13) == 1)) {
          condval_132 = 3;
        } else {
          condval_132 = (((slab * 2) + m_13) - 1);
        }
        condval_131 = condval_132;
      }
      int condval_133;
      if ((((slab * 2) + m_13) == 0)) {
        condval_133 = 0;
      } else {
        int condval_134;
        if ((((slab * 2) + m_13) == 1)) {
          condval_134 = 3;
        } else {
          condval_134 = (((slab * 2) + m_13) - 1);
        }
        condval_133 = condval_134;
      }
      int condval_135;
      if ((((slab * 2) + m_13) == 0)) {
        condval_135 = 0;
      } else {
        int condval_136;
        if ((((slab * 2) + m_13) == 1)) {
          condval_136 = 3;
        } else {
          condval_136 = (((slab * 2) + m_13) - 1);
        }
        condval_135 = condval_136;
      }
      int condval_137;
      if ((((slab * 2) + m_13) == 0)) {
        condval_137 = 0;
      } else {
        int condval_138;
        if ((((slab * 2) + m_13) == 1)) {
          condval_138 = 3;
        } else {
          condval_138 = (((slab * 2) + m_13) - 1);
        }
        condval_137 = condval_138;
      }
      int condval_139;
      if ((((slab * 2) + m_13) == 0)) {
        condval_139 = 0;
      } else {
        int condval_140;
        if ((((slab * 2) + m_13) == 1)) {
          condval_140 = 3;
        } else {
          condval_140 = (((slab * 2) + m_13) - 1);
        }
        condval_139 = condval_140;
      }
      int condval_141;
      if ((((slab * 2) + m_13) == 0)) {
        condval_141 = 0;
      } else {
        int condval_142;
        if ((((slab * 2) + m_13) == 1)) {
          condval_142 = 3;
        } else {
          condval_142 = (((slab * 2) + m_13) - 1);
        }
        condval_141 = condval_142;
      }
      int condval_143;
      if ((((slab * 2) + m_13) == 0)) {
        condval_143 = 0;
      } else {
        int condval_144;
        if ((((slab * 2) + m_13) == 1)) {
          condval_144 = 3;
        } else {
          condval_144 = (((slab * 2) + m_13) - 1);
        }
        condval_143 = condval_144;
      }
      int condval_145;
      if ((((slab * 2) + m_13) == 0)) {
        condval_145 = 0;
      } else {
        int condval_146;
        if ((((slab * 2) + m_13) == 1)) {
          condval_146 = 3;
        } else {
          condval_146 = (((slab * 2) + m_13) - 1);
        }
        condval_145 = condval_146;
      }
      int condval_147;
      if ((((slab * 2) + m_13) == 0)) {
        condval_147 = 0;
      } else {
        int condval_148;
        if ((((slab * 2) + m_13) == 1)) {
          condval_148 = 3;
        } else {
          condval_148 = (((slab * 2) + m_13) - 1);
        }
        condval_147 = condval_148;
      }
      int condval_149;
      if ((((slab * 2) + m_13) == 0)) {
        condval_149 = 0;
      } else {
        int condval_150;
        if ((((slab * 2) + m_13) == 1)) {
          condval_150 = 3;
        } else {
          condval_150 = (((slab * 2) + m_13) - 1);
        }
        condval_149 = condval_150;
      }
      int condval_151;
      if ((((slab * 2) + m_13) == 0)) {
        condval_151 = 0;
      } else {
        int condval_152;
        if ((((slab * 2) + m_13) == 1)) {
          condval_152 = 3;
        } else {
          condval_152 = (((slab * 2) + m_13) - 1);
        }
        condval_151 = condval_152;
      }
      int condval_153;
      if ((((slab * 2) + m_13) == 0)) {
        condval_153 = 0;
      } else {
        int condval_154;
        if ((((slab * 2) + m_13) == 1)) {
          condval_154 = 3;
        } else {
          condval_154 = (((slab * 2) + m_13) - 1);
        }
        condval_153 = condval_154;
      }
      int condval_155;
      if ((((slab * 2) + m_13) == 0)) {
        condval_155 = 0;
      } else {
        int condval_156;
        if ((((slab * 2) + m_13) == 1)) {
          condval_156 = 3;
        } else {
          condval_156 = (((slab * 2) + m_13) - 1);
        }
        condval_155 = condval_156;
      }
      int condval_157;
      if ((((slab * 2) + m_13) == 0)) {
        condval_157 = 0;
      } else {
        int condval_158;
        if ((((slab * 2) + m_13) == 1)) {
          condval_158 = 3;
        } else {
          condval_158 = (((slab * 2) + m_13) - 1);
        }
        condval_157 = condval_158;
      }
      int condval_159;
      if ((((slab * 2) + m_13) == 0)) {
        condval_159 = 0;
      } else {
        int condval_160;
        if ((((slab * 2) + m_13) == 1)) {
          condval_160 = 3;
        } else {
          condval_160 = (((slab * 2) + m_13) - 1);
        }
        condval_159 = condval_160;
      }
      int condval_161;
      if ((((slab * 2) + m_13) == 0)) {
        condval_161 = 0;
      } else {
        int condval_162;
        if ((((slab * 2) + m_13) == 1)) {
          condval_162 = 3;
        } else {
          condval_162 = (((slab * 2) + m_13) - 1);
        }
        condval_161 = condval_162;
      }
      int condval_163;
      if ((((slab * 2) + m_13) == 0)) {
        condval_163 = 0;
      } else {
        int condval_164;
        if ((((slab * 2) + m_13) == 1)) {
          condval_164 = 3;
        } else {
          condval_164 = (((slab * 2) + m_13) - 1);
        }
        condval_163 = condval_164;
      }
      int condval_165;
      if ((((slab * 2) + m_13) == 0)) {
        condval_165 = 0;
      } else {
        int condval_166;
        if ((((slab * 2) + m_13) == 1)) {
          condval_166 = 3;
        } else {
          condval_166 = (((slab * 2) + m_13) - 1);
        }
        condval_165 = condval_166;
      }
      int condval_167;
      if ((((slab * 2) + m_13) == 0)) {
        condval_167 = 0;
      } else {
        int condval_168;
        if ((((slab * 2) + m_13) == 1)) {
          condval_168 = 3;
        } else {
          condval_168 = (((slab * 2) + m_13) - 1);
        }
        condval_167 = condval_168;
      }
      int condval_169;
      if ((((slab * 2) + m_13) == 0)) {
        condval_169 = 0;
      } else {
        int condval_170;
        if ((((slab * 2) + m_13) == 1)) {
          condval_170 = 3;
        } else {
          condval_170 = (((slab * 2) + m_13) - 1);
        }
        condval_169 = condval_170;
      }
      int condval_171;
      if ((((slab * 2) + m_13) == 0)) {
        condval_171 = 0;
      } else {
        int condval_172;
        if ((((slab * 2) + m_13) == 1)) {
          condval_172 = 3;
        } else {
          condval_172 = (((slab * 2) + m_13) - 1);
        }
        condval_171 = condval_172;
      }
      int condval_173;
      if ((((slab * 2) + m_13) == 0)) {
        condval_173 = 0;
      } else {
        int condval_174;
        if ((((slab * 2) + m_13) == 1)) {
          condval_174 = 3;
        } else {
          condval_174 = (((slab * 2) + m_13) - 1);
        }
        condval_173 = condval_174;
      }
      int condval_175;
      if ((((slab * 2) + m_13) == 0)) {
        condval_175 = 0;
      } else {
        int condval_176;
        if ((((slab * 2) + m_13) == 1)) {
          condval_176 = 3;
        } else {
          condval_176 = (((slab * 2) + m_13) - 1);
        }
        condval_175 = condval_176;
      }
      int condval_177;
      if ((((slab * 2) + m_13) == 0)) {
        condval_177 = 0;
      } else {
        int condval_178;
        if ((((slab * 2) + m_13) == 1)) {
          condval_178 = 3;
        } else {
          condval_178 = (((slab * 2) + m_13) - 1);
        }
        condval_177 = condval_178;
      }
      int condval_179;
      if ((((slab * 2) + m_13) == 0)) {
        condval_179 = 0;
      } else {
        int condval_180;
        if ((((slab * 2) + m_13) == 1)) {
          condval_180 = 3;
        } else {
          condval_180 = (((slab * 2) + m_13) - 1);
        }
        condval_179 = condval_180;
      }
      int condval_181;
      if ((((slab * 2) + m_13) == 0)) {
        condval_181 = 0;
      } else {
        int condval_182;
        if ((((slab * 2) + m_13) == 1)) {
          condval_182 = 3;
        } else {
          condval_182 = (((slab * 2) + m_13) - 1);
        }
        condval_181 = condval_182;
      }
      int condval_183;
      if ((((slab * 2) + m_13) == 0)) {
        condval_183 = 0;
      } else {
        int condval_184;
        if ((((slab * 2) + m_13) == 1)) {
          condval_184 = 3;
        } else {
          condval_184 = (((slab * 2) + m_13) - 1);
        }
        condval_183 = condval_184;
      }
      int condval_185;
      if ((((slab * 2) + m_13) == 0)) {
        condval_185 = 0;
      } else {
        int condval_186;
        if ((((slab * 2) + m_13) == 1)) {
          condval_186 = 3;
        } else {
          condval_186 = (((slab * 2) + m_13) - 1);
        }
        condval_185 = condval_186;
      }
      int condval_187;
      if ((((slab * 2) + m_13) == 0)) {
        condval_187 = 0;
      } else {
        int condval_188;
        if ((((slab * 2) + m_13) == 1)) {
          condval_188 = 3;
        } else {
          condval_188 = (((slab * 2) + m_13) - 1);
        }
        condval_187 = condval_188;
      }
      int condval_189;
      if ((((slab * 2) + m_13) == 0)) {
        condval_189 = 0;
      } else {
        int condval_190;
        if ((((slab * 2) + m_13) == 1)) {
          condval_190 = 3;
        } else {
          condval_190 = (((slab * 2) + m_13) - 1);
        }
        condval_189 = condval_190;
      }
      int condval_191;
      if ((((slab * 2) + m_13) == 0)) {
        condval_191 = 0;
      } else {
        int condval_192;
        if ((((slab * 2) + m_13) == 1)) {
          condval_192 = 3;
        } else {
          condval_192 = (((slab * 2) + m_13) - 1);
        }
        condval_191 = condval_192;
      }
      int condval_193;
      if ((((slab * 2) + m_13) == 0)) {
        condval_193 = 0;
      } else {
        int condval_194;
        if ((((slab * 2) + m_13) == 1)) {
          condval_194 = 3;
        } else {
          condval_194 = (((slab * 2) + m_13) - 1);
        }
        condval_193 = condval_194;
      }
      int condval_195;
      if ((((slab * 2) + m_13) == 0)) {
        condval_195 = 0;
      } else {
        int condval_196;
        if ((((slab * 2) + m_13) == 1)) {
          condval_196 = 3;
        } else {
          condval_196 = (((slab * 2) + m_13) - 1);
        }
        condval_195 = condval_196;
      }
      int condval_197;
      if ((((slab * 2) + m_13) == 0)) {
        condval_197 = 0;
      } else {
        int condval_198;
        if ((((slab * 2) + m_13) == 1)) {
          condval_198 = 3;
        } else {
          condval_198 = (((slab * 2) + m_13) - 1);
        }
        condval_197 = condval_198;
      }
      int condval_199;
      if ((((slab * 2) + m_13) == 0)) {
        condval_199 = 0;
      } else {
        int condval_200;
        if ((((slab * 2) + m_13) == 1)) {
          condval_200 = 3;
        } else {
          condval_200 = (((slab * 2) + m_13) - 1);
        }
        condval_199 = condval_200;
      }
      int condval_201;
      if ((((slab * 2) + m_13) == 0)) {
        condval_201 = 0;
      } else {
        int condval_202;
        if ((((slab * 2) + m_13) == 1)) {
          condval_202 = 3;
        } else {
          condval_202 = (((slab * 2) + m_13) - 1);
        }
        condval_201 = condval_202;
      }
      int condval_203;
      if ((((slab * 2) + m_13) == 0)) {
        condval_203 = 0;
      } else {
        int condval_204;
        if ((((slab * 2) + m_13) == 1)) {
          condval_204 = 3;
        } else {
          condval_204 = (((slab * 2) + m_13) - 1);
        }
        condval_203 = condval_204;
      }
      int condval_205;
      if ((((slab * 2) + m_13) == 0)) {
        condval_205 = 0;
      } else {
        int condval_206;
        if ((((slab * 2) + m_13) == 1)) {
          condval_206 = 3;
        } else {
          condval_206 = (((slab * 2) + m_13) - 1);
        }
        condval_205 = condval_206;
      }
      int condval_207;
      if ((((slab * 2) + m_13) == 0)) {
        condval_207 = 0;
      } else {
        int condval_208;
        if ((((slab * 2) + m_13) == 1)) {
          condval_208 = 3;
        } else {
          condval_208 = (((slab * 2) + m_13) - 1);
        }
        condval_207 = condval_208;
      }
      int condval_209;
      if ((((slab * 2) + m_13) == 0)) {
        condval_209 = 0;
      } else {
        int condval_210;
        if ((((slab * 2) + m_13) == 1)) {
          condval_210 = 3;
        } else {
          condval_210 = (((slab * 2) + m_13) - 1);
        }
        condval_209 = condval_210;
      }
      int condval_211;
      if ((((slab * 2) + m_13) == 0)) {
        condval_211 = 0;
      } else {
        int condval_212;
        if ((((slab * 2) + m_13) == 1)) {
          condval_212 = 3;
        } else {
          condval_212 = (((slab * 2) + m_13) - 1);
        }
        condval_211 = condval_212;
      }
      int condval_213;
      if ((((slab * 2) + m_13) == 0)) {
        condval_213 = 0;
      } else {
        int condval_214;
        if ((((slab * 2) + m_13) == 1)) {
          condval_214 = 3;
        } else {
          condval_214 = (((slab * 2) + m_13) - 1);
        }
        condval_213 = condval_214;
      }
      int condval_215;
      if ((((slab * 2) + m_13) == 0)) {
        condval_215 = 0;
      } else {
        int condval_216;
        if ((((slab * 2) + m_13) == 1)) {
          condval_216 = 3;
        } else {
          condval_216 = (((slab * 2) + m_13) - 1);
        }
        condval_215 = condval_216;
      }
      int condval_217;
      if ((((slab * 2) + m_13) == 0)) {
        condval_217 = 0;
      } else {
        int condval_218;
        if ((((slab * 2) + m_13) == 1)) {
          condval_218 = 3;
        } else {
          condval_218 = (((slab * 2) + m_13) - 1);
        }
        condval_217 = condval_218;
      }
      int condval_219;
      if ((((slab * 2) + m_13) == 0)) {
        condval_219 = 0;
      } else {
        int condval_220;
        if ((((slab * 2) + m_13) == 1)) {
          condval_220 = 3;
        } else {
          condval_220 = (((slab * 2) + m_13) - 1);
        }
        condval_219 = condval_220;
      }
      int condval_221;
      if ((((slab * 2) + m_13) == 0)) {
        condval_221 = 0;
      } else {
        int condval_222;
        if ((((slab * 2) + m_13) == 1)) {
          condval_222 = 3;
        } else {
          condval_222 = (((slab * 2) + m_13) - 1);
        }
        condval_221 = condval_222;
      }
      int condval_223;
      if ((((slab * 2) + m_13) == 0)) {
        condval_223 = 0;
      } else {
        int condval_224;
        if ((((slab * 2) + m_13) == 1)) {
          condval_224 = 3;
        } else {
          condval_224 = (((slab * 2) + m_13) - 1);
        }
        condval_223 = condval_224;
      }
      int condval_225;
      if ((((slab * 2) + m_13) == 0)) {
        condval_225 = 0;
      } else {
        int condval_226;
        if ((((slab * 2) + m_13) == 1)) {
          condval_226 = 3;
        } else {
          condval_226 = (((slab * 2) + m_13) - 1);
        }
        condval_225 = condval_226;
      }
      int condval_227;
      if ((((slab * 2) + m_13) == 0)) {
        condval_227 = 0;
      } else {
        int condval_228;
        if ((((slab * 2) + m_13) == 1)) {
          condval_228 = 3;
        } else {
          condval_228 = (((slab * 2) + m_13) - 1);
        }
        condval_227 = condval_228;
      }
      int condval_229;
      if ((((slab * 2) + m_13) == 0)) {
        condval_229 = 0;
      } else {
        int condval_230;
        if ((((slab * 2) + m_13) == 1)) {
          condval_230 = 3;
        } else {
          condval_230 = (((slab * 2) + m_13) - 1);
        }
        condval_229 = condval_230;
      }
      int condval_231;
      if ((((slab * 2) + m_13) == 0)) {
        condval_231 = 0;
      } else {
        int condval_232;
        if ((((slab * 2) + m_13) == 1)) {
          condval_232 = 3;
        } else {
          condval_232 = (((slab * 2) + m_13) - 1);
        }
        condval_231 = condval_232;
      }
      int condval_233;
      if ((((slab * 2) + m_13) == 0)) {
        condval_233 = 0;
      } else {
        int condval_234;
        if ((((slab * 2) + m_13) == 1)) {
          condval_234 = 3;
        } else {
          condval_234 = (((slab * 2) + m_13) - 1);
        }
        condval_233 = condval_234;
      }
      int condval_235;
      if ((((slab * 2) + m_13) == 0)) {
        condval_235 = 0;
      } else {
        int condval_236;
        if ((((slab * 2) + m_13) == 1)) {
          condval_236 = 3;
        } else {
          condval_236 = (((slab * 2) + m_13) - 1);
        }
        condval_235 = condval_236;
      }
      int condval_237;
      if ((((slab * 2) + m_13) == 0)) {
        condval_237 = 0;
      } else {
        int condval_238;
        if ((((slab * 2) + m_13) == 1)) {
          condval_238 = 3;
        } else {
          condval_238 = (((slab * 2) + m_13) - 1);
        }
        condval_237 = condval_238;
      }
      int condval_239;
      if ((((slab * 2) + m_13) == 0)) {
        condval_239 = 0;
      } else {
        int condval_240;
        if ((((slab * 2) + m_13) == 1)) {
          condval_240 = 3;
        } else {
          condval_240 = (((slab * 2) + m_13) - 1);
        }
        condval_239 = condval_240;
      }
      int condval_241;
      if ((((slab * 2) + m_13) == 0)) {
        condval_241 = 0;
      } else {
        int condval_242;
        if ((((slab * 2) + m_13) == 1)) {
          condval_242 = 3;
        } else {
          condval_242 = (((slab * 2) + m_13) - 1);
        }
        condval_241 = condval_242;
      }
      int condval_243;
      if ((((slab * 2) + m_13) == 0)) {
        condval_243 = 0;
      } else {
        int condval_244;
        if ((((slab * 2) + m_13) == 1)) {
          condval_244 = 3;
        } else {
          condval_244 = (((slab * 2) + m_13) - 1);
        }
        condval_243 = condval_244;
      }
      int condval_245;
      if ((((slab * 2) + m_13) == 0)) {
        condval_245 = 0;
      } else {
        int condval_246;
        if ((((slab * 2) + m_13) == 1)) {
          condval_246 = 3;
        } else {
          condval_246 = (((slab * 2) + m_13) - 1);
        }
        condval_245 = condval_246;
      }
      int condval_247;
      if ((((slab * 2) + m_13) == 0)) {
        condval_247 = 0;
      } else {
        int condval_248;
        if ((((slab * 2) + m_13) == 1)) {
          condval_248 = 3;
        } else {
          condval_248 = (((slab * 2) + m_13) - 1);
        }
        condval_247 = condval_248;
      }
      int condval_249;
      if ((((slab * 2) + m_13) == 0)) {
        condval_249 = 0;
      } else {
        int condval_250;
        if ((((slab * 2) + m_13) == 1)) {
          condval_250 = 3;
        } else {
          condval_250 = (((slab * 2) + m_13) - 1);
        }
        condval_249 = condval_250;
      }
      int condval_251;
      if ((((slab * 2) + m_13) == 0)) {
        condval_251 = 0;
      } else {
        int condval_252;
        if ((((slab * 2) + m_13) == 1)) {
          condval_252 = 3;
        } else {
          condval_252 = (((slab * 2) + m_13) - 1);
        }
        condval_251 = condval_252;
      }
      int condval_253;
      if ((((slab * 2) + m_13) == 0)) {
        condval_253 = 0;
      } else {
        int condval_254;
        if ((((slab * 2) + m_13) == 1)) {
          condval_254 = 3;
        } else {
          condval_254 = (((slab * 2) + m_13) - 1);
        }
        condval_253 = condval_254;
      }
      int condval_255;
      if ((((slab * 2) + m_13) == 0)) {
        condval_255 = 0;
      } else {
        int condval_256;
        if ((((slab * 2) + m_13) == 1)) {
          condval_256 = 3;
        } else {
          condval_256 = (((slab * 2) + m_13) - 1);
        }
        condval_255 = condval_256;
      }
      int condval_257;
      if ((((slab * 2) + m_13) == 0)) {
        condval_257 = 0;
      } else {
        int condval_258;
        if ((((slab * 2) + m_13) == 1)) {
          condval_258 = 3;
        } else {
          condval_258 = (((slab * 2) + m_13) - 1);
        }
        condval_257 = condval_258;
      }
      int condval_259;
      if ((((slab * 2) + m_13) == 0)) {
        condval_259 = 0;
      } else {
        int condval_260;
        if ((((slab * 2) + m_13) == 1)) {
          condval_260 = 3;
        } else {
          condval_260 = (((slab * 2) + m_13) - 1);
        }
        condval_259 = condval_260;
      }
      int condval_261;
      if ((((slab * 2) + m_13) == 0)) {
        condval_261 = 0;
      } else {
        int condval_262;
        if ((((slab * 2) + m_13) == 1)) {
          condval_262 = 3;
        } else {
          condval_262 = (((slab * 2) + m_13) - 1);
        }
        condval_261 = condval_262;
      }
      int condval_263;
      if ((((slab * 2) + m_13) == 0)) {
        condval_263 = 0;
      } else {
        int condval_264;
        if ((((slab * 2) + m_13) == 1)) {
          condval_264 = 3;
        } else {
          condval_264 = (((slab * 2) + m_13) - 1);
        }
        condval_263 = condval_264;
      }
      int condval_265;
      if ((((slab * 2) + m_13) == 0)) {
        condval_265 = 0;
      } else {
        int condval_266;
        if ((((slab * 2) + m_13) == 1)) {
          condval_266 = 3;
        } else {
          condval_266 = (((slab * 2) + m_13) - 1);
        }
        condval_265 = condval_266;
      }
      int condval_267;
      if ((((slab * 2) + m_13) == 0)) {
        condval_267 = 0;
      } else {
        int condval_268;
        if ((((slab * 2) + m_13) == 1)) {
          condval_268 = 3;
        } else {
          condval_268 = (((slab * 2) + m_13) - 1);
        }
        condval_267 = condval_268;
      }
      int condval_269;
      if ((((slab * 2) + m_13) == 0)) {
        condval_269 = 0;
      } else {
        int condval_270;
        if ((((slab * 2) + m_13) == 1)) {
          condval_270 = 3;
        } else {
          condval_270 = (((slab * 2) + m_13) - 1);
        }
        condval_269 = condval_270;
      }
      int condval_271;
      if ((((slab * 2) + m_13) == 0)) {
        condval_271 = 0;
      } else {
        int condval_272;
        if ((((slab * 2) + m_13) == 1)) {
          condval_272 = 3;
        } else {
          condval_272 = (((slab * 2) + m_13) - 1);
        }
        condval_271 = condval_272;
      }
      int condval_273;
      if ((((slab * 2) + m_13) == 0)) {
        condval_273 = 0;
      } else {
        int condval_274;
        if ((((slab * 2) + m_13) == 1)) {
          condval_274 = 3;
        } else {
          condval_274 = (((slab * 2) + m_13) - 1);
        }
        condval_273 = condval_274;
      }
      int dest = max(((((((int)blockIdx.x) * 1024) + ((((((int)blockIdx.y) * 2) + ((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_35 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_37 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_39 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_41 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_43 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_45 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_47 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_49 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_51 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_53 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_55 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_57 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 1) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_59 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_61 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_63 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_65 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_67 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_69 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_71 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_73 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_75 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_77 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_79 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_81 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2)))) * (W0 >> 2)) * 512)) + (((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_83 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_85 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_87 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_89 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_91 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_93 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_95 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_97 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_99 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_101 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_103 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_105 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_107 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_109 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_111 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_113 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_115 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_117 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_119 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_121 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_123 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_125 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_127 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_129 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 3))) * 512)) + (((((((((((int)blockIdx.y) * 8) + ((((((condval_131 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_133 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_135 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_137 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_139 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_141 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_143 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_145 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_147 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_149 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_151 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_153 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) & 511)), ((((((int)blockIdx.x) * 1024) + ((((((int)blockIdx.y) * 2) + ((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_155 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_157 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_159 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_161 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_163 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_165 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_167 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_169 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_171 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_173 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_175 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_177 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 1) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_179 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_181 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_183 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_185 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_187 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_189 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_191 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_193 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_195 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_197 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_199 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_201 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2)))) * (W0 >> 2)) * 512)) + (((int)(((((((((((((int)blockIdx.y) * 8) + ((((((condval_203 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_205 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_207 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_209 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_211 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_213 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_215 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_217 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_219 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_221 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_223 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_225 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 2) | ((((((((((((int)blockIdx.y) * 8) + ((((((condval_227 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_229 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_231 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_233 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_235 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_237 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_239 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_241 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_243 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_245 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_247 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_249 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) >> 9) == 3))) * 512)) + (((((((((((int)blockIdx.y) * 8) + ((((((condval_251 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_253 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_255 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 256) + (((((((int)blockIdx.x) * 8) + ((((((condval_257 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_259 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_261 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7) >> 1) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + (((((((((int)blockIdx.y) * 8) + ((((((condval_263 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_265 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_267 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 8) + ((((((condval_269 * 16) + (((int)threadIdx.x) >> 2)) & 48) | ((((condval_271 * 16) + (((int)threadIdx.x) >> 2)) & 7) << 1)) | ((((condval_273 * 16) + (((int)threadIdx.x) >> 2)) & 8) >> 3)) & 7)) & 7)) & 1) * 4)) + (((((int)threadIdx.x) & 3) * 8) & 3)) & 511)));
      if (0 <= dest) {
        if (0 <= (out_offset + dest)) {
          if ((out_offset + dest) < sizes_6) {
            nr_tl_shallow_joint::st128((&(X[(((int64_t)out_offset) + ((int64_t)dest))])), (nr_tl_shallow::e4pair(out[(m_13 * 8)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 2)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 1)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 3)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 4)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 6)]) << (uint)16)), (nr_tl_shallow::e4pair(out[((m_13 * 8) + 5)]) | (nr_tl_shallow::e4pair(out[((m_13 * 8) + 7)]) << (uint)16)));
          }
        }
      }
      #pragma unroll
      for (int n_8 = 0; n_8 < 4; ++n_8) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          #pragma unroll
          for (int j_10 = 0; j_10 < 2; ++j_10) {
            int64_t condval_275;
            if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)0)) {
              condval_275 = (int64_t)0;
            } else {
              int64_t condval_276;
              if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)1)) {
                condval_276 = (int64_t)3;
              } else {
                condval_276 = (((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) - (int64_t)1);
              }
              condval_275 = condval_276;
            }
            int64_t condval_277;
            if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)0)) {
              condval_277 = (int64_t)0;
            } else {
              int64_t condval_278;
              if ((((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) == (int64_t)1)) {
                condval_278 = (int64_t)3;
              } else {
                condval_278 = (((((int64_t)slab) * (int64_t)2) + ((int64_t)m_13)) - (int64_t)1);
              }
              condval_277 = condval_278;
            }
            ((half_t*)rail)[((((((condval_277 * (int64_t)512) + (((int64_t)i_9) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)32)) + (((int64_t)n_8) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + ((int64_t)j_10))] = ((half_t)nr_tl_shallow::unpack(out[(((m_13 * 8) + (n_8 * 2)) + i_9)], j_10));
          }
        }
      }
    }
  }
  nr_tl_shallow::sync();
  int base = max((((((((((((((int)blockIdx.y) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) >> 3)) >> 3) + ((H0 >> 4) * ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10))) * (W0 >> 3)) * 512) + (((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 5) * 512)) + (((W0 >> 6) * (((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + ((((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127)), (((((((((((((int)blockIdx.y) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) >> 3)) >> 3) + ((H0 >> 4) * ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 10))) * (W0 >> 3)) * 512) + (((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 5) * 512)) + (((W0 >> 6) * (((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) >> 7) & 7)) * 512)) + ((((((((int)blockIdx.x) >> 1) * 8) + (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 7)) >> 3) & 3) * 128)) + ((((((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) >> 4) * 1024) + ((((((((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 1) << 4) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 6) >> 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 8)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 16) << 1)) ^ (((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32)) ^ ((((((((((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 3) << 1) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 2)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 4) << 3)) ^ ((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 8)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 16) >> 4)) ^ (((((((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 8) + (((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 7) >> 1)) & 7)) & 32) >> 1)) & 32) >> 3)) * 16)) + ((((((((int)threadIdx.x) & 1) * 4) & 3) | ((((((int)threadIdx.x) & 1) * 4) & 4) << 2)) | ((((((int)threadIdx.x) & 1) * 4) & 24) >> 1)) & 15)) & 127)));
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
  if (0 <= base) {
    if (base < sizes_7) {
      nr_tl_shallow_joint::st128((&(Y[((int64_t)base)])), mp[0], mp[1], mp[2], mp[3]);
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

