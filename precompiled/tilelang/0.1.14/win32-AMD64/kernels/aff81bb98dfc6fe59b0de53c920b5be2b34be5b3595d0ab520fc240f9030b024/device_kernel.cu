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

extern "C" __global__ void kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX) {
  uint temp[4];
  extern __shared__ __align__(1024) int S[];
  int v = 0;
  uint Residual[16];
  uint C[32];
  uint P[16];
  uint A[16];
  uint X[32];
  int a[16];
  int b[4];
  int a_1[4];
  int b_1[4];
  uint v_1 = (uint)0;
  uint v_2 = (uint)0;
  uint v_3 = (uint)0;
  uint v_4 = (uint)0;
  uint v_5 = (uint)0;
  uint v_6 = (uint)0;
  int a_2[16];
  int b_2[4];
  uint v_7 = (uint)0;
  uint v_8 = (uint)0;
  uint v_9 = (uint)0;
  uint Z[96];
  int a_3[16];
  int b_3[4];
  uint QP[16];
  uint KP[16];
  uint VP[16];
  int a_4[4];
  int b_4[4];
  uint Q[16];
  uint K[16];
  uint V[16];
  uint v_10 = (uint)0;
  uint v_11 = (uint)0;
  uint v_12 = (uint)0;
  uint v_13 = (uint)0;
  uint v_14 = (uint)0;
  uint v_15 = (uint)0;
  uint v_16 = (uint)0;
  uint v_17 = (uint)0;
  uint v_18 = (uint)0;
  uint v_19 = (uint)0;
  uint v_20 = (uint)0;
  uint v_21 = (uint)0;
  uint L[32];
  uint v_22 = (uint)0;
  uint v_23 = (uint)0;
  uint v_24 = (uint)0;
  uint acc[2];
  uint v_25 = (uint)0;
  uint v_26 = (uint)0;
  uint v_27 = (uint)0;
  int a_5[8];
  int b_5[4];
  int v_28 = 0;
  #pragma unroll
  for (int slab = 0; slab < 2; ++slab) {
    #pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
      int condval;
      if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
        condval = ((((((((((int)blockIdx.x) / 10) * 81920) + (((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) >> 1) * 40960)) + ((((int)blockIdx.x) % 10) * 4096)) + (((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7) >> 2) * 2048)) + ((((((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 1) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 6) << 3)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 8) >> 2)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 16) >> 1)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 224) << 4))) + ((((((((((((int)blockIdx.x) / 10) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) >> 1) & 1) | (((((((int)blockIdx.x) % 10) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) & 3) << 1)) | ((((((((int)blockIdx.x) / 10) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) & 1) << 3)) & 1) << 2)) + ((((((((((((int)blockIdx.x) / 10) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) >> 1) & 1) | (((((((int)blockIdx.x) % 10) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) & 3) << 1)) | ((((((((int)blockIdx.x) / 10) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) & 1) << 3)) & 14) << 5));
      } else {
        condval = -1;
      }
      v = condval;
      int raw = v;
      uint broadcast_var = (uint)0;
      *(uint4*)(temp + 0) = make_uint4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      if ((0 <= raw) & (raw <= 19234192)) {
        {
          int SW_local_cast[4];
          *(int4*)(SW_local_cast + 0) = *(int4*)(SW + (((raw >> 4) * 4) + 10280960));
          uint4 __1;
          int4 v_ = *(int4*)(SW_local_cast + 0);
          __1.x = (uint)(v_.x);
          __1.y = (uint)(v_.y);
          __1.z = (uint)(v_.z);
          __1.w = (uint)(v_.w);
          *(uint4*)(temp + 0) = __1;
        }
      }
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        S[(((((((((int)threadIdx.x) >> 5) * 512) + (slab * 256)) + ((j & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((j >> 1) * 2)) + parity)] = ((int)temp[j]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int word = 0; word < 16; ++word) {
    int v__1 = S[(((((((int)threadIdx.x) >> 5) * 512) + ((word >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word & 3) >> 1) * 2))];
    int v__2 = S[((((((((int)threadIdx.x) >> 5) * 512) + ((word >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word & 3) >> 1) * 2)) + 1)];
    int condval_1;
    if (((word % 2) == 0)) {
      condval_1 = 21520;
    } else {
      condval_1 = 30258;
    }
    Residual[word] = wide_packet_prmt((*(uint *)(&(v__1))), (*(uint *)(&(v__2))), condval_1);
  }
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      C[((m * 8) + (n * 2))] = (uint)0;
      C[(((m * 8) + (n * 2)) + 1)] = (uint)0;
    }
  }
  for (int slab_1 = 0; slab_1 < 4; ++slab_1) {
    #pragma unroll
    for (int m_1 = 0; m_1 < 4; ++m_1) {
      #pragma unroll
      for (int n_1 = 0; n_1 < 4; ++n_1) {
        X[((m_1 * 8) + (n_1 * 2))] = (uint)0;
        X[(((m_1 * 8) + (n_1 * 2)) + 1)] = (uint)0;
      }
    }
    for (int kp = 0; kp < 3; ++kp) {
      #pragma unroll
      for (int m_2 = 0; m_2 < 4; ++m_2) {
        *(int4*)(a + (m_2 * 4)) = *(int4*)(S + (((kp * 512) + (m_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
      }
      #pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        *(int4*)(b + 0) = *(int4*)(WX + ((((((((int)threadIdx.x) >> 5) * 4096) + (kp * 1024)) + (slab_1 * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4)));
        #pragma unroll
        for (int m_3 = 0; m_3 < 4; ++m_3) {
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            uint64_t r = wide_mma(a[(m_3 * 4)], a[((m_3 * 4) + 1)], a[((m_3 * 4) + 2)], a[((m_3 * 4) + 3)], b[(half * 2)], b[((half * 2) + 1)], X[(((m_3 * 8) + (pair * 4)) + (half * 2))], X[((((m_3 * 8) + (pair * 4)) + (half * 2)) + 1)]);
            X[(((m_3 * 8) + (pair * 4)) + (half * 2))] = ((uint)r);
            X[((((m_3 * 8) + (pair * 4)) + (half * 2)) + 1)] = ((uint)(r >> (uint64_t)32));
          }
        }
      }
    }
    #pragma unroll
    for (int m_4 = 0; m_4 < 4; ++m_4) {
      *(int4*)(a_1 + 0) = *(int4*)(S + (((m_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1536));
      #pragma unroll
      for (int pair_1 = 0; pair_1 < 2; ++pair_1) {
        *(int4*)(b_1 + 0) = *(int4*)(WX + ((((((((int)threadIdx.x) >> 5) * 4096) + (slab_1 * 256)) + (pair_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3072));
        #pragma unroll
        for (int half_1 = 0; half_1 < 2; ++half_1) {
          uint64_t r_1 = wide_mma(a_1[0], a_1[1], a_1[2], a_1[3], b_1[(half_1 * 2)], b_1[((half_1 * 2) + 1)], X[(((m_4 * 8) + (pair_1 * 4)) + (half_1 * 2))], X[((((m_4 * 8) + (pair_1 * 4)) + (half_1 * 2)) + 1)]);
          X[(((m_4 * 8) + (pair_1 * 4)) + (half_1 * 2))] = ((uint)r_1);
          X[((((m_4 * 8) + (pair_1 * 4)) + (half_1 * 2)) + 1)] = ((uint)(r_1 >> (uint64_t)32));
        }
      }
      #pragma unroll
      for (int n_2 = 0; n_2 < 4; ++n_2) {
        P[n_2] = wide_encode(wide_mul(X[((m_4 * 8) + (n_2 * 2))], wide_fma(wide_min(wide_max(X[((m_4 * 8) + (n_2 * 2))], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/)), wide_fma(wide_abs(wide_min(wide_max(X[((m_4 * 8) + (n_2 * 2))], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/))), wide_splat(-0x1.cap-5f/*-5.590820e-02*/), wide_splat(0x1.cap-2f/*4.472656e-01*/)), wide_splat(0x1.cap-1f/*8.945312e-01*/))), wide_mul(X[(((m_4 * 8) + (n_2 * 2)) + 1)], wide_fma(wide_min(wide_max(X[(((m_4 * 8) + (n_2 * 2)) + 1)], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/)), wide_fma(wide_abs(wide_min(wide_max(X[(((m_4 * 8) + (n_2 * 2)) + 1)], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/))), wide_splat(-0x1.cap-5f/*-5.590820e-02*/), wide_splat(0x1.cap-2f/*4.472656e-01*/)), wide_splat(0x1.cap-1f/*8.945312e-01*/))));
      }
      #pragma unroll
      for (int word_1 = 0; word_1 < 4; ++word_1) {
        v_1 = P[((((((((((((0 | ((word_1 * 128) & 1)) | ((((word_1 * 128) >> 1) & 1) << 7)) | ((((word_1 * 128) >> 2) & 1) << 2)) | ((((word_1 * 128) >> 3) & 1) << 3)) | ((((word_1 * 128) >> 4) & 1) << 4)) | ((((word_1 * 128) >> 5) & 1) << 5)) | ((((word_1 * 128) >> 6) & 1) << 6)) | ((((word_1 * 128) >> 7) & 1) << 1)) | ((((word_1 * 128) >> 8) & 1) << 8)) | ((((word_1 * 128) >> 9) & 1) << 9)) | ((((word_1 * 128) >> 10) & 1) << 10)) >> 7)];
        v_2 = P[((((((((((((0 | (((word_1 * 128) + 2) & 1)) | (((((word_1 * 128) + 2) >> 1) & 1) << 7)) | (((((word_1 * 128) + 2) >> 2) & 1) << 2)) | (((((word_1 * 128) + 2) >> 3) & 1) << 3)) | (((((word_1 * 128) + 2) >> 4) & 1) << 4)) | (((((word_1 * 128) + 2) >> 5) & 1) << 5)) | (((((word_1 * 128) + 2) >> 6) & 1) << 6)) | (((((word_1 * 128) + 2) >> 7) & 1) << 1)) | (((((word_1 * 128) + 2) >> 8) & 1) << 8)) | (((((word_1 * 128) + 2) >> 9) & 1) << 9)) | (((((word_1 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
        v_3 = wide_packet_prmt(v_1, v_2, (((((uint)0 | (((uint)((((((((((((0 | (((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
        A[word_1] = v_3;
      }
      #pragma unroll
      for (int n_3 = 0; n_3 < 4; ++n_3) {
        uint64_t r_2 = wide_mma(A[0], A[1], A[2], A[3], WR[(((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16384)], WR[(((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16385)], C[((m_4 * 8) + (n_3 * 2))], C[(((m_4 * 8) + (n_3 * 2)) + 1)]);
        C[((m_4 * 8) + (n_3 * 2))] = ((uint)r_2);
        C[(((m_4 * 8) + (n_3 * 2)) + 1)] = ((uint)(r_2 >> (uint64_t)32));
      }
    }
  }
  #pragma unroll
  for (int m_5 = 0; m_5 < 4; ++m_5) {
    #pragma unroll
    for (int n_4 = 0; n_4 < 4; ++n_4) {
      P[((m_5 * 4) + n_4)] = wide_encode(C[((m_5 * 8) + (n_4 * 2))], C[(((m_5 * 8) + (n_4 * 2)) + 1)]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int word_2 = 0; word_2 < 16; ++word_2) {
    v_4 = P[((((((((((((0 | ((word_2 * 128) & 1)) | ((((word_2 * 128) >> 1) & 1) << 7)) | ((((word_2 * 128) >> 2) & 1) << 2)) | ((((word_2 * 128) >> 3) & 1) << 3)) | ((((word_2 * 128) >> 4) & 1) << 4)) | ((((word_2 * 128) >> 5) & 1) << 5)) | ((((word_2 * 128) >> 6) & 1) << 6)) | ((((word_2 * 128) >> 7) & 1) << 1)) | ((((word_2 * 128) >> 8) & 1) << 8)) | ((((word_2 * 128) >> 9) & 1) << 9)) | ((((word_2 * 128) >> 10) & 1) << 10)) >> 7)];
    v_5 = P[((((((((((((0 | (((word_2 * 128) + 2) & 1)) | (((((word_2 * 128) + 2) >> 1) & 1) << 7)) | (((((word_2 * 128) + 2) >> 2) & 1) << 2)) | (((((word_2 * 128) + 2) >> 3) & 1) << 3)) | (((((word_2 * 128) + 2) >> 4) & 1) << 4)) | (((((word_2 * 128) + 2) >> 5) & 1) << 5)) | (((((word_2 * 128) + 2) >> 6) & 1) << 6)) | (((((word_2 * 128) + 2) >> 7) & 1) << 1)) | (((((word_2 * 128) + 2) >> 8) & 1) << 8)) | (((((word_2 * 128) + 2) >> 9) & 1) << 9)) | (((((word_2 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
    v_6 = wide_packet_prmt(v_4, v_5, (((((uint)0 | (((uint)((((((((((((0 | (((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    uint _reinterpret_tmp = v_6;
    S[(((((((int)threadIdx.x) >> 5) * 512) + ((word_2 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_2 & 3))] = (*(int *)(&(_reinterpret_tmp)));
  }
  __syncthreads();
  #pragma unroll
  for (int m_6 = 0; m_6 < 4; ++m_6) {
    #pragma unroll
    for (int n_5 = 0; n_5 < 4; ++n_5) {
      int g = FG[(((((((int)threadIdx.x) >> 5) * 16) + (n_5 * 4)) + (((int)threadIdx.x) & 3)) + 24580)];
      C[((m_6 * 8) + (n_5 * 2))] = wide_mul(wide_decode(Residual[((m_6 * 4) + n_5)]), g);
      C[(((m_6 * 8) + (n_5 * 2)) + 1)] = wide_mul(wide_decode((Residual[((m_6 * 4) + n_5)] >> (uint)16)), g);
    }
  }
  for (int kp_1 = 0; kp_1 < 4; ++kp_1) {
    #pragma unroll
    for (int m_7 = 0; m_7 < 4; ++m_7) {
      *(int4*)(a_2 + (m_7 * 4)) = *(int4*)(S + (((kp_1 * 512) + (m_7 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int pair_2 = 0; pair_2 < 2; ++pair_2) {
      *(int4*)(b_2 + 0) = *(int4*)(WT + (((((kp_1 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 20480));
      #pragma unroll
      for (int m_8 = 0; m_8 < 4; ++m_8) {
        #pragma unroll
        for (int half_2 = 0; half_2 < 2; ++half_2) {
          uint64_t r_3 = wide_mma(a_2[(m_8 * 4)], a_2[((m_8 * 4) + 1)], a_2[((m_8 * 4) + 2)], a_2[((m_8 * 4) + 3)], b_2[(half_2 * 2)], b_2[((half_2 * 2) + 1)], C[(((m_8 * 8) + (pair_2 * 4)) + (half_2 * 2))], C[((((m_8 * 8) + (pair_2 * 4)) + (half_2 * 2)) + 1)]);
          C[(((m_8 * 8) + (pair_2 * 4)) + (half_2 * 2))] = ((uint)r_3);
          C[((((m_8 * 8) + (pair_2 * 4)) + (half_2 * 2)) + 1)] = ((uint)(r_3 >> (uint64_t)32));
        }
      }
    }
  }
  #pragma unroll
  for (int m_9 = 0; m_9 < 4; ++m_9) {
    #pragma unroll
    for (int n_6 = 0; n_6 < 4; ++n_6) {
      P[((m_9 * 4) + n_6)] = wide_encode(C[((m_9 * 8) + (n_6 * 2))], C[(((m_9 * 8) + (n_6 * 2)) + 1)]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int word_3 = 0; word_3 < 16; ++word_3) {
    v_7 = P[((((((((((((0 | ((word_3 * 128) & 1)) | ((((word_3 * 128) >> 1) & 1) << 7)) | ((((word_3 * 128) >> 2) & 1) << 2)) | ((((word_3 * 128) >> 3) & 1) << 3)) | ((((word_3 * 128) >> 4) & 1) << 4)) | ((((word_3 * 128) >> 5) & 1) << 5)) | ((((word_3 * 128) >> 6) & 1) << 6)) | ((((word_3 * 128) >> 7) & 1) << 1)) | ((((word_3 * 128) >> 8) & 1) << 8)) | ((((word_3 * 128) >> 9) & 1) << 9)) | ((((word_3 * 128) >> 10) & 1) << 10)) >> 7)];
    v_8 = P[((((((((((((0 | (((word_3 * 128) + 2) & 1)) | (((((word_3 * 128) + 2) >> 1) & 1) << 7)) | (((((word_3 * 128) + 2) >> 2) & 1) << 2)) | (((((word_3 * 128) + 2) >> 3) & 1) << 3)) | (((((word_3 * 128) + 2) >> 4) & 1) << 4)) | (((((word_3 * 128) + 2) >> 5) & 1) << 5)) | (((((word_3 * 128) + 2) >> 6) & 1) << 6)) | (((((word_3 * 128) + 2) >> 7) & 1) << 1)) | (((((word_3 * 128) + 2) >> 8) & 1) << 8)) | (((((word_3 * 128) + 2) >> 9) & 1) << 9)) | (((((word_3 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
    v_9 = wide_packet_prmt(v_7, v_8, (((((uint)0 | (((uint)((((((((((((0 | (((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    uint _reinterpret_tmp_1 = v_9;
    S[(((((((int)threadIdx.x) >> 5) * 512) + ((word_3 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_3 & 3))] = (*(int *)(&(_reinterpret_tmp_1)));
  }
  __syncthreads();
  #pragma unroll
  for (int m_10 = 0; m_10 < 4; ++m_10) {
    #pragma unroll
    for (int n_7 = 0; n_7 < 12; ++n_7) {
      Z[((m_10 * 24) + (n_7 * 2))] = (uint)0;
      Z[(((m_10 * 24) + (n_7 * 2)) + 1)] = (uint)0;
    }
  }
  for (int kp_2 = 0; kp_2 < 3; ++kp_2) {
    #pragma unroll
    for (int m_11 = 0; m_11 < 4; ++m_11) {
      *(int4*)(a_3 + (m_11 * 4)) = *(int4*)(S + (((kp_2 * 512) + (m_11 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int pair_3 = 0; pair_3 < 6; ++pair_3) {
      *(int4*)(b_3 + 0) = *(int4*)(WQ + (((((kp_2 * 3072) + ((((int)threadIdx.x) >> 5) * 768)) + (pair_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 24648));
      #pragma unroll
      for (int m_12 = 0; m_12 < 4; ++m_12) {
        #pragma unroll
        for (int half_3 = 0; half_3 < 2; ++half_3) {
          uint64_t r_4 = wide_mma(a_3[(m_12 * 4)], a_3[((m_12 * 4) + 1)], a_3[((m_12 * 4) + 2)], a_3[((m_12 * 4) + 3)], b_3[(half_3 * 2)], b_3[((half_3 * 2) + 1)], Z[(((m_12 * 24) + (pair_3 * 4)) + (half_3 * 2))], Z[((((m_12 * 24) + (pair_3 * 4)) + (half_3 * 2)) + 1)]);
          Z[(((m_12 * 24) + (pair_3 * 4)) + (half_3 * 2))] = ((uint)r_4);
          Z[((((m_12 * 24) + (pair_3 * 4)) + (half_3 * 2)) + 1)] = ((uint)(r_4 >> (uint64_t)32));
        }
      }
    }
  }
  #pragma unroll
  for (int comp = 0; comp < 3; ++comp) {
    #pragma unroll
    for (int dest = 0; dest < 4; ++dest) {
      *(int4*)(a_4 + 0) = *(int4*)(S + (((dest * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1536));
      #pragma unroll
      for (int pair_4 = 0; pair_4 < 2; ++pair_4) {
        *(int4*)(b_4 + 0) = *(int4*)(WQ + ((((((((int)threadIdx.x) >> 5) * 768) + (comp * 256)) + (pair_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 33864));
        #pragma unroll
        for (int half_4 = 0; half_4 < 2; ++half_4) {
          uint64_t r_5 = wide_mma(a_4[0], a_4[1], a_4[2], a_4[3], b_4[(half_4 * 2)], b_4[((half_4 * 2) + 1)], Z[((((dest * 24) + (comp * 8)) + (pair_4 * 4)) + (half_4 * 2))], Z[(((((dest * 24) + (comp * 8)) + (pair_4 * 4)) + (half_4 * 2)) + 1)]);
          Z[((((dest * 24) + (comp * 8)) + (pair_4 * 4)) + (half_4 * 2))] = ((uint)r_5);
          Z[(((((dest * 24) + (comp * 8)) + (pair_4 * 4)) + (half_4 * 2)) + 1)] = ((uint)(r_5 >> (uint64_t)32));
        }
      }
      if (comp == 0) {
        #pragma unroll
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          uint a_6 = Z[(((dest * 24) + (comp * 8)) + j_1)];
          uint b_6 = Z[((((dest * 24) + (comp * 8)) + j_1) + 2)];
          uint c = Z[((((dest * 24) + (comp * 8)) + j_1) + 4)];
          uint d = Z[((((dest * 24) + (comp * 8)) + j_1) + 6)];
          uint s0 = wide_add(wide_fma(a_6, a_6, wide_mul(c, c)), wide_fma(b_6, b_6, wide_mul(d, d)));
          uint s1 = wide_add(s0, wide_shfl(s0, ((((int)threadIdx.x) & 31) ^ 2)));
          uint s2 = wide_add(s1, wide_shfl(s1, ((((int)threadIdx.x) & 31) ^ 1)));
          uint total = wide_max(wide_add(s2, ((s2 >> (uint)16) | (s2 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
          uint inv = wide_rsqrt(total);
          #pragma unroll
          for (int n_8 = 0; n_8 < 4; ++n_8) {
            Z[((((dest * 24) + (comp * 8)) + (n_8 * 2)) + j_1)] = wide_mul(Z[((((dest * 24) + (comp * 8)) + (n_8 * 2)) + j_1)], inv);
            int v__3 = WS[((((int)threadIdx.x) >> 5) + 45128)];
            Z[(((dest * 24) + (n_8 * 2)) + j_1)] = wide_mul(Z[(((dest * 24) + (n_8 * 2)) + j_1)], wide_splat((*(float *)(&(v__3)))));
          }
        }
        #pragma unroll
        for (int n_9 = 0; n_9 < 4; ++n_9) {
          QP[((dest * 4) + n_9)] = wide_encode(Z[(((dest * 24) + (comp * 8)) + (n_9 * 2))], Z[((((dest * 24) + (comp * 8)) + (n_9 * 2)) + 1)]);
        }
      } else {
        if (comp == 1) {
          #pragma unroll
          for (int j_2 = 0; j_2 < 2; ++j_2) {
            uint a_7 = Z[(((dest * 24) + (comp * 8)) + j_2)];
            uint b_7 = Z[((((dest * 24) + (comp * 8)) + j_2) + 2)];
            uint c_1 = Z[((((dest * 24) + (comp * 8)) + j_2) + 4)];
            uint d_1 = Z[((((dest * 24) + (comp * 8)) + j_2) + 6)];
            uint s0_1 = wide_add(wide_fma(a_7, a_7, wide_mul(c_1, c_1)), wide_fma(b_7, b_7, wide_mul(d_1, d_1)));
            uint s1_1 = wide_add(s0_1, wide_shfl(s0_1, ((((int)threadIdx.x) & 31) ^ 2)));
            uint s2_1 = wide_add(s1_1, wide_shfl(s1_1, ((((int)threadIdx.x) & 31) ^ 1)));
            uint total_1 = wide_max(wide_add(s2_1, ((s2_1 >> (uint)16) | (s2_1 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
            uint inv_1 = wide_rsqrt(total_1);
            #pragma unroll
            for (int n_10 = 0; n_10 < 4; ++n_10) {
              Z[((((dest * 24) + (comp * 8)) + (n_10 * 2)) + j_2)] = wide_mul(Z[((((dest * 24) + (comp * 8)) + (n_10 * 2)) + j_2)], inv_1);
            }
          }
          #pragma unroll
          for (int n_11 = 0; n_11 < 4; ++n_11) {
            KP[((dest * 4) + n_11)] = wide_encode(Z[(((dest * 24) + (comp * 8)) + (n_11 * 2))], Z[((((dest * 24) + (comp * 8)) + (n_11 * 2)) + 1)]);
          }
        } else {
          #pragma unroll
          for (int n_12 = 0; n_12 < 4; ++n_12) {
            VP[((dest * 4) + n_12)] = wide_encode(Z[(((dest * 24) + (comp * 8)) + (n_12 * 2))], Z[((((dest * 24) + (comp * 8)) + (n_12 * 2)) + 1)]);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int word_4 = 0; word_4 < 16; ++word_4) {
    v_10 = wide_shfl(QP[((((((((((((0 | ((word_4 * 128) & 1)) | ((((word_4 * 128) >> 1) & 1) << 7)) | ((((word_4 * 128) >> 2) & 1) << 2)) | ((((word_4 * 128) >> 3) & 1) << 3)) | ((((word_4 * 128) >> 4) & 1) << 5)) | ((((word_4 * 128) >> 5) & 1) << 6)) | ((((word_4 * 128) >> 6) & 1) << 4)) | ((((word_4 * 128) >> 7) & 1) << 9)) | ((((word_4 * 128) >> 8) & 1) << 8)) | ((((word_4 * 128) >> 9) & 1) << 1)) | ((((word_4 * 128) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 127) >> 2));
    v_11 = wide_shfl(QP[((((((((((((0 | (((word_4 * 128) + 2) & 1)) | (((((word_4 * 128) + 2) >> 1) & 1) << 7)) | (((((word_4 * 128) + 2) >> 2) & 1) << 2)) | (((((word_4 * 128) + 2) >> 3) & 1) << 3)) | (((((word_4 * 128) + 2) >> 4) & 1) << 5)) | (((((word_4 * 128) + 2) >> 5) & 1) << 6)) | (((((word_4 * 128) + 2) >> 6) & 1) << 4)) | (((((word_4 * 128) + 2) >> 7) & 1) << 9)) | (((((word_4 * 128) + 2) >> 8) & 1) << 8)) | (((((word_4 * 128) + 2) >> 9) & 1) << 1)) | (((((word_4 * 128) + 2) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 127) >> 2));
    v_12 = wide_packet_prmt(v_10, v_11, (((((uint)0 | (((uint)((((((((((((0 | (((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    Q[word_4] = v_12;
    v_13 = wide_shfl(KP[((((((((((((0 | ((word_4 * 128) & 1)) | ((((word_4 * 128) >> 1) & 1) << 7)) | ((((word_4 * 128) >> 2) & 1) << 2)) | ((((word_4 * 128) >> 3) & 1) << 3)) | ((((word_4 * 128) >> 4) & 1) << 5)) | ((((word_4 * 128) >> 5) & 1) << 6)) | ((((word_4 * 128) >> 6) & 1) << 4)) | ((((word_4 * 128) >> 7) & 1) << 8)) | ((((word_4 * 128) >> 8) & 1) << 1)) | ((((word_4 * 128) >> 9) & 1) << 9)) | ((((word_4 * 128) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 127) >> 2));
    v_14 = wide_shfl(KP[((((((((((((0 | (((word_4 * 128) + 2) & 1)) | (((((word_4 * 128) + 2) >> 1) & 1) << 7)) | (((((word_4 * 128) + 2) >> 2) & 1) << 2)) | (((((word_4 * 128) + 2) >> 3) & 1) << 3)) | (((((word_4 * 128) + 2) >> 4) & 1) << 5)) | (((((word_4 * 128) + 2) >> 5) & 1) << 6)) | (((((word_4 * 128) + 2) >> 6) & 1) << 4)) | (((((word_4 * 128) + 2) >> 7) & 1) << 8)) | (((((word_4 * 128) + 2) >> 8) & 1) << 1)) | (((((word_4 * 128) + 2) >> 9) & 1) << 9)) | (((((word_4 * 128) + 2) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 127) >> 2));
    v_15 = wide_packet_prmt(v_13, v_14, (((((uint)0 | (((uint)((((((((((((0 | (((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    K[word_4] = v_15;
    v_16 = wide_shfl(VP[((((((((((((0 | (((word_4 * 128) & 1) << 5)) | ((((word_4 * 128) >> 1) & 1) << 9)) | ((((word_4 * 128) >> 2) & 1) << 6)) | ((((word_4 * 128) >> 3) & 1) << 4)) | (((word_4 * 128) >> 4) & 1)) | ((((word_4 * 128) >> 5) & 1) << 2)) | ((((word_4 * 128) >> 6) & 1) << 3)) | ((((word_4 * 128) >> 7) & 1) << 1)) | ((((word_4 * 128) >> 8) & 1) << 7)) | ((((word_4 * 128) >> 9) & 1) << 8)) | ((((word_4 * 128) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 4)) | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 127) >> 2));
    v_17 = wide_shfl(VP[((((((((((((0 | ((((word_4 * 128) + 1) & 1) << 5)) | (((((word_4 * 128) + 1) >> 1) & 1) << 9)) | (((((word_4 * 128) + 1) >> 2) & 1) << 6)) | (((((word_4 * 128) + 1) >> 3) & 1) << 4)) | ((((word_4 * 128) + 1) >> 4) & 1)) | (((((word_4 * 128) + 1) >> 5) & 1) << 2)) | (((((word_4 * 128) + 1) >> 6) & 1) << 3)) | (((((word_4 * 128) + 1) >> 7) & 1) << 1)) | (((((word_4 * 128) + 1) >> 8) & 1) << 7)) | (((((word_4 * 128) + 1) >> 9) & 1) << 8)) | (((((word_4 * 128) + 1) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 127) >> 2));
    v_18 = wide_shfl(VP[((((((((((((0 | ((((word_4 * 128) + 2) & 1) << 5)) | (((((word_4 * 128) + 2) >> 1) & 1) << 9)) | (((((word_4 * 128) + 2) >> 2) & 1) << 6)) | (((((word_4 * 128) + 2) >> 3) & 1) << 4)) | ((((word_4 * 128) + 2) >> 4) & 1)) | (((((word_4 * 128) + 2) >> 5) & 1) << 2)) | (((((word_4 * 128) + 2) >> 6) & 1) << 3)) | (((((word_4 * 128) + 2) >> 7) & 1) << 1)) | (((((word_4 * 128) + 2) >> 8) & 1) << 7)) | (((((word_4 * 128) + 2) >> 9) & 1) << 8)) | (((((word_4 * 128) + 2) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 127) >> 2));
    v_19 = wide_shfl(VP[((((((((((((0 | ((((word_4 * 128) + 3) & 1) << 5)) | (((((word_4 * 128) + 3) >> 1) & 1) << 9)) | (((((word_4 * 128) + 3) >> 2) & 1) << 6)) | (((((word_4 * 128) + 3) >> 3) & 1) << 4)) | ((((word_4 * 128) + 3) >> 4) & 1)) | (((((word_4 * 128) + 3) >> 5) & 1) << 2)) | (((((word_4 * 128) + 3) >> 6) & 1) << 3)) | (((((word_4 * 128) + 3) >> 7) & 1) << 1)) | (((((word_4 * 128) + 3) >> 8) & 1) << 7)) | (((((word_4 * 128) + 3) >> 9) & 1) << 8)) | (((((word_4 * 128) + 3) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 127) >> 2));
    v_20 = wide_packet_prmt(v_16, v_17, (((uint)0 | (((uint)((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 9)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 4)) | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)(((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3) + 4)) << (uint)4)));
    v_21 = wide_packet_prmt(v_18, v_19, (((uint)0 | (((uint)((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3)) << (uint)8)) | (((uint)(((((((((((((0 | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 9)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    V[word_4] = wide_packet_prmt(v_20, v_21, 30224);
  }
  __syncthreads();
  for (int slab_2 = 0; slab_2 < 2; ++slab_2) {
    #pragma unroll
    for (int m_13 = 0; m_13 < 2; ++m_13) {
      #pragma unroll
      for (int n_13 = 0; n_13 < 8; ++n_13) {
        uint64_t r_6 = wide_mma(Q[((slab_2 * 8) + (m_13 * 4))], Q[(((slab_2 * 8) + (m_13 * 4)) + 1)], Q[(((slab_2 * 8) + (m_13 * 4)) + 2)], Q[(((slab_2 * 8) + (m_13 * 4)) + 3)], K[(n_13 * 2)], K[((n_13 * 2) + 1)], WB[(((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36936)], WB[(((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36937)]);
        L[((m_13 * 16) + (n_13 * 2))] = (((wide_min(wide_max(wide_fma(((uint)r_6), wide_splat(0x1.7p-5f/*4.492188e-02*/), wide_splat(0x1.4dp+0f/*1.300781e+00*/)), wide_splat(0x1.08p+0f/*1.031250e+00*/)), wide_splat(0x1.91cp+0f/*1.569336e+00*/)) << (uint)5) & (uint)4292935648) ^ (uint)2147516416);
        L[(((m_13 * 16) + (n_13 * 2)) + 1)] = (((wide_min(wide_max(wide_fma(((uint)(r_6 >> (uint64_t)32)), wide_splat(0x1.7p-5f/*4.492188e-02*/), wide_splat(0x1.4dp+0f/*1.300781e+00*/)), wide_splat(0x1.08p+0f/*1.031250e+00*/)), wide_splat(0x1.91cp+0f/*1.569336e+00*/)) << (uint)5) & (uint)4292935648) ^ (uint)2147516416);
      }
      #pragma unroll
      for (int j_3 = 0; j_3 < 2; ++j_3) {
        uint g0 = wide_add(L[((m_13 * 16) + j_3)], L[(((m_13 * 16) + j_3) + 4)]);
        uint g1 = wide_add(g0, wide_add(L[(((m_13 * 16) + j_3) + 2)], L[(((m_13 * 16) + j_3) + 6)]));
        uint g2 = wide_add(g1, wide_add(L[(((m_13 * 16) + j_3) + 8)], L[(((m_13 * 16) + j_3) + 12)]));
        uint group = wide_add(g2, wide_add(L[(((m_13 * 16) + j_3) + 10)], L[(((m_13 * 16) + j_3) + 14)]));
        uint t0 = wide_shfl(group, ((((int)threadIdx.x) & 31) & -4));
        uint t1 = wide_add(t0, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 1)));
        uint t2 = wide_add(t1, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 2)));
        uint t3 = wide_add(t2, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 3)));
        uint total_2 = wide_max(wide_add(t3, ((t3 >> (uint)16) | (t3 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
        uint inv_2 = wide_rcp(total_2);
        #pragma unroll
        for (int n_14 = 0; n_14 < 8; ++n_14) {
          L[(((m_13 * 16) + (n_14 * 2)) + j_3)] = wide_mul(L[(((m_13 * 16) + (n_14 * 2)) + j_3)], inv_2);
        }
      }
      #pragma unroll
      for (int n_15 = 0; n_15 < 8; ++n_15) {
        P[((m_13 * 8) + n_15)] = wide_encode(L[((m_13 * 16) + (n_15 * 2))], L[(((m_13 * 16) + (n_15 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int word_5 = 0; word_5 < 16; ++word_5) {
      v_22 = wide_shfl(P[((((((((((((0 | ((word_5 * 128) & 1)) | ((((word_5 * 128) >> 1) & 1) << 8)) | ((((word_5 * 128) >> 2) & 1) << 2)) | ((((word_5 * 128) >> 3) & 1) << 3)) | ((((word_5 * 128) >> 4) & 1) << 6)) | ((((word_5 * 128) >> 5) & 1) << 4)) | ((((word_5 * 128) >> 6) & 1) << 5)) | ((((word_5 * 128) >> 7) & 1) << 10)) | ((((word_5 * 128) >> 8) & 1) << 7)) | ((((word_5 * 128) >> 9) & 1) << 1)) | ((((word_5 * 128) >> 10) & 1) << 9)) >> 7)], (((((((((((((0 | (((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 8)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 6)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 4)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 5)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 10)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 1)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 9)) & 127) >> 2));
      v_23 = wide_shfl(P[((((((((((((0 | (((word_5 * 128) + 2) & 1)) | (((((word_5 * 128) + 2) >> 1) & 1) << 8)) | (((((word_5 * 128) + 2) >> 2) & 1) << 2)) | (((((word_5 * 128) + 2) >> 3) & 1) << 3)) | (((((word_5 * 128) + 2) >> 4) & 1) << 6)) | (((((word_5 * 128) + 2) >> 5) & 1) << 4)) | (((((word_5 * 128) + 2) >> 6) & 1) << 5)) | (((((word_5 * 128) + 2) >> 7) & 1) << 10)) | (((((word_5 * 128) + 2) >> 8) & 1) << 7)) | (((((word_5 * 128) + 2) >> 9) & 1) << 1)) | (((((word_5 * 128) + 2) >> 10) & 1) << 9)) >> 7)], (((((((((((((0 | ((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 8)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 6)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 4)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 5)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 10)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 7)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 9)) & 127) >> 2));
      v_24 = wide_packet_prmt(v_22, v_23, (((((uint)0 | (((uint)((((((((((((0 | (((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 8)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 6)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 4)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 5)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 10)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 1)) | (((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 9)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 8)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 6)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 4)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 5)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 10)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 7)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 9)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 8)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 6)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 4)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 5)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 10)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 7)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 9)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 8)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 6)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 4)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 5)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 10)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 7)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 1)) | ((((((word_5 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 9)) & 3) + 4)) << (uint)12)));
      A[word_5] = v_24;
    }
    #pragma unroll
    for (int m_14 = 0; m_14 < 2; ++m_14) {
      #pragma unroll
      for (int n_16 = 0; n_16 < 4; ++n_16) {
        acc[0] = (uint)0;
        acc[1] = (uint)0;
        #pragma unroll
        for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
          uint64_t r_7 = wide_mma(A[((kp_3 * 8) + (m_14 * 4))], A[(((kp_3 * 8) + (m_14 * 4)) + 1)], A[(((kp_3 * 8) + (m_14 * 4)) + 2)], A[(((kp_3 * 8) + (m_14 * 4)) + 3)], V[((kp_3 * 8) + (n_16 * 2))], V[(((kp_3 * 8) + (n_16 * 2)) + 1)], acc[0], acc[1]);
          acc[0] = ((uint)r_7);
          acc[1] = ((uint)(r_7 >> (uint64_t)32));
        }
        P[((m_14 * 4) + n_16)] = wide_encode(acc[0], acc[1]);
      }
    }
    #pragma unroll
    for (int word_6 = 0; word_6 < 8; ++word_6) {
      int v__4 = S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_6 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_6 & 3) >> 1) * 2))];
      int v__5 = S[(((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_6 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_6 & 3) >> 1) * 2)) + 1)];
      int condval_2;
      if (((word_6 % 2) == 0)) {
        condval_2 = 21520;
      } else {
        condval_2 = 30258;
      }
      Residual[word_6] = wide_packet_prmt((*(uint *)(&(v__4))), (*(uint *)(&(v__5))), condval_2);
    }
    __syncthreads();
    #pragma unroll
    for (int word_7 = 0; word_7 < 8; ++word_7) {
      v_25 = P[((((((((((((0 | ((word_7 * 128) & 1)) | ((((word_7 * 128) >> 1) & 1) << 7)) | ((((word_7 * 128) >> 2) & 1) << 2)) | ((((word_7 * 128) >> 3) & 1) << 3)) | ((((word_7 * 128) >> 4) & 1) << 4)) | ((((word_7 * 128) >> 5) & 1) << 5)) | ((((word_7 * 128) >> 6) & 1) << 6)) | ((((word_7 * 128) >> 7) & 1) << 1)) | ((((word_7 * 128) >> 8) & 1) << 8)) | ((((word_7 * 128) >> 9) & 1) << 9)) | ((((word_7 * 128) >> 10) & 1) << 10)) >> 7)];
      v_26 = P[((((((((((((0 | (((word_7 * 128) + 2) & 1)) | (((((word_7 * 128) + 2) >> 1) & 1) << 7)) | (((((word_7 * 128) + 2) >> 2) & 1) << 2)) | (((((word_7 * 128) + 2) >> 3) & 1) << 3)) | (((((word_7 * 128) + 2) >> 4) & 1) << 4)) | (((((word_7 * 128) + 2) >> 5) & 1) << 5)) | (((((word_7 * 128) + 2) >> 6) & 1) << 6)) | (((((word_7 * 128) + 2) >> 7) & 1) << 1)) | (((((word_7 * 128) + 2) >> 8) & 1) << 8)) | (((((word_7 * 128) + 2) >> 9) & 1) << 9)) | (((((word_7 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
      v_27 = wide_packet_prmt(v_25, v_26, (((((uint)0 | (((uint)((((((((((((0 | (((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
      uint _reinterpret_tmp_2 = v_27;
      S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_7 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_7 & 3))] = (*(int *)(&(_reinterpret_tmp_2)));
    }
    __syncthreads();
    #pragma unroll
    for (int m_15 = 0; m_15 < 2; ++m_15) {
      #pragma unroll
      for (int n_17 = 0; n_17 < 4; ++n_17) {
        int g_1 = AG[(((((((int)threadIdx.x) >> 5) * 16) + (n_17 * 4)) + (((int)threadIdx.x) & 3)) + 49228)];
        C[((m_15 * 8) + (n_17 * 2))] = wide_mul(wide_decode(Residual[((m_15 * 4) + n_17)]), g_1);
        C[(((m_15 * 8) + (n_17 * 2)) + 1)] = wide_mul(wide_decode((Residual[((m_15 * 4) + n_17)] >> (uint)16)), g_1);
      }
    }
    for (int kp_4 = 0; kp_4 < 4; ++kp_4) {
      #pragma unroll
      for (int m_16 = 0; m_16 < 2; ++m_16) {
        *(int4*)(a_5 + (m_16 * 4)) = *(int4*)(S + ((((kp_4 * 512) + (slab_2 * 256)) + (m_16 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
      }
      #pragma unroll
      for (int pair_5 = 0; pair_5 < 2; ++pair_5) {
        *(int4*)(b_5 + 0) = *(int4*)(WP + (((((kp_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_5 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 45132));
        #pragma unroll
        for (int m_17 = 0; m_17 < 2; ++m_17) {
          #pragma unroll
          for (int half_5 = 0; half_5 < 2; ++half_5) {
            uint64_t r_8 = wide_mma(a_5[(m_17 * 4)], a_5[((m_17 * 4) + 1)], a_5[((m_17 * 4) + 2)], a_5[((m_17 * 4) + 3)], b_5[(half_5 * 2)], b_5[((half_5 * 2) + 1)], C[(((m_17 * 8) + (pair_5 * 4)) + (half_5 * 2))], C[((((m_17 * 8) + (pair_5 * 4)) + (half_5 * 2)) + 1)]);
            C[(((m_17 * 8) + (pair_5 * 4)) + (half_5 * 2))] = ((uint)r_8);
            C[((((m_17 * 8) + (pair_5 * 4)) + (half_5 * 2)) + 1)] = ((uint)(r_8 >> (uint64_t)32));
          }
        }
      }
    }
    #pragma unroll
    for (int m_18 = 0; m_18 < 2; ++m_18) {
      #pragma unroll
      for (int n_18 = 0; n_18 < 4; ++n_18) {
        P[((m_18 * 4) + n_18)] = wide_encode(C[((m_18 * 8) + (n_18 * 2))], C[(((m_18 * 8) + (n_18 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int parity_1 = 0; parity_1 < 2; ++parity_1) {
      int condval_3;
      if (((((bool)1 & (bool)1) & (bool)1) & (bool)1)) {
        condval_3 = ((((((((((int)blockIdx.x) / 10) * 81920) + (((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) >> 1) * 40960)) + ((((int)blockIdx.x) % 10) * 4096)) + (((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7) >> 2) * 2048)) + ((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 2)) & 1) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 2)) & 6) << 3)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 2)) & 8) >> 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 2)) & 16) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 2)) & 224) << 4))) + ((((((((((((int)blockIdx.x) / 10) * 8) + ((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) >> 1) & 1) | (((((((int)blockIdx.x) % 10) * 8) + (((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) & 3) << 1)) | ((((((((int)blockIdx.x) / 10) * 8) + ((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) & 1) << 3)) & 1) << 2)) + ((((((((((((int)blockIdx.x) / 10) * 8) + ((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) >> 1) & 1) | (((((((int)blockIdx.x) % 10) * 8) + (((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) & 3) << 1)) | ((((((((int)blockIdx.x) / 10) * 8) + ((((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab_2 * 32) + (parity_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) & 1) << 3)) & 14) << 5));
      } else {
        condval_3 = -1;
      }
      v_28 = condval_3;
      int raw_1 = v_28;
      #pragma unroll
      for (int j_4 = 0; j_4 < 4; ++j_4) {
        int condval_4;
        if ((parity_1 == 0)) {
          condval_4 = 21520;
        } else {
          condval_4 = 30258;
        }
        temp[j_4] = wide_packet_prmt(P[(((j_4 & 1) * 4) + ((j_4 >> 1) * 2))], P[((((j_4 & 1) * 4) + ((j_4 >> 1) * 2)) + 1)], condval_4);
      }
      if ((0 <= raw_1) & (raw_1 <= 18578832)) {
        {
          int RW_local_cast_1[4];
          int4 __2;
          uint4 v__6 = *(uint4*)(temp + 0);
          __2.x = (int)(v__6.x);
          __2.y = (int)(v__6.y);
          __2.z = (int)(v__6.z);
          __2.w = (int)(v__6.w);
          *(int4*)(RW_local_cast_1 + 0) = __2;
          *(int4*)(RW + (((raw_1 >> 4) * 4) + 10444800)) = *(int4*)(RW_local_cast_1 + 0);
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  __threadfence();
  if (((int)threadIdx.x) == 0) {
    RW[(((int)blockIdx.x) + 15082176)] = 0;
  }
}

