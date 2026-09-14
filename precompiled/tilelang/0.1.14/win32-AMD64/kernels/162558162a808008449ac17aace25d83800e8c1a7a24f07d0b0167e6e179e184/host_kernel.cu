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

extern "C" __global__ void kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX, int counter, int gx, int height, int inp, int out, int rn, int sn, int sx, int sy, int tiles, int width, int wns_0, int wns_1, int wns_2, int wns_3, int wns_4, int wns_5, int wns_6, int wns_7, int wns_8);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX, int counter, int gx, int height, int inp, int out, int rn, int sn, int sx, int sy, int tiles, int width, int wns_0, int wns_1, int wns_2, int wns_3, int wns_4, int wns_5, int wns_6, int wns_7, int wns_8) {
  extern __shared__ __align__(1024) int S[];
  int v = 0;
  uint temp[4];
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
  uint v_29 = (uint)0;
  uint v_30 = (uint)0;
  uint v_31 = (uint)0;
  uint v_32 = (uint)0;
  int v_33 = 0;
  uint v_34 = (uint)0;
  uint v_35 = (uint)0;
  uint v_36 = (uint)0;
  uint v_37 = (uint)0;
  int v_38 = 0;
  uint v_39 = (uint)0;
  uint v_40 = (uint)0;
  uint v_41 = (uint)0;
  uint v_42 = (uint)0;
  int v_43 = 0;
  uint v_44 = (uint)0;
  uint v_45 = (uint)0;
  uint v_46 = (uint)0;
  uint v_47 = (uint)0;
  int v_48 = 0;
  #pragma unroll
  for (int slab = 0; slab < 2; ++slab) {
    #pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
      int rmod = (((int)blockIdx.x) % gx);
      int rmod_1 = (((int)blockIdx.x) % gx);
      int rmod_2 = (((int)blockIdx.x) % gx);
      int rdiv = (((int)blockIdx.x) / gx);
      int rmod_3 = (((int)blockIdx.x) % gx);
      int rdiv_1 = (((int)blockIdx.x) / gx);
      int condval;
      if (((((0 <= (((((((0 <= gx) && (0 <= rmod)) || ((gx < 0) && (rmod <= 0))) ? rmod : (rmod + gx)) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx)) & ((((((((0 <= gx) && (0 <= rmod_1)) || ((gx < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + gx)) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx) < width)) & (0 <= ((((((((0 <= gx) && (0 <= rmod_2)) || ((gx < 0) && (rmod_2 <= 0))) ? rdiv : (rdiv - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy))) & (((((((((0 <= gx) && (0 <= rmod_3)) || ((gx < 0) && (rmod_3 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) < height))) {
        int rmod_4 = (((int)blockIdx.x) % gx);
        int rmod_5 = (((int)blockIdx.x) % gx);
        int rdiv_2 = (((int)blockIdx.x) / gx);
        int rmod_6 = (((int)blockIdx.x) % gx);
        int rdiv_3 = (((int)blockIdx.x) / gx);
        int rmod_7 = (((int)blockIdx.x) % gx);
        int rmod_8 = (((int)blockIdx.x) % gx);
        int rdiv_4 = (((int)blockIdx.x) / gx);
        int rmod_9 = (((int)blockIdx.x) % gx);
        int rdiv_5 = (((int)blockIdx.x) / gx);
        int rmod_10 = (((int)blockIdx.x) % gx);
        int rmod_11 = (((int)blockIdx.x) % gx);
        int rdiv_6 = (((int)blockIdx.x) / gx);
        condval = ((((((((((0 <= gx) && (0 <= rmod_4)) || ((gx < 0) && (rmod_4 <= 0))) ? rmod_4 : (rmod_4 + gx)) * 4096) + ((((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7) + sx) >> 2) * 2048)) + (((width >> 2) * ((((((0 <= gx) && (0 <= rmod_5)) || ((gx < 0) && (rmod_5 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) * 2) + ((((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1) + sy) >> 1) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3)) >> 1))) * 2048)) + ((((((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 1) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 6) << 3)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 8) >> 2)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 16) >> 1)) | (((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & -15) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 2) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 4) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 8) >> 1)) & 224) << 4))) + ((((((((((((((0 <= gx) && (0 <= rmod_6)) || ((gx < 0) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) >> 1) & 1) | (((((((((0 <= gx) && (0 <= rmod_7)) || ((gx < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + gx)) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx) & 3) << 1)) | ((((((((((0 <= gx) && (0 <= rmod_8)) || ((gx < 0) && (rmod_8 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) & 1) << 3)) & 1) << 2)) + ((((((((((((((0 <= gx) && (0 <= rmod_9)) || ((gx < 0) && (rmod_9 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) >> 1) & 1) | (((((((((0 <= gx) && (0 <= rmod_10)) || ((gx < 0) && (rmod_10 <= 0))) ? rmod_10 : (rmod_10 + gx)) * 8) + (((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx) & 3) << 1)) | ((((((((((0 <= gx) && (0 <= rmod_11)) || ((gx < 0) && (rmod_11 <= 0))) ? rdiv_6 : (rdiv_6 - 1)) * 8) + ((((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + ((((slab * 32) + (parity * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) & 1) << 3)) & 14) << 5));
      } else {
        condval = -1;
      }
      v = condval;
      int raw = v;
      uint broadcast_var = (uint)0;
      *(uint4*)(temp + 0) = make_uint4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      if ((0 <= raw) & ((raw + 16) <= (sn - inp))) {
        {
          int SW_local_cast[4];
          int broadcast_var_1 = 0;
          int4 condval_1;
          if ((0 <= (inp + raw))) {
            condval_1 = *(int4*)(SW + (((inp + raw) >> 4) * 4));
          } else {
            condval_1 = make_int4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
          }
          *(int4*)(SW_local_cast + 0) = condval_1;
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
    int condval_2;
    if (((word % 2) == 0)) {
      condval_2 = 21520;
    } else {
      condval_2 = 30258;
    }
    Residual[word] = wide_packet_prmt((*(uint *)(&(v__1))), (*(uint *)(&(v__2))), condval_2);
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
        for (int j_s = 0; j_s < 4; ++j_s) {
          int condval_3;
          if (((((((((((int)threadIdx.x) >> 5) * 4096) + (kp * 1024)) + (slab_1 * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s) < wns_0)) {
            condval_3 = WX[(((((((((int)threadIdx.x) >> 5) * 4096) + (kp * 1024)) + (slab_1 * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s)];
          } else {
            condval_3 = 0;
          }
          b[j_s] = condval_3;
        }
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
        for (int j_s_1 = 0; j_s_1 < 4; ++j_s_1) {
          int condval_4;
          if (((((((((((int)threadIdx.x) >> 5) * 4096) + (slab_1 * 256)) + (pair_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_1) + 3072) < wns_0)) {
            condval_4 = WX[(((((((((int)threadIdx.x) >> 5) * 4096) + (slab_1 * 256)) + (pair_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_1) + 3072)];
          } else {
            condval_4 = 0;
          }
          b_1[j_s_1] = condval_4;
        }
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
        int condval_5;
        if (((((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16384) < wns_1)) {
          condval_5 = WR[(((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16384)];
        } else {
          condval_5 = 0;
        }
        int condval_6;
        if (((((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16385) < wns_1)) {
          condval_6 = WR[(((((((((int)threadIdx.x) >> 5) * 1024) + (slab_1 * 256)) + ((n_3 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_3 & 1) * 2)) + 16385)];
        } else {
          condval_6 = 0;
        }
        uint64_t r_2 = wide_mma(A[0], A[1], A[2], A[3], condval_5, condval_6, C[((m_4 * 8) + (n_3 * 2))], C[(((m_4 * 8) + (n_3 * 2)) + 1)]);
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
      int condval_7;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (n_5 * 4)) + (((int)threadIdx.x) & 3)) + 24580) < wns_7)) {
        condval_7 = FG[(((((((int)threadIdx.x) >> 5) * 16) + (n_5 * 4)) + (((int)threadIdx.x) & 3)) + 24580)];
      } else {
        condval_7 = 0;
      }
      int g = condval_7;
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
      for (int j_s_2 = 0; j_s_2 < 4; ++j_s_2) {
        int condval_8;
        if ((((((((kp_1 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_2) + 20480) < wns_2)) {
          condval_8 = WT[((((((kp_1 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_2) + 20480)];
        } else {
          condval_8 = 0;
        }
        b_2[j_s_2] = condval_8;
      }
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
      for (int j_s_3 = 0; j_s_3 < 4; ++j_s_3) {
        int condval_9;
        if ((((((((kp_2 * 3072) + ((((int)threadIdx.x) >> 5) * 768)) + (pair_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_3) + 24648) < wns_3)) {
          condval_9 = WQ[((((((kp_2 * 3072) + ((((int)threadIdx.x) >> 5) * 768)) + (pair_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_3) + 24648)];
        } else {
          condval_9 = 0;
        }
        b_3[j_s_3] = condval_9;
      }
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
        for (int j_s_4 = 0; j_s_4 < 4; ++j_s_4) {
          int condval_10;
          if (((((((((((int)threadIdx.x) >> 5) * 768) + (comp * 256)) + (pair_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_4) + 33864) < wns_3)) {
            condval_10 = WQ[(((((((((int)threadIdx.x) >> 5) * 768) + (comp * 256)) + (pair_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_4) + 33864)];
          } else {
            condval_10 = 0;
          }
          b_4[j_s_4] = condval_10;
        }
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
            int condval_11;
            if ((((((int)threadIdx.x) >> 5) + 45128) < wns_5)) {
              condval_11 = WS[((((int)threadIdx.x) >> 5) + 45128)];
            } else {
              condval_11 = 0;
            }
            int _reinterpret_tmp_2 = condval_11;
            Z[(((dest * 24) + (n_8 * 2)) + j_1)] = wide_mul(Z[(((dest * 24) + (n_8 * 2)) + j_1)], wide_splat((*(float *)(&(_reinterpret_tmp_2)))));
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
        int condval_12;
        if (((((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36936) < wns_4)) {
          condval_12 = WB[(((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36936)];
        } else {
          condval_12 = 0;
        }
        int condval_13;
        if (((((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36937) < wns_4)) {
          condval_13 = WB[(((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_13 * 512)) + ((n_13 >> 2) * 256)) + ((n_13 & 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((n_13 & 3) >> 1) * 2)) + 36937)];
        } else {
          condval_13 = 0;
        }
        uint64_t r_6 = wide_mma(Q[((slab_2 * 8) + (m_13 * 4))], Q[(((slab_2 * 8) + (m_13 * 4)) + 1)], Q[(((slab_2 * 8) + (m_13 * 4)) + 2)], Q[(((slab_2 * 8) + (m_13 * 4)) + 3)], K[(n_13 * 2)], K[((n_13 * 2) + 1)], condval_12, condval_13);
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
      int v__3 = S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_6 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_6 & 3) >> 1) * 2))];
      int v__4 = S[(((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_6 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_6 & 3) >> 1) * 2)) + 1)];
      int condval_14;
      if (((word_6 % 2) == 0)) {
        condval_14 = 21520;
      } else {
        condval_14 = 30258;
      }
      Residual[word_6] = wide_packet_prmt((*(uint *)(&(v__3))), (*(uint *)(&(v__4))), condval_14);
    }
    __syncthreads();
    #pragma unroll
    for (int word_7 = 0; word_7 < 8; ++word_7) {
      v_25 = P[((((((((((((0 | ((word_7 * 128) & 1)) | ((((word_7 * 128) >> 1) & 1) << 7)) | ((((word_7 * 128) >> 2) & 1) << 2)) | ((((word_7 * 128) >> 3) & 1) << 3)) | ((((word_7 * 128) >> 4) & 1) << 4)) | ((((word_7 * 128) >> 5) & 1) << 5)) | ((((word_7 * 128) >> 6) & 1) << 6)) | ((((word_7 * 128) >> 7) & 1) << 1)) | ((((word_7 * 128) >> 8) & 1) << 8)) | ((((word_7 * 128) >> 9) & 1) << 9)) | ((((word_7 * 128) >> 10) & 1) << 10)) >> 7)];
      v_26 = P[((((((((((((0 | (((word_7 * 128) + 2) & 1)) | (((((word_7 * 128) + 2) >> 1) & 1) << 7)) | (((((word_7 * 128) + 2) >> 2) & 1) << 2)) | (((((word_7 * 128) + 2) >> 3) & 1) << 3)) | (((((word_7 * 128) + 2) >> 4) & 1) << 4)) | (((((word_7 * 128) + 2) >> 5) & 1) << 5)) | (((((word_7 * 128) + 2) >> 6) & 1) << 6)) | (((((word_7 * 128) + 2) >> 7) & 1) << 1)) | (((((word_7 * 128) + 2) >> 8) & 1) << 8)) | (((((word_7 * 128) + 2) >> 9) & 1) << 9)) | (((((word_7 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
      v_27 = wide_packet_prmt(v_25, v_26, (((((uint)0 | (((uint)((((((((((((0 | (((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_7 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
      uint _reinterpret_tmp_3 = v_27;
      S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_7 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_7 & 3))] = (*(int *)(&(_reinterpret_tmp_3)));
    }
    __syncthreads();
    #pragma unroll
    for (int m_15 = 0; m_15 < 2; ++m_15) {
      #pragma unroll
      for (int n_17 = 0; n_17 < 4; ++n_17) {
        int condval_15;
        if (((((((((int)threadIdx.x) >> 5) * 16) + (n_17 * 4)) + (((int)threadIdx.x) & 3)) + 49228) < wns_8)) {
          condval_15 = AG[(((((((int)threadIdx.x) >> 5) * 16) + (n_17 * 4)) + (((int)threadIdx.x) & 3)) + 49228)];
        } else {
          condval_15 = 0;
        }
        int g_1 = condval_15;
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
        for (int j_s_5 = 0; j_s_5 < 4; ++j_s_5) {
          int condval_16;
          if ((((((((kp_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_5 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_5) + 45132) < wns_6)) {
            condval_16 = WP[((((((kp_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_5 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_5) + 45132)];
          } else {
            condval_16 = 0;
          }
          b_5[j_s_5] = condval_16;
        }
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
    for (int word_8 = 0; word_8 < 8; ++word_8) {
      v_28 = ((((((((((0 | (((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 9)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 1)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 2)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 3)) | (((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 8));
      v_29 = wide_shfl(P[((((((((((((0 | ((word_8 * 128) & 1)) | ((((word_8 * 128) >> 1) & 1) << 7)) | ((((word_8 * 128) >> 2) & 1) << 9)) | ((((word_8 * 128) >> 3) & 1) << 1)) | ((((word_8 * 128) >> 4) & 1) << 4)) | ((((word_8 * 128) >> 5) & 1) << 5)) | ((((word_8 * 128) >> 6) & 1) << 6)) | ((((word_8 * 128) >> 7) & 1) << 2)) | ((((word_8 * 128) >> 8) & 1) << 3)) | ((((word_8 * 128) >> 9) & 1) << 8)) >> 7) | 0)], ((v_28 & 127) >> 2));
      uint condval_17;
      if (((v_28 >> 7) == ((((((((((((0 | ((word_8 * 128) & 1)) | ((((word_8 * 128) >> 1) & 1) << 7)) | ((((word_8 * 128) >> 2) & 1) << 9)) | ((((word_8 * 128) >> 3) & 1) << 1)) | ((((word_8 * 128) >> 4) & 1) << 4)) | ((((word_8 * 128) >> 5) & 1) << 5)) | ((((word_8 * 128) >> 6) & 1) << 6)) | ((((word_8 * 128) >> 7) & 1) << 2)) | ((((word_8 * 128) >> 8) & 1) << 3)) | ((((word_8 * 128) >> 9) & 1) << 8)) >> 7) | 0))) {
        condval_17 = v_29;
      } else {
        condval_17 = (uint)0;
      }
      v_30 = condval_17;
      v_31 = wide_shfl(P[((((((((((((0 | ((word_8 * 128) & 1)) | ((((word_8 * 128) >> 1) & 1) << 7)) | ((((word_8 * 128) >> 2) & 1) << 9)) | ((((word_8 * 128) >> 3) & 1) << 1)) | ((((word_8 * 128) >> 4) & 1) << 4)) | ((((word_8 * 128) >> 5) & 1) << 5)) | ((((word_8 * 128) >> 6) & 1) << 6)) | ((((word_8 * 128) >> 7) & 1) << 2)) | ((((word_8 * 128) >> 8) & 1) << 3)) | ((((word_8 * 128) >> 9) & 1) << 8)) >> 7) | 4)], ((v_28 & 127) >> 2));
      uint condval_18;
      if (((v_28 >> 7) == ((((((((((((0 | ((word_8 * 128) & 1)) | ((((word_8 * 128) >> 1) & 1) << 7)) | ((((word_8 * 128) >> 2) & 1) << 9)) | ((((word_8 * 128) >> 3) & 1) << 1)) | ((((word_8 * 128) >> 4) & 1) << 4)) | ((((word_8 * 128) >> 5) & 1) << 5)) | ((((word_8 * 128) >> 6) & 1) << 6)) | ((((word_8 * 128) >> 7) & 1) << 2)) | ((((word_8 * 128) >> 8) & 1) << 3)) | ((((word_8 * 128) >> 9) & 1) << 8)) >> 7) | 4))) {
        condval_18 = v_31;
      } else {
        condval_18 = v_30;
      }
      v_32 = condval_18;
      v_33 = ((((((((((0 | ((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 9)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 2)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 3)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 8));
      v_34 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 1) & 1)) | (((((word_8 * 128) + 1) >> 1) & 1) << 7)) | (((((word_8 * 128) + 1) >> 2) & 1) << 9)) | (((((word_8 * 128) + 1) >> 3) & 1) << 1)) | (((((word_8 * 128) + 1) >> 4) & 1) << 4)) | (((((word_8 * 128) + 1) >> 5) & 1) << 5)) | (((((word_8 * 128) + 1) >> 6) & 1) << 6)) | (((((word_8 * 128) + 1) >> 7) & 1) << 2)) | (((((word_8 * 128) + 1) >> 8) & 1) << 3)) | (((((word_8 * 128) + 1) >> 9) & 1) << 8)) >> 7) | 0)], ((v_33 & 127) >> 2));
      uint condval_19;
      if (((v_33 >> 7) == ((((((((((((0 | (((word_8 * 128) + 1) & 1)) | (((((word_8 * 128) + 1) >> 1) & 1) << 7)) | (((((word_8 * 128) + 1) >> 2) & 1) << 9)) | (((((word_8 * 128) + 1) >> 3) & 1) << 1)) | (((((word_8 * 128) + 1) >> 4) & 1) << 4)) | (((((word_8 * 128) + 1) >> 5) & 1) << 5)) | (((((word_8 * 128) + 1) >> 6) & 1) << 6)) | (((((word_8 * 128) + 1) >> 7) & 1) << 2)) | (((((word_8 * 128) + 1) >> 8) & 1) << 3)) | (((((word_8 * 128) + 1) >> 9) & 1) << 8)) >> 7) | 0))) {
        condval_19 = v_34;
      } else {
        condval_19 = (uint)0;
      }
      v_35 = condval_19;
      v_36 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 1) & 1)) | (((((word_8 * 128) + 1) >> 1) & 1) << 7)) | (((((word_8 * 128) + 1) >> 2) & 1) << 9)) | (((((word_8 * 128) + 1) >> 3) & 1) << 1)) | (((((word_8 * 128) + 1) >> 4) & 1) << 4)) | (((((word_8 * 128) + 1) >> 5) & 1) << 5)) | (((((word_8 * 128) + 1) >> 6) & 1) << 6)) | (((((word_8 * 128) + 1) >> 7) & 1) << 2)) | (((((word_8 * 128) + 1) >> 8) & 1) << 3)) | (((((word_8 * 128) + 1) >> 9) & 1) << 8)) >> 7) | 4)], ((v_33 & 127) >> 2));
      uint condval_20;
      if (((v_33 >> 7) == ((((((((((((0 | (((word_8 * 128) + 1) & 1)) | (((((word_8 * 128) + 1) >> 1) & 1) << 7)) | (((((word_8 * 128) + 1) >> 2) & 1) << 9)) | (((((word_8 * 128) + 1) >> 3) & 1) << 1)) | (((((word_8 * 128) + 1) >> 4) & 1) << 4)) | (((((word_8 * 128) + 1) >> 5) & 1) << 5)) | (((((word_8 * 128) + 1) >> 6) & 1) << 6)) | (((((word_8 * 128) + 1) >> 7) & 1) << 2)) | (((((word_8 * 128) + 1) >> 8) & 1) << 3)) | (((((word_8 * 128) + 1) >> 9) & 1) << 8)) >> 7) | 4))) {
        condval_20 = v_36;
      } else {
        condval_20 = v_35;
      }
      v_37 = condval_20;
      v_38 = ((((((((((0 | ((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 9)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 2)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 3)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 8));
      v_39 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 2) & 1)) | (((((word_8 * 128) + 2) >> 1) & 1) << 7)) | (((((word_8 * 128) + 2) >> 2) & 1) << 9)) | (((((word_8 * 128) + 2) >> 3) & 1) << 1)) | (((((word_8 * 128) + 2) >> 4) & 1) << 4)) | (((((word_8 * 128) + 2) >> 5) & 1) << 5)) | (((((word_8 * 128) + 2) >> 6) & 1) << 6)) | (((((word_8 * 128) + 2) >> 7) & 1) << 2)) | (((((word_8 * 128) + 2) >> 8) & 1) << 3)) | (((((word_8 * 128) + 2) >> 9) & 1) << 8)) >> 7) | 0)], ((v_38 & 127) >> 2));
      uint condval_21;
      if (((v_38 >> 7) == ((((((((((((0 | (((word_8 * 128) + 2) & 1)) | (((((word_8 * 128) + 2) >> 1) & 1) << 7)) | (((((word_8 * 128) + 2) >> 2) & 1) << 9)) | (((((word_8 * 128) + 2) >> 3) & 1) << 1)) | (((((word_8 * 128) + 2) >> 4) & 1) << 4)) | (((((word_8 * 128) + 2) >> 5) & 1) << 5)) | (((((word_8 * 128) + 2) >> 6) & 1) << 6)) | (((((word_8 * 128) + 2) >> 7) & 1) << 2)) | (((((word_8 * 128) + 2) >> 8) & 1) << 3)) | (((((word_8 * 128) + 2) >> 9) & 1) << 8)) >> 7) | 0))) {
        condval_21 = v_39;
      } else {
        condval_21 = (uint)0;
      }
      v_40 = condval_21;
      v_41 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 2) & 1)) | (((((word_8 * 128) + 2) >> 1) & 1) << 7)) | (((((word_8 * 128) + 2) >> 2) & 1) << 9)) | (((((word_8 * 128) + 2) >> 3) & 1) << 1)) | (((((word_8 * 128) + 2) >> 4) & 1) << 4)) | (((((word_8 * 128) + 2) >> 5) & 1) << 5)) | (((((word_8 * 128) + 2) >> 6) & 1) << 6)) | (((((word_8 * 128) + 2) >> 7) & 1) << 2)) | (((((word_8 * 128) + 2) >> 8) & 1) << 3)) | (((((word_8 * 128) + 2) >> 9) & 1) << 8)) >> 7) | 4)], ((v_38 & 127) >> 2));
      uint condval_22;
      if (((v_38 >> 7) == ((((((((((((0 | (((word_8 * 128) + 2) & 1)) | (((((word_8 * 128) + 2) >> 1) & 1) << 7)) | (((((word_8 * 128) + 2) >> 2) & 1) << 9)) | (((((word_8 * 128) + 2) >> 3) & 1) << 1)) | (((((word_8 * 128) + 2) >> 4) & 1) << 4)) | (((((word_8 * 128) + 2) >> 5) & 1) << 5)) | (((((word_8 * 128) + 2) >> 6) & 1) << 6)) | (((((word_8 * 128) + 2) >> 7) & 1) << 2)) | (((((word_8 * 128) + 2) >> 8) & 1) << 3)) | (((((word_8 * 128) + 2) >> 9) & 1) << 8)) >> 7) | 4))) {
        condval_22 = v_41;
      } else {
        condval_22 = v_40;
      }
      v_42 = condval_22;
      v_43 = ((((((((((0 | ((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 9)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 1)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 2)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 3)) | ((((((word_8 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 8));
      v_44 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 3) & 1)) | (((((word_8 * 128) + 3) >> 1) & 1) << 7)) | (((((word_8 * 128) + 3) >> 2) & 1) << 9)) | (((((word_8 * 128) + 3) >> 3) & 1) << 1)) | (((((word_8 * 128) + 3) >> 4) & 1) << 4)) | (((((word_8 * 128) + 3) >> 5) & 1) << 5)) | (((((word_8 * 128) + 3) >> 6) & 1) << 6)) | (((((word_8 * 128) + 3) >> 7) & 1) << 2)) | (((((word_8 * 128) + 3) >> 8) & 1) << 3)) | (((((word_8 * 128) + 3) >> 9) & 1) << 8)) >> 7) | 0)], ((v_43 & 127) >> 2));
      uint condval_23;
      if (((v_43 >> 7) == ((((((((((((0 | (((word_8 * 128) + 3) & 1)) | (((((word_8 * 128) + 3) >> 1) & 1) << 7)) | (((((word_8 * 128) + 3) >> 2) & 1) << 9)) | (((((word_8 * 128) + 3) >> 3) & 1) << 1)) | (((((word_8 * 128) + 3) >> 4) & 1) << 4)) | (((((word_8 * 128) + 3) >> 5) & 1) << 5)) | (((((word_8 * 128) + 3) >> 6) & 1) << 6)) | (((((word_8 * 128) + 3) >> 7) & 1) << 2)) | (((((word_8 * 128) + 3) >> 8) & 1) << 3)) | (((((word_8 * 128) + 3) >> 9) & 1) << 8)) >> 7) | 0))) {
        condval_23 = v_44;
      } else {
        condval_23 = (uint)0;
      }
      v_45 = condval_23;
      v_46 = wide_shfl(P[((((((((((((0 | (((word_8 * 128) + 3) & 1)) | (((((word_8 * 128) + 3) >> 1) & 1) << 7)) | (((((word_8 * 128) + 3) >> 2) & 1) << 9)) | (((((word_8 * 128) + 3) >> 3) & 1) << 1)) | (((((word_8 * 128) + 3) >> 4) & 1) << 4)) | (((((word_8 * 128) + 3) >> 5) & 1) << 5)) | (((((word_8 * 128) + 3) >> 6) & 1) << 6)) | (((((word_8 * 128) + 3) >> 7) & 1) << 2)) | (((((word_8 * 128) + 3) >> 8) & 1) << 3)) | (((((word_8 * 128) + 3) >> 9) & 1) << 8)) >> 7) | 4)], ((v_43 & 127) >> 2));
      uint condval_24;
      if (((v_43 >> 7) == ((((((((((((0 | (((word_8 * 128) + 3) & 1)) | (((((word_8 * 128) + 3) >> 1) & 1) << 7)) | (((((word_8 * 128) + 3) >> 2) & 1) << 9)) | (((((word_8 * 128) + 3) >> 3) & 1) << 1)) | (((((word_8 * 128) + 3) >> 4) & 1) << 4)) | (((((word_8 * 128) + 3) >> 5) & 1) << 5)) | (((((word_8 * 128) + 3) >> 6) & 1) << 6)) | (((((word_8 * 128) + 3) >> 7) & 1) << 2)) | (((((word_8 * 128) + 3) >> 8) & 1) << 3)) | (((((word_8 * 128) + 3) >> 9) & 1) << 8)) >> 7) | 4))) {
        condval_24 = v_46;
      } else {
        condval_24 = v_45;
      }
      v_47 = condval_24;
      A[word_8] = (((((uint)0 | (((v_32 >> ((uint)((v_28 & 3) * 8))) & (uint)255) << (uint)0)) | (((v_37 >> ((uint)((v_33 & 3) * 8))) & (uint)255) << (uint)8)) | (((v_42 >> ((uint)((v_38 & 3) * 8))) & (uint)255) << (uint)16)) | (((v_47 >> ((uint)((v_43 & 3) * 8))) & (uint)255) << (uint)24));
    }
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
      int rmod_12 = (((int)blockIdx.x) % gx);
      int rmod_13 = (((int)blockIdx.x) % gx);
      int rmod_14 = (((int)blockIdx.x) % gx);
      int rdiv_7 = (((int)blockIdx.x) / gx);
      int rmod_15 = (((int)blockIdx.x) % gx);
      int rdiv_8 = (((int)blockIdx.x) / gx);
      int condval_25;
      if (((((0 <= (((((((0 <= gx) && (0 <= rmod_12)) || ((gx < 0) && (rmod_12 <= 0))) ? rmod_12 : (rmod_12 + gx)) * 8) + ((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx)) & ((((((((0 <= gx) && (0 <= rmod_13)) || ((gx < 0) && (rmod_13 <= 0))) ? rmod_13 : (rmod_13 + gx)) * 8) + ((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7)) + sx) < width)) & (0 <= ((((((((0 <= gx) && (0 <= rmod_14)) || ((gx < 0) && (rmod_14 <= 0))) ? rdiv_7 : (rdiv_7 - 1)) * 8) + (((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + (((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy))) & (((((((((0 <= gx) && (0 <= rmod_15)) || ((gx < 0) && (rmod_15 <= 0))) ? rdiv_8 : (rdiv_8 - 1)) * 8) + (((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + (((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) < height))) {
        int rmod_16 = (((int)blockIdx.x) % gx);
        int rmod_17 = (((int)blockIdx.x) % gx);
        int rdiv_9 = (((int)blockIdx.x) / gx);
        condval_25 = ((((((((((0 <= gx) && (0 <= rmod_16)) || ((gx < 0) && (rmod_16 <= 0))) ? rmod_16 : (rmod_16 + gx)) * 128) + ((((((((((0 <= gx) && (0 <= rmod_17)) || ((gx < 0) && (rmod_17 <= 0))) ? rdiv_9 : (rdiv_9 - 1)) * 8) + (((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 4) & 3) * 2)) + (((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) & 1)) + sy) * width) * 16)) + (((((((slab_2 * 32) + (((((int)threadIdx.x) & 31) & 1) * 16)) + ((((((int)threadIdx.x) & 31) >> 1) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) >> 1) & 7) * 16)) + (sx * 16)) + ((((((((int)threadIdx.x) >> 5) * 2) + part) * height) * width) * 16)) + ((((((((int)threadIdx.x) >> 5) * 32) + (part * 16)) & 1) | (((((((int)threadIdx.x) >> 5) * 32) + (part * 16)) & 6) << 1)) | (((((((int)threadIdx.x) >> 5) * 32) + (part * 16)) & 8) >> 2)));
      } else {
        condval_25 = -1;
      }
      v_48 = condval_25;
      int raw_1 = v_48;
      if ((0 <= raw_1) & ((raw_1 + 16) <= (rn - out))) {
        {
          int RW_local_cast_1[4];
          int4 __2;
          uint4 v__5 = *(uint4*)(A + (part * 4));
          __2.x = (int)(v__5.x);
          __2.y = (int)(v__5.y);
          __2.z = (int)(v__5.z);
          __2.w = (int)(v__5.w);
          *(int4*)(RW_local_cast_1 + 0) = __2;
          if (0 <= (out + raw_1)) {
            *(int4*)(RW + (((out + raw_1) >> 4) * 4)) = *(int4*)(RW_local_cast_1 + 0);
          }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  __threadfence();
  if (0 <= counter) {
    if (((int)threadIdx.x) == 0) {
      if (((counter >> 2) + ((int)blockIdx.x)) < (rn >> 2)) {
        RW[((((int64_t)counter) >> (int64_t)2) + ((int64_t)((int)blockIdx.x)))] = 0;
      }
    }
  }
}

