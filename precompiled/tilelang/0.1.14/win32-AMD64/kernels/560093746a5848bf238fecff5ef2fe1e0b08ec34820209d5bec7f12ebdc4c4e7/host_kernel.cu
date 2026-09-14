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

extern "C" __global__ void kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX, int counter, int gx, int height, int inp, int out, int rn, int sn, int sx, int sy, int tiles, int width, int wns_0, int wns_1, int wns_2, int wns_3, int wns_4, int wns_6, int wns_7, int wns_8);
extern "C" __global__ void __launch_bounds__(64, 1) kernel_kernel(const int* __restrict__ AG, const int* __restrict__ FG, int* __restrict__ RW, const int* __restrict__ SW, const int* __restrict__ WB, const int* __restrict__ WP, const int* __restrict__ WQ, const int* __restrict__ WR, const int* __restrict__ WS, const int* __restrict__ WT, const int* __restrict__ WX, int counter, int gx, int height, int inp, int out, int rn, int sn, int sx, int sy, int tiles, int width, int wns_0, int wns_1, int wns_2, int wns_3, int wns_4, int wns_6, int wns_7, int wns_8) {
  extern __shared__ __align__(1024) int S[];
  int v = 0;
  int v_1 = 0;
  int v_2 = 0;
  int v_3 = 0;
  uint temp[4];
  uint F[32];
  uint C[32];
  uint P[16];
  uint A[16];
  uint X[32];
  int a[8];
  int b[4];
  uint v_4 = (uint)0;
  uint v_5 = (uint)0;
  uint v_6 = (uint)0;
  int b_1[4];
  uint v_7 = (uint)0;
  uint v_8 = (uint)0;
  uint v_9 = (uint)0;
  int b_2[4];
  uint v_10 = (uint)0;
  uint v_11 = (uint)0;
  uint v_12 = (uint)0;
  uint QP[16];
  uint KP[16];
  uint VP[16];
  uint Z[48];
  int a_1[8];
  int b_3[4];
  uint Q[16];
  uint K[16];
  uint V[16];
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
  uint Residual[16];
  uint v_25 = (uint)0;
  uint v_26 = (uint)0;
  uint v_27 = (uint)0;
  int a_2[8];
  int b_4[4];
  int v_28 = 0;
  uint v_29 = (uint)0;
  uint v_30 = (uint)0;
  int v_31 = 0;
  uint v_32 = (uint)0;
  uint v_33 = (uint)0;
  int v_34 = 0;
  uint v_35 = (uint)0;
  uint v_36 = (uint)0;
  int v_37 = 0;
  uint v_38 = (uint)0;
  uint v_39 = (uint)0;
  int v_40 = 0;
  int v_41 = 0;
  int v_42 = 0;
  int v_43 = 0;
  int v_44 = 0;
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    v = ((((((((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 3) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 12) << 2)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 16) >> 1)) | (((((((int)threadIdx.x) >> 5) * 32) + ((((int)threadIdx.x) & 3) * 4)) & 32) << 4)) | ((((m * 16) + ((((int)threadIdx.x) & 31) >> 2)) & 7) << 6)) | ((((m * 16) + ((((int)threadIdx.x) & 31) >> 2)) & 8) >> 1)) | ((((m * 16) + ((((int)threadIdx.x) & 31) >> 2)) & 48) << 6));
    int rmod = (((int)blockIdx.x) % gx);
    int rdiv = (((int)blockIdx.x) / gx);
    v_1 = (((((((0 <= gx) && (0 <= rmod)) || ((gx < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1)) * 2) + (v >> 11)) + (sy >> 2));
    int rmod_1 = (((int)blockIdx.x) % gx);
    v_2 = (((((((0 <= gx) && (0 <= rmod_1)) || ((gx < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + gx)) * 2) + ((v & 2047) >> 10)) + (sx >> 2));
    int condval;
    if (((((0 <= v_1) & (0 <= v_2)) & (v_1 < (height >> 2))) & (v_2 < (width >> 2)))) {
      condval = (((((v_1 * (width >> 2)) * 1024) + (v_2 * 1024)) + (((v >> 9) & 1) * 512)) + (v & 511));
    } else {
      condval = -1;
    }
    v_3 = condval;
    int raw = v_3;
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
    {
      int S_local_cast_1[4];
      int4 __2;
      uint4 v__1 = *(uint4*)(temp + 0);
      __2.x = (int)(v__1.x);
      __2.y = (int)(v__1.y);
      __2.z = (int)(v__1.z);
      __2.w = (int)(v__1.w);
      *(int4*)(S_local_cast_1 + 0) = __2;
      *(int4*)(S + ((((((int)threadIdx.x) >> 5) * 512) + (m * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(int4*)(S_local_cast_1 + 0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int channel = 0; channel < 2; ++channel) {
    #pragma unroll
    for (int m_1 = 0; m_1 < 2; ++m_1) {
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        int v__2 = S[(((((channel * 512) + ((((int)threadIdx.x) >> 5) * 256)) + (m_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n >> 1) * 2))];
        int v__3 = S[((((((channel * 512) + ((((int)threadIdx.x) >> 5) * 256)) + (m_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n >> 1) * 2)) + 1)];
        int condval_2;
        if (((n % 2) == 0)) {
          condval_2 = 21520;
        } else {
          condval_2 = 30258;
        }
        uint bits = wide_packet_prmt((*(uint *)(&(v__2))), (*(uint *)(&(v__3))), condval_2);
        int condval_3;
        if (((((channel * 16) + (n * 4)) + (((int)threadIdx.x) & 3)) < wns_7)) {
          condval_3 = FG[(((channel * 16) + (n * 4)) + (((int)threadIdx.x) & 3))];
        } else {
          condval_3 = 0;
        }
        int g = condval_3;
        F[(((m_1 * 16) + (channel * 8)) + (n * 2))] = wide_mul(wide_decode(bits), g);
        F[((((m_1 * 16) + (channel * 8)) + (n * 2)) + 1)] = wide_mul(wide_decode((bits >> (uint)16)), g);
      }
    }
  }
  for (int stream = 0; stream < 2; ++stream) {
    #pragma unroll
    for (int m_2 = 0; m_2 < 2; ++m_2) {
      #pragma unroll
      for (int n_1 = 0; n_1 < 4; ++n_1) {
        C[((m_2 * 8) + (n_1 * 2))] = (uint)0;
        C[(((m_2 * 8) + (n_1 * 2)) + 1)] = (uint)0;
      }
    }
    for (int slab = 0; slab < 4; ++slab) {
      #pragma unroll
      for (int m_3 = 0; m_3 < 2; ++m_3) {
        #pragma unroll
        for (int n_2 = 0; n_2 < 4; ++n_2) {
          X[((m_3 * 8) + (n_2 * 2))] = (uint)0;
          X[(((m_3 * 8) + (n_2 * 2)) + 1)] = (uint)0;
        }
      }
      for (int kp = 0; kp < 2; ++kp) {
        #pragma unroll
        for (int m_4 = 0; m_4 < 2; ++m_4) {
          *(int4*)(a + (m_4 * 4)) = *(int4*)(S + ((((kp * 512) + ((((int)threadIdx.x) >> 5) * 256)) + (m_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
        }
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          for (int j_s = 0; j_s < 4; ++j_s) {
            int condval_4;
            if ((((((((stream * 2048) + (kp * 1024)) + (slab * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s) < wns_0)) {
              condval_4 = WX[((((((stream * 2048) + (kp * 1024)) + (slab * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s)];
            } else {
              condval_4 = 0;
            }
            b[j_s] = condval_4;
          }
          #pragma unroll
          for (int m_5 = 0; m_5 < 2; ++m_5) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
              uint64_t r = wide_mma(a[(m_5 * 4)], a[((m_5 * 4) + 1)], a[((m_5 * 4) + 2)], a[((m_5 * 4) + 3)], b[(half * 2)], b[((half * 2) + 1)], X[(((m_5 * 8) + (pair * 4)) + (half * 2))], X[((((m_5 * 8) + (pair * 4)) + (half * 2)) + 1)]);
              X[(((m_5 * 8) + (pair * 4)) + (half * 2))] = ((uint)r);
              X[((((m_5 * 8) + (pair * 4)) + (half * 2)) + 1)] = ((uint)(r >> (uint64_t)32));
            }
          }
        }
      }
      #pragma unroll
      for (int m_6 = 0; m_6 < 2; ++m_6) {
        #pragma unroll
        for (int n_3 = 0; n_3 < 4; ++n_3) {
          P[((m_6 * 4) + n_3)] = wide_encode(wide_mul(X[((m_6 * 8) + (n_3 * 2))], wide_fma(wide_min(wide_max(X[((m_6 * 8) + (n_3 * 2))], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/)), wide_fma(wide_abs(wide_min(wide_max(X[((m_6 * 8) + (n_3 * 2))], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/))), wide_splat(-0x1.cap-5f/*-5.590820e-02*/), wide_splat(0x1.cap-2f/*4.472656e-01*/)), wide_splat(0x1.cap-1f/*8.945312e-01*/))), wide_mul(X[(((m_6 * 8) + (n_3 * 2)) + 1)], wide_fma(wide_min(wide_max(X[(((m_6 * 8) + (n_3 * 2)) + 1)], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/)), wide_fma(wide_abs(wide_min(wide_max(X[(((m_6 * 8) + (n_3 * 2)) + 1)], wide_splat(-0x1p+2f/*-4.000000e+00*/)), wide_splat(0x1p+2f/*4.000000e+00*/))), wide_splat(-0x1.cap-5f/*-5.590820e-02*/), wide_splat(0x1.cap-2f/*4.472656e-01*/)), wide_splat(0x1.cap-1f/*8.945312e-01*/))));
        }
      }
      #pragma unroll
      for (int word = 0; word < 8; ++word) {
        v_4 = P[((((((((((((0 | ((word * 128) & 1)) | ((((word * 128) >> 1) & 1) << 7)) | ((((word * 128) >> 2) & 1) << 2)) | ((((word * 128) >> 3) & 1) << 3)) | ((((word * 128) >> 4) & 1) << 4)) | ((((word * 128) >> 5) & 1) << 5)) | ((((word * 128) >> 6) & 1) << 6)) | ((((word * 128) >> 7) & 1) << 1)) | ((((word * 128) >> 8) & 1) << 8)) | ((((word * 128) >> 9) & 1) << 9)) | ((((word * 128) >> 10) & 1) << 10)) >> 7)];
        v_5 = P[((((((((((((0 | (((word * 128) + 2) & 1)) | (((((word * 128) + 2) >> 1) & 1) << 7)) | (((((word * 128) + 2) >> 2) & 1) << 2)) | (((((word * 128) + 2) >> 3) & 1) << 3)) | (((((word * 128) + 2) >> 4) & 1) << 4)) | (((((word * 128) + 2) >> 5) & 1) << 5)) | (((((word * 128) + 2) >> 6) & 1) << 6)) | (((((word * 128) + 2) >> 7) & 1) << 1)) | (((((word * 128) + 2) >> 8) & 1) << 8)) | (((((word * 128) + 2) >> 9) & 1) << 9)) | (((((word * 128) + 2) >> 10) & 1) << 10)) >> 7)];
        v_6 = wide_packet_prmt(v_4, v_5, (((((uint)0 | (((uint)((((((((((((0 | (((word * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
        A[word] = v_6;
      }
      #pragma unroll
      for (int pair_1 = 0; pair_1 < 2; ++pair_1) {
        for (int j_s_1 = 0; j_s_1 < 4; ++j_s_1) {
          int condval_5;
          if (((((((stream * 1024) + (slab * 256)) + (pair_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_1) < wns_1)) {
            condval_5 = WR[(((((stream * 1024) + (slab * 256)) + (pair_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_1)];
          } else {
            condval_5 = 0;
          }
          b_1[j_s_1] = condval_5;
        }
        #pragma unroll
        for (int m_7 = 0; m_7 < 2; ++m_7) {
          #pragma unroll
          for (int half_1 = 0; half_1 < 2; ++half_1) {
            uint64_t r_1 = wide_mma(A[(m_7 * 4)], A[((m_7 * 4) + 1)], A[((m_7 * 4) + 2)], A[((m_7 * 4) + 3)], b_1[(half_1 * 2)], b_1[((half_1 * 2) + 1)], C[(((m_7 * 8) + (pair_1 * 4)) + (half_1 * 2))], C[((((m_7 * 8) + (pair_1 * 4)) + (half_1 * 2)) + 1)]);
            C[(((m_7 * 8) + (pair_1 * 4)) + (half_1 * 2))] = ((uint)r_1);
            C[((((m_7 * 8) + (pair_1 * 4)) + (half_1 * 2)) + 1)] = ((uint)(r_1 >> (uint64_t)32));
          }
        }
      }
    }
    #pragma unroll
    for (int m_8 = 0; m_8 < 2; ++m_8) {
      #pragma unroll
      for (int n_4 = 0; n_4 < 4; ++n_4) {
        P[((m_8 * 4) + n_4)] = wide_encode(C[((m_8 * 8) + (n_4 * 2))], C[(((m_8 * 8) + (n_4 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int word_1 = 0; word_1 < 8; ++word_1) {
      v_7 = P[((((((((((((0 | ((word_1 * 128) & 1)) | ((((word_1 * 128) >> 1) & 1) << 7)) | ((((word_1 * 128) >> 2) & 1) << 2)) | ((((word_1 * 128) >> 3) & 1) << 3)) | ((((word_1 * 128) >> 4) & 1) << 4)) | ((((word_1 * 128) >> 5) & 1) << 5)) | ((((word_1 * 128) >> 6) & 1) << 6)) | ((((word_1 * 128) >> 7) & 1) << 1)) | ((((word_1 * 128) >> 8) & 1) << 8)) | ((((word_1 * 128) >> 9) & 1) << 9)) | ((((word_1 * 128) >> 10) & 1) << 10)) >> 7)];
      v_8 = P[((((((((((((0 | (((word_1 * 128) + 2) & 1)) | (((((word_1 * 128) + 2) >> 1) & 1) << 7)) | (((((word_1 * 128) + 2) >> 2) & 1) << 2)) | (((((word_1 * 128) + 2) >> 3) & 1) << 3)) | (((((word_1 * 128) + 2) >> 4) & 1) << 4)) | (((((word_1 * 128) + 2) >> 5) & 1) << 5)) | (((((word_1 * 128) + 2) >> 6) & 1) << 6)) | (((((word_1 * 128) + 2) >> 7) & 1) << 1)) | (((((word_1 * 128) + 2) >> 8) & 1) << 8)) | (((((word_1 * 128) + 2) >> 9) & 1) << 9)) | (((((word_1 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
      v_9 = wide_packet_prmt(v_7, v_8, (((((uint)0 | (((uint)((((((((((((0 | (((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_1 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
      A[word_1] = v_9;
    }
    #pragma unroll
    for (int pair_2 = 0; pair_2 < 4; ++pair_2) {
      for (int j_s_2 = 0; j_s_2 < 4; ++j_s_2) {
        int condval_6;
        if ((((((stream * 512) + (pair_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_2) < wns_2)) {
          condval_6 = WT[((((stream * 512) + (pair_2 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_2)];
        } else {
          condval_6 = 0;
        }
        b_2[j_s_2] = condval_6;
      }
      #pragma unroll
      for (int m_9 = 0; m_9 < 2; ++m_9) {
        #pragma unroll
        for (int half_2 = 0; half_2 < 2; ++half_2) {
          uint64_t r_2 = wide_mma(A[(m_9 * 4)], A[((m_9 * 4) + 1)], A[((m_9 * 4) + 2)], A[((m_9 * 4) + 3)], b_2[(half_2 * 2)], b_2[((half_2 * 2) + 1)], F[(((m_9 * 16) + (pair_2 * 4)) + (half_2 * 2))], F[((((m_9 * 16) + (pair_2 * 4)) + (half_2 * 2)) + 1)]);
          F[(((m_9 * 16) + (pair_2 * 4)) + (half_2 * 2))] = ((uint)r_2);
          F[((((m_9 * 16) + (pair_2 * 4)) + (half_2 * 2)) + 1)] = ((uint)(r_2 >> (uint64_t)32));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int channel_1 = 0; channel_1 < 2; ++channel_1) {
    #pragma unroll
    for (int m_10 = 0; m_10 < 2; ++m_10) {
      #pragma unroll
      for (int n_5 = 0; n_5 < 4; ++n_5) {
        P[((m_10 * 4) + n_5)] = wide_encode(F[(((m_10 * 16) + (channel_1 * 8)) + (n_5 * 2))], F[((((m_10 * 16) + (channel_1 * 8)) + (n_5 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int word_2 = 0; word_2 < 8; ++word_2) {
      v_10 = P[((((((((((((0 | ((word_2 * 128) & 1)) | ((((word_2 * 128) >> 1) & 1) << 7)) | ((((word_2 * 128) >> 2) & 1) << 2)) | ((((word_2 * 128) >> 3) & 1) << 3)) | ((((word_2 * 128) >> 4) & 1) << 4)) | ((((word_2 * 128) >> 5) & 1) << 5)) | ((((word_2 * 128) >> 6) & 1) << 6)) | ((((word_2 * 128) >> 7) & 1) << 1)) | ((((word_2 * 128) >> 8) & 1) << 8)) | ((((word_2 * 128) >> 9) & 1) << 9)) | ((((word_2 * 128) >> 10) & 1) << 10)) >> 7)];
      v_11 = P[((((((((((((0 | (((word_2 * 128) + 2) & 1)) | (((((word_2 * 128) + 2) >> 1) & 1) << 7)) | (((((word_2 * 128) + 2) >> 2) & 1) << 2)) | (((((word_2 * 128) + 2) >> 3) & 1) << 3)) | (((((word_2 * 128) + 2) >> 4) & 1) << 4)) | (((((word_2 * 128) + 2) >> 5) & 1) << 5)) | (((((word_2 * 128) + 2) >> 6) & 1) << 6)) | (((((word_2 * 128) + 2) >> 7) & 1) << 1)) | (((((word_2 * 128) + 2) >> 8) & 1) << 8)) | (((((word_2 * 128) + 2) >> 9) & 1) << 9)) | (((((word_2 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
      v_12 = wide_packet_prmt(v_10, v_11, (((((uint)0 | (((uint)((((((((((((0 | (((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_2 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
      uint _reinterpret_tmp = v_12;
      S[(((((channel_1 * 512) + ((((int)threadIdx.x) >> 5) * 256)) + ((word_2 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_2 & 3))] = (*(int *)(&(_reinterpret_tmp)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int slab_1 = 0; slab_1 < 2; ++slab_1) {
    #pragma unroll
    for (int m_11 = 0; m_11 < 2; ++m_11) {
      #pragma unroll
      for (int n_6 = 0; n_6 < 12; ++n_6) {
        Z[((m_11 * 24) + (n_6 * 2))] = (uint)0;
        Z[(((m_11 * 24) + (n_6 * 2)) + 1)] = (uint)0;
      }
    }
    for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
      #pragma unroll
      for (int m_12 = 0; m_12 < 2; ++m_12) {
        *(int4*)(a_1 + (m_12 * 4)) = *(int4*)(S + ((((kp_1 * 512) + (slab_1 * 256)) + (m_12 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
      }
      #pragma unroll
      for (int pair_3 = 0; pair_3 < 6; ++pair_3) {
        for (int j_s_3 = 0; j_s_3 < 4; ++j_s_3) {
          int condval_7;
          if (((((((kp_1 * 1536) + ((((int)threadIdx.x) >> 5) * 768)) + (pair_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_3) < wns_3)) {
            condval_7 = WQ[(((((kp_1 * 1536) + ((((int)threadIdx.x) >> 5) * 768)) + (pair_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_3)];
          } else {
            condval_7 = 0;
          }
          b_3[j_s_3] = condval_7;
        }
        #pragma unroll
        for (int m_13 = 0; m_13 < 2; ++m_13) {
          #pragma unroll
          for (int half_3 = 0; half_3 < 2; ++half_3) {
            uint64_t r_3 = wide_mma(a_1[(m_13 * 4)], a_1[((m_13 * 4) + 1)], a_1[((m_13 * 4) + 2)], a_1[((m_13 * 4) + 3)], b_3[(half_3 * 2)], b_3[((half_3 * 2) + 1)], Z[(((m_13 * 24) + (pair_3 * 4)) + (half_3 * 2))], Z[((((m_13 * 24) + (pair_3 * 4)) + (half_3 * 2)) + 1)]);
            Z[(((m_13 * 24) + (pair_3 * 4)) + (half_3 * 2))] = ((uint)r_3);
            Z[((((m_13 * 24) + (pair_3 * 4)) + (half_3 * 2)) + 1)] = ((uint)(r_3 >> (uint64_t)32));
          }
        }
      }
    }
    #pragma unroll
    for (int comp = 0; comp < 3; ++comp) {
      if (comp == 0) {
        #pragma unroll
        for (int m_14 = 0; m_14 < 2; ++m_14) {
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            uint a_3 = Z[(((m_14 * 24) + (comp * 8)) + j)];
            uint b_5 = Z[((((m_14 * 24) + (comp * 8)) + j) + 2)];
            uint c = Z[((((m_14 * 24) + (comp * 8)) + j) + 4)];
            uint d = Z[((((m_14 * 24) + (comp * 8)) + j) + 6)];
            uint s0 = wide_add(wide_fma(a_3, a_3, wide_mul(c, c)), wide_fma(b_5, b_5, wide_mul(d, d)));
            uint s1 = wide_add(s0, wide_shfl(s0, ((((int)threadIdx.x) & 31) ^ 2)));
            uint s2 = wide_add(s1, wide_shfl(s1, ((((int)threadIdx.x) & 31) ^ 1)));
            uint total = wide_max(wide_add(s2, ((s2 >> (uint)16) | (s2 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
            uint inv = wide_rsqrt(total);
            #pragma unroll
            for (int n_7 = 0; n_7 < 4; ++n_7) {
              Z[((((m_14 * 24) + (comp * 8)) + (n_7 * 2)) + j)] = wide_mul(Z[((((m_14 * 24) + (comp * 8)) + (n_7 * 2)) + j)], inv);
              int v__4 = WS[0];
              uint sraw = (*(uint *)(&(v__4)));
              uint sbits = ((sraw >> ((uint)((((int)threadIdx.x) >> 5) * 16))) & (uint)65535);
              Z[(((m_14 * 24) + (n_7 * 2)) + j)] = wide_mul(Z[(((m_14 * 24) + (n_7 * 2)) + j)], (((sraw >> ((uint)((((int)threadIdx.x) >> 5) * 16))) & (uint)65535) | (((sraw >> ((uint)((((int)threadIdx.x) >> 5) * 16))) & (uint)65535) << (uint)16)));
            }
          }
          #pragma unroll
          for (int n_8 = 0; n_8 < 4; ++n_8) {
            QP[(((slab_1 * 8) + (m_14 * 4)) + n_8)] = wide_encode(Z[(((m_14 * 24) + (comp * 8)) + (n_8 * 2))], Z[((((m_14 * 24) + (comp * 8)) + (n_8 * 2)) + 1)]);
          }
        }
      } else {
        #pragma unroll
        for (int m_15 = 0; m_15 < 2; ++m_15) {
          if (comp == 1) {
            #pragma unroll
            for (int j_1 = 0; j_1 < 2; ++j_1) {
              uint a_4 = Z[(((m_15 * 24) + (comp * 8)) + j_1)];
              uint b_6 = Z[((((m_15 * 24) + (comp * 8)) + j_1) + 2)];
              uint c_1 = Z[((((m_15 * 24) + (comp * 8)) + j_1) + 4)];
              uint d_1 = Z[((((m_15 * 24) + (comp * 8)) + j_1) + 6)];
              uint s0_1 = wide_add(wide_fma(a_4, a_4, wide_mul(c_1, c_1)), wide_fma(b_6, b_6, wide_mul(d_1, d_1)));
              uint s1_1 = wide_add(s0_1, wide_shfl(s0_1, ((((int)threadIdx.x) & 31) ^ 2)));
              uint s2_1 = wide_add(s1_1, wide_shfl(s1_1, ((((int)threadIdx.x) & 31) ^ 1)));
              uint total_1 = wide_max(wide_add(s2_1, ((s2_1 >> (uint)16) | (s2_1 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
              uint inv_1 = wide_rsqrt(total_1);
              #pragma unroll
              for (int n_9 = 0; n_9 < 4; ++n_9) {
                Z[((((m_15 * 24) + (comp * 8)) + (n_9 * 2)) + j_1)] = wide_mul(Z[((((m_15 * 24) + (comp * 8)) + (n_9 * 2)) + j_1)], inv_1);
              }
            }
            #pragma unroll
            for (int n_10 = 0; n_10 < 4; ++n_10) {
              KP[(((slab_1 * 8) + (m_15 * 4)) + n_10)] = wide_encode(Z[(((m_15 * 24) + (comp * 8)) + (n_10 * 2))], Z[((((m_15 * 24) + (comp * 8)) + (n_10 * 2)) + 1)]);
            }
          } else {
            #pragma unroll
            for (int n_11 = 0; n_11 < 4; ++n_11) {
              VP[(((slab_1 * 8) + (m_15 * 4)) + n_11)] = wide_encode(Z[(((m_15 * 24) + (comp * 8)) + (n_11 * 2))], Z[((((m_15 * 24) + (comp * 8)) + (n_11 * 2)) + 1)]);
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int word_3 = 0; word_3 < 16; ++word_3) {
    v_13 = QP[((((((((((((0 | ((word_3 * 128) & 1)) | ((((word_3 * 128) >> 1) & 1) << 7)) | ((((word_3 * 128) >> 2) & 1) << 2)) | ((((word_3 * 128) >> 3) & 1) << 3)) | ((((word_3 * 128) >> 4) & 1) << 4)) | ((((word_3 * 128) >> 5) & 1) << 5)) | ((((word_3 * 128) >> 6) & 1) << 6)) | ((((word_3 * 128) >> 7) & 1) << 1)) | ((((word_3 * 128) >> 8) & 1) << 8)) | ((((word_3 * 128) >> 9) & 1) << 9)) | ((((word_3 * 128) >> 10) & 1) << 10)) >> 7)];
    v_14 = QP[((((((((((((0 | (((word_3 * 128) + 2) & 1)) | (((((word_3 * 128) + 2) >> 1) & 1) << 7)) | (((((word_3 * 128) + 2) >> 2) & 1) << 2)) | (((((word_3 * 128) + 2) >> 3) & 1) << 3)) | (((((word_3 * 128) + 2) >> 4) & 1) << 4)) | (((((word_3 * 128) + 2) >> 5) & 1) << 5)) | (((((word_3 * 128) + 2) >> 6) & 1) << 6)) | (((((word_3 * 128) + 2) >> 7) & 1) << 1)) | (((((word_3 * 128) + 2) >> 8) & 1) << 8)) | (((((word_3 * 128) + 2) >> 9) & 1) << 9)) | (((((word_3 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
    v_15 = wide_packet_prmt(v_13, v_14, (((((uint)0 | (((uint)((((((((((((0 | (((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    Q[word_3] = v_15;
    v_16 = KP[((((((((((((0 | ((word_3 * 128) & 1)) | ((((word_3 * 128) >> 1) & 1) << 7)) | ((((word_3 * 128) >> 2) & 1) << 2)) | ((((word_3 * 128) >> 3) & 1) << 3)) | ((((word_3 * 128) >> 4) & 1) << 4)) | ((((word_3 * 128) >> 5) & 1) << 5)) | ((((word_3 * 128) >> 6) & 1) << 6)) | ((((word_3 * 128) >> 7) & 1) << 8)) | ((((word_3 * 128) >> 8) & 1) << 1)) | ((((word_3 * 128) >> 9) & 1) << 9)) | ((((word_3 * 128) >> 10) & 1) << 10)) >> 7)];
    v_17 = KP[((((((((((((0 | (((word_3 * 128) + 2) & 1)) | (((((word_3 * 128) + 2) >> 1) & 1) << 7)) | (((((word_3 * 128) + 2) >> 2) & 1) << 2)) | (((((word_3 * 128) + 2) >> 3) & 1) << 3)) | (((((word_3 * 128) + 2) >> 4) & 1) << 4)) | (((((word_3 * 128) + 2) >> 5) & 1) << 5)) | (((((word_3 * 128) + 2) >> 6) & 1) << 6)) | (((((word_3 * 128) + 2) >> 7) & 1) << 8)) | (((((word_3 * 128) + 2) >> 8) & 1) << 1)) | (((((word_3 * 128) + 2) >> 9) & 1) << 9)) | (((((word_3 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
    v_18 = wide_packet_prmt(v_16, v_17, (((((uint)0 | (((uint)((((((((((((0 | (((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 8)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    K[word_3] = v_18;
    v_19 = wide_shfl(VP[((((((((((((0 | (((word_3 * 128) & 1) << 4)) | ((((word_3 * 128) >> 1) & 1) << 1)) | ((((word_3 * 128) >> 2) & 1) << 5)) | ((((word_3 * 128) >> 3) & 1) << 6)) | (((word_3 * 128) >> 4) & 1)) | ((((word_3 * 128) >> 5) & 1) << 2)) | ((((word_3 * 128) >> 6) & 1) << 3)) | ((((word_3 * 128) >> 7) & 1) << 9)) | ((((word_3 * 128) >> 8) & 1) << 7)) | ((((word_3 * 128) >> 9) & 1) << 8)) | ((((word_3 * 128) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1) << 4)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 5)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 6)) | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 2)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 3)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 9)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 8)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 127) >> 2));
    v_20 = wide_shfl(VP[((((((((((((0 | ((((word_3 * 128) + 1) & 1) << 4)) | (((((word_3 * 128) + 1) >> 1) & 1) << 1)) | (((((word_3 * 128) + 1) >> 2) & 1) << 5)) | (((((word_3 * 128) + 1) >> 3) & 1) << 6)) | ((((word_3 * 128) + 1) >> 4) & 1)) | (((((word_3 * 128) + 1) >> 5) & 1) << 2)) | (((((word_3 * 128) + 1) >> 6) & 1) << 3)) | (((((word_3 * 128) + 1) >> 7) & 1) << 9)) | (((((word_3 * 128) + 1) >> 8) & 1) << 7)) | (((((word_3 * 128) + 1) >> 9) & 1) << 8)) | (((((word_3 * 128) + 1) >> 10) & 1) << 10)) >> 7)], (((((((((((((0 | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 127) >> 2));
    v_21 = wide_packet_prmt(v_19, v_20, (((((uint)0 | (((uint)((((((((((((0 | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1) << 4)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 5)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 6)) | ((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 2)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 3)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 9)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 7)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 8)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)(((((((((((((0 | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3) + 4)) << (uint)4)) | (((uint)((((((((((((0 | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3)) << (uint)8)) | (((uint)(((((((((((((0 | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1) << 4)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 5)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 6)) | (((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 2)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 3)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 9)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 7)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 8)) | ((((((word_3 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
    V[word_3] = v_21;
  }
  __syncthreads();
  for (int slab_2 = 0; slab_2 < 2; ++slab_2) {
    #pragma unroll
    for (int m_16 = 0; m_16 < 2; ++m_16) {
      #pragma unroll
      for (int n_12 = 0; n_12 < 8; ++n_12) {
        int condval_8;
        if (((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_16 * 512)) + ((n_12 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_12 & 1) * 2)) < wns_4)) {
          condval_8 = WB[(((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_16 * 512)) + ((n_12 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_12 & 1) * 2))];
        } else {
          condval_8 = 0;
        }
        int condval_9;
        if ((((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_16 * 512)) + ((n_12 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_12 & 1) * 2)) + 1) < wns_4)) {
          condval_9 = WB[((((((((((int)threadIdx.x) >> 5) * 2048) + (slab_2 * 1024)) + (m_16 * 512)) + ((n_12 >> 1) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + ((n_12 & 1) * 2)) + 1)];
        } else {
          condval_9 = 0;
        }
        uint64_t r_4 = wide_mma(Q[((slab_2 * 8) + (m_16 * 4))], Q[(((slab_2 * 8) + (m_16 * 4)) + 1)], Q[(((slab_2 * 8) + (m_16 * 4)) + 2)], Q[(((slab_2 * 8) + (m_16 * 4)) + 3)], K[(n_12 * 2)], K[((n_12 * 2) + 1)], condval_8, condval_9);
        L[((m_16 * 16) + (n_12 * 2))] = (((wide_min(wide_max(wide_fma(((uint)r_4), wide_splat(0x1.7p-5f/*4.492188e-02*/), wide_splat(0x1.4dp+0f/*1.300781e+00*/)), wide_splat(0x1.08p+0f/*1.031250e+00*/)), wide_splat(0x1.91cp+0f/*1.569336e+00*/)) << (uint)5) & (uint)4292935648) ^ (uint)2147516416);
        L[(((m_16 * 16) + (n_12 * 2)) + 1)] = (((wide_min(wide_max(wide_fma(((uint)(r_4 >> (uint64_t)32)), wide_splat(0x1.7p-5f/*4.492188e-02*/), wide_splat(0x1.4dp+0f/*1.300781e+00*/)), wide_splat(0x1.08p+0f/*1.031250e+00*/)), wide_splat(0x1.91cp+0f/*1.569336e+00*/)) << (uint)5) & (uint)4292935648) ^ (uint)2147516416);
      }
      #pragma unroll
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        uint g0 = wide_add(L[((m_16 * 16) + j_2)], L[(((m_16 * 16) + j_2) + 2)]);
        uint g1 = wide_add(g0, wide_add(L[(((m_16 * 16) + j_2) + 4)], L[(((m_16 * 16) + j_2) + 6)]));
        uint g2 = wide_add(g1, wide_add(L[(((m_16 * 16) + j_2) + 8)], L[(((m_16 * 16) + j_2) + 10)]));
        uint group = wide_add(g2, wide_add(L[(((m_16 * 16) + j_2) + 12)], L[(((m_16 * 16) + j_2) + 14)]));
        uint t0 = wide_shfl(group, ((((int)threadIdx.x) & 31) & -4));
        uint t1 = wide_add(t0, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 1)));
        uint t2 = wide_add(t1, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 2)));
        uint t3 = wide_add(t2, wide_shfl(group, (((((int)threadIdx.x) & 31) & -4) + 3)));
        uint total_2 = wide_max(wide_add(t3, ((t3 >> (uint)16) | (t3 << (uint)16))), wide_splat(0x1.04p-14f/*6.198883e-05*/));
        uint inv_2 = wide_rcp(total_2);
        #pragma unroll
        for (int n_13 = 0; n_13 < 8; ++n_13) {
          L[(((m_16 * 16) + (n_13 * 2)) + j_2)] = wide_mul(L[(((m_16 * 16) + (n_13 * 2)) + j_2)], inv_2);
        }
      }
      #pragma unroll
      for (int n_14 = 0; n_14 < 8; ++n_14) {
        P[((m_16 * 8) + n_14)] = wide_encode(L[((m_16 * 16) + (n_14 * 2))], L[(((m_16 * 16) + (n_14 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int word_4 = 0; word_4 < 16; ++word_4) {
      v_22 = P[((((((((((((0 | ((word_4 * 128) & 1)) | ((((word_4 * 128) >> 1) & 1) << 7)) | ((((word_4 * 128) >> 2) & 1) << 2)) | ((((word_4 * 128) >> 3) & 1) << 3)) | ((((word_4 * 128) >> 4) & 1) << 4)) | ((((word_4 * 128) >> 5) & 1) << 5)) | ((((word_4 * 128) >> 6) & 1) << 6)) | ((((word_4 * 128) >> 7) & 1) << 1)) | ((((word_4 * 128) >> 8) & 1) << 8)) | ((((word_4 * 128) >> 9) & 1) << 10)) | ((((word_4 * 128) >> 10) & 1) << 9)) >> 7)];
      v_23 = P[((((((((((((0 | (((word_4 * 128) + 2) & 1)) | (((((word_4 * 128) + 2) >> 1) & 1) << 7)) | (((((word_4 * 128) + 2) >> 2) & 1) << 2)) | (((((word_4 * 128) + 2) >> 3) & 1) << 3)) | (((((word_4 * 128) + 2) >> 4) & 1) << 4)) | (((((word_4 * 128) + 2) >> 5) & 1) << 5)) | (((((word_4 * 128) + 2) >> 6) & 1) << 6)) | (((((word_4 * 128) + 2) >> 7) & 1) << 1)) | (((((word_4 * 128) + 2) >> 8) & 1) << 8)) | (((((word_4 * 128) + 2) >> 9) & 1) << 10)) | (((((word_4 * 128) + 2) >> 10) & 1) << 9)) >> 7)];
      v_24 = wide_packet_prmt(v_22, v_23, (((((uint)0 | (((uint)((((((((((((0 | (((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 10)) | (((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 9)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 10)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 9)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 10)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 9)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 10)) | ((((((word_4 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 9)) & 3) + 4)) << (uint)12)));
      A[word_4] = v_24;
    }
    #pragma unroll
    for (int m_17 = 0; m_17 < 2; ++m_17) {
      #pragma unroll
      for (int n_15 = 0; n_15 < 4; ++n_15) {
        acc[0] = (uint)0;
        acc[1] = (uint)0;
        #pragma unroll
        for (int kp_2 = 0; kp_2 < 2; ++kp_2) {
          uint64_t r_5 = wide_mma(A[((kp_2 * 8) + (m_17 * 4))], A[(((kp_2 * 8) + (m_17 * 4)) + 1)], A[(((kp_2 * 8) + (m_17 * 4)) + 2)], A[(((kp_2 * 8) + (m_17 * 4)) + 3)], V[((kp_2 * 8) + (n_15 * 2))], V[(((kp_2 * 8) + (n_15 * 2)) + 1)], acc[0], acc[1]);
          acc[0] = ((uint)r_5);
          acc[1] = ((uint)(r_5 >> (uint64_t)32));
        }
        P[((m_17 * 4) + n_15)] = wide_encode(acc[0], acc[1]);
      }
    }
    #pragma unroll
    for (int word_5 = 0; word_5 < 8; ++word_5) {
      int v__5 = S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_5 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_5 & 3) >> 1) * 2))];
      int v__6 = S[(((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_5 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (((word_5 & 3) >> 1) * 2)) + 1)];
      int condval_10;
      if (((word_5 % 2) == 0)) {
        condval_10 = 21520;
      } else {
        condval_10 = 30258;
      }
      Residual[word_5] = wide_packet_prmt((*(uint *)(&(v__5))), (*(uint *)(&(v__6))), condval_10);
    }
    __syncthreads();
    #pragma unroll
    for (int word_6 = 0; word_6 < 8; ++word_6) {
      v_25 = P[((((((((((((0 | ((word_6 * 128) & 1)) | ((((word_6 * 128) >> 1) & 1) << 7)) | ((((word_6 * 128) >> 2) & 1) << 2)) | ((((word_6 * 128) >> 3) & 1) << 3)) | ((((word_6 * 128) >> 4) & 1) << 4)) | ((((word_6 * 128) >> 5) & 1) << 5)) | ((((word_6 * 128) >> 6) & 1) << 6)) | ((((word_6 * 128) >> 7) & 1) << 1)) | ((((word_6 * 128) >> 8) & 1) << 8)) | ((((word_6 * 128) >> 9) & 1) << 9)) | ((((word_6 * 128) >> 10) & 1) << 10)) >> 7)];
      v_26 = P[((((((((((((0 | (((word_6 * 128) + 2) & 1)) | (((((word_6 * 128) + 2) >> 1) & 1) << 7)) | (((((word_6 * 128) + 2) >> 2) & 1) << 2)) | (((((word_6 * 128) + 2) >> 3) & 1) << 3)) | (((((word_6 * 128) + 2) >> 4) & 1) << 4)) | (((((word_6 * 128) + 2) >> 5) & 1) << 5)) | (((((word_6 * 128) + 2) >> 6) & 1) << 6)) | (((((word_6 * 128) + 2) >> 7) & 1) << 1)) | (((((word_6 * 128) + 2) >> 8) & 1) << 8)) | (((((word_6 * 128) + 2) >> 9) & 1) << 9)) | (((((word_6 * 128) + 2) >> 10) & 1) << 10)) >> 7)];
      v_27 = wide_packet_prmt(v_25, v_26, (((((uint)0 | (((uint)((((((((((((0 | (((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10)) & 3)) << (uint)0)) | (((uint)((((((((((((0 | ((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10)) & 3)) << (uint)4)) | (((uint)(((((((((((((0 | ((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10)) & 3) + 4)) << (uint)8)) | (((uint)(((((((((((((0 | ((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | ((((((word_6 * 128) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10)) & 3) + 4)) << (uint)12)));
      uint _reinterpret_tmp_1 = v_27;
      S[((((((((int)threadIdx.x) >> 5) * 512) + (slab_2 * 256)) + ((word_6 >> 2) * 128)) + ((((int)threadIdx.x) & 31) * 4)) + (word_6 & 3))] = (*(int *)(&(_reinterpret_tmp_1)));
    }
    __syncthreads();
    #pragma unroll
    for (int m_18 = 0; m_18 < 2; ++m_18) {
      #pragma unroll
      for (int n_16 = 0; n_16 < 4; ++n_16) {
        int condval_11;
        if ((((((((int)threadIdx.x) >> 5) * 16) + (n_16 * 4)) + (((int)threadIdx.x) & 3)) < wns_8)) {
          condval_11 = AG[((((((int)threadIdx.x) >> 5) * 16) + (n_16 * 4)) + (((int)threadIdx.x) & 3))];
        } else {
          condval_11 = 0;
        }
        int g_1 = condval_11;
        C[((m_18 * 8) + (n_16 * 2))] = wide_mul(wide_decode(Residual[((m_18 * 4) + n_16)]), g_1);
        C[(((m_18 * 8) + (n_16 * 2)) + 1)] = wide_mul(wide_decode((Residual[((m_18 * 4) + n_16)] >> (uint)16)), g_1);
      }
    }
    for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
      #pragma unroll
      for (int m_19 = 0; m_19 < 2; ++m_19) {
        *(int4*)(a_2 + (m_19 * 4)) = *(int4*)(S + ((((kp_3 * 512) + (slab_2 * 256)) + (m_19 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
      }
      #pragma unroll
      for (int pair_4 = 0; pair_4 < 2; ++pair_4) {
        for (int j_s_4 = 0; j_s_4 < 4; ++j_s_4) {
          int condval_12;
          if (((((((kp_3 * 512) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_4) < wns_6)) {
            condval_12 = WP[(((((kp_3 * 512) + ((((int)threadIdx.x) >> 5) * 256)) + (pair_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + j_s_4)];
          } else {
            condval_12 = 0;
          }
          b_4[j_s_4] = condval_12;
        }
        #pragma unroll
        for (int m_20 = 0; m_20 < 2; ++m_20) {
          #pragma unroll
          for (int half_4 = 0; half_4 < 2; ++half_4) {
            uint64_t r_6 = wide_mma(a_2[(m_20 * 4)], a_2[((m_20 * 4) + 1)], a_2[((m_20 * 4) + 2)], a_2[((m_20 * 4) + 3)], b_4[(half_4 * 2)], b_4[((half_4 * 2) + 1)], C[(((m_20 * 8) + (pair_4 * 4)) + (half_4 * 2))], C[((((m_20 * 8) + (pair_4 * 4)) + (half_4 * 2)) + 1)]);
            C[(((m_20 * 8) + (pair_4 * 4)) + (half_4 * 2))] = ((uint)r_6);
            C[((((m_20 * 8) + (pair_4 * 4)) + (half_4 * 2)) + 1)] = ((uint)(r_6 >> (uint64_t)32));
          }
        }
      }
    }
    #pragma unroll
    for (int m_21 = 0; m_21 < 2; ++m_21) {
      #pragma unroll
      for (int n_17 = 0; n_17 < 4; ++n_17) {
        P[((m_21 * 4) + n_17)] = wide_encode(C[((m_21 * 8) + (n_17 * 2))], C[(((m_21 * 8) + (n_17 * 2)) + 1)]);
      }
    }
    #pragma unroll
    for (int word_7 = 0; word_7 < 8; ++word_7) {
      QP[((slab_2 * 8) + word_7)] = P[word_7];
    }
    __syncthreads();
  }
  #pragma unroll
  for (int part = 0; part < 4; ++part) {
    #pragma unroll
    for (int j_3 = 0; j_3 < 4; ++j_3) {
      v_28 = (((((((((((0 | ((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10));
      v_29 = wide_shfl(QP[(((((((((((((0 | (((part * 512) + (j_3 * 128)) & 1)) | (((((part * 512) + (j_3 * 128)) >> 1) & 1) << 7)) | (((((part * 512) + (j_3 * 128)) >> 2) & 1) << 2)) | (((((part * 512) + (j_3 * 128)) >> 3) & 1) << 3)) | (((((part * 512) + (j_3 * 128)) >> 4) & 1) << 4)) | (((((part * 512) + (j_3 * 128)) >> 5) & 1) << 5)) | (((((part * 512) + (j_3 * 128)) >> 6) & 1) << 6)) | (((((part * 512) + (j_3 * 128)) >> 7) & 1) << 1)) | (((((part * 512) + (j_3 * 128)) >> 8) & 1) << 8)) | (((((part * 512) + (j_3 * 128)) >> 9) & 1) << 9)) | (((((part * 512) + (j_3 * 128)) >> 10) & 1) << 10)) >> 7) | 0)], ((v_28 & 127) >> 2));
      uint condval_13;
      if (((v_28 >> 7) == (((((((((((((0 | (((part * 512) + (j_3 * 128)) & 1)) | (((((part * 512) + (j_3 * 128)) >> 1) & 1) << 7)) | (((((part * 512) + (j_3 * 128)) >> 2) & 1) << 2)) | (((((part * 512) + (j_3 * 128)) >> 3) & 1) << 3)) | (((((part * 512) + (j_3 * 128)) >> 4) & 1) << 4)) | (((((part * 512) + (j_3 * 128)) >> 5) & 1) << 5)) | (((((part * 512) + (j_3 * 128)) >> 6) & 1) << 6)) | (((((part * 512) + (j_3 * 128)) >> 7) & 1) << 1)) | (((((part * 512) + (j_3 * 128)) >> 8) & 1) << 8)) | (((((part * 512) + (j_3 * 128)) >> 9) & 1) << 9)) | (((((part * 512) + (j_3 * 128)) >> 10) & 1) << 10)) >> 7) | 0))) {
        condval_13 = v_29;
      } else {
        condval_13 = (uint)0;
      }
      v_30 = condval_13;
      v_31 = (((((((((((0 | (((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) & 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 1) & 1) << 7)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 2) & 1) << 2)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 3) & 1) << 3)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 4) & 1) << 4)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 5) & 1) << 5)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 6) & 1) << 6)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 7) & 1) << 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 8) & 1) << 8)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 9) & 1) << 9)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 1) >> 10) & 1) << 10));
      v_32 = wide_shfl(QP[(((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 1) & 1)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 10) & 1) << 10)) >> 7) | 0)], ((v_31 & 127) >> 2));
      uint condval_14;
      if (((v_31 >> 7) == (((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 1) & 1)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 1) >> 10) & 1) << 10)) >> 7) | 0))) {
        condval_14 = v_32;
      } else {
        condval_14 = (uint)0;
      }
      v_33 = condval_14;
      v_34 = (((((((((((0 | (((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) & 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 1) & 1) << 7)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 2) & 1) << 2)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 3) & 1) << 3)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 4) & 1) << 4)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 5) & 1) << 5)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 6) & 1) << 6)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 7) & 1) << 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 8) & 1) << 8)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 9) & 1) << 9)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 2) >> 10) & 1) << 10));
      v_35 = wide_shfl(QP[(((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 2) & 1)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 10) & 1) << 10)) >> 7) | 0)], ((v_34 & 127) >> 2));
      uint condval_15;
      if (((v_34 >> 7) == (((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 2) & 1)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 2) >> 10) & 1) << 10)) >> 7) | 0))) {
        condval_15 = v_35;
      } else {
        condval_15 = (uint)0;
      }
      v_36 = condval_15;
      v_37 = (((((((((((0 | (((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) & 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 1) & 1) << 7)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 2) & 1) << 2)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 3) & 1) << 3)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 4) & 1) << 4)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 5) & 1) << 5)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 6) & 1) << 6)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 7) & 1) << 1)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 8) & 1) << 8)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 9) & 1) << 9)) | (((((((part * 512) + (j_3 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 3) >> 10) & 1) << 10));
      v_38 = wide_shfl(QP[(((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 3) & 1)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 10) & 1) << 10)) >> 7) | 0)], ((v_37 & 127) >> 2));
      uint condval_16;
      if (((v_37 >> 7) == (((((((((((((0 | ((((part * 512) + (j_3 * 128)) + 3) & 1)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 1) & 1) << 7)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 2) & 1) << 2)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 3) & 1) << 3)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 4) & 1) << 4)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 5) & 1) << 5)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 6) & 1) << 6)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 7) & 1) << 1)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 8) & 1) << 8)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 9) & 1) << 9)) | ((((((part * 512) + (j_3 * 128)) + 3) >> 10) & 1) << 10)) >> 7) | 0))) {
        condval_16 = v_38;
      } else {
        condval_16 = (uint)0;
      }
      v_39 = condval_16;
      A[((part * 4) + j_3)] = (((((uint)0 | (((v_30 >> ((uint)((v_28 & 3) * 8))) & (uint)255) << (uint)0)) | (((v_33 >> ((uint)((v_31 & 3) * 8))) & (uint)255) << (uint)8)) | (((v_36 >> ((uint)((v_34 & 3) * 8))) & (uint)255) << (uint)16)) | (((v_39 >> ((uint)((v_37 & 3) * 8))) & (uint)255) << (uint)24));
    }
    v_40 = (((((((((((0 | (((part * 512) + ((((int)threadIdx.x) & 31) * 4)) & 1)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 1) & 1) << 7)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 2) & 1) << 2)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 3) & 1) << 3)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 4) & 1) << 4)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 5) & 1) << 5)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 6) & 1) << 6)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 7) & 1) << 1)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 8) & 1) << 8)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 9) & 1) << 9)) | (((((part * 512) + ((((int)threadIdx.x) & 31) * 4)) >> 10) & 1) << 10));
    int ix = v_40;
    v_41 = (((((((((((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 49) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 6) << 1)) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 8) >> 2)) & 3) | ((((((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 49) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 6) << 1)) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 8) >> 2)) & 12) << 2)) | ((((((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 49) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 6) << 1)) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 8) >> 2)) & 16) >> 1)) | ((((((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 49) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 6) << 1)) | (((((((((int)threadIdx.x) >> 5) * 32) + (((ix & 511) >> 7) * 8)) + (((ix & 15) >> 2) * 2)) + (ix & 1)) & 8) >> 2)) & 32) << 4)) | ((((((ix >> 9) * 16) + (((ix & 3) >> 1) * 8)) + ((ix & 127) >> 4)) & 7) << 6)) | ((((((ix >> 9) * 16) + (((ix & 3) >> 1) * 8)) + ((ix & 127) >> 4)) & 8) >> 1)) | ((((((ix >> 9) * 16) + (((ix & 3) >> 1) * 8)) + ((ix & 127) >> 4)) & 48) << 6));
    int rmod_2 = (((int)blockIdx.x) % gx);
    int rdiv_1 = (((int)blockIdx.x) / gx);
    v_42 = (((((((0 <= gx) && (0 <= rmod_2)) || ((gx < 0) && (rmod_2 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 2) + (v_41 >> 11)) + (sy >> 2));
    int rmod_3 = (((int)blockIdx.x) % gx);
    v_43 = (((((((0 <= gx) && (0 <= rmod_3)) || ((gx < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + gx)) * 2) + ((v_41 & 2047) >> 10)) + (sx >> 2));
    int condval_17;
    if (((((0 <= v_42) & (0 <= v_43)) & (v_42 < (height >> 2))) & (v_43 < (width >> 2)))) {
      condval_17 = (((((v_42 * (width >> 2)) * 1024) + (v_43 * 1024)) + (((v_41 >> 9) & 1) * 512)) + (v_41 & 511));
    } else {
      condval_17 = -1;
    }
    v_44 = condval_17;
    int raw_1 = v_44;
    if ((0 <= raw_1) & ((raw_1 + 16) <= (rn - out))) {
      {
        int RW_local_cast_2[4];
        int4 __3;
        uint4 v__7 = *(uint4*)(A + (part * 4));
        __3.x = (int)(v__7.x);
        __3.y = (int)(v__7.y);
        __3.z = (int)(v__7.z);
        __3.w = (int)(v__7.w);
        *(int4*)(RW_local_cast_2 + 0) = __3;
        if (0 <= (out + raw_1)) {
          *(int4*)(RW + (((out + raw_1) >> 4) * 4)) = *(int4*)(RW_local_cast_2 + 0);
        }
      }
    }
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

