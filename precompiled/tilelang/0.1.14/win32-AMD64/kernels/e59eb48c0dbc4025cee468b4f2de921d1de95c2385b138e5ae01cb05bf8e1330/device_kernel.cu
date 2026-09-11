#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
// Instruction-sized primitives only. All loops, addressing and algorithms are TileLang.
__device__ __forceinline__ unsigned nr_ld32(const unsigned char* p) { return *(const unsigned*)p; }
__device__ __forceinline__ void nr_st32(unsigned char* p, unsigned v) { *(unsigned*)p=v; }
__device__ __forceinline__ unsigned nr_vload32(const unsigned* p) {return *(const volatile unsigned*)p;}
__device__ __forceinline__ void nr_vstore32(unsigned* p,unsigned v) {*(volatile unsigned*)p=v;}
__device__ __forceinline__ unsigned nr_hbits(float h) { return __half_as_ushort(__float2half_rn(h)); }
__device__ __forceinline__ float nr_half(unsigned h) { return __half2float(__ushort_as_half((unsigned short)h)); }
__device__ __forceinline__ float nr_dec(unsigned char v) { return __half2float(half(__nv_cvt_fp8_to_halfraw(v,__NV_E4M3))); }
__device__ __forceinline__ unsigned char nr_enc(float v) { return __nv_cvt_halfraw_to_fp8((__half_raw)__float2half_rn(v),__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ unsigned nr_qpair(unsigned v) { union {unsigned u;half2 h;} x; x.u=v;return __nv_cvt_halfraw2_to_fp8x2((__half2_raw)x.h,__NV_SATFINITE,__NV_E4M3); }
__device__ __forceinline__ unsigned nr_decode2(unsigned v) { union {unsigned u;half2 h;} x; x.h=half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)v,__NV_E4M3));return x.u; }
__device__ __forceinline__ float nr_add(float a,float b) {return __half2float(__hadd(__float2half_rn(a),__float2half_rn(b)));}
__device__ __forceinline__ float nr_mul(float a,float b) {return __half2float(__hmul(__float2half_rn(a),__float2half_rn(b)));}
__device__ __forceinline__ float nr_fma(float a,float b,float c) {return __half2float(__hfma(__float2half_rn(a),__float2half_rn(b),__float2half_rn(c)));}
__device__ __forceinline__ unsigned nr_add2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hadd2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_mul2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmul2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_fma2(unsigned a,unsigned b,unsigned c) { union {unsigned u;half2 h;} x,y,z;x.u=a;y.u=b;z.u=c;x.h=__hfma2(x.h,y.h,z.h);return x.u; }
__device__ __forceinline__ unsigned nr_min2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmin2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_max2(unsigned a,unsigned b) { union {unsigned u;half2 h;} x,y;x.u=a;y.u=b;x.h=__hmax2(x.h,y.h);return x.u; }
__device__ __forceinline__ unsigned nr_dup(float v) { union {unsigned u;half2 h;} x;x.h=__float2half2_rn(v);return x.u; }
__device__ __forceinline__ unsigned nr_shuffle(unsigned v,int lane) {return __shfl_sync(0xffffffff,v,lane);}
__device__ __forceinline__ unsigned nr_xor(unsigned v,int mask) {return __shfl_xor_sync(0xffffffff,v,mask);}
__device__ __forceinline__ unsigned nr_perm(unsigned a,unsigned b,unsigned s) {return __byte_perm(a,b,s);}
__device__ __forceinline__ float nr_rcp(float x) {float y;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float nr_rsqrt(float x) {float y;asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ void nr_mma(unsigned* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(d[0]),"+r"(d[1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
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

extern "C" __global__ void main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, const int* __restrict__ Members, const int* __restrict__ Plan, uchar* __restrict__ Raw, const float* __restrict__ Scale, const uchar* __restrict__ Weight);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, const int* __restrict__ Members, const int* __restrict__ Plan, uchar* __restrict__ Raw, const float* __restrict__ Scale, const uchar* __restrict__ Weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* qkv = ((void*)((char*)buf_dyn_shmem + 0));
  void* sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* packed = ((void*)((char*)buf_dyn_shmem + 24576));
  uint c[96];
  uint av[4];
  uint v = (uint)0;
  uint word = (uint)0;
  uint result[8];
  uint score[16];
  uint z = (uint)0;
  uint s = (uint)0;
  uint total = (uint)0;
  for (int r = 0; r < 4; ++r) {
    for (int n = 0; n < 12; ++n) {
      for (int h = 0; h < 2; ++h) {
        c[(((r * 24) + (n * 2)) + h)] = (uint)0;
      }
    }
  }
  for (int p = 0; p < 8; ++p) {
    for (int u = 0; u < 2; ++u) {
      int off = Plan[(((((((int)blockIdx.y) * 6144) + (((int)blockIdx.x) * 2048)) + (p * 256)) + (u * 128)) + ((int)threadIdx.x))];
      for (int z_1 = 0; z_1 < 4; ++z_1) {
        uint condval;
        if ((0 <= off)) {
          condval = nr_ld32((&(A[((((int64_t)z_1) * (int64_t)4) + ((int64_t)off))])));
        } else {
          condval = (uint)0;
        }
        ((uint*)sh)[(((u * 512) + (((int)threadIdx.x) * 4)) + z_1)] = condval;
      }
    }
    __syncthreads();
    for (int k = 0; k < 2; ++k) {
      for (int r_1 = 0; r_1 < 4; ++r_1) {
        for (int z_2 = 0; z_2 < 4; ++z_2) {
          av[z_2] = ((uint*)sh)[((((r_1 * 256) + (k * 128)) + ((((int)threadIdx.x) & 31) * 4)) + z_2)];
        }
        for (int n_1 = 0; n_1 < 12; ++n_1) {
          uint b0 = nr_ld32((&(Weight[(((((((p * 98304) + (k * 49152)) + (((int)blockIdx.z) * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8))])));
          uint b1 = nr_ld32((&(Weight[((((((((p * 98304) + (k * 49152)) + (((int)blockIdx.z) * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8)) + 4)])));
          nr_mma((&(c[((r_1 * 24) + (n_1 * 2))])), av[0], av[1], av[2], av[3], b0, b1);
        }
      }
    }
    __syncthreads();
  }
  for (int r_2 = 0; r_2 < 4; ++r_2) {
    for (int co = 0; co < 3; ++co) {
      if (co < 2) {
        for (int h_1 = 0; h_1 < 2; ++h_1) {
          uint b = c[(((r_2 * 24) + (co * 8)) + h_1)];
          uint b_1 = c[((((r_2 * 24) + (co * 8)) + h_1) + 2)];
          uint b_2 = c[((((r_2 * 24) + (co * 8)) + h_1) + 4)];
          uint b_3 = c[((((r_2 * 24) + (co * 8)) + h_1) + 6)];
          uint c_1 = nr_mul2(b_2, b_2);
          uint c_2 = nr_mul2(b_3, b_3);
          uint a = nr_fma2(b, b, c_1);
          uint b_4 = nr_fma2(b_1, b_1, c_2);
          uint a_1 = nr_add2(a, b_4);
          uint b_5 = nr_xor(a_1, 2);
          uint a_2 = nr_add2(a_1, b_5);
          uint b_6 = nr_xor(a_2, 1);
          uint a_3 = nr_add2(a_2, b_6);
          uint a_4 = (a_3 >> (uint)16);
          half_t a_5 = ((half_t)nr_half(a_3));
          half_t b_7 = ((half_t)nr_half((a_3 >> (uint)16)));
          half_t den = cutlass::half_t(__hmax((((half_t)nr_add(((float)a_5), ((float)b_7)))).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half()));
          half_t inv = ((half_t)nr_rsqrt(((float)den)));
          for (int n_2 = 0; n_2 < 4; ++n_2) {
            float a_6 = ((float)inv);
            uint a_7 = c[((((r_2 * 24) + (co * 8)) + (n_2 * 2)) + h_1)];
            uint b_8 = nr_dup(((float)inv));
            v = nr_mul2(a_7, b_8);
            if (co == 0) {
              uint a_8 = v;
              uint b_9 = nr_dup(((float)((half_t)Scale[(((((int)blockIdx.z) * 4) + (((int)threadIdx.x) >> 5)) + 229376)])));
              v = nr_mul2(a_8, b_9);
            }
            c[((((r_2 * 24) + (co * 8)) + (n_2 * 2)) + h_1)] = v;
          }
        }
        for (int h_2 = 0; h_2 < 2; ++h_2) {
          for (int n_3 = 0; n_3 < 2; ++n_3) {
            uint a_9 = c[((((r_2 * 24) + (co * 8)) + (n_3 * 4)) + h_2)];
            uint a_10 = c[(((((r_2 * 24) + (co * 8)) + (n_3 * 4)) + h_2) + 2)];
            uint condval_1;
            if ((0 <= Members[((((((((int)blockIdx.y) * 192) + (((int)blockIdx.x) * 64)) + (r_2 * 16)) + (((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8)) + ((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1))])) {
              condval_1 = (nr_qpair(a_9) | (nr_qpair(a_10) << (uint)16));
            } else {
              condval_1 = (uint)0;
            }
            uint value = condval_1;
            nr_st32((&(((uchar*)qkv)[(((((((((((int)threadIdx.x) >> 5) * 6144) + (co * 2048)) + (r_2 * 512)) + (((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 256)) + ((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 64)) + (((((h_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1) * 32)) + (n_3 * 16)) + (((((int)threadIdx.x) & 31) & 3) * 4))])), value);
          }
        }
      } else {
        bool vx = (0 <= Members[((((((((int)blockIdx.y) * 192) + (((int)blockIdx.x) * 64)) + (r_2 * 16)) + (((((((int)threadIdx.x) & 31) >> 2) >> 2) & 1) * 8)) + ((((((int)threadIdx.x) & 31) >> 2) & 3) * 2)) + ((((((int)threadIdx.x) & 31) >> 2) >> 3) & 1))]);
        bool vy = (0 <= Members[((((((((int)blockIdx.y) * 192) + (((int)blockIdx.x) * 64)) + (r_2 * 16)) + ((((((((int)threadIdx.x) & 31) >> 2) + 8) >> 2) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 2) + 8) & 3) * 2)) + (((((((int)threadIdx.x) & 31) >> 2) + 8) >> 3) & 1))]);
        for (int n_4 = 0; n_4 < 4; ++n_4) {
          uint a_11 = c[(((r_2 * 24) + (n_4 * 2)) + 16)];
          uint a_12 = c[(((r_2 * 24) + (n_4 * 2)) + 17)];
          uint condval_2;
          if (vx) {
            condval_2 = nr_qpair(a_11);
          } else {
            condval_2 = (uint)0;
          }
          uint condval_3;
          if (vy) {
            condval_3 = nr_qpair(a_12);
          } else {
            condval_3 = (uint)0;
          }
          uint a_13 = (condval_2 | (condval_3 << (uint)16));
          word = (uint)0;
          for (int b_10 = 0; b_10 < 4; ++b_10) {
            uint val = nr_shuffle(a_13, ((((((((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 1) << 3) + (((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 6) >> 1)) + (((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 8) >> 1)) & 7) * 4) + ((((int)threadIdx.x) & 31) >> 3)));
            word = (word | (((val >> ((uint)((((((((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 1) << 3) + (((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 6) >> 1)) + (((((((((int)threadIdx.x) & 31) & 3) * 4) + ((b_10 & 1) << 1)) + ((b_10 & 2) >> 1)) & 8) >> 1)) >> 3) * 16) + ((((((int)threadIdx.x) & 31) >> 2) & 1) * 8)))) & (uint)255) << ((uint)(b_10 * 8))));
          }
          nr_st32((&(((uchar*)qkv)[((((((((int)threadIdx.x) >> 5) * 6144) + (r_2 * 512)) + (n_4 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 4096)])), word);
        }
      }
    }
  }
  __syncthreads();
  for (int r_3 = 0; r_3 < 4; ++r_3) {
    for (int n_5 = 0; n_5 < 4; ++n_5) {
      for (int h_3 = 0; h_3 < 2; ++h_3) {
        result[((n_5 * 2) + h_3)] = (uint)0;
      }
    }
    for (int z_3 = 0; z_3 < 4; ++z_3) {
      av[z_3] = nr_ld32((&(((uchar*)qkv)[(((((((((int)threadIdx.x) >> 5) * 6144) + (r_3 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((z_3 & 1) * 32)) + ((z_3 >> 1) * 16)) + (((((int)threadIdx.x) & 31) & 3) * 4))])));
    }
    for (int n_6 = 0; n_6 < 8; ++n_6) {
      for (int h_4 = 0; h_4 < 2; ++h_4) {
        score[((n_6 * 2) + h_4)] = nr_ld32((&(Weight[((((((((((int)blockIdx.z) * 32768) + ((((int)threadIdx.x) >> 5) * 8192)) + (r_3 * 2048)) + ((n_6 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_6 & 1) * 8)) + (h_4 * 4)) + 786432)])));
      }
      uint b0_1 = nr_ld32((&(((uchar*)qkv)[(((((((((int)threadIdx.x) >> 5) * 6144) + ((n_6 >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((n_6 & 1) * 32)) + (((((int)threadIdx.x) & 31) & 3) * 4)) + 2048)])));
      uint b1_1 = nr_ld32((&(((uchar*)qkv)[(((((((((int)threadIdx.x) >> 5) * 6144) + ((n_6 >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((n_6 & 1) * 32)) + (((((int)threadIdx.x) & 31) & 3) * 4)) + 2064)])));
      nr_mma((&(score[(n_6 * 2)])), av[0], av[1], av[2], av[3], b0_1, b1_1);
      for (int h_5 = 0; h_5 < 2; ++h_5) {
        uint a_14 = score[((n_6 * 2) + h_5)];
        uint b_11 = nr_dup(0x1.7p-5f/*4.492188e-02*/);
        uint c_3 = nr_dup(0x1.4dp+0f/*1.300781e+00*/);
        z = nr_fma2(a_14, b_11, c_3);
        z = nr_min2(nr_max2(z, nr_dup(0x1.08p+0f/*1.031250e+00*/)), nr_dup(0x1.91cp+0f/*1.569336e+00*/));
        z = (((z & (uint)134154239) << (uint)5) ^ (uint)2147516416);
        score[((n_6 * 2) + h_5)] = z;
      }
    }
    for (int h_6 = 0; h_6 < 2; ++h_6) {
      uint a_15 = score[h_6];
      uint b_12 = score[(h_6 + 2)];
      s = nr_add2(a_15, b_12);
      for (int n_7 = 1; n_7 < 4; ++n_7) {
        uint a_16 = score[((n_7 * 4) + h_6)];
        uint b_13 = score[(((n_7 * 4) + h_6) + 2)];
        uint a_17 = s;
        uint b_14 = nr_add2(a_16, b_13);
        s = nr_add2(a_17, b_14);
      }
      uint a_18 = s;
      total = nr_shuffle(a_18, ((((int)threadIdx.x) & 31) & -4));
      for (int n_8 = 1; n_8 < 4; ++n_8) {
        uint a_19 = s;
        uint a_20 = total;
        uint b_15 = nr_shuffle(a_19, (((((int)threadIdx.x) & 31) & -4) + n_8));
        total = nr_add2(a_20, b_15);
      }
      uint a_21 = total;
      uint a_22 = (total >> (uint)16);
      half_t a_23 = ((half_t)nr_half(a_21));
      half_t b_16 = ((half_t)nr_half(a_22));
      half_t b_17 = ((half_t)nr_add(((float)a_23), ((float)b_16)));
      half_t den_1 = ((half_t)nr_add(0x0p+0f/*0.000000e+00*/, ((float)b_17)));
      half_t inv_1 = ((half_t)nr_rcp(((float)cutlass::half_t(__hmax((den_1).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half())))));
      for (int n_9 = 0; n_9 < 8; ++n_9) {
        float a_24 = ((float)inv_1);
        uint a_25 = score[((n_9 * 2) + h_6)];
        uint b_18 = nr_dup(((float)inv_1));
        uint a_26 = nr_mul2(a_25, b_18);
        score[((n_9 * 2) + h_6)] = nr_qpair(a_26);
      }
    }
    for (int p_1 = 0; p_1 < 2; ++p_1) {
      for (int z_4 = 0; z_4 < 4; ++z_4) {
        av[z_4] = ((score[(((p_1 * 8) + ((z_4 >> 1) * 4)) + (z_4 & 1))] & (uint)65535) | ((score[((((p_1 * 8) + ((z_4 >> 1) * 4)) + (z_4 & 1)) + 2)] & (uint)65535) << (uint)16));
      }
      for (int n_10 = 0; n_10 < 4; ++n_10) {
        uint b0_2 = nr_ld32((&(((uchar*)qkv)[((((((((int)threadIdx.x) >> 5) * 6144) + (p_1 * 1024)) + (n_10 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 4096)])));
        uint b1_2 = nr_ld32((&(((uchar*)qkv)[((((((((int)threadIdx.x) >> 5) * 6144) + (p_1 * 1024)) + (n_10 * 128)) + ((((int)threadIdx.x) & 31) * 4)) + 4608)])));
        nr_mma((&(result[(n_10 * 2)])), av[0], av[1], av[2], av[3], b0_2, b1_2);
      }
    }
    for (int n_11 = 0; n_11 < 4; ++n_11) {
      for (int h_7 = 0; h_7 < 2; ++h_7) {
        uint a_27 = result[((n_11 * 2) + h_7)];
        uint q = nr_qpair(a_27);
        ((uchar*)packed)[(((((((((((int)threadIdx.x) >> 5) * 2048) + (r_3 * 512)) + (((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((((((int)threadIdx.x) & 31) >> 2) * 2) + h_7) & 1) << 2)) + ((((((((int)threadIdx.x) & 31) >> 2) * 2) + h_7) & 14) << 5))] = ((uchar)q);
        ((uchar*)packed)[((((((((((((int)threadIdx.x) >> 5) * 2048) + (r_3 * 512)) + (((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_11 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((((((int)threadIdx.x) & 31) >> 2) * 2) + h_7) & 1) << 2)) + ((((((((int)threadIdx.x) & 31) >> 2) * 2) + h_7) & 14) << 5)) + 1)] = ((uchar)(q >> (uint)8));
      }
    }
  }
  __syncthreads();
  for (int r_4 = 0; r_4 < 4; ++r_4) {
    int off_1 = Plan[((((((int)blockIdx.y) * 6144) + (((int)blockIdx.x) * 2048)) + (r_4 * 64)) + (((int)threadIdx.x) & 31))];
    if (0 <= off_1) {
      for (int z_5 = 0; z_5 < 4; ++z_5) {
        uint v_1 = nr_ld32((&(((uchar*)packed)[(((((((int)threadIdx.x) >> 5) * 2048) + (r_4 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z_5 * 4))])));
        if (((((((int)blockIdx.z) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + (z_5 * 4)) + off_1) < 163840) {
          nr_st32((&(Raw[((((((int)blockIdx.z) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + (z_5 * 4)) + off_1)])), v_1);
        }
      }
    }
  }
  bool first_block = (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0));
  if (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0)) {
    if (((int)threadIdx.x) < 24) {
      C0[((int)threadIdx.x)] = 0;
    }
  }
}

