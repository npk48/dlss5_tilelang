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

extern "C" __global__ void main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ Raw, const uchar* __restrict__ Weight, int H, int W, int n0, int n1, int sizes_3, int v0, int v1);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ Raw, const uchar* __restrict__ Weight, int H, int W, int n0, int n1, int sizes_3, int v0, int v1) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* packed = ((void*)((char*)buf_dyn_shmem + 0));
  void* route = ((void*)((char*)buf_dyn_shmem + 0));
  void* sh = ((void*)((char*)buf_dyn_shmem + 0));
  uint c[64];
  uint av[4];
  uint total[64];
  uint hidden[32];
  for (int r = 0; r < 4; ++r) {
    for (int n = 0; n < 8; ++n) {
      for (int h = 0; h < 2; ++h) {
        c[(((r * 16) + (n * 2)) + h)] = (uint)0;
      }
    }
  }
  for (int p = 0; p < 8; ++p) {
    for (int u = 0; u < 8; ++u) {
      int i = max((((((((int)blockIdx.x) * 128) + (((((u * 128) + ((int)threadIdx.x)) >> 8) & 1) * 64)) + (((((((((int)blockIdx.y) * 8) + (((((u * 128) + ((int)threadIdx.x)) >> 9) & 1) * 4)) + ((((u * 128) + ((int)threadIdx.x)) & 1) * 2)) + (W * (((p * 4) + (((((u * 128) + ((int)threadIdx.x)) >> 7) & 1) * 2)) + ((((u * 128) + ((int)threadIdx.x)) >> 1) & 1)))) + ((((u * 128) + ((int)threadIdx.x)) >> 6) & 1)) * H) * 16)) + (((((u * 128) + ((int)threadIdx.x)) >> 4) & 3) * 16)) + ((((((int)threadIdx.x) & 31) >> 2) & 3) * 4)), (((((((int)blockIdx.x) * 128) + (((((u * 128) + ((int)threadIdx.x)) >> 8) & 1) * 64)) + (((((((((int)blockIdx.y) * 8) + (((((u * 128) + ((int)threadIdx.x)) >> 9) & 1) * 4)) + ((((u * 128) + ((int)threadIdx.x)) & 1) * 2)) + (W * (((p * 4) + (((((u * 128) + ((int)threadIdx.x)) >> 7) & 1) * 2)) + ((((u * 128) + ((int)threadIdx.x)) >> 1) & 1)))) + ((((u * 128) + ((int)threadIdx.x)) >> 6) & 1)) * H) * 16)) + (((((u * 128) + ((int)threadIdx.x)) >> 4) & 3) * 16)) + ((((((int)threadIdx.x) & 31) >> 2) & 3) * 4)));
      uint condval;
      if ((((((((int)blockIdx.x) * 8) + (((((u * 128) + ((int)threadIdx.x)) >> 8) & 1) * 4)) + ((((u * 128) + ((int)threadIdx.x)) >> 4) & 3)) < H) && (((((((int)blockIdx.y) * 8) + (((((u * 128) + ((int)threadIdx.x)) >> 9) & 1) * 4)) + ((((u * 128) + ((int)threadIdx.x)) & 1) * 2)) + ((((u * 128) + ((int)threadIdx.x)) >> 6) & 1)) < W))) {
        condval = nr_ld32((&(A[((int64_t)i)])));
      } else {
        condval = (uint)0;
      }
      ((uint*)sh)[((u * 128) + ((int)threadIdx.x))] = condval;
    }
    __syncthreads();
    for (int k = 0; k < 2; ++k) {
      for (int r_1 = 0; r_1 < 4; ++r_1) {
        for (int z = 0; z < 4; ++z) {
          av[z] = ((uint*)sh)[((((r_1 * 256) + (k * 128)) + ((((int)threadIdx.x) & 31) * 4)) + z)];
        }
        for (int n_1 = 0; n_1 < 8; ++n_1) {
          uint b0 = nr_ld32((&(Weight[(((((((p * 32768) + (k * 16384)) + (((int)blockIdx.z) * 8192)) + (((((int)threadIdx.x) >> 5) & 3) * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8))])));
          uint b1 = nr_ld32((&(Weight[((((((((p * 32768) + (k * 16384)) + (((int)blockIdx.z) * 8192)) + (((((int)threadIdx.x) >> 5) & 3) * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8)) + 4)])));
          nr_mma((&(c[((r_1 * 16) + (n_1 * 2))])), av[0], av[1], av[2], av[3], b0, b1);
        }
      }
    }
    __syncthreads();
  }
  for (int r_2 = 0; r_2 < 4; ++r_2) {
    for (int n_2 = 0; n_2 < 8; ++n_2) {
      for (int h_1 = 0; h_1 < 2; ++h_1) {
        uint a = c[(((r_2 * 16) + (n_2 * 2)) + h_1)];
        uint q = nr_qpair(a);
        ((uchar*)route)[(((((((((((int)threadIdx.x) >> 5) * 6144) + (r_2 * 1024)) + ((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) >> 5) * 512)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 7) * 64)) + (h_1 * 4)) + ((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 3)) + (((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 12) << 2)) + (((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 16) >> 1))] = ((uchar)q);
        ((uchar*)route)[((((((((((((int)threadIdx.x) >> 5) * 6144) + (r_2 * 1024)) + ((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) >> 5) * 512)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 7) * 64)) + (h_1 * 4)) + ((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 3)) + (((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 12) << 2)) + (((((((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & -15) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 2) << 1)) | ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 4) << 1)) & 31) & 16) >> 1)) + 1)] = ((uchar)(q >> (uint)8));
      }
    }
  }
  __syncthreads();
  for (int r_3 = 0; r_3 < 4; ++r_3) {
    for (int n_3 = 0; n_3 < 8; ++n_3) {
      for (int h_2 = 0; h_2 < 2; ++h_2) {
        total[(((r_3 * 16) + (n_3 * 2)) + h_2)] = (uint)0;
      }
    }
  }
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    for (int r_4 = 0; r_4 < 4; ++r_4) {
      for (int n_4 = 0; n_4 < 4; ++n_4) {
        for (int h_3 = 0; h_3 < 2; ++h_3) {
          hidden[(((r_4 * 8) + (n_4 * 2)) + h_3)] = (uint)0;
        }
      }
    }
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int r_5 = 0; r_5 < 4; ++r_5) {
        for (int z_1 = 0; z_1 < 4; ++z_1) {
          av[z_1] = nr_ld32((&(((uchar*)route)[((((((((int)threadIdx.x) >> 5) * 6144) + (r_5 * 1024)) + (k_1 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z_1 * 4))])));
        }
        for (int n_5 = 0; n_5 < 4; ++n_5) {
          uint b0_1 = nr_ld32((&(Weight[((((((((((int)blockIdx.z) * 65536) + (((((int)threadIdx.x) >> 5) & 3) * 16384)) + (k_1 * 8192)) + (i_1 * 1024)) + ((n_5 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_5 & 1) * 8)) + 262144)])));
          uint b1_1 = nr_ld32((&(Weight[((((((((((int)blockIdx.z) * 65536) + (((((int)threadIdx.x) >> 5) & 3) * 16384)) + (k_1 * 8192)) + (i_1 * 1024)) + ((n_5 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_5 & 1) * 8)) + 262148)])));
          nr_mma((&(hidden[((r_5 * 8) + (n_5 * 2))])), av[0], av[1], av[2], av[3], b0_1, b1_1);
        }
      }
    }
    for (int r_6 = 0; r_6 < 4; ++r_6) {
      for (int n_6 = 0; n_6 < 4; ++n_6) {
        for (int h_4 = 0; h_4 < 2; ++h_4) {
          uint a_1 = hidden[(((r_6 * 8) + (n_6 * 2)) + h_4)];
          uint a_2 = nr_min2(nr_max2(a_1, nr_dup(-0x1p+2f/*-4.000000e+00*/)), nr_dup(0x1p+2f/*4.000000e+00*/));
          uint a_3 = (a_2 & (uint)2147450879);
          uint b = nr_dup(-0x1.cap-5f/*-5.590820e-02*/);
          uint c_1 = nr_dup(0x1.cap-2f/*4.472656e-01*/);
          uint b_1 = nr_fma2((a_2 & (uint)2147450879), b, c_1);
          uint c_2 = nr_dup(0x1.cap-1f/*8.945312e-01*/);
          uint b_2 = nr_fma2(a_2, b_1, c_2);
          uint a_4 = nr_mul2(a_1, b_2);
          uint q_1 = nr_qpair(a_4);
          ((uchar*)route)[(((((((((((int)threadIdx.x) >> 5) * 6144) + (r_6 * 512)) + ((((h_4 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 7) * 64)) + (h_4 * 4)) + (((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 3)) + ((((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 12) << 2)) + ((((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 16) >> 1)) + 4096)] = ((uchar)q_1);
          ((uchar*)route)[(((((((((((int)threadIdx.x) >> 5) * 6144) + (r_6 * 512)) + ((((h_4 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 7) * 64)) + (h_4 * 4)) + (((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 3)) + ((((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 12) << 2)) + ((((((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 1)) + ((((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + (((n_6 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16)) & 16) >> 1)) + 4097)] = ((uchar)(q_1 >> (uint)8));
        }
      }
    }
    __syncthreads();
    for (int r_7 = 0; r_7 < 4; ++r_7) {
      for (int z_2 = 0; z_2 < 4; ++z_2) {
        av[z_2] = nr_ld32((&(((uchar*)route)[((((((((int)threadIdx.x) >> 5) * 6144) + (r_7 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z_2 * 4)) + 4096)])));
      }
      for (int n_7 = 0; n_7 < 8; ++n_7) {
        uint b0_2 = nr_ld32((&(Weight[(((((((((int)blockIdx.z) * 65536) + (((((int)threadIdx.x) >> 5) & 3) * 16384)) + (i_1 * 2048)) + ((n_7 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_7 & 1) * 8)) + 393216)])));
        uint b1_2 = nr_ld32((&(Weight[(((((((((int)blockIdx.z) * 65536) + (((((int)threadIdx.x) >> 5) & 3) * 16384)) + (i_1 * 2048)) + ((n_7 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_7 & 1) * 8)) + 393220)])));
        nr_mma((&(total[((r_7 * 16) + (n_7 * 2))])), av[0], av[1], av[2], av[3], b0_2, b1_2);
      }
    }
    __syncthreads();
  }
  for (int r_8 = 0; r_8 < 4; ++r_8) {
    for (int n_8 = 0; n_8 < 8; ++n_8) {
      for (int h_5 = 0; h_5 < 2; ++h_5) {
        c[(((r_8 * 16) + (n_8 * 2)) + h_5)] = total[(((r_8 * 16) + (n_8 * 2)) + h_5)];
      }
    }
  }
  for (int r_9 = 0; r_9 < 4; ++r_9) {
    for (int n_9 = 0; n_9 < 8; ++n_9) {
      for (int h_6 = 0; h_6 < 2; ++h_6) {
        uint a_5 = c[(((r_9 * 16) + (n_9 * 2)) + h_6)];
        uint q_2 = nr_qpair(a_5);
        ((uchar*)packed)[((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_9 * 1024)) + (((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5))] = ((uchar)q_2);
        ((uchar*)packed)[(((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_9 * 1024)) + (((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_9 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_6 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5)) + 1)] = ((uchar)(q_2 >> (uint)8));
      }
    }
  }
  __syncthreads();
  for (int t = 0; t < 4; ++t) {
    if ((((((int)blockIdx.y) * 2) + (t >> 1)) < (W >> 2)) && (((((int)blockIdx.x) * 2) + (t & 1)) < (H >> 2))) {
      for (int z_3 = 0; z_3 < 8; ++z_3) {
        int i_2 = max(((((((((((int)blockIdx.x) * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + (t >> 1))) * 8192)) + ((t & 1) * 8192)) + (((int)blockIdx.z) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + ((z_3 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_3 & 3) * 4)), ((((((((((int)blockIdx.x) * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + (t >> 1))) * 8192)) + ((t & 1) * 8192)) + (((int)blockIdx.z) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + ((z_3 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_3 & 3) * 4)));
        uint v = nr_ld32((&(((uchar*)packed)[((((((((int)threadIdx.x) >> 5) * 4096) + (t * 1024)) + ((z_3 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_3 & 3) * 4))])));
        if (0 <= i_2) {
          if (i_2 < sizes_3) {
            nr_st32((&(Raw[((int64_t)i_2)])), v);
          }
        }
      }
    }
  }
  bool first_block = (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0));
  if (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0)) {
    for (int j = 0; j < ((n0 + 127) >> 7); ++j) {
      if (((j * 128) + ((int)threadIdx.x)) < n0) {
        C0[((((int64_t)j) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = v0;
      }
    }
    for (int j_1 = 0; j_1 < ((n1 + 127) >> 7); ++j_1) {
      if (((j_1 * 128) + ((int)threadIdx.x)) < n1) {
        C1[((((int64_t)j_1) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = v1;
      }
    }
  }
}

