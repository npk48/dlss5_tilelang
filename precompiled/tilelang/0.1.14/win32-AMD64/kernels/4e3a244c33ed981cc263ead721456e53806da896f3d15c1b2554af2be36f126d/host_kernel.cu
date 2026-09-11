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

extern "C" __global__ void main_kernel(const uchar* __restrict__ A, uchar* __restrict__ PoolRaw, uchar* __restrict__ Raw, const uchar* __restrict__ Skip, const uchar* __restrict__ Weight);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const uchar* __restrict__ A, uchar* __restrict__ PoolRaw, uchar* __restrict__ Raw, const uchar* __restrict__ Skip, const uchar* __restrict__ Weight) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* keep = ((void*)((char*)buf_dyn_shmem + 0));
  void* sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* packed = ((void*)((char*)buf_dyn_shmem + 32768));
  uint c[64];
  uint v = (uint)0;
  uint av[4];
  for (int t = 0; t < 4; ++t) {
    for (int n = 0; n < 8; ++n) {
      for (int h = 0; h < 2; ++h) {
        c[(((t * 16) + (n * 2)) + h)] = (uint)0;
        v = (uint)0;
        if ((((((int)blockIdx.x) % 3) * 2) + (t & 1)) < 5) {
          v = ((nr_ld32((&(Skip[((((((((((((((((int)blockIdx.y) * 81920) + ((t >> 1) * 40960)) + ((((int)blockIdx.x) % 3) * 16384)) + ((t & 1) * 8192)) + ((((int)blockIdx.x) / 3) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + (((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5)) & -4)]))) >> ((uint)(((((((((((((((((int)blockIdx.y) * 81920) + ((t >> 1) * 40960)) + ((((int)blockIdx.x) % 3) * 16384)) + ((t & 1) * 8192)) + ((((int)blockIdx.x) / 3) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + (((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5)) & 3) * 8))) & (uint)65535);
        }
        uint a = nr_decode2(v);
        uint b = nr_ld32((&(Weight[((((((((int)blockIdx.x) / 3) * 512) + (((((int)threadIdx.x) >> 5) & 3) * 128)) + (n * 16)) + (((((int)threadIdx.x) & 31) & 3) * 4)) + 262144)])));
        c[(((t * 16) + (n * 2)) + h)] = nr_mul2(a, b);
      }
    }
  }
  for (int g = 0; g < 8; ++g) {
    for (int u = 0; u < 2; ++u) {
      for (int z = 0; z < 4; ++z) {
        uint condval;
        if (((((((int)blockIdx.x) % 3) * 2) + (((u * 2) + (((int)threadIdx.x) >> 6)) & 1)) < 5)) {
          condval = nr_ld32((&(A[((((((((((int)blockIdx.y) * 81920) + (u * 40960)) + ((((int)blockIdx.x) % 3) * 16384)) + ((((u * 2) + (((int)threadIdx.x) >> 6)) & 1) * 8192)) + (g * 1024)) + ((((u * 4) + (((int)threadIdx.x) >> 5)) & 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z * 4))])));
        } else {
          condval = (uint)0;
        }
        ((uint*)sh)[(((u * 512) + (((int)threadIdx.x) * 4)) + z)] = condval;
      }
    }
    __syncthreads();
    for (int k = 0; k < 2; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int z_1 = 0; z_1 < 4; ++z_1) {
          av[z_1] = ((uint*)sh)[((((r * 256) + (k * 128)) + ((((int)threadIdx.x) & 31) * 4)) + z_1)];
        }
        for (int n_1 = 0; n_1 < 8; ++n_1) {
          uint b0 = nr_ld32((&(Weight[(((((((g * 32768) + (k * 16384)) + ((((int)blockIdx.x) / 3) * 8192)) + (((((int)threadIdx.x) >> 5) & 3) * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8))])));
          uint b1 = nr_ld32((&(Weight[((((((((g * 32768) + (k * 16384)) + ((((int)blockIdx.x) / 3) * 8192)) + (((((int)threadIdx.x) >> 5) & 3) * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8)) + 4)])));
          nr_mma((&(c[((r * 16) + (n_1 * 2))])), av[0], av[1], av[2], av[3], b0, b1);
        }
      }
    }
    __syncthreads();
  }
  for (int r_1 = 0; r_1 < 4; ++r_1) {
    for (int n_2 = 0; n_2 < 8; ++n_2) {
      for (int h_1 = 0; h_1 < 2; ++h_1) {
        uint a_1 = c[(((r_1 * 16) + (n_2 * 2)) + h_1)];
        uint q = nr_qpair(a_1);
        ((uchar*)packed)[((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_1 * 1024)) + (((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5))] = ((uchar)q);
        ((uchar*)packed)[(((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_1 * 1024)) + (((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5)) + 1)] = ((uchar)(q >> (uint)8));
        uint a_2 = c[(((r_1 * 16) + (n_2 * 2)) + h_1)];
        ((half_t*)keep)[(((((((((int)threadIdx.x) >> 5) * 4096) + (r_1 * 1024)) + (h_1 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (n_2 * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2))] = ((half_t)nr_half(a_2));
        uint a_3 = (c[(((r_1 * 16) + (n_2 * 2)) + h_1)] >> (uint)16);
        ((half_t*)keep)[((((((((((int)threadIdx.x) >> 5) * 4096) + (r_1 * 1024)) + (h_1 * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (n_2 * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)) + 1)] = ((half_t)nr_half(a_3));
      }
    }
  }
  __syncthreads();
  for (int t_1 = 0; t_1 < 4; ++t_1) {
    if ((((((int)blockIdx.x) % 3) * 2) + (t_1 & 1)) < 5) {
      for (int z_2 = 0; z_2 < 8; ++z_2) {
        uint v_1 = nr_ld32((&(((uchar*)packed)[((((((((int)threadIdx.x) >> 5) * 4096) + (t_1 * 1024)) + ((z_2 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_2 & 3) * 4))])));
        nr_st32((&(Raw[(((((((((((int)blockIdx.y) * 81920) + ((t_1 >> 1) * 40960)) + ((((int)blockIdx.x) % 3) * 16384)) + ((t_1 & 1) * 8192)) + ((((int)blockIdx.x) / 3) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + ((z_2 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_2 & 3) * 4))])), v_1);
      }
    }
  }
  for (int p = 0; p < 16; ++p) {
    for (int j = 0; j < 2; ++j) {
      half_t a_4 = ((half_t*)keep)[(((((((((int)threadIdx.x) >> 5) * 4096) + (((p & 3) >> 1) * 2048)) + ((p >> 2) * 512)) + ((p & 1) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j)];
      half_t b_1 = ((half_t*)keep)[((((((((((int)threadIdx.x) >> 5) * 4096) + (((p & 3) >> 1) * 2048)) + ((p >> 2) * 512)) + ((p & 1) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j) + 64)];
      half_t a_5 = ((half_t*)keep)[((((((((((int)threadIdx.x) >> 5) * 4096) + (((p & 3) >> 1) * 2048)) + ((p >> 2) * 512)) + ((p & 1) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j) + 256)];
      half_t b_2 = ((half_t*)keep)[((((((((((int)threadIdx.x) >> 5) * 4096) + (((p & 3) >> 1) * 2048)) + ((p >> 2) * 512)) + ((p & 1) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + j) + 320)];
      half_t a_6 = ((half_t)nr_add(((float)a_4), ((float)b_1)));
      half_t b_3 = ((half_t)nr_add(((float)a_5), ((float)b_2)));
      half_t a_7 = ((half_t)nr_add(((float)a_6), ((float)b_3)));
      half_t a_8 = ((half_t)nr_mul(((float)a_7), 0x1p-2f/*2.500000e-01*/));
      uchar q_1 = nr_enc(((float)a_8));
      ((uchar*)packed)[((((((((((((int)threadIdx.x) >> 5) * 4096) + (((((int)threadIdx.x) & 31) * 2) & 1)) + ((((((int)threadIdx.x) & 31) * 2) & 6) << 3)) + ((((((int)threadIdx.x) & 31) * 2) & 8) >> 2)) + ((((((int)threadIdx.x) & 31) * 2) & 16) >> 1)) + ((((((int)threadIdx.x) & 31) * 2) & 32) << 4)) + ((((((((p & 3) >> 1) & 1) + (((p & 3) & 1) << 1)) + ((((p >> 2) >> 1) & 1) << 2)) + (((p >> 2) & 1) << 3)) & 1) << 2)) + ((((((((p & 3) >> 1) & 1) + (((p & 3) & 1) << 1)) + ((((p >> 2) >> 1) & 1) << 2)) + (((p >> 2) & 1) << 3)) & 14) << 5)) + j)] = q_1;
    }
  }
  __syncthreads();
  for (int z_3 = 0; z_3 < 8; ++z_3) {
    uint v_2 = nr_ld32((&(((uchar*)packed)[(((((((int)threadIdx.x) >> 5) * 4096) + ((z_3 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_3 & 3) * 4))])));
    nr_st32((&(PoolRaw[(((((((((int)blockIdx.y) * 24576) + ((((int)blockIdx.x) % 3) * 8192)) + ((((int)blockIdx.x) / 3) * 4096)) + (((((int)threadIdx.x) >> 5) & 3) * 1024)) + ((z_3 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_3 & 3) * 4))])), v_2);
  }
  bool first_block = ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0));
}

