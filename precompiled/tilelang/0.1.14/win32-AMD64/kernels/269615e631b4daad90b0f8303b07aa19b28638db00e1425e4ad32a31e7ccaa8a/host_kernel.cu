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

extern "C" __global__ void main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ Raw, const uchar* __restrict__ Skip, const uchar* __restrict__ Weight, int H, int W, int n0, int n1, int sizes_3, int v0, int v1);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ Raw, const uchar* __restrict__ Skip, const uchar* __restrict__ Weight, int H, int W, int n0, int n1, int sizes_3, int v0, int v1) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* packed = ((void*)((char*)buf_dyn_shmem + 0));
  void* sh = ((void*)((char*)buf_dyn_shmem + 0));
  uint c[64];
  uint v = (uint)0;
  uint av[4];
  int rmod = (((int)blockIdx.x) % ((H + 7) >> 3));
  int rmod_1 = (((int)blockIdx.x) % ((H + 7) >> 3));
  int sx = max(((((0 <= ((H + 7) >> 3)) && (0 <= rmod)) || ((((H + 7) >> 3) < 0) && (rmod <= 0))) ? rmod : (rmod + ((H + 7) >> 3))), ((((0 <= ((H + 7) >> 3)) && (0 <= rmod_1)) || ((((H + 7) >> 3) < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + ((H + 7) >> 3))));
  int rmod_2 = (((int)blockIdx.x) % ((H + 7) >> 3));
  int rdiv = (((int)blockIdx.x) / ((H + 7) >> 3));
  int rmod_3 = (((int)blockIdx.x) % ((H + 7) >> 3));
  int rdiv_1 = (((int)blockIdx.x) / ((H + 7) >> 3));
  int g = max(((((((0 <= ((H + 7) >> 3)) && (0 <= rmod_2)) || ((((H + 7) >> 3) < 0) && (rmod_2 <= 0))) ? rdiv : (rdiv - 1)) * 4) + ((((int)threadIdx.x) >> 5) & 3)), ((((((0 <= ((H + 7) >> 3)) && (0 <= rmod_3)) || ((((H + 7) >> 3) < 0) && (rmod_3 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 4) + ((((int)threadIdx.x) >> 5) & 3)));
  for (int r = 0; r < 4; ++r) {
    for (int n = 0; n < 8; ++n) {
      for (int h = 0; h < 2; ++h) {
        c[(((r * 16) + (n * 2)) + h)] = (uint)0;
        int y = max(((((sx * 8) + ((r & 1) * 4)) + (h * 2)) + ((((int)threadIdx.x) & 31) >> 4)), ((((sx * 8) + ((r & 1) * 4)) + (h * 2)) + ((((int)threadIdx.x) & 31) >> 4)));
        v = (uint)0;
        if ((y < H) && ((((((int)blockIdx.y) * 8) + ((r >> 1) * 4)) + ((((int)threadIdx.x) & 15) >> 2)) < W)) {
          int ch = max((((g * 64) + (n * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)), (((g * 64) + (n * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)));
          int pix = max(((((y >> 2) * 4) + ((((int)threadIdx.x) & 15) >> 2)) + ((((((int)blockIdx.y) * 8) + ((r >> 1) * 4)) + (y & 3)) * H)), ((((y >> 2) * 4) + ((((int)threadIdx.x) & 15) >> 2)) + ((((((int)blockIdx.y) * 8) + ((r >> 1) * 4)) + (y & 3)) * H)));
          int off = max((((((pix * 16) + ((((ch >> 4) * H) * W) * 16)) + (ch & 1)) + ((ch & 6) << 1)) + ((ch & 8) >> 2)), (((((pix * 16) + ((((ch >> 4) * H) * W) * 16)) + (ch & 1)) + ((ch & 6) << 1)) + ((ch & 8) >> 2)));
          int i = max((off & -4), (off & -4));
          v = ((nr_ld32((&(Skip[((int64_t)i)]))) >> ((uint)((off & 3) * 8))) & (uint)65535);
        }
        int i_1 = max(((((g * 128) + (n * 16)) + (((((int)threadIdx.x) & 31) & 3) * 4)) + 262144), ((((g * 128) + (n * 16)) + (((((int)threadIdx.x) & 31) & 3) * 4)) + 262144));
        uint a = nr_decode2(v);
        uint b = nr_ld32((&(Weight[((int64_t)i_1)])));
        c[(((r * 16) + (n * 2)) + h)] = nr_mul2(a, b);
      }
    }
  }
  for (int g_1 = 0; g_1 < 8; ++g_1) {
    for (int u = 0; u < 2; ++u) {
      for (int z = 0; z < 4; ++z) {
        int i_2 = max((((((((sx * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + u)) * 8192)) + ((((u * 2) + (((int)threadIdx.x) >> 6)) & 1) * 8192)) + (g_1 * 1024)) + ((((u * 4) + (((int)threadIdx.x) >> 5)) & 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z * 4)), (((((((sx * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + u)) * 8192)) + ((((u * 2) + (((int)threadIdx.x) >> 6)) & 1) * 8192)) + (g_1 * 1024)) + ((((u * 4) + (((int)threadIdx.x) >> 5)) & 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + (z * 4)));
        uint condval;
        if (((((((int)blockIdx.y) * 2) + u) < (W >> 2)) && (((sx * 2) + (((u * 2) + (((int)threadIdx.x) >> 6)) & 1)) < (H >> 2)))) {
          condval = nr_ld32((&(A[((int64_t)i_2)])));
        } else {
          condval = (uint)0;
        }
        ((uint*)sh)[(((u * 512) + (((int)threadIdx.x) * 4)) + z)] = condval;
      }
    }
    __syncthreads();
    for (int k = 0; k < 2; ++k) {
      for (int r_1 = 0; r_1 < 4; ++r_1) {
        for (int z_1 = 0; z_1 < 4; ++z_1) {
          av[z_1] = ((uint*)sh)[((((r_1 * 256) + (k * 128)) + ((((int)threadIdx.x) & 31) * 4)) + z_1)];
        }
        for (int n_1 = 0; n_1 < 8; ++n_1) {
          int i_3 = max(((((((g_1 * 32768) + (k * 16384)) + (g * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8)), ((((((g_1 * 32768) + (k * 16384)) + (g * 2048)) + ((n_1 >> 1) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((n_1 & 1) * 8)));
          int i_4 = max((i_3 + 4), (i_3 + 4));
          uint b0 = nr_ld32((&(Weight[((int64_t)i_3)])));
          uint b1 = nr_ld32((&(Weight[((int64_t)i_4)])));
          nr_mma((&(c[((r_1 * 16) + (n_1 * 2))])), av[0], av[1], av[2], av[3], b0, b1);
        }
      }
    }
    __syncthreads();
  }
  for (int r_2 = 0; r_2 < 4; ++r_2) {
    for (int n_2 = 0; n_2 < 8; ++n_2) {
      for (int h_1 = 0; h_1 < 2; ++h_1) {
        uint a_1 = c[(((r_2 * 16) + (n_2 * 2)) + h_1)];
        uint q = nr_qpair(a_1);
        ((uchar*)packed)[((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_2 * 1024)) + (((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5))] = ((uchar)q);
        ((uchar*)packed)[(((((((((((((int)threadIdx.x) >> 5) * 4096) + (r_2 * 1024)) + (((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) + ((((n_2 * 8) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 32) << 4)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 1) << 2)) + (((((((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 2) & 1) * 8) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) & 3) * 2)) + ((((h_1 * 8) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1)) & 14) << 5)) + 1)] = ((uchar)(q >> (uint)8));
        int y_1 = max(((((sx * 8) + ((r_2 & 1) * 4)) + (h_1 * 2)) + ((((int)threadIdx.x) & 31) >> 4)), ((((sx * 8) + ((r_2 & 1) * 4)) + (h_1 * 2)) + ((((int)threadIdx.x) & 31) >> 4)));
      }
    }
  }
  __syncthreads();
  for (int t = 0; t < 4; ++t) {
    if ((((((int)blockIdx.y) * 2) + (t >> 1)) < (W >> 2)) && (((sx * 2) + (t & 1)) < (H >> 2))) {
      for (int z_2 = 0; z_2 < 8; ++z_2) {
        int i_5 = max((((((((sx * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + (t >> 1))) * 8192)) + ((t & 1) * 8192)) + (g * 1024)) + ((z_2 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_2 & 3) * 4)), (((((((sx * 16384) + (((H >> 2) * ((((int)blockIdx.y) * 2) + (t >> 1))) * 8192)) + ((t & 1) * 8192)) + (g * 1024)) + ((z_2 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_2 & 3) * 4)));
        uint v_1 = nr_ld32((&(((uchar*)packed)[((((((((int)threadIdx.x) >> 5) * 4096) + (t * 1024)) + ((z_2 >> 2) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + ((z_2 & 3) * 4))])));
        if (0 <= i_5) {
          if (i_5 < sizes_3) {
            nr_st32((&(Raw[((int64_t)i_5)])), v_1);
          }
        }
      }
    }
  }
  bool first_block = ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0));
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
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

