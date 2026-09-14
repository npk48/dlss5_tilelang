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

// Instruction-sized operations only; page production, ownership, loops and MMA dataflow live in TileLang.
namespace nr_tl_vit_joint {
__device__ __forceinline__ void cp(void* d,const unsigned char* a,int off,int valid) {
 unsigned s=__cvta_generic_to_shared(d);
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(s),"l"(a+(valid?off:0)),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void commit() {asm volatile("cp.async.commit_group;":::"memory");}
__device__ __forceinline__ void wait0() {asm volatile("cp.async.wait_group 0;":::"memory");}
__device__ __forceinline__ void wait1() {asm volatile("cp.async.wait_group 1;":::"memory");}
__device__ __forceinline__ void shared4(unsigned* d,const void* p) {
 unsigned s=__cvta_generic_to_shared(p);
 asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(s):"memory");
}
__device__ __forceinline__ void global4(unsigned* d,const void* p) {
 asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"l"(p):"memory");
}
__device__ __forceinline__ void store4(void* p,unsigned a,unsigned b,unsigned c,unsigned d) {
 asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"::"l"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");
}
__device__ __forceinline__ void store2(void* p,unsigned v) {*(unsigned short*)p=(unsigned short)v;}
__device__ __forceinline__ unsigned long long mma(unsigned a,unsigned b,unsigned c,unsigned d,unsigned e,unsigned f,unsigned x,unsigned y) {
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};":"+r"(x),"+r"(y):"r"(a),"r"(b),"r"(c),"r"(d),"r"(e),"r"(f));
 return (unsigned long long)x|((unsigned long long)y<<32);
}
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

extern "C" __global__ void main_kernel(uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, const half_t* __restrict__ G, uint* __restrict__ P, uchar* __restrict__ S, uchar* __restrict__ W, int PH, int PW, int n0, int n1, int rows, int sizes_2, int sizes_3, int sizes_8, int v0, int v1);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, const half_t* __restrict__ G, uint* __restrict__ P, uchar* __restrict__ S, uchar* __restrict__ W, int PH, int PW, int n0, int n1, int rows, int sizes_2, int sizes_3, int sizes_8, int v0, int v1) {
  uint c[64];
  uint res[4];
  extern __shared__ __align__(1024) uchar sh[];
  uint bv[16];
  uint av[4];
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        c[(((r * 16) + (n * 2)) + h)] = (uint)0;
      }
    }
  }
  if (((int)blockIdx.z) == 0) {
    #pragma unroll
    for (int r_1 = 0; r_1 < 4; ++r_1) {
      #pragma unroll
      for (int g = 0; g < 2; ++g) {
        if (((((((((((((int)blockIdx.x) * 131072) + ((((int)threadIdx.x) >> 6) * 65536)) + (r_1 * 16384)) + ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (((((int)threadIdx.x) & 31) & 3) * 2)) >> 5) * 512)) + ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_1 * 16)) + ((((int)threadIdx.x) & 31) >> 2)) & 7) * 64)) + (((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_1 * 16)) + ((((int)threadIdx.x) & 31) >> 2)) >> 3) & 1) * 4)) + (((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 1)) + ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 6) << 3)) + ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 8) >> 2)) + ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (((((int)threadIdx.x) & 31) & 3) * 2)) & 16) >> 1)) < sizes_2) {
          nr_tl_vit_joint::global4((&(res[0])), (&(S[((((((((((((int64_t)((int)blockIdx.x)) * (int64_t)131072) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)65536)) + (((int64_t)r_1) * (int64_t)16384)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + (((int64_t)g) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3) * (int64_t)2)) >> (int64_t)5) * (int64_t)512)) + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((int64_t)r_1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) & (int64_t)7) * (int64_t)64)) + (((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)64)) + (((int64_t)r_1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) >> (int64_t)3) & (int64_t)1) * (int64_t)4)) + (((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + (((int64_t)g) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3) * (int64_t)2)) & (int64_t)1)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + (((int64_t)g) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3) * (int64_t)2)) & (int64_t)6) << (int64_t)3)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + (((int64_t)g) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3) * (int64_t)2)) & (int64_t)8) >> (int64_t)2)) + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)64)) + (((int64_t)g) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3) * (int64_t)2)) & (int64_t)16) >> (int64_t)1))])));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          half_t condval;
          if ((((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (j * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)) + 524288) < sizes_8)) {
            condval = G[((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (j * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)) + 524288)];
          } else {
            condval = half_t(0x0p+0f/*0.000000e+00*/);
          }
          half_t a = condval;
          half_t condval_1;
          if ((((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (j * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)) + 524289) < sizes_8)) {
            condval_1 = G[((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + (g * 32)) + (j * 8)) + (((((int)threadIdx.x) & 31) & 3) * 2)) + 524289)];
          } else {
            condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          half_t a_1 = condval_1;
          uint b = (nr_hbits(((float)a)) | (nr_hbits(((float)a_1)) << (uint)16));
          #pragma unroll
          for (int h_1 = 0; h_1 < 2; ++h_1) {
            uint bits = ((res[(((j >> 1) * 2) + h_1)] >> ((uint)((j & 1) * 16))) & (uint)65535);
            uint a_2 = nr_decode2(bits);
            c[((((r_1 * 16) + (g * 8)) + (j * 2)) + h_1)] = nr_mul2(a_2, b);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int h_2 = 0; h_2 < 2; ++h_2) {
    nr_tl_vit_joint::cp((&(sh[((h_2 * 2048) + (((int)threadIdx.x) * 16))])), (&(A[0])), (((((((int)blockIdx.x) * 131072) + (h_2 * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + (((int)blockIdx.z) * 4096)) + ((((int)threadIdx.x) & 31) * 16)), ((int)((((((int)blockIdx.x) * 128) + (h_2 * 64)) + ((((int)threadIdx.x) >> 5) * 16)) < rows)));
  }
  #pragma unroll
  for (int j_1 = 0; j_1 < 2; ++j_1) {
    nr_tl_vit_joint::cp((&(sh[(((j_1 * 2048) + (((int)threadIdx.x) * 16)) + 4096)])), (&(W[0])), ((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 4096)) + (j_1 * 2048)) + (((int)threadIdx.x) * 16)), 1);
  }
  nr_tl_vit_joint::commit();
  for (int phase = 0; phase < 8; ++phase) {
    nr_tl_vit_joint::wait0();
    __syncthreads();
    if (phase < 7) {
      #pragma unroll
      for (int h_3 = 0; h_3 < 2; ++h_3) {
        nr_tl_vit_joint::cp((&(sh[(((((phase + 1) & 1) * 8192) + (h_3 * 2048)) + (((int)threadIdx.x) * 16))])), (&(A[0])), (((((((((int)blockIdx.x) * 131072) + (h_3 * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + (((int)blockIdx.z) * 4096)) + (phase * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 512), ((int)((((((int)blockIdx.x) * 128) + (h_3 * 64)) + ((((int)threadIdx.x) >> 5) * 16)) < rows)));
      }
      #pragma unroll
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        nr_tl_vit_joint::cp((&(sh[((((((phase + 1) & 1) * 8192) + (j_2 * 2048)) + (((int)threadIdx.x) * 16)) + 4096)])), (&(W[0])), ((((((((int)blockIdx.z) * 262144) + (phase * 32768)) + (((int)blockIdx.y) * 4096)) + (j_2 * 2048)) + (((int)threadIdx.x) * 16)) + 32768), 1);
      }
      nr_tl_vit_joint::commit();
    }
    #pragma unroll
    for (int n_1 = 0; n_1 < 4; ++n_1) {
      nr_tl_vit_joint::shared4((&(bv[(n_1 * 4)])), (&(sh[((((((phase & 1) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (n_1 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 4096)])));
    }
    #pragma unroll
    for (int r_2 = 0; r_2 < 4; ++r_2) {
      nr_tl_vit_joint::shared4((&(av[0])), (&(sh[(((((phase & 1) * 8192) + ((((int)threadIdx.x) >> 6) * 2048)) + (r_2 * 512)) + ((((int)threadIdx.x) & 31) * 16))])));
      #pragma unroll
      for (int n_2 = 0; n_2 < 8; ++n_2) {
        uint64_t v = nr_tl_vit_joint::mma(av[0], av[1], av[2], av[3], bv[(n_2 * 2)], bv[((n_2 * 2) + 1)], c[((r_2 * 16) + (n_2 * 2))], c[(((r_2 * 16) + (n_2 * 2)) + 1)]);
        c[((r_2 * 16) + (n_2 * 2))] = ((uint)v);
        c[(((r_2 * 16) + (n_2 * 2)) + 1)] = ((uint)(v >> (uint64_t)32));
      }
    }
    __syncthreads();
  }
  nr_tl_vit_joint::wait0();
  __syncthreads();
  #pragma unroll
  for (int r_3 = 0; r_3 < 4; ++r_3) {
    #pragma unroll
    for (int n_3 = 0; n_3 < 8; ++n_3) {
      #pragma unroll
      for (int h_4 = 0; h_4 < 2; ++h_4) {
        if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < rows) {
          int rmod = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rmod_1 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rdiv = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) / PH);
          int rmod_2 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rdiv_1 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) / PH);
          int rmod_3 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rmod_4 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rmod_5 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rdiv_2 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) / PH);
          int rmod_6 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int rdiv_3 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) / PH);
          int rmod_7 = ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (r_3 * 16)) + (h_4 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) % PH);
          int logical = max(((((((((0 <= PH) && (0 <= rmod)) || ((PH < 0) && (rmod <= 0))) ? rmod : (rmod + PH)) & -2) | (((((0 <= PH) && (0 <= rmod_1)) || ((PH < 0) && (rmod_1 <= 0))) ? rdiv : (rdiv - 1)) & 1)) * PW) + ((((((0 <= PH) && (0 <= rmod_2)) || ((PH < 0) && (rmod_2 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) & -2) | (((((0 <= PH) && (0 <= rmod_3)) || ((PH < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + PH)) & 1))), ((((((((0 <= PH) && (0 <= rmod_4)) || ((PH < 0) && (rmod_4 <= 0))) ? rmod_4 : (rmod_4 + PH)) & -2) | (((((0 <= PH) && (0 <= rmod_5)) || ((PH < 0) && (rmod_5 <= 0))) ? rdiv_2 : (rdiv_2 - 1)) & 1)) * PW) + ((((((0 <= PH) && (0 <= rmod_6)) || ((PH < 0) && (rmod_6 <= 0))) ? rdiv_3 : (rdiv_3 - 1)) & -2) | (((((0 <= PH) && (0 <= rmod_7)) || ((PH < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + PH)) & 1))));
          if (0 <= ((((int)blockIdx.z) * rows) + logical)) {
            if ((((((((((int)blockIdx.z) * rows) * 512) + (logical * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.x) & 63) >> 5) * 32)) + (n_3 * 4)) + ((((int)threadIdx.x) & 31) & 3)) < sizes_3) {
              P[(((((((((int64_t)((int)blockIdx.z)) * ((int64_t)rows)) * (int64_t)512) + (((int64_t)logical) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)5) * (int64_t)32)) + (((int64_t)n_3) * (int64_t)4)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) & (int64_t)3))] = c[(((r_3 * 16) + (n_3 * 2)) + h_4)];
            }
          }
        }
      }
    }
  }
  bool first_block = (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0));
  if (((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) && (((int)blockIdx.z) == 0)) {
    for (int j_3 = 0; j_3 < ((n0 + 127) >> 7); ++j_3) {
      if (((j_3 * 128) + ((int)threadIdx.x)) < n0) {
        C0[((((int64_t)j_3) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = v0;
      }
    }
    for (int j_4 = 0; j_4 < ((n1 + 127) >> 7); ++j_4) {
      if (((j_4 * 128) + ((int)threadIdx.x)) < n1) {
        C1[((((int64_t)j_4) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = v1;
      }
    }
  }
}

