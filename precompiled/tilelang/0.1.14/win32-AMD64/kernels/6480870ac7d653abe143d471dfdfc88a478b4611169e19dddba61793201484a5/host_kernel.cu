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

extern "C" __global__ void main_kernel(int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ K, uchar* __restrict__ O, uchar* __restrict__ Q, uchar* __restrict__ V, int n0, int n1, int rows, int sizes_0, int sizes_3, int v0, int v1);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ K, uchar* __restrict__ O, uchar* __restrict__ Q, uchar* __restrict__ V, int n0, int n1, int rows, int sizes_0, int sizes_3, int v0, int v1) {
  uint query[16];
  uint result[32];
  half_t den[8];
  extern __shared__ __align__(1024) uchar sh[];
  uint score[64];
  uint bv[8];
  uint av[4];
  uint z = (uint)0;
  uint s = (uint)0;
  uint total = (uint)0;
  uint packet[4];
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    #pragma unroll
    for (int z_1 = 0; z_1 < 4; ++z_1) {
      query[((r * 4) + z_1)] = (uint)0;
    }
    if ((((((int)blockIdx.y) * 16) + ((((int)threadIdx.x) >> 5) * 4)) + r) < (rows >> 4)) {
      if ((((((((int)blockIdx.y) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (r * 16384)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) & 31) * 16)) < sizes_0) {
        nr_tl_vit_joint::global4((&(query[(r * 4)])), (&(Q[(((((((int64_t)((int)blockIdx.y)) * (int64_t)262144) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)65536)) + (((int64_t)r) * (int64_t)16384)) + (((int64_t)((int)blockIdx.x)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)16))])));
      }
    }
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        result[(((r * 8) + (n * 2)) + h)] = (uint)0;
      }
    }
    #pragma unroll
    for (int h_1 = 0; h_1 < 2; ++h_1) {
      den[((r * 2) + h_1)] = half_t(0x0p+0f/*0.000000e+00*/);
    }
  }
  nr_tl_vit_joint::cp((&(sh[(((int)threadIdx.x) * 16)])), (&(K[0])), ((((((int)threadIdx.x) >> 5) * 16384) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) & 31) * 16)), ((int)(((((int)threadIdx.x) >> 5) * 16) < rows)));
  nr_tl_vit_joint::cp((&(sh[((((int)threadIdx.x) * 16) + 2048)])), (&(V[0])), ((((((int)threadIdx.x) >> 6) * 32768) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) & 63) * 16)), 1);
  nr_tl_vit_joint::commit();
  for (int phase = 0; phase < ((rows + 63) >> 6); ++phase) {
    nr_tl_vit_joint::wait0();
    __syncthreads();
    if ((phase + 1) < ((rows + 63) >> 6)) {
      nr_tl_vit_joint::cp((&(sh[(((((int)threadIdx.x) * 16) + 4096) - ((phase & 1) * 4096))])), (&(K[0])), (((((phase * 65536) + ((((int)threadIdx.x) >> 5) * 16384)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 65536), ((int)((((phase * 64) + ((((int)threadIdx.x) >> 5) * 16)) + 64) < rows)));
      nr_tl_vit_joint::cp((&(sh[(((((int)threadIdx.x) * 16) + 6144) - ((phase & 1) * 4096))])), (&(V[0])), (((((phase * 65536) + ((((int)threadIdx.x) >> 6) * 32768)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) & 63) * 16)) + 65536), 1);
      nr_tl_vit_joint::commit();
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 4; ++r_1) {
      #pragma unroll
      for (int n_1 = 0; n_1 < 8; ++n_1) {
        #pragma unroll
        for (int h_2 = 0; h_2 < 2; ++h_2) {
          score[(((r_1 * 16) + (n_1 * 2)) + h_2)] = (uint)0;
        }
      }
    }
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      nr_tl_vit_joint::shared4((&(bv[0])), (&(sh[((((phase & 1) * 4096) + (pair * 512)) + ((((int)threadIdx.x) & 31) * 16))])));
      #pragma unroll
      for (int r_2 = 0; r_2 < 4; ++r_2) {
        #pragma unroll
        for (int z_2 = 0; z_2 < 4; ++z_2) {
          av[z_2] = query[((r_2 * 4) + z_2)];
        }
        #pragma unroll
        for (int n_2 = 0; n_2 < 2; ++n_2) {
          uint64_t v = nr_tl_vit_joint::mma(av[0], av[1], av[2], av[3], bv[(n_2 * 2)], bv[((n_2 * 2) + 1)], score[(((r_2 * 16) + (pair * 4)) + (n_2 * 2))], score[((((r_2 * 16) + (pair * 4)) + (n_2 * 2)) + 1)]);
          score[(((r_2 * 16) + (pair * 4)) + (n_2 * 2))] = ((uint)v);
          score[((((r_2 * 16) + (pair * 4)) + (n_2 * 2)) + 1)] = ((uint)(v >> (uint64_t)32));
        }
      }
    }
    #pragma unroll
    for (int r_3 = 0; r_3 < 4; ++r_3) {
      #pragma unroll
      for (int n_3 = 0; n_3 < 8; ++n_3) {
        #pragma unroll
        for (int h_3 = 0; h_3 < 2; ++h_3) {
          uint a = score[(((r_3 * 16) + (n_3 * 2)) + h_3)];
          uint b = nr_dup(0x1.6ecp-4f/*8.953857e-02*/);
          uint c = nr_dup(0x1.b58p+0f/*1.708984e+00*/);
          z = nr_fma2(a, b, c);
          z = nr_min2(nr_max2(z, nr_dup(0x1.708p+0f/*1.439453e+00*/)), nr_dup(0x1.fa4p+0f/*1.977539e+00*/));
          z = (((z & (uint)268374015) << (uint)4) - (uint)3221274624);
          score[(((r_3 * 16) + (n_3 * 2)) + h_3)] = z;
        }
      }
      #pragma unroll
      for (int h_4 = 0; h_4 < 2; ++h_4) {
        uint a_1 = score[((r_3 * 16) + h_4)];
        uint b_1 = score[(((r_3 * 16) + h_4) + 2)];
        s = nr_add2(a_1, b_1);
        #pragma unroll
        for (int n_4 = 1; n_4 < 4; ++n_4) {
          uint a_2 = score[(((r_3 * 16) + (n_4 * 4)) + h_4)];
          uint b_2 = score[((((r_3 * 16) + (n_4 * 4)) + h_4) + 2)];
          uint a_3 = s;
          uint b_3 = nr_add2(a_2, b_2);
          s = nr_add2(a_3, b_3);
        }
        uint a_4 = s;
        total = nr_shuffle(a_4, ((((int)threadIdx.x) & 31) & -4));
        #pragma unroll
        for (int n_5 = 1; n_5 < 4; ++n_5) {
          uint a_5 = s;
          uint a_6 = total;
          uint b_4 = nr_shuffle(a_5, (((((int)threadIdx.x) & 31) & -4) + n_5));
          total = nr_add2(a_6, b_4);
        }
        uint a_7 = total;
        uint a_8 = (total >> (uint)16);
        half_t a_9 = ((half_t)nr_half(a_7));
        half_t b_5 = ((half_t)nr_half(a_8));
        half_t a_10 = den[((r_3 * 2) + h_4)];
        half_t b_6 = ((half_t)nr_add(((float)a_9), ((float)b_5)));
        den[((r_3 * 2) + h_4)] = ((half_t)nr_add(((float)a_10), ((float)b_6)));
      }
      #pragma unroll
      for (int kp = 0; kp < 2; ++kp) {
        #pragma unroll
        for (int z_3 = 0; z_3 < 4; ++z_3) {
          uint a_11 = score[(((r_3 * 16) + (kp * 8)) + ((z_3 >> 1) * 4))];
          uint a_12 = score[((((r_3 * 16) + (kp * 8)) + ((z_3 >> 1) * 4)) + 1)];
          uint a_13 = (nr_qpair(a_11) | (nr_qpair(a_12) << (uint)16));
          uint a_14 = score[((((r_3 * 16) + (kp * 8)) + ((z_3 >> 1) * 4)) + 2)];
          uint a_15 = score[((((r_3 * 16) + (kp * 8)) + ((z_3 >> 1) * 4)) + 3)];
          uint b_7 = (nr_qpair(a_14) | (nr_qpair(a_15) << (uint)16));
          int condval;
          if ((0 < (z_3 & 1))) {
            condval = 30258;
          } else {
            condval = 21520;
          }
          av[z_3] = nr_perm(a_13, b_7, ((uint)condval));
        }
        #pragma unroll
        for (int n_6 = 0; n_6 < 2; ++n_6) {
          nr_tl_vit_joint::shared4((&(bv[(n_6 * 4)])), (&(sh[((((((phase & 1) * 4096) + (kp * 1024)) + (n_6 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 2048)])));
        }
        #pragma unroll
        for (int n_7 = 0; n_7 < 4; ++n_7) {
          uint64_t v_1 = nr_tl_vit_joint::mma(av[0], av[1], av[2], av[3], bv[(n_7 * 2)], bv[((n_7 * 2) + 1)], result[((r_3 * 8) + (n_7 * 2))], result[(((r_3 * 8) + (n_7 * 2)) + 1)]);
          result[((r_3 * 8) + (n_7 * 2))] = ((uint)v_1);
          result[(((r_3 * 8) + (n_7 * 2)) + 1)] = ((uint)(v_1 >> (uint64_t)32));
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r_4 = 0; r_4 < 4; ++r_4) {
    #pragma unroll
    for (int h_5 = 0; h_5 < 2; ++h_5) {
      half_t a_16 = den[((r_4 * 2) + h_5)];
      half_t b_8 = ((half_t)(((float)(((rows + 63) & 63) - 63)) * 0x1.58p-4f/*8.398438e-02*/));
      half_t d = ((half_t)nr_add(((float)a_16), ((float)((half_t)(((float)(((rows + 63) & 63) - 63)) * 0x1.58p-4f/*8.398438e-02*/)))));
      half_t inv = ((half_t)nr_rcp(((float)cutlass::half_t(__hmax((d).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half())))));
      #pragma unroll
      for (int n_8 = 0; n_8 < 4; ++n_8) {
        float a_17 = ((float)inv);
        uint a_18 = result[(((r_4 * 8) + (n_8 * 2)) + h_5)];
        uint b_9 = nr_dup(((float)inv));
        result[(((r_4 * 8) + (n_8 * 2)) + h_5)] = nr_mul2(a_18, b_9);
      }
    }
    if ((((((int)blockIdx.y) * 16) + ((((int)threadIdx.x) >> 5) * 4)) + r_4) < (rows >> 4)) {
      #pragma unroll
      for (int n_9 = 0; n_9 < 2; ++n_9) {
        #pragma unroll
        for (int h_6 = 0; h_6 < 2; ++h_6) {
          uint a_19 = result[(((r_4 * 8) + (n_9 * 4)) + h_6)];
          uint a_20 = result[((((r_4 * 8) + (n_9 * 4)) + h_6) + 2)];
          packet[((n_9 * 2) + h_6)] = (nr_qpair(a_19) | (nr_qpair(a_20) << (uint)16));
        }
      }
      if ((((((((int)blockIdx.y) * 262144) + ((((int)threadIdx.x) >> 5) * 65536)) + (r_4 * 16384)) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) & 31) * 16)) < sizes_3) {
        nr_tl_vit_joint::store4((&(O[(((((((int64_t)((int)blockIdx.y)) * (int64_t)262144) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)65536)) + (((int64_t)r_4) * (int64_t)16384)) + (((int64_t)((int)blockIdx.x)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)16))])), packet[0], packet[1], packet[2], packet[3]);
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

