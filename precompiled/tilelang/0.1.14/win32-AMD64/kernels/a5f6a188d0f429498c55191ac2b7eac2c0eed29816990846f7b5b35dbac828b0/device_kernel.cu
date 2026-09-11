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

extern "C" __global__ void main_kernel(uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ K, uchar* __restrict__ Q, uint* __restrict__ Raw, const float* __restrict__ Scale, uchar* __restrict__ V, uchar* __restrict__ W);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(uchar* __restrict__ A, int* __restrict__ C0, int* __restrict__ C1, uchar* __restrict__ K, uchar* __restrict__ Q, uint* __restrict__ Raw, const float* __restrict__ Scale, uchar* __restrict__ V, uchar* __restrict__ W) {
  uint c[96];
  extern __shared__ __align__(1024) uchar sh[];
  uint bv[24];
  uint av[4];
  uint rv[4];
  uint z = (uint)0;
  for (int split = 0; split < 2; ++split) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      #pragma unroll
      for (int n = 0; n < 12; ++n) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          c[(((r * 24) + (n * 2)) + h)] = (uint)0;
        }
      }
    }
    #pragma unroll
    for (int h_1 = 0; h_1 < 2; ++h_1) {
      nr_tl_vit_joint::cp((&(sh[((h_1 * 2048) + (((int)threadIdx.x) * 16))])), (&(A[0])), ((((h_1 * 65536) + ((((int)threadIdx.x) >> 5) * 16384)) + (split * 8192)) + ((((int)threadIdx.x) & 31) * 16)), 1);
    }
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
      nr_tl_vit_joint::cp((&(sh[(((j * 2048) + (((int)threadIdx.x) * 16)) + 4096)])), (&(W[0])), (((((split * 1572864) + (((int)blockIdx.x) * 6144)) + (j * 2048)) + (((int)threadIdx.x) * 16)) + 128), 1);
    }
    nr_tl_vit_joint::commit();
    for (int step = 0; step < 16; ++step) {
      nr_tl_vit_joint::wait0();
      __syncthreads();
      if (step < 15) {
        #pragma unroll
        for (int h_2 = 0; h_2 < 2; ++h_2) {
          nr_tl_vit_joint::cp((&(sh[((((h_2 * 2048) + (((int)threadIdx.x) * 16)) + 10240) - ((step & 1) * 10240))])), (&(A[0])), ((((((h_2 * 65536) + ((((int)threadIdx.x) >> 5) * 16384)) + (split * 8192)) + (step * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 512), 1);
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 3; ++j_1) {
          nr_tl_vit_joint::cp((&(sh[((((j_1 * 2048) + (((int)threadIdx.x) * 16)) + 14336) - ((step & 1) * 10240))])), (&(W[0])), ((((((split * 1572864) + (step * 98304)) + (((int)blockIdx.x) * 6144)) + (j_1 * 2048)) + (((int)threadIdx.x) * 16)) + 98432), 1);
        }
        nr_tl_vit_joint::commit();
      }
      #pragma unroll
      for (int n_1 = 0; n_1 < 6; ++n_1) {
        nr_tl_vit_joint::shared4((&(bv[(n_1 * 4)])), (&(sh[((((((step & 1) * 10240) + (((((int)threadIdx.x) & 63) >> 5) * 3072)) + (n_1 * 512)) + ((((int)threadIdx.x) & 31) * 16)) + 4096)])));
      }
      #pragma unroll
      for (int r_1 = 0; r_1 < 4; ++r_1) {
        nr_tl_vit_joint::shared4((&(av[0])), (&(sh[(((((step & 1) * 10240) + ((((int)threadIdx.x) >> 6) * 2048)) + (r_1 * 512)) + ((((int)threadIdx.x) & 31) * 16))])));
        #pragma unroll
        for (int n_2 = 0; n_2 < 12; ++n_2) {
          uint64_t v = nr_tl_vit_joint::mma(av[0], av[1], av[2], av[3], bv[(n_2 * 2)], bv[((n_2 * 2) + 1)], c[((r_1 * 24) + (n_2 * 2))], c[(((r_1 * 24) + (n_2 * 2)) + 1)]);
          c[((r_1 * 24) + (n_2 * 2))] = ((uint)v);
          c[(((r_1 * 24) + (n_2 * 2)) + 1)] = ((uint)(v >> (uint64_t)32));
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int r_2 = 0; r_2 < 4; ++r_2) {
      #pragma unroll
      for (int co = 0; co < 3; ++co) {
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          if (split == 0) {
            nr_tl_vit_joint::store4((&(Raw[(((((((co * 65536) + ((((int)threadIdx.x) >> 6) * 32768)) + (r_2 * 8192)) + (((int)blockIdx.x) * 512)) + (((((int)threadIdx.x) & 63) >> 5) * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4))])), c[(((r_2 * 24) + (co * 8)) + (pair * 4))], c[((((r_2 * 24) + (co * 8)) + (pair * 4)) + 1)], c[((((r_2 * 24) + (co * 8)) + (pair * 4)) + 2)], c[((((r_2 * 24) + (co * 8)) + (pair * 4)) + 3)]);
          } else {
            nr_tl_vit_joint::global4((&(rv[0])), (&(Raw[(((((((co * 65536) + ((((int)threadIdx.x) >> 6) * 32768)) + (r_2 * 8192)) + (((int)blockIdx.x) * 512)) + (((((int)threadIdx.x) & 63) >> 5) * 256)) + (pair * 128)) + ((((int)threadIdx.x) & 31) * 4))])));
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; ++j_2) {
              uint a = rv[j_2];
              uint b = c[((((r_2 * 24) + (co * 8)) + (pair * 4)) + j_2)];
              c[((((r_2 * 24) + (co * 8)) + (pair * 4)) + j_2)] = nr_add2(a, b);
            }
          }
        }
        if (split == 1) {
          if (co < 2) {
            #pragma unroll
            for (int h_3 = 0; h_3 < 2; ++h_3) {
              uint b_1 = c[(((r_2 * 24) + (co * 8)) + h_3)];
              uint b_2 = c[((((r_2 * 24) + (co * 8)) + h_3) + 2)];
              uint b_3 = c[((((r_2 * 24) + (co * 8)) + h_3) + 4)];
              uint b_4 = c[((((r_2 * 24) + (co * 8)) + h_3) + 6)];
              uint c_1 = nr_mul2(b_3, b_3);
              uint a_1 = nr_fma2(b_1, b_1, c_1);
              uint c_2 = nr_mul2(b_4, b_4);
              uint b_5 = nr_fma2(b_2, b_2, c_2);
              uint a_2 = nr_add2(a_1, b_5);
              uint b_6 = nr_xor(a_2, 2);
              uint a_3 = nr_add2(a_2, b_6);
              uint b_7 = nr_xor(a_3, 1);
              uint a_4 = nr_add2(a_3, b_7);
              uint a_5 = (a_4 >> (uint)16);
              half_t a_6 = ((half_t)nr_half(a_4));
              half_t b_8 = ((half_t)nr_half((a_4 >> (uint)16)));
              half_t den = cutlass::half_t(__hmax((((half_t)nr_add(((float)a_6), ((float)b_8)))).to_half(), (half_t(0x1.04p-14f/*6.198883e-05*/)).to_half()));
              half_t inv = ((half_t)nr_rsqrt(((float)den)));
              float a_7 = ((float)inv);
              uint b_9 = nr_dup(((float)inv));
              #pragma unroll
              for (int n_3 = 0; n_3 < 4; ++n_3) {
                uint a_8 = c[((((r_2 * 24) + (co * 8)) + (n_3 * 2)) + h_3)];
                z = nr_mul2(a_8, b_9);
                if (co == 0) {
                  uint a_9 = z;
                  uint b_10 = nr_dup(0x1.6ap+2f/*5.656250e+00*/);
                  z = nr_mul2(a_9, b_10);
                  float a_10 = ((float)((half_t)Scale[((((int)blockIdx.x) * 2) + ((((int)threadIdx.x) & 63) >> 5))]));
                  uint a_11 = z;
                  uint b_11 = nr_dup(a_10);
                  z = nr_mul2(a_11, b_11);
                }
                c[((((r_2 * 24) + (co * 8)) + (n_3 * 2)) + h_3)] = z;
              }
            }
          }
          if (co < 2) {
            if (co == 0) {
              #pragma unroll
              for (int n_4 = 0; n_4 < 2; ++n_4) {
                #pragma unroll
                for (int h_4 = 0; h_4 < 2; ++h_4) {
                  uint a_12 = c[((((r_2 * 24) + (co * 8)) + (n_4 * 4)) + h_4)];
                  uint a_13 = c[(((((r_2 * 24) + (co * 8)) + (n_4 * 4)) + h_4) + 2)];
                  uint v_1 = (nr_qpair(a_12) | (nr_qpair(a_13) << (uint)16));
                  nr_st32((&(Q[(((((((((int)threadIdx.x) >> 6) * 65536) + (r_2 * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) & 63) * 16)) + (n_4 * 8)) + (h_4 * 4))])), v_1);
                }
              }
            } else {
              #pragma unroll
              for (int n_5 = 0; n_5 < 2; ++n_5) {
                #pragma unroll
                for (int h_5 = 0; h_5 < 2; ++h_5) {
                  uint a_14 = c[((((r_2 * 24) + (co * 8)) + (n_5 * 4)) + h_5)];
                  uint a_15 = c[(((((r_2 * 24) + (co * 8)) + (n_5 * 4)) + h_5) + 2)];
                  uint v_2 = (nr_qpair(a_14) | (nr_qpair(a_15) << (uint)16));
                  nr_st32((&(K[(((((((((int)threadIdx.x) >> 6) * 65536) + (r_2 * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) & 63) * 16)) + (h_5 * 8)) + (n_5 * 4))])), v_2);
                }
              }
            }
          } else {
            #pragma unroll
            for (int n_6 = 0; n_6 < 4; ++n_6) {
              uint a_16 = c[(((r_2 * 24) + (n_6 * 2)) + 16)];
              uint a_17 = c[(((r_2 * 24) + (n_6 * 2)) + 17)];
              uint a_18 = (nr_qpair(a_16) | (nr_qpair(a_17) << (uint)16));
              uint a_19 = nr_shuffle(a_18, ((((int)threadIdx.x) & 31) & -5));
              uint b_12 = nr_shuffle(a_18, (((((int)threadIdx.x) & 31) & -5) | 4));
              int condval;
              if ((0 < (((((int)threadIdx.x) & 31) >> 2) & 1))) {
                condval = 29521;
              } else {
                condval = 25152;
              }
              uint v_3 = nr_perm(a_19, b_12, ((uint)condval));
              nr_st32((&(V[(((((((((((((int)threadIdx.x) >> 6) * 65536) + ((r_2 >> 1) * 32768)) + (((int)blockIdx.x) * 2048)) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((n_6 >> 1) * 512)) + (((((int)threadIdx.x) & 31) & 3) * 128)) + ((((((int)threadIdx.x) & 31) >> 2) & 1) * 64)) + ((((((int)threadIdx.x) & 31) & -5) >> 3) * 16)) + ((n_6 & 1) * 8)) + ((r_2 & 1) * 4))])), v_3);
            }
          }
        }
      }
    }
    __syncthreads();
  }
  if (((int)blockIdx.x) == 0) {
    if (((int)threadIdx.x) < 16) {
      C0[((int)threadIdx.x)] = 1;
      C1[((int)threadIdx.x)] = 0;
    }
  }
}

