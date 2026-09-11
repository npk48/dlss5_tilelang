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
