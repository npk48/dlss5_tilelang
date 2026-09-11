#pragma once
// Single-instruction memory operations only; all producer/consumer logic is TL.
namespace nr_tl_eight_joint {
__device__ __forceinline__ int ld128(unsigned *q, const void *p) {
 asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(q[0]),"=r"(q[1]),"=r"(q[2]),"=r"(q[3]) : "l"(p) : "memory"); return 0;
}
__device__ __forceinline__ unsigned ld32(const void *p) {
 unsigned q; asm volatile("ld.global.u32 %0, [%1];" : "=r"(q) : "l"(p) : "memory"); return q;
}
__device__ __forceinline__ int lds128(unsigned *q, const void *p) {
 unsigned a=__cvta_generic_to_shared(p);
 asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(q[0]),"=r"(q[1]),"=r"(q[2]),"=r"(q[3]) : "r"(a) : "memory"); return 0;
}
__device__ __forceinline__ int sts128(void *p, unsigned a, unsigned b, unsigned c, unsigned d) {
 unsigned s=__cvta_generic_to_shared(p);
 asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(s),"r"(a),"r"(b),"r"(c),"r"(d) : "memory"); return 0;
}
__device__ __forceinline__ int sts16(void *p, unsigned offset, unsigned v) {
 unsigned s=__cvta_generic_to_shared(p)+offset;
 asm volatile("st.shared.u16 [%0], %1;" :: "r"(s),"h"((unsigned short)v) : "memory"); return 0;
}
__device__ __forceinline__ int st128(void *p, unsigned a, unsigned b, unsigned c, unsigned d) {
 asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p),"r"(a),"r"(b),"r"(c),"r"(d) : "memory"); return 0;
}
__device__ __forceinline__ int st32(void *p, unsigned a) {
 asm volatile("st.global.u32 [%0], %1;" :: "l"(p),"r"(a) : "memory"); return 0;
}
}
