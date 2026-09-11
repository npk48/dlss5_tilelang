__device__ __forceinline__ unsigned wide_packet_prmt(unsigned a,unsigned b,unsigned s) { unsigned r; asm("prmt.b32 %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s)); return r; }
