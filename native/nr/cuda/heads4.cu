// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// OUTER wide_input_packet/wide4.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_wide4_5 {
// Organized full 4H/8H: accepted uniform-bank math; real dispatch metadata.

namespace wide_activation {
using u32 = unsigned int;
__device__ __forceinline__ int lane() {
    return threadIdx.x;
}
template <int I> struct Tag {
    static constexpr int value = I;
};
template <int N, int I = 0, class F> __device__ __forceinline__ void each(F f) {
    if constexpr (I < N) {
        f(Tag<I>{});
        each<N, I + 1>(f);
    }
}
template <int N> struct Words {
    u32 v[N];
};
__device__ __forceinline__ Words<16> hidden_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value4_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 8);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 8);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 8);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 8);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 8);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 8);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 8);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 8);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 8);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 8);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 8);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 8);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 8);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 8);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 8);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 8);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 8) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_a(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 1);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 1);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 1);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 1);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 1);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 1);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 1);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 1);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 1) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value8_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 4);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 4);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 4);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 4);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 4);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 4);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 4);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 4);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 4);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 4);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 4);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 4);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 4);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 4);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 4);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 4);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 4) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[3] = (((lane() >> 1) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[4] = (((lane() >> 1) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[9] = (((lane() >> 1) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[11] = (((lane() >> 1) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[12] = (((lane() >> 1) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[14] = (((lane() >> 1) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[4] = (((lane() >> 4) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[6] = (((lane() >> 4) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[9] = (((lane() >> 4) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[11] = (((lane() >> 4) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<32> prob8_a(Words<32> x) {
    {
        Words<32> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        y.v[16] = __byte_perm(x.v[16], x.v[18], 0x5410);
        y.v[17] = __byte_perm(x.v[17], x.v[19], 0x5410);
        y.v[18] = __byte_perm(x.v[18], x.v[16], 0x3276);
        y.v[19] = __byte_perm(x.v[19], x.v[17], 0x3276);
        y.v[20] = __byte_perm(x.v[20], x.v[22], 0x5410);
        y.v[21] = __byte_perm(x.v[21], x.v[23], 0x5410);
        y.v[22] = __byte_perm(x.v[22], x.v[20], 0x3276);
        y.v[23] = __byte_perm(x.v[23], x.v[21], 0x3276);
        y.v[24] = __byte_perm(x.v[24], x.v[26], 0x5410);
        y.v[25] = __byte_perm(x.v[25], x.v[27], 0x5410);
        y.v[26] = __byte_perm(x.v[26], x.v[24], 0x3276);
        y.v[27] = __byte_perm(x.v[27], x.v[25], 0x3276);
        y.v[28] = __byte_perm(x.v[28], x.v[30], 0x5410);
        y.v[29] = __byte_perm(x.v[29], x.v[31], 0x5410);
        y.v[30] = __byte_perm(x.v[30], x.v[28], 0x3276);
        y.v[31] = __byte_perm(x.v[31], x.v[29], 0x3276);
        x = y;
    }
    {
        Words<32> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        u32 t16 = __shfl_xor_sync(0xffffffff, x.v[17], 2);
        y.v[16] = (((lane() >> 1) & 1) == 0) ? x.v[16] : t16;
        u32 t17 = __shfl_xor_sync(0xffffffff, x.v[16], 2);
        y.v[17] = (((lane() >> 1) & 1) == 1) ? x.v[17] : t17;
        u32 t18 = __shfl_xor_sync(0xffffffff, x.v[19], 2);
        y.v[18] = (((lane() >> 1) & 1) == 0) ? x.v[18] : t18;
        u32 t19 = __shfl_xor_sync(0xffffffff, x.v[18], 2);
        y.v[19] = (((lane() >> 1) & 1) == 1) ? x.v[19] : t19;
        u32 t20 = __shfl_xor_sync(0xffffffff, x.v[21], 2);
        y.v[20] = (((lane() >> 1) & 1) == 0) ? x.v[20] : t20;
        u32 t21 = __shfl_xor_sync(0xffffffff, x.v[20], 2);
        y.v[21] = (((lane() >> 1) & 1) == 1) ? x.v[21] : t21;
        u32 t22 = __shfl_xor_sync(0xffffffff, x.v[23], 2);
        y.v[22] = (((lane() >> 1) & 1) == 0) ? x.v[22] : t22;
        u32 t23 = __shfl_xor_sync(0xffffffff, x.v[22], 2);
        y.v[23] = (((lane() >> 1) & 1) == 1) ? x.v[23] : t23;
        u32 t24 = __shfl_xor_sync(0xffffffff, x.v[25], 2);
        y.v[24] = (((lane() >> 1) & 1) == 0) ? x.v[24] : t24;
        u32 t25 = __shfl_xor_sync(0xffffffff, x.v[24], 2);
        y.v[25] = (((lane() >> 1) & 1) == 1) ? x.v[25] : t25;
        u32 t26 = __shfl_xor_sync(0xffffffff, x.v[27], 2);
        y.v[26] = (((lane() >> 1) & 1) == 0) ? x.v[26] : t26;
        u32 t27 = __shfl_xor_sync(0xffffffff, x.v[26], 2);
        y.v[27] = (((lane() >> 1) & 1) == 1) ? x.v[27] : t27;
        u32 t28 = __shfl_xor_sync(0xffffffff, x.v[29], 2);
        y.v[28] = (((lane() >> 1) & 1) == 0) ? x.v[28] : t28;
        u32 t29 = __shfl_xor_sync(0xffffffff, x.v[28], 2);
        y.v[29] = (((lane() >> 1) & 1) == 1) ? x.v[29] : t29;
        u32 t30 = __shfl_xor_sync(0xffffffff, x.v[31], 2);
        y.v[30] = (((lane() >> 1) & 1) == 0) ? x.v[30] : t30;
        u32 t31 = __shfl_xor_sync(0xffffffff, x.v[30], 2);
        y.v[31] = (((lane() >> 1) & 1) == 1) ? x.v[31] : t31;
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[18];
        y.v[18] = x.v[17];
        y.v[19] = x.v[19];
        y.v[20] = x.v[20];
        y.v[21] = x.v[22];
        y.v[22] = x.v[21];
        y.v[23] = x.v[23];
        y.v[24] = x.v[24];
        y.v[25] = x.v[26];
        y.v[26] = x.v[25];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[30];
        y.v[30] = x.v[29];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[17];
        y.v[18] = x.v[18];
        y.v[19] = x.v[19];
        y.v[20] = x.v[24];
        y.v[21] = x.v[25];
        y.v[22] = x.v[26];
        y.v[23] = x.v[27];
        y.v[24] = x.v[20];
        y.v[25] = x.v[21];
        y.v[26] = x.v[22];
        y.v[27] = x.v[23];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[16];
        y.v[9] = x.v[17];
        y.v[10] = x.v[18];
        y.v[11] = x.v[19];
        y.v[12] = x.v[20];
        y.v[13] = x.v[21];
        y.v[14] = x.v[22];
        y.v[15] = x.v[23];
        y.v[16] = x.v[8];
        y.v[17] = x.v[9];
        y.v[18] = x.v[10];
        y.v[19] = x.v[11];
        y.v[20] = x.v[12];
        y.v[21] = x.v[13];
        y.v[22] = x.v[14];
        y.v[23] = x.v[15];
        y.v[24] = x.v[24];
        y.v[25] = x.v[25];
        y.v[26] = x.v[26];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_slab_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<4> hidden_slab_a(Words<4> x) {
    {
        Words<4> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> publish_output(Words<8> x) {
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 2) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        x = y;
    }
    return x;
}
} // namespace wide_activation
namespace wide_activation {
__device__ __forceinline__ Words<16> query4_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key4_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query8_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key8_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[5] = (((lane() >> 4) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[7] = (((lane() >> 4) & 1) == 0) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[8] = (((lane() >> 4) & 1) == 1) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[10] = (((lane() >> 4) & 1) == 1) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
} // namespace wide_activation

// Address-only ownership metadata, never reordered weights or activation values.
namespace physical {
struct Cross {
    const int *pool_input, *pool_output, *up_inverse;
    const half *projection, *gate;
    const unsigned char *matrix;
    int pool_out, skip;
};
__device__ __forceinline__ unsigned char encode(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half decode(unsigned char x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ int permute(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ void multiply(uint2 &c, uint4 a, uint2 b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c.x), "+r"(c.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ half component(uint2 c, int j) {
    return __ushort_as_half((j < 2 ? c.x : c.y) >> ((j & 1) * 16));
}
__device__ __forceinline__ unsigned char fused_input(const unsigned char *r, Cross x, int raw) {
    if (raw < 0)
        return 0;
    int2 inverse = ((const int2 *)x.up_inverse)[raw];
    return encode(__hfma(decode(r[x.skip + raw]), x.gate[inverse.y], x.projection[inverse.x]));
}
// Called after ALL projection reads of A finish. pi maps the original pair tree
// to canonical Half positions produced by this CTA, including invalid -1 items.
template <int H> __device__ void pool(unsigned char *r, const half *values, Cross x) {
    constexpr int K = H * 32, N = K * 2;
    int lane = threadIdx.x, warp = threadIdx.y;
    const int *pi = x.pool_input + blockIdx.x * 16 * 4 * K;
    const int *po = x.pool_output + blockIdx.x * 16 * N;
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int kp = 0; kp < H; kp++) {
            uint4 a = {0, 0, 0, 0};
            unsigned *aw = (unsigned *)&a;
#pragma unroll
            for (int word = 0; word < 4; word++) {
                int row = lane / 4 + (word & 1) * 8;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    int k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                    if constexpr (H != 2)
                        k = permute(k);
                    half v[4];
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        int index = pi[(row * 4 + t) * K + k];
                        v[t] = index < 0 ? __float2half(0) : values[index];
                    }
                    aw[word] |=
                        unsigned(encode(__hmul(__hadd(__hadd(v[0], v[1]), __hadd(v[2], v[3])),
                                               __float2half(.25f))))
                        << (b * 8);
                }
            }
#pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + lane * 16);
                multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            }
        }
#pragma unroll
        for (int n = 0; n < 4; n++)
            for (int j = 0; j < 4; j++) {
                int row = lane / 4 + j / 2 * 8, col = column + n * 8 + (lane & 3) * 2 + (j & 1),
                    out = po[row * N + col];
                if (out >= 0)
                    r[x.pool_out + out] = encode(component(c[n], j));
            }
    }
}
template <int H>
__device__ void project(const unsigned char *r, half *dest, const unsigned char *w, const int *im,
                        int input, int rows) {
    constexpr int K = H * 64, N = H * 32;
    int lane = threadIdx.x, warp = threadIdx.y, first = blockIdx.x * 16;
    uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (int kp = 0; kp < K / 32; kp++) {
        uint4 a = {0, 0, 0, 0};
        unsigned *aw = (unsigned *)&a;
#pragma unroll
        for (int word = 0; word < 4; word++)
            for (int b = 0; b < 4; b++) {
                int row = first + lane / 4 + (word & 1) * 8,
                    k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                if constexpr (H != 2)
                    k = permute(k);
                int index = row < rows ? im[row * K + k] : -1;
                aw[word] |= unsigned(index < 0 ? 0 : r[input + index]) << (8 * b);
            }
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            uint4 b = *(const uint4 *)(w + kp * N * 32 + warp * 1024 + pair * 512 + lane * 16);
            multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        }
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int j = 0; j < 4; j++) {
            int row = first + lane / 4 + j / 2 * 8,
                col = warp * 32 + n * 8 + (lane & 3) * 2 + (j & 1);
            if (row < rows)
                dest[row * N + col] = component(c[n], j);
        }
}
} // namespace physical
using namespace wide_activation;
using u32 = unsigned int;
using u8 = unsigned char;
struct RawHalf2 {
    u32 bits;
};
struct E4x4 {
    u32 bits;
};
struct A128 {
    u32 v[4];
};
struct B128 {
    u32 v[4];
};
struct Frag {
    u32 x, y;
};
__device__ __forceinline__ half hh(u32 x) {
    return __ushort_as_half((unsigned short)x);
}
__device__ __forceinline__ u32 hb(half x) {
    return __half_as_ushort(x);
}
__device__ __forceinline__ half dec(u8 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ u8 enc(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half part(Frag x, int i) {
    return hh((i < 2 ? x.x : x.y) >> ((i & 1) * 16));
}
__device__ __forceinline__ void put(Frag &x, int i, half h) {
    u32 &v = i < 2 ? x.x : x.y;
    int s = (i & 1) * 16;
    v = (v & ~(65535u << s)) | (hb(h) << s);
}
__device__ __forceinline__ void mma(Frag &d, const u32 *a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int route(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ half2 h2(u32 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.bits = x;
    return v.pair;
}
__device__ __forceinline__ u32 bits2(half2 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.pair = x;
    return v.bits;
}
__device__ __forceinline__ half2 constant2(float x) {
    return __float2half2_rn(x);
}
__device__ __forceinline__ half2 activation2(half2 x) {
    half2 z = __hmin2(__hmax2(x, constant2(-4)), constant2(4));
    half2 g = __hfma2(__habs2(z), constant2(-0.055908203125f), constant2(0.447265625f));
    return __hmul2(x, __hfma2(z, g, constant2(0.89453125f)));
}
__device__ __forceinline__ u32 encode4(u32 a, u32 b) {
    unsigned short x = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(a), __NV_SATFINITE, __NV_E4M3);
    unsigned short y = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(b), __NV_SATFINITE, __NV_E4M3);
    return u32(x) | (u32(y) << 16);
}
template <int M = 4, int N = 4> struct Packed {
    u32 v[M][N];
    // Call sites have warp-uniform fragment indices; only the byte/lane varies.
    __device__ __forceinline__ u8 get(int row, int col) const {
        u32 result = __shfl_sync(0xffffffff, v[row / 16][col / 8], (row & 7) * 4 + (col & 7) / 2);
        return (result >> (((row & 8) ? 2 : 0) + (col & 1)) * 8) & 255;
    }
    // PV8 A is the sole exception: column bit3 depends on destination lane.
    // Shuffle both fixed registers BEFORE choosing; never gather the full fragment.
    __device__ __forceinline__ u8 pv8(int row, int col) const {
        int n = (col / 8) & ~1, src = (row & 7) * 4 + (col & 7) / 2;
        u32 a = __shfl_sync(0xffffffff, v[row / 16][n], src);
        u32 b = __shfl_sync(0xffffffff, v[row / 16][n + 1], src);
        return (((col & 8) ? b : a) >> ((((row & 8) ? 2 : 0) + (col & 1)) * 8)) & 255;
    }
};
template <int M, int N>
__device__ __forceinline__ Packed<M, N> pack(Frag (&f)[M][N], bool act = false) {
    Packed<M, N> p;
#pragma unroll

#pragma unroll
    for (int m = 0; m < M; m++) {
#pragma unroll

#pragma unroll
        for (int n = 0; n < N; n++) {
            u32 x = f[m][n].x, y = f[m][n].y;
            if (act) {
                x = bits2(activation2(h2(x)));
                y = bits2(activation2(h2(y)));
            }
            p.v[m][n] = encode4(x, y);
        }
    }
    return p;
}
// Packet [head/K32][M16][lane][A word]. Same 64*32*H bytes.
// A words are the exact routed MMA bytes, not row-major values reinterpreted.
template <int H> __device__ __forceinline__ void shared_a(const u8 *s, u32 *a, int m, int kp) {
    unsigned addr = __cvta_generic_to_shared(s + kp * 2048 + m * 512 + threadIdx.x * 16);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}
__device__ __forceinline__ int qorder(int x) {
    return (x & 17) | ((x & 2) << 2) | ((x & 12) >> 1);
}
template <int H> __device__ __forceinline__ int pvorder(int x) {
    if constexpr (H == 4)
        return (x & 32) | ((x & 1) << 1) | ((x & 2) << 3) | ((x & 4)) | ((x & 8) >> 3) |
               ((x & 16) >> 1);
    else
        return (x & 33) | ((x & 2) << 3) | ((x & 4) >> 1) | (x & 8) | ((x & 16) >> 2);
}
template <int H> __device__ __forceinline__ int sumorder(int x) {
    if constexpr (H == 4)
        return ((x & 1) << 4) | ((x & 2) << 2) | ((x & 4) << 3) | ((x & 8) >> 1) | ((x & 16) >> 4) |
               ((x & 32) >> 4);
    else
        return ((x & 1) << 4) | ((x & 2) << 1) | ((x & 4) << 3) | ((x & 8) >> 2) | ((x & 16) >> 1) |
               ((x & 32) >> 5);
}
template <int Mode, int H = 4, int M, int N>
__device__ __forceinline__ void reg_a(const Packed<M, N> &p, u32 *a, int m, int kp) {
    int l = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        u32 v = 0;
        int row = m * 16 + l / 4 + (r & 1) * 8;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int k = kp * 32 + (r / 2) * 16 + (l & 3) * 4 + b;
            if constexpr (Mode == 1)
                k = route(k);
            if constexpr (Mode == 2)
                k = qorder(k);
            if constexpr (Mode == 3)
                k = pvorder<H>(k);
            if constexpr (Mode == 3 && H == 8)
                v |= u32(p.pv8(row, k)) << (b * 8);
            else
                v |= u32(p.get(row, k)) << (b * 8);
        }
        a[r] = v;
    }
}
__device__ __forceinline__ void weighted(Frag &c, const u32 *a, const u8 *w, int off, int n) {
    const u32 *b = (const u32 *)(w + off + (n / 2) * 512 + threadIdx.x * 16 + (n & 1) * 8);
    mma(c, a, b[0], b[1]);
}
// A B128 lives outside the entire owned M slab; both N8 halves consume it.
template <int M, int First = 0>
__device__ __forceinline__ void weighted_shared(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                                int kp) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            u32 a[4];
            shared_a<1>(s, a, m, kp);
            mma(c[m][pair * 2], a, b.x, b.y);
            mma(c[m][pair * 2 + 1], a, b.z, b.w);
        });
    });
}
__device__ __forceinline__ void weighted_register(Frag (&c)[4][4], const Words<16> &a, const u8 *w,
                                                  int off) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma(c[m][pair * 2], aa, b.x, b.y);
            mma(c[m][pair * 2 + 1], aa, b.z, b.w);
        });
    });
}
__device__ __forceinline__ u8 packet_byte(const u8 *s, int row, int col) {
    int k = (col & ~14) | ((col & 8) >> 2) | ((col & 2) << 1) | ((col & 4) << 1);
    return s[(k / 32) * 2048 + (row / 16) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 +
             ((row & 8) ? 4 : 0) + ((k & 16) ? 8 : 0) + (k & 3)];
}
// Exact inverse of the existing encoded FF packet publication.
template <int M = 4, int First = 0> __device__ __forceinline__ Packed<> read_encoded(const u8 *s) {
    Packed<> p = {};
    unsigned base = __cvta_generic_to_shared(s + threadIdx.y * 2048 + threadIdx.x * 16);
    each<M>([&](auto mt) {
        constexpr int m = First + decltype(mt)::value;
        uint4 a;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4+%5];"
                     : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                     : "r"(base), "n"(m * 512)
                     : "memory");
        p.v[m][0] = __byte_perm(a.x, a.y, 0x5410);
        p.v[m][1] = __byte_perm(a.x, a.y, 0x7632);
        p.v[m][2] = __byte_perm(a.z, a.w, 0x5410);
        p.v[m][3] = __byte_perm(a.z, a.w, 0x7632);
    });
    return p;
}
template <int H>
__device__ __forceinline__ void exchange(const Packed<> &p, u8 *s, int rowbase = 0, int count = 4) {
    Words<16> x;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            x.v[m * 4 + n] = p.v[m][n];
        });
    });
    auto a = hidden_a(x);
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        if (m < count) {
            unsigned addr = __cvta_generic_to_shared(s + threadIdx.y * 2048 +
                                                     (rowbase / 16 + m) * 512 + threadIdx.x * 16);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a.v[m * 4]),
                         "r"(a.v[m * 4 + 1]), "r"(a.v[m * 4 + 2]), "r"(a.v[m * 4 + 3])
                         : "memory");
        }
    });
}
template <int H, int M = 4, int First = 0>
__device__ __forceinline__ void initial(Frag (&c)[4][4], const Packed<> &p, const u8 *w, int off) {
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        int col = threadIdx.y * 32 + n * 8 + (threadIdx.x & 3) * 2;
        RawHalf2 gate{*(const u32 *)(w + off + col * 2)};
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            E4x4 packed{p.v[m][n]};
            half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)packed.bits, __NV_E4M3));
            half2 hi =
                __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)(packed.bits >> 16), __NV_E4M3));
            c[m][n] = {bits2(__hmul2(lo, h2(gate.bits))), bits2(__hmul2(hi, h2(gate.bits)))};
        });
    });
}
template <int M> __device__ __forceinline__ Packed<M, 4> normpack(Frag (&z)[M][4], half scale) {

#pragma unroll
    for (int m = 0; m < M; m++)
        for (int row = 0; row < 2; row++) {
            u32 a = row ? z[m][0].y : z[m][0].x, b = row ? z[m][1].y : z[m][1].x;
            u32 c = row ? z[m][2].y : z[m][2].x, d = row ? z[m][3].y : z[m][3].x;
            u32 sum = bits2(__hadd2(__hfma2(h2(a), h2(a), __hmul2(h2(c), h2(c))),
                                    __hfma2(h2(b), h2(b), __hmul2(h2(d), h2(d)))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 2))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 1))));
            float x = __half2float(
                      __hmax(__hadd(hh(sum), hh(sum >> 16)), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(x));
            half h = __float2half_rn(inv);

#pragma unroll
            for (int n = 0; n < 4; n++) {
                u32 &v = row ? z[m][n].y : z[m][n].x;
                v = bits2(
                    __hmul2(__hmul2(h2(v), __halves2half2(h, h)), __halves2half2(scale, scale)));
            }
        }
    return pack(z);
}
// Included after the unchanged wide arithmetic and packing primitives.
template <int M, int N> __device__ __forceinline__ Words<M * N> words(const Packed<M, N> &p) {
    Words<M * N> x;
    each<M>([&](auto mt) {
        each<N>([&](auto nt) {
            constexpr int m = decltype(mt)::value, n = decltype(nt)::value;
            x.v[m * N + n] = p.v[m][n];
        });
    });
    return x;
}
template <int H, int M>
__device__ __forceinline__ Packed<M, 8> physicalprobability(Frag (&c)[M][8]) {
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<2>([&](auto rt) {
                constexpr int row = decltype(rt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                half2 z = __hfma2(h2(v), constant2(0.044921875f), constant2(1.30078125f));
                z = __hmin2(__hmax2(z, constant2(1.03125f)), constant2(1.5693359375f));
                v = ((bits2(z) << 5) & 0xffe0ffe0u) ^ 0x80008000u;
            });
        });
        each<2>([&](auto rt) {
            constexpr int row = decltype(rt)::value;
            Words<8> d;
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                d.v[n] = row ? c[m][n].y : c[m][n].x;
            });
            if constexpr (H == 4)
                d = den4_physical_paired(d);
            else
                d = den8_physical_paired(d);
            u32 group;
            each<4>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                u32 pair = bits2(__hadd2(h2(d.v[p]), h2(d.v[4 + p])));
                if constexpr (p == 0)
                    group = pair;
                else
                    group = bits2(__hadd2(h2(group), h2(pair)));
            });
            u32 total;
            each<4>([&](auto gt) {
                constexpr int g = decltype(gt)::value;
                u32 term = __shfl_sync(0xffffffff, group, (threadIdx.x & ~3) + g);
                if constexpr (g == 0)
                    total = term;
                else
                    total = bits2(__hadd2(h2(total), h2(term)));
            });
            float den = __half2float(__hmax(__hadd(hh(total), hh(total >> 16)),
                                            __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half r = __float2half_rn(inv);
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                v = bits2(__hmul2(h2(v), __halves2half2(r, r)));
            });
        });
    });
    return pack(c);
}
template <int H, int M, int First>
__device__ __forceinline__ Packed<> physicalattention(const Words<16> &q, const Words<16> &k,
                                                      const Words<16> &v, const u8 *w) {
    static_assert(M == 2 && (First == 0 || First == 2));
    Words<16> p;
    {
        Frag logits[2][8];
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4] = {q.v[(First + m) * 4], q.v[(First + m) * 4 + 1], q.v[(First + m) * 4 + 2],
                        q.v[(First + m) * 4 + 3]};
            each<4>([&](auto pt) {
                constexpr int packetlocal = decltype(pt)::value;
                constexpr int n = (packetlocal & 1) + 4 * (packetlocal >> 1),
                              packet = 8 * (First / 2) + 4 * m + packetlocal;
                // One raw aligned packet directly occupies its two future QK C pairs.
                asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(logits[m][n].x), "=r"(logits[m][n].y), "=r"(logits[m][n + 2].x),
                               "=r"(logits[m][n + 2].y)
                             : "l"(w + (H == 4 ? 0x24120 : 0x88220) + 8192 * threadIdx.y +
                                   512 * packet + 16 * threadIdx.x));
                mma(logits[m][n], a, k.v[n * 2], k.v[n * 2 + 1]);
                mma(logits[m][n + 2], a, k.v[(n + 2) * 2], k.v[(n + 2) * 2 + 1]);
            });
        });
        if constexpr (H == 4)
            p = prob4_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
        else
            p = prob8_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
    }
    Frag result[4][4] = {};
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            constexpr int off = (kp * M + m) * 4;
            u32 a[4] = {p.v[off], p.v[off + 1], p.v[off + 2], p.v[off + 3]};
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                mma(result[m][n], a, v.v[(kp * 4 + n) * 2], v.v[(kp * 4 + n) * 2 + 1]);
            });
        });
    });
    return pack(result);
}

// By-value ABI: decoded shape/grid/shift and symbol-derived input/output flags.
struct Layout {
    int height, width, grid_x, shift_x, shift_y, flags;
};
template <int H, bool Output>
__device__ __forceinline__ int address(Layout p, int tile, int slot, int n) {
    int x = (tile % p.grid_x) * 8 + (H == 4 ? ((slot >> 1) & 7) : (slot & 7)) + p.shift_x;
    int y = (tile / p.grid_x) * 8 + (H == 4 ? (((slot >> 4) & 3) * 2 + (slot & 1)) : (slot / 8)) +
            p.shift_y;
    if (x < 0 || x >= p.width || y < 0 || y >= p.height)
        return -1;
    if (p.flags & (Output ? 2 : 1)) {
        int lo = (n & 1) | ((n & 6) << 1) | ((n & 8) >> 2);
        return (y * p.width + x) * 16 + (n / 16) * p.height * p.width * 16 + lo;
    }
    int nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4);
    int st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3);
    if constexpr (H == 4)
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + (x / 4 + (p.width / 4) * (y / 4)) * 2048;
    else {
        st |= (x & 4) << 2;
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + ((st & 16) << 8) +
               (x / 8 + ((p.width + 7) / 8) * (y / 4)) * 8192;
    }
}
// Original4 chained: expand K4 unrolled (0x1000..); tail loop 0x2c10..2d80;
// QKV M64N96 K4 loop 0x33d0..3760; projection M32 K4 unrolled 0x66c0...
// A is loaded once per owned M16/K32, outside all component/pair consumers.
template <int M, int First = 0>
__device__ __forceinline__ void four_step(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                          int kp) {
    u32 a[M][4];
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<4>(s, a[m], First + m, kp);
    });
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            mma(c[First + m][pair * 2], a[m], b.x, b.y);
            mma(c[First + m][pair * 2 + 1], a[m], b.z, b.w);
        });
    });
}
template <int M, int First = 0>
__device__ __forceinline__ void four_unrolled(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                              int stride) {
    each<4>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        four_step<M, First>(c, s, w, off + kp * stride, kp);
    });
}
__device__ __forceinline__ void four_tail(Frag (&c)[4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++)
        four_step<4>(c, s, w, off + kp * 4096, kp);
}
__device__ __forceinline__ void four_qkv(Frag (&c)[3][4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}
__device__ __forceinline__ void four_qkv_prefix(Frag (&c)[3][4][4], const u8 *s, const u8 *w,
                                                int off) {
#pragma unroll 1
    for (int kp = 0; kp < 3; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}

// hidden_a swaps unit bit1 with register bit0 only: each M16 is closed.
// Both expand and reduce B128 pairs are loaded once and reused across M64.
template <int H>
__device__ __forceinline__ void finish_hidden(Frag (&expanded)[4][4], Frag (&reduced)[4][4],
                                              const u8 *s, const u8 *w, int off, int stride,
                                              int reduceoff) {
    uint4 eb[2], rb[2];
    each<2>([&](auto pt) {
        constexpr int p = decltype(pt)::value;
        eb[p] = *(const uint4 *)(w + off + (H - 1) * stride + p * 512 + threadIdx.x * 16);
        rb[p] = *(const uint4 *)(w + reduceoff + p * 512 + threadIdx.x * 16);
    });
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        A128 a;
        shared_a<H>(s, a.v, m, H - 1);
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(expanded[m][p * 2], a.v, eb[p].x, eb[p].y);
            mma(expanded[m][p * 2 + 1], a.v, eb[p].z, eb[p].w);
        });
        Words<4> e;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            RawHalf2 x{bits2(activation2(h2(expanded[m][n].x)))},
                y{bits2(activation2(h2(expanded[m][n].y)))};
            e.v[n] = encode4(x.bits, y.bits);
        });
        auto packet = hidden_slab_a(e);
        A128 active{{packet.v[0], packet.v[1], packet.v[2], packet.v[3]}};
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(reduced[m][p * 2], active.v, rb[p].x, rb[p].y);
            mma(reduced[m][p * 2 + 1], active.v, rb[p].z, rb[p].w);
        });
    });
}
// Finish one component at a time; Q/K normalization consumes a closed M16.
// Q/K physical routes cross canonical M16: route only at this component tail.
// V's complete physical transpose waits for all 64 rows. Attention is outside
// this function, after the complete K/V packets have been produced.
template <int H>
__device__ __forceinline__ void finish_qkv(Frag (&c)[3][4][4], Words<16> &q, Words<16> &k,
                                           Words<16> &v, const u8 *s, const u8 *w, int off,
                                           int stride, half scale) {
    A128 a[4];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<H>(s, a[m].v, m, H - 1);
    });
    each<3>([&](auto ct) {
        constexpr int component = decltype(ct)::value;
        uint4 b[2];
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            b[p] = *(const uint4 *)(w + off + (H - 1) * stride + component * 1024 + p * 512 +
                                    threadIdx.x * 16);
        });
        Words<16> encoded;
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<2>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                mma(c[component][m][p * 2], a[m].v, b[p].x, b[p].y);
                mma(c[component][m][p * 2 + 1], a[m].v, b[p].z, b[p].w);
            });
            Frag row[1][4];
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                row[0][n] = c[component][m][n];
            });
            Packed<1, 4> packed;
            if constexpr (component < 2)
                packed = normpack(row, component == 0 ? scale : __float2half(1));
            else
                packed = pack(row);
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                encoded.v[m * 4 + n] = packed.v[0][n];
            });
        });
        if constexpr (component == 0) {
            if constexpr (H == 4)
                q = query4_physical_a(encoded);
            else
                q = query8_physical_a(encoded);
        } else if constexpr (component == 1) {
            if constexpr (H == 4)
                k = key4_physical_b(encoded);
            else
                k = key8_physical_b(encoded);
        } else if constexpr (H == 4)
            v = value4_b(encoded);
        else
            v = value8_b(encoded);
    });
}
// M32 encoded C -> raw output packets. No shared scratch or dynamic word indexing.
template <int First, int Role>
__device__ __forceinline__ void publish(u8 *resource, int output, Layout layout, int tile,
                                        Frag (&out)[4][4]) {
    int l = threadIdx.x, warp = threadIdx.y;
    Words<8> x;
    each<8>([&](auto it) {
        constexpr int i = decltype(it)::value;
        x.v[i] = encode4(out[First + i / 4][i % 4].x, out[First + i / 4][i % 4].y);
    });
    if constexpr (Role == 4) {
        // source bits [byte2,lane5,word3] -> destination [0,3,7,8,4,5,6,1,9,2].
        // All source M values participate before any destination-dependent selection.
        x = publish_output(x);
        int row = First * 16 + l / 4 + (l & 1) * 16 + ((l >> 1) & 1) * 8;
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            int idx = address<4, true>(layout, tile, row, warp * 32 + p * 16);
            if (idx >= 0)
                *(uint4 *)(resource + output + idx) = {x.v[p * 4], x.v[p * 4 + 1], x.v[p * 4 + 2],
                                                       x.v[p * 4 + 3]};
        });
    } else {
        // Adapter enforces the fully proved real payload domain: H%4==0, W%4==0,
        // shiftY=0/-4. Each valid base implies all 16 bytes valid in this CTA/slab.
        each<2>([&](auto jt) {
            constexpr int j2 = decltype(jt)::value;
            int row = First * 16 + l / 4 + j2 * 8;
            int idx = address<4, true>(layout, tile, row, warp * 32 + (l & 3) * 2);
            if (idx >= 0) {
                constexpr int mask = j2 ? 0x7632 : 0x5410;
                uint4 v = {__byte_perm(x.v[0], x.v[1], mask), __byte_perm(x.v[4], x.v[5], mask),
                           __byte_perm(x.v[2], x.v[3], mask), __byte_perm(x.v[6], x.v[7], mask)};
                *(uint4 *)(resource + output + idx) = v;
            }
        });
    }
}
#include "routes4.cuh"
// Only final encoded pool A crosses warps, after both live body slabs die.
template <int H>
__device__ __forceinline__ void pool_packet(u8 *r, const u8 *s, physical::Cross x) {
    constexpr int N = H * 64;
    int l = threadIdx.x, warp = threadIdx.y;
    const int *po = x.pool_output + blockIdx.x * 16 * N;
#pragma unroll 1
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {};
#pragma unroll 1
        for (int kp = 0; kp < H; kp++) {
            uint4 a;
            unsigned addr = __cvta_generic_to_shared(s + kp * 512 + l * 16);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                         : "r"(addr));
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + l * 16);
                physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            });
        }
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                int row = l / 4 + j / 2 * 8, col = column + n * 8 + (l & 3) * 2 + (j & 1),
                    dest = po[row * N + col];
                if (dest >= 0)
                    r[x.pool_out + dest] = enc(physical::component(c[n], j));
            });
        });
    }
}
// Local projection rows are a permutation/subset of the original global rows.
// No inverse device read and no canonical Half reload. Canonical stores remain.
template <int H>
__device__ __forceinline__ void project_input(u8 *r, const u8 *w, Layout layout, int input,
                                              physical::Cross x, const int *ui, u8 *s) {
    constexpr int K = H * 64, N = H * 32;
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    uint2 c[4] = {};
#pragma unroll 1
    for (int kp = 0; kp < H * 2; kp++) {
        uint4 a = {};
        u32 *aw = (u32 *)&a;
        each<4>([&](auto wt) {
            constexpr int word = decltype(wt)::value;
            int row = up_global[tile][l / 4 + (word & 1) * 8],
                k = route(kp * 32 + (word / 2) * 16 + (l & 3) * 4);
            if (row >= 0) {
                int off = ui[row * K + k];
                aw[word] = off < 0 ? 0 : *(const u32 *)(r + input + off);
            }
        });
        each<2>([&](auto pt) {
            constexpr int pair = decltype(pt)::value;
            uint4 b = *(const uint4 *)(w + (H == 4 ? 0x18000 : 0x58000) + kp * N * 32 +
                                       warp * 1024 + pair * 512 + l * 16);
            physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        });
    }
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        each<4>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int row = up_global[tile][l / 4 + j / 2 * 8],
                col = warp * 32 + n * 8 + (l & 3) * 2 + (j & 1);
            if (row >= 0)
                ((half *)x.projection)[row * N + col] = physical::component(c[n], j);
        });
    });
    int cid = up_ids[tile];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4] = {};
        each<4>([&](auto wt) {
            constexpr int word = decltype(wt)::value;
            int token = m * 16 + l / 4 + (word & 1) * 8, rr = up_local[cid][token];
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                int col = route((word / 2) * 16 + (l & 3) * 4 + pair * 2),
                    bank = (col / 8) * 2 + (rr & 8) / 8, src = (rr & 7) * 4 + (col & 7) / 2;
                u32 value = 0;
                // Every source register index is compile-time fixed before the shuffle.
                each<8>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    u32 v = __shfl_sync(0xffffffff, (b & 1) ? c[b / 2].y : c[b / 2].x, src);
                    if (bank == b)
                        value = v;
                });
                each<2>([&](auto ht) {
                    constexpr int h = decltype(ht)::value;
                    int channel = warp * 32 + col + h,
                        raw = address<H, false>(layout, tile, token, channel);
                    if (raw >= 0 && rr >= 0)
                        a[word] |= u32(enc(__hfma(dec(r[x.skip + raw]), x.gate[channel],
                                                  hh(value >> (h * 16)))))
                                   << ((pair * 2 + h) * 8);
                });
            });
        });
        *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = make_uint4(a[0], a[1], a[2], a[3]);
    });
}
template <int Role>
__device__ __forceinline__ void four_body(u8 *resource, const u8 *w, Layout layout, int input,
                                          int output, int counter, int up, const u8 *inputResource,
                                          physical::Cross cross, const int *ui) {
    constexpr int H = 4, C = 128;
    extern __shared__ u8 s[];
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    u32 poolA[4] = {0, 0, 0, 0};
    if constexpr (Role == 3)
        project_input<H>(resource, w, layout, input, cross, ui, s);
    else {
        if constexpr (Role == 1) {
            each<4>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                uint4 a = {0, 0, 0, 0};
                u32 *av = (u32 *)&a;
                each<4>([&](auto rt) {
                    constexpr int word = decltype(rt)::value;
                    int row = m * 16 + l / 4 + (word & 1) * 8,
                        k = warp * 32 + (word / 2) * 16 + (l & 3) * 4;
                    int idx = address<H, false>(layout, tile, row, route(k));
                    av[word] = idx < 0 ? 0 : *(const u32 *)(inputResource + input + idx);
                });
                *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = a;
            });
        } else {
            each<2>([&](auto qt) {
                constexpr int q = decltype(qt)::value;
                int base0 =
                    address<H, false>(layout, tile, 32 * q + l / 4, route(32 * warp + 4 * (l & 3)));
                int base1 = address<H, false>(layout, tile, 32 * q + l / 4 + 8,
                                              route(32 * warp + 4 * (l & 3)));
                uint4 v0 = {0, 0, 0, 0}, v1 = {0, 0, 0, 0};
                if (base0 >= 0)
                    v0 = *(const uint4 *)(inputResource + input + base0);
                if (base1 >= 0)
                    v1 = *(const uint4 *)(inputResource + input + base1);
                *(uint4 *)(s + warp * 2048 + (2 * q) * 512 + l * 16) =
                    make_uint4(v0.x, v1.x, v0.z, v1.z);
                *(uint4 *)(s + warp * 2048 + (2 * q + 1) * 512 + l * 16) =
                    make_uint4(v0.y, v1.y, v0.w, v1.w);
            });
        }
    }
    __syncthreads();
    auto original = read_encoded<4, 0>(s);
    Frag reduced[4][4] = {};
#pragma unroll 1
    for (int hidden = 0; hidden < 4; hidden++) {
        Frag expanded[4][4] = {};
        each<3>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            four_step<4>(expanded, s, w, warp * C * 128 + hidden * 1024 + kp * 4096, kp);
        });
        finish_hidden<4>(expanded, reduced, s, w, warp * C * 128 + hidden * 1024, 4096,
                         0x10000 + warp * 4096 + hidden * 1024);
    }
    auto mid = pack(reduced);
    __syncthreads();
    exchange<H>(mid, s);
    __syncthreads();
    Frag ff[4][4];
    initial<H>(ff, original, w, Role == 3 ? 0x20000 : 0x18010);
    four_tail(ff, s, w, 0x14000 + warp * 1024);
    const u8 *aw = w + (Role == 3 ? 0x80e0 : 0);
    {
        auto feature = pack(ff);
        __syncthreads();
        exchange<H>(feature, s);
    }
    __syncthreads();
    Words<16> q, k, v;
    {
        Frag qkv[3][4][4] = {};
        four_qkv_prefix(qkv, s, aw, 0x18120 + warp * 3072);
        finish_qkv<4>(qkv, q, k, v, s, aw, 0x18120 + warp * 3072, 12288,
                      __float2half_rn(((const float *)(aw + 0x2c120))[warp]));
    }
    __syncthreads();
    each<2>([&](auto ft) {
        constexpr int first = decltype(ft)::value * 2;
        auto pv = physicalattention<H, 2, first>(q, k, v, aw);
        // FF stays in its original 4096-byte M32 slab until every warp's late read.
        auto feature = read_encoded<2, first>(s);
        __syncthreads();
        exchange<H>(pv, s, first * 16, 2);
        __syncthreads();
        Frag out[4][4];
        initial<H, 2, first>(out, feature, aw, 0x30130);
        four_unrolled<2, first>(out, s, aw, 0x2c130 + warp * 1024, 4096);
        if constexpr (Role == 2)
            collect_pool<first>(out, poolA);
        publish<first, Role>(resource, output, layout, tile, out);
        // All projection warps have consumed PV before the next slab is overwritten.
        __syncthreads();
    });
    if constexpr (Role == 2) {
        *(uint4 *)(s + warp * 512 + l * 16) = make_uint4(poolA[0], poolA[1], poolA[2], poolA[3]);
        __syncthreads();
        pool_packet<H>(resource, s, cross);
    }
    __threadfence();
    if (counter >= 0 && l == 0 && warp == 0)
        ((int *)(resource + counter))[tile] = 0;
}
extern "C" __global__ void four_input(u8 *r, const u8 *w, Layout layout, int in, int out, int cb,
                                      int up, const u8 *src, physical::Cross cross) {
    four_body<1>(r, w, layout, in, out, cb, up, src, cross, nullptr);
}
extern "C" __global__ void four_ordinary(u8 *r, const u8 *w, Layout layout, int in, int out, int cb,
                                         int up, const u8 *src, physical::Cross cross) {
    four_body<0>(r, w, layout, in, out, cb, up, src, cross, nullptr);
}
// Unused exported entry removed: wide4_ds

// Unused exported entry removed: wide4_up

extern "C" __global__ void four_output(u8 *r, const u8 *w, Layout layout, int in, int out, int cb,
                                       int up, const u8 *src, physical::Cross cross) {
    four_body<4>(r, w, layout, in, out, cb, up, src, cross, nullptr);
}

} // namespace nr_outer_wide4_5
// ============================================================================
// OUTER wide_transition_packet/ds4.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_ds4_6 {
// Organized full 4H/8H: accepted uniform-bank math; real dispatch metadata.

namespace wide_activation {
using u32 = unsigned int;
__device__ __forceinline__ int lane() {
    return threadIdx.x;
}
template <int I> struct Tag {
    static constexpr int value = I;
};
template <int N, int I = 0, class F> __device__ __forceinline__ void each(F f) {
    if constexpr (I < N) {
        f(Tag<I>{});
        each<N, I + 1>(f);
    }
}
template <int N> struct Words {
    u32 v[N];
};
__device__ __forceinline__ Words<16> hidden_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value4_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 8);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 8);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 8);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 8);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 8);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 8);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 8);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 8);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 8);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 8);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 8);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 8);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 8);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 8);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 8);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 8);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 8) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_a(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 1);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 1);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 1);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 1);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 1);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 1);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 1);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 1);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 1) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value8_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 4);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 4);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 4);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 4);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 4);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 4);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 4);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 4);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 4);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 4);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 4);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 4);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 4);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 4);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 4);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 4);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 4) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[3] = (((lane() >> 1) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[4] = (((lane() >> 1) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[9] = (((lane() >> 1) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[11] = (((lane() >> 1) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[12] = (((lane() >> 1) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[14] = (((lane() >> 1) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[4] = (((lane() >> 4) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[6] = (((lane() >> 4) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[9] = (((lane() >> 4) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[11] = (((lane() >> 4) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<32> prob8_a(Words<32> x) {
    {
        Words<32> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        y.v[16] = __byte_perm(x.v[16], x.v[18], 0x5410);
        y.v[17] = __byte_perm(x.v[17], x.v[19], 0x5410);
        y.v[18] = __byte_perm(x.v[18], x.v[16], 0x3276);
        y.v[19] = __byte_perm(x.v[19], x.v[17], 0x3276);
        y.v[20] = __byte_perm(x.v[20], x.v[22], 0x5410);
        y.v[21] = __byte_perm(x.v[21], x.v[23], 0x5410);
        y.v[22] = __byte_perm(x.v[22], x.v[20], 0x3276);
        y.v[23] = __byte_perm(x.v[23], x.v[21], 0x3276);
        y.v[24] = __byte_perm(x.v[24], x.v[26], 0x5410);
        y.v[25] = __byte_perm(x.v[25], x.v[27], 0x5410);
        y.v[26] = __byte_perm(x.v[26], x.v[24], 0x3276);
        y.v[27] = __byte_perm(x.v[27], x.v[25], 0x3276);
        y.v[28] = __byte_perm(x.v[28], x.v[30], 0x5410);
        y.v[29] = __byte_perm(x.v[29], x.v[31], 0x5410);
        y.v[30] = __byte_perm(x.v[30], x.v[28], 0x3276);
        y.v[31] = __byte_perm(x.v[31], x.v[29], 0x3276);
        x = y;
    }
    {
        Words<32> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        u32 t16 = __shfl_xor_sync(0xffffffff, x.v[17], 2);
        y.v[16] = (((lane() >> 1) & 1) == 0) ? x.v[16] : t16;
        u32 t17 = __shfl_xor_sync(0xffffffff, x.v[16], 2);
        y.v[17] = (((lane() >> 1) & 1) == 1) ? x.v[17] : t17;
        u32 t18 = __shfl_xor_sync(0xffffffff, x.v[19], 2);
        y.v[18] = (((lane() >> 1) & 1) == 0) ? x.v[18] : t18;
        u32 t19 = __shfl_xor_sync(0xffffffff, x.v[18], 2);
        y.v[19] = (((lane() >> 1) & 1) == 1) ? x.v[19] : t19;
        u32 t20 = __shfl_xor_sync(0xffffffff, x.v[21], 2);
        y.v[20] = (((lane() >> 1) & 1) == 0) ? x.v[20] : t20;
        u32 t21 = __shfl_xor_sync(0xffffffff, x.v[20], 2);
        y.v[21] = (((lane() >> 1) & 1) == 1) ? x.v[21] : t21;
        u32 t22 = __shfl_xor_sync(0xffffffff, x.v[23], 2);
        y.v[22] = (((lane() >> 1) & 1) == 0) ? x.v[22] : t22;
        u32 t23 = __shfl_xor_sync(0xffffffff, x.v[22], 2);
        y.v[23] = (((lane() >> 1) & 1) == 1) ? x.v[23] : t23;
        u32 t24 = __shfl_xor_sync(0xffffffff, x.v[25], 2);
        y.v[24] = (((lane() >> 1) & 1) == 0) ? x.v[24] : t24;
        u32 t25 = __shfl_xor_sync(0xffffffff, x.v[24], 2);
        y.v[25] = (((lane() >> 1) & 1) == 1) ? x.v[25] : t25;
        u32 t26 = __shfl_xor_sync(0xffffffff, x.v[27], 2);
        y.v[26] = (((lane() >> 1) & 1) == 0) ? x.v[26] : t26;
        u32 t27 = __shfl_xor_sync(0xffffffff, x.v[26], 2);
        y.v[27] = (((lane() >> 1) & 1) == 1) ? x.v[27] : t27;
        u32 t28 = __shfl_xor_sync(0xffffffff, x.v[29], 2);
        y.v[28] = (((lane() >> 1) & 1) == 0) ? x.v[28] : t28;
        u32 t29 = __shfl_xor_sync(0xffffffff, x.v[28], 2);
        y.v[29] = (((lane() >> 1) & 1) == 1) ? x.v[29] : t29;
        u32 t30 = __shfl_xor_sync(0xffffffff, x.v[31], 2);
        y.v[30] = (((lane() >> 1) & 1) == 0) ? x.v[30] : t30;
        u32 t31 = __shfl_xor_sync(0xffffffff, x.v[30], 2);
        y.v[31] = (((lane() >> 1) & 1) == 1) ? x.v[31] : t31;
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[18];
        y.v[18] = x.v[17];
        y.v[19] = x.v[19];
        y.v[20] = x.v[20];
        y.v[21] = x.v[22];
        y.v[22] = x.v[21];
        y.v[23] = x.v[23];
        y.v[24] = x.v[24];
        y.v[25] = x.v[26];
        y.v[26] = x.v[25];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[30];
        y.v[30] = x.v[29];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[17];
        y.v[18] = x.v[18];
        y.v[19] = x.v[19];
        y.v[20] = x.v[24];
        y.v[21] = x.v[25];
        y.v[22] = x.v[26];
        y.v[23] = x.v[27];
        y.v[24] = x.v[20];
        y.v[25] = x.v[21];
        y.v[26] = x.v[22];
        y.v[27] = x.v[23];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[16];
        y.v[9] = x.v[17];
        y.v[10] = x.v[18];
        y.v[11] = x.v[19];
        y.v[12] = x.v[20];
        y.v[13] = x.v[21];
        y.v[14] = x.v[22];
        y.v[15] = x.v[23];
        y.v[16] = x.v[8];
        y.v[17] = x.v[9];
        y.v[18] = x.v[10];
        y.v[19] = x.v[11];
        y.v[20] = x.v[12];
        y.v[21] = x.v[13];
        y.v[22] = x.v[14];
        y.v[23] = x.v[15];
        y.v[24] = x.v[24];
        y.v[25] = x.v[25];
        y.v[26] = x.v[26];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_slab_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<4> hidden_slab_a(Words<4> x) {
    {
        Words<4> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> publish_output(Words<8> x) {
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 2) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        x = y;
    }
    return x;
}
} // namespace wide_activation
namespace wide_activation {
__device__ __forceinline__ Words<16> query4_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key4_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query8_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key8_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[5] = (((lane() >> 4) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[7] = (((lane() >> 4) & 1) == 0) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[8] = (((lane() >> 4) & 1) == 1) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[10] = (((lane() >> 4) & 1) == 1) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
} // namespace wide_activation

// Address-only ownership metadata, never reordered weights or activation values.
namespace physical {
struct Cross {
    const int *pool_input, *pool_output, *up_inverse;
    const half *projection, *gate;
    const unsigned char *matrix;
    int pool_out, skip;
};
__device__ __forceinline__ unsigned char encode(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half decode(unsigned char x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ int permute(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ void multiply(uint2 &c, uint4 a, uint2 b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c.x), "+r"(c.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ half component(uint2 c, int j) {
    return __ushort_as_half((j < 2 ? c.x : c.y) >> ((j & 1) * 16));
}
__device__ __forceinline__ unsigned char fused_input(const unsigned char *r, Cross x, int raw) {
    if (raw < 0)
        return 0;
    int2 inverse = ((const int2 *)x.up_inverse)[raw];
    return encode(__hfma(decode(r[x.skip + raw]), x.gate[inverse.y], x.projection[inverse.x]));
}
// Called after ALL projection reads of A finish. pi maps the original pair tree
// to canonical Half positions produced by this CTA, including invalid -1 items.
template <int H> __device__ void pool(unsigned char *r, const half *values, Cross x) {
    constexpr int K = H * 32, N = K * 2;
    int lane = threadIdx.x, warp = threadIdx.y;
    const int *pi = x.pool_input + blockIdx.x * 16 * 4 * K;
    const int *po = x.pool_output + blockIdx.x * 16 * N;
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int kp = 0; kp < H; kp++) {
            uint4 a = {0, 0, 0, 0};
            unsigned *aw = (unsigned *)&a;
#pragma unroll
            for (int word = 0; word < 4; word++) {
                int row = lane / 4 + (word & 1) * 8;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    int k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                    if constexpr (H != 2)
                        k = permute(k);
                    half v[4];
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        int index = pi[(row * 4 + t) * K + k];
                        v[t] = index < 0 ? __float2half(0) : values[index];
                    }
                    aw[word] |=
                        unsigned(encode(__hmul(__hadd(__hadd(v[0], v[1]), __hadd(v[2], v[3])),
                                               __float2half(.25f))))
                        << (b * 8);
                }
            }
#pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + lane * 16);
                multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            }
        }
#pragma unroll
        for (int n = 0; n < 4; n++)
            for (int j = 0; j < 4; j++) {
                int row = lane / 4 + j / 2 * 8, col = column + n * 8 + (lane & 3) * 2 + (j & 1),
                    out = po[row * N + col];
                if (out >= 0)
                    r[x.pool_out + out] = encode(component(c[n], j));
            }
    }
}
template <int H>
__device__ void project(const unsigned char *r, half *dest, const unsigned char *w, const int *im,
                        int input, int rows) {
    constexpr int K = H * 64, N = H * 32;
    int lane = threadIdx.x, warp = threadIdx.y, first = blockIdx.x * 16;
    uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (int kp = 0; kp < K / 32; kp++) {
        uint4 a = {0, 0, 0, 0};
        unsigned *aw = (unsigned *)&a;
#pragma unroll
        for (int word = 0; word < 4; word++)
            for (int b = 0; b < 4; b++) {
                int row = first + lane / 4 + (word & 1) * 8,
                    k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                if constexpr (H != 2)
                    k = permute(k);
                int index = row < rows ? im[row * K + k] : -1;
                aw[word] |= unsigned(index < 0 ? 0 : r[input + index]) << (8 * b);
            }
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            uint4 b = *(const uint4 *)(w + kp * N * 32 + warp * 1024 + pair * 512 + lane * 16);
            multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        }
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int j = 0; j < 4; j++) {
            int row = first + lane / 4 + j / 2 * 8,
                col = warp * 32 + n * 8 + (lane & 3) * 2 + (j & 1);
            if (row < rows)
                dest[row * N + col] = component(c[n], j);
        }
}
} // namespace physical
using namespace wide_activation;
using u32 = unsigned int;
using u8 = unsigned char;
struct RawHalf2 {
    u32 bits;
};
struct E4x4 {
    u32 bits;
};
struct A128 {
    u32 v[4];
};
struct B128 {
    u32 v[4];
};
struct Frag {
    u32 x, y;
};
__device__ __forceinline__ half hh(u32 x) {
    return __ushort_as_half((unsigned short)x);
}
__device__ __forceinline__ u32 hb(half x) {
    return __half_as_ushort(x);
}
__device__ __forceinline__ half dec(u8 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ u8 enc(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half part(Frag x, int i) {
    return hh((i < 2 ? x.x : x.y) >> ((i & 1) * 16));
}
__device__ __forceinline__ void put(Frag &x, int i, half h) {
    u32 &v = i < 2 ? x.x : x.y;
    int s = (i & 1) * 16;
    v = (v & ~(65535u << s)) | (hb(h) << s);
}
__device__ __forceinline__ void mma(Frag &d, const u32 *a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int route(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ half2 h2(u32 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.bits = x;
    return v.pair;
}
__device__ __forceinline__ u32 bits2(half2 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.pair = x;
    return v.bits;
}
__device__ __forceinline__ half2 constant2(float x) {
    return __float2half2_rn(x);
}
__device__ __forceinline__ half2 activation2(half2 x) {
    half2 z = __hmin2(__hmax2(x, constant2(-4)), constant2(4));
    half2 g = __hfma2(__habs2(z), constant2(-0.055908203125f), constant2(0.447265625f));
    return __hmul2(x, __hfma2(z, g, constant2(0.89453125f)));
}
__device__ __forceinline__ u32 encode4(u32 a, u32 b) {
    unsigned short x = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(a), __NV_SATFINITE, __NV_E4M3);
    unsigned short y = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(b), __NV_SATFINITE, __NV_E4M3);
    return u32(x) | (u32(y) << 16);
}
template <int M = 4, int N = 4> struct Packed {
    u32 v[M][N];
    // Call sites have warp-uniform fragment indices; only the byte/lane varies.
    __device__ __forceinline__ u8 get(int row, int col) const {
        u32 result = __shfl_sync(0xffffffff, v[row / 16][col / 8], (row & 7) * 4 + (col & 7) / 2);
        return (result >> (((row & 8) ? 2 : 0) + (col & 1)) * 8) & 255;
    }
    // PV8 A is the sole exception: column bit3 depends on destination lane.
    // Shuffle both fixed registers BEFORE choosing; never gather the full fragment.
    __device__ __forceinline__ u8 pv8(int row, int col) const {
        int n = (col / 8) & ~1, src = (row & 7) * 4 + (col & 7) / 2;
        u32 a = __shfl_sync(0xffffffff, v[row / 16][n], src);
        u32 b = __shfl_sync(0xffffffff, v[row / 16][n + 1], src);
        return (((col & 8) ? b : a) >> ((((row & 8) ? 2 : 0) + (col & 1)) * 8)) & 255;
    }
};
template <int M, int N>
__device__ __forceinline__ Packed<M, N> pack(Frag (&f)[M][N], bool act = false) {
    Packed<M, N> p;
#pragma unroll

#pragma unroll
    for (int m = 0; m < M; m++) {
#pragma unroll

#pragma unroll
        for (int n = 0; n < N; n++) {
            u32 x = f[m][n].x, y = f[m][n].y;
            if (act) {
                x = bits2(activation2(h2(x)));
                y = bits2(activation2(h2(y)));
            }
            p.v[m][n] = encode4(x, y);
        }
    }
    return p;
}
// Packet [head/K32][M16][lane][A word]. Same 64*32*H bytes.
// A words are the exact routed MMA bytes, not row-major values reinterpreted.
template <int H> __device__ __forceinline__ void shared_a(const u8 *s, u32 *a, int m, int kp) {
    unsigned addr = __cvta_generic_to_shared(s + kp * 2048 + m * 512 + threadIdx.x * 16);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}
__device__ __forceinline__ int qorder(int x) {
    return (x & 17) | ((x & 2) << 2) | ((x & 12) >> 1);
}
template <int H> __device__ __forceinline__ int pvorder(int x) {
    if constexpr (H == 4)
        return (x & 32) | ((x & 1) << 1) | ((x & 2) << 3) | ((x & 4)) | ((x & 8) >> 3) |
               ((x & 16) >> 1);
    else
        return (x & 33) | ((x & 2) << 3) | ((x & 4) >> 1) | (x & 8) | ((x & 16) >> 2);
}
template <int H> __device__ __forceinline__ int sumorder(int x) {
    if constexpr (H == 4)
        return ((x & 1) << 4) | ((x & 2) << 2) | ((x & 4) << 3) | ((x & 8) >> 1) | ((x & 16) >> 4) |
               ((x & 32) >> 4);
    else
        return ((x & 1) << 4) | ((x & 2) << 1) | ((x & 4) << 3) | ((x & 8) >> 2) | ((x & 16) >> 1) |
               ((x & 32) >> 5);
}
template <int Mode, int H = 4, int M, int N>
__device__ __forceinline__ void reg_a(const Packed<M, N> &p, u32 *a, int m, int kp) {
    int l = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        u32 v = 0;
        int row = m * 16 + l / 4 + (r & 1) * 8;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int k = kp * 32 + (r / 2) * 16 + (l & 3) * 4 + b;
            if constexpr (Mode == 1)
                k = route(k);
            if constexpr (Mode == 2)
                k = qorder(k);
            if constexpr (Mode == 3)
                k = pvorder<H>(k);
            if constexpr (Mode == 3 && H == 8)
                v |= u32(p.pv8(row, k)) << (b * 8);
            else
                v |= u32(p.get(row, k)) << (b * 8);
        }
        a[r] = v;
    }
}
__device__ __forceinline__ void weighted(Frag &c, const u32 *a, const u8 *w, int off, int n) {
    const u32 *b = (const u32 *)(w + off + (n / 2) * 512 + threadIdx.x * 16 + (n & 1) * 8);
    mma(c, a, b[0], b[1]);
}
// A B128 lives outside the entire owned M slab; both N8 halves consume it.
template <int M, int First = 0>
__device__ __forceinline__ void weighted_shared(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                                int kp) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            u32 a[4];
            shared_a<1>(s, a, m, kp);
            mma(c[m][pair * 2], a, b.x, b.y);
            mma(c[m][pair * 2 + 1], a, b.z, b.w);
        });
    });
}
__device__ __forceinline__ void weighted_register(Frag (&c)[4][4], const Words<16> &a, const u8 *w,
                                                  int off) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma(c[m][pair * 2], aa, b.x, b.y);
            mma(c[m][pair * 2 + 1], aa, b.z, b.w);
        });
    });
}
__device__ __forceinline__ u8 packet_byte(const u8 *s, int row, int col) {
    int k = (col & ~14) | ((col & 8) >> 2) | ((col & 2) << 1) | ((col & 4) << 1);
    return s[(k / 32) * 2048 + (row / 16) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 +
             ((row & 8) ? 4 : 0) + ((k & 16) ? 8 : 0) + (k & 3)];
}
// Exact inverse of the existing encoded FF packet publication.
template <int M = 4, int First = 0> __device__ __forceinline__ Packed<> read_encoded(const u8 *s) {
    Packed<> p = {};
    unsigned base = __cvta_generic_to_shared(s + threadIdx.y * 2048 + threadIdx.x * 16);
    each<M>([&](auto mt) {
        constexpr int m = First + decltype(mt)::value;
        uint4 a;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4+%5];"
                     : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                     : "r"(base), "n"(m * 512)
                     : "memory");
        p.v[m][0] = __byte_perm(a.x, a.y, 0x5410);
        p.v[m][1] = __byte_perm(a.x, a.y, 0x7632);
        p.v[m][2] = __byte_perm(a.z, a.w, 0x5410);
        p.v[m][3] = __byte_perm(a.z, a.w, 0x7632);
    });
    return p;
}
template <int H>
__device__ __forceinline__ void exchange(const Packed<> &p, u8 *s, int rowbase = 0, int count = 4) {
    Words<16> x;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            x.v[m * 4 + n] = p.v[m][n];
        });
    });
    auto a = hidden_a(x);
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        if (m < count) {
            unsigned addr = __cvta_generic_to_shared(s + threadIdx.y * 2048 +
                                                     (rowbase / 16 + m) * 512 + threadIdx.x * 16);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a.v[m * 4]),
                         "r"(a.v[m * 4 + 1]), "r"(a.v[m * 4 + 2]), "r"(a.v[m * 4 + 3])
                         : "memory");
        }
    });
}
template <int H, int M = 4, int First = 0>
__device__ __forceinline__ void initial(Frag (&c)[4][4], const Packed<> &p, const u8 *w, int off) {
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        int col = threadIdx.y * 32 + n * 8 + (threadIdx.x & 3) * 2;
        RawHalf2 gate{*(const u32 *)(w + off + col * 2)};
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            E4x4 packed{p.v[m][n]};
            half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)packed.bits, __NV_E4M3));
            half2 hi =
                __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)(packed.bits >> 16), __NV_E4M3));
            c[m][n] = {bits2(__hmul2(lo, h2(gate.bits))), bits2(__hmul2(hi, h2(gate.bits)))};
        });
    });
}
template <int M> __device__ __forceinline__ Packed<M, 4> normpack(Frag (&z)[M][4], half scale) {

#pragma unroll
    for (int m = 0; m < M; m++)
        for (int row = 0; row < 2; row++) {
            u32 a = row ? z[m][0].y : z[m][0].x, b = row ? z[m][1].y : z[m][1].x;
            u32 c = row ? z[m][2].y : z[m][2].x, d = row ? z[m][3].y : z[m][3].x;
            u32 sum = bits2(__hadd2(__hfma2(h2(a), h2(a), __hmul2(h2(c), h2(c))),
                                    __hfma2(h2(b), h2(b), __hmul2(h2(d), h2(d)))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 2))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 1))));
            float x = __half2float(
                      __hmax(__hadd(hh(sum), hh(sum >> 16)), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(x));
            half h = __float2half_rn(inv);

#pragma unroll
            for (int n = 0; n < 4; n++) {
                u32 &v = row ? z[m][n].y : z[m][n].x;
                v = bits2(
                    __hmul2(__hmul2(h2(v), __halves2half2(h, h)), __halves2half2(scale, scale)));
            }
        }
    return pack(z);
}
// Included after the unchanged wide arithmetic and packing primitives.
template <int M, int N> __device__ __forceinline__ Words<M * N> words(const Packed<M, N> &p) {
    Words<M * N> x;
    each<M>([&](auto mt) {
        each<N>([&](auto nt) {
            constexpr int m = decltype(mt)::value, n = decltype(nt)::value;
            x.v[m * N + n] = p.v[m][n];
        });
    });
    return x;
}
template <int H, int M>
__device__ __forceinline__ Packed<M, 8> physicalprobability(Frag (&c)[M][8]) {
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<2>([&](auto rt) {
                constexpr int row = decltype(rt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                half2 z = __hfma2(h2(v), constant2(0.044921875f), constant2(1.30078125f));
                z = __hmin2(__hmax2(z, constant2(1.03125f)), constant2(1.5693359375f));
                v = ((bits2(z) << 5) & 0xffe0ffe0u) ^ 0x80008000u;
            });
        });
        each<2>([&](auto rt) {
            constexpr int row = decltype(rt)::value;
            Words<8> d;
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                d.v[n] = row ? c[m][n].y : c[m][n].x;
            });
            if constexpr (H == 4)
                d = den4_physical_paired(d);
            else
                d = den8_physical_paired(d);
            u32 group;
            each<4>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                u32 pair = bits2(__hadd2(h2(d.v[p]), h2(d.v[4 + p])));
                if constexpr (p == 0)
                    group = pair;
                else
                    group = bits2(__hadd2(h2(group), h2(pair)));
            });
            u32 total;
            each<4>([&](auto gt) {
                constexpr int g = decltype(gt)::value;
                u32 term = __shfl_sync(0xffffffff, group, (threadIdx.x & ~3) + g);
                if constexpr (g == 0)
                    total = term;
                else
                    total = bits2(__hadd2(h2(total), h2(term)));
            });
            float den = __half2float(__hmax(__hadd(hh(total), hh(total >> 16)),
                                            __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half r = __float2half_rn(inv);
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                v = bits2(__hmul2(h2(v), __halves2half2(r, r)));
            });
        });
    });
    return pack(c);
}
template <int H, int M, int First>
__device__ __forceinline__ Packed<> physicalattention(const Words<16> &q, const Words<16> &k,
                                                      const Words<16> &v, const u8 *w) {
    static_assert(M == 2 && (First == 0 || First == 2));
    Words<16> p;
    {
        Frag logits[2][8];
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4] = {q.v[(First + m) * 4], q.v[(First + m) * 4 + 1], q.v[(First + m) * 4 + 2],
                        q.v[(First + m) * 4 + 3]};
            each<4>([&](auto pt) {
                constexpr int packetlocal = decltype(pt)::value;
                constexpr int n = (packetlocal & 1) + 4 * (packetlocal >> 1),
                              packet = 8 * (First / 2) + 4 * m + packetlocal;
                // One raw aligned packet directly occupies its two future QK C pairs.
                asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(logits[m][n].x), "=r"(logits[m][n].y), "=r"(logits[m][n + 2].x),
                               "=r"(logits[m][n + 2].y)
                             : "l"(w + (H == 4 ? 0x24120 : 0x88220) + 8192 * threadIdx.y +
                                   512 * packet + 16 * threadIdx.x));
                mma(logits[m][n], a, k.v[n * 2], k.v[n * 2 + 1]);
                mma(logits[m][n + 2], a, k.v[(n + 2) * 2], k.v[(n + 2) * 2 + 1]);
            });
        });
        if constexpr (H == 4)
            p = prob4_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
        else
            p = prob8_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
    }
    Frag result[4][4] = {};
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            constexpr int off = (kp * M + m) * 4;
            u32 a[4] = {p.v[off], p.v[off + 1], p.v[off + 2], p.v[off + 3]};
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                mma(result[m][n], a, v.v[(kp * 4 + n) * 2], v.v[(kp * 4 + n) * 2 + 1]);
            });
        });
    });
    return pack(result);
}

// By-value ABI: decoded shape/grid/shift and symbol-derived input/output flags.
struct Layout {
    int height, width, grid_x, shift_x, shift_y, flags;
};
template <int H, bool Output>
__device__ __forceinline__ int address(Layout p, int tile, int slot, int n) {
    int x = (tile % p.grid_x) * 8 + (H == 4 ? ((slot >> 1) & 7) : (slot & 7)) + p.shift_x;
    int y = (tile / p.grid_x) * 8 + (H == 4 ? (((slot >> 4) & 3) * 2 + (slot & 1)) : (slot / 8)) +
            p.shift_y;
    if (x < 0 || x >= p.width || y < 0 || y >= p.height)
        return -1;
    if (p.flags & (Output ? 2 : 1)) {
        int lo = (n & 1) | ((n & 6) << 1) | ((n & 8) >> 2);
        return (y * p.width + x) * 16 + (n / 16) * p.height * p.width * 16 + lo;
    }
    int nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4);
    int st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3);
    if constexpr (H == 4)
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + (x / 4 + (p.width / 4) * (y / 4)) * 2048;
    else {
        st |= (x & 4) << 2;
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + ((st & 16) << 8) +
               (x / 8 + ((p.width + 7) / 8) * (y / 4)) * 8192;
    }
}
// Original4 chained: expand K4 unrolled (0x1000..); tail loop 0x2c10..2d80;
// QKV M64N96 K4 loop 0x33d0..3760; projection M32 K4 unrolled 0x66c0...
// A is loaded once per owned M16/K32, outside all component/pair consumers.
template <int M, int First = 0>
__device__ __forceinline__ void four_step(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                          int kp) {
    u32 a[M][4];
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<4>(s, a[m], First + m, kp);
    });
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            mma(c[First + m][pair * 2], a[m], b.x, b.y);
            mma(c[First + m][pair * 2 + 1], a[m], b.z, b.w);
        });
    });
}
template <int M, int First = 0>
__device__ __forceinline__ void four_unrolled(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                              int stride) {
    each<4>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        four_step<M, First>(c, s, w, off + kp * stride, kp);
    });
}
__device__ __forceinline__ void four_tail(Frag (&c)[4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++)
        four_step<4>(c, s, w, off + kp * 4096, kp);
}
__device__ __forceinline__ void four_qkv(Frag (&c)[3][4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}
__device__ __forceinline__ void four_qkv_prefix(Frag (&c)[3][4][4], const u8 *s, const u8 *w,
                                                int off) {
#pragma unroll 1
    for (int kp = 0; kp < 3; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}

// hidden_a swaps unit bit1 with register bit0 only: each M16 is closed.
// Both expand and reduce B128 pairs are loaded once and reused across M64.
template <int H>
__device__ __forceinline__ void finish_hidden(Frag (&expanded)[4][4], Frag (&reduced)[4][4],
                                              const u8 *s, const u8 *w, int off, int stride,
                                              int reduceoff) {
    uint4 eb[2], rb[2];
    each<2>([&](auto pt) {
        constexpr int p = decltype(pt)::value;
        eb[p] = *(const uint4 *)(w + off + (H - 1) * stride + p * 512 + threadIdx.x * 16);
        rb[p] = *(const uint4 *)(w + reduceoff + p * 512 + threadIdx.x * 16);
    });
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        A128 a;
        shared_a<H>(s, a.v, m, H - 1);
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(expanded[m][p * 2], a.v, eb[p].x, eb[p].y);
            mma(expanded[m][p * 2 + 1], a.v, eb[p].z, eb[p].w);
        });
        Words<4> e;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            RawHalf2 x{bits2(activation2(h2(expanded[m][n].x)))},
                y{bits2(activation2(h2(expanded[m][n].y)))};
            e.v[n] = encode4(x.bits, y.bits);
        });
        auto packet = hidden_slab_a(e);
        A128 active{{packet.v[0], packet.v[1], packet.v[2], packet.v[3]}};
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(reduced[m][p * 2], active.v, rb[p].x, rb[p].y);
            mma(reduced[m][p * 2 + 1], active.v, rb[p].z, rb[p].w);
        });
    });
}
// Finish one component at a time; Q/K normalization consumes a closed M16.
// Q/K physical routes cross canonical M16: route only at this component tail.
// V's complete physical transpose waits for all 64 rows. Attention is outside
// this function, after the complete K/V packets have been produced.
template <int H>
__device__ __forceinline__ void finish_qkv(Frag (&c)[3][4][4], Words<16> &q, Words<16> &k,
                                           Words<16> &v, const u8 *s, const u8 *w, int off,
                                           int stride, half scale) {
    A128 a[4];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<H>(s, a[m].v, m, H - 1);
    });
    each<3>([&](auto ct) {
        constexpr int component = decltype(ct)::value;
        uint4 b[2];
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            b[p] = *(const uint4 *)(w + off + (H - 1) * stride + component * 1024 + p * 512 +
                                    threadIdx.x * 16);
        });
        Words<16> encoded;
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<2>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                mma(c[component][m][p * 2], a[m].v, b[p].x, b[p].y);
                mma(c[component][m][p * 2 + 1], a[m].v, b[p].z, b[p].w);
            });
            Frag row[1][4];
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                row[0][n] = c[component][m][n];
            });
            Packed<1, 4> packed;
            if constexpr (component < 2)
                packed = normpack(row, component == 0 ? scale : __float2half(1));
            else
                packed = pack(row);
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                encoded.v[m * 4 + n] = packed.v[0][n];
            });
        });
        if constexpr (component == 0) {
            if constexpr (H == 4)
                q = query4_physical_a(encoded);
            else
                q = query8_physical_a(encoded);
        } else if constexpr (component == 1) {
            if constexpr (H == 4)
                k = key4_physical_b(encoded);
            else
                k = key8_physical_b(encoded);
        } else if constexpr (H == 4)
            v = value4_b(encoded);
        else
            v = value8_b(encoded);
    });
}
// M32 encoded C -> raw output packets. No shared scratch or dynamic word indexing.
template <int First, int Role>
__device__ __forceinline__ void publish(u8 *resource, int output, Layout layout, int tile,
                                        Frag (&out)[4][4]) {
    int l = threadIdx.x, warp = threadIdx.y;
    Words<8> x;
    each<8>([&](auto it) {
        constexpr int i = decltype(it)::value;
        x.v[i] = encode4(out[First + i / 4][i % 4].x, out[First + i / 4][i % 4].y);
    });
    if constexpr (Role == 4) {
        // source bits [byte2,lane5,word3] -> destination [0,3,7,8,4,5,6,1,9,2].
        // All source M values participate before any destination-dependent selection.
        x = publish_output(x);
        int row = First * 16 + l / 4 + (l & 1) * 16 + ((l >> 1) & 1) * 8;
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            int idx = address<4, true>(layout, tile, row, warp * 32 + p * 16);
            if (idx >= 0)
                *(uint4 *)(resource + output + idx) = {x.v[p * 4], x.v[p * 4 + 1], x.v[p * 4 + 2],
                                                       x.v[p * 4 + 3]};
        });
    } else {
        // Adapter enforces the fully proved real payload domain: H%4==0, W%4==0,
        // shiftY=0/-4. Each valid base implies all 16 bytes valid in this CTA/slab.
        each<2>([&](auto jt) {
            constexpr int j2 = decltype(jt)::value;
            int row = First * 16 + l / 4 + j2 * 8;
            int idx = address<4, true>(layout, tile, row, warp * 32 + (l & 3) * 2);
            if (idx >= 0) {
                constexpr int mask = j2 ? 0x7632 : 0x5410;
                uint4 v = {__byte_perm(x.v[0], x.v[1], mask), __byte_perm(x.v[4], x.v[5], mask),
                           __byte_perm(x.v[2], x.v[3], mask), __byte_perm(x.v[6], x.v[7], mask)};
                *(uint4 *)(resource + output + idx) = v;
            }
        });
    }
}
#include "routes4.cuh"
#include "addresses.cuh"
// Only final encoded pool A crosses warps, after both live body slabs die.
template <int H> __device__ __forceinline__ void pool_packet(u8 *r, u8 *s, physical::Cross x) {
    constexpr int N = H * 64;
    int l = threadIdx.x, warp = threadIdx.y;
    u8 *published = s + H * 512;
#pragma unroll 1
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {};
#pragma unroll 1
        for (int kp = 0; kp < H; kp++) {
            uint4 a;
            unsigned addr = __cvta_generic_to_shared(s + kp * 512 + l * 16);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                         : "r"(addr));
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + l * 16);
                physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            });
        }
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                int row = l / 4 + j / 2 * 8, col = column + n * 8 + (l & 3) * 2 + (j & 1);
                published[row * N + col] = enc(physical::component(c[n], j));
            });
        });
    }
    __syncthreads();
    for (int packet = warp * 32 + l; packet < 16 * N / 16; packet += H * 32) {
        int row = packet / (N / 16), group = packet % (N / 16),
            pixel = transition_packet::pixel(H, blockIdx.x, row);
        if (pixel >= 0) {
            u32 v[4] = {};
            each<4>([&](auto wt) {
                constexpr int w = decltype(wt)::value;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    constexpr int raw = w * 4 + b,
                                  c = (raw & 1) | ((raw & 12) >> 1) | ((raw & 2) << 2);
                    v[w] |= u32(published[row * N + group * 16 + c]) << (8 * b);
                });
            });
            *(uint4 *)(r + x.pool_out + pixel * 16 + group * transition_packet::count(H) * 16) =
                make_uint4(v[0], v[1], v[2], v[3]);
        }
    }
}
// Local projection rows are a permutation/subset of the original global rows.
// No inverse device read and no canonical Half reload. Canonical stores remain.
template <int H>
__device__ __forceinline__ void project_input(u8 *r, const u8 *w, Layout layout, int input,
                                              physical::Cross x, const int *ui, u8 *s) {
    constexpr int K = H * 64, N = H * 32;
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    // Each owned physical planar packet is loaded once by this CTA, then routed to MMA A.
    for (int packet = warp * 32 + l; packet < 16 * (K / 16); packet += H * 32) {
        int local = packet / (K / 16), group = packet % (K / 16), row = up_global[tile][local];
        uint4 raw = {};
        if (row >= 0)
            raw = *(const uint4 *)(r + input + row * 16 + group * transition_packet::count(H) * 16);
        int base = (group / 2) * 512 + (local % 8) * 64 + (local / 8) * 4 + (group % 2) * 8;
        *(u32 *)(s + base) = raw.x;
        *(u32 *)(s + base + 16) = raw.y;
        *(u32 *)(s + base + 32) = raw.z;
        *(u32 *)(s + base + 48) = raw.w;
    }
    __syncthreads();
    uint2 c[4] = {};
#pragma unroll 1
    for (int kp = 0; kp < H * 2; kp++) {
        uint4 a = *(const uint4 *)(s + kp * 512 + l * 16);
        each<2>([&](auto pt) {
            constexpr int pair = decltype(pt)::value;
            uint4 b = *(const uint4 *)(w + (H == 4 ? 0x18000 : 0x58000) + kp * N * 32 +
                                       warp * 1024 + pair * 512 + l * 16);
            physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        });
    }
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        each<4>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int row = up_global[tile][l / 4 + j / 2 * 8],
                col = warp * 32 + n * 8 + (l & 3) * 2 + (j & 1);
            if (row >= 0)
                ((half *)x.projection)[row * N + col] = physical::component(c[n], j);
        });
    });
    __syncthreads(); // all projection readers finish before gate input overwrites A
    int cid = up_ids[tile];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4] = {};
        each<4>([&](auto wt) {
            constexpr int word = decltype(wt)::value;
            int token = m * 16 + l / 4 + (word & 1) * 8, rr = up_local[cid][token];
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                int col = route((word / 2) * 16 + (l & 3) * 4 + pair * 2),
                    src = (rr & 7) * 4 + (col & 7) / 2;
                constexpr int n = 2 * (word / 2) + pair;
                // Every source register index is compile-time fixed before the shuffle.
                u32 lo = __shfl_sync(0xffffffff, c[n].x, src),
                    hi = __shfl_sync(0xffffffff, c[n].y, src);
                u32 value = (rr & 8) ? hi : lo;
                each<2>([&](auto ht) {
                    constexpr int h = decltype(ht)::value;
                    int channel = warp * 32 + col + h,
                        raw = address<H, false>(layout, tile, token, channel);
                    if (raw >= 0 && rr >= 0)
                        a[word] |= u32(enc(__hfma(dec(r[x.skip + raw]), x.gate[channel],
                                                  hh(value >> (h * 16)))))
                                   << ((pair * 2 + h) * 8);
                });
            });
        });
        *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = make_uint4(a[0], a[1], a[2], a[3]);
    });
}
template <int Role>
__device__ __forceinline__ void four_body(u8 *resource, const u8 *w, Layout layout, int input,
                                          int output, int counter, int up, const u8 *inputResource,
                                          physical::Cross cross, const int *ui) {
    constexpr int H = 4, C = 128;
    extern __shared__ u8 s[];
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    u32 poolA[4] = {0, 0, 0, 0};
    if constexpr (Role == 3)
        project_input<H>(resource, w, layout, input, cross, ui, s);
    else {
        if constexpr (Role == 1) {
            each<4>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                uint4 a = {0, 0, 0, 0};
                u32 *av = (u32 *)&a;
                each<4>([&](auto rt) {
                    constexpr int word = decltype(rt)::value;
                    int row = m * 16 + l / 4 + (word & 1) * 8,
                        k = warp * 32 + (word / 2) * 16 + (l & 3) * 4;
                    int idx = address<H, false>(layout, tile, row, route(k));
                    av[word] = idx < 0 ? 0 : *(const u32 *)(inputResource + input + idx);
                });
                *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = a;
            });
        } else {
            each<2>([&](auto qt) {
                constexpr int q = decltype(qt)::value;
                int base0 =
                    address<H, false>(layout, tile, 32 * q + l / 4, route(32 * warp + 4 * (l & 3)));
                int base1 = address<H, false>(layout, tile, 32 * q + l / 4 + 8,
                                              route(32 * warp + 4 * (l & 3)));
                uint4 v0 = {0, 0, 0, 0}, v1 = {0, 0, 0, 0};
                if (base0 >= 0)
                    v0 = *(const uint4 *)(inputResource + input + base0);
                if (base1 >= 0)
                    v1 = *(const uint4 *)(inputResource + input + base1);
                *(uint4 *)(s + warp * 2048 + (2 * q) * 512 + l * 16) =
                    make_uint4(v0.x, v1.x, v0.z, v1.z);
                *(uint4 *)(s + warp * 2048 + (2 * q + 1) * 512 + l * 16) =
                    make_uint4(v0.y, v1.y, v0.w, v1.w);
            });
        }
    }
    __syncthreads();
    auto original = read_encoded<4, 0>(s);
    Frag reduced[4][4] = {};
#pragma unroll 1
    for (int hidden = 0; hidden < 4; hidden++) {
        Frag expanded[4][4] = {};
        each<3>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            four_step<4>(expanded, s, w, warp * C * 128 + hidden * 1024 + kp * 4096, kp);
        });
        finish_hidden<4>(expanded, reduced, s, w, warp * C * 128 + hidden * 1024, 4096,
                         0x10000 + warp * 4096 + hidden * 1024);
    }
    auto mid = pack(reduced);
    __syncthreads();
    exchange<H>(mid, s);
    __syncthreads();
    Frag ff[4][4];
    initial<H>(ff, original, w, Role == 3 ? 0x20000 : 0x18010);
    four_tail(ff, s, w, 0x14000 + warp * 1024);
    const u8 *aw = w + (Role == 3 ? 0x80e0 : 0);
    {
        auto feature = pack(ff);
        __syncthreads();
        exchange<H>(feature, s);
    }
    __syncthreads();
    Words<16> q, k, v;
    {
        Frag qkv[3][4][4] = {};
        four_qkv_prefix(qkv, s, aw, 0x18120 + warp * 3072);
        finish_qkv<4>(qkv, q, k, v, s, aw, 0x18120 + warp * 3072, 12288,
                      __float2half_rn(((const float *)(aw + 0x2c120))[warp]));
    }
    __syncthreads();
    each<2>([&](auto ft) {
        constexpr int first = decltype(ft)::value * 2;
        auto pv = physicalattention<H, 2, first>(q, k, v, aw);
        // FF stays in its original 4096-byte M32 slab until every warp's late read.
        auto feature = read_encoded<2, first>(s);
        __syncthreads();
        exchange<H>(pv, s, first * 16, 2);
        __syncthreads();
        Frag out[4][4];
        initial<H, 2, first>(out, feature, aw, 0x30130);
        four_unrolled<2, first>(out, s, aw, 0x2c130 + warp * 1024, 4096);
        if constexpr (Role == 2)
            collect_pool<first>(out, poolA);
        publish<first, Role>(resource, output, layout, tile, out);
        // All projection warps have consumed PV before the next slab is overwritten.
        __syncthreads();
    });
    if constexpr (Role == 2) {
        *(uint4 *)(s + warp * 512 + l * 16) = make_uint4(poolA[0], poolA[1], poolA[2], poolA[3]);
        __syncthreads();
        pool_packet<H>(resource, s, cross);
    }
    __threadfence();
    if (counter >= 0 && l == 0 && warp == 0)
        ((int *)(resource + counter))[tile] = 0;
}
extern "C" __global__ void packet4_ds(u8 *r, const u8 *w, Layout layout, int in, int out, int cb,
                                      int up, const u8 *src, physical::Cross cross, const int *ui) {
    four_body<2>(r, w, layout, in, out, cb, up, src, cross, ui);
}

} // namespace nr_outer_ds4_6
// ============================================================================
// OUTER wide_transition_packet/up4.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_up4_7 {
// Organized full 4H/8H: accepted uniform-bank math; real dispatch metadata.

namespace wide_activation {
using u32 = unsigned int;
__device__ __forceinline__ int lane() {
    return threadIdx.x;
}
template <int I> struct Tag {
    static constexpr int value = I;
};
template <int N, int I = 0, class F> __device__ __forceinline__ void each(F f) {
    if constexpr (I < N) {
        f(Tag<I>{});
        each<N, I + 1>(f);
    }
}
template <int N> struct Words {
    u32 v[N];
};
__device__ __forceinline__ Words<16> hidden_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value4_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 8);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 8);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 8);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 8);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 8);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 8);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 8);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 8);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 8);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 8);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 8);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 8);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 8);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 8);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 8);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 8) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 8);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 8) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 4)) & 1) * 17));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 1) ^ (lane() >> 2)) & 1) * 6));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_a(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 1);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 1);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 1);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 1);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 1);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 1);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 1);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 1) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 1);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 1) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> value8_b(Words<16> x) {
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 4);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 4);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 4);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 4);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 4);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 4);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 4);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 4);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[8], 4);
        y.v[8] = __byte_perm(x.v[8], t8, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[9], 4);
        y.v[9] = __byte_perm(x.v[9], t9, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[10], 4);
        y.v[10] = __byte_perm(x.v[10], t10, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[11], 4);
        y.v[11] = __byte_perm(x.v[11], t11, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[12], 4);
        y.v[12] = __byte_perm(x.v[12], t12, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[13], 4);
        y.v[13] = __byte_perm(x.v[13], t13, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[14], 4);
        y.v[14] = __byte_perm(x.v[14], t14, ((lane() & 4) ? 0x3715 : 0x6240));
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[15], 4);
        y.v[15] = __byte_perm(x.v[15], t15, ((lane() & 4) ? 0x3715 : 0x6240));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[12], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[13], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[14], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[15], 0x5410);
        y.v[12] = __byte_perm(x.v[12], x.v[8], 0x3276);
        y.v[13] = __byte_perm(x.v[13], x.v[9], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[10], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[11], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 0) ^ (lane() >> 3)) & 1) * 9));
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[3] = (((lane() >> 1) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[4] = (((lane() >> 1) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[9] = (((lane() >> 1) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[11] = (((lane() >> 1) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[12] = (((lane() >> 1) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[14] = (((lane() >> 1) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[4] = (((lane() >> 4) & 1) == 1) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[6] = (((lane() >> 4) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[9] = (((lane() >> 4) & 1) == 0) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[11] = (((lane() >> 4) & 1) == 0) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[9];
        y.v[10] = x.v[12];
        y.v[11] = x.v[13];
        y.v[12] = x.v[10];
        y.v[13] = x.v[11];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<32> prob8_a(Words<32> x) {
    {
        Words<32> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        y.v[16] = __byte_perm(x.v[16], x.v[18], 0x5410);
        y.v[17] = __byte_perm(x.v[17], x.v[19], 0x5410);
        y.v[18] = __byte_perm(x.v[18], x.v[16], 0x3276);
        y.v[19] = __byte_perm(x.v[19], x.v[17], 0x3276);
        y.v[20] = __byte_perm(x.v[20], x.v[22], 0x5410);
        y.v[21] = __byte_perm(x.v[21], x.v[23], 0x5410);
        y.v[22] = __byte_perm(x.v[22], x.v[20], 0x3276);
        y.v[23] = __byte_perm(x.v[23], x.v[21], 0x3276);
        y.v[24] = __byte_perm(x.v[24], x.v[26], 0x5410);
        y.v[25] = __byte_perm(x.v[25], x.v[27], 0x5410);
        y.v[26] = __byte_perm(x.v[26], x.v[24], 0x3276);
        y.v[27] = __byte_perm(x.v[27], x.v[25], 0x3276);
        y.v[28] = __byte_perm(x.v[28], x.v[30], 0x5410);
        y.v[29] = __byte_perm(x.v[29], x.v[31], 0x5410);
        y.v[30] = __byte_perm(x.v[30], x.v[28], 0x3276);
        y.v[31] = __byte_perm(x.v[31], x.v[29], 0x3276);
        x = y;
    }
    {
        Words<32> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        u32 t16 = __shfl_xor_sync(0xffffffff, x.v[17], 2);
        y.v[16] = (((lane() >> 1) & 1) == 0) ? x.v[16] : t16;
        u32 t17 = __shfl_xor_sync(0xffffffff, x.v[16], 2);
        y.v[17] = (((lane() >> 1) & 1) == 1) ? x.v[17] : t17;
        u32 t18 = __shfl_xor_sync(0xffffffff, x.v[19], 2);
        y.v[18] = (((lane() >> 1) & 1) == 0) ? x.v[18] : t18;
        u32 t19 = __shfl_xor_sync(0xffffffff, x.v[18], 2);
        y.v[19] = (((lane() >> 1) & 1) == 1) ? x.v[19] : t19;
        u32 t20 = __shfl_xor_sync(0xffffffff, x.v[21], 2);
        y.v[20] = (((lane() >> 1) & 1) == 0) ? x.v[20] : t20;
        u32 t21 = __shfl_xor_sync(0xffffffff, x.v[20], 2);
        y.v[21] = (((lane() >> 1) & 1) == 1) ? x.v[21] : t21;
        u32 t22 = __shfl_xor_sync(0xffffffff, x.v[23], 2);
        y.v[22] = (((lane() >> 1) & 1) == 0) ? x.v[22] : t22;
        u32 t23 = __shfl_xor_sync(0xffffffff, x.v[22], 2);
        y.v[23] = (((lane() >> 1) & 1) == 1) ? x.v[23] : t23;
        u32 t24 = __shfl_xor_sync(0xffffffff, x.v[25], 2);
        y.v[24] = (((lane() >> 1) & 1) == 0) ? x.v[24] : t24;
        u32 t25 = __shfl_xor_sync(0xffffffff, x.v[24], 2);
        y.v[25] = (((lane() >> 1) & 1) == 1) ? x.v[25] : t25;
        u32 t26 = __shfl_xor_sync(0xffffffff, x.v[27], 2);
        y.v[26] = (((lane() >> 1) & 1) == 0) ? x.v[26] : t26;
        u32 t27 = __shfl_xor_sync(0xffffffff, x.v[26], 2);
        y.v[27] = (((lane() >> 1) & 1) == 1) ? x.v[27] : t27;
        u32 t28 = __shfl_xor_sync(0xffffffff, x.v[29], 2);
        y.v[28] = (((lane() >> 1) & 1) == 0) ? x.v[28] : t28;
        u32 t29 = __shfl_xor_sync(0xffffffff, x.v[28], 2);
        y.v[29] = (((lane() >> 1) & 1) == 1) ? x.v[29] : t29;
        u32 t30 = __shfl_xor_sync(0xffffffff, x.v[31], 2);
        y.v[30] = (((lane() >> 1) & 1) == 0) ? x.v[30] : t30;
        u32 t31 = __shfl_xor_sync(0xffffffff, x.v[30], 2);
        y.v[31] = (((lane() >> 1) & 1) == 1) ? x.v[31] : t31;
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[18];
        y.v[18] = x.v[17];
        y.v[19] = x.v[19];
        y.v[20] = x.v[20];
        y.v[21] = x.v[22];
        y.v[22] = x.v[21];
        y.v[23] = x.v[23];
        y.v[24] = x.v[24];
        y.v[25] = x.v[26];
        y.v[26] = x.v[25];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[30];
        y.v[30] = x.v[29];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        y.v[16] = x.v[16];
        y.v[17] = x.v[17];
        y.v[18] = x.v[18];
        y.v[19] = x.v[19];
        y.v[20] = x.v[24];
        y.v[21] = x.v[25];
        y.v[22] = x.v[26];
        y.v[23] = x.v[27];
        y.v[24] = x.v[20];
        y.v[25] = x.v[21];
        y.v[26] = x.v[22];
        y.v[27] = x.v[23];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    {
        Words<32> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        y.v[8] = x.v[16];
        y.v[9] = x.v[17];
        y.v[10] = x.v[18];
        y.v[11] = x.v[19];
        y.v[12] = x.v[20];
        y.v[13] = x.v[21];
        y.v[14] = x.v[22];
        y.v[15] = x.v[23];
        y.v[16] = x.v[8];
        y.v[17] = x.v[9];
        y.v[18] = x.v[10];
        y.v[19] = x.v[11];
        y.v[20] = x.v[12];
        y.v[21] = x.v[13];
        y.v[22] = x.v[14];
        y.v[23] = x.v[15];
        y.v[24] = x.v[24];
        y.v[25] = x.v[25];
        y.v[26] = x.v[26];
        y.v[27] = x.v[27];
        y.v[28] = x.v[28];
        y.v[29] = x.v[29];
        y.v[30] = x.v[30];
        y.v[31] = x.v[31];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_slab_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 2);
        y.v[8] = (((lane() >> 1) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 2);
        y.v[9] = (((lane() >> 1) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 2);
        y.v[10] = (((lane() >> 1) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 2);
        y.v[11] = (((lane() >> 1) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 2);
        y.v[12] = (((lane() >> 1) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 2);
        y.v[13] = (((lane() >> 1) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 2);
        y.v[14] = (((lane() >> 1) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 2);
        y.v[15] = (((lane() >> 1) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<4> hidden_slab_a(Words<4> x) {
    {
        Words<4> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 0) ^ (lane() >> 1)) & 1) * 3));
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[1] = (((lane() >> 1) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[2] = (((lane() >> 1) & 1) == 1) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[5] = (((lane() >> 1) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[6] = (((lane() >> 1) & 1) == 1) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[0] = (((lane() >> 1) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[1] = (((lane() >> 1) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[2] = (((lane() >> 1) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[3] = (((lane() >> 1) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[4] = (((lane() >> 1) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[5] = (((lane() >> 1) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[6] = (((lane() >> 1) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[7] = (((lane() >> 1) & 1) == 1) ? x.v[7] : t7;
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> publish_output(Words<8> x) {
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 2);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 2);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 2);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 2);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 2);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 2);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 2);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 2) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 2);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 2) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[4], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[5], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[6], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[7], 0x5410);
        y.v[4] = __byte_perm(x.v[4], x.v[0], 0x3276);
        y.v[5] = __byte_perm(x.v[5], x.v[1], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[2], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[3], 0x3276);
        x = y;
    }
    {
        Words<8> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[0], 1);
        y.v[0] = __byte_perm(x.v[0], t0, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[1], 1);
        y.v[1] = __byte_perm(x.v[1], t1, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[2], 1);
        y.v[2] = __byte_perm(x.v[2], t2, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[3], 1);
        y.v[3] = __byte_perm(x.v[3], t3, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[4], 1);
        y.v[4] = __byte_perm(x.v[4], t4, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[5], 1);
        y.v[5] = __byte_perm(x.v[5], t5, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[6], 1);
        y.v[6] = __byte_perm(x.v[6], t6, ((lane() & 1) ? 0x3276 : 0x5410));
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[7], 1);
        y.v[7] = __byte_perm(x.v[7], t7, ((lane() & 1) ? 0x3276 : 0x5410));
        x = y;
    }
    {
        Words<8> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        x = y;
    }
    return x;
}
} // namespace wide_activation
namespace wide_activation {
__device__ __forceinline__ Words<16> query4_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key4_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 3)) & 1) * 12));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den4_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob4_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 2) ^ (lane() >> 4)) & 1) * 20));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] =
            __shfl_sync(0xffffffff, x.v[0], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[1] =
            __shfl_sync(0xffffffff, x.v[1], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[2] =
            __shfl_sync(0xffffffff, x.v[2], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[3] =
            __shfl_sync(0xffffffff, x.v[3], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[4] =
            __shfl_sync(0xffffffff, x.v[4], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[5] =
            __shfl_sync(0xffffffff, x.v[5], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[6] =
            __shfl_sync(0xffffffff, x.v[6], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[7] =
            __shfl_sync(0xffffffff, x.v[7], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[8] =
            __shfl_sync(0xffffffff, x.v[8], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[9] =
            __shfl_sync(0xffffffff, x.v[9], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[10] =
            __shfl_sync(0xffffffff, x.v[10], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[11] =
            __shfl_sync(0xffffffff, x.v[11], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[12] =
            __shfl_sync(0xffffffff, x.v[12], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[13] =
            __shfl_sync(0xffffffff, x.v[13], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[14] =
            __shfl_sync(0xffffffff, x.v[14], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        y.v[15] =
            __shfl_sync(0xffffffff, x.v[15], lane() ^ ((((lane() >> 3) ^ (lane() >> 4)) & 1) * 24));
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> query8_physical_a(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[4];
        y.v[2] = x.v[2];
        y.v[3] = x.v[6];
        y.v[4] = x.v[1];
        y.v[5] = x.v[5];
        y.v[6] = x.v[3];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[12];
        y.v[10] = x.v[10];
        y.v[11] = x.v[14];
        y.v[12] = x.v[9];
        y.v[13] = x.v[13];
        y.v[14] = x.v[11];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> key8_physical_b(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[1], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[0], 0x3276);
        y.v[2] = __byte_perm(x.v[2], x.v[3], 0x5410);
        y.v[3] = __byte_perm(x.v[3], x.v[2], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[5], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[4], 0x3276);
        y.v[6] = __byte_perm(x.v[6], x.v[7], 0x5410);
        y.v[7] = __byte_perm(x.v[7], x.v[6], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[9], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[8], 0x3276);
        y.v[10] = __byte_perm(x.v[10], x.v[11], 0x5410);
        y.v[11] = __byte_perm(x.v[11], x.v[10], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[13], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[12], 0x3276);
        y.v[14] = __byte_perm(x.v[14], x.v[15], 0x5410);
        y.v[15] = __byte_perm(x.v[15], x.v[14], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[1] = (((lane() >> 4) & 1) == 1) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[3] = (((lane() >> 4) & 1) == 1) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[5] = (((lane() >> 4) & 1) == 1) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[7] = (((lane() >> 4) & 1) == 1) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[8] = (((lane() >> 4) & 1) == 0) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[10] = (((lane() >> 4) & 1) == 0) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[12] = (((lane() >> 4) & 1) == 0) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[14] = (((lane() >> 4) & 1) == 0) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[2];
        y.v[2] = x.v[1];
        y.v[3] = x.v[3];
        y.v[4] = x.v[4];
        y.v[5] = x.v[6];
        y.v[6] = x.v[5];
        y.v[7] = x.v[7];
        y.v[8] = x.v[8];
        y.v[9] = x.v[10];
        y.v[10] = x.v[9];
        y.v[11] = x.v[11];
        y.v[12] = x.v[12];
        y.v[13] = x.v[14];
        y.v[14] = x.v[13];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<8> den8_physical_paired(Words<8> x) {
    {
        Words<8> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[4];
        y.v[3] = x.v[5];
        y.v[4] = x.v[2];
        y.v[5] = x.v[3];
        y.v[6] = x.v[6];
        y.v[7] = x.v[7];
        x = y;
    }
    return x;
}
__device__ __forceinline__ Words<16> prob8_physical_to_accepted_PV_A(Words<16> x) {
    {
        Words<16> y;
        y.v[0] = __byte_perm(x.v[0], x.v[2], 0x5410);
        y.v[1] = __byte_perm(x.v[1], x.v[3], 0x5410);
        y.v[2] = __byte_perm(x.v[2], x.v[0], 0x3276);
        y.v[3] = __byte_perm(x.v[3], x.v[1], 0x3276);
        y.v[4] = __byte_perm(x.v[4], x.v[6], 0x5410);
        y.v[5] = __byte_perm(x.v[5], x.v[7], 0x5410);
        y.v[6] = __byte_perm(x.v[6], x.v[4], 0x3276);
        y.v[7] = __byte_perm(x.v[7], x.v[5], 0x3276);
        y.v[8] = __byte_perm(x.v[8], x.v[10], 0x5410);
        y.v[9] = __byte_perm(x.v[9], x.v[11], 0x5410);
        y.v[10] = __byte_perm(x.v[10], x.v[8], 0x3276);
        y.v[11] = __byte_perm(x.v[11], x.v[9], 0x3276);
        y.v[12] = __byte_perm(x.v[12], x.v[14], 0x5410);
        y.v[13] = __byte_perm(x.v[13], x.v[15], 0x5410);
        y.v[14] = __byte_perm(x.v[14], x.v[12], 0x3276);
        y.v[15] = __byte_perm(x.v[15], x.v[13], 0x3276);
        x = y;
    }
    {
        Words<16> y;
        u32 t0 = __shfl_xor_sync(0xffffffff, x.v[8], 16);
        y.v[0] = (((lane() >> 4) & 1) == 0) ? x.v[0] : t0;
        u32 t1 = __shfl_xor_sync(0xffffffff, x.v[9], 16);
        y.v[1] = (((lane() >> 4) & 1) == 0) ? x.v[1] : t1;
        u32 t2 = __shfl_xor_sync(0xffffffff, x.v[10], 16);
        y.v[2] = (((lane() >> 4) & 1) == 0) ? x.v[2] : t2;
        u32 t3 = __shfl_xor_sync(0xffffffff, x.v[11], 16);
        y.v[3] = (((lane() >> 4) & 1) == 0) ? x.v[3] : t3;
        u32 t4 = __shfl_xor_sync(0xffffffff, x.v[12], 16);
        y.v[4] = (((lane() >> 4) & 1) == 0) ? x.v[4] : t4;
        u32 t5 = __shfl_xor_sync(0xffffffff, x.v[13], 16);
        y.v[5] = (((lane() >> 4) & 1) == 0) ? x.v[5] : t5;
        u32 t6 = __shfl_xor_sync(0xffffffff, x.v[14], 16);
        y.v[6] = (((lane() >> 4) & 1) == 0) ? x.v[6] : t6;
        u32 t7 = __shfl_xor_sync(0xffffffff, x.v[15], 16);
        y.v[7] = (((lane() >> 4) & 1) == 0) ? x.v[7] : t7;
        u32 t8 = __shfl_xor_sync(0xffffffff, x.v[0], 16);
        y.v[8] = (((lane() >> 4) & 1) == 1) ? x.v[8] : t8;
        u32 t9 = __shfl_xor_sync(0xffffffff, x.v[1], 16);
        y.v[9] = (((lane() >> 4) & 1) == 1) ? x.v[9] : t9;
        u32 t10 = __shfl_xor_sync(0xffffffff, x.v[2], 16);
        y.v[10] = (((lane() >> 4) & 1) == 1) ? x.v[10] : t10;
        u32 t11 = __shfl_xor_sync(0xffffffff, x.v[3], 16);
        y.v[11] = (((lane() >> 4) & 1) == 1) ? x.v[11] : t11;
        u32 t12 = __shfl_xor_sync(0xffffffff, x.v[4], 16);
        y.v[12] = (((lane() >> 4) & 1) == 1) ? x.v[12] : t12;
        u32 t13 = __shfl_xor_sync(0xffffffff, x.v[5], 16);
        y.v[13] = (((lane() >> 4) & 1) == 1) ? x.v[13] : t13;
        u32 t14 = __shfl_xor_sync(0xffffffff, x.v[6], 16);
        y.v[14] = (((lane() >> 4) & 1) == 1) ? x.v[14] : t14;
        u32 t15 = __shfl_xor_sync(0xffffffff, x.v[7], 16);
        y.v[15] = (((lane() >> 4) & 1) == 1) ? x.v[15] : t15;
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[8];
        y.v[2] = x.v[2];
        y.v[3] = x.v[10];
        y.v[4] = x.v[4];
        y.v[5] = x.v[12];
        y.v[6] = x.v[6];
        y.v[7] = x.v[14];
        y.v[8] = x.v[1];
        y.v[9] = x.v[9];
        y.v[10] = x.v[3];
        y.v[11] = x.v[11];
        y.v[12] = x.v[5];
        y.v[13] = x.v[13];
        y.v[14] = x.v[7];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[8];
        y.v[3] = x.v[9];
        y.v[4] = x.v[4];
        y.v[5] = x.v[5];
        y.v[6] = x.v[12];
        y.v[7] = x.v[13];
        y.v[8] = x.v[2];
        y.v[9] = x.v[3];
        y.v[10] = x.v[10];
        y.v[11] = x.v[11];
        y.v[12] = x.v[6];
        y.v[13] = x.v[7];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    {
        Words<16> y;
        y.v[0] = x.v[0];
        y.v[1] = x.v[1];
        y.v[2] = x.v[2];
        y.v[3] = x.v[3];
        y.v[4] = x.v[8];
        y.v[5] = x.v[9];
        y.v[6] = x.v[10];
        y.v[7] = x.v[11];
        y.v[8] = x.v[4];
        y.v[9] = x.v[5];
        y.v[10] = x.v[6];
        y.v[11] = x.v[7];
        y.v[12] = x.v[12];
        y.v[13] = x.v[13];
        y.v[14] = x.v[14];
        y.v[15] = x.v[15];
        x = y;
    }
    return x;
}
} // namespace wide_activation

// Address-only ownership metadata, never reordered weights or activation values.
namespace physical {
struct Cross {
    const int *pool_input, *pool_output, *up_inverse;
    const half *projection, *gate;
    const unsigned char *matrix;
    int pool_out, skip;
};
__device__ __forceinline__ unsigned char encode(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half decode(unsigned char x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ int permute(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ void multiply(uint2 &c, uint4 a, uint2 b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c.x), "+r"(c.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ half component(uint2 c, int j) {
    return __ushort_as_half((j < 2 ? c.x : c.y) >> ((j & 1) * 16));
}
__device__ __forceinline__ unsigned char fused_input(const unsigned char *r, Cross x, int raw) {
    if (raw < 0)
        return 0;
    int2 inverse = ((const int2 *)x.up_inverse)[raw];
    return encode(__hfma(decode(r[x.skip + raw]), x.gate[inverse.y], x.projection[inverse.x]));
}
// Called after ALL projection reads of A finish. pi maps the original pair tree
// to canonical Half positions produced by this CTA, including invalid -1 items.
template <int H> __device__ void pool(unsigned char *r, const half *values, Cross x) {
    constexpr int K = H * 32, N = K * 2;
    int lane = threadIdx.x, warp = threadIdx.y;
    const int *pi = x.pool_input + blockIdx.x * 16 * 4 * K;
    const int *po = x.pool_output + blockIdx.x * 16 * N;
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int kp = 0; kp < H; kp++) {
            uint4 a = {0, 0, 0, 0};
            unsigned *aw = (unsigned *)&a;
#pragma unroll
            for (int word = 0; word < 4; word++) {
                int row = lane / 4 + (word & 1) * 8;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    int k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                    if constexpr (H != 2)
                        k = permute(k);
                    half v[4];
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        int index = pi[(row * 4 + t) * K + k];
                        v[t] = index < 0 ? __float2half(0) : values[index];
                    }
                    aw[word] |=
                        unsigned(encode(__hmul(__hadd(__hadd(v[0], v[1]), __hadd(v[2], v[3])),
                                               __float2half(.25f))))
                        << (b * 8);
                }
            }
#pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + lane * 16);
                multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            }
        }
#pragma unroll
        for (int n = 0; n < 4; n++)
            for (int j = 0; j < 4; j++) {
                int row = lane / 4 + j / 2 * 8, col = column + n * 8 + (lane & 3) * 2 + (j & 1),
                    out = po[row * N + col];
                if (out >= 0)
                    r[x.pool_out + out] = encode(component(c[n], j));
            }
    }
}
template <int H>
__device__ void project(const unsigned char *r, half *dest, const unsigned char *w, const int *im,
                        int input, int rows) {
    constexpr int K = H * 64, N = H * 32;
    int lane = threadIdx.x, warp = threadIdx.y, first = blockIdx.x * 16;
    uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (int kp = 0; kp < K / 32; kp++) {
        uint4 a = {0, 0, 0, 0};
        unsigned *aw = (unsigned *)&a;
#pragma unroll
        for (int word = 0; word < 4; word++)
            for (int b = 0; b < 4; b++) {
                int row = first + lane / 4 + (word & 1) * 8,
                    k = kp * 32 + (word / 2) * 16 + (lane & 3) * 4 + b;
                if constexpr (H != 2)
                    k = permute(k);
                int index = row < rows ? im[row * K + k] : -1;
                aw[word] |= unsigned(index < 0 ? 0 : r[input + index]) << (8 * b);
            }
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            uint4 b = *(const uint4 *)(w + kp * N * 32 + warp * 1024 + pair * 512 + lane * 16);
            multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        }
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int j = 0; j < 4; j++) {
            int row = first + lane / 4 + j / 2 * 8,
                col = warp * 32 + n * 8 + (lane & 3) * 2 + (j & 1);
            if (row < rows)
                dest[row * N + col] = component(c[n], j);
        }
}
} // namespace physical
using namespace wide_activation;
using u32 = unsigned int;
using u8 = unsigned char;
struct RawHalf2 {
    u32 bits;
};
struct E4x4 {
    u32 bits;
};
struct A128 {
    u32 v[4];
};
struct B128 {
    u32 v[4];
};
struct Frag {
    u32 x, y;
};
__device__ __forceinline__ half hh(u32 x) {
    return __ushort_as_half((unsigned short)x);
}
__device__ __forceinline__ u32 hb(half x) {
    return __half_as_ushort(x);
}
__device__ __forceinline__ half dec(u8 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ u8 enc(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half part(Frag x, int i) {
    return hh((i < 2 ? x.x : x.y) >> ((i & 1) * 16));
}
__device__ __forceinline__ void put(Frag &x, int i, half h) {
    u32 &v = i < 2 ? x.x : x.y;
    int s = (i & 1) * 16;
    v = (v & ~(65535u << s)) | (hb(h) << s);
}
__device__ __forceinline__ void mma(Frag &d, const u32 *a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int route(int k) {
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
}
__device__ __forceinline__ half2 h2(u32 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.bits = x;
    return v.pair;
}
__device__ __forceinline__ u32 bits2(half2 x) {
    union {
        u32 bits;
        half2 pair;
    } v;
    v.pair = x;
    return v.bits;
}
__device__ __forceinline__ half2 constant2(float x) {
    return __float2half2_rn(x);
}
__device__ __forceinline__ half2 activation2(half2 x) {
    half2 z = __hmin2(__hmax2(x, constant2(-4)), constant2(4));
    half2 g = __hfma2(__habs2(z), constant2(-0.055908203125f), constant2(0.447265625f));
    return __hmul2(x, __hfma2(z, g, constant2(0.89453125f)));
}
__device__ __forceinline__ u32 encode4(u32 a, u32 b) {
    unsigned short x = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(a), __NV_SATFINITE, __NV_E4M3);
    unsigned short y = __nv_cvt_halfraw2_to_fp8x2((__half2_raw)h2(b), __NV_SATFINITE, __NV_E4M3);
    return u32(x) | (u32(y) << 16);
}
template <int M = 4, int N = 4> struct Packed {
    u32 v[M][N];
    // Call sites have warp-uniform fragment indices; only the byte/lane varies.
    __device__ __forceinline__ u8 get(int row, int col) const {
        u32 result = __shfl_sync(0xffffffff, v[row / 16][col / 8], (row & 7) * 4 + (col & 7) / 2);
        return (result >> (((row & 8) ? 2 : 0) + (col & 1)) * 8) & 255;
    }
    // PV8 A is the sole exception: column bit3 depends on destination lane.
    // Shuffle both fixed registers BEFORE choosing; never gather the full fragment.
    __device__ __forceinline__ u8 pv8(int row, int col) const {
        int n = (col / 8) & ~1, src = (row & 7) * 4 + (col & 7) / 2;
        u32 a = __shfl_sync(0xffffffff, v[row / 16][n], src);
        u32 b = __shfl_sync(0xffffffff, v[row / 16][n + 1], src);
        return (((col & 8) ? b : a) >> ((((row & 8) ? 2 : 0) + (col & 1)) * 8)) & 255;
    }
};
template <int M, int N>
__device__ __forceinline__ Packed<M, N> pack(Frag (&f)[M][N], bool act = false) {
    Packed<M, N> p;
#pragma unroll

#pragma unroll
    for (int m = 0; m < M; m++) {
#pragma unroll

#pragma unroll
        for (int n = 0; n < N; n++) {
            u32 x = f[m][n].x, y = f[m][n].y;
            if (act) {
                x = bits2(activation2(h2(x)));
                y = bits2(activation2(h2(y)));
            }
            p.v[m][n] = encode4(x, y);
        }
    }
    return p;
}
// Packet [head/K32][M16][lane][A word]. Same 64*32*H bytes.
// A words are the exact routed MMA bytes, not row-major values reinterpreted.
template <int H> __device__ __forceinline__ void shared_a(const u8 *s, u32 *a, int m, int kp) {
    unsigned addr = __cvta_generic_to_shared(s + kp * 2048 + m * 512 + threadIdx.x * 16);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr));
}
__device__ __forceinline__ int qorder(int x) {
    return (x & 17) | ((x & 2) << 2) | ((x & 12) >> 1);
}
template <int H> __device__ __forceinline__ int pvorder(int x) {
    if constexpr (H == 4)
        return (x & 32) | ((x & 1) << 1) | ((x & 2) << 3) | ((x & 4)) | ((x & 8) >> 3) |
               ((x & 16) >> 1);
    else
        return (x & 33) | ((x & 2) << 3) | ((x & 4) >> 1) | (x & 8) | ((x & 16) >> 2);
}
template <int H> __device__ __forceinline__ int sumorder(int x) {
    if constexpr (H == 4)
        return ((x & 1) << 4) | ((x & 2) << 2) | ((x & 4) << 3) | ((x & 8) >> 1) | ((x & 16) >> 4) |
               ((x & 32) >> 4);
    else
        return ((x & 1) << 4) | ((x & 2) << 1) | ((x & 4) << 3) | ((x & 8) >> 2) | ((x & 16) >> 1) |
               ((x & 32) >> 5);
}
template <int Mode, int H = 4, int M, int N>
__device__ __forceinline__ void reg_a(const Packed<M, N> &p, u32 *a, int m, int kp) {
    int l = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        u32 v = 0;
        int row = m * 16 + l / 4 + (r & 1) * 8;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int k = kp * 32 + (r / 2) * 16 + (l & 3) * 4 + b;
            if constexpr (Mode == 1)
                k = route(k);
            if constexpr (Mode == 2)
                k = qorder(k);
            if constexpr (Mode == 3)
                k = pvorder<H>(k);
            if constexpr (Mode == 3 && H == 8)
                v |= u32(p.pv8(row, k)) << (b * 8);
            else
                v |= u32(p.get(row, k)) << (b * 8);
        }
        a[r] = v;
    }
}
__device__ __forceinline__ void weighted(Frag &c, const u32 *a, const u8 *w, int off, int n) {
    const u32 *b = (const u32 *)(w + off + (n / 2) * 512 + threadIdx.x * 16 + (n & 1) * 8);
    mma(c, a, b[0], b[1]);
}
// A B128 lives outside the entire owned M slab; both N8 halves consume it.
template <int M, int First = 0>
__device__ __forceinline__ void weighted_shared(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                                int kp) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            u32 a[4];
            shared_a<1>(s, a, m, kp);
            mma(c[m][pair * 2], a, b.x, b.y);
            mma(c[m][pair * 2 + 1], a, b.z, b.w);
        });
    });
}
__device__ __forceinline__ void weighted_register(Frag (&c)[4][4], const Words<16> &a, const u8 *w,
                                                  int off) {
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma(c[m][pair * 2], aa, b.x, b.y);
            mma(c[m][pair * 2 + 1], aa, b.z, b.w);
        });
    });
}
__device__ __forceinline__ u8 packet_byte(const u8 *s, int row, int col) {
    int k = (col & ~14) | ((col & 8) >> 2) | ((col & 2) << 1) | ((col & 4) << 1);
    return s[(k / 32) * 2048 + (row / 16) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 +
             ((row & 8) ? 4 : 0) + ((k & 16) ? 8 : 0) + (k & 3)];
}
// Exact inverse of the existing encoded FF packet publication.
template <int M = 4, int First = 0> __device__ __forceinline__ Packed<> read_encoded(const u8 *s) {
    Packed<> p = {};
    unsigned base = __cvta_generic_to_shared(s + threadIdx.y * 2048 + threadIdx.x * 16);
    each<M>([&](auto mt) {
        constexpr int m = First + decltype(mt)::value;
        uint4 a;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4+%5];"
                     : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                     : "r"(base), "n"(m * 512)
                     : "memory");
        p.v[m][0] = __byte_perm(a.x, a.y, 0x5410);
        p.v[m][1] = __byte_perm(a.x, a.y, 0x7632);
        p.v[m][2] = __byte_perm(a.z, a.w, 0x5410);
        p.v[m][3] = __byte_perm(a.z, a.w, 0x7632);
    });
    return p;
}
template <int H>
__device__ __forceinline__ void exchange(const Packed<> &p, u8 *s, int rowbase = 0, int count = 4) {
    Words<16> x;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            x.v[m * 4 + n] = p.v[m][n];
        });
    });
    auto a = hidden_a(x);
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        if (m < count) {
            unsigned addr = __cvta_generic_to_shared(s + threadIdx.y * 2048 +
                                                     (rowbase / 16 + m) * 512 + threadIdx.x * 16);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a.v[m * 4]),
                         "r"(a.v[m * 4 + 1]), "r"(a.v[m * 4 + 2]), "r"(a.v[m * 4 + 3])
                         : "memory");
        }
    });
}
template <int H, int M = 4, int First = 0>
__device__ __forceinline__ void initial(Frag (&c)[4][4], const Packed<> &p, const u8 *w, int off) {
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        int col = threadIdx.y * 32 + n * 8 + (threadIdx.x & 3) * 2;
        RawHalf2 gate{*(const u32 *)(w + off + col * 2)};
        each<M>([&](auto mt) {
            constexpr int m = First + decltype(mt)::value;
            E4x4 packed{p.v[m][n]};
            half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)packed.bits, __NV_E4M3));
            half2 hi =
                __half2(__nv_cvt_fp8x2_to_halfraw2((unsigned short)(packed.bits >> 16), __NV_E4M3));
            c[m][n] = {bits2(__hmul2(lo, h2(gate.bits))), bits2(__hmul2(hi, h2(gate.bits)))};
        });
    });
}
template <int M> __device__ __forceinline__ Packed<M, 4> normpack(Frag (&z)[M][4], half scale) {

#pragma unroll
    for (int m = 0; m < M; m++)
        for (int row = 0; row < 2; row++) {
            u32 a = row ? z[m][0].y : z[m][0].x, b = row ? z[m][1].y : z[m][1].x;
            u32 c = row ? z[m][2].y : z[m][2].x, d = row ? z[m][3].y : z[m][3].x;
            u32 sum = bits2(__hadd2(__hfma2(h2(a), h2(a), __hmul2(h2(c), h2(c))),
                                    __hfma2(h2(b), h2(b), __hmul2(h2(d), h2(d)))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 2))));
            sum = bits2(__hadd2(h2(sum), h2(__shfl_xor_sync(0xffffffff, sum, 1))));
            float x = __half2float(
                      __hmax(__hadd(hh(sum), hh(sum >> 16)), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(x));
            half h = __float2half_rn(inv);

#pragma unroll
            for (int n = 0; n < 4; n++) {
                u32 &v = row ? z[m][n].y : z[m][n].x;
                v = bits2(
                    __hmul2(__hmul2(h2(v), __halves2half2(h, h)), __halves2half2(scale, scale)));
            }
        }
    return pack(z);
}
// Included after the unchanged wide arithmetic and packing primitives.
template <int M, int N> __device__ __forceinline__ Words<M * N> words(const Packed<M, N> &p) {
    Words<M * N> x;
    each<M>([&](auto mt) {
        each<N>([&](auto nt) {
            constexpr int m = decltype(mt)::value, n = decltype(nt)::value;
            x.v[m * N + n] = p.v[m][n];
        });
    });
    return x;
}
template <int H, int M>
__device__ __forceinline__ Packed<M, 8> physicalprobability(Frag (&c)[M][8]) {
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<2>([&](auto rt) {
                constexpr int row = decltype(rt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                half2 z = __hfma2(h2(v), constant2(0.044921875f), constant2(1.30078125f));
                z = __hmin2(__hmax2(z, constant2(1.03125f)), constant2(1.5693359375f));
                v = ((bits2(z) << 5) & 0xffe0ffe0u) ^ 0x80008000u;
            });
        });
        each<2>([&](auto rt) {
            constexpr int row = decltype(rt)::value;
            Words<8> d;
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                d.v[n] = row ? c[m][n].y : c[m][n].x;
            });
            if constexpr (H == 4)
                d = den4_physical_paired(d);
            else
                d = den8_physical_paired(d);
            u32 group;
            each<4>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                u32 pair = bits2(__hadd2(h2(d.v[p]), h2(d.v[4 + p])));
                if constexpr (p == 0)
                    group = pair;
                else
                    group = bits2(__hadd2(h2(group), h2(pair)));
            });
            u32 total;
            each<4>([&](auto gt) {
                constexpr int g = decltype(gt)::value;
                u32 term = __shfl_sync(0xffffffff, group, (threadIdx.x & ~3) + g);
                if constexpr (g == 0)
                    total = term;
                else
                    total = bits2(__hadd2(h2(total), h2(term)));
            });
            float den = __half2float(__hmax(__hadd(hh(total), hh(total >> 16)),
                                            __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half r = __float2half_rn(inv);
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 &v = row ? c[m][n].y : c[m][n].x;
                v = bits2(__hmul2(h2(v), __halves2half2(r, r)));
            });
        });
    });
    return pack(c);
}
template <int H, int M, int First>
__device__ __forceinline__ Packed<> physicalattention(const Words<16> &q, const Words<16> &k,
                                                      const Words<16> &v, const u8 *w) {
    static_assert(M == 2 && (First == 0 || First == 2));
    Words<16> p;
    {
        Frag logits[2][8];
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4] = {q.v[(First + m) * 4], q.v[(First + m) * 4 + 1], q.v[(First + m) * 4 + 2],
                        q.v[(First + m) * 4 + 3]};
            each<4>([&](auto pt) {
                constexpr int packetlocal = decltype(pt)::value;
                constexpr int n = (packetlocal & 1) + 4 * (packetlocal >> 1),
                              packet = 8 * (First / 2) + 4 * m + packetlocal;
                // One raw aligned packet directly occupies its two future QK C pairs.
                asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(logits[m][n].x), "=r"(logits[m][n].y), "=r"(logits[m][n + 2].x),
                               "=r"(logits[m][n + 2].y)
                             : "l"(w + (H == 4 ? 0x24120 : 0x88220) + 8192 * threadIdx.y +
                                   512 * packet + 16 * threadIdx.x));
                mma(logits[m][n], a, k.v[n * 2], k.v[n * 2 + 1]);
                mma(logits[m][n + 2], a, k.v[(n + 2) * 2], k.v[(n + 2) * 2 + 1]);
            });
        });
        if constexpr (H == 4)
            p = prob4_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
        else
            p = prob8_physical_to_accepted_PV_A(words(physicalprobability<H>(logits)));
    }
    Frag result[4][4] = {};
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            constexpr int off = (kp * M + m) * 4;
            u32 a[4] = {p.v[off], p.v[off + 1], p.v[off + 2], p.v[off + 3]};
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                mma(result[m][n], a, v.v[(kp * 4 + n) * 2], v.v[(kp * 4 + n) * 2 + 1]);
            });
        });
    });
    return pack(result);
}

// By-value ABI: decoded shape/grid/shift and symbol-derived input/output flags.
struct Layout {
    int height, width, grid_x, shift_x, shift_y, flags;
};
template <int H, bool Output>
__device__ __forceinline__ int address(Layout p, int tile, int slot, int n) {
    int x = (tile % p.grid_x) * 8 + (H == 4 ? ((slot >> 1) & 7) : (slot & 7)) + p.shift_x;
    int y = (tile / p.grid_x) * 8 + (H == 4 ? (((slot >> 4) & 3) * 2 + (slot & 1)) : (slot / 8)) +
            p.shift_y;
    if (x < 0 || x >= p.width || y < 0 || y >= p.height)
        return -1;
    if (p.flags & (Output ? 2 : 1)) {
        int lo = (n & 1) | ((n & 6) << 1) | ((n & 8) >> 2);
        return (y * p.width + x) * 16 + (n / 16) * p.height * p.width * 16 + lo;
    }
    int nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4);
    int st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3);
    if constexpr (H == 4)
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + (x / 4 + (p.width / 4) * (y / 4)) * 2048;
    else {
        st |= (x & 4) << 2;
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + ((st & 16) << 8) +
               (x / 8 + ((p.width + 7) / 8) * (y / 4)) * 8192;
    }
}
// Original4 chained: expand K4 unrolled (0x1000..); tail loop 0x2c10..2d80;
// QKV M64N96 K4 loop 0x33d0..3760; projection M32 K4 unrolled 0x66c0...
// A is loaded once per owned M16/K32, outside all component/pair consumers.
template <int M, int First = 0>
__device__ __forceinline__ void four_step(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                          int kp) {
    u32 a[M][4];
    each<M>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<4>(s, a[m], First + m, kp);
    });
    each<2>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + off + pair * 512 + threadIdx.x * 16);
        each<M>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            mma(c[First + m][pair * 2], a[m], b.x, b.y);
            mma(c[First + m][pair * 2 + 1], a[m], b.z, b.w);
        });
    });
}
template <int M, int First = 0>
__device__ __forceinline__ void four_unrolled(Frag (&c)[4][4], const u8 *s, const u8 *w, int off,
                                              int stride) {
    each<4>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        four_step<M, First>(c, s, w, off + kp * stride, kp);
    });
}
__device__ __forceinline__ void four_tail(Frag (&c)[4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++)
        four_step<4>(c, s, w, off + kp * 4096, kp);
}
__device__ __forceinline__ void four_qkv(Frag (&c)[3][4][4], const u8 *s, const u8 *w, int off) {
#pragma unroll 1
    for (int kp = 0; kp < 4; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}
__device__ __forceinline__ void four_qkv_prefix(Frag (&c)[3][4][4], const u8 *s, const u8 *w,
                                                int off) {
#pragma unroll 1
    for (int kp = 0; kp < 3; kp++) {
        u32 a[4][4];
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            shared_a<4>(s, a[m], m, kp);
        });
        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b = *(const uint4 *)(w + off + kp * 12288 + component * 1024 + pair * 512 +
                                           threadIdx.x * 16);
                each<4>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    mma(c[component][m][pair * 2], a[m], b.x, b.y);
                    mma(c[component][m][pair * 2 + 1], a[m], b.z, b.w);
                });
            });
        });
    }
}

// hidden_a swaps unit bit1 with register bit0 only: each M16 is closed.
// Both expand and reduce B128 pairs are loaded once and reused across M64.
template <int H>
__device__ __forceinline__ void finish_hidden(Frag (&expanded)[4][4], Frag (&reduced)[4][4],
                                              const u8 *s, const u8 *w, int off, int stride,
                                              int reduceoff) {
    uint4 eb[2], rb[2];
    each<2>([&](auto pt) {
        constexpr int p = decltype(pt)::value;
        eb[p] = *(const uint4 *)(w + off + (H - 1) * stride + p * 512 + threadIdx.x * 16);
        rb[p] = *(const uint4 *)(w + reduceoff + p * 512 + threadIdx.x * 16);
    });
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        A128 a;
        shared_a<H>(s, a.v, m, H - 1);
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(expanded[m][p * 2], a.v, eb[p].x, eb[p].y);
            mma(expanded[m][p * 2 + 1], a.v, eb[p].z, eb[p].w);
        });
        Words<4> e;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            RawHalf2 x{bits2(activation2(h2(expanded[m][n].x)))},
                y{bits2(activation2(h2(expanded[m][n].y)))};
            e.v[n] = encode4(x.bits, y.bits);
        });
        auto packet = hidden_slab_a(e);
        A128 active{{packet.v[0], packet.v[1], packet.v[2], packet.v[3]}};
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            mma(reduced[m][p * 2], active.v, rb[p].x, rb[p].y);
            mma(reduced[m][p * 2 + 1], active.v, rb[p].z, rb[p].w);
        });
    });
}
// Finish one component at a time; Q/K normalization consumes a closed M16.
// Q/K physical routes cross canonical M16: route only at this component tail.
// V's complete physical transpose waits for all 64 rows. Attention is outside
// this function, after the complete K/V packets have been produced.
template <int H>
__device__ __forceinline__ void finish_qkv(Frag (&c)[3][4][4], Words<16> &q, Words<16> &k,
                                           Words<16> &v, const u8 *s, const u8 *w, int off,
                                           int stride, half scale) {
    A128 a[4];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        shared_a<H>(s, a[m].v, m, H - 1);
    });
    each<3>([&](auto ct) {
        constexpr int component = decltype(ct)::value;
        uint4 b[2];
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            b[p] = *(const uint4 *)(w + off + (H - 1) * stride + component * 1024 + p * 512 +
                                    threadIdx.x * 16);
        });
        Words<16> encoded;
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<2>([&](auto pt) {
                constexpr int p = decltype(pt)::value;
                mma(c[component][m][p * 2], a[m].v, b[p].x, b[p].y);
                mma(c[component][m][p * 2 + 1], a[m].v, b[p].z, b[p].w);
            });
            Frag row[1][4];
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                row[0][n] = c[component][m][n];
            });
            Packed<1, 4> packed;
            if constexpr (component < 2)
                packed = normpack(row, component == 0 ? scale : __float2half(1));
            else
                packed = pack(row);
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                encoded.v[m * 4 + n] = packed.v[0][n];
            });
        });
        if constexpr (component == 0) {
            if constexpr (H == 4)
                q = query4_physical_a(encoded);
            else
                q = query8_physical_a(encoded);
        } else if constexpr (component == 1) {
            if constexpr (H == 4)
                k = key4_physical_b(encoded);
            else
                k = key8_physical_b(encoded);
        } else if constexpr (H == 4)
            v = value4_b(encoded);
        else
            v = value8_b(encoded);
    });
}
// M32 encoded C -> raw output packets. No shared scratch or dynamic word indexing.
template <int First, int Role>
__device__ __forceinline__ void publish(u8 *resource, int output, Layout layout, int tile,
                                        Frag (&out)[4][4]) {
    int l = threadIdx.x, warp = threadIdx.y;
    Words<8> x;
    each<8>([&](auto it) {
        constexpr int i = decltype(it)::value;
        x.v[i] = encode4(out[First + i / 4][i % 4].x, out[First + i / 4][i % 4].y);
    });
    if constexpr (Role == 4) {
        // source bits [byte2,lane5,word3] -> destination [0,3,7,8,4,5,6,1,9,2].
        // All source M values participate before any destination-dependent selection.
        x = publish_output(x);
        int row = First * 16 + l / 4 + (l & 1) * 16 + ((l >> 1) & 1) * 8;
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            int idx = address<4, true>(layout, tile, row, warp * 32 + p * 16);
            if (idx >= 0)
                *(uint4 *)(resource + output + idx) = {x.v[p * 4], x.v[p * 4 + 1], x.v[p * 4 + 2],
                                                       x.v[p * 4 + 3]};
        });
    } else {
        // Adapter enforces the fully proved real payload domain: H%4==0, W%4==0,
        // shiftY=0/-4. Each valid base implies all 16 bytes valid in this CTA/slab.
        each<2>([&](auto jt) {
            constexpr int j2 = decltype(jt)::value;
            int row = First * 16 + l / 4 + j2 * 8;
            int idx = address<4, true>(layout, tile, row, warp * 32 + (l & 3) * 2);
            if (idx >= 0) {
                constexpr int mask = j2 ? 0x7632 : 0x5410;
                uint4 v = {__byte_perm(x.v[0], x.v[1], mask), __byte_perm(x.v[4], x.v[5], mask),
                           __byte_perm(x.v[2], x.v[3], mask), __byte_perm(x.v[6], x.v[7], mask)};
                *(uint4 *)(resource + output + idx) = v;
            }
        });
    }
}
#include "routes4.cuh"
#include "addresses.cuh"
// Only final encoded pool A crosses warps, after both live body slabs die.
template <int H> __device__ __forceinline__ void pool_packet(u8 *r, u8 *s, physical::Cross x) {
    constexpr int N = H * 64;
    int l = threadIdx.x, warp = threadIdx.y;
    u8 *published = s + H * 512;
#pragma unroll 1
    for (int group = 0; group < 2; group++) {
        int column = (group * H + warp) * 32;
        uint2 c[4] = {};
#pragma unroll 1
        for (int kp = 0; kp < H; kp++) {
            uint4 a;
            unsigned addr = __cvta_generic_to_shared(s + kp * 512 + l * 16);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                         : "r"(addr));
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * N * 32 + column * 32 + pair * 512 + l * 16);
                physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            });
        }
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                int row = l / 4 + j / 2 * 8, col = column + n * 8 + (l & 3) * 2 + (j & 1);
                published[row * N + col] = enc(physical::component(c[n], j));
            });
        });
    }
    __syncthreads();
    for (int packet = warp * 32 + l; packet < 16 * N / 16; packet += H * 32) {
        int row = packet / (N / 16), group = packet % (N / 16),
            pixel = transition_packet::pixel(H, blockIdx.x, row);
        if (pixel >= 0) {
            u32 v[4] = {};
            each<4>([&](auto wt) {
                constexpr int w = decltype(wt)::value;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    constexpr int raw = w * 4 + b,
                                  c = (raw & 1) | ((raw & 12) >> 1) | ((raw & 2) << 2);
                    v[w] |= u32(published[row * N + group * 16 + c]) << (8 * b);
                });
            });
            *(uint4 *)(r + x.pool_out + pixel * 16 + group * transition_packet::count(H) * 16) =
                make_uint4(v[0], v[1], v[2], v[3]);
        }
    }
}
// Local projection rows are a permutation/subset of the original global rows.
// No inverse device read and no canonical Half reload. Canonical stores remain.
template <int H>
__device__ __forceinline__ void project_input(u8 *r, const u8 *w, Layout layout, int input,
                                              physical::Cross x, const int *ui, u8 *s) {
    constexpr int K = H * 64, N = H * 32;
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    // Each owned physical planar packet is loaded once by this CTA, then routed to MMA A.
    for (int packet = warp * 32 + l; packet < 16 * (K / 16); packet += H * 32) {
        int local = packet / (K / 16), group = packet % (K / 16), row = up_global[tile][local];
        uint4 raw = {};
        if (row >= 0)
            raw = *(const uint4 *)(r + input + row * 16 + group * transition_packet::count(H) * 16);
        int base = (group / 2) * 512 + (local % 8) * 64 + (local / 8) * 4 + (group % 2) * 8;
        *(u32 *)(s + base) = raw.x;
        *(u32 *)(s + base + 16) = raw.y;
        *(u32 *)(s + base + 32) = raw.z;
        *(u32 *)(s + base + 48) = raw.w;
    }
    __syncthreads();
    uint2 c[4] = {};
#pragma unroll 1
    for (int kp = 0; kp < H * 2; kp++) {
        uint4 a = *(const uint4 *)(s + kp * 512 + l * 16);
        each<2>([&](auto pt) {
            constexpr int pair = decltype(pt)::value;
            uint4 b = *(const uint4 *)(w + (H == 4 ? 0x18000 : 0x58000) + kp * N * 32 +
                                       warp * 1024 + pair * 512 + l * 16);
            physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        });
    }
    each<4>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        each<4>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int row = up_global[tile][l / 4 + j / 2 * 8],
                col = warp * 32 + n * 8 + (l & 3) * 2 + (j & 1);
            if (row >= 0)
                ((half *)x.projection)[row * N + col] = physical::component(c[n], j);
        });
    });
    __syncthreads(); // all projection readers finish before gate input overwrites A
    int cid = up_ids[tile];
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4] = {};
        each<4>([&](auto wt) {
            constexpr int word = decltype(wt)::value;
            int token = m * 16 + l / 4 + (word & 1) * 8, rr = up_local[cid][token];
            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;
                int col = route((word / 2) * 16 + (l & 3) * 4 + pair * 2),
                    src = (rr & 7) * 4 + (col & 7) / 2;
                constexpr int n = 2 * (word / 2) + pair;
                // Every source register index is compile-time fixed before the shuffle.
                u32 lo = __shfl_sync(0xffffffff, c[n].x, src),
                    hi = __shfl_sync(0xffffffff, c[n].y, src);
                u32 value = (rr & 8) ? hi : lo;
                each<2>([&](auto ht) {
                    constexpr int h = decltype(ht)::value;
                    int channel = warp * 32 + col + h,
                        raw = address<H, false>(layout, tile, token, channel);
                    if (raw >= 0 && rr >= 0)
                        a[word] |= u32(enc(__hfma(dec(r[x.skip + raw]), x.gate[channel],
                                                  hh(value >> (h * 16)))))
                                   << ((pair * 2 + h) * 8);
                });
            });
        });
        *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = make_uint4(a[0], a[1], a[2], a[3]);
    });
}
template <int Role>
__device__ __forceinline__ void four_body(u8 *resource, const u8 *w, Layout layout, int input,
                                          int output, int counter, int up, const u8 *inputResource,
                                          physical::Cross cross, const int *ui) {
    constexpr int H = 4, C = 128;
    extern __shared__ u8 s[];
    int l = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
    u32 poolA[4] = {0, 0, 0, 0};
    if constexpr (Role == 3)
        project_input<H>(resource, w, layout, input, cross, ui, s);
    else {
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            uint4 a = {0, 0, 0, 0};
            u32 *av = (u32 *)&a;
            each<4>([&](auto rt) {
                constexpr int word = decltype(rt)::value;
                int row = m * 16 + l / 4 + (word & 1) * 8,
                    k = warp * 32 + (word / 2) * 16 + (l & 3) * 4;
                {
                    int idx = address<H, false>(layout, tile, row, route(k));
                    uint4 raw = {0, 0, 0, 0};
                    if (idx >= 0)
                        raw = *(const uint4 *)(inputResource + input + (idx & ~15));
                    int selector = (idx & 15) >> 2;
                    u32 value = selector == 0   ? raw.x
                                : selector == 1 ? raw.y
                                : selector == 2 ? raw.z
                                                : raw.w;
                    av[word] = idx < 0 ? 0 : value;
                }
            });
            *(uint4 *)(s + warp * 2048 + m * 512 + l * 16) = a;
        });
    }
    __syncthreads();
    auto original = read_encoded<4, 0>(s);
    Frag reduced[4][4] = {};
#pragma unroll 1
    for (int hidden = 0; hidden < 4; hidden++) {
        Frag expanded[4][4] = {};
        each<3>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            four_step<4>(expanded, s, w, warp * C * 128 + hidden * 1024 + kp * 4096, kp);
        });
        finish_hidden<4>(expanded, reduced, s, w, warp * C * 128 + hidden * 1024, 4096,
                         0x10000 + warp * 4096 + hidden * 1024);
    }
    auto mid = pack(reduced);
    __syncthreads();
    exchange<H>(mid, s);
    __syncthreads();
    Frag ff[4][4];
    initial<H>(ff, original, w, Role == 3 ? 0x20000 : 0x18010);
    four_tail(ff, s, w, 0x14000 + warp * 1024);
    const u8 *aw = w + (Role == 3 ? 0x80e0 : 0);
    {
        auto feature = pack(ff);
        __syncthreads();
        exchange<H>(feature, s);
    }
    __syncthreads();
    Words<16> q, k, v;
    {
        Frag qkv[3][4][4] = {};
        four_qkv_prefix(qkv, s, aw, 0x18120 + warp * 3072);
        finish_qkv<4>(qkv, q, k, v, s, aw, 0x18120 + warp * 3072, 12288,
                      __float2half_rn(((const float *)(aw + 0x2c120))[warp]));
    }
    __syncthreads();
    each<2>([&](auto ft) {
        constexpr int first = decltype(ft)::value * 2;
        auto pv = physicalattention<H, 2, first>(q, k, v, aw);
        // FF stays in its original 4096-byte M32 slab until every warp's late read.
        auto feature = read_encoded<2, first>(s);
        __syncthreads();
        exchange<H>(pv, s, first * 16, 2);
        __syncthreads();
        Frag out[4][4];
        initial<H, 2, first>(out, feature, aw, 0x30130);
        four_unrolled<2, first>(out, s, aw, 0x2c130 + warp * 1024, 4096);
        if constexpr (Role == 2)
            collect_pool<first>(out, poolA);
        publish<first, Role>(resource, output, layout, tile, out);
        // All projection warps have consumed PV before the next slab is overwritten.
        __syncthreads();
    });
    if constexpr (Role == 2) {
        *(uint4 *)(s + warp * 512 + l * 16) = make_uint4(poolA[0], poolA[1], poolA[2], poolA[3]);
        __syncthreads();
        pool_packet<H>(resource, s, cross);
    }
    __threadfence();
    if (counter >= 0 && l == 0 && warp == 0)
        ((int *)(resource + counter))[tile] = 0;
}
extern "C" __global__ void packet4_up(u8 *r, const u8 *w, Layout layout, int in, int out, int cb,
                                      int up, const u8 *src, physical::Cross cross, const int *ui) {
    four_body<3>(r, w, layout, in, out, cb, up, src, cross, ui);
}

} // namespace nr_outer_up4_7
