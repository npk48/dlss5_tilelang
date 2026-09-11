// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// DEEP transition.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_transition_5 {
// Generic-shape derivative of the selected native arithmetic.

struct CompletionRegion {
    unsigned long long pointer;
    int count;
    int value;
};
struct CompletionPublications {
    CompletionRegion regions[2];
    int count;
    int padding;
};
static_assert(sizeof(CompletionPublications) == 40, "publication ABI");
// Not a tile-readiness signal. Every real reader remains a later default-stream
// kernel. Each thread reaches this tail only after its original mathematical body.
__device__ __forceinline__ void completion_tail(CompletionPublications p) {
    unsigned int local = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned int threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned int tid = local + threads * block,
                 stride = threads * gridDim.x * gridDim.y * gridDim.z;
#pragma unroll
    for (int region = 0; region < 2; region++) {
        if (region >= p.count)
            continue;
        CompletionRegion r = p.regions[region];
        for (unsigned int i = tid; i < (unsigned int)r.count; i += stride) {
            // Volatile stores plus the memory clobber keep publication after original
            // per-thread memory effects. No global fence/spin or new readiness protocol.
            asm volatile("st.global.u32 [%0], %1;" ::"l"(r.pointer + 4ull * i), "r"(r.value)
                         : "memory");
        }
    }
}
using u8 = unsigned char;
using u32 = unsigned int;
struct Frag {
    u32 x, y;
};
__device__ __forceinline__ half hh(u32 x) {
    return __ushort_as_half((unsigned short)x);
}
__device__ __forceinline__ half part(Frag f, int j) {
    return hh((j < 2 ? f.x : f.y) >> ((j & 1) * 16));
}
__device__ __forceinline__ void put(Frag &f, int j, half h) {
    u32 &v = j < 2 ? f.x : f.y;
    int s = (j & 1) * 16;
    v = (v & ~(65535u << s)) | (u32(__half_as_ushort(h)) << s);
}
__device__ __forceinline__ half dec(u8 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}
__device__ __forceinline__ u8 enc(half x) {
    return __nv_cvt_halfraw_to_fp8((__half_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half activation(half x) {
    half z = __hmin(__hmax(x, __float2half(-4)), __float2half(4));
    half g = __hfma(__habs(z), __float2half(-.055908203125f), __float2half(.447265625f));
    return __hmul(x, __hfma(z, g, __float2half(.89453125f)));
}
__device__ __forceinline__ void cp(void *dst, const void *src, int bytes) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void mma(Frag &c, const u32 *a, const u32 *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c.x), "+r"(c.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void norm(Frag *f, int l, int component, int head, const float *scale,
                                     int vit) {
    if (component < 2) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half sum[2];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int i = row * 2 + j;
                half a = part(f[0], i), b = part(f[1], i), c = part(f[2], i), d = part(f[3], i);
                sum[j] = __hadd(__hfma(a, a, __hmul(c, c)), __hfma(b, b, __hmul(d, d)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 2)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 1)));
            }
            float den = __half2float(
                      __hmax(__hadd(sum[0], sum[1]), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half h = __float2half_rn(inv);
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    half v = __hmul(part(f[n], row * 2 + j), h);
                    if (component == 0) {
                        if (vit)
                            v = __hmul(v, __float2half(5.65625f));
                        v = __hmul(v, __float2half(scale[head]));
                    }
                    put(f[n], row * 2 + j, v);
                }
        }
    }
}
// Producer-owned double K64 buffer. Every CTA shares each A packet across N128
// and each B packet across M128. No alternate weights or persistent layout.
__device__ __forceinline__ u32 aword(const u8 *s, const unsigned short *route, int idx) {
    u32 v = 0;
#pragma unroll
    for (int b = 0; b < 4; b++)
        v |= u32(s[route[idx + b]]) << (8 * b);
    return v;
}
__device__ __forceinline__ void stage(u8 *s, const u8 *a, const u8 *w, int mb, int nb, int base,
                                      int rows, int K, int N, const int *packets) {
    packets += (mb / 128 * (K / 64) + base / 64) * 1024;
    int t = threadIdx.x;
    for (int i = t; i < 1024; i += 128) {
        int off = packets[i];
        cp(s + i * 16, a + (off < 0 ? 0 : off), off < 0 ? 0 : 16);
    }
    for (int i = t; i < 512; i += 128) {
        int step = i / 256, packet = i % 256;
        cp(s + 16384 + i * 16, w + (base / 32 + step) * 32 * N + nb * 32 + packet * 16, 16);
    }
    asm volatile("cp.async.commit_group;" ::);
}
extern "C" __global__ void matrix_phase(const u8 *a, const u8 *w, const u8 *skip,
                                        const int *skiproute, half *out, half *partials, int rows,
                                        int K, int N, int mode, int splits, int gate, int kind,
                                        u8 *quant, int act, u8 *raw, const int *map, u8 *q, u8 *k,
                                        u8 *v, const float *scale, const int *packets,
                                        const unsigned short *routes, const int *skipmap) {
    __shared__ __align__(16) u8 s[2][24576];
    int t = threadIdx.x, l = t & 31, wm = (t / 32) / 2, wn = (t / 32) % 2, mb = blockIdx.x * 128,
        nb = blockIdx.y * 128, p = blockIdx.z;
    Frag c[4][8] = {};
    if (skip && p == 0)
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int n = 0; n < 8; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int m = mb + wm * 64 + r * 16 + l / 4 + (j / 2) * 8,
                        col = nb + wn * 64 + n * 8 + (l & 3) * 2 + (j & 1);
                    if (m < rows)
                        put(c[r][n], j,
                            __hmul(dec(skip[skipmap[m * N + (skiproute ? skiproute[col] : col)]]),
                                   ((const half *)(w + gate))[col]));
                }
    int begin = p * (K / splits), end = (p + 1) * (K / splits);
    stage(s[0], a, w, mb, nb, begin, rows, K, N, packets);
#pragma unroll 1
    for (int base = begin, phase = 0; base < end; base += 64, phase ^= 1) {
        asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if (base + 64 < end)
            stage(s[phase ^ 1], a, w, mb, nb, base + 64, rows, K, N, packets);
        const unsigned short *ar = routes + (mb / 128 * (K / 64) + base / 64) * 8192;
#pragma unroll
        for (int step = 0; step < 2; step++) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                u32 av[4];
#pragma unroll
                for (int z = 0; z < 4; z++) {
                    int m = wm * 64 + r * 16 + l / 4 + (z & 1) * 8,
                        kp = step * 32 + (z / 2) * 16 + (l & 3) * 4;
                    u32 val = 0;
                    if (!mode)
                        val = aword(s[phase], ar, m * 64 + kp);
                    else {
                        int at = m * 64 + step * 32 + (z / 2) * 16 + (l & 3) * 2;
                        val = u32(s[phase][ar[at]]) | (u32(s[phase][ar[at + 1]]) << 8) |
                              (u32(s[phase][ar[at + 8]]) << 16) | (u32(s[phase][ar[at + 9]]) << 24);
                    }
                    av[z] = val;
                }
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    const u32 *bp = (u32 *)(s[phase] + 16384 + step * 4096 + wn * 2048 +
                                            (n / 2) * 512 + (n % 2) * 8 + l * 16);
                    mma(c[r][n], av, bp);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int m = mb + wm * 64 + r * 16 + l / 4 + (j / 2) * 8,
                    col = nb + wn * 64 + n * 8 + (l & 3) * 2 + (j & 1);
                if (m < rows) {
                    int idx = m * N + col;
                    half h = part(c[r][n], j);
                    if (splits > 1)
                        partials[p * rows * N + idx] = h;
                    else {
                        if (out)
                            out[idx] = h;
                        if (quant || raw) {
                            u8 val = enc(act ? activation(h) : h);
                            if (quant)
                                quant[idx] = val;
                            if (raw)
                                raw[map[idx]] = val;
                        }
                    }
                }
            }
        if (q) {
#pragma unroll
            for (int g = 0; g < 2; g++) {
                int col = nb + wn * 64 + g * 32, component = (col / 32) % 3, head = col / 96;
                norm(c[r] + g * 4, l, component, head, scale, 0);
                u8 *dst = component == 0 ? q : component == 1 ? k : v;
#pragma unroll
                for (int n = 0; n < 4; n++)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int m = mb + wm * 64 + r * 16 + l / 4 + (j / 2) * 8;
                        if (m < rows)
                            dst[m * 512 + head * 32 + n * 8 + (l & 3) * 2 + (j & 1)] =
                                enc(part(c[r][g * 4 + n], j));
                    }
            }
        }
    }
}
// Kept after the real raw-scratch publication. The raw load is not replaced by
// a private partial; Half merge still precedes FP8 conversion and raw scatter.
extern "C" __global__ void merge_phase(const half *partials, const half *raw, const int *map,
                                       half *out, int count, int splits, u8 *quant, int act,
                                       u8 *published, const int *pubmap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        half h = __hadd(raw[map[i]], partials[(splits - 1) * count + i]);
        if (out)
            out[i] = h;
        if (quant || published) {
            u8 v = enc(act ? activation(h) : h);
            if (quant)
                quant[i] = v;
            if (published)
                published[pubmap[i]] = v;
        }
    }
}
extern "C" __global__ void merge_phase_completion(const half *partials, const half *raw,
                                                  const int *map, half *out, int count, int splits,
                                                  u8 *quant, int act, u8 *published,
                                                  const int *pubmap,
                                                  CompletionPublications completion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        half h = __hadd(raw[map[i]], partials[(splits - 1) * count + i]);
        if (out)
            out[i] = h;
        if (quant || published) {
            u8 v = enc(act ? activation(h) : h);
            if (quant)
                quant[i] = v;
            if (published)
                published[pubmap[i]] = v;
        }
    }
    completion_tail(completion);
}

} // namespace nr_deep_transition_5
// ============================================================================
// DEEP repack.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_repack_6 {
// Generic-shape derivative of the selected native arithmetic.

struct CompletionRegion {
    unsigned long long pointer;
    int count;
    int value;
};
struct CompletionPublications {
    CompletionRegion regions[2];
    int count;
    int padding;
};
static_assert(sizeof(CompletionPublications) == 40, "publication ABI");
// Not a tile-readiness signal. Every real reader remains a later default-stream
// kernel. Each thread reaches this tail only after its original mathematical body.
__device__ __forceinline__ void completion_tail(CompletionPublications p) {
    unsigned int local = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned int threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned int block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned int tid = local + threads * block,
                 stride = threads * gridDim.x * gridDim.y * gridDim.z;
#pragma unroll
    for (int region = 0; region < 2; region++) {
        if (region >= p.count)
            continue;
        CompletionRegion r = p.regions[region];
        for (unsigned int i = tid; i < (unsigned int)r.count; i += stride) {
            // Volatile stores plus the memory clobber keep publication after original
            // per-thread memory effects. No global fence/spin or new readiness protocol.
            asm volatile("st.global.u32 [%0], %1;" ::"l"(r.pointer + 4ull * i), "r"(r.value)
                         : "memory");
        }
    }
}
using u8 = unsigned char;
using u32 = unsigned int;
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
__device__ __forceinline__ half part(Frag f, int i) {
    return hh((i < 2 ? f.x : f.y) >> ((i & 1) * 16));
}
__device__ __forceinline__ void put(Frag &f, int i, half h) {
    u32 &v = i < 2 ? f.x : f.y;
    int s = (i & 1) * 16;
    v = (v & ~(65535u << s)) | (hb(h) << s);
}
__device__ __forceinline__ void mma(Frag &d, u32 *a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int route(int k, int mode) {
    if (mode == 1)
        return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
    if (mode == 2)
        return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
    return k;
}
__device__ __forceinline__ half activation(half x) {
    half z = __hmin(__hmax(x, __float2half(-4)), __float2half(4));
    half g = __hfma(__habs(z), __float2half(-.055908203125f), __float2half(.447265625f));
    return __hmul(x, __hfma(z, g, __float2half(.89453125f)));
}
__device__ __forceinline__ int dep(int v, const int *bits, int n) {
    int r = 0;
    for (int i = 0; i < n; i++)
        r |= ((v >> i) & 1) << bits[i];
    return r;
}
__device__ __forceinline__ void norm(Frag *f, int l, int component, int head, const float *scale,
                                     int vit) {
    if (component < 2) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half sum[2];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int i = row * 2 + j;
                half a = part(f[0], i), b = part(f[1], i), c = part(f[2], i), d = part(f[3], i);
                sum[j] = __hadd(__hfma(a, a, __hmul(c, c)), __hfma(b, b, __hmul(d, d)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 2)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 1)));
            }
            float den = __half2float(
                      __hmax(__hadd(sum[0], sum[1]), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half h = __float2half_rn(inv);
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    half v = __hmul(part(f[n], row * 2 + j), h);
                    if (component == 0) {
                        if (vit)
                            v = __hmul(v, __float2half(5.65625f));
                        v = __hmul(v, __float2half(scale[head]));
                    }
                    put(f[n], row * 2 + j, v);
                }
        }
    }
}
__device__ __constant__ int QORDER[32] = {0,  1,  8,  9,  2,  3,  10, 11, 4,  5,  12,
                                          13, 6,  7,  14, 15, 16, 17, 24, 25, 18, 19,
                                          26, 27, 20, 21, 28, 29, 22, 23, 30, 31};
__device__ __constant__ int SUM16[64] = {
    0,  1,  16, 17, 32, 33, 48, 49, 4,  5,  20, 21, 36, 37, 52, 53, 8,  9,  24, 25, 40, 41,
    56, 57, 12, 13, 28, 29, 44, 45, 60, 61, 2,  3,  18, 19, 34, 35, 50, 51, 6,  7,  22, 23,
    38, 39, 54, 55, 10, 11, 26, 27, 42, 43, 58, 59, 14, 15, 30, 31, 46, 47, 62, 63};
__device__ __constant__ int SUMV[64] = {
    0,  8,  16, 24, 32, 40, 48, 56, 2,  10, 18, 26, 34, 42, 50, 58, 4,  12, 20, 28, 36, 44,
    52, 60, 6,  14, 22, 30, 38, 46, 54, 62, 1,  9,  17, 25, 33, 41, 49, 57, 3,  11, 19, 27,
    35, 43, 51, 59, 5,  13, 21, 29, 37, 45, 53, 61, 7,  15, 23, 31, 39, 47, 55, 63};
__device__ __forceinline__ half scoreget(Frag *f, int row, int key) {
    u32 v = row ? f[key / 8].y : f[key / 8].x;
    v = __shfl_sync(0xffffffff, v, ((threadIdx.x & 31) / 4) * 4 + (key & 7) / 2);
    return hh(v >> ((key & 1) * 16));
}
__device__ __forceinline__ half sum64(Frag *f, int row, int vit) {
    const int *order = vit ? SUMV : SUM16;
    half halves[2];
    for (int h = 0; h < 2; h++) {
        half total = hh(0);
        for (int g = 0; g < 4; g++) {
            half group = hh(0);
            for (int p = 0; p < 4; p++) {
                int i = h * 32 + g * 8 + p * 2;
                half pair = __hadd(scoreget(f, row, order[i]), scoreget(f, row, order[i + 1]));
                group = p ? __hadd(group, pair) : pair;
            }
            total = g ? __hadd(total, group) : group;
        }
        halves[h] = total;
    }
    return __hadd(halves[0], halves[1]);
}
__device__ __forceinline__ int pvorder(int k, int vit) {
    return vit ? (k & ~15) + ((k % 16) / 4) * 2 + (k & 1) + ((k >> 1) & 1) * 8
               : (k & ~3) | ((k & 1) << 1) | ((k & 2) >> 1);
}
union H2 {
    u32 u;
    half2 h;
};
__device__ __forceinline__ half2 pair(u32 x) {
    H2 v;
    v.u = x;
    return v.h;
}
__device__ __forceinline__ u32 bits(half2 x) {
    H2 v;
    v.h = x;
    return v.u;
}
__device__ __forceinline__ half2 dup(float x) {
    return __float2half2_rn(x);
}
__device__ __forceinline__ unsigned short qpair(half2 x) {
    return __nv_cvt_halfraw2_to_fp8x2((__half2_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half2 act2(half2 x) {
    half2 z = __hmin2(__hmax2(x, dup(-4)), dup(4));
    half2 g = __hfma2(__habs2(z), dup(-.055908203125f), dup(.447265625f));
    return __hmul2(x, __hfma2(z, g, dup(.89453125f)));
}
extern "C" __global__ void raw_repack_completion(const u8 *a, u8 *out, const int *im, const int *om,
                                                 int count, CompletionPublications completion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        out[om ? om[i] : i] = a[im[i]];
    completion_tail(completion);
}
extern "C" __global__ void up_exit_completion(const half *projected, const u8 *skip,
                                              const half *gain, u8 *raw, const int *perm,
                                              const int *om, int count, int W, int pool_width,
                                              CompletionPublications completion) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < count) {
        int i = perm[j], p = i / 512, n = i % 512;
        raw[om[j]] = enc(__hfma(dec(skip[i]), gain[n],
                                projected[((p / W / 2) * pool_width + (p % W / 2)) * 512 + n]));
    }
    completion_tail(completion);
}

} // namespace nr_deep_repack_6
// ============================================================================
// DEEP split.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_split_7 {
using u8 = unsigned char;
using u32 = unsigned int;
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
__device__ __forceinline__ half part(Frag f, int i) {
    return hh((i < 2 ? f.x : f.y) >> ((i & 1) * 16));
}
__device__ __forceinline__ void put(Frag &f, int i, half h) {
    u32 &v = i < 2 ? f.x : f.y;
    int s = (i & 1) * 16;
    v = (v & ~(65535u << s)) | (hb(h) << s);
}
__device__ __forceinline__ void mma(Frag &d, u32 *a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int route(int k, int mode) {
    if (mode == 1)
        return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
    if (mode == 2)
        return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1);
    return k;
}
__device__ __forceinline__ half activation(half x) {
    half z = __hmin(__hmax(x, __float2half(-4)), __float2half(4));
    half g = __hfma(__habs(z), __float2half(-.055908203125f), __float2half(.447265625f));
    return __hmul(x, __hfma(z, g, __float2half(.89453125f)));
}
__device__ __forceinline__ int dep(int v, const int *bits, int n) {
    int r = 0;
    for (int i = 0; i < n; i++)
        r |= ((v >> i) & 1) << bits[i];
    return r;
}
__device__ __forceinline__ void norm(Frag *f, int l, int component, int head, const float *scale,
                                     int vit) {
    if (component < 2) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half sum[2];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int i = row * 2 + j;
                half a = part(f[0], i), b = part(f[1], i), c = part(f[2], i), d = part(f[3], i);
                sum[j] = __hadd(__hfma(a, a, __hmul(c, c)), __hfma(b, b, __hmul(d, d)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 2)));
                sum[j] = __hadd(sum[j],
                                hh(__shfl_xor_sync(0xffffffff, (u32)__half_as_ushort(sum[j]), 1)));
            }
            float den = __half2float(
                      __hmax(__hadd(sum[0], sum[1]), __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half h = __float2half_rn(inv);
#pragma unroll
            for (int n = 0; n < 4; n++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    half v = __hmul(part(f[n], row * 2 + j), h);
                    if (component == 0) {
                        if (vit)
                            v = __hmul(v, __float2half(5.65625f));
                        v = __hmul(v, __float2half(scale[head]));
                    }
                    put(f[n], row * 2 + j, v);
                }
        }
    }
}
__device__ __constant__ int QORDER[32] = {0,  1,  8,  9,  2,  3,  10, 11, 4,  5,  12,
                                          13, 6,  7,  14, 15, 16, 17, 24, 25, 18, 19,
                                          26, 27, 20, 21, 28, 29, 22, 23, 30, 31};
__device__ __constant__ int SUM16[64] = {
    0,  1,  16, 17, 32, 33, 48, 49, 4,  5,  20, 21, 36, 37, 52, 53, 8,  9,  24, 25, 40, 41,
    56, 57, 12, 13, 28, 29, 44, 45, 60, 61, 2,  3,  18, 19, 34, 35, 50, 51, 6,  7,  22, 23,
    38, 39, 54, 55, 10, 11, 26, 27, 42, 43, 58, 59, 14, 15, 30, 31, 46, 47, 62, 63};
__device__ __constant__ int SUMV[64] = {
    0,  8,  16, 24, 32, 40, 48, 56, 2,  10, 18, 26, 34, 42, 50, 58, 4,  12, 20, 28, 36, 44,
    52, 60, 6,  14, 22, 30, 38, 46, 54, 62, 1,  9,  17, 25, 33, 41, 49, 57, 3,  11, 19, 27,
    35, 43, 51, 59, 5,  13, 21, 29, 37, 45, 53, 61, 7,  15, 23, 31, 39, 47, 55, 63};
__device__ __forceinline__ half scoreget(Frag *f, int row, int key) {
    u32 v = row ? f[key / 8].y : f[key / 8].x;
    v = __shfl_sync(0xffffffff, v, ((threadIdx.x & 31) / 4) * 4 + (key & 7) / 2);
    return hh(v >> ((key & 1) * 16));
}
__device__ __forceinline__ half sum64(Frag *f, int row, int vit) {
    const int *order = vit ? SUMV : SUM16;
    half halves[2];
    for (int h = 0; h < 2; h++) {
        half total = hh(0);
        for (int g = 0; g < 4; g++) {
            half group = hh(0);
            for (int p = 0; p < 4; p++) {
                int i = h * 32 + g * 8 + p * 2;
                half pair = __hadd(scoreget(f, row, order[i]), scoreget(f, row, order[i + 1]));
                group = p ? __hadd(group, pair) : pair;
            }
            total = g ? __hadd(total, group) : group;
        }
        halves[h] = total;
    }
    return __hadd(halves[0], halves[1]);
}
__device__ __forceinline__ int pvorder(int k, int vit) {
    return vit ? (k & ~15) + ((k % 16) / 4) * 2 + (k & 1) + ((k >> 1) & 1) * 8
               : (k & ~3) | ((k & 1) << 1) | ((k & 2) >> 1);
}
union H2 {
    u32 u;
    half2 h;
};
__device__ __forceinline__ half2 pair(u32 x) {
    H2 v;
    v.u = x;
    return v.h;
}
__device__ __forceinline__ u32 bits(half2 x) {
    H2 v;
    v.h = x;
    return v.u;
}
__device__ __forceinline__ half2 dup(float x) {
    return __float2half2_rn(x);
}
__device__ __forceinline__ unsigned short qpair(half2 x) {
    return __nv_cvt_halfraw2_to_fp8x2((__half2_raw)x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half2 act2(half2 x) {
    half2 z = __hmin2(__hmax2(x, dup(-4)), dup(4));
    half2 g = __hfma2(__habs2(z), dup(-.055908203125f), dup(.447265625f));
    return __hmul2(x, __hfma2(z, g, dup(.89453125f)));
}
// Producer partitions 0,1,2 are merged in original Half order and published.
// The separate merge_phase kernel really reloads this raw scratch before p3.
extern "C" __global__ void publish_split(const half *partials, half *raw, const int *map, int count,
                                         int splits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        half v = partials[i];
        for (int p = 1; p < splits - 1; p++)
            v = __hadd(v, partials[p * count + i]);
        raw[map[i]] = v;
    }
}
extern "C" __global__ void permute(const u8 *a, u8 *b, const int *map, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        b[i] = a[map[i]];
}

} // namespace nr_deep_split_7
