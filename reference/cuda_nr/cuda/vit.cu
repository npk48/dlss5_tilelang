// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// DEEP matrix.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_matrix_2 {
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
// New equivalent CTA-local organization. No original barrier/spin implementation.
__device__ __forceinline__ int logical(int p) {
    return (((p % PH) & ~1) | ((p / PH) & 1)) * PW + (((p / PH) & ~1) | ((p % PH) & 1));
}
__device__ __forceinline__ int rawc(int p, int c, int N) {
    return (p / 16) * 16 * N + (p & 7) * 64 + ((p >> 3) & 1) * 4 + (c & 1) + ((c & 6) << 3) +
           ((c & 8) >> 2) + ((c & 16) >> 1) + (c >> 5) * 512;
}
__device__ __forceinline__ void copy16(void *dst, const void *src) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ __forceinline__ uint4 shared4(const void *ptr) {
    uint4 v;
    u32 s = __cvta_generic_to_shared(ptr);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(s));
    return v;
}
__device__ __forceinline__ void copyA(void *dst, const u8 *a, int off, bool valid) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s),
                 "l"(a + (valid ? off : 0)), "r"(valid ? 16 : 0));
}
template <int K, int N, int PK>
__device__ __forceinline__ void abstage(u8 *dst, const u8 *a, const u8 *weights, int mt, int nb,
                                        int base) {
    int l = threadIdx.x, w = threadIdx.y;
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int st = 0; st < PK / 32; st++)
            copyA(dst + (w + h * 4) * PK * 16 + st * 512 + l * 16, a,
                  ((mt * 8 + w + h * 4) * K * 16 + (base / 32 + st) * 512 + l * 16),
                  (mt * 8 + w + h * 4) * 16 < DT);
    }
// One commit contains the complete A and unique B page for every thread.
#pragma unroll
    for (int x = w * 32 + l; x < (128 * PK) / 16; x += 128) {
        int st = x / 256, packet = x % 256;
        copy16(dst + 128 * PK + x * 16,
               weights + (base / 32 + st) * 32 * N + nb * 32 + packet * 16);
    }
    asm volatile("cp.async.commit_group;" ::);
}
__device__ __forceinline__ half2 decoded(unsigned short v) {
    return half2(__nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3));
}
__device__ __forceinline__ void row_issue(Frag *c, const u8 *s, const uint4 *b) {
    u32 addr = __cvta_generic_to_shared(s);
    asm volatile(
        "{ .reg .b32 a<4>;\n"
        "ld.shared.v4.u32 {a0,a1,a2,a3}, [%32];\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {a0,a1,a2,a3}, {%16,%17}, {%0,%1};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%2,%3}, {a0,a1,a2,a3}, {%18,%19}, {%2,%3};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%4,%5}, {a0,a1,a2,a3}, {%20,%21}, {%4,%5};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%6,%7}, {a0,a1,a2,a3}, {%22,%23}, {%6,%7};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%8,%9}, {a0,a1,a2,a3}, {%24,%25}, {%8,%9};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%10,%11}, {a0,a1,a2,a3}, {%26,%27}, {%10,%11};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%12,%13}, {a0,a1,a2,a3}, {%28,%29}, {%12,%13};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%14,%15}, {a0,a1,a2,a3}, {%30,%31}, {%14,%15};\n"
        "}"
        : "+r"(c[0].x), "+r"(c[0].y), "+r"(c[1].x), "+r"(c[1].y), "+r"(c[2].x), "+r"(c[2].y),
          "+r"(c[3].x), "+r"(c[3].y), "+r"(c[4].x), "+r"(c[4].y), "+r"(c[5].x), "+r"(c[5].y),
          "+r"(c[6].x), "+r"(c[6].y), "+r"(c[7].x), "+r"(c[7].y)
        : "r"(b[0].x), "r"(b[0].y), "r"(b[0].z), "r"(b[0].w), "r"(b[1].x), "r"(b[1].y), "r"(b[1].z),
          "r"(b[1].w), "r"(b[2].x), "r"(b[2].y), "r"(b[2].z), "r"(b[2].w), "r"(b[3].x), "r"(b[3].y),
          "r"(b[3].z), "r"(b[3].w), "r"(addr));
}
template <int ROLE>
__device__ __forceinline__ void matrix(const u8 *a, const u8 *w, const u8 *skip, half *partials,
                                       u8 *raw, u8 *s) {
    constexpr int K = ROLE == 1 ? 4096 : 1024, N = ROLE == 0 ? 4096 : 1024,
                  PK = ROLE == 2 ? 32 : 64, PAGES = ROLE == 0 ? 3 : 2, APAGE = 128 * PK,
                  PAGE = 2 * APAGE, PHASES = ROLE == 2 ? 8 : 16;
    int l = threadIdx.x, warp = threadIdx.y, wm = warp / 2, wn = warp % 2, mt = blockIdx.x % MT,
        nb = (blockIdx.x / MT) * 128, p = blockIdx.z;
    Frag c[4][8] = {};
    if constexpr (ROLE != 0) {
        if (p == 0) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
#pragma unroll
                for (int n32 = 0; n32 < 2; n32++) {
                    int prow = mt * 128 + wm * 64 + r * 16 + l / 4,
                        col0 = nb + wn * 64 + n32 * 32 + (l & 3) * 2;
                    uint4 packet = *(const uint4 *)(skip + rawc(prow, col0, N));
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        u32 lo = j < 2 ? packet.x : packet.z, hi = j < 2 ? packet.y : packet.w;
                        half2 gate = *(const half2 *)(w + K * N + 2 * (col0 + j * 8));
                        c[r][n32 * 4 + j].x =
                            bits(__hmul2(decoded((unsigned short)(lo >> ((j & 1) * 16))), gate));
                        c[r][n32 * 4 + j].y =
                            bits(__hmul2(decoded((unsigned short)(hi >> ((j & 1) * 16))), gate));
                    }
                }
            }
        }
    }
    int begin = p * (K / (ROLE == 0 ? 1 : 4));
    abstage<K, N, PK>(s, a, w, mt, nb, begin);
    if constexpr (ROLE == 0)
        abstage<K, N, PK>(s + PAGE, a, w, mt, nb, begin + PK);
#pragma unroll 1
    for (int phase = 0; phase < PHASES; phase++) {
        int page = phase % PAGES, base = begin + phase * PK;
        if constexpr (ROLE == 0) {
            // Two committed groups until the final phase; the tail has only one.
            if (phase + 1 < PHASES)
                asm volatile("cp.async.wait_group 1;" ::);
            else
                asm volatile("cp.async.wait_group 0;" ::);
        } else
            asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if constexpr (ROLE == 0) {
            if (phase + 2 < PHASES)
                abstage<K, N, PK>(s + ((phase + 2) % PAGES) * PAGE, a, w, mt, nb, base + 2 * PK);
        } else {
            if (phase + 1 < PHASES)
                abstage<K, N, PK>(s + ((phase + 1) % PAGES) * PAGE, a, w, mt, nb, base + PK);
        }
#pragma unroll
        for (int st = 0; st < PK / 32; st++) {
            uint4 currentB[4];
#pragma unroll
            for (int n = 0; n < 4; n++)
                currentB[n] =
                    shared4(s + page * PAGE + APAGE + st * 4096 + wn * 2048 + n * 512 + l * 16);
#pragma unroll
            for (int r = 0; r < 4; r++)
                row_issue(c[r], s + page * PAGE + (wm * 4 + r) * PK * 16 + st * 512 + l * 16,
                          currentB);
        }
        // Every reader is finished before any page can be reused by any warp.
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;" ::);
    __syncthreads();
    if constexpr (ROLE == 0) {
        // Dead A pages: four 4096B warp-owned raw64 tiles, no global shadow.
        u8 *scratch = s + warp * 4096;
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int col = n * 8 + (l & 3) * 2;
                *(unsigned short *)(scratch + rawc(r * 16 + l / 4, col, 64)) =
                    qpair(act2(pair(c[r][n].x)));
                *(unsigned short *)(scratch + rawc(r * 16 + l / 4 + 8, col, 64)) =
                    qpair(act2(pair(c[r][n].y)));
            }
        }
        __syncwarp();
#pragma unroll
        for (int t = 0; t < 8; t++) {
            int off = t * 512 + l * 16;
            uint4 value = shared4(scratch + off);
            *(uint4 *)(raw + rawc(mt * 128 + wm * 64 + (off / 1024) * 16, nb + wn * 64, N) +
                       off % 1024) = value;
        }
    } else {
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int prow = mt * 128 + wm * 64 + r * 16 + l / 4,
                    col = nb + wn * 64 + n * 8 + (l & 3) * 2;
                if (prow < DT)
                    *(u32 *)(partials + p * DT * N + logical(prow) * N + col) = c[r][n].x;
                if (prow + 8 < DT)
                    *(u32 *)(partials + p * DT * N + logical(prow + 8) * N + col) = c[r][n].y;
            }
        }
    }
}
// Preserve the complete physical_matrix ABI/arggeometry; dead args remain identity-bound.
#define ARGS                                                                                       \
    const u8 *a, const u8 *w, const u8 *skip, const int *skiproute, half *out, half *partials,     \
        int rows, int K, int N, int mode, int splits, int gate, int kind, u8 *quant, int act,      \
        u8 *raw, const int *map, u8 *q, u8 *k, u8 *v, const float *scale, const int *packets,      \
        const unsigned short *routes, const int *skipmap, int phaseK
extern "C" __global__ void matrix_expand_completion(ARGS, CompletionPublications completion) {
    __shared__ __align__(16) u8 s[3 * 16384];
    matrix<0>(a, w, skip, partials, raw, s);
    completion_tail(completion);
}
extern "C" __global__ void matrix_contract(ARGS) {
    __shared__ __align__(16) u8 s[2 * 16384];
    matrix<1>(a, w, skip, partials, raw, s);
}
extern "C" __global__ void matrix_projection(ARGS) {
    __shared__ __align__(16) u8 s[2 * 8192];
    matrix<2>(a, w, skip, partials, raw, s);
}

} // namespace nr_deep_matrix_2
#undef ARGS
// ============================================================================
// DEEP qkv.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_qkv_3 {
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
// Generic-shape derivative of the selected native arithmetic.
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
// All addresses are in the fixed physical H32W20 view, not HWC.
__device__ inline void copy16(void *dst, const void *src) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ inline uint4 shared4(const void *ptr) {
    uint4 v;
    u32 s = __cvta_generic_to_shared(ptr);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(s));
    return v;
}
__device__ inline void astage(u8 *dst, const u8 *a, int mt, int base) {
    int l = threadIdx.x, w = threadIdx.y;
#pragma unroll
    for (int h = 0; h < 2; h++)
        copy16(dst + (w + h * 4) * 512 + l * 16,
               a + (mt * 8 + w + h * 4) * 16384 + (base / 32) * 512 + l * 16);
    asm volatile("cp.async.commit_group;" ::);
}
__device__ inline void kvstage(u8 *s, const u8 *k, const u8 *v, int head, int phase) {
    int l = threadIdx.x, w = threadIdx.y;
    if ((phase * 4 + w) * 16 < DT)
        copy16(s + w * 512 + l * 16, k + ((phase * 4 + w) * 32 + head) * 512 + l * 16);
    else
        *(uint4 *)(s + w * 512 + l * 16) = make_uint4(0, 0, 0, 0);
    copy16(s + 2048 + w * 512 + l * 16,
           v + phase * 65536 + (w / 2) * 32768 + head * 1024 + (w % 2) * 512 + l * 16);
    asm volatile("cp.async.commit_group;" ::);
}
__device__ inline void rawstore(half *dst, Frag *f, int base) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint4 v = make_uint4(f[h * 2].x, f[h * 2].y, f[h * 2 + 1].x, f[h * 2 + 1].y);
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst + base + h * 256), "r"(v.x),
                     "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
}
__device__ inline void rawmerge(const half *src, Frag *f, int base) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint4 v;
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(src + base + h * 256)
                     : "memory");
        f[h * 2].x = bits(__hadd2(pair(v.x), pair(f[h * 2].x)));
        f[h * 2].y = bits(__hadd2(pair(v.y), pair(f[h * 2].y)));
        f[h * 2 + 1].x = bits(__hadd2(pair(v.z), pair(f[h * 2 + 1].x)));
        f[h * 2 + 1].y = bits(__hadd2(pair(v.w), pair(f[h * 2 + 1].y)));
    }
}
// Two independent Half columns; same square/FMA and xor2,xor1,then column tree.
template <int CO> __device__ inline void normalize(Frag *f, int head, const float *scale) {
    if constexpr (CO < 2) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half2 a = pair(row ? f[0].y : f[0].x), b = pair(row ? f[1].y : f[1].x),
                  c = pair(row ? f[2].y : f[2].x), d = pair(row ? f[3].y : f[3].x);
            half2 sum = __hadd2(__hfma2(a, a, __hmul2(c, c)), __hfma2(b, b, __hmul2(d, d)));
            sum = __hadd2(sum, pair(__shfl_xor_sync(0xffffffff, bits(sum), 2)));
            sum = __hadd2(sum, pair(__shfl_xor_sync(0xffffffff, bits(sum), 1)));
            float den = __half2float(__hmax(__hadd(__low2half(sum), __high2half(sum)),
                                            __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half2 factor = __half2half2(__float2half_rn(inv));
#pragma unroll
            for (int n = 0; n < 4; n++) {
                half2 v = __hmul2(pair(row ? f[n].y : f[n].x), factor);
                if constexpr (CO == 0) {
                    v = __hmul2(v, dup(5.65625f));
                    v = __hmul2(v, __half2half2(__float2half(scale[head])));
                }
                (row ? f[n].y : f[n].x) = bits(v);
            }
        }
    }
}
__device__ inline uint4 qwords(Frag *f) {
    return make_uint4(u32(qpair(pair(f[0].x))) | (u32(qpair(pair(f[1].x))) << 16),
                      u32(qpair(pair(f[0].y))) | (u32(qpair(pair(f[1].y))) << 16),
                      u32(qpair(pair(f[2].x))) | (u32(qpair(pair(f[3].x))) << 16),
                      u32(qpair(pair(f[2].y))) | (u32(qpair(pair(f[3].y))) << 16));
}
template <int CO> __device__ inline void publish(u8 *dst, Frag *f, int g, int head) {
    int l = threadIdx.x;
    if constexpr (CO < 2) {
        uint4 v = qwords(f);
        if constexpr (CO == 1) {
            u32 t = v.y;
            v.y = v.z;
            v.z = t;
        }
        *(uint4 *)(dst + g * 16384 + head * 512 + l * 16) = v;
    } else {
        // Four row bytes per dword. Pair source lanes separated by bit2; all warp lanes
        // participate.
        int src = l & ~4, j = (l >> 2) & 1,
            base = (g / 2) * 32768 + (g % 2) * 4 + head * 1024 + (src / 8) * 16 + (l & 3) * 128 +
                   j * 64;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            u32 e = u32(qpair(pair(f[n].x))) | (u32(qpair(pair(f[n].y))) << 16);
            u32 a = __shfl_sync(0xffffffff, e, src), b = __shfl_sync(0xffffffff, e, src | 4);
            *(u32 *)(dst + base + (n & 1) * 8 + (n / 2) * 512) =
                __byte_perm(a, b, j ? 0x7351 : 0x6240);
        }
    }
}
__device__ inline u32 exponent(u32 value) {
    half2 z = __hfma2(pair(value), dup(.08953857421875f), dup(1.708984375f));
    z = __hmin2(__hmax2(z, dup(1.439453125f)), dup(1.9775390625f));
    // Clamped Half bits lie in [0x3dc2,0x3fe9]; neither lane borrows.
    return ((bits(z) & 0x0fff0fffU) << 4) - 0xc000c000U;
}
// SUMV's exact pair -> four-pair sequential group -> four-lane sequential
// total -> even/odd Half tree. Packing never reassociates additions.
__device__ inline half denominator(Frag *f, int row) {
    half2 t = __hadd2(pair(row ? f[0].y : f[0].x), pair(row ? f[1].y : f[1].x));
#pragma unroll
    for (int n = 2; n < 8; n += 2)
        t = __hadd2(t, __hadd2(pair(row ? f[n].y : f[n].x), pair(row ? f[n + 1].y : f[n + 1].x)));
    int src = threadIdx.x & ~3;
    half2 total = pair(__shfl_sync(0xffffffff, bits(t), src));
#pragma unroll
    for (int n = 1; n < 4; n++)
        total = __hadd2(total, pair(__shfl_sync(0xffffffff, bits(t), src + n)));
    return __hadd(__low2half(total), __high2half(total));
}
// A and two heads of original B are committed as one page group.
__device__ __forceinline__ void abstage(u8 *dst, const u8 *a, const u8 *weights, int mt,
                                        int headpair, int base) {
    int l = threadIdx.x, warp = threadIdx.y;
#pragma unroll
    for (int h = 0; h < 2; h++)
        copy16(dst + (warp + h * 4) * 512 + l * 16,
               a + (mt * 8 + warp + h * 4) * 16384 + (base / 32) * 512 + l * 16);
#pragma unroll
    for (int x = warp * 32 + l; x < 384; x += 128)
        copy16(dst + 4096 + x * 16, weights + 0x80 + (base / 32) * 0x18000 +
                                        (headpair + x / 192) * 0xc00 + (x % 192) * 16);
    asm volatile("cp.async.commit_group;" ::);
}
__device__ __forceinline__ void row_issue(Frag *c, const u8 *s, const uint4 *b) {
    u32 addr = __cvta_generic_to_shared(s);
    asm volatile(
        "{ .reg .b32 a<4>;\n"
        "ld.shared.v4.u32 {a0,a1,a2,a3}, [%48];\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {a0,a1,a2,a3}, {%24,%25}, {%0,%1};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%2,%3}, {a0,a1,a2,a3}, {%26,%27}, {%2,%3};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%4,%5}, {a0,a1,a2,a3}, {%28,%29}, {%4,%5};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%6,%7}, {a0,a1,a2,a3}, {%30,%31}, {%6,%7};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%8,%9}, {a0,a1,a2,a3}, {%32,%33}, {%8,%9};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%10,%11}, {a0,a1,a2,a3}, {%34,%35}, {%10,%11};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%12,%13}, {a0,a1,a2,a3}, {%36,%37}, {%12,%13};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%14,%15}, {a0,a1,a2,a3}, {%38,%39}, {%14,%15};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%16,%17}, {a0,a1,a2,a3}, {%40,%41}, {%16,%17};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%18,%19}, {a0,a1,a2,a3}, {%42,%43}, {%18,%19};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%20,%21}, {a0,a1,a2,a3}, {%44,%45}, {%20,%21};\n"
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%22,%23}, {a0,a1,a2,a3}, {%46,%47}, {%22,%23};\n"
        "}"
        : "+r"(c[0].x), "+r"(c[0].y), "+r"(c[1].x), "+r"(c[1].y), "+r"(c[2].x), "+r"(c[2].y),
          "+r"(c[3].x), "+r"(c[3].y), "+r"(c[4].x), "+r"(c[4].y), "+r"(c[5].x), "+r"(c[5].y),
          "+r"(c[6].x), "+r"(c[6].y), "+r"(c[7].x), "+r"(c[7].y), "+r"(c[8].x), "+r"(c[8].y),
          "+r"(c[9].x), "+r"(c[9].y), "+r"(c[10].x), "+r"(c[10].y), "+r"(c[11].x), "+r"(c[11].y)
        : "r"(b[0].x), "r"(b[0].y), "r"(b[0].z), "r"(b[0].w), "r"(b[1].x), "r"(b[1].y), "r"(b[1].z),
          "r"(b[1].w), "r"(b[2].x), "r"(b[2].y), "r"(b[2].z), "r"(b[2].w), "r"(b[3].x), "r"(b[3].y),
          "r"(b[3].z), "r"(b[3].w), "r"(b[4].x), "r"(b[4].y), "r"(b[4].z), "r"(b[4].w), "r"(b[5].x),
          "r"(b[5].y), "r"(b[5].z), "r"(b[5].w), "r"(addr));
}
// Fixed native raw ABI. Nullable canonical outputs are required null by adapter.
extern "C" __global__ void joint_qkv_completion(const u8 *a, const u8 *w, half *oq, half *ok,
                                                half *ov, half *partial, half *raw, const int *mq,
                                                const int *mk, const int *mv, half *z, u8 *q, u8 *k,
                                                u8 *v, const float *scale, int rows,
                                                const int *packets, const unsigned short *routes,
                                                const int *qmap, const int *kmap, const int *vmap,
                                                CompletionPublications completion) {
    __shared__ __align__(16) u8 s[2][10240];
    int l = threadIdx.x, wm = threadIdx.y / 2, head = (blockIdx.x / MT) * 2 + threadIdx.y % 2,
        mt = blockIdx.x % MT;
#pragma unroll 1
    for (int p = 0; p < 2; p++) {
        Frag c[4][3][4] = {};
        int begin = p * 512, end = begin + 512;
        abstage(s[0], a, w, mt, (blockIdx.x / MT) * 2, begin);
#pragma unroll 1
        for (int base = begin, page = 0; base < end; base += 32, page ^= 1) {
            asm volatile("cp.async.wait_group 0;" ::);
            __syncthreads();
            if (base + 32 < end)
                abstage(s[page ^ 1], a, w, mt, (blockIdx.x / MT) * 2, base + 32);
            uint4 currentB[6];
#pragma unroll
            for (int n = 0; n < 6; n++)
                currentB[n] = shared4(s[page] + 4096 + (threadIdx.y % 2) * 3072 + n * 512 + l * 16);
#pragma unroll
            for (int r = 0; r < 4; r++)
                row_issue(&c[r][0][0], s[page] + (wm * 4 + r) * 512 + l * 16, currentB);
            __syncthreads();
        }
        if (!p) {
#pragma unroll
            for (int r = 0; r < 4; r++)
                for (int co = 0; co < 3; co++)
                    rawstore(raw, c[r][co],
                             co * (DP * 1024) + (mt * 8 + wm * 4 + r) * 16384 + head * 512 + l * 8);
        } else {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                int g = mt * 8 + wm * 4 + r;
                rawmerge(raw, c[r][0], g * 16384 + head * 512 + l * 8);
                normalize<0>(c[r][0], head, scale);
                publish<0>(q, c[r][0], g, head);
                rawmerge(raw, c[r][1], (DP * 1024) + g * 16384 + head * 512 + l * 8);
                normalize<1>(c[r][1], head, scale);
                publish<1>(k, c[r][1], g, head);
                rawmerge(raw, c[r][2], (2 * DP * 1024) + g * 16384 + head * 512 + l * 8);
                publish<2>(v, c[r][2], g, head);
            }
        }
        __syncthreads();
    }

    completion_tail(completion);
}

} // namespace nr_deep_qkv_3
// ============================================================================
// DEEP attention.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_attention_4 {
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
// Generic-shape derivative of the selected native arithmetic.
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
// All addresses are in the fixed physical H32W20 view, not HWC.
__device__ inline void copy16(void *dst, const void *src) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ inline uint4 shared4(const void *ptr) {
    uint4 v;
    u32 s = __cvta_generic_to_shared(ptr);
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(s));
    return v;
}
__device__ inline void astage(u8 *dst, const u8 *a, int mt, int base) {
    int l = threadIdx.x, w = threadIdx.y;
#pragma unroll
    for (int h = 0; h < 2; h++)
        copy16(dst + (w + h * 4) * 512 + l * 16,
               a + (mt * 8 + w + h * 4) * 16384 + (base / 32) * 512 + l * 16);
    asm volatile("cp.async.commit_group;" ::);
}
__device__ inline void kvstage(u8 *s, const u8 *k, const u8 *v, int head, int phase) {
    int l = threadIdx.x, w = threadIdx.y;
    if ((phase * 4 + w) * 16 < DT)
        copy16(s + w * 512 + l * 16, k + ((phase * 4 + w) * 32 + head) * 512 + l * 16);
    else
        *(uint4 *)(s + w * 512 + l * 16) = make_uint4(0, 0, 0, 0);
    copy16(s + 2048 + w * 512 + l * 16,
           v + phase * 65536 + (w / 2) * 32768 + head * 1024 + (w % 2) * 512 + l * 16);
    asm volatile("cp.async.commit_group;" ::);
}
__device__ inline void rawstore(half *dst, Frag *f, int base) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint4 v = make_uint4(f[h * 2].x, f[h * 2].y, f[h * 2 + 1].x, f[h * 2 + 1].y);
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(dst + base + h * 256), "r"(v.x),
                     "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
}
__device__ inline void rawmerge(const half *src, Frag *f, int base) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint4 v;
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(src + base + h * 256)
                     : "memory");
        f[h * 2].x = bits(__hadd2(pair(v.x), pair(f[h * 2].x)));
        f[h * 2].y = bits(__hadd2(pair(v.y), pair(f[h * 2].y)));
        f[h * 2 + 1].x = bits(__hadd2(pair(v.z), pair(f[h * 2 + 1].x)));
        f[h * 2 + 1].y = bits(__hadd2(pair(v.w), pair(f[h * 2 + 1].y)));
    }
}
// Two independent Half columns; same square/FMA and xor2,xor1,then column tree.
template <int CO> __device__ inline void normalize(Frag *f, int head, const float *scale) {
    if constexpr (CO < 2) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half2 a = pair(row ? f[0].y : f[0].x), b = pair(row ? f[1].y : f[1].x),
                  c = pair(row ? f[2].y : f[2].x), d = pair(row ? f[3].y : f[3].x);
            half2 sum = __hadd2(__hfma2(a, a, __hmul2(c, c)), __hfma2(b, b, __hmul2(d, d)));
            sum = __hadd2(sum, pair(__shfl_xor_sync(0xffffffff, bits(sum), 2)));
            sum = __hadd2(sum, pair(__shfl_xor_sync(0xffffffff, bits(sum), 1)));
            float den = __half2float(__hmax(__hadd(__low2half(sum), __high2half(sum)),
                                            __float2half(6.198883056640625e-5f))),
                  inv;
            asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(den));
            half2 factor = __half2half2(__float2half_rn(inv));
#pragma unroll
            for (int n = 0; n < 4; n++) {
                half2 v = __hmul2(pair(row ? f[n].y : f[n].x), factor);
                if constexpr (CO == 0) {
                    v = __hmul2(v, dup(5.65625f));
                    v = __hmul2(v, __half2half2(__float2half(scale[head])));
                }
                (row ? f[n].y : f[n].x) = bits(v);
            }
        }
    }
}
__device__ inline uint4 qwords(Frag *f) {
    return make_uint4(u32(qpair(pair(f[0].x))) | (u32(qpair(pair(f[1].x))) << 16),
                      u32(qpair(pair(f[0].y))) | (u32(qpair(pair(f[1].y))) << 16),
                      u32(qpair(pair(f[2].x))) | (u32(qpair(pair(f[3].x))) << 16),
                      u32(qpair(pair(f[2].y))) | (u32(qpair(pair(f[3].y))) << 16));
}
template <int CO> __device__ inline void publish(u8 *dst, Frag *f, int g, int head) {
    int l = threadIdx.x;
    if constexpr (CO < 2) {
        uint4 v = qwords(f);
        if constexpr (CO == 1) {
            u32 t = v.y;
            v.y = v.z;
            v.z = t;
        }
        *(uint4 *)(dst + g * 16384 + head * 512 + l * 16) = v;
    } else {
        // Four row bytes per dword. Pair source lanes separated by bit2; all warp lanes
        // participate.
        int src = l & ~4, j = (l >> 2) & 1,
            base = (g / 2) * 32768 + (g % 2) * 4 + head * 1024 + (src / 8) * 16 + (l & 3) * 128 +
                   j * 64;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            u32 e = u32(qpair(pair(f[n].x))) | (u32(qpair(pair(f[n].y))) << 16);
            u32 a = __shfl_sync(0xffffffff, e, src), b = __shfl_sync(0xffffffff, e, src | 4);
            *(u32 *)(dst + base + (n & 1) * 8 + (n / 2) * 512) =
                __byte_perm(a, b, j ? 0x7351 : 0x6240);
        }
    }
}
__device__ inline u32 exponent(u32 value) {
    half2 z = __hfma2(pair(value), dup(.08953857421875f), dup(1.708984375f));
    z = __hmin2(__hmax2(z, dup(1.439453125f)), dup(1.9775390625f));
    // Clamped Half bits lie in [0x3dc2,0x3fe9]; neither lane borrows.
    return ((bits(z) & 0x0fff0fffU) << 4) - 0xc000c000U;
}
// SUMV's exact pair -> four-pair sequential group -> four-lane sequential
// total -> even/odd Half tree. Packing never reassociates additions.
__device__ inline half denominator(Frag *f, int row) {
    half2 t = __hadd2(pair(row ? f[0].y : f[0].x), pair(row ? f[1].y : f[1].x));
#pragma unroll
    for (int n = 2; n < 8; n += 2)
        t = __hadd2(t, __hadd2(pair(row ? f[n].y : f[n].x), pair(row ? f[n + 1].y : f[n + 1].x)));
    int src = threadIdx.x & ~3;
    half2 total = pair(__shfl_sync(0xffffffff, bits(t), src));
#pragma unroll
    for (int n = 1; n < 4; n++)
        total = __hadd2(total, pair(__shfl_sync(0xffffffff, bits(t), src + n)));
    return __hadd(__low2half(total), __high2half(total));
}
extern "C" __global__ void joint_attention_completion(const u8 *q, const u8 *k, const u8 *v,
                                                      const half *bias, u8 *out, const int *members,
                                                      const int *keyorder, int rows, int heads,
                                                      int vit, const int *qm, const int *km,
                                                      const int *vm, const int *om,
                                                      CompletionPublications completion) {
    __shared__ __align__(16) u8 s[2][4096];
    int l = threadIdx.x, w = threadIdx.y, head = blockIdx.x;
    uint4 query[4];
    Frag result[4][4] = {};
    half den[4][2];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int g = blockIdx.y * 16 + w * 4 + r;
        query[r] = g < (DT / 16) ? *(const uint4 *)(q + 16384 * g + 512 * head + l * 16)
                                 : make_uint4(0, 0, 0, 0);
        den[r][0] = den[r][1] = hh(0);
    }
    kvstage(s[0], k, v, head, 0);
#pragma unroll 1
    for (int phase = 0; phase < ((DT + 63) / 64); phase++) {
        int page = phase & 1;
        asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if (phase + 1 < ((DT + 63) / 64))
            kvstage(s[page ^ 1], k, v, head, phase + 1);
        Frag score[4][8] = {};
#pragma unroll
        for (int n16 = 0; n16 < 4; n16++) {
            uint4 b = shared4(s[page] + n16 * 512 + l * 16);
#pragma unroll
            for (int r = 0; r < 4; r++) {
                mma(score[r][n16 * 2], (u32 *)&query[r], b.x, b.y);
                mma(score[r][n16 * 2 + 1], (u32 *)&query[r], b.z, b.w);
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int n = 0; n < 8; n++) {
                score[r][n].x = exponent(score[r][n].x);
                score[r][n].y = exponent(score[r][n].y);
            }
#pragma unroll
            for (int row = 0; row < 2; row++)
                den[r][row] = __hadd(den[r][row], denominator(score[r], row));
            u32 packed[8];
#pragma unroll
            for (int n = 0; n < 8; n++)
                packed[n] =
                    u32(qpair(pair(score[r][n].x))) | (u32(qpair(pair(score[r][n].y))) << 16);
#pragma unroll
            for (int kp = 0; kp < 64; kp += 32) {
                u32 ap[4];
#pragma unroll
                for (int z = 0; z < 4; z++)
                    ap[z] =
                        __byte_perm(packed[kp / 8 + 2 * (z / 2)], packed[kp / 8 + 2 * (z / 2) + 1],
                                    (z % 2) ? 0x7632 : 0x5410);
#pragma unroll
                for (int n16 = 0; n16 < 2; n16++) {
                    uint4 b = shared4(s[page] + 2048 + (kp / 32) * 1024 + n16 * 512 + l * 16);
                    mma(result[r][n16 * 2], ap, b.x, b.y);
                    mma(result[r][n16 * 2 + 1], ap, b.z, b.w);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int row = 0; row < 2; row++) {
            half d =
                __hadd(den[r][row], __float2half(-(((rows + 63) / 64) * 64 - rows) * .083984375f));
            float f = __half2float(__hmax(d, __float2half(6.198883056640625e-5f))), inv;
            asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(f));
            half2 recip = __half2half2(__float2half(inv));
#pragma unroll
            for (int n = 0; n < 4; n++)
                (row ? result[r][n].y : result[r][n].x) =
                    bits(__hmul2(pair(row ? result[r][n].y : result[r][n].x), recip));
        }
        int g = blockIdx.y * 16 + w * 4 + r;
        if (g < (DT / 16))
            publish<0>(out, result[r], g, head);
    }

    completion_tail(completion);
}

} // namespace nr_deep_attention_4
