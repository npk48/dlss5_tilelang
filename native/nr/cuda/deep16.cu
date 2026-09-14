// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// DEEP sixteen.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_sixteen_0 {
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
__device__ __forceinline__ void cp16(void *dst, const void *src, bool valid) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp4(void *dst, const void *src, bool valid) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(s), "l"(src),
                 "r"(valid ? 4 : 0));
}
// Fixed-v7 native spatial tiles, original unmodified 128-bit weight packets.
__device__ __forceinline__ uint4 ld128(const void *p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ void st128(void *p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w));
}
__device__ __forceinline__ void dot(Frag *c, u32 *a, const u8 *w, int off, int count) {
#pragma unroll
    for (int n = 0; n < count; n++) {
        uint4 b = ld128(w + off + 512 * n);
        mma(c[2 * n], a, b.x, b.y);
        mma(c[2 * n + 1], a, b.z, b.w);
    }
}
// One original B128 packet lives across every M16 fragment, not one dot call.
template <int T, int Count>
__device__ __forceinline__ void dot_rows(Frag (&c)[T][Count * 2], const u8 *sh, int t0, int k,
                                         const u8 *w, int off, int l) {
    u32 av[T][4];
#pragma unroll
    for (int t = 0; t < T; t++) {
        uint4 v = *(const uint4 *)(sh + 1024 * (t0 + t) + 512 * k + 16 * l);
        av[t][0] = v.x;
        av[t][1] = v.y;
        av[t][2] = v.z;
        av[t][3] = v.w;
    }
#pragma unroll
    for (int n = 0; n < Count; n++) {
        uint4 b = ld128(w + off + 512 * n);
#pragma unroll
        for (int t = 0; t < T; t++) {
            mma(c[t][2 * n], av[t], b.x, b.y);
            mma(c[t][2 * n + 1], av[t], b.z, b.w);
        }
    }
}
__device__ __forceinline__ int tok(int r) {
    return ((r >> 3) & 1) + 2 * (r & 3) + 8 * ((r >> 2) & 1);
}
__device__ __forceinline__ int loc(int r, int n) {
    return (n & 1) + ((n & 6) << 3) + ((n & 8) >> 2) + ((n & 16) >> 1) + ((n & 32) << 4) +
           ((tok(r) & 1) << 2) + ((tok(r) & 14) << 5);
}
__device__ __forceinline__ int yy(int sx, int t, int r) {
    return 8 * sx + 4 * (t & 1) + r / 4;
}
__device__ __forceinline__ int xx(int sy, int t, int r) {
    return 8 * sy + 4 * (t / 2) + r % 4;
}
__device__ __forceinline__ int idof(int sx, int sy, int t, int r, int decmode) {
    int y = yy(sx, t, r), x = xx(sy, t, r);
    return decmode ? 16 * ((DH / 4) * (2 * sy + t / 2) + 2 * sx + (t & 1)) + tok(r) : y * DW + x;
}
__device__ __forceinline__ int firstoff(int y, int x, int n) {
    int pixel = (x / 4 * 4 + y % 4) * DH + (y / 4 * 4 + x % 4);
    return pixel * 16 + (n / 16) * (DH * DW * 16) + (n & 1) + ((n & 6) << 1) + ((n & 8) >> 2);
}
__device__ __forceinline__ int rawbase(int sx, int sy, int t, int g) {
    return 8192 * ((DH / 4) * (2 * sy + t / 2) + 2 * sx + (t & 1)) + 1024 * g;
}
__device__ __forceinline__ int akoff(int row, int k) {
    return (row & 7) * 64 + (row / 8) * 4 + (k & 3) + ((k & 12) << 2) + ((k & 16) >> 1);
}
// Literal expansion/down B128 K identities are natural K32; Half pairs survive.
__device__ __forceinline__ int rhoi(int k) {
    return (k & ~14) | ((k & 8) >> 2) | ((k & 2) << 1) | ((k & 4) << 1);
}
__device__ __forceinline__ int expcol(int n8, int paircol) {
    int v = n8 * 8 + paircol;
    return (v & 1) + ((v & 6) << 1) + ((v & 8) >> 2) + (v & 16);
}
__device__ __forceinline__ void input_page(u8 *sh, const u8 *a, int sx, int sy, int p, int warps,
                                           bool first) {
    int l = threadIdx.x, w = threadIdx.y;
    if (first) {
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int j = 32 * w + l + 128 * u, yp = 8 * sx + ((j >> 4) & 3) + 4 * ((j >> 8) & 1),
                xp = 8 * sy + ((j >> 6) & 1) + 2 * (j & 1) + 4 * ((j >> 9) & 1),
                c16 = 4 * p + ((j >> 1) & 1) + 2 * ((j >> 7) & 1);
            bool valid = yp < DH && xp < DW;
            cp4(sh + 4 * j, a + (valid ? 16 * ((DW * c16 + xp) * DH + yp) + 4 * ((l >> 2) & 3) : 0),
                valid);
        }
    } else {
        for (int j = w; j < 8; j += warps) {
            int t = j / 2, k = j & 1;
            bool valid = 2 * sy + t / 2 < (DW / 4) && 2 * sx + (t & 1) < (DH / 4);
            cp16(sh + 512 * j + 16 * l, a + (valid ? rawbase(sx, sy, t, p) + 512 * k + 16 * l : 0),
                 valid);
        }
    }
}
__device__ __forceinline__ void packtile(const Frag *c, u8 *sh, int l) {
#pragma unroll
    for (int n = 0; n < 8; n++) {
        *(unsigned short *)(sh + loc(l / 4, n * 8 + 2 * (l & 3))) = qpair(pair(c[n].x));
        *(unsigned short *)(sh + loc(l / 4 + 8, n * 8 + 2 * (l & 3))) = qpair(pair(c[n].y));
    }
}
__device__ __forceinline__ void publish(const u8 *packed, u8 *raw, u8 *canonical, int sx, int sy,
                                        int t, int g, int decoder, int l) {
    bool valid = 2 * sy + t / 2 < (DW / 4) && 2 * sx + (t & 1) < (DH / 4);
    if (valid) {
        st128(raw + rawbase(sx, sy, t, g) + 16 * l, *(const uint4 *)(packed + 16 * l));
        st128(raw + rawbase(sx, sy, t, g) + 512 + 16 * l, *(const uint4 *)(packed + 512 + 16 * l));
    }
    if (canonical) {
#pragma unroll
        for (int n = 0; n < 8; n++)
            for (int h = 0; h < 2; h++) {
                int r = l / 4 + 8 * h, col = n * 8 + 2 * (l & 3);
                if (valid)
                    *(unsigned short *)(canonical + idof(sx, sy, t, r, decoder) * 512 + g * 64 +
                                        col) = *(const unsigned short *)(packed + loc(r, col));
            }
    }
}
template <bool First> __device__ void ffn(const u8 *a, const u8 *w, u8 *out, u8 *raw, int decoder) {
    constexpr int T = First ? 4 : 2, W = First ? 4 : 8;
    __shared__ __align__(16) u8 pages[8192];
    __shared__ __align__(16) u8 routebuf[W][T * 1536];
    int l = threadIdx.x, warp = threadIdx.y, sx = blockIdx.x, sy = blockIdx.y,
        g = 4 * blockIdx.z + (warp & 3), t0 = First ? 0 : 2 * (warp / 4);
    Frag dense[T][8] = {};
    input_page(pages, a, sx, sy, 0, W, First);
    asm volatile("cp.async.commit_group;" ::);
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        u8 *sh = pages + 4096 * (p & 1);
        asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if (p + 1 < 8) {
            input_page(pages + 4096 * ((p + 1) & 1), a, sx, sy, p + 1, W, First);
            asm volatile("cp.async.commit_group;" ::);
        }
#pragma unroll
        for (int k = 0; k < 2; k++)
            dot_rows<T, 4>(dense, sh, t0, k, w, 2048 * g + 16384 * (2 * p + k) + 16 * l, l);
        __syncthreads();
    }
    // All M tiles share each expansion/down B packet. Each hidden32 is consumed
    // immediately into its own persistent Half C, before advancing hidden index.
    u8 *da = routebuf[warp];
    u8 *ha = da + T * 1024;
#pragma unroll
    for (int t = 0; t < T; t++)
        for (int n = 0; n < 8; n++)
            for (int h = 0; h < 2; h++) {
                int row = l / 4 + 8 * h, col = n * 8 + 2 * (l & 3), k = rhoi(col), hw = k & 31;
                *(unsigned short *)(da + t * 1024 + 512 * (k / 32) + akoff(row, hw)) =
                    qpair(pair(h ? dense[t][n].y : dense[t][n].x));
            }
    __syncwarp();
    Frag total[T][8] = {};
#pragma unroll 1
    for (int i = 0; i < 8; i++) {
        Frag hidden[T][4] = {};
#pragma unroll
        for (int k = 0; k < 2; k++)
            dot_rows<T, 2>(hidden, da, 0, k, w,
                           0x40000 + 0x4000 * g + 0x400 * i + 0x2000 * k + 16 * l, l);
#pragma unroll
        for (int t = 0; t < T; t++)
            for (int n = 0; n < 4; n++)
                for (int h = 0; h < 2; h++) {
                    int col = expcol(n, 2 * (l & 3));
                    *(unsigned short *)(ha + t * 512 + akoff(l / 4 + 8 * h, col)) =
                        qpair(act2(pair(h ? hidden[t][n].y : hidden[t][n].x)));
                }
        __syncwarp();
        u32 av[T][4];
#pragma unroll
        for (int t = 0; t < T; t++) {
            uint4 v = *(uint4 *)(ha + t * 512 + 16 * l);
            av[t][0] = v.x;
            av[t][1] = v.y;
            av[t][2] = v.z;
            av[t][3] = v.w;
        }
#pragma unroll
        for (int n = 0; n < 4; n++) {
            uint4 b = ld128(w + 0x60000 + 0x4000 * g + 0x800 * i + 16 * l + 512 * n);
#pragma unroll
            for (int t = 0; t < T; t++) {
                mma(total[t][2 * n], av[t], b.x, b.y);
                mma(total[t][2 * n + 1], av[t], b.z, b.w);
            }
        }
        __syncwarp();
    }
// Dense operands are dead; reuse their space for original raw output packing.
#pragma unroll
    for (int t = 0; t < T; t++) {
        packtile(total[t], da, l);
        __syncwarp();
        publish(da, raw, out, sx, sy, t0 + t, g, decoder, l);
        __syncwarp();
    }
}
extern "C" __global__ void first25_completion(const u8 *a, const u8 *w, u8 *out, u8 *raw,
                                              int decoder, CompletionPublications completion) {
    ffn<true>(a, w, out, raw, decoder);
    completion_tail(completion);
}
extern "C" __global__ void chained_completion(const u8 *a, const u8 *w, u8 *out, u8 *raw,
                                              int decoder, CompletionPublications completion) {
    ffn<false>(a, w, out, raw, decoder);
    completion_tail(completion);
}

template <int Variant>
__device__ void projection(const u8 *a, const u8 *w, const u8 *skip, half *halfout, u8 *quant,
                           u8 *raw, u8 *pooled, u8 *poolraw, int decoder) {
    constexpr int W = Variant == 1 ? 8 : 4, T = Variant == 1 ? 2 : 4;
    union Workspace {
        u8 pages[12288];
        half keep[Variant == 3 ? W : 1][Variant == 3 ? 4096 : 1];
    };
    __shared__ __align__(16) Workspace workspace;
    u8 *pages = workspace.pages;
    __shared__ __align__(16) u8 packed[W][4096];
    auto &keep = workspace.keep;
    int l = threadIdx.x, warp = threadIdx.y, sx = blockIdx.x % SH, sy = blockIdx.y,
        g = 4 * (blockIdx.x / SH) + (warp & 3), t0 = Variant == 1 ? 2 * (warp / 4) : 0;
    Frag c[T][8] = {};
#pragma unroll
    for (int t = 0; t < T; t++)
        for (int n = 0; n < 8; n++)
            for (int h = 0; h < 2; h++) {
                int r = l / 4 + 8 * h, col = n * 8 + 2 * (l & 3), y = yy(sx, t0 + t, r),
                    x = xx(sy, t0 + t, r);
                unsigned short q = 0;
                if (y < DH && x < DW) {
                    int off = Variant == 0 ? firstoff(y, x, g * 64 + col)
                                           : rawbase(sx, sy, t0 + t, g) + loc(r, col);
                    u32 word = *(const u32 *)(skip + (off & ~3));
                    q = (word >> ((off & 3) * 8)) & 65535;
                }
                half2 v = __halves2half2(dec(q & 255), dec(q >> 8));
                u32 value = bits(__hmul2(v, *(const half2 *)(w + 0x40000 + 2 * (g * 64 + col))));
                if (h)
                    c[t][n].y = value;
                else
                    c[t][n].x = value;
            }
    input_page(pages, a, sx, sy, 0, W, false);
    asm volatile("cp.async.commit_group;" ::);
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        u8 *sh = pages + 4096 * (p % 3);
        asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if (p + 1 < 8) {
            input_page(pages + 4096 * ((p + 1) % 3), a, sx, sy, p + 1, W, false);
            asm volatile("cp.async.commit_group;" ::);
        }
#pragma unroll
        for (int k = 0; k < 2; k++)
            dot_rows<T, 4>(c, sh, t0, k, w, 2048 * g + 16384 * (2 * p + k) + 16 * l, l);
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < T; t++) {
        packtile(c[t], packed[warp] + 1024 * t, l);
#pragma unroll
        for (int n = 0; n < 8; n++)
            for (int h = 0; h < 2; h++) {
                int r = l / 4 + 8 * h, col = n * 8 + 2 * (l & 3);
                u32 v = h ? c[t][n].y : c[t][n].x;
                if (Variant == 3)
                    *(u32 *)(keep[warp] + (16 * t + r) * 64 + col) = v;
                if (halfout && yy(sx, t0 + t, r) < DH && xx(sy, t0 + t, r) < DW)
                    *(u32 *)(halfout + idof(sx, sy, t0 + t, r, decoder) * 512 + g * 64 + col) = v;
            }
    }
    __syncwarp();
    if (Variant != 4) {
        for (int t = 0; t < T; t++)
            publish(packed[warp] + 1024 * t, raw, quant, sx, sy, t0 + t, g, decoder, l);
    } else {
        // Exact rawfirst outview word ownership: 32 predicated dwords per lane.
        for (int aa = 0; aa < 4; aa++)
            for (int h = 0; h < 4; h++)
                for (int v = 0; v < 2; v++) {
                    int xp = 8 * sy + l / 16 + 2 * h, yp = 8 * sx + (l / 4) % 4 + 4 * v;
                    int y = (xp % 4), x = (yp % 4), t = v + 2 * ((xp % 8) / 4), r = y * 4 + x,
                        col = aa * 16 + 2 * (l % 4);
                    u32 lo = *(unsigned short *)(packed[warp] + 1024 * t + loc(r, col)),
                        hi = *(unsigned short *)(packed[warp] + 1024 * t + loc(r, col + 8));
                    if (xp < DW && yp < DH) {
                        int off = 16 * ((DW * (4 * g + aa) + xp) * DH + yp) + 4 * (l % 4);
                        *(u32 *)(raw + off) = lo | (hi << 16);
                    }
                }
        if (quant)
            for (int t = 0; t < T; t++)
                for (int n = 0; n < 8; n++)
                    for (int h = 0; h < 2; h++) {
                        int r = l / 4 + 8 * h, col = n * 8 + 2 * (l & 3);
                        if (yy(sx, t, r) < DH && xx(sy, t, r) < DW)
                            *(unsigned short *)(quant + idof(sx, sy, t, r, decoder) * 512 + g * 64 +
                                                col) =
                                *(unsigned short *)(packed[warp] + 1024 * t + loc(r, col));
                    }
    }
    if (Variant == 3) {
        // Each lane pair owns a channel pair of one pooled row. Preserve the Half C tree.
        for (int p = 0; p < 16; p++) {
            int py = p / 4, px = p % 4;
            {
                int cc = 2 * l, y = 2 * py, x = 2 * px, t = (y / 4) + 2 * (x / 4),
                    r = (y % 4) * 4 + x % 4;
                half *src = keep[warp] + (16 * t + r) * 64 + cc;
                half2 sum = __hadd2(__hadd2(*(half2 *)src, *(half2 *)(src + 64)),
                                    __hadd2(*(half2 *)(src + 256), *(half2 *)(src + 320)));
                unsigned short z = qpair(__hmul2(sum, dup(.25f)));
                int token =
                    ((px >> 1) & 1) + ((px & 1) << 1) + (((py >> 1) & 1) << 2) + ((py & 1) << 3);
                int local = (cc & 1) + ((cc & 6) << 3) + ((cc & 8) >> 2) + ((cc & 16) >> 1) +
                            ((cc & 32) << 4) + ((token & 1) << 2) + ((token & 14) << 5);
                *(unsigned short *)(packed[warp] + local) = z;
                if (pooled)
                    *(unsigned short *)(pooled + ((4 * sx + py) * PW + 4 * sy + px) * 512 + 64 * g +
                                        cc) = z;
            }
        }
        __syncwarp();
        st128(poolraw + 8192 * (SH * sy + sx) + 1024 * g + 16 * l,
              *(uint4 *)(packed[warp] + 16 * l));
        st128(poolraw + 8192 * (SH * sy + sx) + 1024 * g + 512 + 16 * l,
              *(uint4 *)(packed[warp] + 512 + 16 * l));
    }
}
#define PROJ(NAME, V)                                                                              \
    extern "C" __global__ void NAME(const u8 *a, const u8 *w, const u8 *s, half *h, u8 *q, u8 *r,  \
                                    u8 *p, u8 *pr, int d) {                                        \
        projection<V>(a, w, s, h, q, r, p, pr, d);                                                 \
    }
extern "C" __global__ void special26_completion(const u8 *a, const u8 *w, const u8 *s, half *h,
                                                u8 *q, u8 *r, u8 *p, u8 *pr, int d,
                                                CompletionPublications completion) {
    projection<0>(a, w, s, h, q, r, p, pr, d);
    completion_tail(completion);
}
extern "C" __global__ void project8_completion(const u8 *a, const u8 *w, const u8 *s, half *h,
                                               u8 *q, u8 *r, u8 *p, u8 *pr, int d,
                                               CompletionPublications completion) {
    projection<1>(a, w, s, h, q, r, p, pr, d);
    completion_tail(completion);
}
extern "C" __global__ void project4_completion(const u8 *a, const u8 *w, const u8 *s, half *h,
                                               u8 *q, u8 *r, u8 *p, u8 *pr, int d,
                                               CompletionPublications completion) {
    projection<2>(a, w, s, h, q, r, p, pr, d);
    completion_tail(completion);
}
PROJ(pool56, 3)
PROJ(out132, 4)

} // namespace nr_deep_sixteen_0
#undef PROJ
// ============================================================================
// DEEP local.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_deep_local_1 {
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
// Same two independent Half columns, xor2 then xor1 then adjacent column.
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
                half2 z = __hmul2(pair(row ? f[n].y : f[n].x), factor);
                if constexpr (CO == 0)
                    z = __hmul2(z, __half2half2(__float2half(scale[head])));
                (row ? f[n].y : f[n].x) = bits(z);
            }
        }
    }
}
__device__ inline int token(int r) {
    return ((r >> 3) & 1) + 2 * (r & 3) + 8 * ((r >> 2) & 1);
}
__device__ inline int untoken(int s) {
    return ((s & 1) << 3) + ((s & 6) >> 1) + ((s & 8) >> 1);
}
template <int CO>
__device__ inline void publish_qkv(Frag *f, u8 *dst, int t, const int *members, int head,
                                   const float *scale) {
    normalize<CO>(f, head, scale);
    int l = threadIdx.x;
    if constexpr (CO < 2) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int slot = 16 * t + token(l / 4 + 8 * h);
            bool valid = members[slot] >= 0;
#pragma unroll
            for (int n = 0; n < 2; n++) {
                u32 a = qpair(pair(h ? f[2 * n].y : f[2 * n].x)),
                    b = qpair(pair(h ? f[2 * n + 1].y : f[2 * n + 1].x));
                *(u32 *)(dst + slot * 32 + 16 * n + 4 * (l & 3)) = valid ? (a | (b << 16)) : 0;
            }
        }
    } else {
        // Direct final V B words. All lanes shuffle even when their member is padding.
        bool vx = members[16 * t + token(l / 4)] >= 0, vy = members[16 * t + token(l / 4 + 8)] >= 0;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            u32 e = (vx ? u32(qpair(pair(f[n].x))) : 0) |
                    ((vy ? u32(qpair(pair(f[n].y))) : 0) << 16),
                v[4];
#pragma unroll
            for (int b = 0; b < 4; b++) {
                int r = untoken(4 * (l & 3) + ((b & 1) << 1) + ((b & 2) >> 1));
                u32 s = __shfl_sync(0xffffffff, e, 4 * (r & 7) + l / 8);
                v[b] = (s >> (8 * (2 * (r / 8) + ((l / 4) & 1)))) & 255;
            }
            *(u32 *)(dst + t * 512 + n * 128 + l * 4) =
                v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24);
        }
    }
}
__device__ inline u32 exponent(u32 x) {
    half2 z = __hfma2(pair(x), dup(.044921875f), dup(1.30078125f));
    z = __hmin2(__hmax2(z, dup(1.03125f)), dup(1.5693359375f));
    // Clamp bits 0x3c20..0x3e47. After mask/shift each half fits: no cross-half carry.
    return ((bits(z) & 0x07ff07ffU) << 5) ^ 0x80008000U;
}
__device__ inline half denominator(Frag *f, int row) {
    half2 sums;
    half e =
        __hadd(__low2half(pair(row ? f[0].y : f[0].x)), __high2half(pair(row ? f[0].y : f[0].x)));
    half o =
        __hadd(__low2half(pair(row ? f[1].y : f[1].x)), __high2half(pair(row ? f[1].y : f[1].x)));
#pragma unroll
    for (int n = 2; n < 8; n += 2) {
        half2 a = pair(row ? f[n].y : f[n].x), b = pair(row ? f[n + 1].y : f[n + 1].x);
        e = __hadd(e, __hadd(__low2half(a), __high2half(a)));
        o = __hadd(o, __hadd(__low2half(b), __high2half(b)));
    }
    sums = __halves2half2(e, o);
    int src = threadIdx.x & ~3;
    u32 s[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
        s[j] = __shfl_sync(0xffffffff, bits(sums), src + j);
    half2 total = pair((s[0] & 65535) | (s[1] << 16));
    total = __hadd2(total, pair((s[2] & 65535) | (s[3] << 16)));
    total = __hadd2(total, pair((s[0] >> 16) | (s[1] & 0xffff0000U)));
    total = __hadd2(total, pair((s[2] >> 16) | (s[3] & 0xffff0000U)));
    return __hadd(__low2half(total), __high2half(total));
}
template <int P> __device__ inline void pv(Frag *result, const Frag *q, const u8 *v) {
    int l = threadIdx.x;
    u32 a[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int n = 4 * P + 2 * (r / 2);
        u32 x = (r & 1) ? q[n].y : q[n].x, y = (r & 1) ? q[n + 1].y : q[n + 1].x;
        int src = (l & ~3) + 2 * (l & 1);
        u32 lo0 = __shfl_sync(0xffffffff, x, src), hi0 = __shfl_sync(0xffffffff, x, src | 1);
        u32 lo1 = __shfl_sync(0xffffffff, y, src), hi1 = __shfl_sync(0xffffffff, y, src | 1);
        // Choose N after the shuffle: source bit1 differs from destination bit1.
        a[r] = __byte_perm((l & 2) ? lo1 : lo0, (l & 2) ? hi1 : hi0, 0x5140);
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        mma(result[n], a, *(const u32 *)(v + P * 1024 + n * 128 + l * 4),
            *(const u32 *)(v + P * 1024 + 512 + n * 128 + l * 4));
}
__device__ inline void query16(const u8 *q, const u8 *k, const u8 *v, const half *bias, u8 *output,
                               int head, int base) {
    int l = threadIdx.x;
    Frag score[8] = {}, result[4] = {};
    u32 a[4];
#pragma unroll
    for (int r = 0; r < 4; r++)
        a[r] = *(const u32 *)(q + (base + l / 4 + 8 * (r & 1)) * 32 + 16 * (r / 2) + 4 * (l & 3));
#pragma unroll
    for (int n = 0; n < 8; n++) {
        const int qb[6] = {1, 5, 6, 7, 10, 11}, kb[6] = {2, 0, 3, 4, 8, 9};
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int off =
                head * 4096 + dep(base + l / 4 + 8 * h, qb, 6) + dep(n * 8 + 2 * (l & 3), kb, 6);
            (h ? score[n].y : score[n].x) = hb(bias[off]) | (hb(bias[off + 4]) << 16);
        }
        mma(score[n], a, *(const u32 *)(k + (n * 8 + l / 4) * 32 + 4 * (l & 3)),
            *(const u32 *)(k + (n * 8 + l / 4) * 32 + 16 + 4 * (l & 3)));
        score[n].x = exponent(score[n].x);
        score[n].y = exponent(score[n].y);
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
        half den = __hadd(hh(0), denominator(score, h));
        float d = __half2float(__hmax(den, __float2half(6.198883056640625e-5f))), inv;
        asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(d));
        half2 recip = __half2half2(__float2half(inv));
#pragma unroll
        for (int n = 0; n < 8; n++)
            (h ? score[n].y : score[n].x) =
                qpair(__hmul2(pair(h ? score[n].y : score[n].x), recip));
    }
    pv<0>(result, score, v);
    pv<1>(result, score, v);
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int h = 0; h < 2; h++) {
            int slot = base + l / 4 + 8 * h, col = n * 8 + 2 * (l & 3), tok = slot % 16;
            int off = (col & 1) + ((col & 6) << 3) + ((col & 8) >> 2) + ((col & 16) >> 1) +
                      ((tok & 1) << 2) + ((tok & 14) << 5) + 512 * (slot / 16);
            *(unsigned short *)(output + off) = qpair(pair(h ? result[n].y : result[n].x));
        }
    __syncwarp();
}
// One physical M16 representation. Producer, exponent, V and quant primitives inherited.
__device__ inline half physical_denominator(Frag *f, int row) {
    half2 sums = __hadd2(pair(row ? f[0].y : f[0].x), pair(row ? f[1].y : f[1].x));
#pragma unroll
    for (int n = 2; n < 8; n += 2)
        sums = __hadd2(sums,
                       __hadd2(pair(row ? f[n].y : f[n].x), pair(row ? f[n + 1].y : f[n + 1].x)));
    int src = threadIdx.x & ~3;
    u32 s[4];
#pragma unroll
    for (int g = 0; g < 4; g++)
        s[g] = __shfl_sync(0xffffffff, bits(sums), src + g);
    half2 total = __hadd2(__hadd2(__hadd2(pair(s[0]), pair(s[1])), pair(s[2])), pair(s[3]));
    return __hadd(__low2half(total), __high2half(total));
}
template <int P> __device__ inline void physical_pv(Frag *result, const Frag *q, const u8 *v) {
    int l = threadIdx.x;
    u32 a[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int n = 4 * P + 2 * (r / 2);
        u32 x = (r & 1) ? q[n].y : q[n].x, y = (r & 1) ? q[n + 1].y : q[n + 1].x;
        a[r] = (x & 0xffffU) | ((y & 0xffffU) << 16);
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        mma(result[n], a, *(const u32 *)(v + P * 1024 + n * 128 + l * 4),
            *(const u32 *)(v + P * 1024 + 512 + n * 128 + l * 4));
}
__device__ inline void physical_query16(const u8 *q, const u8 *k, const u8 *v, const half *bias,
                                        u8 *output, int head, int base) {
    int l = threadIdx.x;
    Frag score[8] = {}, result[4] = {};
    u32 a[4];
#pragma unroll
    for (int r = 0; r < 4; r++)
        a[r] = *(const u32 *)(q + (base + 2 * (l / 4) + (r & 1)) * 32 + 16 * (r / 2) + 4 * (l & 3));
#pragma unroll
    for (int j = 0; j < 4; j++) {
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(score[2 * j].x), "=r"(score[2 * j].y), "=r"(score[2 * j + 1].x),
                       "=r"(score[2 * j + 1].y)
                     : "l"((const u8 *)bias + head * 8192 + (base / 16) * 2048 + j * 512 + l * 16));
#pragma unroll
        for (int b = 0; b < 2; b++) {
            int n = 2 * j + b;
            mma(score[n], a,
                *(const u32 *)(k + (16 * (n / 2) + 2 * (l / 4) + (n & 1)) * 32 + 4 * (l & 3)),
                *(const u32 *)(k + (16 * (n / 2) + 2 * (l / 4) + (n & 1)) * 32 + 16 + 4 * (l & 3)));
            score[n].x = exponent(score[n].x);
            score[n].y = exponent(score[n].y);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
        half den = __hadd(hh(0), physical_denominator(score, h));
        float d = __half2float(__hmax(den, __float2half(6.198883056640625e-5f))), inv;
        asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(inv) : "f"(d));
        half2 recip = __half2half2(__float2half(inv));
#pragma unroll
        for (int n = 0; n < 8; n++)
            (h ? score[n].y : score[n].x) =
                qpair(__hmul2(pair(h ? score[n].y : score[n].x), recip));
    }
    physical_pv<0>(result, score, v);
    physical_pv<1>(result, score, v);
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int h = 0; h < 2; h++) {
            int slot = base + 2 * (l / 4) + h, col = n * 8 + 2 * (l & 3), tok = slot % 16;
            int off = (col & 1) + ((col & 6) << 3) + ((col & 8) >> 2) + ((col & 16) >> 1) +
                      ((tok & 1) << 2) + ((tok & 14) << 5) + 512 * (slot / 16);
            *(unsigned short *)(output + off) = qpair(pair(h ? result[n].y : result[n].x));
        }
    __syncwarp();
}
__device__ __forceinline__ void cp16(void *dst, const void *src, bool valid) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp4(void *dst, const void *src, bool valid) {
    u32 s = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(s), "l"(src),
                 "r"(valid ? 4 : 0));
}
// Fixed-v7 native spatial tiles, original unmodified 128-bit weight packets.
__device__ __forceinline__ uint4 ld128(const void *p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ void st128(void *p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w));
}
__device__ __forceinline__ void dot(Frag *c, u32 *a, const u8 *w, int off, int count) {
#pragma unroll
    for (int n = 0; n < count; n++) {
        uint4 b = ld128(w + off + 512 * n);
        mma(c[2 * n], a, b.x, b.y);
        mma(c[2 * n + 1], a, b.z, b.w);
    }
}
// One original B128 packet lives across every M16 fragment, not one dot call.
template <int T, int Count>
__device__ __forceinline__ void dot_rows(Frag (&c)[T][Count * 2], const u8 *sh, int t0, int k,
                                         const u8 *w, int off, int l) {
    u32 av[T][4];
#pragma unroll
    for (int t = 0; t < T; t++) {
        uint4 v = *(const uint4 *)(sh + 1024 * (t0 + t) + 512 * k + 16 * l);
        av[t][0] = v.x;
        av[t][1] = v.y;
        av[t][2] = v.z;
        av[t][3] = v.w;
    }
#pragma unroll
    for (int n = 0; n < Count; n++) {
        uint4 b = ld128(w + off + 512 * n);
#pragma unroll
        for (int t = 0; t < T; t++) {
            mma(c[t][2 * n], av[t], b.x, b.y);
            mma(c[t][2 * n + 1], av[t], b.z, b.w);
        }
    }
}
__device__ __forceinline__ void local_page(u8 *sh, const u8 *a, const int *plan, int group, int p,
                                           int warp, int l) {
    for (int j = warp; j < 8; j += 4) {
        int off = plan[(((group * 8 + p) * 4 + j / 2) * 2 + j % 2) * 32 + l];
        cp16(sh + 512 * j + 16 * l, a + (off < 0 ? 0 : off), off >= 0);
    }
    asm volatile("cp.async.commit_group;" ::);
}
extern "C" __global__ void joint_local64_completion(const u8 *a, const int *plan, const u8 *w,
                                                    u8 *out, u8 *raw, const int *omap,
                                                    const int *members,
                                                    CompletionPublications completion) {
    __shared__ __align__(16) u8 pages[8192];
    __shared__ __align__(16) u8 qkv[4][3][2048];
    int l = threadIdx.x, warp = threadIdx.y, head = 4 * blockIdx.z + warp,
        group = blockIdx.y * gridDim.x + blockIdx.x;
    Frag c[4][12] = {};
    local_page(pages, a, plan, group, 0, warp, l);
#pragma unroll 1
    for (int p = 0; p < 8; p++) {
        u8 *sh = pages + 4096 * (p & 1);
        asm volatile("cp.async.wait_group 0;" ::);
        __syncthreads();
        if (p + 1 < 8)
            local_page(pages + 4096 * ((p + 1) & 1), a, plan, group, p + 1, warp, l);
#pragma unroll
        for (int k = 0; k < 2; k++)
            dot_rows<4, 6>(c, sh, 0, k, w, 3072 * head + 49152 * (2 * p + k) + 16 * l, l);
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
        publish_qkv<0>(c[t], qkv[warp][0], t, members + group * 64, head,
                       (const float *)(w + 0xe0000));
        publish_qkv<1>(c[t] + 4, qkv[warp][1], t, members + group * 64, head,
                       (const float *)(w + 0xe0000));
        publish_qkv<2>(c[t] + 8, qkv[warp][2], t, members + group * 64, head,
                       (const float *)(w + 0xe0000));
    }
    __syncwarp();
    u8 *packed = pages + 2048 * warp;
    for (int t = 0; t < 4; t++)
        physical_query16(qkv[warp][0], qkv[warp][1], qkv[warp][2], (const half *)(w + 0xc0000),
                         packed, head, 16 * t);
    for (int t = 0; t < 4; t++) {
        int off = plan[((group * 8 * 4 + t) * 2) * 32 + l];
        if (off >= 0)
            st128(raw + off + 512 * head, *(uint4 *)(packed + 512 * t + 16 * l));
        for (int n = 0; n < 4; n++)
            for (int h = 0; h < 2; h++) {
                int slot = 16 * t + l / 4 + 8 * h, id = members[group * 64 + slot],
                    col = n * 8 + 2 * (l & 3), token = slot % 16;
                int ix = (col & 1) + ((col & 6) << 3) + ((col & 8) >> 2) + ((col & 16) >> 1) +
                         ((token & 1) << 2) + ((token & 14) << 5) + 512 * t;
                if (out && id >= 0)
                    *(unsigned short *)(out + (id * 16 + head) * 32 + col) =
                        *(unsigned short *)(packed + ix);
            }
    }

    completion_tail(completion);
}

} // namespace nr_deep_local_1
