// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// ============================================================================
// OUTER shallow_static_geometry/one.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_one_0 {
// Experimental shared arithmetic component. Not a qualified native endpoint.
// All Tile storage is distributed over exactly one warp in MMA C coordinates.
namespace endpoint {
using u32 = unsigned;
using u8 = unsigned char;
__device__ __forceinline__ int lane() {
    return threadIdx.x & 31;
}
__device__ __forceinline__ half h(float x) {
    return __float2half_rn(x);
}
__device__ __forceinline__ float f(half x) {
    return __half2float(x);
}
__device__ __forceinline__ u32 pack(half a, half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}
__device__ __forceinline__ half unpack(u32 x, int i) {
    return __ushort_as_half(x >> (16 * i));
}
__device__ __forceinline__ u8 e4(half x) {
    return __nv_cvt_halfraw_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half une4(u8 x) {
    return __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
}
__device__ __forceinline__ half q(half x) {
    return une4(e4(x));
}
__device__ __forceinline__ half activate(half x) {
    float c = fminf(4.f, fmaxf(-4.f, f(x)));
    half g = h(__fmaf_rn(fabsf(c), -.055908203125f, .447265625f));
    return __hmul(x, h(__fmaf_rn(c, f(g), .89453125f)));
}
__device__ __forceinline__ half exponent(half x) {
    half a = h(__fmaf_rn(f(x), .044921875f, 1.30078125f));
    a = h(fminf(1.5693359375f, fmaxf(1.03125f, f(a))));
    return __ushort_as_half((unsigned(__half_as_ushort(a)) << 5) + 32768);
}
__device__ __forceinline__ half reciprocal(half x) {
    float z;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
__device__ __forceinline__ half rsqrt_half(half x) {
    float z;
    asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
struct Fragment {
    u32 v[2];
};
__device__ __forceinline__ void mma8(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma16(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int M, int N> struct Tile {
    static_assert(M % 16 == 0 && N % 8 == 0, "MMA tile dimensions");
    Fragment c[M / 16][N / 8];
    // Collective arbitrary gather. Both words of every bank are shuffled before
    // selection: selecting a dynamic word BEFORE shfl is incorrect when the
    // destination lanes request different rows. No lane-private full window.
    __device__ half get(int row, int col) const {
        u32 bits = 0;
        int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
                u32 a = __shfl_sync(0xffffffff, c[m][n].v[0], src);
                u32 b = __shfl_sync(0xffffffff, c[m][n].v[1], src);
                if (m == row / 16 && n == col / 8)
                    bits = (row & 8) ? b : a;
            }
        }
        return unpack(bits, col & 1);
    }
    template <class F> __device__ void fill(F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r = m * 16 + (lane() >> 2) + i * 8, k = n * 8 + (lane() & 3) * 2;
                    half a = fn(r, k), b = fn(r, k + 1);
                    c[m][n].v[i] = pack(a, b);
                }
            }
        }
    }
};
// Accessors return encoded E4 bytes in physical MMA K order.
template <int M, int N, class A, class B> __device__ void gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = 0;
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    a[i] |= u32(av(r, k + j)) << (j * 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
template <int M, int N, class A, class B> __device__ void gemm16(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
                half x = av(r, k), y = av(r, k + 1);
                a[i] = pack(x, y);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + (lane() >> 2);
                half x = bv(k, col), y = bv(k + 1, col);
                b[i] = pack(x, y);
            }
            mma16(c.c[m][n], a, b);
        }
    }
}
struct BodyWeights {
    // Dense decoded logical matrices, NOT a compressed/repacked static-weight
    // candidate. Host route construction must follow physical_1h and archive.
    const u8 *expand;                 // [32,128], hidden_inverse applied to each stream
    const u8 *contract;               // [4,32,32], stream,Khalf order 0/0,1/0,0/32,1/32
    const u8 *qkv;                    // [32,96], Qe Qo Ke Ko Ve Vo
    const u8 *projection;             // [32,32]
    const half *bias;                 // [64,64], physical bias half -> query/key route
    const half *ffn_gate, *attn_gate; // output-N order
    const int *cp, *rc, *oi, *ai, *pk;
    half scale;
};
__device__ half norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = z.get(row, base + j), v1 = z.get(row, base + 8 + j);
        half v2 = z.get(row, base + 16 + j), v3 = z.get(row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = e.get(row, j), b = e.get(row, j + 8);
        g[j] = __hadd(a, b);
        a = e.get(row, j + 48);
        b = e.get(row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 16);
        b = e.get(row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 32);
        b = e.get(row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ int qk_order(int k) {
    return (k / 16) * 16 + (k % 2) + ((k / 2) % 2) * 8 + ((k / 4) % 4) * 2;
}
// Complete shared 64-token arithmetic; input is unquantized Half in canonical
// channels and physical token order. Output callback consumes two 32-row slabs
// without a global latent publication. Caller supplies sampling/codec/readout.
// This function has NOT been independently oracle-qualified.
template <class Publish>
__device__ void body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill([&](int r, int c) { return __hmul(input.get(r, w.rc[c]), w.ffn_gate[c]); });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return e4(input.get(r, w.cp[k])); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
// Direct C-word transformation has no cross-lane reads/aliasing.
#pragma unroll 1
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = hidden.c[m][n].v[i];
                    hidden.c[m][n].v[i] =
                        pack(q(activate(unpack(a, 0))), q(activate(unpack(a, 1))));
                }
        gemm8(
            ff, [&](int r, int k) { return e4(hidden.get(r, k)); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    gemm8(
        z, [&](int r, int k) { return e4(ff.get(r, w.oi[k])); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = norm(z, r, 0);
            ik[m * 2 + i] = norm(z, r, 32);
        }
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n)
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                z.c[m][n].v[i] = pack(q(a), q(b));
            }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return e4(z.get(slab * 32 + r, qk_order(k))); },
            [&](int k, int n) { return e4(z.get(n, 32 + qk_order(k))); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = denominator(logits, m * 16 + (lane() >> 2) + i * 8);
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    logits.c[m][n].v[i] =
                        pack(q(__hmul(unpack(a, 0), d)), q(__hmul(unpack(a, 1), d)));
                }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return e4(logits.get(r, w.pk[part * 32 + k])); },
                [&](int k, int n) { return e4(z.get(w.pk[part * 32 + k], 64 + n)); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) { return __hmul(ff.get(slab * 32 + r, n), w.attn_gate[n]); });
        gemm8(
            out, [&](int r, int k) { return e4(attended.get(r, w.ai[k])); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// Call sites have warp-uniform row/16 and row/8. Channel routes may vary
// between lanes, so only that bank dimension is scanned before selection.
template <int M, int N> __device__ half packed_half_row(const Tile<M, N> &t, int row, int col) {
    u32 bits = 0;
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll
    for (int n = 0; n < N / 8; ++n) {
        u32 x = __shfl_sync(0xffffffff, t.c[row / 16][n].v[(row >> 3) & 1], src);
        if (n == col / 8)
            bits = x;
    }
    return unpack(bits, col & 1);
}
template <int M, int N> __device__ half packed_half_uniform(const Tile<M, N> &t, int row, int col) {
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
    return unpack(__shfl_sync(0xffffffff, t.c[row / 16][col / 8].v[(row >> 3) & 1], src), col & 1);
}
// Four encoded E4 values per C bank: row0 pair in bytes 0/1, row8
// pair in bytes 2/3. This is a copy, never the Half residual storage.
// One word shuffle selects both rows; no decode-to-Half/re-encode roundtrip.
template <int M, int N> struct PackedE4 {
    u32 c[M / 16][N / 8];
    template <class F> __device__ void encode(const Tile<M, N> &t, F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m)
            for (int n = 0; n < N / 8; ++n) {
                u32 a = t.c[m][n].v[0], b = t.c[m][n].v[1];
                c[m][n] = u32(e4(fn(unpack(a, 0)))) | (u32(e4(fn(unpack(a, 1)))) << 8) |
                          (u32(e4(fn(unpack(b, 0)))) << 16) | (u32(e4(fn(unpack(b, 1)))) << 24);
            }
    }
    __device__ u8 row(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
            u32 x = __shfl_sync(0xffffffff, c[r / 16][n], src);
            if (n == k / 8)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
    __device__ u8 col(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int m = 0; m < M / 16; ++m) {
            u32 x = __shfl_sync(0xffffffff, c[m][k / 8], src);
            if (m == r / 16)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
};
__device__ half packed_norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = packed_half_uniform(z, row, base + j),
             v1 = packed_half_uniform(z, row, base + 8 + j);
        half v2 = packed_half_uniform(z, row, base + 16 + j),
             v3 = packed_half_uniform(z, row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half packed_denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = packed_half_uniform(e, row, j), b = packed_half_uniform(e, row, j + 8);
        g[j] = __hadd(a, b);
        a = packed_half_uniform(e, row, j + 48);
        b = packed_half_uniform(e, row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 16);
        b = packed_half_uniform(e, row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 32);
        b = packed_half_uniform(e, row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
template <class Publish>
__device__ void packed_body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill(
        [&](int r, int c) { return __hmul(packed_half_row(input, r, w.rc[c]), w.ffn_gate[c]); });
    PackedE4<64, 32> input8;
    input8.encode(input, [](half x) { return x; });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return input8.row(r, w.cp[k]); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
        PackedE4<64, 32> hidden8;
        hidden8.encode(hidden, [](half x) { return activate(x); });
        gemm8(
            ff, [&](int r, int k) { return hidden8.row(r, k); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    PackedE4<64, 32> ff8;
    ff8.encode(ff, [](half x) { return x; });
    gemm8(
        z, [&](int r, int k) { return ff8.row(r, w.oi[k]); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = packed_norm(z, r, 0);
            ik[m * 2 + i] = packed_norm(z, r, 32);
        }
    PackedE4<64, 96> z8;
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n) {
            u32 word = 0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                word |= (u32(e4(a)) | (u32(e4(b)) << 8)) << (16 * i);
            }
            z8.c[m][n] = word;
        }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return z8.row(slab * 32 + r, qk_order(k)); },
            [&](int k, int n) { return z8.row(n, 32 + qk_order(k)); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = packed_denominator(logits, m * 16 + (lane() >> 2) + i * 8);
        PackedE4<32, 64> probs;
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n) {
                u32 word = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    word |=
                        (u32(e4(__hmul(unpack(a, 0), d))) | (u32(e4(__hmul(unpack(a, 1), d))) << 8))
                        << (16 * i);
                }
                probs.c[m][n] = word;
            }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return probs.row(r, w.pk[part * 32 + k]); },
                [&](int k, int n) { return z8.col(w.pk[part * 32 + k], 64 + n); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) {
            return __hmul(packed_half_row(ff, slab * 32 + r, n), w.attn_gate[n]);
        });
        PackedE4<32, 32> attended8;
        attended8.encode(attended, [](half x) { return x; });
        gemm8(
            out, [&](int r, int k) { return attended8.row(r, w.ai[k]); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// A is the same physical K packet for every N bank. Gather once per M,
// then consume it in original N order. Each accumulator sees identical MMA.
template <int M, int N, class A, class B>
__device__ void organized_gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
        u32 a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[i] |= u32(av(r, k + j)) << (j * 8);
        }
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
} // namespace endpoint
namespace endpoint {
// Lossless same-capacity B8 layout: [K/32][N/8][lane][8].
// Bytes are identical to the traced raw B8, but packets are lane-permuted
// to match the accepted decoder's N order. Not native register scheduling.
struct WeightPacket {
    const u8 *data;
    int kbase, nbase, width;
    __device__ uint2 load(int bank) const {
        return *(const uint2 *)(data +
                                ((kbase / 32 * (width / 8) + nbase / 8 + bank) * 32 + lane()) * 8);
    }
};
template <int M, int N, class A>
__device__ void organized_gemm8(Tile<M, N> &c, A av, WeightPacket bv) {
    u32 a[M / 16][4];
#pragma unroll
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[m][i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[m][i] |= u32(av(r, k + j)) << (j * 8);
        }
    }
#pragma unroll 1
    for (int n = 0; n < N / 8; ++n) {
        uint2 packet = bv.load(n);
        u32 b[2] = {packet.x, packet.y};
#pragma unroll
        for (int m = 0; m < M / 16; ++m)
            mma8(c.c[m][n], a[m], b);
    }
}
} // namespace endpoint
namespace packed12 {
using namespace endpoint;
union HalfBits {
    u32 u;
    half2 h;
    __device__ HalfBits(u32 x) : u(x) {}
    __device__ HalfBits(half2 x) : h(x) {}
};
__device__ __forceinline__ half2 hh(u32 x) {
    return HalfBits(x).h;
}
__device__ __forceinline__ u32 bits(half2 x) {
    return HalfBits(x).u;
}
__device__ __forceinline__ u32 mul(u32 a, u32 b) {
    return bits(__hmul2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 splat(half a) {
    return pack(a, a);
}
__device__ __forceinline__ u32 add(u32 a, u32 b) {
    return bits(__hadd2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 fma(u32 a, u32 b, u32 c) {
    return bits(__hfma2(hh(a), hh(b), hh(c)));
}
__device__ __forceinline__ u32 activate_pair(u32 x) {
    half2 c = __hmin2(__float2half2_rn(4.f), __hmax2(__float2half2_rn(-4.f), hh(x)));
    half2 g = __hfma2(__habs2(c), __float2half2_rn(-.055908203125f), __float2half2_rn(.447265625f));
    return bits(__hmul2(hh(x), __hfma2(c, g, __float2half2_rn(.89453125f))));
}
__device__ __forceinline__ u32 exponent_pair(u32 x) {
    half2 a = __hfma2(hh(x), __float2half2_rn(.044921875f), __float2half2_rn(1.30078125f));
    a = __hmin2(__float2half2_rn(1.5693359375f), __hmax2(__float2half2_rn(1.03125f), a));
    u32 v = bits(a);
    return (((v & 0xffffu) << 5) + 32768u) & 0xffffu | ((((v >> 16) << 5) + 32768u) << 16);
}
__device__ __forceinline__ u32 e4pair(u32 x) {
    return __nv_cvt_halfraw2_to_fp8x2(hh(x), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ u32 e4four(u32 a, u32 b) {
    return e4pair(a) | (e4pair(b) << 16);
}
struct Identity {
    __device__ u32 operator()(u32 x) const {
        return x;
    }
};
struct Activate {
    __device__ u32 operator()(u32 x) const {
        return activate_pair(x);
    }
};
} // namespace packed12
namespace packed12 {
// Each four-lane row group owns four distinct adjacent-channel partial pairs.
__device__ __forceinline__ u32 row_broadcast(u32 v, int owner) {
    return __shfl_sync(0xffffffff, v, (lane() & 28) + owner);
}
__device__ __forceinline__ half norm_finish(u32 t) {
    u32 a = add(row_broadcast(t, 0), row_broadcast(t, 2));
    u32 b = add(row_broadcast(t, 1), row_broadcast(t, 3));
    u32 s = add(a, b);
    return rsqrt_half(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
__device__ __forceinline__ half den_finish(u32 t) {
    // NOT a balanced xor tree: (((partial0+partial1)+partial2)+partial3).
    u32 s = add(add(add(row_broadcast(t, 0), row_broadcast(t, 1)), row_broadcast(t, 2)),
                row_broadcast(t, 3));
    return reciprocal(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
template <int M, int N> __device__ half norm_tile(const Tile<M, N> &z, int row, int base) {
    int m = row / 16, i = (row / 8) & 1, n = base / 8;
    u32 a = z.c[m][n].v[i], b = z.c[m][n + 1].v[i], c = z.c[m][n + 2].v[i], d = z.c[m][n + 3].v[i];
    return norm_finish(add(fma(a, a, mul(c, c)), fma(b, b, mul(d, d))));
}
__device__ half den_two(const Tile<32, 64> &z, int row) {
    int m = row / 16, i = (row / 8) & 1;
    // lane%4 = g; two halves are s=0/1. Each t contributes (16t+2g)+(16t+2g+8).
    u32 s = add(z.c[m][0].v[i], z.c[m][1].v[i]);
    s = add(s, add(z.c[m][2].v[i], z.c[m][3].v[i]));
    s = add(s, add(z.c[m][4].v[i], z.c[m][5].v[i]));
    s = add(s, add(z.c[m][6].v[i], z.c[m][7].v[i]));
    return den_finish(s);
}
} // namespace packed12
namespace activation {
using namespace endpoint;
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
template <int M, int N> using HC = Words<(M / 16) * (N / 8) * 2>;
template <int M, int N, class F> __device__ __forceinline__ HC<M, N> fill(F f) {
    HC<M, N> x;
    each<M / 16>([&](auto mt) {
        each<N / 8>([&](auto nt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, n = decltype(nt)::value,
                              i = decltype(it)::value;
                int r = m * 16 + lane() / 4 + i * 8, c = n * 8 + (lane() % 4) * 2;
                x.v[(m * (N / 8) + n) * 2 + i] = pack(f(r, c), f(r, c + 1));
            });
        });
    });
    return x;
}
template <int M, int N, class F>
__device__ __forceinline__ Words<M / 16 * (N / 8)> encode(const HC<M, N> &x, F f) {
    Words<M / 16 * (N / 8)> y;
    each<M / 16 * (N / 8)>([&](auto t) {
        constexpr int j = decltype(t)::value;
        u32 a = x.v[j * 2], b = x.v[j * 2 + 1];
        y.v[j] = packed12::e4four(f(a), f(b));
    });
    return y;
}
template <int M, int N>
__device__ __forceinline__ void weight(HC<M, N> &c, const Words<M / 4> &a, WeightPacket w) {
    each<N / 8>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        uint2 b = w.load(n);
        u32 bb[2] = {b.x, b.y};
        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma8(z, aa, bb);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// Uniform fixed bank, row owner is lane/4; norm/den do not scan any banks.
template <int M, int N, int Row, int Col>
__device__ __forceinline__ half component(const HC<M, N> &x) {
    constexpr int bank = (Row / 2 * (N / 8) + Col / 8) * 2 + Row % 2;
    return unpack(__shfl_sync(0xffffffff, x.v[bank], (lane() & 28) + (Col % 8) / 2), Col % 2);
}
template <int M, int N, int Row, int Base = 0>
__device__ __forceinline__ half norm(const HC<M, N> &x) {
    constexpr int j = (Row / 2 * (N / 8) + Base / 8) * 2 + Row % 2;
    u32 a = x.v[j], b = x.v[j + 2], c = x.v[j + 4], d = x.v[j + 6];
    return packed12::norm_finish(packed12::add(packed12::fma(a, a, packed12::mul(c, c)),
                                               packed12::fma(b, b, packed12::mul(d, d))));
}
template <int Row> __device__ __forceinline__ half denominator(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    return packed12::den_finish(s);
}
} // namespace activation

namespace packet_reference {

using u32 = unsigned;

template <int N> using Words = activation::Words<N>;

__device__ __forceinline__ u32 transpose_half2(u32 a) {
    u32 d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;" : "=r"(d) : "r"(a));
    return d;
}

// input_a: input is ALREADY prepared encoded E4, except residual_c is original Half2.

__device__ __forceinline__ Words<16> input_a(const Words<16> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s5 = __shfl_sync(0xffffffff, x.v[1],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s6 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s7 = __shfl_sync(0xffffffff, x.v[3],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[1],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s17 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s18 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s19 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s37 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s38 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s39 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s49 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s50 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s51 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    Words<16> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[1] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[3] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[5] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[7] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[9] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[11] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[13] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[15] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    return out;
}

__device__ __forceinline__ Words<32> residual_c(const Words<32> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s5 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s6 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s7 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[1],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[3],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 = __shfl_sync(0xffffffff, x.v[0],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s17 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s18 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s19 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[1],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s37 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s38 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s39 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s49 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s50 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s51 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s64 =
        __shfl_sync(0xffffffff, x.v[16],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s65 =
        __shfl_sync(0xffffffff, x.v[18],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s66 =
        __shfl_sync(0xffffffff, x.v[20],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s67 =
        __shfl_sync(0xffffffff, x.v[22],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s68 =
        __shfl_sync(0xffffffff, x.v[17],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s69 =
        __shfl_sync(0xffffffff, x.v[19],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s70 =
        __shfl_sync(0xffffffff, x.v[21],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s71 =
        __shfl_sync(0xffffffff, x.v[23],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s72 = __shfl_sync(0xffffffff, x.v[16],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s73 = __shfl_sync(0xffffffff, x.v[18],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s74 = __shfl_sync(0xffffffff, x.v[20],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s75 = __shfl_sync(0xffffffff, x.v[22],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s76 = __shfl_sync(0xffffffff, x.v[17],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s77 = __shfl_sync(0xffffffff, x.v[19],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s78 = __shfl_sync(0xffffffff, x.v[21],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s79 = __shfl_sync(0xffffffff, x.v[23],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s80 = __shfl_sync(0xffffffff, x.v[16],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s81 = __shfl_sync(0xffffffff, x.v[18],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s82 = __shfl_sync(0xffffffff, x.v[20],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s83 = __shfl_sync(0xffffffff, x.v[22],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s84 = __shfl_sync(0xffffffff, x.v[17],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s85 = __shfl_sync(0xffffffff, x.v[19],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s86 = __shfl_sync(0xffffffff, x.v[21],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s87 = __shfl_sync(0xffffffff, x.v[23],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s88 = __shfl_sync(0xffffffff, x.v[16],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s89 = __shfl_sync(0xffffffff, x.v[18],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s90 = __shfl_sync(0xffffffff, x.v[20],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s91 = __shfl_sync(0xffffffff, x.v[22],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s92 = __shfl_sync(0xffffffff, x.v[17],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s93 = __shfl_sync(0xffffffff, x.v[19],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s94 = __shfl_sync(0xffffffff, x.v[21],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s95 = __shfl_sync(0xffffffff, x.v[23],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s96 =
        __shfl_sync(0xffffffff, x.v[24],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s97 =
        __shfl_sync(0xffffffff, x.v[26],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s98 =
        __shfl_sync(0xffffffff, x.v[28],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s99 =
        __shfl_sync(0xffffffff, x.v[30],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s100 =
        __shfl_sync(0xffffffff, x.v[25],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s101 =
        __shfl_sync(0xffffffff, x.v[27],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s102 =
        __shfl_sync(0xffffffff, x.v[29],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s103 =
        __shfl_sync(0xffffffff, x.v[31],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s104 = __shfl_sync(0xffffffff, x.v[24],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s105 = __shfl_sync(0xffffffff, x.v[26],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s106 = __shfl_sync(0xffffffff, x.v[28],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s107 = __shfl_sync(0xffffffff, x.v[30],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s108 = __shfl_sync(0xffffffff, x.v[25],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s109 = __shfl_sync(0xffffffff, x.v[27],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s110 = __shfl_sync(0xffffffff, x.v[29],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s111 = __shfl_sync(0xffffffff, x.v[31],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s112 = __shfl_sync(0xffffffff, x.v[24],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s113 = __shfl_sync(0xffffffff, x.v[26],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s114 = __shfl_sync(0xffffffff, x.v[28],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s115 = __shfl_sync(0xffffffff, x.v[30],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s116 = __shfl_sync(0xffffffff, x.v[25],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s117 = __shfl_sync(0xffffffff, x.v[27],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s118 = __shfl_sync(0xffffffff, x.v[29],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s119 = __shfl_sync(0xffffffff, x.v[31],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s120 = __shfl_sync(0xffffffff, x.v[24],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s121 = __shfl_sync(0xffffffff, x.v[26],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s122 = __shfl_sync(0xffffffff, x.v[28],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s123 = __shfl_sync(0xffffffff, x.v[30],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s124 = __shfl_sync(0xffffffff, x.v[25],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s125 = __shfl_sync(0xffffffff, x.v[27],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s126 = __shfl_sync(0xffffffff, x.v[29],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s127 = __shfl_sync(0xffffffff, x.v[31],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    Words<32> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[1] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[3] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[5] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[7] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[9] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[11] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[13] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[15] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[16] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[17] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[18] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[19] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[20] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[21] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[22] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[23] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[24] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[25] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[26] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[27] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[28] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[29] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[30] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[31] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    return out;
}

// Prepare each Half2 ONCE, with original norm/scale/den math, BEFORE this call.

template <class F> __device__ __forceinline__ Words<16> ff_a(F prepare) {

    Words<16> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> query_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> key_b(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> prob_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<8> attended_a(F prepare) {

    Words<8> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> value_b(F get) {
    Words<16> out;

    {
        u32 a = transpose_half2(get(activation::Tag<0>{})),
            b = transpose_half2(get(activation::Tag<1>{}));
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<24>{})),
            b = transpose_half2(get(activation::Tag<25>{}));
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<2>{})),
            b = transpose_half2(get(activation::Tag<3>{}));
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<26>{})),
            b = transpose_half2(get(activation::Tag<27>{}));
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<4>{})),
            b = transpose_half2(get(activation::Tag<5>{}));
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<28>{})),
            b = transpose_half2(get(activation::Tag<29>{}));
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<6>{})),
            b = transpose_half2(get(activation::Tag<7>{}));
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<30>{})),
            b = transpose_half2(get(activation::Tag<31>{}));
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<8>{})),
            b = transpose_half2(get(activation::Tag<9>{}));
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<16>{})),
            b = transpose_half2(get(activation::Tag<17>{}));
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<10>{})),
            b = transpose_half2(get(activation::Tag<11>{}));
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<18>{})),
            b = transpose_half2(get(activation::Tag<19>{}));
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<12>{})),
            b = transpose_half2(get(activation::Tag<13>{}));
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<20>{})),
            b = transpose_half2(get(activation::Tag<21>{}));
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<14>{})),
            b = transpose_half2(get(activation::Tag<15>{}));
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<22>{})),
            b = transpose_half2(get(activation::Tag<23>{}));
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

} // namespace packet_reference

namespace shallow {

using namespace endpoint;

using namespace activation;

template <int P> constexpr int logical_m16() {
    return P == 0 ? 0 : P == 1 ? 3 : P == 2 ? 1 : 2;
}
template <int Row> __device__ __forceinline__ half physical_den(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    return packed12::den_finish(s);
}
struct Entry {
    HC<64, 32> residual;
    Words<16> expand;
};

template <class F> __device__ __forceinline__ Entry entry(F get, const BodyWeights &w) {

    Entry e;

    Words<32> raw;
    each<32>([&](auto jt) { raw.v[decltype(jt)::value] = get(jt); });

    e.residual = packet_reference::residual_c(raw);

    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });

    Words<16> encoded;
    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;

        encoded.v[j] = packed12::e4four(raw.v[j * 2], raw.v[j * 2 + 1]);
    });
    e.expand = packet_reference::input_a(encoded);
    return e;
}

template <class F> __device__ __forceinline__ Entry load_entry(F get, const BodyWeights &w) {

    // Producer is scalar only at the memory/sample edge; no dynamically indexed Tile.

    auto x = fill<64, 32>(get);

    return entry([&](auto t) { return x.v[decltype(t)::value]; }, w);
}

template <int M, int N>
__device__ __forceinline__ void paired_weight(HC<M, N> &c, const Words<M / 4> &a, const u8 *w) {

    each<N / 16>([&](auto pt) {
        constexpr int p = decltype(pt)::value;

        // Original immutable B128 packet, shared by both N8 and all M16.

        const uint4 raw = *reinterpret_cast<const uint4 *>(w + p * 512 + lane() * 16);

        const uint2 b0 = make_uint2(raw.x, raw.y), b1 = make_uint2(raw.z, raw.w);

        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;

            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};

            each<2>([&](auto nt) {
                constexpr int n = p * 2 + decltype(nt)::value;

                uint2 b = decltype(nt)::value ? b1 : b0;
                u32 bb[2] = {b.x, b.y};

                Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
                mma8(z, aa, bb);

                c.v[(m * (N / 8) + n) * 2] = z.v[0];
                c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
            });
        });
    });
}

template <class F> __device__ __forceinline__ void publish_words(const HC<32, 32> &x, F publish) {

    each<2>([&](auto m) {
        each<4>([&](auto n) {
            each<2>([&](auto i) {
                publish(
                    m, n, i,
                    x.v[(decltype(m)::value * 4 + decltype(n)::value) * 2 + decltype(i)::value]);
            });
        });
    });
}

// Readout calls the original gemm16 twice with its original C handoff. Its A

// row/16, col/8 and row/8 are warp-uniform at this call site. Fixed references

// avoid rebuilding the projection Tile. This is not a general varying gather.

__device__ __forceinline__ half packet_half_uniform(const HC<32, 32> &x, int row, int col) {

    u32 word = 0;
    int bank = (row / 16 * 4 + col / 8) * 2 + (row / 8 % 2);

    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        if (bank == j)
            word = x.v[j];
    });

    return unpack(__shfl_sync(0xffffffff, word, (row % 8) * 4 + (col % 8) / 2), col % 2);
}

template <class Publish>
__device__ __forceinline__ void packet_body(Entry input, const BodyWeights &w, Publish publish) {

    auto ff = input.residual;

    {

        const auto a = input.expand;

#pragma unroll 1

        for (int part = 0; part < 4; ++part) {

            Words<16> ha;

            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;

                auto hidden = fill<64, 16>([](int, int) { return h(0); });

                paired_weight<64, 16>(hidden, a, w.expand + part * 1024 + pair * 512);

                // A hidden N16 pair dies after activation/encoding into final A.

                each<4>([&](auto mt) {
                    each<2>([&](auto it) {
                        constexpr int dest =
                            decltype(mt)::value * 4 + pair * 2 + decltype(it)::value;

                        // rawExpand C ownership, NOT the old decoded hidden consumer.

                        constexpr int bank = decltype(mt)::value * 4 + decltype(it)::value;

                        u32 x = packed12::activate_pair(hidden.v[bank]);

                        u32 y = packed12::activate_pair(hidden.v[bank + 2]);

                        ha.v[dest] = packed12::e4four(x, y);
                    });
                });
            });

            paired_weight<64, 32>(ff, ha, w.contract + part * 1024);
        }
    }

    Words<16> qa, kb, vb;

    {

        auto a = packet_reference::ff_a([&](auto jt) { return ff.v[decltype(jt)::value]; });

        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;

            auto z = fill<64, 32>([](int, int) { return h(0); });

            paired_weight<64, 32>(z, a, w.qkv + component * 1024);

            Words<8> inv;

            if constexpr (component < 2)
                each<8>([&](auto rt) {
                    constexpr int row = decltype(rt)::value;
                    inv.v[row] = packed12::splat(activation::norm<64, 32, row>(z));
                });

            auto get = [&](auto jt) {
                constexpr int j = decltype(jt)::value;
                u32 x = z.v[j];

                if constexpr (component < 2)
                    x = packed12::mul(x, inv.v[j / 8 * 2 + j % 2]);

                if constexpr (component == 0)
                    x = packed12::mul(x, packed12::splat(w.scale));

                return x;
            };

            // Directly encode final consumer words. No z96 or canonical Q/K/V.

            if constexpr (component == 0)
                qa = packet_reference::query_a(get);

            if constexpr (component == 1)
                kb = packet_reference::key_b(get);

            if constexpr (component == 2)
                vb = packet_reference::value_b(get);
        });
    }

    each<2>([&](auto st) {
        constexpr int slab = decltype(st)::value;

        Words<16> pa;

        {

            HC<32, 64> logits;
            each<2>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                u32 a[4] = {qa.v[(slab * 2 + m) * 4], qa.v[(slab * 2 + m) * 4 + 1],
                            qa.v[(slab * 2 + m) * 4 + 2], qa.v[(slab * 2 + m) * 4 + 3]};
                each<4>([&](auto pt) {
                    constexpr int p = decltype(pt)::value;
                    // qkv already includes the pre/UP/post ABI shift; original raw40 only.
                    u32 s0, s1, s2, s3;
                    const u8 *seed = w.qkv + 0xc00 + ((slab * 2 + m) * 4 + p) * 512 + lane() * 16;
                    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3)
                                 : "l"(seed));
                    each<2>([&](auto nt) {
                        constexpr int n = p * 2 + decltype(nt)::value;
                        u32 b[2] = {kb.v[n * 2], kb.v[n * 2 + 1]};
                        Fragment c{{decltype(nt)::value ? s2 : s0, decltype(nt)::value ? s3 : s1}};
                        mma8(c, a, b);
                        logits.v[(m * 8 + n) * 2] = c.v[0];
                        logits.v[(m * 8 + n) * 2 + 1] = c.v[1];
                    });
                });
            });
            each<32>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                logits.v[j] = packed12::exponent_pair(logits.v[j]);
            });

            Words<4> den;

            each<4>([&](auto rt) {
                constexpr int r = decltype(rt)::value;
                den.v[r] = packed12::splat(physical_den<r>(logits));
            });

            pa = packet_reference::prob_a([&](auto jt) {
                constexpr int j = decltype(jt)::value;

                return packed12::mul(logits.v[j], den.v[j / 16 * 2 + j % 2]);
            });
        }

        auto attended = fill<32, 32>([](int, int) { return h(0); });

        each<2>([&](auto pt) {
            constexpr int part = decltype(pt)::value;

            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 b[2] = {vb.v[part * 8 + n * 2], vb.v[part * 8 + n * 2 + 1]};

                each<2>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    u32 a[4] = {pa.v[part * 8 + m * 4], pa.v[part * 8 + m * 4 + 1],
                                pa.v[part * 8 + m * 4 + 2], pa.v[part * 8 + m * 4 + 3]};

                    Fragment c{{attended.v[(m * 4 + n) * 2], attended.v[(m * 4 + n) * 2 + 1]}};
                    mma8(c, a, b);

                    attended.v[(m * 4 + n) * 2] = c.v[0];
                    attended.v[(m * 4 + n) * 2 + 1] = c.v[1];
                });
            });
        });

        HC<32, 32> out;

        each<16>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

            out.v[j] = packed12::mul(ff.v[logical_m16<slab * 2 + j / 8>() * 8 + j % 8],
                                     pack(w.attn_gate[c], w.attn_gate[c + 1]));
        });

        auto aa = packet_reference::attended_a([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            return attended.v[j];
        });

        paired_weight<32, 32>(out, aa, w.projection);

        publish(Tag<logical_m16<slab * 2>() * 16>{}, Tag<logical_m16<slab * 2 + 1>() * 16>{}, out);
    });
}

} // namespace shallow

namespace joint {
using namespace shallow;
// Canonical channel represented by a raw-body residual C owner.
__device__ __forceinline__ int input_channel(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void gate(Entry &e, const BodyWeights &w) {
    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() & 3) * 2;
        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });
}
// Input is already in the residual owner's fixed Half registers. Quantization
// reads those original registers, independently of the gated C rail.
__device__ __forceinline__ Entry true_half(HC<64, 32> raw, const BodyWeights &w) {
    Entry e;
    e.residual = raw;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                e.expand.v[m * 4 + p * 2 + i] =
                    packed12::e4four(raw.v[m * 8 + p * 4 + i], raw.v[m * 8 + p * 4 + 2 + i]);
            });
        });
    });
    gate(e, w);
    return e;
}
template <class F> __device__ __forceinline__ Entry produce_half(F get, const BodyWeights &w) {
    return true_half(fill<64, 32>([&](int r, int c) { return get(r, input_channel(c)); }), w);
}
// Byte loads are predicated BEFORE memory access. Only a real aligned,
// consecutive, all-valid quartet may use the vector path.
template <class Address>
__device__ __forceinline__ u32 load4(const u8 *data, Address address, int row, int col) {
    int a = address(row, col), b = address(row, col + 1), c = address(row, col + 2),
        d = address(row, col + 3);
    if (a >= 0 && b == a + 1 && c == a + 2 && d == a + 3 &&
        ((reinterpret_cast<unsigned long long>(data + a) & 3) == 0))
        return *reinterpret_cast<const u32 *>(data + a);
    return u32(a < 0 ? 0 : data[a]) | (u32(b < 0 ? 0 : data[b]) << 8) |
           (u32(c < 0 ? 0 : data[c]) << 16) | (u32(d < 0 ? 0 : data[d]) << 24);
}
template <class Address>
__device__ __forceinline__ Entry e4_edge(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                int row = m * 16 + lane() / 4 + i * 8, col = (lane() & 3) * 8 + p * 4;
                u32 raw = load4(data, address, row, col), a = raw;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    u8 v = raw >> (b * 8);
                    // Keep the old NaN round-trip, including its sign/canonicalization.
                    if ((v & 127) == 127)
                        a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
                });
                e.expand.v[m * 4 + p * 2 + i] = a;
                e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
                e.residual.v[m * 8 + p * 4 + 2 + i] =
                    pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
            });
        });
    });
    gate(e, w);
    return e;
}
// HalfSlab consumer: same address recipe is used for both bytes of the pair.
// Optional pre never controls the separate mandatory shared Half pool result.
template <class Address>
__device__ __forceinline__ void publish_pair(u8 *data, half *pre, Address address, int row, int col,
                                             u32 word) {
    int a = address(row, col), b = address(row, col + 1);
    half x = unpack(word, 0), y = unpack(word, 1);
    if (a >= 0 && b == a + 1 && ((reinterpret_cast<unsigned long long>(data + a) & 1) == 0))
        *reinterpret_cast<unsigned short *>(data + a) = unsigned(e4(x)) | (unsigned(e4(y)) << 8);
    else {
        if (a >= 0)
            data[a] = e4(x);
        if (b >= 0)
            data[b] = e4(y);
    }
    if (pre) {
        if (a >= 0)
            pre[a] = x;
        if (b >= 0)
            pre[b] = y;
    }
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed8(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
            u32 v = 0;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                v |= u32(av(r, k + j)) << (j * 8);
            });
            a[i] = v;
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + lane() / 4;
                u32 v = 0;
                each<4>([&](auto jt) {
                    constexpr int j = decltype(jt)::value;
                    v |= u32(bv(k + j, col)) << (j * 8);
                });
                b[i] = v;
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma8(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed16(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
            a[i] = pack(av(r, k), av(r, k + 1));
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + lane() / 4;
                b[i] = pack(bv(k, col), bv(k + 1, col));
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma16(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// A for the original readout K16 is exactly the projection C's pair at
// [m, kpart*2+i/2, i%2]. No runtime bank selection or projection Tile.
__device__ __forceinline__ HC<32, 8> read_head(const HC<32, 32> &out, const half *weights) {
    auto head = fill<32, 8>([](int, int) { return h(0); });
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4], b[2];
            each<4>([&](auto it) {
                constexpr int i = decltype(it)::value;
                a[i] = out.v[m * 8 + kp * 4 + (i / 2) * 2 + i % 2];
            });
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = kp * 16 + (lane() & 3) * 2 + i * 8, n = lane() / 4;
                b[i] = pack(weights[k * 8 + n], weights[(k + 1) * 8 + n]);
            });
            Fragment z{{head.v[m * 2], head.v[m * 2 + 1]}};
            mma16(z, a, b);
            head.v[m * 2] = z.v[0];
            head.v[m * 2 + 1] = z.v[1];
        });
    });
    return head;
}
template <int Col> __device__ __forceinline__ half head_channel(const HC<32, 8> &head) {
    u32 result = 0;
    each<4>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        u32 v = __shfl_sync(0xffffffff, head.v[j], (lane() % 8) * 4 + Col / 2);
        if (lane() / 8 == j)
            result = v;
    });
    return unpack(result, Col % 2);
}
} // namespace joint
namespace physical {
using namespace shallow;
// Each tin packet holds (row,row+8) x (four+four channels). The raw A
// consumer and projection C producer use this exact same lane ownership.
__device__ __forceinline__ uint4 load(const u8 *p, int a) {
    if (a < 0)
        return make_uint4(0, 0, 0, 0);
    return *reinterpret_cast<const uint4 *>(p + a);
}
__device__ __forceinline__ int nc(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ int result_index(int r, int c) {
    return (r / 16) * 512 + (r % 8) * 64 + (r % 16 / 8) * 4 + 16 * (c % 8 / 2) + (c / 16) * 8 +
           (c % 16 / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void raw_word(Entry &e, int m, int p, int i, u32 raw) {
    u32 a = raw;
    each<4>([&](auto bt) {
        constexpr int b = decltype(bt)::value;
        u8 v = raw >> (b * 8);
        if ((v & 127) == 127)
            a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
    });
    e.expand.v[m * 4 + p * 2 + i] = a;
    e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
    e.residual.v[m * 8 + p * 4 + 2 + i] = pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
}
template <class Address>
__device__ __forceinline__ Entry input_tin(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 4, (lane() & 3) * 8));
        raw_word(e, m, 0, 0, v.x);
        raw_word(e, m, 0, 1, v.y);
        raw_word(e, m, 1, 0, v.z);
        raw_word(e, m, 1, 1, v.w);
    });
    joint::gate(e, w);
    return e;
}
template <class Address>
__device__ __forceinline__ Entry input_compact(const u8 *data, Address address,
                                               const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 2, (lane() & 1) * 4));
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int p = decltype(pt)::value, i = decltype(it)::value;
                int src = (lane() / 4 + i * 8) * 2 + p;
                u32 a = __shfl_sync(0xffffffff, v.x, src), b = __shfl_sync(0xffffffff, v.y, src);
                u32 c = __shfl_sync(0xffffffff, v.z, src), d = __shfl_sync(0xffffffff, v.w, src);
                u32 raw = (lane() % 4 == 0) ? a : (lane() % 4 == 1) ? b : (lane() % 4 == 2) ? c : d;
                raw_word(e, m, p, i, raw);
            });
        });
    });
    joint::gate(e, w);
    return e;
}
// Same C owner publishes E4 memory and the unquantized shared Half rail.
// The shared layout is consumed directly by the four-point pool, not redecoded.
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_tin(const HC<32, 32> &z, u8 *data, half *pre, half *result,
                                            Address address) {
    each<2>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        int row = (m == 0 ? Base0 : Base1) + lane() / 4, col = (lane() & 3) * 2;
        int a = address(row, col);
        uint4 v = make_uint4(packed12::e4four(z.v[m * 8], z.v[m * 8 + 2]),
                             packed12::e4four(z.v[m * 8 + 1], z.v[m * 8 + 3]),
                             packed12::e4four(z.v[m * 8 + 4], z.v[m * 8 + 6]),
                             packed12::e4four(z.v[m * 8 + 5], z.v[m * 8 + 7]));
        if (a >= 0)
            *reinterpret_cast<uint4 *>(data + a) = v;
        if (pre || result) {
            uint4 lo = make_uint4(z.v[m * 8], z.v[m * 8 + 2], z.v[m * 8 + 1], z.v[m * 8 + 3]);
            uint4 hi = make_uint4(z.v[m * 8 + 4], z.v[m * 8 + 6], z.v[m * 8 + 5], z.v[m * 8 + 7]);
            if (pre && a >= 0) {
                *reinterpret_cast<uint4 *>(pre + a) = lo;
                *reinterpret_cast<uint4 *>(pre + a + 8) = hi;
            }
            if (result) {
                int s = (m == 0 ? Base0 : Base1) / 16 * 512 + lane() * 16;
                *reinterpret_cast<uint4 *>(result + s) = lo;
                *reinterpret_cast<uint4 *>(result + s + 8) = hi;
            }
        }
    });
}
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_compact(const HC<32, 32> &z, u8 *data, half *pre,
                                                Address address) {
    each<2>([&](auto mt) {
        each<2>([&](auto pt) {
            constexpr int m = decltype(mt)::value, p = decltype(pt)::value;
            u32 lo = packed12::e4four(z.v[m * 8 + p * 4], z.v[m * 8 + p * 4 + 2]);
            u32 hi = packed12::e4four(z.v[m * 8 + p * 4 + 1], z.v[m * 8 + p * 4 + 3]);
            uint4 v;
            each<4>([&](auto kt) {
                constexpr int k = decltype(kt)::value;
                int src = (lane() % 8) * 4 + k;
                u32 a = __shfl_sync(0xffffffff, lo, src), b = __shfl_sync(0xffffffff, hi, src),
                    q = lane() < 8 ? a : b;
                if constexpr (k == 0)
                    v.x = q;
                if constexpr (k == 1)
                    v.y = q;
                if constexpr (k == 2)
                    v.z = q;
                if constexpr (k == 3)
                    v.w = q;
            });
            if (lane() < 16) {
                int a = address((m == 0 ? Base0 : Base1) + lane(), p * 16);
                if (a >= 0)
                    *reinterpret_cast<uint4 *>(data + a) = v;
            }
            // Optional true-Half ABI remains in C ownership; no E4 reconstruction.
            if (pre) {
                each<2>([&](auto nt) {
                    each<2>([&](auto it) {
                        constexpr int n = p * 2 + decltype(nt)::value, i = decltype(it)::value;
                        int row = (m == 0 ? Base0 : Base1) + lane() / 4 + i * 8,
                            col = n * 8 + (lane() & 3) * 2, a = address(row, col);
                        if (a >= 0)
                            *reinterpret_cast<u32 *>(pre + a) = z.v[m * 8 + n * 2 + i];
                    });
                });
            }
        });
    });
}
} // namespace physical

#ifndef NR_H
#error Native NR must compile with its real packet height
#endif
#ifndef NR_W
#error Native NR must compile with its real packet width
#endif
namespace shallow_static {
__device__ __forceinline__ int height() { return NR_H / 2; }
__device__ __forceinline__ int width() { return NR_W / 2; }
__device__ __forceinline__ int token(int r) {
    return (r & 48) | ((r & 7) << 1) | ((r & 8) >> 3);
}
__device__ __forceinline__ int pixel(int r) {
    return (r & 3) ^ ((r & 12) << 1) ^ ((r & 16) << 1) ^ ((r & 32)) ^ ((r & 32) >> 3);
}
__device__ __forceinline__ int pool_local(int p) {
    return (p & 3) ^ ((p & 4) << 2) ^ ((p & 4) << 3) ^ ((p & 56) >> 1);
}
__device__ __forceinline__ int ctop(int t) {
    return ((t & 1) << 4) ^ ((t & 6) >> 1) ^ (t & 8) ^ ((t & 16) << 1) ^ (t & 32) ^ ((t & 32) >> 3);
}
__device__ __forceinline__ int ptoc(int t) {
    return ((t & 3) << 1) ^ ((t & 4) << 2) ^ ((t & 4) << 3) ^ (t & 8) ^ ((t & 16) >> 4) ^
           ((t & 32) >> 1);
}
__device__ __forceinline__ int compact(int y, int x, int c) {
    int t = ctop((y & 7) * 8 + (x & 7)), k = (c & 3) | ((c & 4) << 2) | ((c & 24) >> 1);
    int l = (k >> 4) * 1024 + t * 16 + (k & 15);
    return (((y >> 3) + (height() / 8) * (l >> 10)) * (width() / 4) + (x >> 5) + (width() / 32) * ((l >> 7) & 7)) *
               512 +
           ((x >> 3) & 3) * 128 + (l & 127);
}
template <int Seq, bool Output = false> __device__ __forceinline__ int address(int row, int c) {
    constexpr int phase = Seq >= 151 ? Seq - 151 : Seq - 3;
    const int gx = width() / 8 + ((phase == 1 || phase == 2) ? 1 : 0);
    constexpr int sx = (phase == 1 || phase == 2) ? -1 : 0,
                  sy = (phase == 1 || phase == 3) ? -1 : 0;
    int stripe = row >> 4, cy = int(blockIdx.x) / gx * 2 + sy + (stripe == 1 || stripe == 2),
        cx = int(blockIdx.x) % gx * 2 + sx + (stripe == 2 || stripe == 3);
    if (cy < 0 || cy >= height() / 4 || cx < 0 || cx >= width() / 4)
        return -1;
    if constexpr (Output)
        c = physical::nc(c);
    if constexpr ((Seq == 3 && !Output) || (Seq == 154 && Output)) {
        int s0 = (cy & 1) ? 1 + (cx & 1) : (cx & 1) * 3;
        int y = (cy >> 1) * 8 + s0 * 2 + ((row >> 2) & 1),
            x = (cx >> 1) * 8 + ((row & 3) << 1) + ((row >> 3) & 1);
        return compact(y, x, c);
    }
    return (cy * (width() / 4) + cx) * 512 + (row & 7) * 64 + ((row >> 3) & 1) * 4 + 8 * (c >> 2) +
           (c & 3);
}
__device__ __forceinline__ int up_address(int row, int c) {
    int low = row & 63, outer = row >> 6;
    int y = (outer / (width() / 32)) * 4 + ((low >> 3) & 1) + 2 * (low & 1);
    int x = (outer % (width() / 32)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) +
            4 * ((low >> 4) & 1) + 8 * ((low >> 5) & 1);
    return (y * (width() / 2) + x) * 16 + (c >> 4) * (height() / 2) * (width() / 2) * 16 + (c & 15);
}
__device__ __forceinline__ int pool_row(int cta, int owner) {
    int by = cta / (width() / 8), bx = cta % (width() / 8);
    if ((by == 0 || by == height() / 8) && owner >= 8)
        return -1;
    int y = (by == 0 ? 0 : by * 4 - 2) + (owner >> 2);
    return y * (width() / 2) + bx * 4 + (owner & 3);
}
__device__ __forceinline__ int pool_address(int row, int c) {
    return row * 16 + (c >> 4) * (height() / 2) * (width() / 2) * 16 + (c & 1) + ((c & 6) << 1) + ((c & 8) >> 2);
}
} // namespace shallow_static
using namespace activation;
using namespace shallow;
using namespace endpoint;
__device__ int tin_offset(int y, int x, int c, int width) {
    int t = (y & 7) * 8 + (x & 7), p = 64 * (t / 2) + 4 * (t & 1) + 8 * (c / 4) + (c & 3),
        s = p / 512;
    int cy = (y / 8) * 2 + (s == 1 || s == 2), cx = (x / 8) * 2 + (s == 2 || s == 3);
    return (cy * (width / 4) + cx) * 512 + (p & 511);
}
extern "C" __global__ __launch_bounds__(32) void block0_native_packet(BodyWeights w,
                                                                      const float *features,
                                                                      const half *adapter,
                                                                      u8 *fullskip, u8 *compact,
                                                                      int *status) {
    // 64x16 Half packet shared storage = 2048 B, only one 32-thread warp.
    __shared__ half packet[64][16];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int t = lane() + 32 * i, x = blockIdx.x * 8 + (t & 7), y = blockIdx.y * 8 + t / 8;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            float v = features[(c * NR_H + y) * NR_W + x];
            half hv = h(v);
            // InputBlock.forward explicitly casts the Float32 packet to Half.
            // Conditioning values (e.g. tone=1.2) need not already equal Half.
            if (!isfinite(v) || !isfinite(f(hv)))
                atomicOr(status, 1);
            packet[t][c] = hv;
        }
    }
    __syncwarp();
    auto adapted = fill<64, 32>([](int, int) { return h(0); });
    __shared__ __align__(16) half result[2048];
    joint::fixed16<64, 32>(
        adapted, [&](int r, int k) { return packet[shallow_static::pixel(r)][k]; },
        [&](int k, int n) { return adapter[k * 32 + joint::input_channel(n)]; });
    auto input = joint::true_half(adapted, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &out) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        physical::publish_tin<base0, base1>(out, fullskip, nullptr, result, [&](int row, int c) {
            int slot = shallow_static::token(row);
            return tin_offset(blockIdx.y * 8 + slot / 8, blockIdx.x * 8 + (slot & 7),
                              physical::nc(c), NR_W);
        });
    });
    __syncwarp();
    // One lane owns a complete compact packet on the unquantized Half rail.
    int p = lane() / 2, kh = (lane() & 1) * 16, py = (p / 4) * 2, px = (p % 4) * 2;
    int a = shallow_static::pool_local(py * 8 + px),
        b = shallow_static::pool_local(py * 8 + px + 1);
    int c = shallow_static::pool_local((py + 1) * 8 + px),
        d = shallow_static::pool_local((py + 1) * 8 + px + 1);
    uint4 packed;
    each<4>([&](auto wt) {
        constexpr int j = decltype(wt)::value;
        u32 word = 0;
        each<4>([&](auto bt) {
            constexpr int bit = decltype(bt)::value;
            int k = kh + j * 4 + bit;
            int phys = (k / 4 % 4) * 8 + (k / 16) * 4 + k % 4, col = physical::nc(phys);
            half va = result[physical::result_index(a, col)],
                 vb = result[physical::result_index(b, col)];
            half vc = result[physical::result_index(c, col)],
                 vd = result[physical::result_index(d, col)];
            half avg = __hmul(__hadd(__hadd(va, vb), __hadd(vc, vd)), h(.25f));
            word |= u32(e4(avg)) << (bit * 8);
        });
        if constexpr (j == 0)
            packed.x = word;
        if constexpr (j == 1)
            packed.y = word;
        if constexpr (j == 2)
            packed.z = word;
        if constexpr (j == 3)
            packed.w = word;
    });
    int y = blockIdx.y * 4 + p / 4, x = blockIdx.x * 4 + p % 4,
        slot = shallow_static::ptoc((y & 7) * 8 + (x & 7));
    y = (y / 8) * 8 + slot / 8;
    x = (x / 8) * 8 + (slot & 7);
    int first = (kh / 16) * 4;
    *reinterpret_cast<uint4 *>(compact + shallow_static::compact(y, x, first)) = packed;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_3(u8 *r, BodyWeights w,
                                                                 const int *im, const int *om,
                                                                 int in, int out, int counter,
                                                                 const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<3>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (true)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) { return shallow_static::address<3, true>(row, col); };
        if (false)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_4(u8 *r, BodyWeights w,
                                                                 const int *im, const int *om,
                                                                 int in, int out, int counter,
                                                                 const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<4>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (false)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) { return shallow_static::address<4, true>(row, col); };
        if (false)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_5(u8 *r, BodyWeights w,
                                                                 const int *im, const int *om,
                                                                 int in, int out, int counter,
                                                                 const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<5>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (false)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) { return shallow_static::address<5, true>(row, col); };
        if (false)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_152(u8 *r, BodyWeights w,
                                                                   const int *im, const int *om,
                                                                   int in, int out, int counter,
                                                                   const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<152>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (false)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) {
            return shallow_static::address<152, true>(row, col);
        };
        if (false)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_153(u8 *r, BodyWeights w,
                                                                   const int *im, const int *om,
                                                                   int in, int out, int counter,
                                                                   const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<153>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (false)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) {
            return shallow_static::address<153, true>(row, col);
        };
        if (false)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

extern "C" __global__ __launch_bounds__(32) void outer1_static_154(u8 *r, BodyWeights w,
                                                                   const int *im, const int *om,
                                                                   int in, int out, int counter,
                                                                   const half *mixed, half *pre) {
    auto address = [&](int t, int c) { return shallow_static::address<154>(t, c); };
    Entry input;
    if (mixed)
        input = joint::produce_half(
            [&](int t, int c) {
                int a = address(t, c);
                return a < 0 ? h(0) : mixed[a];
            },
            w);
    else if (false)
        input = physical::input_compact(r + in, address, w);
    else
        input = physical::input_tin(r + in, address, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &z) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto publish = [&](int row, int col) {
            return shallow_static::address<154, true>(row, col);
        };
        if (true)
            physical::publish_compact<base0, base1>(z, r + out, pre, publish);
        else
            physical::publish_tin<base0, base1>(z, r + out, pre, nullptr, publish);
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

} // namespace nr_outer_one_0
// ============================================================================
// OUTER shallow_static_geometry/transition.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_transition_1 {
// Experimental shared arithmetic component. Not a qualified native endpoint.
// All Tile storage is distributed over exactly one warp in MMA C coordinates.
namespace endpoint {
using u32 = unsigned;
using u8 = unsigned char;
__device__ __forceinline__ int lane() {
    return threadIdx.x & 31;
}
__device__ __forceinline__ half h(float x) {
    return __float2half_rn(x);
}
__device__ __forceinline__ float f(half x) {
    return __half2float(x);
}
__device__ __forceinline__ u32 pack(half a, half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}
__device__ __forceinline__ half unpack(u32 x, int i) {
    return __ushort_as_half(x >> (16 * i));
}
__device__ __forceinline__ u8 e4(half x) {
    return __nv_cvt_halfraw_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half une4(u8 x) {
    return __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
}
__device__ __forceinline__ half q(half x) {
    return une4(e4(x));
}
__device__ __forceinline__ half activate(half x) {
    float c = fminf(4.f, fmaxf(-4.f, f(x)));
    half g = h(__fmaf_rn(fabsf(c), -.055908203125f, .447265625f));
    return __hmul(x, h(__fmaf_rn(c, f(g), .89453125f)));
}
__device__ __forceinline__ half exponent(half x) {
    half a = h(__fmaf_rn(f(x), .044921875f, 1.30078125f));
    a = h(fminf(1.5693359375f, fmaxf(1.03125f, f(a))));
    return __ushort_as_half((unsigned(__half_as_ushort(a)) << 5) + 32768);
}
__device__ __forceinline__ half reciprocal(half x) {
    float z;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
__device__ __forceinline__ half rsqrt_half(half x) {
    float z;
    asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
struct Fragment {
    u32 v[2];
};
__device__ __forceinline__ void mma8(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma16(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int M, int N> struct Tile {
    static_assert(M % 16 == 0 && N % 8 == 0, "MMA tile dimensions");
    Fragment c[M / 16][N / 8];
    // Collective arbitrary gather. Both words of every bank are shuffled before
    // selection: selecting a dynamic word BEFORE shfl is incorrect when the
    // destination lanes request different rows. No lane-private full window.
    __device__ half get(int row, int col) const {
        u32 bits = 0;
        int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
                u32 a = __shfl_sync(0xffffffff, c[m][n].v[0], src);
                u32 b = __shfl_sync(0xffffffff, c[m][n].v[1], src);
                if (m == row / 16 && n == col / 8)
                    bits = (row & 8) ? b : a;
            }
        }
        return unpack(bits, col & 1);
    }
    template <class F> __device__ void fill(F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r = m * 16 + (lane() >> 2) + i * 8, k = n * 8 + (lane() & 3) * 2;
                    half a = fn(r, k), b = fn(r, k + 1);
                    c[m][n].v[i] = pack(a, b);
                }
            }
        }
    }
};
// Accessors return encoded E4 bytes in physical MMA K order.
template <int M, int N, class A, class B> __device__ void gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = 0;
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    a[i] |= u32(av(r, k + j)) << (j * 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
template <int M, int N, class A, class B> __device__ void gemm16(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
                half x = av(r, k), y = av(r, k + 1);
                a[i] = pack(x, y);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + (lane() >> 2);
                half x = bv(k, col), y = bv(k + 1, col);
                b[i] = pack(x, y);
            }
            mma16(c.c[m][n], a, b);
        }
    }
}
struct BodyWeights {
    // Dense decoded logical matrices, NOT a compressed/repacked static-weight
    // candidate. Host route construction must follow physical_1h and archive.
    const u8 *expand;                 // [32,128], hidden_inverse applied to each stream
    const u8 *contract;               // [4,32,32], stream,Khalf order 0/0,1/0,0/32,1/32
    const u8 *qkv;                    // [32,96], Qe Qo Ke Ko Ve Vo
    const u8 *projection;             // [32,32]
    const half *bias;                 // [64,64], physical bias half -> query/key route
    const half *ffn_gate, *attn_gate; // output-N order
    const int *cp, *rc, *oi, *ai, *pk;
    half scale;
};
__device__ half norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = z.get(row, base + j), v1 = z.get(row, base + 8 + j);
        half v2 = z.get(row, base + 16 + j), v3 = z.get(row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = e.get(row, j), b = e.get(row, j + 8);
        g[j] = __hadd(a, b);
        a = e.get(row, j + 48);
        b = e.get(row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 16);
        b = e.get(row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 32);
        b = e.get(row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ int qk_order(int k) {
    return (k / 16) * 16 + (k % 2) + ((k / 2) % 2) * 8 + ((k / 4) % 4) * 2;
}
// Complete shared 64-token arithmetic; input is unquantized Half in canonical
// channels and physical token order. Output callback consumes two 32-row slabs
// without a global latent publication. Caller supplies sampling/codec/readout.
// This function has NOT been independently oracle-qualified.
template <class Publish>
__device__ void body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill([&](int r, int c) { return __hmul(input.get(r, w.rc[c]), w.ffn_gate[c]); });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return e4(input.get(r, w.cp[k])); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
// Direct C-word transformation has no cross-lane reads/aliasing.
#pragma unroll 1
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = hidden.c[m][n].v[i];
                    hidden.c[m][n].v[i] =
                        pack(q(activate(unpack(a, 0))), q(activate(unpack(a, 1))));
                }
        gemm8(
            ff, [&](int r, int k) { return e4(hidden.get(r, k)); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    gemm8(
        z, [&](int r, int k) { return e4(ff.get(r, w.oi[k])); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = norm(z, r, 0);
            ik[m * 2 + i] = norm(z, r, 32);
        }
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n)
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                z.c[m][n].v[i] = pack(q(a), q(b));
            }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return e4(z.get(slab * 32 + r, qk_order(k))); },
            [&](int k, int n) { return e4(z.get(n, 32 + qk_order(k))); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = denominator(logits, m * 16 + (lane() >> 2) + i * 8);
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    logits.c[m][n].v[i] =
                        pack(q(__hmul(unpack(a, 0), d)), q(__hmul(unpack(a, 1), d)));
                }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return e4(logits.get(r, w.pk[part * 32 + k])); },
                [&](int k, int n) { return e4(z.get(w.pk[part * 32 + k], 64 + n)); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) { return __hmul(ff.get(slab * 32 + r, n), w.attn_gate[n]); });
        gemm8(
            out, [&](int r, int k) { return e4(attended.get(r, w.ai[k])); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// Call sites have warp-uniform row/16 and row/8. Channel routes may vary
// between lanes, so only that bank dimension is scanned before selection.
template <int M, int N> __device__ half packed_half_row(const Tile<M, N> &t, int row, int col) {
    u32 bits = 0;
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll
    for (int n = 0; n < N / 8; ++n) {
        u32 x = __shfl_sync(0xffffffff, t.c[row / 16][n].v[(row >> 3) & 1], src);
        if (n == col / 8)
            bits = x;
    }
    return unpack(bits, col & 1);
}
template <int M, int N> __device__ half packed_half_uniform(const Tile<M, N> &t, int row, int col) {
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
    return unpack(__shfl_sync(0xffffffff, t.c[row / 16][col / 8].v[(row >> 3) & 1], src), col & 1);
}
// Four encoded E4 values per C bank: row0 pair in bytes 0/1, row8
// pair in bytes 2/3. This is a copy, never the Half residual storage.
// One word shuffle selects both rows; no decode-to-Half/re-encode roundtrip.
template <int M, int N> struct PackedE4 {
    u32 c[M / 16][N / 8];
    template <class F> __device__ void encode(const Tile<M, N> &t, F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m)
            for (int n = 0; n < N / 8; ++n) {
                u32 a = t.c[m][n].v[0], b = t.c[m][n].v[1];
                c[m][n] = u32(e4(fn(unpack(a, 0)))) | (u32(e4(fn(unpack(a, 1)))) << 8) |
                          (u32(e4(fn(unpack(b, 0)))) << 16) | (u32(e4(fn(unpack(b, 1)))) << 24);
            }
    }
    __device__ u8 row(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
            u32 x = __shfl_sync(0xffffffff, c[r / 16][n], src);
            if (n == k / 8)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
    __device__ u8 col(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int m = 0; m < M / 16; ++m) {
            u32 x = __shfl_sync(0xffffffff, c[m][k / 8], src);
            if (m == r / 16)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
};
__device__ half packed_norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = packed_half_uniform(z, row, base + j),
             v1 = packed_half_uniform(z, row, base + 8 + j);
        half v2 = packed_half_uniform(z, row, base + 16 + j),
             v3 = packed_half_uniform(z, row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half packed_denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = packed_half_uniform(e, row, j), b = packed_half_uniform(e, row, j + 8);
        g[j] = __hadd(a, b);
        a = packed_half_uniform(e, row, j + 48);
        b = packed_half_uniform(e, row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 16);
        b = packed_half_uniform(e, row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 32);
        b = packed_half_uniform(e, row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
template <class Publish>
__device__ void packed_body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill(
        [&](int r, int c) { return __hmul(packed_half_row(input, r, w.rc[c]), w.ffn_gate[c]); });
    PackedE4<64, 32> input8;
    input8.encode(input, [](half x) { return x; });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return input8.row(r, w.cp[k]); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
        PackedE4<64, 32> hidden8;
        hidden8.encode(hidden, [](half x) { return activate(x); });
        gemm8(
            ff, [&](int r, int k) { return hidden8.row(r, k); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    PackedE4<64, 32> ff8;
    ff8.encode(ff, [](half x) { return x; });
    gemm8(
        z, [&](int r, int k) { return ff8.row(r, w.oi[k]); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = packed_norm(z, r, 0);
            ik[m * 2 + i] = packed_norm(z, r, 32);
        }
    PackedE4<64, 96> z8;
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n) {
            u32 word = 0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                word |= (u32(e4(a)) | (u32(e4(b)) << 8)) << (16 * i);
            }
            z8.c[m][n] = word;
        }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return z8.row(slab * 32 + r, qk_order(k)); },
            [&](int k, int n) { return z8.row(n, 32 + qk_order(k)); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = packed_denominator(logits, m * 16 + (lane() >> 2) + i * 8);
        PackedE4<32, 64> probs;
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n) {
                u32 word = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    word |=
                        (u32(e4(__hmul(unpack(a, 0), d))) | (u32(e4(__hmul(unpack(a, 1), d))) << 8))
                        << (16 * i);
                }
                probs.c[m][n] = word;
            }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return probs.row(r, w.pk[part * 32 + k]); },
                [&](int k, int n) { return z8.col(w.pk[part * 32 + k], 64 + n); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) {
            return __hmul(packed_half_row(ff, slab * 32 + r, n), w.attn_gate[n]);
        });
        PackedE4<32, 32> attended8;
        attended8.encode(attended, [](half x) { return x; });
        gemm8(
            out, [&](int r, int k) { return attended8.row(r, w.ai[k]); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// A is the same physical K packet for every N bank. Gather once per M,
// then consume it in original N order. Each accumulator sees identical MMA.
template <int M, int N, class A, class B>
__device__ void organized_gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
        u32 a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[i] |= u32(av(r, k + j)) << (j * 8);
        }
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
} // namespace endpoint
namespace endpoint {
// Lossless same-capacity B8 layout: [K/32][N/8][lane][8].
// Bytes are identical to the traced raw B8, but packets are lane-permuted
// to match the accepted decoder's N order. Not native register scheduling.
struct WeightPacket {
    const u8 *data;
    int kbase, nbase, width;
    __device__ uint2 load(int bank) const {
        return *(const uint2 *)(data +
                                ((kbase / 32 * (width / 8) + nbase / 8 + bank) * 32 + lane()) * 8);
    }
};
template <int M, int N, class A>
__device__ void organized_gemm8(Tile<M, N> &c, A av, WeightPacket bv) {
    u32 a[M / 16][4];
#pragma unroll
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[m][i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[m][i] |= u32(av(r, k + j)) << (j * 8);
        }
    }
#pragma unroll 1
    for (int n = 0; n < N / 8; ++n) {
        uint2 packet = bv.load(n);
        u32 b[2] = {packet.x, packet.y};
#pragma unroll
        for (int m = 0; m < M / 16; ++m)
            mma8(c.c[m][n], a[m], b);
    }
}
} // namespace endpoint
namespace packed12 {
using namespace endpoint;
union HalfBits {
    u32 u;
    half2 h;
    __device__ HalfBits(u32 x) : u(x) {}
    __device__ HalfBits(half2 x) : h(x) {}
};
__device__ __forceinline__ half2 hh(u32 x) {
    return HalfBits(x).h;
}
__device__ __forceinline__ u32 bits(half2 x) {
    return HalfBits(x).u;
}
__device__ __forceinline__ u32 mul(u32 a, u32 b) {
    return bits(__hmul2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 splat(half a) {
    return pack(a, a);
}
__device__ __forceinline__ u32 add(u32 a, u32 b) {
    return bits(__hadd2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 fma(u32 a, u32 b, u32 c) {
    return bits(__hfma2(hh(a), hh(b), hh(c)));
}
__device__ __forceinline__ u32 activate_pair(u32 x) {
    half2 c = __hmin2(__float2half2_rn(4.f), __hmax2(__float2half2_rn(-4.f), hh(x)));
    half2 g = __hfma2(__habs2(c), __float2half2_rn(-.055908203125f), __float2half2_rn(.447265625f));
    return bits(__hmul2(hh(x), __hfma2(c, g, __float2half2_rn(.89453125f))));
}
__device__ __forceinline__ u32 exponent_pair(u32 x) {
    half2 a = __hfma2(hh(x), __float2half2_rn(.044921875f), __float2half2_rn(1.30078125f));
    a = __hmin2(__float2half2_rn(1.5693359375f), __hmax2(__float2half2_rn(1.03125f), a));
    u32 v = bits(a);
    return (((v & 0xffffu) << 5) + 32768u) & 0xffffu | ((((v >> 16) << 5) + 32768u) << 16);
}
__device__ __forceinline__ u32 e4pair(u32 x) {
    return __nv_cvt_halfraw2_to_fp8x2(hh(x), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ u32 e4four(u32 a, u32 b) {
    return e4pair(a) | (e4pair(b) << 16);
}
struct Identity {
    __device__ u32 operator()(u32 x) const {
        return x;
    }
};
struct Activate {
    __device__ u32 operator()(u32 x) const {
        return activate_pair(x);
    }
};
} // namespace packed12
namespace packed12 {
// Each four-lane row group owns four distinct adjacent-channel partial pairs.
__device__ __forceinline__ u32 row_broadcast(u32 v, int owner) {
    return __shfl_sync(0xffffffff, v, (lane() & 28) + owner);
}
__device__ __forceinline__ half norm_finish(u32 t) {
    u32 a = add(row_broadcast(t, 0), row_broadcast(t, 2));
    u32 b = add(row_broadcast(t, 1), row_broadcast(t, 3));
    u32 s = add(a, b);
    return rsqrt_half(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
__device__ __forceinline__ half den_finish(u32 t) {
    // NOT a balanced xor tree: (((partial0+partial1)+partial2)+partial3).
    u32 s = add(add(add(row_broadcast(t, 0), row_broadcast(t, 1)), row_broadcast(t, 2)),
                row_broadcast(t, 3));
    return reciprocal(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
template <int M, int N> __device__ half norm_tile(const Tile<M, N> &z, int row, int base) {
    int m = row / 16, i = (row / 8) & 1, n = base / 8;
    u32 a = z.c[m][n].v[i], b = z.c[m][n + 1].v[i], c = z.c[m][n + 2].v[i], d = z.c[m][n + 3].v[i];
    return norm_finish(add(fma(a, a, mul(c, c)), fma(b, b, mul(d, d))));
}
__device__ half den_two(const Tile<32, 64> &z, int row) {
    int m = row / 16, i = (row / 8) & 1;
    // lane%4 = g; two halves are s=0/1. Each t contributes (16t+2g)+(16t+2g+8).
    u32 s = add(z.c[m][0].v[i], z.c[m][1].v[i]);
    s = add(s, add(z.c[m][2].v[i], z.c[m][3].v[i]));
    s = add(s, add(z.c[m][4].v[i], z.c[m][5].v[i]));
    s = add(s, add(z.c[m][6].v[i], z.c[m][7].v[i]));
    return den_finish(s);
}
} // namespace packed12
namespace activation {
using namespace endpoint;
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
template <int M, int N> using HC = Words<(M / 16) * (N / 8) * 2>;
template <int M, int N, class F> __device__ __forceinline__ HC<M, N> fill(F f) {
    HC<M, N> x;
    each<M / 16>([&](auto mt) {
        each<N / 8>([&](auto nt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, n = decltype(nt)::value,
                              i = decltype(it)::value;
                int r = m * 16 + lane() / 4 + i * 8, c = n * 8 + (lane() % 4) * 2;
                x.v[(m * (N / 8) + n) * 2 + i] = pack(f(r, c), f(r, c + 1));
            });
        });
    });
    return x;
}
template <int M, int N, class F>
__device__ __forceinline__ Words<M / 16 * (N / 8)> encode(const HC<M, N> &x, F f) {
    Words<M / 16 * (N / 8)> y;
    each<M / 16 * (N / 8)>([&](auto t) {
        constexpr int j = decltype(t)::value;
        u32 a = x.v[j * 2], b = x.v[j * 2 + 1];
        y.v[j] = packed12::e4four(f(a), f(b));
    });
    return y;
}
template <int M, int N>
__device__ __forceinline__ void weight(HC<M, N> &c, const Words<M / 4> &a, WeightPacket w) {
    each<N / 8>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        uint2 b = w.load(n);
        u32 bb[2] = {b.x, b.y};
        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma8(z, aa, bb);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// Uniform fixed bank, row owner is lane/4; norm/den do not scan any banks.
template <int M, int N, int Row, int Col>
__device__ __forceinline__ half component(const HC<M, N> &x) {
    constexpr int bank = (Row / 2 * (N / 8) + Col / 8) * 2 + Row % 2;
    return unpack(__shfl_sync(0xffffffff, x.v[bank], (lane() & 28) + (Col % 8) / 2), Col % 2);
}
template <int M, int N, int Row, int Base = 0>
__device__ __forceinline__ half norm(const HC<M, N> &x) {
    constexpr int j = (Row / 2 * (N / 8) + Base / 8) * 2 + Row % 2;
    u32 a = x.v[j], b = x.v[j + 2], c = x.v[j + 4], d = x.v[j + 6];
    return packed12::norm_finish(packed12::add(packed12::fma(a, a, packed12::mul(c, c)),
                                               packed12::fma(b, b, packed12::mul(d, d))));
}
template <int Row> __device__ __forceinline__ half denominator(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    return packed12::den_finish(s);
}
} // namespace activation

namespace packet_reference {

using u32 = unsigned;

template <int N> using Words = activation::Words<N>;

__device__ __forceinline__ u32 transpose_half2(u32 a) {
    u32 d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;" : "=r"(d) : "r"(a));
    return d;
}

// input_a: input is ALREADY prepared encoded E4, except residual_c is original Half2.

__device__ __forceinline__ Words<16> input_a(const Words<16> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s5 = __shfl_sync(0xffffffff, x.v[1],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s6 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s7 = __shfl_sync(0xffffffff, x.v[3],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[1],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s17 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s18 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s19 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s37 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s38 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s39 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s49 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s50 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s51 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    Words<16> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[1] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[3] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[5] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[7] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[9] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[11] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[13] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[15] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    return out;
}

__device__ __forceinline__ Words<32> residual_c(const Words<32> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s5 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s6 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s7 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[1],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[3],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 = __shfl_sync(0xffffffff, x.v[0],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s17 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s18 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s19 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[1],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s37 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s38 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s39 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s49 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s50 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s51 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s64 =
        __shfl_sync(0xffffffff, x.v[16],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s65 =
        __shfl_sync(0xffffffff, x.v[18],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s66 =
        __shfl_sync(0xffffffff, x.v[20],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s67 =
        __shfl_sync(0xffffffff, x.v[22],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s68 =
        __shfl_sync(0xffffffff, x.v[17],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s69 =
        __shfl_sync(0xffffffff, x.v[19],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s70 =
        __shfl_sync(0xffffffff, x.v[21],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s71 =
        __shfl_sync(0xffffffff, x.v[23],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s72 = __shfl_sync(0xffffffff, x.v[16],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s73 = __shfl_sync(0xffffffff, x.v[18],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s74 = __shfl_sync(0xffffffff, x.v[20],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s75 = __shfl_sync(0xffffffff, x.v[22],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s76 = __shfl_sync(0xffffffff, x.v[17],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s77 = __shfl_sync(0xffffffff, x.v[19],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s78 = __shfl_sync(0xffffffff, x.v[21],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s79 = __shfl_sync(0xffffffff, x.v[23],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s80 = __shfl_sync(0xffffffff, x.v[16],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s81 = __shfl_sync(0xffffffff, x.v[18],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s82 = __shfl_sync(0xffffffff, x.v[20],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s83 = __shfl_sync(0xffffffff, x.v[22],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s84 = __shfl_sync(0xffffffff, x.v[17],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s85 = __shfl_sync(0xffffffff, x.v[19],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s86 = __shfl_sync(0xffffffff, x.v[21],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s87 = __shfl_sync(0xffffffff, x.v[23],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s88 = __shfl_sync(0xffffffff, x.v[16],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s89 = __shfl_sync(0xffffffff, x.v[18],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s90 = __shfl_sync(0xffffffff, x.v[20],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s91 = __shfl_sync(0xffffffff, x.v[22],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s92 = __shfl_sync(0xffffffff, x.v[17],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s93 = __shfl_sync(0xffffffff, x.v[19],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s94 = __shfl_sync(0xffffffff, x.v[21],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s95 = __shfl_sync(0xffffffff, x.v[23],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s96 =
        __shfl_sync(0xffffffff, x.v[24],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s97 =
        __shfl_sync(0xffffffff, x.v[26],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s98 =
        __shfl_sync(0xffffffff, x.v[28],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s99 =
        __shfl_sync(0xffffffff, x.v[30],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s100 =
        __shfl_sync(0xffffffff, x.v[25],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s101 =
        __shfl_sync(0xffffffff, x.v[27],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s102 =
        __shfl_sync(0xffffffff, x.v[29],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s103 =
        __shfl_sync(0xffffffff, x.v[31],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s104 = __shfl_sync(0xffffffff, x.v[24],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s105 = __shfl_sync(0xffffffff, x.v[26],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s106 = __shfl_sync(0xffffffff, x.v[28],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s107 = __shfl_sync(0xffffffff, x.v[30],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s108 = __shfl_sync(0xffffffff, x.v[25],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s109 = __shfl_sync(0xffffffff, x.v[27],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s110 = __shfl_sync(0xffffffff, x.v[29],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s111 = __shfl_sync(0xffffffff, x.v[31],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s112 = __shfl_sync(0xffffffff, x.v[24],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s113 = __shfl_sync(0xffffffff, x.v[26],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s114 = __shfl_sync(0xffffffff, x.v[28],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s115 = __shfl_sync(0xffffffff, x.v[30],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s116 = __shfl_sync(0xffffffff, x.v[25],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s117 = __shfl_sync(0xffffffff, x.v[27],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s118 = __shfl_sync(0xffffffff, x.v[29],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s119 = __shfl_sync(0xffffffff, x.v[31],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s120 = __shfl_sync(0xffffffff, x.v[24],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s121 = __shfl_sync(0xffffffff, x.v[26],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s122 = __shfl_sync(0xffffffff, x.v[28],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s123 = __shfl_sync(0xffffffff, x.v[30],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s124 = __shfl_sync(0xffffffff, x.v[25],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s125 = __shfl_sync(0xffffffff, x.v[27],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s126 = __shfl_sync(0xffffffff, x.v[29],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s127 = __shfl_sync(0xffffffff, x.v[31],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    Words<32> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[1] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[3] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[5] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[7] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[9] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[11] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[13] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[15] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[16] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[17] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[18] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[19] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[20] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[21] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[22] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[23] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[24] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[25] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[26] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[27] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[28] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[29] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[30] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[31] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    return out;
}

// Prepare each Half2 ONCE, with original norm/scale/den math, BEFORE this call.

template <class F> __device__ __forceinline__ Words<16> ff_a(F prepare) {

    Words<16> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> query_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> key_b(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> prob_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<8> attended_a(F prepare) {

    Words<8> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> value_b(F get) {
    Words<16> out;

    {
        u32 a = transpose_half2(get(activation::Tag<0>{})),
            b = transpose_half2(get(activation::Tag<1>{}));
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<24>{})),
            b = transpose_half2(get(activation::Tag<25>{}));
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<2>{})),
            b = transpose_half2(get(activation::Tag<3>{}));
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<26>{})),
            b = transpose_half2(get(activation::Tag<27>{}));
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<4>{})),
            b = transpose_half2(get(activation::Tag<5>{}));
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<28>{})),
            b = transpose_half2(get(activation::Tag<29>{}));
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<6>{})),
            b = transpose_half2(get(activation::Tag<7>{}));
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<30>{})),
            b = transpose_half2(get(activation::Tag<31>{}));
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<8>{})),
            b = transpose_half2(get(activation::Tag<9>{}));
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<16>{})),
            b = transpose_half2(get(activation::Tag<17>{}));
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<10>{})),
            b = transpose_half2(get(activation::Tag<11>{}));
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<18>{})),
            b = transpose_half2(get(activation::Tag<19>{}));
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<12>{})),
            b = transpose_half2(get(activation::Tag<13>{}));
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<20>{})),
            b = transpose_half2(get(activation::Tag<21>{}));
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<14>{})),
            b = transpose_half2(get(activation::Tag<15>{}));
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<22>{})),
            b = transpose_half2(get(activation::Tag<23>{}));
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

} // namespace packet_reference

namespace shallow {

using namespace endpoint;

using namespace activation;

template <int P> constexpr int logical_m16() {
    return P == 0 ? 0 : P == 1 ? 3 : P == 2 ? 1 : 2;
}
template <int Row> __device__ __forceinline__ half physical_den(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    return packed12::den_finish(s);
}
struct Entry {
    HC<64, 32> residual;
    Words<16> expand;
};

template <class F> __device__ __forceinline__ Entry entry(F get, const BodyWeights &w) {

    Entry e;

    Words<32> raw;
    each<32>([&](auto jt) { raw.v[decltype(jt)::value] = get(jt); });

    e.residual = packet_reference::residual_c(raw);

    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });

    Words<16> encoded;
    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;

        encoded.v[j] = packed12::e4four(raw.v[j * 2], raw.v[j * 2 + 1]);
    });
    e.expand = packet_reference::input_a(encoded);
    return e;
}

template <class F> __device__ __forceinline__ Entry load_entry(F get, const BodyWeights &w) {

    // Producer is scalar only at the memory/sample edge; no dynamically indexed Tile.

    auto x = fill<64, 32>(get);

    return entry([&](auto t) { return x.v[decltype(t)::value]; }, w);
}

template <int M, int N>
__device__ __forceinline__ void paired_weight(HC<M, N> &c, const Words<M / 4> &a, const u8 *w) {

    each<N / 16>([&](auto pt) {
        constexpr int p = decltype(pt)::value;

        // Original immutable B128 packet, shared by both N8 and all M16.

        const uint4 raw = *reinterpret_cast<const uint4 *>(w + p * 512 + lane() * 16);

        const uint2 b0 = make_uint2(raw.x, raw.y), b1 = make_uint2(raw.z, raw.w);

        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;

            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};

            each<2>([&](auto nt) {
                constexpr int n = p * 2 + decltype(nt)::value;

                uint2 b = decltype(nt)::value ? b1 : b0;
                u32 bb[2] = {b.x, b.y};

                Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
                mma8(z, aa, bb);

                c.v[(m * (N / 8) + n) * 2] = z.v[0];
                c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
            });
        });
    });
}

template <class F> __device__ __forceinline__ void publish_words(const HC<32, 32> &x, F publish) {

    each<2>([&](auto m) {
        each<4>([&](auto n) {
            each<2>([&](auto i) {
                publish(
                    m, n, i,
                    x.v[(decltype(m)::value * 4 + decltype(n)::value) * 2 + decltype(i)::value]);
            });
        });
    });
}

// Readout calls the original gemm16 twice with its original C handoff. Its A

// row/16, col/8 and row/8 are warp-uniform at this call site. Fixed references

// avoid rebuilding the projection Tile. This is not a general varying gather.

__device__ __forceinline__ half packet_half_uniform(const HC<32, 32> &x, int row, int col) {

    u32 word = 0;
    int bank = (row / 16 * 4 + col / 8) * 2 + (row / 8 % 2);

    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        if (bank == j)
            word = x.v[j];
    });

    return unpack(__shfl_sync(0xffffffff, word, (row % 8) * 4 + (col % 8) / 2), col % 2);
}

template <class Publish>
__device__ __forceinline__ void packet_body(Entry input, const BodyWeights &w, Publish publish) {

    auto ff = input.residual;

    {

        const auto a = input.expand;

#pragma unroll 1

        for (int part = 0; part < 4; ++part) {

            Words<16> ha;

            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;

                auto hidden = fill<64, 16>([](int, int) { return h(0); });

                paired_weight<64, 16>(hidden, a, w.expand + part * 1024 + pair * 512);

                // A hidden N16 pair dies after activation/encoding into final A.

                each<4>([&](auto mt) {
                    each<2>([&](auto it) {
                        constexpr int dest =
                            decltype(mt)::value * 4 + pair * 2 + decltype(it)::value;

                        // rawExpand C ownership, NOT the old decoded hidden consumer.

                        constexpr int bank = decltype(mt)::value * 4 + decltype(it)::value;

                        u32 x = packed12::activate_pair(hidden.v[bank]);

                        u32 y = packed12::activate_pair(hidden.v[bank + 2]);

                        ha.v[dest] = packed12::e4four(x, y);
                    });
                });
            });

            paired_weight<64, 32>(ff, ha, w.contract + part * 1024);
        }
    }

    Words<16> qa, kb, vb;

    {

        auto a = packet_reference::ff_a([&](auto jt) { return ff.v[decltype(jt)::value]; });

        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;

            auto z = fill<64, 32>([](int, int) { return h(0); });

            paired_weight<64, 32>(z, a, w.qkv + component * 1024);

            Words<8> inv;

            if constexpr (component < 2)
                each<8>([&](auto rt) {
                    constexpr int row = decltype(rt)::value;
                    inv.v[row] = packed12::splat(activation::norm<64, 32, row>(z));
                });

            auto get = [&](auto jt) {
                constexpr int j = decltype(jt)::value;
                u32 x = z.v[j];

                if constexpr (component < 2)
                    x = packed12::mul(x, inv.v[j / 8 * 2 + j % 2]);

                if constexpr (component == 0)
                    x = packed12::mul(x, packed12::splat(w.scale));

                return x;
            };

            // Directly encode final consumer words. No z96 or canonical Q/K/V.

            if constexpr (component == 0)
                qa = packet_reference::query_a(get);

            if constexpr (component == 1)
                kb = packet_reference::key_b(get);

            if constexpr (component == 2)
                vb = packet_reference::value_b(get);
        });
    }

    each<2>([&](auto st) {
        constexpr int slab = decltype(st)::value;

        Words<16> pa;

        {

            HC<32, 64> logits;
            each<2>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                u32 a[4] = {qa.v[(slab * 2 + m) * 4], qa.v[(slab * 2 + m) * 4 + 1],
                            qa.v[(slab * 2 + m) * 4 + 2], qa.v[(slab * 2 + m) * 4 + 3]};
                each<4>([&](auto pt) {
                    constexpr int p = decltype(pt)::value;
                    // qkv already includes the pre/UP/post ABI shift; original raw40 only.
                    u32 s0, s1, s2, s3;
                    const u8 *seed = w.qkv + 0xc00 + ((slab * 2 + m) * 4 + p) * 512 + lane() * 16;
                    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3)
                                 : "l"(seed));
                    each<2>([&](auto nt) {
                        constexpr int n = p * 2 + decltype(nt)::value;
                        u32 b[2] = {kb.v[n * 2], kb.v[n * 2 + 1]};
                        Fragment c{{decltype(nt)::value ? s2 : s0, decltype(nt)::value ? s3 : s1}};
                        mma8(c, a, b);
                        logits.v[(m * 8 + n) * 2] = c.v[0];
                        logits.v[(m * 8 + n) * 2 + 1] = c.v[1];
                    });
                });
            });
            each<32>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                logits.v[j] = packed12::exponent_pair(logits.v[j]);
            });

            Words<4> den;

            each<4>([&](auto rt) {
                constexpr int r = decltype(rt)::value;
                den.v[r] = packed12::splat(physical_den<r>(logits));
            });

            pa = packet_reference::prob_a([&](auto jt) {
                constexpr int j = decltype(jt)::value;

                return packed12::mul(logits.v[j], den.v[j / 16 * 2 + j % 2]);
            });
        }

        auto attended = fill<32, 32>([](int, int) { return h(0); });

        each<2>([&](auto pt) {
            constexpr int part = decltype(pt)::value;

            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 b[2] = {vb.v[part * 8 + n * 2], vb.v[part * 8 + n * 2 + 1]};

                each<2>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    u32 a[4] = {pa.v[part * 8 + m * 4], pa.v[part * 8 + m * 4 + 1],
                                pa.v[part * 8 + m * 4 + 2], pa.v[part * 8 + m * 4 + 3]};

                    Fragment c{{attended.v[(m * 4 + n) * 2], attended.v[(m * 4 + n) * 2 + 1]}};
                    mma8(c, a, b);

                    attended.v[(m * 4 + n) * 2] = c.v[0];
                    attended.v[(m * 4 + n) * 2 + 1] = c.v[1];
                });
            });
        });

        HC<32, 32> out;

        each<16>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

            out.v[j] = packed12::mul(ff.v[logical_m16<slab * 2 + j / 8>() * 8 + j % 8],
                                     pack(w.attn_gate[c], w.attn_gate[c + 1]));
        });

        auto aa = packet_reference::attended_a([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            return attended.v[j];
        });

        paired_weight<32, 32>(out, aa, w.projection);

        publish(Tag<logical_m16<slab * 2>() * 16>{}, Tag<logical_m16<slab * 2 + 1>() * 16>{}, out);
    });
}

} // namespace shallow

namespace joint {
using namespace shallow;
// Canonical channel represented by a raw-body residual C owner.
__device__ __forceinline__ int input_channel(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void gate(Entry &e, const BodyWeights &w) {
    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() & 3) * 2;
        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });
}
// Input is already in the residual owner's fixed Half registers. Quantization
// reads those original registers, independently of the gated C rail.
__device__ __forceinline__ Entry true_half(HC<64, 32> raw, const BodyWeights &w) {
    Entry e;
    e.residual = raw;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                e.expand.v[m * 4 + p * 2 + i] =
                    packed12::e4four(raw.v[m * 8 + p * 4 + i], raw.v[m * 8 + p * 4 + 2 + i]);
            });
        });
    });
    gate(e, w);
    return e;
}
template <class F> __device__ __forceinline__ Entry produce_half(F get, const BodyWeights &w) {
    return true_half(fill<64, 32>([&](int r, int c) { return get(r, input_channel(c)); }), w);
}
// Byte loads are predicated BEFORE memory access. Only a real aligned,
// consecutive, all-valid quartet may use the vector path.
template <class Address>
__device__ __forceinline__ u32 load4(const u8 *data, Address address, int row, int col) {
    int a = address(row, col), b = address(row, col + 1), c = address(row, col + 2),
        d = address(row, col + 3);
    if (a >= 0 && b == a + 1 && c == a + 2 && d == a + 3 &&
        ((reinterpret_cast<unsigned long long>(data + a) & 3) == 0))
        return *reinterpret_cast<const u32 *>(data + a);
    return u32(a < 0 ? 0 : data[a]) | (u32(b < 0 ? 0 : data[b]) << 8) |
           (u32(c < 0 ? 0 : data[c]) << 16) | (u32(d < 0 ? 0 : data[d]) << 24);
}
template <class Address>
__device__ __forceinline__ Entry e4_edge(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                int row = m * 16 + lane() / 4 + i * 8, col = (lane() & 3) * 8 + p * 4;
                u32 raw = load4(data, address, row, col), a = raw;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    u8 v = raw >> (b * 8);
                    // Keep the old NaN round-trip, including its sign/canonicalization.
                    if ((v & 127) == 127)
                        a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
                });
                e.expand.v[m * 4 + p * 2 + i] = a;
                e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
                e.residual.v[m * 8 + p * 4 + 2 + i] =
                    pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
            });
        });
    });
    gate(e, w);
    return e;
}
// HalfSlab consumer: same address recipe is used for both bytes of the pair.
// Optional pre never controls the separate mandatory shared Half pool result.
template <class Address>
__device__ __forceinline__ void publish_pair(u8 *data, half *pre, Address address, int row, int col,
                                             u32 word) {
    int a = address(row, col), b = address(row, col + 1);
    half x = unpack(word, 0), y = unpack(word, 1);
    if (a >= 0 && b == a + 1 && ((reinterpret_cast<unsigned long long>(data + a) & 1) == 0))
        *reinterpret_cast<unsigned short *>(data + a) = unsigned(e4(x)) | (unsigned(e4(y)) << 8);
    else {
        if (a >= 0)
            data[a] = e4(x);
        if (b >= 0)
            data[b] = e4(y);
    }
    if (pre) {
        if (a >= 0)
            pre[a] = x;
        if (b >= 0)
            pre[b] = y;
    }
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed8(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
            u32 v = 0;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                v |= u32(av(r, k + j)) << (j * 8);
            });
            a[i] = v;
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + lane() / 4;
                u32 v = 0;
                each<4>([&](auto jt) {
                    constexpr int j = decltype(jt)::value;
                    v |= u32(bv(k + j, col)) << (j * 8);
                });
                b[i] = v;
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma8(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed16(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
            a[i] = pack(av(r, k), av(r, k + 1));
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + lane() / 4;
                b[i] = pack(bv(k, col), bv(k + 1, col));
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma16(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// A for the original readout K16 is exactly the projection C's pair at
// [m, kpart*2+i/2, i%2]. No runtime bank selection or projection Tile.
__device__ __forceinline__ HC<32, 8> read_head(const HC<32, 32> &out, const half *weights) {
    auto head = fill<32, 8>([](int, int) { return h(0); });
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4], b[2];
            each<4>([&](auto it) {
                constexpr int i = decltype(it)::value;
                a[i] = out.v[m * 8 + kp * 4 + (i / 2) * 2 + i % 2];
            });
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = kp * 16 + (lane() & 3) * 2 + i * 8, n = lane() / 4;
                b[i] = pack(weights[k * 8 + n], weights[(k + 1) * 8 + n]);
            });
            Fragment z{{head.v[m * 2], head.v[m * 2 + 1]}};
            mma16(z, a, b);
            head.v[m * 2] = z.v[0];
            head.v[m * 2 + 1] = z.v[1];
        });
    });
    return head;
}
template <int Col> __device__ __forceinline__ half head_channel(const HC<32, 8> &head) {
    u32 result = 0;
    each<4>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        u32 v = __shfl_sync(0xffffffff, head.v[j], (lane() % 8) * 4 + Col / 2);
        if (lane() / 8 == j)
            result = v;
    });
    return unpack(result, Col % 2);
}
} // namespace joint
namespace physical {
using namespace shallow;
// Each tin packet holds (row,row+8) x (four+four channels). The raw A
// consumer and projection C producer use this exact same lane ownership.
__device__ __forceinline__ uint4 load(const u8 *p, int a) {
    if (a < 0)
        return make_uint4(0, 0, 0, 0);
    return *reinterpret_cast<const uint4 *>(p + a);
}
__device__ __forceinline__ int nc(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ int result_index(int r, int c) {
    return (r / 16) * 512 + (r % 8) * 64 + (r % 16 / 8) * 4 + 16 * (c % 8 / 2) + (c / 16) * 8 +
           (c % 16 / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void raw_word(Entry &e, int m, int p, int i, u32 raw) {
    u32 a = raw;
    each<4>([&](auto bt) {
        constexpr int b = decltype(bt)::value;
        u8 v = raw >> (b * 8);
        if ((v & 127) == 127)
            a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
    });
    e.expand.v[m * 4 + p * 2 + i] = a;
    e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
    e.residual.v[m * 8 + p * 4 + 2 + i] = pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
}
template <class Address>
__device__ __forceinline__ Entry input_tin(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 4, (lane() & 3) * 8));
        raw_word(e, m, 0, 0, v.x);
        raw_word(e, m, 0, 1, v.y);
        raw_word(e, m, 1, 0, v.z);
        raw_word(e, m, 1, 1, v.w);
    });
    joint::gate(e, w);
    return e;
}
template <class Address>
__device__ __forceinline__ Entry input_compact(const u8 *data, Address address,
                                               const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 2, (lane() & 1) * 4));
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int p = decltype(pt)::value, i = decltype(it)::value;
                int src = (lane() / 4 + i * 8) * 2 + p;
                u32 a = __shfl_sync(0xffffffff, v.x, src), b = __shfl_sync(0xffffffff, v.y, src);
                u32 c = __shfl_sync(0xffffffff, v.z, src), d = __shfl_sync(0xffffffff, v.w, src);
                u32 raw = (lane() % 4 == 0) ? a : (lane() % 4 == 1) ? b : (lane() % 4 == 2) ? c : d;
                raw_word(e, m, p, i, raw);
            });
        });
    });
    joint::gate(e, w);
    return e;
}
// Same C owner publishes E4 memory and the unquantized shared Half rail.
// The shared layout is consumed directly by the four-point pool, not redecoded.
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_tin(const HC<32, 32> &z, u8 *data, half *pre, half *result,
                                            Address address) {
    each<2>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        int row = (m == 0 ? Base0 : Base1) + lane() / 4, col = (lane() & 3) * 2;
        int a = address(row, col);
        uint4 v = make_uint4(packed12::e4four(z.v[m * 8], z.v[m * 8 + 2]),
                             packed12::e4four(z.v[m * 8 + 1], z.v[m * 8 + 3]),
                             packed12::e4four(z.v[m * 8 + 4], z.v[m * 8 + 6]),
                             packed12::e4four(z.v[m * 8 + 5], z.v[m * 8 + 7]));
        if (a >= 0)
            *reinterpret_cast<uint4 *>(data + a) = v;
        if (pre || result) {
            uint4 lo = make_uint4(z.v[m * 8], z.v[m * 8 + 2], z.v[m * 8 + 1], z.v[m * 8 + 3]);
            uint4 hi = make_uint4(z.v[m * 8 + 4], z.v[m * 8 + 6], z.v[m * 8 + 5], z.v[m * 8 + 7]);
            if (pre && a >= 0) {
                *reinterpret_cast<uint4 *>(pre + a) = lo;
                *reinterpret_cast<uint4 *>(pre + a + 8) = hi;
            }
            if (result) {
                int s = (m == 0 ? Base0 : Base1) / 16 * 512 + lane() * 16;
                *reinterpret_cast<uint4 *>(result + s) = lo;
                *reinterpret_cast<uint4 *>(result + s + 8) = hi;
            }
        }
    });
}
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_compact(const HC<32, 32> &z, u8 *data, half *pre,
                                                Address address) {
    each<2>([&](auto mt) {
        each<2>([&](auto pt) {
            constexpr int m = decltype(mt)::value, p = decltype(pt)::value;
            u32 lo = packed12::e4four(z.v[m * 8 + p * 4], z.v[m * 8 + p * 4 + 2]);
            u32 hi = packed12::e4four(z.v[m * 8 + p * 4 + 1], z.v[m * 8 + p * 4 + 3]);
            uint4 v;
            each<4>([&](auto kt) {
                constexpr int k = decltype(kt)::value;
                int src = (lane() % 8) * 4 + k;
                u32 a = __shfl_sync(0xffffffff, lo, src), b = __shfl_sync(0xffffffff, hi, src),
                    q = lane() < 8 ? a : b;
                if constexpr (k == 0)
                    v.x = q;
                if constexpr (k == 1)
                    v.y = q;
                if constexpr (k == 2)
                    v.z = q;
                if constexpr (k == 3)
                    v.w = q;
            });
            if (lane() < 16) {
                int a = address((m == 0 ? Base0 : Base1) + lane(), p * 16);
                if (a >= 0)
                    *reinterpret_cast<uint4 *>(data + a) = v;
            }
            // Optional true-Half ABI remains in C ownership; no E4 reconstruction.
            if (pre) {
                each<2>([&](auto nt) {
                    each<2>([&](auto it) {
                        constexpr int n = p * 2 + decltype(nt)::value, i = decltype(it)::value;
                        int row = (m == 0 ? Base0 : Base1) + lane() / 4 + i * 8,
                            col = n * 8 + (lane() & 3) * 2, a = address(row, col);
                        if (a >= 0)
                            *reinterpret_cast<u32 *>(pre + a) = z.v[m * 8 + n * 2 + i];
                    });
                });
            }
        });
    });
}
} // namespace physical

#ifndef NR_H
#error Native NR must compile with its real packet height
#endif
#ifndef NR_W
#error Native NR must compile with its real packet width
#endif
namespace shallow_static {
__device__ __forceinline__ int height() { return NR_H / 2; }
__device__ __forceinline__ int width() { return NR_W / 2; }
__device__ __forceinline__ int token(int r) {
    return (r & 48) | ((r & 7) << 1) | ((r & 8) >> 3);
}
__device__ __forceinline__ int pixel(int r) {
    return (r & 3) ^ ((r & 12) << 1) ^ ((r & 16) << 1) ^ ((r & 32)) ^ ((r & 32) >> 3);
}
__device__ __forceinline__ int pool_local(int p) {
    return (p & 3) ^ ((p & 4) << 2) ^ ((p & 4) << 3) ^ ((p & 56) >> 1);
}
__device__ __forceinline__ int ctop(int t) {
    return ((t & 1) << 4) ^ ((t & 6) >> 1) ^ (t & 8) ^ ((t & 16) << 1) ^ (t & 32) ^ ((t & 32) >> 3);
}
__device__ __forceinline__ int ptoc(int t) {
    return ((t & 3) << 1) ^ ((t & 4) << 2) ^ ((t & 4) << 3) ^ (t & 8) ^ ((t & 16) >> 4) ^
           ((t & 32) >> 1);
}
__device__ __forceinline__ int compact(int y, int x, int c) {
    int t = ctop((y & 7) * 8 + (x & 7)), k = (c & 3) | ((c & 4) << 2) | ((c & 24) >> 1);
    int l = (k >> 4) * 1024 + t * 16 + (k & 15);
    return (((y >> 3) + (height() / 8) * (l >> 10)) * (width() / 4) + (x >> 5) + (width() / 32) * ((l >> 7) & 7)) *
               512 +
           ((x >> 3) & 3) * 128 + (l & 127);
}
template <int Seq, bool Output = false> __device__ __forceinline__ int address(int row, int c) {
    constexpr int phase = Seq >= 151 ? Seq - 151 : Seq - 3;
    const int gx = width() / 8 + ((phase == 1 || phase == 2) ? 1 : 0);
    constexpr int sx = (phase == 1 || phase == 2) ? -1 : 0,
                  sy = (phase == 1 || phase == 3) ? -1 : 0;
    int stripe = row >> 4, cy = int(blockIdx.x) / gx * 2 + sy + (stripe == 1 || stripe == 2),
        cx = int(blockIdx.x) % gx * 2 + sx + (stripe == 2 || stripe == 3);
    if (cy < 0 || cy >= height() / 4 || cx < 0 || cx >= width() / 4)
        return -1;
    if constexpr (Output)
        c = physical::nc(c);
    if constexpr ((Seq == 3 && !Output) || (Seq == 154 && Output)) {
        int s0 = (cy & 1) ? 1 + (cx & 1) : (cx & 1) * 3;
        int y = (cy >> 1) * 8 + s0 * 2 + ((row >> 2) & 1),
            x = (cx >> 1) * 8 + ((row & 3) << 1) + ((row >> 3) & 1);
        return compact(y, x, c);
    }
    return (cy * (width() / 4) + cx) * 512 + (row & 7) * 64 + ((row >> 3) & 1) * 4 + 8 * (c >> 2) +
           (c & 3);
}
__device__ __forceinline__ int up_address(int row, int c) {
    int low = row & 63, outer = row >> 6;
    int y = (outer / (width() / 32)) * 4 + ((low >> 3) & 1) + 2 * (low & 1);
    int x = (outer % (width() / 32)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) +
            4 * ((low >> 4) & 1) + 8 * ((low >> 5) & 1);
    return (y * (width() / 2) + x) * 16 + (c >> 4) * (height() / 2) * (width() / 2) * 16 + (c & 15);
}
__device__ __forceinline__ int pool_row(int cta, int owner) {
    int by = cta / (width() / 8), bx = cta % (width() / 8);
    if ((by == 0 || by == height() / 8) && owner >= 8)
        return -1;
    int y = (by == 0 ? 0 : by * 4 - 2) + (owner >> 2);
    return y * (width() / 2) + bx * 4 + (owner & 3);
}
__device__ __forceinline__ int pool_address(int row, int c) {
    return row * 16 + (c >> 4) * (height() / 2) * (width() / 2) * 16 + (c & 1) + ((c & 6) << 1) + ((c & 8) >> 2);
}
} // namespace shallow_static
using namespace activation;
using namespace shallow;
using namespace endpoint;
__device__ __forceinline__ int tin_offset(int y, int x, int c, int width) {
    int t = (y & 7) * 8 + (x & 7), p = 64 * (t / 2) + 4 * (t & 1) + 8 * (c / 4) + (c & 3),
        s = p / 512;
    int cy = (y / 8) * 2 + (s == 1 || s == 2), cx = (x / 8) * 2 + (s == 2 || s == 3);
    return (cy * (width / 4) + cx) * 512 + (p & 511);
}
namespace transition {
// Only canonical butterfly lanes are gathered; both banks shuffle uniformly
// before the destination chooses. No lane-dependent indexing of C registers.
template <int Bank> __device__ __forceinline__ u32 average(const HC<32, 32> &z) {
    u32 p = packed12::add(z.v[Bank], __shfl_xor_sync(0xffffffff, z.v[Bank], 4));
    return packed12::mul(packed12::add(p, __shfl_xor_sync(0xffffffff, p, 16)),
                         pack(h(.25f), h(.25f)));
}
template <int Bank> __device__ __forceinline__ u32 gather(const HC<32, 32> &z) {
    int src = (lane() & 3) + ((lane() & 4) << 1);
    u32 lo = __shfl_sync(0xffffffff, average<Bank>(z), src);
    u32 hi = __shfl_sync(0xffffffff, average<Bank + 1>(z), src);
    return (lane() & 16) ? hi : lo;
}
template <int Base0, int Base1>
__device__ __forceinline__ void pool_a(Words<4> &a, const HC<32, 32> &z, int by) {
    each<2>([&](auto mt) {
        constexpr int m = decltype(mt)::value, logical = (m == 0 ? Base0 : Base1) / 16,
                      Slab = logical / 2, cm = logical % 2;
        // These conditions are CTA-uniform, including top and bottom ragged rows.
        if ((by != 0 || cm == 1 - Slab) && (by != NR_H / 16 || cm == Slab)) {
            each<2>([&](auto kt) {
                constexpr int k = decltype(kt)::value;
                u32 lo = gather<m * 8 + k * 4>(z), hi = gather<m * 8 + k * 4 + 2>(z);
                u32 word = packed12::e4four(lo, hi);
                if (((lane() >> 3) & 1) == Slab) {
                    if (by == 0 || by == NR_H / 16)
                        a.v[k * 2] |= word;
                    else if constexpr ((cm ^ Slab) == 0)
                        a.v[k * 2] |= word;
                    else
                        a.v[k * 2 + 1] |= word;
                }
            });
        }
    });
}
// Every pixel/N16 is one aligned packet in the original planar physical order.
__device__ __forceinline__ void publish_pool(const HC<16, 64> &z, u8 *data, const int *rows,
                                             const int *offsets) {
    each<4>([&](auto pt) {
        constexpr int p = decltype(pt)::value;
        u32 lo = packed12::e4four(z.v[p * 4], z.v[p * 4 + 2]);
        u32 hi = packed12::e4four(z.v[p * 4 + 1], z.v[p * 4 + 3]);
        uint4 v;
        each<4>([&](auto kt) {
            constexpr int k = decltype(kt)::value;
            int src = (lane() % 8) * 4 + k;
            u32 x = __shfl_sync(0xffffffff, lo, src), y = __shfl_sync(0xffffffff, hi, src);
            u32 q = lane() < 8 ? x : y;
            if constexpr (k == 0)
                v.x = q;
            if constexpr (k == 1)
                v.y = q;
            if constexpr (k == 2)
                v.z = q;
            if constexpr (k == 3)
                v.w = q;
        });
        if (lane() < 16) {
            int row = shallow_static::pool_row(int(blockIdx.x), lane());
            if (row >= 0)
                *reinterpret_cast<uint4 *>(data + shallow_static::pool_address(row, p * 16)) = v;
        }
    });
}
} // namespace transition
extern "C" __global__ __launch_bounds__(32) void outer1_static_ds(
    u8 *r, BodyWeights w, const int *im, const int *om, int in, int out, int counter,
    const half *mixed, half *pre, const u8 *pool_weight, const int *pool_rows, const int *pool_cols,
    const int *pool_out, int compact_out) {
    Words<4> a{{0, 0, 0, 0}};
    auto input = physical::input_tin(
        r + in, [&](int t, int c) { return shallow_static::address<6>(t, c); }, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &output) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        physical::publish_tin<base0, base1>(output, r + out, pre, nullptr, [&](int t, int c) {
            return shallow_static::address<6, true>(t, c);
        });
        transition::pool_a<base0, base1>(a, output, int(blockIdx.x) / (NR_W / 16));
    });
    auto pooled = fill<16, 64>([](int, int) { return h(0); });
    // BodyWeights.expand is the original live resource40 record, not canonical B.
    paired_weight<16, 64>(pooled, a, w.expand + 0x50b0);
    transition::publish_pool(pooled, r + compact_out, pool_rows, pool_out);
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}
extern "C" __global__ __launch_bounds__(32) void outer1_static_up(
    u8 *r, BodyWeights w, const int *im, const int *om, int in, int out, int counter, half *mixed,
    half *pre, const u8 *up_weight, const int *up_input, const int *source_rows,
    const int *local_rows, const int *local_cols, const half *gate, int skip) {
    auto z = fill<16, 32>([](int, int) { return h(0); });
#pragma unroll 1
    for (int kp = 0; kp < 2; kp++) {
        Words<4> a;
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int row = int(blockIdx.x) * 16 + lane() / 4 + (i & 1) * 8;
            int k = kp * 32 + (lane() & 3) * 4 + (i / 2) * 16;
            // Four K4 consumers share exactly one aligned original 16-byte packet.
            uint4 packet = make_uint4(0, 0, 0, 0);
            if ((lane() & 3) == 0)
                packet =
                    *reinterpret_cast<const uint4 *>(r + in + shallow_static::up_address(row, k));
            int src = lane() & ~3;
            u32 q0 = __shfl_sync(0xffffffff, packet.x, src),
                q1 = __shfl_sync(0xffffffff, packet.y, src);
            u32 q2 = __shfl_sync(0xffffffff, packet.z, src),
                q3 = __shfl_sync(0xffffffff, packet.w, src);
            a.v[i] = (lane() & 3) == 0 ? q0 : (lane() & 3) == 1 ? q1 : (lane() & 3) == 2 ? q2 : q3;
        });
        paired_weight<16, 32>(z, a, w.expand + 0x2000 + kp * 1024);
    }
    HC<64, 32> raw;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        int base = shallow_static::address<151>(m * 16 + lane() / 4, (lane() & 3) * 8);
        uint4 sk = physical::load(r + skip, base);
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int p = decltype(pt)::value, i = decltype(it)::value;
                int col = (lane() & 3) * 8 + p * 4;
                u32 word;
                if constexpr (p == 0 && i == 0)
                    word = sk.x;
                if constexpr (p == 0 && i == 1)
                    word = sk.y;
                if constexpr (p == 1 && i == 0)
                    word = sk.z;
                if constexpr (p == 1 && i == 1)
                    word = sk.w;
                each<2>([&](auto jt) {
                    constexpr int j = decltype(jt)::value, b = p * 4 + j * 2 + i;
                    constexpr int origin = m == 0 ? 0 : m == 1 ? 4 : m == 2 ? 20 : 16;
                    u32 projected = __shfl_sync(0xffffffff, z.v[b], (lane() & 11) + origin);
                    half x = h(0), y = h(0);
                    if (base >= 0) {
                        x = __hfma(une4(u8(word >> (j * 16))), gate[col + j * 2],
                                   unpack(projected, 0));
                        y = __hfma(une4(u8(word >> (j * 16 + 8))), gate[col + j * 2 + 1],
                                   unpack(projected, 1));
                    }
                    raw.v[m * 8 + b] = pack(x, y);
                });
            });
        });
        if (mixed && base >= 0) {
            *reinterpret_cast<uint4 *>(mixed + base) =
                make_uint4(raw.v[m * 8], raw.v[m * 8 + 2], raw.v[m * 8 + 1], raw.v[m * 8 + 3]);
            *reinterpret_cast<uint4 *>(mixed + base + 8) =
                make_uint4(raw.v[m * 8 + 4], raw.v[m * 8 + 6], raw.v[m * 8 + 5], raw.v[m * 8 + 7]);
        }
    });
    auto input = joint::true_half(raw, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &output) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        physical::publish_tin<base0, base1>(output, r + out, pre, nullptr, [&](int t, int c) {
            return shallow_static::address<151, true>(t, c);
        });
    });
    if (counter >= 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

} // namespace nr_outer_transition_1
// ============================================================================
// OUTER post_static_packet/post.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_post_2 {
// Experimental shared arithmetic component. Not a qualified native endpoint.
// All Tile storage is distributed over exactly one warp in MMA C coordinates.
namespace endpoint {
using u32 = unsigned;
using u8 = unsigned char;
__device__ __forceinline__ int lane() {
    return threadIdx.x & 31;
}
__device__ __forceinline__ half h(float x) {
    return __float2half_rn(x);
}
__device__ __forceinline__ float f(half x) {
    return __half2float(x);
}
__device__ __forceinline__ u32 pack(half a, half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}
__device__ __forceinline__ half unpack(u32 x, int i) {
    return __ushort_as_half(x >> (16 * i));
}
__device__ __forceinline__ u8 e4(half x) {
    return __nv_cvt_halfraw_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ half une4(u8 x) {
    return __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
}
__device__ __forceinline__ half q(half x) {
    return une4(e4(x));
}
__device__ __forceinline__ half activate(half x) {
    float c = fminf(4.f, fmaxf(-4.f, f(x)));
    half g = h(__fmaf_rn(fabsf(c), -.055908203125f, .447265625f));
    return __hmul(x, h(__fmaf_rn(c, f(g), .89453125f)));
}
__device__ __forceinline__ half exponent(half x) {
    half a = h(__fmaf_rn(f(x), .044921875f, 1.30078125f));
    a = h(fminf(1.5693359375f, fmaxf(1.03125f, f(a))));
    return __ushort_as_half((unsigned(__half_as_ushort(a)) << 5) + 32768);
}
__device__ __forceinline__ half reciprocal(half x) {
    float z;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
__device__ __forceinline__ half rsqrt_half(half x) {
    float z;
    asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(z) : "f"(f(x)));
    return h(z);
}
struct Fragment {
    u32 v[2];
};
__device__ __forceinline__ void mma8(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma16(Fragment &c, const u32 a[4], const u32 b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c.v[0]), "+r"(c.v[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int M, int N> struct Tile {
    static_assert(M % 16 == 0 && N % 8 == 0, "MMA tile dimensions");
    Fragment c[M / 16][N / 8];
    // Collective arbitrary gather. Both words of every bank are shuffled before
    // selection: selecting a dynamic word BEFORE shfl is incorrect when the
    // destination lanes request different rows. No lane-private full window.
    __device__ half get(int row, int col) const {
        u32 bits = 0;
        int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
                u32 a = __shfl_sync(0xffffffff, c[m][n].v[0], src);
                u32 b = __shfl_sync(0xffffffff, c[m][n].v[1], src);
                if (m == row / 16 && n == col / 8)
                    bits = (row & 8) ? b : a;
            }
        }
        return unpack(bits, col & 1);
    }
    template <class F> __device__ void fill(F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
            for (int n = 0; n < N / 8; ++n) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r = m * 16 + (lane() >> 2) + i * 8, k = n * 8 + (lane() & 3) * 2;
                    half a = fn(r, k), b = fn(r, k + 1);
                    c[m][n].v[i] = pack(a, b);
                }
            }
        }
    }
};
// Accessors return encoded E4 bytes in physical MMA K order.
template <int M, int N, class A, class B> __device__ void gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = 0;
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    a[i] |= u32(av(r, k + j)) << (j * 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
template <int M, int N, class A, class B> __device__ void gemm16(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
                half x = av(r, k), y = av(r, k + 1);
                a[i] = pack(x, y);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + (lane() >> 2);
                half x = bv(k, col), y = bv(k + 1, col);
                b[i] = pack(x, y);
            }
            mma16(c.c[m][n], a, b);
        }
    }
}
struct BodyWeights {
    // Dense decoded logical matrices, NOT a compressed/repacked static-weight
    // candidate. Host route construction must follow physical_1h and archive.
    const u8 *expand;                 // [32,128], hidden_inverse applied to each stream
    const u8 *contract;               // [4,32,32], stream,Khalf order 0/0,1/0,0/32,1/32
    const u8 *qkv;                    // [32,96], Qe Qo Ke Ko Ve Vo
    const u8 *projection;             // [32,32]
    const half *bias;                 // [64,64], physical bias half -> query/key route
    const half *ffn_gate, *attn_gate; // output-N order
    const int *cp, *rc, *oi, *ai, *pk;
    half scale;
};
__device__ half norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = z.get(row, base + j), v1 = z.get(row, base + 8 + j);
        half v2 = z.get(row, base + 16 + j), v3 = z.get(row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = e.get(row, j), b = e.get(row, j + 8);
        g[j] = __hadd(a, b);
        a = e.get(row, j + 48);
        b = e.get(row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 16);
        b = e.get(row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = e.get(row, j + 32);
        b = e.get(row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ int qk_order(int k) {
    return (k / 16) * 16 + (k % 2) + ((k / 2) % 2) * 8 + ((k / 4) % 4) * 2;
}
// Complete shared 64-token arithmetic; input is unquantized Half in canonical
// channels and physical token order. Output callback consumes two 32-row slabs
// without a global latent publication. Caller supplies sampling/codec/readout.
// This function has NOT been independently oracle-qualified.
template <class Publish>
__device__ void body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill([&](int r, int c) { return __hmul(input.get(r, w.rc[c]), w.ffn_gate[c]); });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return e4(input.get(r, w.cp[k])); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
// Direct C-word transformation has no cross-lane reads/aliasing.
#pragma unroll 1
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = hidden.c[m][n].v[i];
                    hidden.c[m][n].v[i] =
                        pack(q(activate(unpack(a, 0))), q(activate(unpack(a, 1))));
                }
        gemm8(
            ff, [&](int r, int k) { return e4(hidden.get(r, k)); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    gemm8(
        z, [&](int r, int k) { return e4(ff.get(r, w.oi[k])); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = norm(z, r, 0);
            ik[m * 2 + i] = norm(z, r, 32);
        }
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n)
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                z.c[m][n].v[i] = pack(q(a), q(b));
            }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return e4(z.get(slab * 32 + r, qk_order(k))); },
            [&](int k, int n) { return e4(z.get(n, 32 + qk_order(k))); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = denominator(logits, m * 16 + (lane() >> 2) + i * 8);
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    logits.c[m][n].v[i] =
                        pack(q(__hmul(unpack(a, 0), d)), q(__hmul(unpack(a, 1), d)));
                }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return e4(logits.get(r, w.pk[part * 32 + k])); },
                [&](int k, int n) { return e4(z.get(w.pk[part * 32 + k], 64 + n)); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) { return __hmul(ff.get(slab * 32 + r, n), w.attn_gate[n]); });
        gemm8(
            out, [&](int r, int k) { return e4(attended.get(r, w.ai[k])); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// Call sites have warp-uniform row/16 and row/8. Channel routes may vary
// between lanes, so only that bank dimension is scanned before selection.
template <int M, int N> __device__ half packed_half_row(const Tile<M, N> &t, int row, int col) {
    u32 bits = 0;
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
#pragma unroll
    for (int n = 0; n < N / 8; ++n) {
        u32 x = __shfl_sync(0xffffffff, t.c[row / 16][n].v[(row >> 3) & 1], src);
        if (n == col / 8)
            bits = x;
    }
    return unpack(bits, col & 1);
}
template <int M, int N> __device__ half packed_half_uniform(const Tile<M, N> &t, int row, int col) {
    int src = ((row & 7) << 2) | ((col & 7) >> 1);
    return unpack(__shfl_sync(0xffffffff, t.c[row / 16][col / 8].v[(row >> 3) & 1], src), col & 1);
}
// Four encoded E4 values per C bank: row0 pair in bytes 0/1, row8
// pair in bytes 2/3. This is a copy, never the Half residual storage.
// One word shuffle selects both rows; no decode-to-Half/re-encode roundtrip.
template <int M, int N> struct PackedE4 {
    u32 c[M / 16][N / 8];
    template <class F> __device__ void encode(const Tile<M, N> &t, F fn) {
#pragma unroll 1
        for (int m = 0; m < M / 16; ++m)
            for (int n = 0; n < N / 8; ++n) {
                u32 a = t.c[m][n].v[0], b = t.c[m][n].v[1];
                c[m][n] = u32(e4(fn(unpack(a, 0)))) | (u32(e4(fn(unpack(a, 1)))) << 8) |
                          (u32(e4(fn(unpack(b, 0)))) << 16) | (u32(e4(fn(unpack(b, 1)))) << 24);
            }
    }
    __device__ u8 row(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int n = 0; n < N / 8; ++n) {
            u32 x = __shfl_sync(0xffffffff, c[r / 16][n], src);
            if (n == k / 8)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
    __device__ u8 col(int r, int k) const {
        u32 bits = 0;
        int src = ((r & 7) << 2) | ((k & 7) >> 1);
#pragma unroll
        for (int m = 0; m < M / 16; ++m) {
            u32 x = __shfl_sync(0xffffffff, c[m][k / 8], src);
            if (m == r / 16)
                bits = x;
        }
        return bits >> (((r & 8) ? 16 : 0) + (k & 1) * 8);
    }
};
__device__ half packed_norm(const Tile<64, 96> &z, int row, int base) {
    half c[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half v0 = packed_half_uniform(z, row, base + j),
             v1 = packed_half_uniform(z, row, base + 8 + j);
        half v2 = packed_half_uniform(z, row, base + 16 + j),
             v3 = packed_half_uniform(z, row, base + 24 + j);
        c[j] = __hadd(__hfma(v0, v0, __hmul(v2, v2)), __hfma(v1, v1, __hmul(v3, v3)));
    }
    half a = __hadd(__hadd(c[0], c[4]), __hadd(c[2], c[6]));
    half b = __hadd(__hadd(c[1], c[5]), __hadd(c[3], c[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
__device__ half packed_denominator(const Tile<32, 64> &e, int row) {
    half g[8];
#pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        half a = packed_half_uniform(e, row, j), b = packed_half_uniform(e, row, j + 8);
        g[j] = __hadd(a, b);
        a = packed_half_uniform(e, row, j + 48);
        b = packed_half_uniform(e, row, j + 56);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 16);
        b = packed_half_uniform(e, row, j + 24);
        g[j] = __hadd(g[j], __hadd(a, b));
        a = packed_half_uniform(e, row, j + 32);
        b = packed_half_uniform(e, row, j + 40);
        g[j] = __hadd(g[j], __hadd(a, b));
    }
    half a = __hadd(__hadd(__hadd(g[0], g[2]), g[4]), g[6]);
    half b = __hadd(__hadd(__hadd(g[1], g[3]), g[5]), g[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
template <class Publish>
__device__ void packed_body(const Tile<64, 32> &input, const BodyWeights &w, Publish publish) {
    Tile<64, 32> ff;
    ff.fill(
        [&](int r, int c) { return __hmul(packed_half_row(input, r, w.rc[c]), w.ffn_gate[c]); });
    PackedE4<64, 32> input8;
    input8.encode(input, [](half x) { return x; });
#pragma unroll 1
    for (int part = 0; part < 4; ++part) {
        Tile<64, 32> hidden;
        hidden.fill([](int, int) { return h(0); });
        int start = (part % 2) * 64 + (part / 2) * 32;
        gemm8(
            hidden, [&](int r, int k) { return input8.row(r, w.cp[k]); },
            [&](int k, int n) { return w.expand[k * 128 + start + n]; });
        PackedE4<64, 32> hidden8;
        hidden8.encode(hidden, [](half x) { return activate(x); });
        gemm8(
            ff, [&](int r, int k) { return hidden8.row(r, k); },
            [&](int k, int n) { return w.contract[(part * 32 + k) * 32 + n]; });
    }
    Tile<64, 96> z;
    z.fill([](int, int) { return h(0); });
    PackedE4<64, 32> ff8;
    ff8.encode(ff, [](half x) { return x; });
    gemm8(
        z, [&](int r, int k) { return ff8.row(r, w.oi[k]); },
        [&](int k, int n) { return w.qkv[k * 96 + n]; });
    half iq[8], ik[8];
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 2; ++i) {
            int r = m * 16 + (lane() >> 2) + i * 8;
            iq[m * 2 + i] = packed_norm(z, r, 0);
            ik[m * 2 + i] = packed_norm(z, r, 32);
        }
    PackedE4<64, 96> z8;
#pragma unroll 1
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 12; ++n) {
            u32 word = 0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                u32 bits = z.c[m][n].v[i];
                half a = unpack(bits, 0), b = unpack(bits, 1);
                if (n < 4) {
                    a = __hmul(__hmul(a, iq[m * 2 + i]), w.scale);
                    b = __hmul(__hmul(b, iq[m * 2 + i]), w.scale);
                } else if (n < 8) {
                    a = __hmul(a, ik[m * 2 + i]);
                    b = __hmul(b, ik[m * 2 + i]);
                }
                word |= (u32(e4(a)) | (u32(e4(b)) << 8)) << (16 * i);
            }
            z8.c[m][n] = word;
        }
#pragma unroll 1
    for (int slab = 0; slab < 2; ++slab) {
        Tile<32, 64> logits;
        logits.fill([&](int r, int c) { return w.bias[(slab * 32 + r) * 64 + c]; });
        gemm8(
            logits, [&](int r, int k) { return z8.row(slab * 32 + r, qk_order(k)); },
            [&](int k, int n) { return z8.row(n, 32 + qk_order(k)); });
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n)
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    logits.c[m][n].v[i] = pack(exponent(unpack(a, 0)), exponent(unpack(a, 1)));
                }
        half inv[4];
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int i = 0; i < 2; ++i)
                inv[m * 2 + i] = packed_denominator(logits, m * 16 + (lane() >> 2) + i * 8);
        PackedE4<32, 64> probs;
#pragma unroll 1
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 8; ++n) {
                u32 word = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u32 a = logits.c[m][n].v[i];
                    half d = inv[m * 2 + i];
                    word |=
                        (u32(e4(__hmul(unpack(a, 0), d))) | (u32(e4(__hmul(unpack(a, 1), d))) << 8))
                        << (16 * i);
                }
                probs.c[m][n] = word;
            }
        Tile<32, 32> attended;
        attended.fill([](int, int) { return h(0); });
#pragma unroll 1
        for (int part = 0; part < 2; ++part)
            gemm8(
                attended, [&](int r, int k) { return probs.row(r, w.pk[part * 32 + k]); },
                [&](int k, int n) { return z8.col(w.pk[part * 32 + k], 64 + n); });
        Tile<32, 32> out;
        out.fill([&](int r, int n) {
            return __hmul(packed_half_row(ff, slab * 32 + r, n), w.attn_gate[n]);
        });
        PackedE4<32, 32> attended8;
        attended8.encode(attended, [](half x) { return x; });
        gemm8(
            out, [&](int r, int k) { return attended8.row(r, w.ai[k]); },
            [&](int k, int n) { return w.projection[k * 32 + n]; });
        publish(slab * 32, out);
    }
}
} // namespace endpoint
namespace endpoint {
// A is the same physical K packet for every N bank. Gather once per M,
// then consume it in original N order. Each accumulator sees identical MMA.
template <int M, int N, class A, class B>
__device__ void organized_gemm8(Tile<M, N> &c, A av, B bv) {
#pragma unroll 1
    for (int m = 0; m < M / 16; ++m) {
        u32 a[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[i] |= u32(av(r, k + j)) << (j * 8);
        }
#pragma unroll 1
        for (int n = 0; n < N / 8; ++n) {
            u32 b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                b[i] = 0;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + (lane() >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    b[i] |= u32(bv(k + j, col)) << (j * 8);
            }
            mma8(c.c[m][n], a, b);
        }
    }
}
} // namespace endpoint
namespace endpoint {
// Lossless same-capacity B8 layout: [K/32][N/8][lane][8].
// Bytes are identical to the traced raw B8, but packets are lane-permuted
// to match the accepted decoder's N order. Not native register scheduling.
struct WeightPacket {
    const u8 *data;
    int kbase, nbase, width;
    __device__ uint2 load(int bank) const {
        return *(const uint2 *)(data +
                                ((kbase / 32 * (width / 8) + nbase / 8 + bank) * 32 + lane()) * 8);
    }
};
template <int M, int N, class A>
__device__ void organized_gemm8(Tile<M, N> &c, A av, WeightPacket bv) {
    u32 a[M / 16][4];
#pragma unroll
    for (int m = 0; m < M / 16; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[m][i] = 0;
            int r = m * 16 + (lane() >> 2) + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[m][i] |= u32(av(r, k + j)) << (j * 8);
        }
    }
#pragma unroll 1
    for (int n = 0; n < N / 8; ++n) {
        uint2 packet = bv.load(n);
        u32 b[2] = {packet.x, packet.y};
#pragma unroll
        for (int m = 0; m < M / 16; ++m)
            mma8(c.c[m][n], a[m], b);
    }
}
} // namespace endpoint
namespace packed12 {
using namespace endpoint;
union HalfBits {
    u32 u;
    half2 h;
    __device__ HalfBits(u32 x) : u(x) {}
    __device__ HalfBits(half2 x) : h(x) {}
};
__device__ __forceinline__ half2 hh(u32 x) {
    return HalfBits(x).h;
}
__device__ __forceinline__ u32 bits(half2 x) {
    return HalfBits(x).u;
}
__device__ __forceinline__ u32 mul(u32 a, u32 b) {
    return bits(__hmul2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 splat(half a) {
    return pack(a, a);
}
__device__ __forceinline__ u32 add(u32 a, u32 b) {
    return bits(__hadd2(hh(a), hh(b)));
}
__device__ __forceinline__ u32 fma(u32 a, u32 b, u32 c) {
    return bits(__hfma2(hh(a), hh(b), hh(c)));
}
__device__ __forceinline__ u32 activate_pair(u32 x) {
    half2 c = __hmin2(__float2half2_rn(4.f), __hmax2(__float2half2_rn(-4.f), hh(x)));
    half2 g = __hfma2(__habs2(c), __float2half2_rn(-.055908203125f), __float2half2_rn(.447265625f));
    return bits(__hmul2(hh(x), __hfma2(c, g, __float2half2_rn(.89453125f))));
}
__device__ __forceinline__ u32 exponent_pair(u32 x) {
    half2 a = __hfma2(hh(x), __float2half2_rn(.044921875f), __float2half2_rn(1.30078125f));
    a = __hmin2(__float2half2_rn(1.5693359375f), __hmax2(__float2half2_rn(1.03125f), a));
    u32 v = bits(a);
    return (((v & 0xffffu) << 5) + 32768u) & 0xffffu | ((((v >> 16) << 5) + 32768u) << 16);
}
__device__ __forceinline__ u32 e4pair(u32 x) {
    return __nv_cvt_halfraw2_to_fp8x2(hh(x), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ u32 e4four(u32 a, u32 b) {
    return e4pair(a) | (e4pair(b) << 16);
}
struct Identity {
    __device__ u32 operator()(u32 x) const {
        return x;
    }
};
struct Activate {
    __device__ u32 operator()(u32 x) const {
        return activate_pair(x);
    }
};
} // namespace packed12
namespace packed12 {
// Each four-lane row group owns four distinct adjacent-channel partial pairs.
__device__ __forceinline__ u32 row_broadcast(u32 v, int owner) {
    return __shfl_sync(0xffffffff, v, (lane() & 28) + owner);
}
__device__ __forceinline__ half norm_finish(u32 t) {
    u32 a = add(row_broadcast(t, 0), row_broadcast(t, 2));
    u32 b = add(row_broadcast(t, 1), row_broadcast(t, 3));
    u32 s = add(a, b);
    return rsqrt_half(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
__device__ __forceinline__ half den_finish(u32 t) {
    // NOT a balanced xor tree: (((partial0+partial1)+partial2)+partial3).
    u32 s = add(add(add(row_broadcast(t, 0), row_broadcast(t, 1)), row_broadcast(t, 2)),
                row_broadcast(t, 3));
    return reciprocal(h(fmaxf(f(__hadd(unpack(s, 0), unpack(s, 1))), 6.198883056640625e-05f)));
}
template <int M, int N> __device__ half norm_tile(const Tile<M, N> &z, int row, int base) {
    int m = row / 16, i = (row / 8) & 1, n = base / 8;
    u32 a = z.c[m][n].v[i], b = z.c[m][n + 1].v[i], c = z.c[m][n + 2].v[i], d = z.c[m][n + 3].v[i];
    return norm_finish(add(fma(a, a, mul(c, c)), fma(b, b, mul(d, d))));
}
__device__ half den_two(const Tile<32, 64> &z, int row) {
    int m = row / 16, i = (row / 8) & 1;
    // lane%4 = g; two halves are s=0/1. Each t contributes (16t+2g)+(16t+2g+8).
    u32 s = add(z.c[m][0].v[i], z.c[m][1].v[i]);
    s = add(s, add(z.c[m][2].v[i], z.c[m][3].v[i]));
    s = add(s, add(z.c[m][4].v[i], z.c[m][5].v[i]));
    s = add(s, add(z.c[m][6].v[i], z.c[m][7].v[i]));
    return den_finish(s);
}
} // namespace packed12
namespace activation {
using namespace endpoint;
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
template <int M, int N> using HC = Words<(M / 16) * (N / 8) * 2>;
template <int M, int N, class F> __device__ __forceinline__ HC<M, N> fill(F f) {
    HC<M, N> x;
    each<M / 16>([&](auto mt) {
        each<N / 8>([&](auto nt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, n = decltype(nt)::value,
                              i = decltype(it)::value;
                int r = m * 16 + lane() / 4 + i * 8, c = n * 8 + (lane() % 4) * 2;
                x.v[(m * (N / 8) + n) * 2 + i] = pack(f(r, c), f(r, c + 1));
            });
        });
    });
    return x;
}
template <int M, int N, class F>
__device__ __forceinline__ Words<M / 16 * (N / 8)> encode(const HC<M, N> &x, F f) {
    Words<M / 16 * (N / 8)> y;
    each<M / 16 * (N / 8)>([&](auto t) {
        constexpr int j = decltype(t)::value;
        u32 a = x.v[j * 2], b = x.v[j * 2 + 1];
        y.v[j] = packed12::e4four(f(a), f(b));
    });
    return y;
}
template <int M, int N>
__device__ __forceinline__ void weight(HC<M, N> &c, const Words<M / 4> &a, WeightPacket w) {
    each<N / 8>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        uint2 b = w.load(n);
        u32 bb[2] = {b.x, b.y};
        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            mma8(z, aa, bb);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// Uniform fixed bank, row owner is lane/4; norm/den do not scan any banks.
template <int M, int N, int Row, int Col>
__device__ __forceinline__ half component(const HC<M, N> &x) {
    constexpr int bank = (Row / 2 * (N / 8) + Col / 8) * 2 + Row % 2;
    return unpack(__shfl_sync(0xffffffff, x.v[bank], (lane() & 28) + (Col % 8) / 2), Col % 2);
}
template <int M, int N, int Row, int Base = 0>
__device__ __forceinline__ half norm(const HC<M, N> &x) {
    constexpr int j = (Row / 2 * (N / 8) + Base / 8) * 2 + Row % 2;
    u32 a = x.v[j], b = x.v[j + 2], c = x.v[j + 4], d = x.v[j + 6];
    return packed12::norm_finish(packed12::add(packed12::fma(a, a, packed12::mul(c, c)),
                                               packed12::fma(b, b, packed12::mul(d, d))));
}
template <int Row> __device__ __forceinline__ half denominator(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    return packed12::den_finish(s);
}
} // namespace activation

namespace packet_reference {

using u32 = unsigned;

template <int N> using Words = activation::Words<N>;

__device__ __forceinline__ u32 transpose_half2(u32 a) {
    u32 d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;" : "=r"(d) : "r"(a));
    return d;
}

// input_a: input is ALREADY prepared encoded E4, except residual_c is original Half2.

__device__ __forceinline__ Words<16> input_a(const Words<16> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s5 = __shfl_sync(0xffffffff, x.v[1],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s6 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s7 = __shfl_sync(0xffffffff, x.v[3],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[1],
                         2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s17 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s18 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s19 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s37 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s38 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s39 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s49 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s50 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s51 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    Words<16> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[1] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s3
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s2
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s1
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s7
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s6
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s5
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s4 : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[3] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s11
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s10
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s9
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 3
             ? s15
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 2
                    ? s14
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 1
                           ? s13
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1)) == 0 ? s12
                                                                                       : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[5] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s19
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s18
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s17
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s16
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s23
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s22
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s21
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s20
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[7] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s27
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s26
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s25
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s24
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 7
             ? s31
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 6
                    ? s30
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 5
                           ? s29
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4) == 4 ? s28
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[9] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s35
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s34
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s33
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s39
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s38
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s37
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s36
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[11] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s43
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s42
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s41
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 11
             ? s47
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 10
                    ? s46
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 9
                           ? s45
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 8) == 8 ? s44
                                                                                           : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[13] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s51
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s50
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s49
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s48
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s55
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s54
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s53
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s52
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        16 | 1024 | 4096 | 16384);

    out.v[15] = __byte_perm(
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s59
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s58
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s57
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s56
                                  : 0)))),
        ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 15
             ? s63
             : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 14
                    ? s62
                    : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 13
                           ? s61
                           : ((((((l >> 0) & 1)) << 0) | ((((l >> 1) & 1)) << 1) | 4 | 8) == 12
                                  ? s60
                                  : 0)))),
        2 | 16 | 32 | 512 | 1024 | 4096 | 8192 | 16384);

    return out;
}

__device__ __forceinline__ Words<32> residual_c(const Words<32> &x) {

    int l = threadIdx.x & 31;

    u32 s0 =
        __shfl_sync(0xffffffff, x.v[0],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s1 =
        __shfl_sync(0xffffffff, x.v[2],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s2 =
        __shfl_sync(0xffffffff, x.v[4],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s3 =
        __shfl_sync(0xffffffff, x.v[6],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s4 =
        __shfl_sync(0xffffffff, x.v[1],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s5 =
        __shfl_sync(0xffffffff, x.v[3],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s6 =
        __shfl_sync(0xffffffff, x.v[5],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s7 =
        __shfl_sync(0xffffffff, x.v[7],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s8 = __shfl_sync(0xffffffff, x.v[0],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s9 = __shfl_sync(0xffffffff, x.v[2],
                         1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                             ((((l >> 4) & 1)) << 4));

    u32 s10 = __shfl_sync(0xffffffff, x.v[4],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s11 = __shfl_sync(0xffffffff, x.v[6],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s12 = __shfl_sync(0xffffffff, x.v[1],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s13 = __shfl_sync(0xffffffff, x.v[3],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s14 = __shfl_sync(0xffffffff, x.v[5],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s15 = __shfl_sync(0xffffffff, x.v[7],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s16 = __shfl_sync(0xffffffff, x.v[0],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s17 = __shfl_sync(0xffffffff, x.v[2],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s18 = __shfl_sync(0xffffffff, x.v[4],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s19 = __shfl_sync(0xffffffff, x.v[6],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s20 = __shfl_sync(0xffffffff, x.v[1],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s21 = __shfl_sync(0xffffffff, x.v[3],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s22 = __shfl_sync(0xffffffff, x.v[5],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s23 = __shfl_sync(0xffffffff, x.v[7],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s24 = __shfl_sync(0xffffffff, x.v[0],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s25 = __shfl_sync(0xffffffff, x.v[2],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s26 = __shfl_sync(0xffffffff, x.v[4],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s27 = __shfl_sync(0xffffffff, x.v[6],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s28 = __shfl_sync(0xffffffff, x.v[1],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s29 = __shfl_sync(0xffffffff, x.v[3],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s30 = __shfl_sync(0xffffffff, x.v[5],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s31 = __shfl_sync(0xffffffff, x.v[7],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s32 =
        __shfl_sync(0xffffffff, x.v[8],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s33 =
        __shfl_sync(0xffffffff, x.v[10],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s34 =
        __shfl_sync(0xffffffff, x.v[12],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s35 =
        __shfl_sync(0xffffffff, x.v[14],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s36 =
        __shfl_sync(0xffffffff, x.v[9],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s37 =
        __shfl_sync(0xffffffff, x.v[11],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s38 =
        __shfl_sync(0xffffffff, x.v[13],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s39 =
        __shfl_sync(0xffffffff, x.v[15],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s40 = __shfl_sync(0xffffffff, x.v[8],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s41 = __shfl_sync(0xffffffff, x.v[10],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s42 = __shfl_sync(0xffffffff, x.v[12],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s43 = __shfl_sync(0xffffffff, x.v[14],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s44 = __shfl_sync(0xffffffff, x.v[9],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s45 = __shfl_sync(0xffffffff, x.v[11],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s46 = __shfl_sync(0xffffffff, x.v[13],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s47 = __shfl_sync(0xffffffff, x.v[15],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s48 = __shfl_sync(0xffffffff, x.v[8],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s49 = __shfl_sync(0xffffffff, x.v[10],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s50 = __shfl_sync(0xffffffff, x.v[12],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s51 = __shfl_sync(0xffffffff, x.v[14],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s52 = __shfl_sync(0xffffffff, x.v[9],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s53 = __shfl_sync(0xffffffff, x.v[11],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s54 = __shfl_sync(0xffffffff, x.v[13],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s55 = __shfl_sync(0xffffffff, x.v[15],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s56 = __shfl_sync(0xffffffff, x.v[8],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s57 = __shfl_sync(0xffffffff, x.v[10],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s58 = __shfl_sync(0xffffffff, x.v[12],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s59 = __shfl_sync(0xffffffff, x.v[14],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s60 = __shfl_sync(0xffffffff, x.v[9],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s61 = __shfl_sync(0xffffffff, x.v[11],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s62 = __shfl_sync(0xffffffff, x.v[13],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s63 = __shfl_sync(0xffffffff, x.v[15],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s64 =
        __shfl_sync(0xffffffff, x.v[16],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s65 =
        __shfl_sync(0xffffffff, x.v[18],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s66 =
        __shfl_sync(0xffffffff, x.v[20],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s67 =
        __shfl_sync(0xffffffff, x.v[22],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s68 =
        __shfl_sync(0xffffffff, x.v[17],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s69 =
        __shfl_sync(0xffffffff, x.v[19],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s70 =
        __shfl_sync(0xffffffff, x.v[21],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s71 =
        __shfl_sync(0xffffffff, x.v[23],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s72 = __shfl_sync(0xffffffff, x.v[16],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s73 = __shfl_sync(0xffffffff, x.v[18],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s74 = __shfl_sync(0xffffffff, x.v[20],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s75 = __shfl_sync(0xffffffff, x.v[22],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s76 = __shfl_sync(0xffffffff, x.v[17],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s77 = __shfl_sync(0xffffffff, x.v[19],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s78 = __shfl_sync(0xffffffff, x.v[21],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s79 = __shfl_sync(0xffffffff, x.v[23],
                          1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s80 = __shfl_sync(0xffffffff, x.v[16],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s81 = __shfl_sync(0xffffffff, x.v[18],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s82 = __shfl_sync(0xffffffff, x.v[20],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s83 = __shfl_sync(0xffffffff, x.v[22],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s84 = __shfl_sync(0xffffffff, x.v[17],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s85 = __shfl_sync(0xffffffff, x.v[19],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s86 = __shfl_sync(0xffffffff, x.v[21],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s87 = __shfl_sync(0xffffffff, x.v[23],
                          2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s88 = __shfl_sync(0xffffffff, x.v[16],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s89 = __shfl_sync(0xffffffff, x.v[18],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s90 = __shfl_sync(0xffffffff, x.v[20],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s91 = __shfl_sync(0xffffffff, x.v[22],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s92 = __shfl_sync(0xffffffff, x.v[17],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s93 = __shfl_sync(0xffffffff, x.v[19],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s94 = __shfl_sync(0xffffffff, x.v[21],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s95 = __shfl_sync(0xffffffff, x.v[23],
                          1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                              ((((l >> 4) & 1)) << 4));

    u32 s96 =
        __shfl_sync(0xffffffff, x.v[24],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s97 =
        __shfl_sync(0xffffffff, x.v[26],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s98 =
        __shfl_sync(0xffffffff, x.v[28],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s99 =
        __shfl_sync(0xffffffff, x.v[30],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s100 =
        __shfl_sync(0xffffffff, x.v[25],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s101 =
        __shfl_sync(0xffffffff, x.v[27],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s102 =
        __shfl_sync(0xffffffff, x.v[29],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s103 =
        __shfl_sync(0xffffffff, x.v[31],
                    ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) | ((((l >> 4) & 1)) << 4));

    u32 s104 = __shfl_sync(0xffffffff, x.v[24],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s105 = __shfl_sync(0xffffffff, x.v[26],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s106 = __shfl_sync(0xffffffff, x.v[28],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s107 = __shfl_sync(0xffffffff, x.v[30],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s108 = __shfl_sync(0xffffffff, x.v[25],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s109 = __shfl_sync(0xffffffff, x.v[27],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s110 = __shfl_sync(0xffffffff, x.v[29],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s111 = __shfl_sync(0xffffffff, x.v[31],
                           1 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s112 = __shfl_sync(0xffffffff, x.v[24],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s113 = __shfl_sync(0xffffffff, x.v[26],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s114 = __shfl_sync(0xffffffff, x.v[28],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s115 = __shfl_sync(0xffffffff, x.v[30],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s116 = __shfl_sync(0xffffffff, x.v[25],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s117 = __shfl_sync(0xffffffff, x.v[27],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s118 = __shfl_sync(0xffffffff, x.v[29],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s119 = __shfl_sync(0xffffffff, x.v[31],
                           2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s120 = __shfl_sync(0xffffffff, x.v[24],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s121 = __shfl_sync(0xffffffff, x.v[26],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s122 = __shfl_sync(0xffffffff, x.v[28],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s123 = __shfl_sync(0xffffffff, x.v[30],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s124 = __shfl_sync(0xffffffff, x.v[25],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s125 = __shfl_sync(0xffffffff, x.v[27],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s126 = __shfl_sync(0xffffffff, x.v[29],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    u32 s127 = __shfl_sync(0xffffffff, x.v[31],
                           1 | 2 | ((((l >> 2) & 1)) << 2) | ((((l >> 3) & 1)) << 3) |
                               ((((l >> 4) & 1)) << 4));

    Words<32> out;

    out.v[0] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s3
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s2
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s1
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s0 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[1] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s7
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s6
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s5
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s4
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[2] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s11
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s10
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s9
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s8 : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[3] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s15
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s14
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s13
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s12
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[4] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s19
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s18
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s17
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s16
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[5] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s23
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s22
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s21
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s20
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[6] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 6
             ? s27
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 4
                    ? s26
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 2
                           ? s25
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 0 ? s24
                                                                                       : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[7] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 7
             ? s31
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 5
                    ? s30
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 3
                           ? s29
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2)) == 1 ? s28
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[8] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s35
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s34
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s33
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s32
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[9] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s39
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s38
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s37
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s36
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[10] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s43
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s42
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s41
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s40
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[11] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s47
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s46
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s45
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s44
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[12] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s51
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s50
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s49
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s48
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[13] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s55
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s54
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s53
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s52
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[14] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 14
             ? s59
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 12
                    ? s58
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 10
                           ? s57
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 8 ? s56
                                                                                           : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[15] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 15
             ? s63
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 13
                    ? s62
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 11
                           ? s61
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8) == 9
                                  ? s60
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[16] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s67
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s66
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s65
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s64
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[17] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s71
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s70
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s69
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s68
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[18] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s75
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s74
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s73
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s72
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[19] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s79
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s78
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s77
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s76
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[20] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s83
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s82
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s81
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s80
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[21] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s87
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s86
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s85
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s84
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[22] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 22
             ? s91
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 20
                    ? s90
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 18
                           ? s89
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 16
                                  ? s88
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[23] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 23
             ? s95
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 21
                    ? s94
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 19
                           ? s93
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 16) == 17
                                  ? s92
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[24] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s99
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s98
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s97
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s96
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[25] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s103
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s102
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s101
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s100
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[26] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s107
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s106
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s105
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s104
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[27] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s111
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s110
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s109
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s108
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[28] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s115
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s114
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s113
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s112
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[29] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s119
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s118
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s117
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s116
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[30] = __byte_perm(
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 30
             ? s123
             : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 28
                    ? s122
                    : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 26
                           ? s121
                           : ((((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 24
                                  ? s120
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    out.v[31] = __byte_perm(
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 31
             ? s127
             : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 29
                    ? s126
                    : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 27
                           ? s125
                           : ((1 | ((((l >> 0) & 1)) << 1) | ((((l >> 1) & 1)) << 2) | 8 | 16) == 25
                                  ? s124
                                  : 0)))),
        16 | 512 | 4096 | 8192);

    return out;
}

// Prepare each Half2 ONCE, with original norm/scale/den math, BEFORE this call.

template <class F> __device__ __forceinline__ Words<16> ff_a(F prepare) {

    Words<16> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> query_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> key_b(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<16> prob_a(F prepare) {
    Words<16> out;
    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<16>{}), b = prepare(activation::Tag<18>{});
        out.v[4] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<17>{}), b = prepare(activation::Tag<19>{});
        out.v[5] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<20>{}), b = prepare(activation::Tag<22>{});
        out.v[6] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<21>{}), b = prepare(activation::Tag<23>{});
        out.v[7] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[8] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[9] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[10] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[11] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<24>{}), b = prepare(activation::Tag<26>{});
        out.v[12] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<25>{}), b = prepare(activation::Tag<27>{});
        out.v[13] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<28>{}), b = prepare(activation::Tag<30>{});
        out.v[14] = packed12::e4four(a, b);
    }
    {
        u32 a = prepare(activation::Tag<29>{}), b = prepare(activation::Tag<31>{});
        out.v[15] = packed12::e4four(a, b);
    }
    return out;
}

template <class F> __device__ __forceinline__ Words<8> attended_a(F prepare) {

    Words<8> out;

    {
        u32 a = prepare(activation::Tag<0>{}), b = prepare(activation::Tag<2>{});
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<1>{}), b = prepare(activation::Tag<3>{});
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<4>{}), b = prepare(activation::Tag<6>{});
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<5>{}), b = prepare(activation::Tag<7>{});
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<8>{}), b = prepare(activation::Tag<10>{});
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<9>{}), b = prepare(activation::Tag<11>{});
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<12>{}), b = prepare(activation::Tag<14>{});
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = prepare(activation::Tag<13>{}), b = prepare(activation::Tag<15>{});
        out.v[7] = packed12::e4four(a, b);
    }

    return out;
}

template <class F> __device__ __forceinline__ Words<16> value_b(F get) {
    Words<16> out;

    {
        u32 a = transpose_half2(get(activation::Tag<0>{})),
            b = transpose_half2(get(activation::Tag<1>{}));
        out.v[0] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<24>{})),
            b = transpose_half2(get(activation::Tag<25>{}));
        out.v[1] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<2>{})),
            b = transpose_half2(get(activation::Tag<3>{}));
        out.v[2] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<26>{})),
            b = transpose_half2(get(activation::Tag<27>{}));
        out.v[3] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<4>{})),
            b = transpose_half2(get(activation::Tag<5>{}));
        out.v[4] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<28>{})),
            b = transpose_half2(get(activation::Tag<29>{}));
        out.v[5] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<6>{})),
            b = transpose_half2(get(activation::Tag<7>{}));
        out.v[6] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<30>{})),
            b = transpose_half2(get(activation::Tag<31>{}));
        out.v[7] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<8>{})),
            b = transpose_half2(get(activation::Tag<9>{}));
        out.v[8] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<16>{})),
            b = transpose_half2(get(activation::Tag<17>{}));
        out.v[9] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<10>{})),
            b = transpose_half2(get(activation::Tag<11>{}));
        out.v[10] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<18>{})),
            b = transpose_half2(get(activation::Tag<19>{}));
        out.v[11] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<12>{})),
            b = transpose_half2(get(activation::Tag<13>{}));
        out.v[12] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<20>{})),
            b = transpose_half2(get(activation::Tag<21>{}));
        out.v[13] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<14>{})),
            b = transpose_half2(get(activation::Tag<15>{}));
        out.v[14] = packed12::e4four(a, b);
    }

    {
        u32 a = transpose_half2(get(activation::Tag<22>{})),
            b = transpose_half2(get(activation::Tag<23>{}));
        out.v[15] = packed12::e4four(a, b);
    }

    return out;
}

} // namespace packet_reference

namespace shallow {

using namespace endpoint;

using namespace activation;

template <int P> constexpr int logical_m16() {
    return P == 0 ? 0 : P == 1 ? 3 : P == 2 ? 1 : 2;
}
template <int Row> __device__ __forceinline__ half physical_den(const HC<32, 64> &x) {
    constexpr int j = Row / 2 * 16 + Row % 2;
    u32 s = packed12::add(x.v[j], x.v[j + 2]);
    s = packed12::add(s, packed12::add(x.v[j + 4], x.v[j + 6]));
    s = packed12::add(s, packed12::add(x.v[j + 8], x.v[j + 10]));
    s = packed12::add(s, packed12::add(x.v[j + 12], x.v[j + 14]));
    return packed12::den_finish(s);
}
struct Entry {
    HC<64, 32> residual;
    Words<16> expand;
};

template <class F> __device__ __forceinline__ Entry entry(F get, const BodyWeights &w) {

    Entry e;

    Words<32> raw;
    each<32>([&](auto jt) { raw.v[decltype(jt)::value] = get(jt); });

    e.residual = packet_reference::residual_c(raw);

    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });

    Words<16> encoded;
    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;

        encoded.v[j] = packed12::e4four(raw.v[j * 2], raw.v[j * 2 + 1]);
    });
    e.expand = packet_reference::input_a(encoded);
    return e;
}

template <class F> __device__ __forceinline__ Entry load_entry(F get, const BodyWeights &w) {

    // Producer is scalar only at the memory/sample edge; no dynamically indexed Tile.

    auto x = fill<64, 32>(get);

    return entry([&](auto t) { return x.v[decltype(t)::value]; }, w);
}

template <int M, int N>
__device__ __forceinline__ void paired_weight(HC<M, N> &c, const Words<M / 4> &a, const u8 *w) {

    each<N / 16>([&](auto pt) {
        constexpr int p = decltype(pt)::value;

        // Original immutable B128 packet, shared by both N8 and all M16.

        const uint4 raw = *reinterpret_cast<const uint4 *>(w + p * 512 + lane() * 16);

        const uint2 b0 = make_uint2(raw.x, raw.y), b1 = make_uint2(raw.z, raw.w);

        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;

            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};

            each<2>([&](auto nt) {
                constexpr int n = p * 2 + decltype(nt)::value;

                uint2 b = decltype(nt)::value ? b1 : b0;
                u32 bb[2] = {b.x, b.y};

                Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
                mma8(z, aa, bb);

                c.v[(m * (N / 8) + n) * 2] = z.v[0];
                c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
            });
        });
    });
}

template <class F> __device__ __forceinline__ void publish_words(const HC<32, 32> &x, F publish) {

    each<2>([&](auto m) {
        each<4>([&](auto n) {
            each<2>([&](auto i) {
                publish(
                    m, n, i,
                    x.v[(decltype(m)::value * 4 + decltype(n)::value) * 2 + decltype(i)::value]);
            });
        });
    });
}

// Readout calls the original gemm16 twice with its original C handoff. Its A

// row/16, col/8 and row/8 are warp-uniform at this call site. Fixed references

// avoid rebuilding the projection Tile. This is not a general varying gather.

__device__ __forceinline__ half packet_half_uniform(const HC<32, 32> &x, int row, int col) {

    u32 word = 0;
    int bank = (row / 16 * 4 + col / 8) * 2 + (row / 8 % 2);

    each<16>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        if (bank == j)
            word = x.v[j];
    });

    return unpack(__shfl_sync(0xffffffff, word, (row % 8) * 4 + (col % 8) / 2), col % 2);
}

template <class Publish>
__device__ __forceinline__ void packet_body(Entry input, const BodyWeights &w, Publish publish) {

    auto ff = input.residual;

    {

        const auto a = input.expand;

#pragma unroll 1

        for (int part = 0; part < 4; ++part) {

            Words<16> ha;

            each<2>([&](auto pt) {
                constexpr int pair = decltype(pt)::value;

                auto hidden = fill<64, 16>([](int, int) { return h(0); });

                paired_weight<64, 16>(hidden, a, w.expand + part * 1024 + pair * 512);

                // A hidden N16 pair dies after activation/encoding into final A.

                each<4>([&](auto mt) {
                    each<2>([&](auto it) {
                        constexpr int dest =
                            decltype(mt)::value * 4 + pair * 2 + decltype(it)::value;

                        // rawExpand C ownership, NOT the old decoded hidden consumer.

                        constexpr int bank = decltype(mt)::value * 4 + decltype(it)::value;

                        u32 x = packed12::activate_pair(hidden.v[bank]);

                        u32 y = packed12::activate_pair(hidden.v[bank + 2]);

                        ha.v[dest] = packed12::e4four(x, y);
                    });
                });
            });

            paired_weight<64, 32>(ff, ha, w.contract + part * 1024);
        }
    }

    Words<16> qa, kb, vb;

    {

        auto a = packet_reference::ff_a([&](auto jt) { return ff.v[decltype(jt)::value]; });

        each<3>([&](auto ct) {
            constexpr int component = decltype(ct)::value;

            auto z = fill<64, 32>([](int, int) { return h(0); });

            paired_weight<64, 32>(z, a, w.qkv + component * 1024);

            Words<8> inv;

            if constexpr (component < 2)
                each<8>([&](auto rt) {
                    constexpr int row = decltype(rt)::value;
                    inv.v[row] = packed12::splat(activation::norm<64, 32, row>(z));
                });

            auto get = [&](auto jt) {
                constexpr int j = decltype(jt)::value;
                u32 x = z.v[j];

                if constexpr (component < 2)
                    x = packed12::mul(x, inv.v[j / 8 * 2 + j % 2]);

                if constexpr (component == 0)
                    x = packed12::mul(x, packed12::splat(w.scale));

                return x;
            };

            // Directly encode final consumer words. No z96 or canonical Q/K/V.

            if constexpr (component == 0)
                qa = packet_reference::query_a(get);

            if constexpr (component == 1)
                kb = packet_reference::key_b(get);

            if constexpr (component == 2)
                vb = packet_reference::value_b(get);
        });
    }

    each<2>([&](auto st) {
        constexpr int slab = decltype(st)::value;

        Words<16> pa;

        {

            HC<32, 64> logits;
            each<2>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                u32 a[4] = {qa.v[(slab * 2 + m) * 4], qa.v[(slab * 2 + m) * 4 + 1],
                            qa.v[(slab * 2 + m) * 4 + 2], qa.v[(slab * 2 + m) * 4 + 3]};
                each<4>([&](auto pt) {
                    constexpr int p = decltype(pt)::value;
                    // qkv already includes the pre/UP/post ABI shift; original raw40 only.
                    u32 s0, s1, s2, s3;
                    const u8 *seed = w.qkv + 0xc00 + ((slab * 2 + m) * 4 + p) * 512 + lane() * 16;
                    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(s0), "=r"(s1), "=r"(s2), "=r"(s3)
                                 : "l"(seed));
                    each<2>([&](auto nt) {
                        constexpr int n = p * 2 + decltype(nt)::value;
                        u32 b[2] = {kb.v[n * 2], kb.v[n * 2 + 1]};
                        Fragment c{{decltype(nt)::value ? s2 : s0, decltype(nt)::value ? s3 : s1}};
                        mma8(c, a, b);
                        logits.v[(m * 8 + n) * 2] = c.v[0];
                        logits.v[(m * 8 + n) * 2 + 1] = c.v[1];
                    });
                });
            });
            each<32>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                logits.v[j] = packed12::exponent_pair(logits.v[j]);
            });

            Words<4> den;

            each<4>([&](auto rt) {
                constexpr int r = decltype(rt)::value;
                den.v[r] = packed12::splat(physical_den<r>(logits));
            });

            pa = packet_reference::prob_a([&](auto jt) {
                constexpr int j = decltype(jt)::value;

                return packed12::mul(logits.v[j], den.v[j / 16 * 2 + j % 2]);
            });
        }

        auto attended = fill<32, 32>([](int, int) { return h(0); });

        each<2>([&](auto pt) {
            constexpr int part = decltype(pt)::value;

            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 b[2] = {vb.v[part * 8 + n * 2], vb.v[part * 8 + n * 2 + 1]};

                each<2>([&](auto mt) {
                    constexpr int m = decltype(mt)::value;
                    u32 a[4] = {pa.v[part * 8 + m * 4], pa.v[part * 8 + m * 4 + 1],
                                pa.v[part * 8 + m * 4 + 2], pa.v[part * 8 + m * 4 + 3]};

                    Fragment c{{attended.v[(m * 4 + n) * 2], attended.v[(m * 4 + n) * 2 + 1]}};
                    mma8(c, a, b);

                    attended.v[(m * 4 + n) * 2] = c.v[0];
                    attended.v[(m * 4 + n) * 2 + 1] = c.v[1];
                });
            });
        });

        HC<32, 32> out;

        each<16>([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            int c = (j / 2 % 4) * 8 + (lane() % 4) * 2;

            out.v[j] = packed12::mul(ff.v[logical_m16<slab * 2 + j / 8>() * 8 + j % 8],
                                     pack(w.attn_gate[c], w.attn_gate[c + 1]));
        });

        auto aa = packet_reference::attended_a([&](auto jt) {
            constexpr int j = decltype(jt)::value;
            return attended.v[j];
        });

        paired_weight<32, 32>(out, aa, w.projection);

        publish(Tag<logical_m16<slab * 2>() * 16>{}, Tag<logical_m16<slab * 2 + 1>() * 16>{}, out);
    });
}

} // namespace shallow

namespace joint {
using namespace shallow;
// Canonical channel represented by a raw-body residual C owner.
__device__ __forceinline__ int input_channel(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void gate(Entry &e, const BodyWeights &w) {
    each<32>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        int c = (j / 2 % 4) * 8 + (lane() & 3) * 2;
        e.residual.v[j] = packed12::mul(e.residual.v[j], pack(w.ffn_gate[c], w.ffn_gate[c + 1]));
    });
}
// Input is already in the residual owner's fixed Half registers. Quantization
// reads those original registers, independently of the gated C rail.
__device__ __forceinline__ Entry true_half(HC<64, 32> raw, const BodyWeights &w) {
    Entry e;
    e.residual = raw;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                e.expand.v[m * 4 + p * 2 + i] =
                    packed12::e4four(raw.v[m * 8 + p * 4 + i], raw.v[m * 8 + p * 4 + 2 + i]);
            });
        });
    });
    gate(e, w);
    return e;
}
template <class F> __device__ __forceinline__ Entry produce_half(F get, const BodyWeights &w) {
    return true_half(fill<64, 32>([&](int r, int c) { return get(r, input_channel(c)); }), w);
}
// Byte loads are predicated BEFORE memory access. Only a real aligned,
// consecutive, all-valid quartet may use the vector path.
template <class Address>
__device__ __forceinline__ u32 load4(const u8 *data, Address address, int row, int col) {
    int a = address(row, col), b = address(row, col + 1), c = address(row, col + 2),
        d = address(row, col + 3);
    if (a >= 0 && b == a + 1 && c == a + 2 && d == a + 3 &&
        ((reinterpret_cast<unsigned long long>(data + a) & 3) == 0))
        return *reinterpret_cast<const u32 *>(data + a);
    return u32(a < 0 ? 0 : data[a]) | (u32(b < 0 ? 0 : data[b]) << 8) |
           (u32(c < 0 ? 0 : data[c]) << 16) | (u32(d < 0 ? 0 : data[d]) << 24);
}
template <class Address>
__device__ __forceinline__ Entry e4_edge(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int m = decltype(mt)::value, p = decltype(pt)::value,
                              i = decltype(it)::value;
                int row = m * 16 + lane() / 4 + i * 8, col = (lane() & 3) * 8 + p * 4;
                u32 raw = load4(data, address, row, col), a = raw;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    u8 v = raw >> (b * 8);
                    // Keep the old NaN round-trip, including its sign/canonicalization.
                    if ((v & 127) == 127)
                        a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
                });
                e.expand.v[m * 4 + p * 2 + i] = a;
                e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
                e.residual.v[m * 8 + p * 4 + 2 + i] =
                    pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
            });
        });
    });
    gate(e, w);
    return e;
}
// HalfSlab consumer: same address recipe is used for both bytes of the pair.
// Optional pre never controls the separate mandatory shared Half pool result.
template <class Address>
__device__ __forceinline__ void publish_pair(u8 *data, half *pre, Address address, int row, int col,
                                             u32 word) {
    int a = address(row, col), b = address(row, col + 1);
    half x = unpack(word, 0), y = unpack(word, 1);
    if (a >= 0 && b == a + 1 && ((reinterpret_cast<unsigned long long>(data + a) & 1) == 0))
        *reinterpret_cast<unsigned short *>(data + a) = unsigned(e4(x)) | (unsigned(e4(y)) << 8);
    else {
        if (a >= 0)
            data[a] = e4(x);
        if (b >= 0)
            data[b] = e4(y);
    }
    if (pre) {
        if (a >= 0)
            pre[a] = x;
        if (b >= 0)
            pre[b] = y;
    }
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed8(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 4 + (i / 2) * 16;
            u32 v = 0;
            each<4>([&](auto jt) {
                constexpr int j = decltype(jt)::value;
                v |= u32(av(r, k + j)) << (j * 8);
            });
            a[i] = v;
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 4 + i * 16, col = n * 8 + lane() / 4;
                u32 v = 0;
                each<4>([&](auto jt) {
                    constexpr int j = decltype(jt)::value;
                    v |= u32(bv(k + j, col)) << (j * 8);
                });
                b[i] = v;
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma8(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
template <int M, int N, class A, class B>
__device__ __forceinline__ void fixed16(HC<M, N> &c, A av, B bv) {
    each<M / 16>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        u32 a[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            int r = m * 16 + lane() / 4 + (i & 1) * 8, k = (lane() & 3) * 2 + (i / 2) * 8;
            a[i] = pack(av(r, k), av(r, k + 1));
        });
        each<N / 8>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 b[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = (lane() & 3) * 2 + i * 8, col = n * 8 + lane() / 4;
                b[i] = pack(bv(k, col), bv(k + 1, col));
            });
            Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
            mma16(z, a, b);
            c.v[(m * (N / 8) + n) * 2] = z.v[0];
            c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
        });
    });
}
// A for the original readout K16 is exactly the projection C's pair at
// [m, kpart*2+i/2, i%2]. No runtime bank selection or projection Tile.
__device__ __forceinline__ HC<32, 8> read_head(const HC<32, 32> &out, const half *weights) {
    auto head = fill<32, 8>([](int, int) { return h(0); });
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 a[4], b[2];
            each<4>([&](auto it) {
                constexpr int i = decltype(it)::value;
                a[i] = out.v[m * 8 + kp * 4 + (i / 2) * 2 + i % 2];
            });
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int k = kp * 16 + (lane() & 3) * 2 + i * 8, n = lane() / 4;
                b[i] = pack(weights[k * 8 + n], weights[(k + 1) * 8 + n]);
            });
            Fragment z{{head.v[m * 2], head.v[m * 2 + 1]}};
            mma16(z, a, b);
            head.v[m * 2] = z.v[0];
            head.v[m * 2 + 1] = z.v[1];
        });
    });
    return head;
}
template <int Col> __device__ __forceinline__ half head_channel(const HC<32, 8> &head) {
    u32 result = 0;
    each<4>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        u32 v = __shfl_sync(0xffffffff, head.v[j], (lane() % 8) * 4 + Col / 2);
        if (lane() / 8 == j)
            result = v;
    });
    return unpack(result, Col % 2);
}
} // namespace joint
namespace physical {
using namespace shallow;
// Each tin packet holds (row,row+8) x (four+four channels). The raw A
// consumer and projection C producer use this exact same lane ownership.
__device__ __forceinline__ uint4 load(const u8 *p, int a) {
    if (a < 0)
        return make_uint4(0, 0, 0, 0);
    return *reinterpret_cast<const uint4 *>(p + a);
}
__device__ __forceinline__ int nc(int c) {
    return (c % 8 / 2) * 8 + (c / 8) * 2 + (c & 1);
}
__device__ __forceinline__ int result_index(int r, int c) {
    return (r / 16) * 512 + (r % 8) * 64 + (r % 16 / 8) * 4 + 16 * (c % 8 / 2) + (c / 16) * 8 +
           (c % 16 / 8) * 2 + (c & 1);
}
__device__ __forceinline__ void raw_word(Entry &e, int m, int p, int i, u32 raw) {
    u32 a = raw;
    each<4>([&](auto bt) {
        constexpr int b = decltype(bt)::value;
        u8 v = raw >> (b * 8);
        if ((v & 127) == 127)
            a = (a & ~(255u << (b * 8))) | (u32(e4(une4(v))) << (b * 8));
    });
    e.expand.v[m * 4 + p * 2 + i] = a;
    e.residual.v[m * 8 + p * 4 + i] = pack(une4(u8(raw)), une4(u8(raw >> 8)));
    e.residual.v[m * 8 + p * 4 + 2 + i] = pack(une4(u8(raw >> 16)), une4(u8(raw >> 24)));
}
template <class Address>
__device__ __forceinline__ Entry input_tin(const u8 *data, Address address, const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 4, (lane() & 3) * 8));
        raw_word(e, m, 0, 0, v.x);
        raw_word(e, m, 0, 1, v.y);
        raw_word(e, m, 1, 0, v.z);
        raw_word(e, m, 1, 1, v.w);
    });
    joint::gate(e, w);
    return e;
}
template <class Address>
__device__ __forceinline__ Entry input_compact(const u8 *data, Address address,
                                               const BodyWeights &w) {
    Entry e;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        uint4 v = load(data, address(m * 16 + lane() / 2, (lane() & 1) * 4));
        each<2>([&](auto pt) {
            each<2>([&](auto it) {
                constexpr int p = decltype(pt)::value, i = decltype(it)::value;
                int src = (lane() / 4 + i * 8) * 2 + p;
                u32 a = __shfl_sync(0xffffffff, v.x, src), b = __shfl_sync(0xffffffff, v.y, src);
                u32 c = __shfl_sync(0xffffffff, v.z, src), d = __shfl_sync(0xffffffff, v.w, src);
                u32 raw = (lane() % 4 == 0) ? a : (lane() % 4 == 1) ? b : (lane() % 4 == 2) ? c : d;
                raw_word(e, m, p, i, raw);
            });
        });
    });
    joint::gate(e, w);
    return e;
}
// Same C owner publishes E4 memory and the unquantized shared Half rail.
// The shared layout is consumed directly by the four-point pool, not redecoded.
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_tin(const HC<32, 32> &z, u8 *data, half *pre, half *result,
                                            Address address) {
    each<2>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        int row = (m == 0 ? Base0 : Base1) + lane() / 4, col = (lane() & 3) * 2;
        int a = address(row, col);
        uint4 v = make_uint4(packed12::e4four(z.v[m * 8], z.v[m * 8 + 2]),
                             packed12::e4four(z.v[m * 8 + 1], z.v[m * 8 + 3]),
                             packed12::e4four(z.v[m * 8 + 4], z.v[m * 8 + 6]),
                             packed12::e4four(z.v[m * 8 + 5], z.v[m * 8 + 7]));
        if (a >= 0)
            *reinterpret_cast<uint4 *>(data + a) = v;
        if (pre || result) {
            uint4 lo = make_uint4(z.v[m * 8], z.v[m * 8 + 2], z.v[m * 8 + 1], z.v[m * 8 + 3]);
            uint4 hi = make_uint4(z.v[m * 8 + 4], z.v[m * 8 + 6], z.v[m * 8 + 5], z.v[m * 8 + 7]);
            if (pre && a >= 0) {
                *reinterpret_cast<uint4 *>(pre + a) = lo;
                *reinterpret_cast<uint4 *>(pre + a + 8) = hi;
            }
            if (result) {
                int s = (m == 0 ? Base0 : Base1) / 16 * 512 + lane() * 16;
                *reinterpret_cast<uint4 *>(result + s) = lo;
                *reinterpret_cast<uint4 *>(result + s + 8) = hi;
            }
        }
    });
}
template <int Base0, int Base1, class Address>
__device__ __forceinline__ void publish_compact(const HC<32, 32> &z, u8 *data, half *pre,
                                                Address address) {
    each<2>([&](auto mt) {
        each<2>([&](auto pt) {
            constexpr int m = decltype(mt)::value, p = decltype(pt)::value;
            u32 lo = packed12::e4four(z.v[m * 8 + p * 4], z.v[m * 8 + p * 4 + 2]);
            u32 hi = packed12::e4four(z.v[m * 8 + p * 4 + 1], z.v[m * 8 + p * 4 + 3]);
            uint4 v;
            each<4>([&](auto kt) {
                constexpr int k = decltype(kt)::value;
                int src = (lane() % 8) * 4 + k;
                u32 a = __shfl_sync(0xffffffff, lo, src), b = __shfl_sync(0xffffffff, hi, src),
                    q = lane() < 8 ? a : b;
                if constexpr (k == 0)
                    v.x = q;
                if constexpr (k == 1)
                    v.y = q;
                if constexpr (k == 2)
                    v.z = q;
                if constexpr (k == 3)
                    v.w = q;
            });
            if (lane() < 16) {
                int a = address((m == 0 ? Base0 : Base1) + lane(), p * 16);
                if (a >= 0)
                    *reinterpret_cast<uint4 *>(data + a) = v;
            }
            // Optional true-Half ABI remains in C ownership; no E4 reconstruction.
            if (pre) {
                each<2>([&](auto nt) {
                    each<2>([&](auto it) {
                        constexpr int n = p * 2 + decltype(nt)::value, i = decltype(it)::value;
                        int row = (m == 0 ? Base0 : Base1) + lane() / 4 + i * 8,
                            col = n * 8 + (lane() & 3) * 2, a = address(row, col);
                        if (a >= 0)
                            *reinterpret_cast<u32 *>(pre + a) = z.v[m * 8 + n * 2 + i];
                    });
                });
            }
        });
    });
}
} // namespace physical

// Shape-specialized native packet/head boundary. These are composed endpoint coordinates, not seven
// LUT substitutes.
namespace post_static {
using namespace shallow;
template <int M> __device__ __forceinline__ int2 pixel(int row) {
    constexpr int dx = M == 2 || M == 3, dy = M == 1 || M == 2;
    return make_int2(int(blockIdx.x) * 8 - 4 + dx * 4 + (row & 3),
                     int(blockIdx.y) * 8 - 4 + dy * 4 + (row >> 2));
}
__device__ __forceinline__ int main_packet(int sx, int sy, int p) {
    // canonical_to_pixel and pixel_to_canonical cancel inside compact token.
    int t = (sy & 7) * 8 + (sx & 7), l = p * 1024 + t * 16;
    return (((sy >> 3) + (NR_H / 16) * (l >> 10)) * (NR_W / 8) + (sx >> 5) +
            (NR_W / 64) * ((l >> 7) & 7)) *
               512 +
           ((sx >> 3) & 3) * 128 + (l & 127);
}
template <int P, int I> __device__ __forceinline__ u32 main_word(uint4 packet) {
    int src = (lane() & 8) + P + 2 * I;
    u32 a = __shfl_sync(0xffffffff, packet.x, src), b = __shfl_sync(0xffffffff, packet.y, src);
    u32 c = __shfl_sync(0xffffffff, packet.z, src), d = __shfl_sync(0xffffffff, packet.w, src);
    return (lane() & 3) == 0 ? a : (lane() & 3) == 1 ? b : (lane() & 3) == 2 ? c : d;
}
__device__ __forceinline__ uint2 head_packet(const HC<32, 8> &head) {
    uint2 q = make_uint2(0, 0);
    each<4>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        // Bank is compile-time until after both full-warp collectives.
        u32 rg = __shfl_sync(0xffffffff, head.v[j], (lane() & 7) * 4);
        u32 ba = __shfl_sync(0xffffffff, head.v[j], (lane() & 7) * 4 + 1);
        if ((lane() >> 3) == j)
            q = make_uint2(rg, ba);
    });
    return q;
}
} // namespace post_static
using namespace activation;
using namespace shallow;
using namespace endpoint;
extern "C" __global__
__launch_bounds__(32) void post70_native_head(BodyWeights w, const u8 *main, const u8 *skip,
                                              const half *main_gate, const half *skip_gate,
                                              const half *readout, float *head_out, int *status) {
    HC<64, 32> raw;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        int r = lane() / 4;
        int2 xy = post_static::pixel<m>(r);
        // Exactly the old base-row predicate also protects row+8.
        bool valid = xy.x >= 0 && xy.x < NR_W && xy.y >= 0 && xy.y < NR_H;
        constexpr int dx = m == 2 || m == 3, dy = m == 1 || m == 2;
        int stripe_y = int(blockIdx.y) * 2 - 1 + dy, stripe_x = int(blockIdx.x) * 2 - 1 + dx;
        int skip_address = (stripe_y * (NR_W / 4) + stripe_x) * 512 + (r & 3) * 64 +
                           (r >> 2) * 256 + (lane() & 3) * 16;
        uint4 sk = physical::load(skip, valid ? skip_address : -1);
        // Eight actual main packets per CTA quadrant; no added bytes. Every
        // consumer's owner shares the original paired predicate at all edges.
        uint4 mp = make_uint4(0, 0, 0, 0);
        if (valid && (lane() & 20) == 0) {
            int sx = int(blockIdx.x) * 4 - 2 + dx * 2 + ((lane() >> 3) & 1);
            int sy = int(blockIdx.y) * 4 - 2 + dy * 2 + ((lane() >> 1) & 1);
            mp = *reinterpret_cast<const uint4 *>(main +
                                                  post_static::main_packet(sx, sy, lane() & 1));
        }
        each<2>([&](auto pt) {
            constexpr int p = decltype(pt)::value;
            int col = (lane() & 3) * 8 + p * 4;
            uint2 mg = make_uint2(0, 0), sg = make_uint2(0, 0);
            if (valid) {
                mg = *reinterpret_cast<const uint2 *>(main_gate + col);
                sg = *reinterpret_cast<const uint2 *>(skip_gate + col);
            }
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                u32 mv = post_static::main_word<p, i>(mp);
                u32 sv;
                if constexpr (p == 0 && i == 0)
                    sv = sk.x;
                if constexpr (p == 0 && i == 1)
                    sv = sk.y;
                if constexpr (p == 1 && i == 0)
                    sv = sk.z;
                if constexpr (p == 1 && i == 1)
                    sv = sk.w;
                each<2>([&](auto jt) {
                    constexpr int j = decltype(jt)::value;
                    half a = h(0), b = h(0);
                    u32 gm = j == 0 ? mg.x : mg.y, gs = j == 0 ? sg.x : sg.y;
                    if (valid) {
                        a = __hfma(une4(u8(sv >> (j * 16))), unpack(gs, 0),
                                   __hmul(une4(u8(mv >> (j * 16))), unpack(gm, 0)));
                        b = __hfma(une4(u8(sv >> (j * 16 + 8))), unpack(gs, 1),
                                   __hmul(une4(u8(mv >> (j * 16 + 8))), unpack(gm, 1)));
                    }
                    raw.v[m * 8 + p * 4 + j * 2 + i] = pack(a, b);
                });
            });
        });
    });
    auto input = joint::true_half(raw, w);
    packet_body(input, w, [&](auto b0, auto b1, const HC<32, 32> &out) {
        constexpr int base0 = decltype(b0)::value, base1 = decltype(b1)::value;
        auto head = joint::read_head(out, readout);
        {
            int r = lane(); // Varying rows: generic get shuffles ALL banks before selection.
            // All lanes enter collective gathers; only then predicate writes.
            uint2 hp = post_static::head_packet(head);
            half hr = unpack(hp.x, 0), hg = unpack(hp.x, 1), hb = unpack(hp.y, 0),
                 ha = unpack(hp.y, 1);
            { // All rows gathered before pixel validity predicates.
                int2 xy = r < 16 ? post_static::pixel<base0 / 16>(r % 16)
                                 : post_static::pixel<base1 / 16>(r % 16);
                int y = xy.y, x = xy.x;
                if (y >= 0 && y < NR_H && x >= 0 && x < NR_W) {
                    float4 value = make_float4(f(hr), f(hg), f(hb), f(ha));
                    if (!isfinite(value.x) || !isfinite(value.y) || !isfinite(value.z) ||
                        !isfinite(value.w))
                        atomicOr(status, 4);
                    *reinterpret_cast<float4 *>(head_out + (y * NR_W + x) * 4) = value;
                }
            }
        }
    });
}

} // namespace nr_outer_post_2
