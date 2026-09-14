// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================================
// OUTER two_physical_seed/two.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_two_3 {
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
// Index bits are [register][lane][unit]. All register indexing below is
// compile-time. A unit is one E4 byte or one Half, never a numeric conversion.
template <int U, int A, int B, int N>
__device__ __forceinline__ Words<N> xor_bits(const Words<N> &x) {
    constexpr int S = U == 4 ? 2 : 1, L = S + 5;
    Words<N> y;
    each<N>([&](auto it) {
        constexpr int I = decltype(it)::value;
        if constexpr (A >= L) {
            constexpr int J = I ^ (1 << (A - L));
            if constexpr (B >= L)
                y.v[I] = x.v[((I >> (B - L)) & 1) ? J : I];
            else if constexpr (B >= S)
                y.v[I] = (lane() & (1 << (B - S))) ? x.v[J] : x.v[I];
            else {
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b + ((((b / (4 / U)) >> B) & 1) ? 4 : 0)) << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], x.v[J], mask);
            }
        } else if constexpr (A >= S) {
            if constexpr (B >= L)
                y.v[I] =
                    __shfl_sync(0xffffffff, x.v[I], lane() ^ ((((I >> (B - L)) & 1)) << (A - S)));
            else if constexpr (B >= S)
                y.v[I] = __shfl_sync(0xffffffff, x.v[I],
                                     lane() ^ (((lane() >> (B - S)) & 1) << (A - S)));
            else {
                u32 other = __shfl_xor_sync(0xffffffff, x.v[I], 1 << (A - S));
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b + ((((b / (4 / U)) >> B) & 1) ? 4 : 0)) << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], other, mask);
            }
        } else {
            constexpr unsigned switched = []() {
                unsigned m = 0;
                for (int b = 0; b < 4; ++b)
                    m |= (b ^ (1 << (A + (U == 4 ? 0 : 1)))) << (b * 4);
                return m;
            }();
            if constexpr (B >= L)
                y.v[I] = __byte_perm(x.v[I], 0, ((I >> (B - L)) & 1) ? switched : 0x3210);
            else if constexpr (B >= S)
                y.v[I] = __byte_perm(x.v[I], 0, (lane() & (1 << (B - S))) ? switched : 0x3210);
            else {
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b ^ ((((b / (4 / U)) >> B) & 1) << (A + (U == 4 ? 0 : 1))))
                             << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], 0, mask);
            }
        }
    });
    return y;
}
template <int U, int A, int B, int N> __device__ __forceinline__ Words<N> swap_bits(Words<N> x) {
    return xor_bits<U, A, B>(xor_bits<U, B, A>(xor_bits<U, A, B>(x)));
}
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
        y.v[j] = u32(e4(f(unpack(a, 0)))) | (u32(e4(f(unpack(a, 1)))) << 8) |
                 (u32(e4(f(unpack(b, 0)))) << 16) | (u32(e4(f(unpack(b, 1)))) << 24);
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
    half t[8];
    each<8>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        half a = component<M, N, Row, Base + j>(x), b = component<M, N, Row, Base + 8 + j>(x);
        half c = component<M, N, Row, Base + 16 + j>(x), d = component<M, N, Row, Base + 24 + j>(x);
        t[j] = __hadd(__hfma(a, a, __hmul(c, c)), __hfma(b, b, __hmul(d, d)));
    });
    half a = __hadd(__hadd(t[0], t[4]), __hadd(t[2], t[6]));
    half b = __hadd(__hadd(t[1], t[5]), __hadd(t[3], t[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
template <int Row> __device__ __forceinline__ half denominator(const HC<32, 64> &x) {
    half t[8];
    each<8>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        half v = __hadd(component<32, 64, Row, j>(x), component<32, 64, Row, j + 8>(x));
        v = __hadd(v, __hadd(component<32, 64, Row, j + 48>(x), component<32, 64, Row, j + 56>(x)));
        v = __hadd(v, __hadd(component<32, 64, Row, j + 16>(x), component<32, 64, Row, j + 24>(x)));
        t[j] =
            __hadd(v, __hadd(component<32, 64, Row, j + 32>(x), component<32, 64, Row, j + 40>(x)));
    });
    half a = __hadd(__hadd(__hadd(t[0], t[2]), t[4]), t[6]);
    half b = __hadd(__hadd(__hadd(t[1], t[3]), t[5]), t[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
} // namespace activation
namespace two_phase {
using namespace activation;
__device__ __forceinline__ Words<16> hidden_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<4> tail_a(Words<4> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> query_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> prob_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 9, 10>(x);
    return x;
}
__device__ __forceinline__ Words<16> key_b(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 7, 8>(x);
    return x;
}
__device__ __forceinline__ Words<16> value_b(Words<16> x) {
    x = swap_bits<4, 0, 4>(x);
    x = swap_bits<4, 2, 5>(x);
    x = swap_bits<4, 3, 6>(x);
    x = swap_bits<4, 7, 9>(x);
    x = swap_bits<4, 8, 9>(x);
    return x;
}
} // namespace two_phase
namespace packed12 {
using namespace endpoint;
__device__ __forceinline__ half2 hh(u32 x) {
    return __halves2half2(unpack(x, 0), unpack(x, 1));
}
__device__ __forceinline__ u32 bits(half2 x) {
    return pack(__low2half(x), __high2half(x));
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
namespace two_phase {
using namespace endpoint;
template <int Offset, int Count, int N>
__device__ __forceinline__ Words<Count> slice(const Words<N> &x) {
    Words<Count> y;
    each<Count>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        y.v[j] = x.v[Offset + j];
    });
    return y;
}
template <int M, int N, class F>
__device__ __forceinline__ Words<M / 16 * (N / 8)> encode(const HC<M, N> &x, F f) {
    Words<M / 16 * (N / 8)> y;
    each<M / 16 * (N / 8)>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        y.v[j] = packed12::e4four(f(x.v[2 * j]), f(x.v[2 * j + 1]));
    });
    return y;
}
template <int M, int N>
__device__ __forceinline__ void matrix(HC<M, N> &c, const Words<M / 4> &a, const Words<N / 4> &b) {
    each<N / 8>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        u32 bb[2] = {b.v[n * 2], b.v[n * 2 + 1]};
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
template <int Row, int Base> __device__ __forceinline__ half norm(const HC<16, 96> &x) {
    constexpr int n = Base / 8;
    u32 a = x.v[n * 2 + Row], b = x.v[(n + 1) * 2 + Row], c = x.v[(n + 2) * 2 + Row],
        d = x.v[(n + 3) * 2 + Row];
    return packed12::norm_finish(packed12::add(packed12::fma(a, a, packed12::mul(c, c)),
                                               packed12::fma(b, b, packed12::mul(d, d))));
}
template <int Row> __device__ __forceinline__ half den(const HC<32, 64> &x) {
    constexpr int base = (Row / 2) * 16, i = Row % 2;
    u32 s = packed12::add(x.v[base + i], x.v[base + 2 + i]);
    s = packed12::add(s, packed12::add(x.v[base + 4 + i], x.v[base + 6 + i]));
    s = packed12::add(s, packed12::add(x.v[base + 8 + i], x.v[base + 10 + i]));
    s = packed12::add(s, packed12::add(x.v[base + 12 + i], x.v[base + 14 + i]));
    return packed12::den_finish(s);
}
__device__ __forceinline__ int perm(int c) {
    return (c & 49) | ((c & 6) << 1) | ((c & 8) >> 2);
}
} // namespace two_phase

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

struct Layout2 {
    int h, w, gx, sx, sy, seq;
};
__device__ int label2(int t, int c) {
    return (c & 3) | ((c & 12) << 2) | ((c & 16) >> 1) | ((c & 32) << 4) | ((t & 7) << 6) |
           ((t & 8) >> 1) | ((t & 48) << 6);
}
__device__ int token2(int l) {
    return ((l >> 6) & 7) | ((l & 4) << 1) | ((l >> 6) & 48);
}
__device__ int channel2(int l) {
    return (l & 3) | ((l >> 2) & 12) | ((l & 8) << 1) | ((l >> 4) & 32);
}
__device__ int generic_address(Layout2 d, int l) {
    int y = (8 * int(blockIdx.x / d.gx) + d.sy) / 4 + l / 2048;
    int x = (8 * int(blockIdx.x % d.gx) + d.sx) / 4 + (l % 2048) / 1024;
    if (y < 0 || x < 0 || y >= d.h / 4 || x >= d.w / 4)
        return -1;
    return (y * (d.w / 4) + x) * 1024 + ((l / 512) & 1) * 512 + (l & 511);
}
__device__ int input_address(Layout2 d, int t, int c) {
    if (d.seq != 7)
        return generic_address(d, label2(t, c));
    int z = (t & 1) | ((t & 2) << 3) | ((t & 4) >> 1) | ((t & 8) >> 1) | ((t & 16) << 1) |
            ((t & 32) >> 2);
    int pixel = (z & 1) | ((z & 14) << 2) | ((z >> 3) & 6);
    int row =
        (int(blockIdx.x / d.gx) * 8 + pixel / 8) * d.w + int(blockIdx.x % d.gx) * 8 + pixel % 8;
    return row * 16 + (c / 16) * d.h * d.w * 16 + c % 16;
}
__device__ int output_address(Layout2 d, int t, int c) {
    int a = input_address(d, t, c);
    if (a < 0)
        return -1;
    if (d.seq == 7) {
        int plane = a / (d.h * d.w * 16), z = a % (d.h * d.w * 16), y = z / (d.w * 16),
            x = (z / 16) % d.w, k = z % 16;
        return ((y / 4) * (d.w / 4) + x / 4) * 1024 + (k & 3) + ((x % 4 * 4 + k / 4) << 4) +
               (((y >> 1) & 1) << 2) + ((plane & 1) << 3) + ((y & 1) << 8) + ((plane >> 1) << 9);
    }
    if (d.seq != 150)
        return a;
    int band = a / 512, z = a % 512, c0 = band / ((d.h / 8) * (d.w / 4)) * 16 + z % 16;
    int rem = band % ((d.h / 8) * (d.w / 4));
    int y = (rem / (d.w / 4)) * 8 + ((rem % 2) * 32 + z / 16) / 8,
        x = (rem % (d.w / 4) / 2) * 8 + (z / 16) % 8;
    int n = (c0 & 1) | ((x & 1) << 1) | (((x >> 1) & 1) << 2) | (((c0 >> 1) & 1) << 3) |
            (((c0 >> 3) & 1) << 4) | (((y >> 2) & 1) << 5);
    int low = ((c0 >> 2) & 1) | (((x >> 2) & 1) << 1) | ((y & 1) << 2) | (((y >> 1) & 1) << 3) |
              (((x >> 3) & 1) << 4);
    int token = low + 32 * (x / 16 + (d.w / 16) * (y / 8 + (d.h / 8) * (c0 / 16)));
    low = token & 63;
    int outer = token >> 6;
    int row = (outer / (d.w / 16)) * 4 + ((low >> 3) & 1) + 2 * (low & 1);
    int col = (outer % (d.w / 16)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) +
              4 * ((low >> 4) & 1) + 8 * ((low >> 5) & 1);
    return (row * d.w + col) * 16 + (n / 16) * d.h * d.w * 16 + (n & 1) + ((n & 6) << 1) +
           ((n & 8) >> 2);
}
// Unused exported entry removed: prove_addresses

using namespace endpoint;
using namespace two_phase;

// Generated exhaustive byte permutations; every input register is initialized.
__device__ __forceinline__ Words<16> feature_packet(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 9, 10>(x);
    return x;
}
__device__ __forceinline__ Words<8> pv_packet(Words<8> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> output_packet(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}

__device__ __forceinline__ Words<16> special150_packet(Words<16> x) {
    x = swap_bits<4, 1, 6>(x);
    x = swap_bits<4, 2, 7>(x);
    x = swap_bits<4, 3, 8>(x);
    x = swap_bits<4, 4, 2>(x);
    x = swap_bits<4, 5, 3>(x);
    x = swap_bits<4, 1, 5>(x);
    x = swap_bits<4, 4, 1>(x);
    x = swap_bits<4, 4, 10>(x);
    x = swap_bits<4, 9, 4>(x);
    return x;
}
__device__ __forceinline__ int rank150_source(int rank) {
    return ((rank >> 0 & 1) << 0) | ((rank >> 8 & 1) << 1) | ((rank >> 2 & 1) << 2) |
           ((rank >> 3 & 1) << 3) | ((rank >> 4 & 1) << 4) | ((rank >> 5 & 1) << 5) |
           ((rank >> 7 & 1) << 6) | ((rank >> 1 & 1) << 7) | ((rank >> 10 & 1) << 8) |
           ((rank >> 6 & 1) << 9) | ((rank >> 9 & 1) << 10);
}

// Bounded four-class recipes, no runtime table allocation or global-domain LUT.
__device__ __constant__ short ds_bases[4][16][4] = {{{0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1}},
                                                    {{0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984},
                                                     {2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032}},
                                                    {{2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1}},
                                                    {{2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032},
                                                     {0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984}}};
__device__ __constant__ short ds_counts[4] = {64, 128, 64, 128};
__device__ __constant__ short ds_first[4][128] = {
    {0,   512, 256, 768, 128, 640, 384,  896, 16,  528, 272, 784, 144, 656, 400, 912, 32,  544, 288,
     800, 160, 672, 416, 928, 48,  560,  304, 816, 176, 688, 432, 944, 64,  576, 320, 832, 192, 704,
     448, 960, 80,  592, 336, 848, 208,  720, 464, 976, 96,  608, 352, 864, 224, 736, 480, 992, 112,
     624, 368, 880, 240, 752, 496, 1008, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0},
    {0,   512, 256, 768, 128, 640, 384, 896,  1024, 1536, 1280, 1792, 1152, 1664, 1408, 1920,
     16,  528, 272, 784, 144, 656, 400, 912,  1040, 1552, 1296, 1808, 1168, 1680, 1424, 1936,
     32,  544, 288, 800, 160, 672, 416, 928,  1056, 1568, 1312, 1824, 1184, 1696, 1440, 1952,
     48,  560, 304, 816, 176, 688, 432, 944,  1072, 1584, 1328, 1840, 1200, 1712, 1456, 1968,
     64,  576, 320, 832, 192, 704, 448, 960,  1088, 1600, 1344, 1856, 1216, 1728, 1472, 1984,
     80,  592, 336, 848, 208, 720, 464, 976,  1104, 1616, 1360, 1872, 1232, 1744, 1488, 2000,
     96,  608, 352, 864, 224, 736, 480, 992,  1120, 1632, 1376, 1888, 1248, 1760, 1504, 2016,
     112, 624, 368, 880, 240, 752, 496, 1008, 1136, 1648, 1392, 1904, 1264, 1776, 1520, 2032},
    {0,   512, 256, 768, 128, 640, 384,  896, 16,  528, 272, 784, 144, 656, 400, 912, 32,  544, 288,
     800, 160, 672, 416, 928, 48,  560,  304, 816, 176, 688, 432, 944, 64,  576, 320, 832, 192, 704,
     448, 960, 80,  592, 336, 848, 208,  720, 464, 976, 96,  608, 352, 864, 224, 736, 480, 992, 112,
     624, 368, 880, 240, 752, 496, 1008, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0},
    {1024, 1536, 1280, 1792, 1152, 1664, 1408, 1920, 0,   512, 256, 768, 128, 640, 384, 896,
     1040, 1552, 1296, 1808, 1168, 1680, 1424, 1936, 16,  528, 272, 784, 144, 656, 400, 912,
     1056, 1568, 1312, 1824, 1184, 1696, 1440, 1952, 32,  544, 288, 800, 160, 672, 416, 928,
     1072, 1584, 1328, 1840, 1200, 1712, 1456, 1968, 48,  560, 304, 816, 176, 688, 432, 944,
     1088, 1600, 1344, 1856, 1216, 1728, 1472, 1984, 64,  576, 320, 832, 192, 704, 448, 960,
     1104, 1616, 1360, 1872, 1232, 1744, 1488, 2000, 80,  592, 336, 848, 208, 720, 464, 976,
     1120, 1632, 1376, 1888, 1248, 1760, 1504, 2016, 96,  608, 352, 864, 224, 736, 480, 992,
     1136, 1648, 1392, 1904, 1264, 1776, 1520, 2032, 112, 624, 368, 880, 240, 752, 496, 1008}};
__device__ __constant__ unsigned char ds_ids[4][128] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
__device__ __constant__ short ds_deltas[4][1][16] = {
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}}};
namespace streamed {
__device__ __forceinline__ uint4 up_packet(const u8 *r, physical::Cross x, int raw, int height,
                                           int width) {
    int band = raw / 512, z = raw % 512, plane = (height / 8) * (width / 4);
    int n = band / plane * 16 + z % 16, rem = band % plane;
    int row = rem / (width / 4) * 8 + ((rem % 2) * 32 + z / 16) / 8;
    int col = rem % (width / 4) / 2 * 8 + (z / 16) % 8;
    int y = row / 16 + (height / 16) * (n >> 4);
    int st = (y & 1) | ((row & 1) << 1) | (((col >> 3) & 1) << 2) | (((n >> 2) & 1) << 3);
    st += 16 * (col / 16 + (width / 16) * (((row >> 3) & 1) + 2 * (y >> 1)));
    int sn = (n & 1) | (((col >> 1) & 1) << 1) | (((n >> 1) & 1) << 2) | ((col & 1) << 3) |
             (((n >> 3) & 1) << 4) | (((row >> 2) & 1) << 5);
    int pb = st * 64 + perm(sn);
    int gb = (n & 1) | ((col & 1) << 1) | (((col >> 1) & 1) << 2) | (((n >> 1) & 1) << 3) |
             (((n >> 3) & 1) << 4) | (((row >> 2) & 1) << 5);
    uint4 skip = *(const uint4 *)(r + x.skip + raw);
    Words<4> s{{skip.x, skip.y, skip.z, skip.w}}, o;
    each<4>([&](auto wt) {
        constexpr int w = decltype(wt)::value;
        u32 out = 0;
        each<2>([&](auto ht) {
            constexpr int pair = w * 2 + decltype(ht)::value;
            constexpr int pd = (pair & 1) * 8 + ((pair >> 1) & 1) * 512 + (pair >> 2) * 16;
            constexpr int gd = (pair & 1) * 8 + (pair >> 2) * 16;
            u32 bits = s.v[w] >> (decltype(ht)::value * 16);
            half2 a = __halves2half2(une4(bits & 255), une4((bits >> 8) & 255));
            half2 f = __hfma2(a, *(const half2 *)(x.gate + gb + gd),
                              *(const half2 *)(x.projection + pb + pd));
            out |= packed12::e4pair(pack(__low2half(f), __high2half(f)))
                   << (decltype(ht)::value * 16);
        });
        o.v[w] = out;
    });
    return {o.v[0], o.v[1], o.v[2], o.v[3]};
}
__device__ __forceinline__ int project_a(int row, int k) {
    return (k / 32) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 + ((row & 8) ? 4 : 0) +
           ((k & 16) ? 8 : 0) + (k & 3);
}
__device__ __forceinline__ int inverse_perm(int k) {
    return (k & 49) | ((k & 12) >> 1) | ((k & 2) << 2);
}
__device__ __forceinline__ int pool_class(int gx) {
    int row = blockIdx.x / gx;
    return row == 0 ? 2 : row == NR_H / 64 ? 3 : row == NR_H / 32 ? 0 : 1;
}
__device__ void pool(u8 *r, const half *values, u8 *scratch, physical::Cross x, int gx) {
    int cls = pool_class(gx), warp = threadIdx.y;
    const int *po = x.pool_output + blockIdx.x * 2048;
    // Body output and all projection readers are done. Only the dead low page is reused.
    for (int i = warp * 32 + lane(); i < 1024; i += 64)
        ((u32 *)scratch)[i] = 0;
    __syncthreads();
    for (int group = 0; group < 2; group++) {
        int column = (group * 2 + warp) * 32;
        uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#pragma unroll
        for (int kp = 0; kp < 2; kp++) {
            Words<4> words;
            each<4>([&](auto wt) {
                constexpr int word = decltype(wt)::value;
                u32 bits = 0;
                int row = lane() / 4 + (word & 1) * 8;
                each<2>([&](auto bt) {
                    constexpr int b = decltype(bt)::value * 2;
                    int k = kp * 32 + (word / 2) * 16 + (lane() & 3) * 4 + b;
                    u32 v[4];
                    each<4>([&](auto st) {
                        constexpr int s = decltype(st)::value;
                        int base = ds_bases[cls][row][s];
                        v[s] = base < 0 ? 0 : *(const u32 *)(values + base + inverse_perm(k));
                    });
                    u32 mean = packed12::mul(
                        packed12::add(packed12::add(v[0], v[1]), packed12::add(v[2], v[3])),
                        packed12::splat(h(.25f)));
                    bits |= packed12::e4pair(mean) << (b * 8);
                });
                words.v[word] = bits;
            });
            uint4 a = {words.v[0], words.v[1], words.v[2], words.v[3]};
#pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * 4096 + column * 32 + pair * 512 + lane() * 16);
                physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            }
        }
#pragma unroll
        for (int n = 0; n < 4; n++)
            for (int i = 0; i < 2; i++) {
                int row = lane() / 4 + i * 8, col = column + n * 8 + (lane() & 3) * 2;
                *(unsigned short *)(scratch + row * 128 + col) =
                    packed12::e4pair(i ? c[n].y : c[n].x);
            }
    }
    __syncthreads(); // all canonical local pool bytes exist before finite packet gather
    for (int packet = warp * 32 + lane(); packet < ds_counts[cls]; packet += 64) {
        int first = ds_first[cls][packet], id = ds_ids[cls][packet];
        Words<4> words;
        each<4>([&](auto wt) {
            constexpr int word = decltype(wt)::value;
            u32 bits = 0;
            each<4>([&](auto bt) {
                constexpr int b = decltype(bt)::value;
                bits |= u32(scratch[first + ds_deltas[cls][id][word * 4 + b]]) << (8 * b);
            });
            words.v[word] = bits;
        });
        int address = po[first];
        if (address >= 0)
            *(uint4 *)(r + x.pool_out + address) = {words.v[0], words.v[1], words.v[2], words.v[3]};
    }
}
} // namespace streamed
// Unused exported entry removed: streamed_project2

struct TwoWeights {
    const u8 *expand, *reduce, *tail, *qkv, *projection;
    const half *bias, *ffn_gate, *attn_gate, *scale;
    const int *p, *ip, *pk;
};

static_assert(sizeof(TwoWeights) == 96, "Retain original TwoWeights ABI");

// Raw B128 offsets are relative to live p.r40 fields, not lane-B8 copies.
template <int M, int N>
__device__ __forceinline__ void raw_weight(HC<M, N> &c, const Words<M / 4> &a, const u8 *w) {
    each<N / 16>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + pair * 512 + lane() * 16);
        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            each<2>([&](auto ht) {
                constexpr int n = pair * 2 + decltype(ht)::value;
                u32 bb[2];
                if constexpr (decltype(ht)::value == 0) {
                    bb[0] = b.x;
                    bb[1] = b.y;
                } else {
                    bb[0] = b.z;
                    bb[1] = b.w;
                }
                Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
                mma8(z, aa, bb);
                c.v[(m * (N / 8) + n) * 2] = z.v[0];
                c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
            });
        });
    });
}
__device__ __forceinline__ int aoffset(int row, int k) {
    return (k / 32) * 2048 + (row / 16) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 +
           ((row & 8) ? 4 : 0) + ((k & 16) ? 8 : 0) + (k & 3);
}
extern "C" __global__ __launch_bounds__(64) void streamed2(u8 *r, TwoWeights w, const Layout2 *meta,
                                                           const int *unused, int in, int out,
                                                           int counter, physical::Cross cross) {
    extern __shared__ __align__(16) u8 exchange[];
    Layout2 layout = *meta;
    int warp = threadIdx.y;
    for (int part = 0; part < 4; part++) {
        int l = (warp * 32 + lane()) * 16 + part * 1024;
        if (layout.seq == 7) {
            int t = l / 64, c = l % 64, a = input_address(layout, t, c);
            uint4 packet = *(const uint4 *)(r + in + a);
            *(u32 *)(exchange + label2(t, c)) = packet.x;
            *(u32 *)(exchange + label2(t, c + 4)) = packet.y;
            *(u32 *)(exchange + label2(t, c + 8)) = packet.z;
            *(u32 *)(exchange + label2(t, c + 12)) = packet.w;
        } else {
            int a = generic_address(layout, l);
            uint4 packet = {0, 0, 0, 0};
            if (a >= 0) {
                if (cross.up_inverse) {
                    packet = streamed::up_packet(r, cross, a, layout.h, layout.w);
                } else
                    packet = *(const uint4 *)(r + in + a);
            }
            *(uint4 *)(exchange + l) = packet;
        }
    }
    __syncthreads();
    // One M32 warp. Materialize its input directly as both K32 A packets,
    // and its skip as Half C. No intermediate dynamic input Tile survives.
    Words<16> input;
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            int t = warp * 32 + m * 16 + lane() / 4, k = kp * 32 + (lane() & 3) * 4;
            uint4 a = *(const uint4 *)(exchange + label2(t, k));
            input.v[kp * 8 + m * 4] = a.x;
            input.v[kp * 8 + m * 4 + 1] = a.y;
            input.v[kp * 8 + m * 4 + 2] = a.z;
            input.v[kp * 8 + m * 4 + 3] = a.w;
        });
    });
    auto ff = activation::fill<32, 64>([&](int t, int c) {
        return __hmul(une4(exchange[label2(warp * 32 + t, perm(c))]), w.ffn_gate[c]);
    });
// New equivalent: persistent M32 reduce C, only one N32 hidden slab live.
#pragma unroll 1
    for (int stream = 0; stream < 2; stream++) {
        auto reduced = activation::fill<32, 32>([](int, int) { return h(0); });
#pragma unroll 1
        for (int slab = 0; slab < 4; slab++) {
            auto hidden = activation::fill<32, 32>([](int, int) { return h(0); });
            each<2>([&](auto kt) {
                constexpr int kp = decltype(kt)::value;
                raw_weight<32, 32>(hidden, slice<kp * 8, 8>(input),
                                   w.expand + stream * 8192 + kp * 4096 + slab * 1024);
            });
            auto ha = pv_packet(two_phase::encode<32, 32>(hidden, packed12::Activate{}));
            raw_weight<32, 32>(reduced, ha, w.reduce + stream * 4096 + slab * 1024);
        }
        auto ra = pv_packet(two_phase::encode<32, 32>(reduced, packed12::Identity{}));
        raw_weight<32, 64>(ff, ra, w.tail + stream * 2048);
    }
    auto feature = feature_packet(two_phase::encode<32, 64>(ff, packed12::Identity{}));
    __syncthreads(); // both warps finish all input reads before FF replaces the page
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            *(uint4 *)(exchange + kp * 2048 + (warp * 2 + m) * 512 + lane() * 16) = {
                feature.v[kp * 8 + m * 4], feature.v[kp * 8 + m * 4 + 1],
                feature.v[kp * 8 + m * 4 + 2], feature.v[kp * 8 + m * 4 + 3]};
        });
    });
    __syncthreads(); // rowgroup producers hand all 64 rows to head consumers
    // Residual is the original E4 FF shared reread, NOT 1H Half FF residual.
    // Save its encoded bits, postponing Half gate multiplication to projection.
    Words<16> residual, q, k, v;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 bits = 0;
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int t = m * 16 + lane() / 4 + i * 8, c = warp * 32 + n * 8 + (lane() & 3) * 2;
                bits |= u32(*(unsigned short *)(exchange + aoffset(t, perm(c)))) << (16 * i);
            });
            residual.v[m * 4 + n] = bits;
        });
    });
    each<2>([&](auto st) {
        constexpr int first = decltype(st)::value * 2;
        auto z = activation::fill<32, 96>([](int, int) { return h(0); });
        each<2>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            Words<8> a;
            each<2>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                uint4 packet =
                    *(const uint4 *)(exchange + kp * 2048 + (first + m) * 512 + lane() * 16);
                a.v[m * 4] = packet.x;
                a.v[m * 4 + 1] = packet.y;
                a.v[m * 4 + 2] = packet.z;
                a.v[m * 4 + 3] = packet.w;
            });
            raw_weight<32, 96>(z, a, w.qkv + warp * 3072 + kp * 6144);
        });
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            HC<16, 96> row;
            each<24>([&](auto it) {
                constexpr int i = decltype(it)::value;
                row.v[i] = z.v[m * 24 + i];
            });
            half iq[2], ik[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                iq[i] = two_phase::norm<i, 0>(row);
                ik[i] = two_phase::norm<i, 32>(row);
            });
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 qb = 0, kb = 0;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    qb |= packed12::e4pair(
                              packed12::mul(packed12::mul(row.v[n * 2 + i], packed12::splat(iq[i])),
                                            packed12::splat(w.scale[warp])))
                          << (16 * i);
                    kb |= packed12::e4pair(
                              packed12::mul(row.v[(n + 4) * 2 + i], packed12::splat(ik[i])))
                          << (16 * i);
                });
                q.v[(first + m) * 4 + n] = qb;
                k.v[(first + m) * 4 + n] = kb;
                v.v[(first + m) * 4 + n] =
                    packed12::e4four(row.v[(n + 8) * 2], row.v[(n + 8) * 2 + 1]);
            });
        });
    });
    __syncthreads(); // residual and all QKV reads finish before slab reuse
    auto qa = query_a(q);
    auto kb = key_b(k);
    auto vb = value_b(v);
    each<2>([&](auto st) {
        constexpr int slab = decltype(st)::value;
        HC<32, 64> logits;
        each<8>([&](auto pt) {
            constexpr int packet = decltype(pt)::value;
            const u8 *seed =
                w.qkv + 0x3000 + warp * 8192 + slab * 4096 + packet * 512 + lane() * 16;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(logits.v[packet * 4]), "=r"(logits.v[packet * 4 + 1]),
                           "=r"(logits.v[packet * 4 + 2]), "=r"(logits.v[packet * 4 + 3])
                         : "l"(seed));
        });
        two_phase::matrix<32, 64>(logits, slice<slab * 8, 8>(qa), kb);
        each<32>([&](auto it) {
            constexpr int i = decltype(it)::value;
            logits.v[i] = packed12::exponent_pair(logits.v[i]);
        });
        half inv[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            inv[i] = two_phase::den<i>(logits);
        });
        Words<16> prob;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 word = 0;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    word |= packed12::e4pair(packed12::mul(logits.v[(m * 8 + n) * 2 + i],
                                                           packed12::splat(inv[m * 2 + i])))
                            << (16 * i);
                });
                prob.v[m * 8 + n] = word;
            });
        });
        auto pa = prob_a(prob);
        auto a = activation::fill<32, 32>([](int, int) { return h(0); });
        each<2>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            two_phase::matrix<32, 32>(a, slice<kp * 8, 8>(pa), slice<kp * 8, 8>(vb));
        });
        auto pv = pv_packet(two_phase::encode<32, 32>(a, packed12::Identity{}));
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            *(uint4 *)(exchange + warp * 2048 + (slab * 2 + m) * 512 + lane() * 16) = {
                pv.v[m * 4], pv.v[m * 4 + 1], pv.v[m * 4 + 2], pv.v[m * 4 + 3]};
        });
    });
    __syncthreads();
    HC<64, 32> result;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                u32 bits = residual.v[m * 4 + n] >> (16 * i);
                int c = warp * 32 + n * 8 + (lane() & 3) * 2;
                result.v[(m * 4 + n) * 2 + i] =
                    packed12::mul(pack(une4(bits & 255), une4((bits >> 8) & 255)),
                                  pack(w.attn_gate[c], w.attn_gate[c + 1]));
            });
        });
    });
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        Words<16> a;
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            uint4 packet = *(const uint4 *)(exchange + kp * 2048 + m * 512 + lane() * 16);
            a.v[m * 4] = packet.x;
            a.v[m * 4 + 1] = packet.y;
            a.v[m * 4 + 2] = packet.z;
            a.v[m * 4 + 3] = packet.w;
        });
        raw_weight<64, 32>(result, a, w.projection + warp * 1024 + kp * 2048);
    });
    __syncthreads();
    if (cross.pool_input) {
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    int t = m * 16 + lane() / 4 + i * 8, c = warp * 32 + n * 8 + (lane() & 3) * 2;
                    *(u32 *)((half *)(exchange + 4096) + t * 64 + c) =
                        result.v[(m * 4 + n) * 2 + i];
                });
            });
        });
    }
    auto encoded = two_phase::encode<64, 32>(result, packed12::Identity{});
    if (layout.seq == 150) {
        auto published = special150_packet(encoded);
        each<4>([&](auto pt) {
            constexpr int part = decltype(pt)::value;
            int src = rank150_source(part * 512 + lane() * 16), word = src / 128,
                ll = (src % 128) / 4, byte = src % 4;
            int t = word / 4 * 16 + ll / 4 + byte / 2 * 8,
                c = warp * 32 + word % 4 * 8 + (ll & 3) * 2 + (byte & 1);
            int address = output_address(layout, t, perm(c));
            if (address >= 0)
                *(uint4 *)(r + out + address) = {published.v[part * 4], published.v[part * 4 + 1],
                                                 published.v[part * 4 + 2],
                                                 published.v[part * 4 + 3]};
        });
    } else {
        auto published = output_packet(encoded);
        each<4>([&](auto pt) {
            constexpr int part = decltype(pt)::value;
            int address = generic_address(layout, warp * 512 + lane() * 16 + part * 1024);
            if (address >= 0)
                *(uint4 *)(r + out + address) = {published.v[part * 4], published.v[part * 4 + 1],
                                                 published.v[part * 4 + 2],
                                                 published.v[part * 4 + 3]};
        });
    }
    if (cross.pool_input) {
        __syncthreads();
        streamed::pool(r, (half *)(exchange + 4096), exchange, cross, layout.gx);
    }
    if (counter >= 0 && warp == 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

} // namespace nr_outer_two_3
// ============================================================================
// OUTER wide_transition_packet/two.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_two_4 {
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
// Index bits are [register][lane][unit]. All register indexing below is
// compile-time. A unit is one E4 byte or one Half, never a numeric conversion.
template <int U, int A, int B, int N>
__device__ __forceinline__ Words<N> xor_bits(const Words<N> &x) {
    constexpr int S = U == 4 ? 2 : 1, L = S + 5;
    Words<N> y;
    each<N>([&](auto it) {
        constexpr int I = decltype(it)::value;
        if constexpr (A >= L) {
            constexpr int J = I ^ (1 << (A - L));
            if constexpr (B >= L)
                y.v[I] = x.v[((I >> (B - L)) & 1) ? J : I];
            else if constexpr (B >= S)
                y.v[I] = (lane() & (1 << (B - S))) ? x.v[J] : x.v[I];
            else {
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b + ((((b / (4 / U)) >> B) & 1) ? 4 : 0)) << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], x.v[J], mask);
            }
        } else if constexpr (A >= S) {
            if constexpr (B >= L)
                y.v[I] =
                    __shfl_sync(0xffffffff, x.v[I], lane() ^ ((((I >> (B - L)) & 1)) << (A - S)));
            else if constexpr (B >= S)
                y.v[I] = __shfl_sync(0xffffffff, x.v[I],
                                     lane() ^ (((lane() >> (B - S)) & 1) << (A - S)));
            else {
                u32 other = __shfl_xor_sync(0xffffffff, x.v[I], 1 << (A - S));
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b + ((((b / (4 / U)) >> B) & 1) ? 4 : 0)) << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], other, mask);
            }
        } else {
            constexpr unsigned switched = []() {
                unsigned m = 0;
                for (int b = 0; b < 4; ++b)
                    m |= (b ^ (1 << (A + (U == 4 ? 0 : 1)))) << (b * 4);
                return m;
            }();
            if constexpr (B >= L)
                y.v[I] = __byte_perm(x.v[I], 0, ((I >> (B - L)) & 1) ? switched : 0x3210);
            else if constexpr (B >= S)
                y.v[I] = __byte_perm(x.v[I], 0, (lane() & (1 << (B - S))) ? switched : 0x3210);
            else {
                constexpr unsigned mask = []() {
                    unsigned m = 0;
                    for (int b = 0; b < 4; ++b)
                        m |= (b ^ ((((b / (4 / U)) >> B) & 1) << (A + (U == 4 ? 0 : 1))))
                             << (b * 4);
                    return m;
                }();
                y.v[I] = __byte_perm(x.v[I], 0, mask);
            }
        }
    });
    return y;
}
template <int U, int A, int B, int N> __device__ __forceinline__ Words<N> swap_bits(Words<N> x) {
    return xor_bits<U, A, B>(xor_bits<U, B, A>(xor_bits<U, A, B>(x)));
}
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
        y.v[j] = u32(e4(f(unpack(a, 0)))) | (u32(e4(f(unpack(a, 1)))) << 8) |
                 (u32(e4(f(unpack(b, 0)))) << 16) | (u32(e4(f(unpack(b, 1)))) << 24);
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
    half t[8];
    each<8>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        half a = component<M, N, Row, Base + j>(x), b = component<M, N, Row, Base + 8 + j>(x);
        half c = component<M, N, Row, Base + 16 + j>(x), d = component<M, N, Row, Base + 24 + j>(x);
        t[j] = __hadd(__hfma(a, a, __hmul(c, c)), __hfma(b, b, __hmul(d, d)));
    });
    half a = __hadd(__hadd(t[0], t[4]), __hadd(t[2], t[6]));
    half b = __hadd(__hadd(t[1], t[5]), __hadd(t[3], t[7]));
    return rsqrt_half(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
template <int Row> __device__ __forceinline__ half denominator(const HC<32, 64> &x) {
    half t[8];
    each<8>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        half v = __hadd(component<32, 64, Row, j>(x), component<32, 64, Row, j + 8>(x));
        v = __hadd(v, __hadd(component<32, 64, Row, j + 48>(x), component<32, 64, Row, j + 56>(x)));
        v = __hadd(v, __hadd(component<32, 64, Row, j + 16>(x), component<32, 64, Row, j + 24>(x)));
        t[j] =
            __hadd(v, __hadd(component<32, 64, Row, j + 32>(x), component<32, 64, Row, j + 40>(x)));
    });
    half a = __hadd(__hadd(__hadd(t[0], t[2]), t[4]), t[6]);
    half b = __hadd(__hadd(__hadd(t[1], t[3]), t[5]), t[7]);
    return reciprocal(h(fmaxf(f(__hadd(a, b)), 6.198883056640625e-05f)));
}
} // namespace activation
namespace two_phase {
using namespace activation;
__device__ __forceinline__ Words<16> hidden_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<4> tail_a(Words<4> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> query_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> prob_a(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 9, 10>(x);
    return x;
}
__device__ __forceinline__ Words<16> key_b(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 7, 8>(x);
    return x;
}
__device__ __forceinline__ Words<16> value_b(Words<16> x) {
    x = swap_bits<4, 0, 4>(x);
    x = swap_bits<4, 2, 5>(x);
    x = swap_bits<4, 3, 6>(x);
    x = swap_bits<4, 7, 9>(x);
    x = swap_bits<4, 8, 9>(x);
    return x;
}
} // namespace two_phase
namespace packed12 {
using namespace endpoint;
__device__ __forceinline__ half2 hh(u32 x) {
    return __halves2half2(unpack(x, 0), unpack(x, 1));
}
__device__ __forceinline__ u32 bits(half2 x) {
    return pack(__low2half(x), __high2half(x));
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
namespace two_phase {
using namespace endpoint;
template <int Offset, int Count, int N>
__device__ __forceinline__ Words<Count> slice(const Words<N> &x) {
    Words<Count> y;
    each<Count>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        y.v[j] = x.v[Offset + j];
    });
    return y;
}
template <int M, int N, class F>
__device__ __forceinline__ Words<M / 16 * (N / 8)> encode(const HC<M, N> &x, F f) {
    Words<M / 16 * (N / 8)> y;
    each<M / 16 * (N / 8)>([&](auto jt) {
        constexpr int j = decltype(jt)::value;
        y.v[j] = packed12::e4four(f(x.v[2 * j]), f(x.v[2 * j + 1]));
    });
    return y;
}
template <int M, int N>
__device__ __forceinline__ void matrix(HC<M, N> &c, const Words<M / 4> &a, const Words<N / 4> &b) {
    each<N / 8>([&](auto nt) {
        constexpr int n = decltype(nt)::value;
        u32 bb[2] = {b.v[n * 2], b.v[n * 2 + 1]};
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
template <int Row, int Base> __device__ __forceinline__ half norm(const HC<16, 96> &x) {
    constexpr int n = Base / 8;
    u32 a = x.v[n * 2 + Row], b = x.v[(n + 1) * 2 + Row], c = x.v[(n + 2) * 2 + Row],
        d = x.v[(n + 3) * 2 + Row];
    return packed12::norm_finish(packed12::add(packed12::fma(a, a, packed12::mul(c, c)),
                                               packed12::fma(b, b, packed12::mul(d, d))));
}
template <int Row> __device__ __forceinline__ half den(const HC<32, 64> &x) {
    constexpr int base = (Row / 2) * 16, i = Row % 2;
    u32 s = packed12::add(x.v[base + i], x.v[base + 2 + i]);
    s = packed12::add(s, packed12::add(x.v[base + 4 + i], x.v[base + 6 + i]));
    s = packed12::add(s, packed12::add(x.v[base + 8 + i], x.v[base + 10 + i]));
    s = packed12::add(s, packed12::add(x.v[base + 12 + i], x.v[base + 14 + i]));
    return packed12::den_finish(s);
}
__device__ __forceinline__ int perm(int c) {
    return (c & 49) | ((c & 6) << 1) | ((c & 8) >> 2);
}
} // namespace two_phase

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

struct Layout2 {
    int h, w, gx, sx, sy, seq;
};
__device__ int label2(int t, int c) {
    return (c & 3) | ((c & 12) << 2) | ((c & 16) >> 1) | ((c & 32) << 4) | ((t & 7) << 6) |
           ((t & 8) >> 1) | ((t & 48) << 6);
}
__device__ int token2(int l) {
    return ((l >> 6) & 7) | ((l & 4) << 1) | ((l >> 6) & 48);
}
__device__ int channel2(int l) {
    return (l & 3) | ((l >> 2) & 12) | ((l & 8) << 1) | ((l >> 4) & 32);
}
__device__ int generic_address(Layout2 d, int l) {
    int y = (8 * int(blockIdx.x / d.gx) + d.sy) / 4 + l / 2048;
    int x = (8 * int(blockIdx.x % d.gx) + d.sx) / 4 + (l % 2048) / 1024;
    if (y < 0 || x < 0 || y >= d.h / 4 || x >= d.w / 4)
        return -1;
    return (y * (d.w / 4) + x) * 1024 + ((l / 512) & 1) * 512 + (l & 511);
}
__device__ int input_address(Layout2 d, int t, int c) {
    if (d.seq != 7)
        return generic_address(d, label2(t, c));
    int z = (t & 1) | ((t & 2) << 3) | ((t & 4) >> 1) | ((t & 8) >> 1) | ((t & 16) << 1) |
            ((t & 32) >> 2);
    int pixel = (z & 1) | ((z & 14) << 2) | ((z >> 3) & 6);
    int row =
        (int(blockIdx.x / d.gx) * 8 + pixel / 8) * d.w + int(blockIdx.x % d.gx) * 8 + pixel % 8;
    return row * 16 + (c / 16) * d.h * d.w * 16 + c % 16;
}
__device__ int output_address(Layout2 d, int t, int c) {
    int a = input_address(d, t, c);
    if (a < 0)
        return -1;
    if (d.seq == 7) {
        int plane = a / (d.h * d.w * 16), z = a % (d.h * d.w * 16), y = z / (d.w * 16),
            x = (z / 16) % d.w, k = z % 16;
        return ((y / 4) * (d.w / 4) + x / 4) * 1024 + (k & 3) + ((x % 4 * 4 + k / 4) << 4) +
               (((y >> 1) & 1) << 2) + ((plane & 1) << 3) + ((y & 1) << 8) + ((plane >> 1) << 9);
    }
    if (d.seq != 150)
        return a;
    int band = a / 512, z = a % 512, c0 = band / ((d.h / 8) * (d.w / 4)) * 16 + z % 16;
    int rem = band % ((d.h / 8) * (d.w / 4));
    int y = (rem / (d.w / 4)) * 8 + ((rem % 2) * 32 + z / 16) / 8,
        x = (rem % (d.w / 4) / 2) * 8 + (z / 16) % 8;
    int n = (c0 & 1) | ((x & 1) << 1) | (((x >> 1) & 1) << 2) | (((c0 >> 1) & 1) << 3) |
            (((c0 >> 3) & 1) << 4) | (((y >> 2) & 1) << 5);
    int low = ((c0 >> 2) & 1) | (((x >> 2) & 1) << 1) | ((y & 1) << 2) | (((y >> 1) & 1) << 3) |
              (((x >> 3) & 1) << 4);
    int token = low + 32 * (x / 16 + (d.w / 16) * (y / 8 + (d.h / 8) * (c0 / 16)));
    low = token & 63;
    int outer = token >> 6;
    int row = (outer / (d.w / 16)) * 4 + ((low >> 3) & 1) + 2 * (low & 1);
    int col = (outer % (d.w / 16)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) +
              4 * ((low >> 4) & 1) + 8 * ((low >> 5) & 1);
    return (row * d.w + col) * 16 + (n / 16) * d.h * d.w * 16 + (n & 1) + ((n & 6) << 1) +
           ((n & 8) >> 2);
}
// Unused exported entry removed: prove_addresses

using namespace endpoint;
using namespace two_phase;

// Generated exhaustive byte permutations; every input register is initialized.
__device__ __forceinline__ Words<16> feature_packet(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    x = swap_bits<4, 9, 10>(x);
    return x;
}
__device__ __forceinline__ Words<8> pv_packet(Words<8> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}
__device__ __forceinline__ Words<16> output_packet(Words<16> x) {
    x = swap_bits<4, 1, 7>(x);
    return x;
}

__device__ __forceinline__ Words<16> special150_packet(Words<16> x) {
    x = swap_bits<4, 1, 6>(x);
    x = swap_bits<4, 2, 7>(x);
    x = swap_bits<4, 3, 8>(x);
    x = swap_bits<4, 4, 2>(x);
    x = swap_bits<4, 5, 3>(x);
    x = swap_bits<4, 1, 5>(x);
    x = swap_bits<4, 4, 1>(x);
    x = swap_bits<4, 4, 10>(x);
    x = swap_bits<4, 9, 4>(x);
    return x;
}
__device__ __forceinline__ int rank150_source(int rank) {
    return ((rank >> 0 & 1) << 0) | ((rank >> 8 & 1) << 1) | ((rank >> 2 & 1) << 2) |
           ((rank >> 3 & 1) << 3) | ((rank >> 4 & 1) << 4) | ((rank >> 5 & 1) << 5) |
           ((rank >> 7 & 1) << 6) | ((rank >> 1 & 1) << 7) | ((rank >> 10 & 1) << 8) |
           ((rank >> 6 & 1) << 9) | ((rank >> 9 & 1) << 10);
}

// Bounded four-class recipes, no runtime table allocation or global-domain LUT.
__device__ __constant__ short ds_bases[4][16][4] = {{{0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1}},
                                                    {{0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984},
                                                     {2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032}},
                                                    {{2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1},
                                                     {-1, -1, -1, -1}},
                                                    {{2048, 2112, 2304, 2368},
                                                     {2560, 2624, 2816, 2880},
                                                     {3072, 3136, 3328, 3392},
                                                     {3584, 3648, 3840, 3904},
                                                     {2176, 2240, 2432, 2496},
                                                     {2688, 2752, 2944, 3008},
                                                     {3200, 3264, 3456, 3520},
                                                     {3712, 3776, 3968, 4032},
                                                     {0, 64, 256, 320},
                                                     {512, 576, 768, 832},
                                                     {1024, 1088, 1280, 1344},
                                                     {1536, 1600, 1792, 1856},
                                                     {128, 192, 384, 448},
                                                     {640, 704, 896, 960},
                                                     {1152, 1216, 1408, 1472},
                                                     {1664, 1728, 1920, 1984}}};
__device__ __constant__ short ds_counts[4] = {64, 128, 64, 128};
__device__ __constant__ short ds_first[4][128] = {
    {0,   512, 256, 768, 128, 640, 384,  896, 16,  528, 272, 784, 144, 656, 400, 912, 32,  544, 288,
     800, 160, 672, 416, 928, 48,  560,  304, 816, 176, 688, 432, 944, 64,  576, 320, 832, 192, 704,
     448, 960, 80,  592, 336, 848, 208,  720, 464, 976, 96,  608, 352, 864, 224, 736, 480, 992, 112,
     624, 368, 880, 240, 752, 496, 1008, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0},
    {0,   512, 256, 768, 128, 640, 384, 896,  1024, 1536, 1280, 1792, 1152, 1664, 1408, 1920,
     16,  528, 272, 784, 144, 656, 400, 912,  1040, 1552, 1296, 1808, 1168, 1680, 1424, 1936,
     32,  544, 288, 800, 160, 672, 416, 928,  1056, 1568, 1312, 1824, 1184, 1696, 1440, 1952,
     48,  560, 304, 816, 176, 688, 432, 944,  1072, 1584, 1328, 1840, 1200, 1712, 1456, 1968,
     64,  576, 320, 832, 192, 704, 448, 960,  1088, 1600, 1344, 1856, 1216, 1728, 1472, 1984,
     80,  592, 336, 848, 208, 720, 464, 976,  1104, 1616, 1360, 1872, 1232, 1744, 1488, 2000,
     96,  608, 352, 864, 224, 736, 480, 992,  1120, 1632, 1376, 1888, 1248, 1760, 1504, 2016,
     112, 624, 368, 880, 240, 752, 496, 1008, 1136, 1648, 1392, 1904, 1264, 1776, 1520, 2032},
    {0,   512, 256, 768, 128, 640, 384,  896, 16,  528, 272, 784, 144, 656, 400, 912, 32,  544, 288,
     800, 160, 672, 416, 928, 48,  560,  304, 816, 176, 688, 432, 944, 64,  576, 320, 832, 192, 704,
     448, 960, 80,  592, 336, 848, 208,  720, 464, 976, 96,  608, 352, 864, 224, 736, 480, 992, 112,
     624, 368, 880, 240, 752, 496, 1008, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0},
    {1024, 1536, 1280, 1792, 1152, 1664, 1408, 1920, 0,   512, 256, 768, 128, 640, 384, 896,
     1040, 1552, 1296, 1808, 1168, 1680, 1424, 1936, 16,  528, 272, 784, 144, 656, 400, 912,
     1056, 1568, 1312, 1824, 1184, 1696, 1440, 1952, 32,  544, 288, 800, 160, 672, 416, 928,
     1072, 1584, 1328, 1840, 1200, 1712, 1456, 1968, 48,  560, 304, 816, 176, 688, 432, 944,
     1088, 1600, 1344, 1856, 1216, 1728, 1472, 1984, 64,  576, 320, 832, 192, 704, 448, 960,
     1104, 1616, 1360, 1872, 1232, 1744, 1488, 2000, 80,  592, 336, 848, 208, 720, 464, 976,
     1120, 1632, 1376, 1888, 1248, 1760, 1504, 2016, 96,  608, 352, 864, 224, 736, 480, 992,
     1136, 1648, 1392, 1904, 1264, 1776, 1520, 2032, 112, 624, 368, 880, 240, 752, 496, 1008}};
__device__ __constant__ unsigned char ds_ids[4][128] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
__device__ __constant__ short ds_deltas[4][1][16] = {
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
    {{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}}};
#include "addresses.cuh"
namespace streamed {
__device__ __forceinline__ uint4 up_packet(const u8 *r, physical::Cross x, int raw, int height,
                                           int width) {
    int band = raw / 512, z = raw % 512, plane = (height / 8) * (width / 4);
    int n = band / plane * 16 + z % 16, rem = band % plane;
    int row = rem / (width / 4) * 8 + ((rem % 2) * 32 + z / 16) / 8;
    int col = rem % (width / 4) / 2 * 8 + (z / 16) % 8;
    int y = row / 16 + (height / 16) * (n >> 4);
    int st = (y & 1) | ((row & 1) << 1) | (((col >> 3) & 1) << 2) | (((n >> 2) & 1) << 3);
    st += 16 * (col / 16 + (width / 16) * (((row >> 3) & 1) + 2 * (y >> 1)));
    int sn = (n & 1) | (((col >> 1) & 1) << 1) | (((n >> 1) & 1) << 2) | ((col & 1) << 3) |
             (((n >> 3) & 1) << 4) | (((row >> 2) & 1) << 5);
    int pb = st * 64 + perm(sn);
    int gb = (n & 1) | ((col & 1) << 1) | (((col >> 1) & 1) << 2) | (((n >> 1) & 1) << 3) |
             (((n >> 3) & 1) << 4) | (((row >> 2) & 1) << 5);
    uint4 skip = *(const uint4 *)(r + x.skip + raw);
    Words<4> s{{skip.x, skip.y, skip.z, skip.w}}, o;
    each<4>([&](auto wt) {
        constexpr int w = decltype(wt)::value;
        u32 out = 0;
        each<2>([&](auto ht) {
            constexpr int pair = w * 2 + decltype(ht)::value;
            constexpr int pd = (pair & 1) * 8 + ((pair >> 1) & 1) * 512 + (pair >> 2) * 16;
            constexpr int gd = (pair & 1) * 8 + (pair >> 2) * 16;
            u32 bits = s.v[w] >> (decltype(ht)::value * 16);
            half2 a = __halves2half2(une4(bits & 255), une4((bits >> 8) & 255));
            half2 f = __hfma2(a, *(const half2 *)(x.gate + gb + gd),
                              *(const half2 *)(x.projection + pb + pd));
            out |= packed12::e4pair(pack(__low2half(f), __high2half(f)))
                   << (decltype(ht)::value * 16);
        });
        o.v[w] = out;
    });
    return {o.v[0], o.v[1], o.v[2], o.v[3]};
}
__device__ __forceinline__ int project_a(int row, int k) {
    return (k / 32) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 + ((row & 8) ? 4 : 0) +
           ((k & 16) ? 8 : 0) + (k & 3);
}
__device__ __forceinline__ int inverse_perm(int k) {
    return (k & 49) | ((k & 12) >> 1) | ((k & 2) << 2);
}
__device__ __forceinline__ int pool_class(int gx) {
    int row = blockIdx.x / gx;
    return row == 0 ? 2 : row == NR_H / 64 ? 3 : row == NR_H / 32 ? 0 : 1;
}
__device__ void pool(u8 *r, const half *values, u8 *scratch, physical::Cross x, int gx) {
    int warp = threadIdx.y;
    const int *po = x.pool_output + blockIdx.x * 2048;
    const int *pi = x.pool_input + blockIdx.x * 4096;
    // The fixed odd-half-band recipes are not valid for every packet height.
    // Consume the exact per-shape full-K ownership map in Half element units.
    // Body output and all projection readers are done. Only the dead low page is reused.
    for (int i = warp * 32 + lane(); i < 1024; i += 64)
        ((u32 *)scratch)[i] = 0;
    __syncthreads();
    for (int group = 0; group < 2; group++) {
        int column = (group * 2 + warp) * 32;
        uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#pragma unroll
        for (int kp = 0; kp < 2; kp++) {
            Words<4> words;
            each<4>([&](auto wt) {
                constexpr int word = decltype(wt)::value;
                u32 bits = 0;
                int row = lane() / 4 + (word & 1) * 8;
                each<2>([&](auto bt) {
                    constexpr int b = decltype(bt)::value * 2;
                    int k = kp * 32 + (word / 2) * 16 + (lane() & 3) * 4 + b;
                    u32 v[4];
                    each<4>([&](auto st) {
                        constexpr int s = decltype(st)::value;
                        int base = pi[(row * 4 + s) * 64 + k];
                        v[s] = base < 0 ? 0 : *(const u32 *)(values + base);
                    });
                    u32 mean = packed12::mul(
                        packed12::add(packed12::add(v[0], v[1]), packed12::add(v[2], v[3])),
                        packed12::splat(h(.25f)));
                    bits |= packed12::e4pair(mean) << (b * 8);
                });
                words.v[word] = bits;
            });
            uint4 a = {words.v[0], words.v[1], words.v[2], words.v[3]};
#pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                uint4 b =
                    *(const uint4 *)(x.matrix + kp * 4096 + column * 32 + pair * 512 + lane() * 16);
                physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
                physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
            }
        }
#pragma unroll
        for (int n = 0; n < 4; n++)
            for (int i = 0; i < 2; i++) {
                int row = lane() / 4 + i * 8, col = column + n * 8 + (lane() & 3) * 2;
                *(unsigned short *)(scratch + row * 128 + col) =
                    packed12::e4pair(i ? c[n].y : c[n].x);
            }
    }
    __syncthreads(); // all canonical local pool bytes exist before finite packet gather
    for (int packet = warp * 32 + lane(); packet < 128; packet += 64) {
        int row = packet / 8, n = (packet % 8) * 16, address = po[row * 128 + n];
        if (address >= 0) {
            Words<4> words;
            each<4>([&](auto wt) {
                constexpr int word = decltype(wt)::value;
                u32 bits = 0;
                each<4>([&](auto bt) {
                    constexpr int b = decltype(bt)::value;
                    bits |= u32(scratch[row * 128 + n + inverse_perm(word * 4 + b)]) << (8 * b);
                });
                words.v[word] = bits;
            });
            *(uint4 *)(r + x.pool_out + address) = {words.v[0], words.v[1], words.v[2], words.v[3]};
        }
    }
}
} // namespace streamed
extern "C" __global__ __launch_bounds__(64) void streamed_project2(const u8 *r, half *dest,
                                                                   const u8 *w, const int *im,
                                                                   int input, int rows) {
    __shared__ __align__(16) u8 a_page[2048];
    int warp = threadIdx.y, first = blockIdx.x * 16;
    for (int packet = warp * 32 + lane(); packet < 128; packet += 64) {
        int row = packet / 8, k = (packet % 8) * 16;
        int address = first + row < rows ? im[(first + row) * 128 + k] : -1;
        uint4 a = {0, 0, 0, 0};
        if (address >= 0)
            a = *(const uint4 *)(r + input + address);
        *(u32 *)(a_page + streamed::project_a(row, k)) = a.x;
        *(u32 *)(a_page + streamed::project_a(row, k + 4)) = a.y;
        *(u32 *)(a_page + streamed::project_a(row, k + 8)) = a.z;
        *(u32 *)(a_page + streamed::project_a(row, k + 12)) = a.w;
    }
    __syncthreads();
    uint2 c[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
#pragma unroll
    for (int kp = 0; kp < 4; kp++) {
        uint4 a = *(const uint4 *)(a_page + kp * 512 + lane() * 16);
#pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            uint4 b = *(const uint4 *)(w + kp * 2048 + warp * 1024 + pair * 512 + lane() * 16);
            physical::multiply(c[pair * 2], a, make_uint2(b.x, b.y));
            physical::multiply(c[pair * 2 + 1], a, make_uint2(b.z, b.w));
        }
    }
#pragma unroll
    for (int n = 0; n < 4; n++)
        for (int i = 0; i < 2; i++) {
            int row = first + lane() / 4 + i * 8, col = warp * 32 + n * 8 + (lane() & 3) * 2;
            if (row < rows)
                *(u32 *)(dest + row * 64 + col) = i ? c[n].y : c[n].x;
        }
}
struct TwoWeights {
    const u8 *expand, *reduce, *tail, *qkv, *projection;
    const half *bias, *ffn_gate, *attn_gate, *scale;
    const int *p, *ip, *pk;
};

static_assert(sizeof(TwoWeights) == 96, "Retain original TwoWeights ABI");

// Raw B128 offsets are relative to live p.r40 fields, not lane-B8 copies.
template <int M, int N>
__device__ __forceinline__ void raw_weight(HC<M, N> &c, const Words<M / 4> &a, const u8 *w) {
    each<N / 16>([&](auto pt) {
        constexpr int pair = decltype(pt)::value;
        uint4 b = *(const uint4 *)(w + pair * 512 + lane() * 16);
        each<M / 16>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            u32 aa[4] = {a.v[m * 4], a.v[m * 4 + 1], a.v[m * 4 + 2], a.v[m * 4 + 3]};
            each<2>([&](auto ht) {
                constexpr int n = pair * 2 + decltype(ht)::value;
                u32 bb[2];
                if constexpr (decltype(ht)::value == 0) {
                    bb[0] = b.x;
                    bb[1] = b.y;
                } else {
                    bb[0] = b.z;
                    bb[1] = b.w;
                }
                Fragment z{{c.v[(m * (N / 8) + n) * 2], c.v[(m * (N / 8) + n) * 2 + 1]}};
                mma8(z, aa, bb);
                c.v[(m * (N / 8) + n) * 2] = z.v[0];
                c.v[(m * (N / 8) + n) * 2 + 1] = z.v[1];
            });
        });
    });
}
__device__ __forceinline__ int aoffset(int row, int k) {
    return (k / 32) * 2048 + (row / 16) * 512 + ((row & 7) * 4 + (k & 15) / 4) * 16 +
           ((row & 8) ? 4 : 0) + ((k & 16) ? 8 : 0) + (k & 3);
}
extern "C" __global__ __launch_bounds__(64) void packet2_ds(u8 *r, TwoWeights w,
                                                            const Layout2 *meta, const int *unused,
                                                            int in, int out, int counter,
                                                            physical::Cross cross) {
    extern __shared__ __align__(16) u8 exchange[];
    Layout2 layout = *meta;
    int warp = threadIdx.y;
    for (int part = 0; part < 4; part++) {
        int l = (warp * 32 + lane()) * 16 + part * 1024;
        if (layout.seq == 7) {
            int t = l / 64, c = l % 64, a = input_address(layout, t, c);
            uint4 packet = *(const uint4 *)(r + in + a);
            *(u32 *)(exchange + label2(t, c)) = packet.x;
            *(u32 *)(exchange + label2(t, c + 4)) = packet.y;
            *(u32 *)(exchange + label2(t, c + 8)) = packet.z;
            *(u32 *)(exchange + label2(t, c + 12)) = packet.w;
        } else {
            int a = generic_address(layout, l);
            uint4 packet = {0, 0, 0, 0};
            if (a >= 0) {
                if (cross.up_inverse) {
                    packet = streamed::up_packet(r, cross, a, layout.h, layout.w);
                } else
                    packet = *(const uint4 *)(r + in + a);
            }
            *(uint4 *)(exchange + l) = packet;
        }
    }
    __syncthreads();
    // One M32 warp. Materialize its input directly as both K32 A packets,
    // and its skip as Half C. No intermediate dynamic input Tile survives.
    Words<16> input;
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            int t = warp * 32 + m * 16 + lane() / 4, k = kp * 32 + (lane() & 3) * 4;
            uint4 a = *(const uint4 *)(exchange + label2(t, k));
            input.v[kp * 8 + m * 4] = a.x;
            input.v[kp * 8 + m * 4 + 1] = a.y;
            input.v[kp * 8 + m * 4 + 2] = a.z;
            input.v[kp * 8 + m * 4 + 3] = a.w;
        });
    });
    auto ff = activation::fill<32, 64>([&](int t, int c) {
        return __hmul(une4(exchange[label2(warp * 32 + t, perm(c))]), w.ffn_gate[c]);
    });
// New equivalent: persistent M32 reduce C, only one N32 hidden slab live.
#pragma unroll 1
    for (int stream = 0; stream < 2; stream++) {
        auto reduced = activation::fill<32, 32>([](int, int) { return h(0); });
#pragma unroll 1
        for (int slab = 0; slab < 4; slab++) {
            auto hidden = activation::fill<32, 32>([](int, int) { return h(0); });
            each<2>([&](auto kt) {
                constexpr int kp = decltype(kt)::value;
                raw_weight<32, 32>(hidden, slice<kp * 8, 8>(input),
                                   w.expand + stream * 8192 + kp * 4096 + slab * 1024);
            });
            auto ha = pv_packet(two_phase::encode<32, 32>(hidden, packed12::Activate{}));
            raw_weight<32, 32>(reduced, ha, w.reduce + stream * 4096 + slab * 1024);
        }
        auto ra = pv_packet(two_phase::encode<32, 32>(reduced, packed12::Identity{}));
        raw_weight<32, 64>(ff, ra, w.tail + stream * 2048);
    }
    auto feature = feature_packet(two_phase::encode<32, 64>(ff, packed12::Identity{}));
    __syncthreads(); // both warps finish all input reads before FF replaces the page
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            *(uint4 *)(exchange + kp * 2048 + (warp * 2 + m) * 512 + lane() * 16) = {
                feature.v[kp * 8 + m * 4], feature.v[kp * 8 + m * 4 + 1],
                feature.v[kp * 8 + m * 4 + 2], feature.v[kp * 8 + m * 4 + 3]};
        });
    });
    __syncthreads(); // rowgroup producers hand all 64 rows to head consumers
    // Residual is the original E4 FF shared reread, NOT 1H Half FF residual.
    // Save its encoded bits, postponing Half gate multiplication to projection.
    Words<16> residual, q, k, v;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            u32 bits = 0;
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                int t = m * 16 + lane() / 4 + i * 8, c = warp * 32 + n * 8 + (lane() & 3) * 2;
                bits |= u32(*(unsigned short *)(exchange + aoffset(t, perm(c)))) << (16 * i);
            });
            residual.v[m * 4 + n] = bits;
        });
    });
    each<2>([&](auto st) {
        constexpr int first = decltype(st)::value * 2;
        auto z = activation::fill<32, 96>([](int, int) { return h(0); });
        each<2>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            Words<8> a;
            each<2>([&](auto mt) {
                constexpr int m = decltype(mt)::value;
                uint4 packet =
                    *(const uint4 *)(exchange + kp * 2048 + (first + m) * 512 + lane() * 16);
                a.v[m * 4] = packet.x;
                a.v[m * 4 + 1] = packet.y;
                a.v[m * 4 + 2] = packet.z;
                a.v[m * 4 + 3] = packet.w;
            });
            raw_weight<32, 96>(z, a, w.qkv + warp * 3072 + kp * 6144);
        });
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            HC<16, 96> row;
            each<24>([&](auto it) {
                constexpr int i = decltype(it)::value;
                row.v[i] = z.v[m * 24 + i];
            });
            half iq[2], ik[2];
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                iq[i] = two_phase::norm<i, 0>(row);
                ik[i] = two_phase::norm<i, 32>(row);
            });
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 qb = 0, kb = 0;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    qb |= packed12::e4pair(
                              packed12::mul(packed12::mul(row.v[n * 2 + i], packed12::splat(iq[i])),
                                            packed12::splat(w.scale[warp])))
                          << (16 * i);
                    kb |= packed12::e4pair(
                              packed12::mul(row.v[(n + 4) * 2 + i], packed12::splat(ik[i])))
                          << (16 * i);
                });
                q.v[(first + m) * 4 + n] = qb;
                k.v[(first + m) * 4 + n] = kb;
                v.v[(first + m) * 4 + n] =
                    packed12::e4four(row.v[(n + 8) * 2], row.v[(n + 8) * 2 + 1]);
            });
        });
    });
    __syncthreads(); // residual and all QKV reads finish before slab reuse
    auto qa = query_a(q);
    auto kb = key_b(k);
    auto vb = value_b(v);
    each<2>([&](auto st) {
        constexpr int slab = decltype(st)::value;
        HC<32, 64> logits;
        each<8>([&](auto pt) {
            constexpr int packet = decltype(pt)::value;
            const u8 *seed =
                w.qkv + 0x3000 + warp * 8192 + slab * 4096 + packet * 512 + lane() * 16;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(logits.v[packet * 4]), "=r"(logits.v[packet * 4 + 1]),
                           "=r"(logits.v[packet * 4 + 2]), "=r"(logits.v[packet * 4 + 3])
                         : "l"(seed));
        });
        two_phase::matrix<32, 64>(logits, slice<slab * 8, 8>(qa), kb);
        each<32>([&](auto it) {
            constexpr int i = decltype(it)::value;
            logits.v[i] = packed12::exponent_pair(logits.v[i]);
        });
        half inv[4];
        each<4>([&](auto it) {
            constexpr int i = decltype(it)::value;
            inv[i] = two_phase::den<i>(logits);
        });
        Words<16> prob;
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<8>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                u32 word = 0;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    word |= packed12::e4pair(packed12::mul(logits.v[(m * 8 + n) * 2 + i],
                                                           packed12::splat(inv[m * 2 + i])))
                            << (16 * i);
                });
                prob.v[m * 8 + n] = word;
            });
        });
        auto pa = prob_a(prob);
        auto a = activation::fill<32, 32>([](int, int) { return h(0); });
        each<2>([&](auto kt) {
            constexpr int kp = decltype(kt)::value;
            two_phase::matrix<32, 32>(a, slice<kp * 8, 8>(pa), slice<kp * 8, 8>(vb));
        });
        auto pv = pv_packet(two_phase::encode<32, 32>(a, packed12::Identity{}));
        each<2>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            *(uint4 *)(exchange + warp * 2048 + (slab * 2 + m) * 512 + lane() * 16) = {
                pv.v[m * 4], pv.v[m * 4 + 1], pv.v[m * 4 + 2], pv.v[m * 4 + 3]};
        });
    });
    __syncthreads();
    HC<64, 32> result;
    each<4>([&](auto mt) {
        constexpr int m = decltype(mt)::value;
        each<4>([&](auto nt) {
            constexpr int n = decltype(nt)::value;
            each<2>([&](auto it) {
                constexpr int i = decltype(it)::value;
                u32 bits = residual.v[m * 4 + n] >> (16 * i);
                int c = warp * 32 + n * 8 + (lane() & 3) * 2;
                result.v[(m * 4 + n) * 2 + i] =
                    packed12::mul(pack(une4(bits & 255), une4((bits >> 8) & 255)),
                                  pack(w.attn_gate[c], w.attn_gate[c + 1]));
            });
        });
    });
    each<2>([&](auto kt) {
        constexpr int kp = decltype(kt)::value;
        Words<16> a;
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            uint4 packet = *(const uint4 *)(exchange + kp * 2048 + m * 512 + lane() * 16);
            a.v[m * 4] = packet.x;
            a.v[m * 4 + 1] = packet.y;
            a.v[m * 4 + 2] = packet.z;
            a.v[m * 4 + 3] = packet.w;
        });
        raw_weight<64, 32>(result, a, w.projection + warp * 1024 + kp * 2048);
    });
    __syncthreads();
    if (cross.pool_input) {
        each<4>([&](auto mt) {
            constexpr int m = decltype(mt)::value;
            each<4>([&](auto nt) {
                constexpr int n = decltype(nt)::value;
                each<2>([&](auto it) {
                    constexpr int i = decltype(it)::value;
                    int t = m * 16 + lane() / 4 + i * 8, c = warp * 32 + n * 8 + (lane() & 3) * 2;
                    *(u32 *)((half *)(exchange + 4096) + t * 64 + c) =
                        result.v[(m * 4 + n) * 2 + i];
                });
            });
        });
    }
    auto encoded = two_phase::encode<64, 32>(result, packed12::Identity{});
    if (layout.seq == 150) {
        auto published = special150_packet(encoded);
        each<4>([&](auto pt) {
            constexpr int part = decltype(pt)::value;
            int src = rank150_source(part * 512 + lane() * 16), word = src / 128,
                ll = (src % 128) / 4, byte = src % 4;
            int t = word / 4 * 16 + ll / 4 + byte / 2 * 8,
                c = warp * 32 + word % 4 * 8 + (ll & 3) * 2 + (byte & 1);
            int address = output_address(layout, t, perm(c));
            if (address >= 0)
                *(uint4 *)(r + out + address) = {published.v[part * 4], published.v[part * 4 + 1],
                                                 published.v[part * 4 + 2],
                                                 published.v[part * 4 + 3]};
        });
    } else {
        auto published = output_packet(encoded);
        each<4>([&](auto pt) {
            constexpr int part = decltype(pt)::value;
            int address = generic_address(layout, warp * 512 + lane() * 16 + part * 1024);
            if (address >= 0)
                *(uint4 *)(r + out + address) = {published.v[part * 4], published.v[part * 4 + 1],
                                                 published.v[part * 4 + 2],
                                                 published.v[part * 4 + 3]};
        });
    }
    if (cross.pool_input) {
        __syncthreads();
        streamed::pool(r, (half *)(exchange + 4096), exchange, cross, layout.gx);
    }
    if (counter >= 0 && warp == 0 && lane() == 0)
        ((unsigned *)(r + counter))[blockIdx.x] = 0;
}

} // namespace nr_outer_two_4
