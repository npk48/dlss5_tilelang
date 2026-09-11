"""16H blocks and shared raw publication, merge and repack bridges.

The 797fd63 math, layouts, K32 order and launch geometry are retained.
"""
from pathlib import Path
from functools import lru_cache
import bootstrap
import torch
import tilelang
import tilelang.language as T
from fp8_toolchain import private_compile
from device_policy import TARGET, EXECUTION_BACKEND

from tilelang_nr.common.deep import (
    _CONFIG,
    _HEADER,
    _completion,
    _finish,
    _jit,
    _producer,
    _view,
    act,
    act2,
    add,
    add2,
    akoff,
    completion,
    dec,
    denominator,
    dup,
    enc,
    exponent,
    fma,
    fma2,
    hb,
    hh,
    issue,
    ld,
    loc,
    mul,
    mul2,
    normalize,
    perm,
    qp,
    rawbase,
    rawc,
    shfl,
    st,
    sxor,
    tok
)


@_jit
def _bridge(kind, sizes, count, splits, activate, width, poolwidth, flags, tails):
    (n0, v0), (n1, v1) = tails
    s0, s1, s2, s3, s4, s5 = sizes
    f0, f1, f2 = flags

    @T.prim_func
    def main(
        A: T.Tensor((s0, ), 'uint8'),
        B: T.Tensor((s1, ), 'uint8'),
        M: T.Tensor((s2, ), 'int32'),
        O: T.Tensor((s3, ), 'uint8'),
        P: T.Tensor((s4, ), 'int32'),
        G: T.Tensor((s5, ), 'float16'),
        AH: T.Tensor((s0 // 2, ), 'float16'),
        BH: T.Tensor((s1 // 2, ), 'float16'),
        OH: T.Tensor((s3 // 2, ), 'float16'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(T.ceildiv(count, 256), threads=256) as block:
            T.import_source(_HEADER)
            i = block * 256 + T.get_thread_binding()
            if i < count:
                if kind == 0:
                    O[T.if_then_else(f0 != 0, P[i], i)] = A[M[i]]
                elif kind == 1:
                    h = T.alloc_var('float16')
                    h = AH[i]
                    for p in T.serial(1, splits - 1):
                        h = add(h, AH[p * count + i])
                    OH[M[i]] = h
                elif kind == 2:
                    h = add(BH[M[i]], AH[(splits - 1) * count + i])
                    if f0:
                        OH[i] = h
                    if f1:
                        O[i] = enc(T.if_then_else(activate != 0, act(h), h))
                    if f2:
                        O[P[i]] = enc(T.if_then_else(activate != 0, act(h), h))
                else:
                    j = M[i]
                    p = j // 512
                    n = j % 512
                    h = fma(
                        dec(B[j]),
                        G[n],
                        AH[((p // width // 2) * poolwidth + (p % width // 2)) * 512 + n]
                    )
                    O[P[i]] = enc(h)
            completion(C0, C1, n0, v0, n1, v1, block == 0, 256)

    return main


@_jit
def _local(sizes, gx, gy, gz, hasout, tails):
    (n0, v0), (n1, v1) = tails
    sa, splan, sw, sraw, smem, sout = sizes

    @T.prim_func
    def main(
        A: T.Tensor((sa, ), 'uint8'),
        Plan: T.Tensor((splan, ), 'int32'),
        Weight: T.Tensor((sw, ), 'uint8'),
        Raw: T.Tensor((sraw, ), 'uint8'),
        Members: T.Tensor((smem, ), 'int32'),
        Out: T.Tensor((sout, ), 'uint8'),
        Scale: T.Tensor((sw // 4, ), 'float32'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(gx, gy, gz, threads=128) as (bx, by, bz):
            T.import_source(_HEADER)
            tid = T.get_thread_binding()
            l = tid % 32
            warp = tid // 32
            head = 4 * bz + warp
            group = by * gx + bx
            sh = T.alloc_shared((1024, ), 'uint32')
            qkv = T.alloc_shared((4, 3, 2048), 'uint8')
            packed = T.alloc_shared((4, 2048), 'uint8')
            c = T.alloc_local((4, 12, 2), 'uint32')
            av = T.alloc_local((4, ), 'uint32')
            for r, n, h in T.grid(4, 12, 2):
                c[r, n, h] = T.uint32(0)
            for p in T.serial(8):
                for u in T.serial(2):
                    j = warp + 4 * u
                    off = Plan[(((group * 8 + p) * 4 + j // 2) * 2 + j % 2) * 32 + l]
                    for z in T.serial(4):
                        sh[j * 128 + l * 4 +
                           z] = T.if_then_else(off >= 0, ld(A, off + 4 * z), T.uint32(0))
                T.evaluate(T.tvm_storage_sync('shared'))
                for k in T.serial(2):
                    for r in T.serial(4):
                        for z in T.serial(4):
                            av[z] = sh[(1024 * r + 512 * k + 16 * l) // 4 + z]
                        for n in T.serial(12):
                            off = 3072 * head + 49152 * (2 * p +
                                                         k) + 16 * l + (n // 2) * 512 + (n % 2) * 8
                            issue(c, r, n, av, ld(Weight, off), ld(Weight, off + 4))
                T.evaluate(T.tvm_storage_sync('shared'))
            for r, co in T.grid(4, 3):
                # Scale points at the original weight buffer; no rebased allocation.
                if co < 2:
                    for h in T.serial(2):
                        a = c[r, co * 4, h]
                        b = c[r, co * 4 + 1, h]
                        cc = c[r, co * 4 + 2, h]
                        d = c[r, co * 4 + 3, h]
                        s = add2(fma2(a, a, mul2(cc, cc)), fma2(b, b, mul2(d, d)))
                        s1 = add2(s, sxor(s, 2))
                        s2 = add2(s1, sxor(s1, 1))
                        den = T.max(add(hh(s2), hh(s2 >> 16)), T.float16(6.198883056640625e-5))
                        inv = T.cast(
                            T.call_extern('float32', 'nr_rsqrt', T.cast(den, 'float32')), 'float16'
                        )
                        for n in T.serial(4):
                            v = T.alloc_var('uint32')
                            v = mul2(c[r, co * 4 + n, h], dup(T.cast(inv, 'float32')))
                            if co == 0:
                                v = mul2(
                                    v,
                                    dup(
                                        T.cast(
                                            T.cast(Scale[0xe0000 // 4 + head], 'float16'),
                                            'float32'
                                        )
                                    )
                                )
                            c[r, co * 4 + n, h] = v
                    for h, n in T.grid(2, 2):
                        slot = 16 * r + tok(l // 4 + 8 * h)
                        value = T.if_then_else(
                            Members[group * 64 + slot] >= 0,
                            qp(c[r, co * 4 + 2 * n, h]) | (qp(c[r, co * 4 + 2 * n + 1, h]) << 16),
                            T.uint32(0)
                        )
                        T.evaluate(
                            T.call_extern(
                                'handle',
                                'nr_st32',
                                T.access_ptr(
                                    qkv[warp, co, slot * 32 + 16 * n + 4 * (l & 3)], 'w', 4
                                ),
                                value
                            )
                        )
                else:
                    vx = Members[group * 64 + 16 * r + tok(l // 4)] >= 0
                    vy = Members[group * 64 + 16 * r + tok(l // 4 + 8)] >= 0
                    for n in T.serial(4):
                        e = T.if_then_else(
                            vx, qp(c[r, 8 + n, 0]), T.uint32(0)
                        ) | (T.if_then_else(vy, qp(c[r, 8 + n, 1]), T.uint32(0)) << 16)
                        word = T.alloc_var('uint32')
                        word = T.uint32(0)
                        for b in T.serial(4):
                            slot = 4 * (l & 3) + ((b & 1) << 1) + ((b & 2) >> 1)
                            rr = ((slot & 1) << 3) + ((slot & 6) >> 1) + ((slot & 8) >> 1)
                            val = shfl(e, 4 * (rr & 7) + l // 8)
                            word = word | (
                                ((val >> (8 * (2 * (rr // 8) + ((l // 4) & 1)))) & 255) << (8 * b)
                            )
                        T.evaluate(
                            T.call_extern(
                                'handle',
                                'nr_st32',
                                T.access_ptr(qkv[warp, 2, r * 512 + n * 128 + l * 4], 'w', 4),
                                word
                            )
                        )
            T.evaluate(T.tvm_storage_sync('shared'))
            score = T.alloc_local((1, 8, 2), 'uint32')
            result = T.alloc_local((1, 4, 2), 'uint32')
            for r in T.serial(4):
                for n, h in T.grid(4, 2):
                    result[0, n, h] = T.uint32(0)
                for z in T.serial(4):
                    av[z] = T.call_extern(
                        'uint32',
                        'nr_ld32',
                        T.access_ptr(
                            qkv[warp,
                                0, (16 * r + 2 * (l // 4) + (z & 1)) * 32 + 16 * (z // 2) + 4 *
                                (l & 3)],
                            'r',
                            4
                        )
                    )
                for n in T.serial(8):
                    for h in T.serial(2):
                        score[0, n, h] = ld(
                            Weight,
                            0xc0000 + head * 8192 + r * 2048 + (n // 2) * 512 + l * 16 +
                            (n % 2) * 8 + h * 4
                        )
                    off = (16 * (n // 2) + 2 * (l // 4) + (n & 1)) * 32 + 4 * (l & 3)
                    b0 = T.call_extern('uint32', 'nr_ld32', T.access_ptr(qkv[warp, 1, off], 'r', 4))
                    b1 = T.call_extern(
                        'uint32', 'nr_ld32', T.access_ptr(qkv[warp, 1, off + 16], 'r', 4)
                    )
                    issue(score, 0, n, av, b0, b1)
                    for h in T.serial(2):
                        score[0, n, h] = exponent(score[0, n, h], False)
                for h in T.serial(2):
                    den = add(T.float16(0), denominator(score, 0, h, l))
                    inv = T.cast(
                        T.call_extern(
                            'float32',
                            'nr_rcp',
                            T.cast(T.max(den, T.float16(6.198883056640625e-5)), 'float32')
                        ),
                        'float16'
                    )
                    for n in T.serial(8):
                        score[0, n, h] = qp(mul2(score[0, n, h], dup(T.cast(inv, 'float32'))))
                for p in T.serial(2):
                    for z in T.serial(4):
                        n = 4 * p + 2 * (z // 2)
                        av[z] = (score[0, n, z & 1]
                                 & 65535) | ((score[0, n + 1, z & 1] & 65535) << 16)
                    for n in T.serial(4):
                        off = p * 1024 + n * 128 + l * 4
                        b0 = T.call_extern(
                            'uint32', 'nr_ld32', T.access_ptr(qkv[warp, 2, off], 'r', 4)
                        )
                        b1 = T.call_extern(
                            'uint32', 'nr_ld32', T.access_ptr(qkv[warp, 2, off + 512], 'r', 4)
                        )
                        issue(result, 0, n, av, b0, b1)
                for n, h in T.grid(4, 2):
                    slot = 16 * r + 2 * (l // 4) + h
                    col = n * 8 + 2 * (l & 3)
                    token = slot % 16
                    off = (col & 1) + ((col & 6) << 3) + ((col & 8) >> 2) + ((col & 16) >> 1) + (
                        (token & 1) << 2
                    ) + ((token & 14) << 5) + 512 * r
                    q = qp(result[0, n, h])
                    packed[warp, off] = T.cast(q, 'uint8')
                    packed[warp, off + 1] = T.cast(q >> 8, 'uint8')
            T.evaluate(T.tvm_storage_sync('shared'))
            for r in T.serial(4):
                off = Plan[((group * 8 * 4 + r) * 2) * 32 + l]
                if off >= 0:
                    for z in T.serial(4):
                        st(
                            Raw,
                            off + 512 * head + z * 4,
                            T.call_extern(
                                'uint32',
                                'nr_ld32',
                                T.access_ptr(packed[warp, 512 * r + 16 * l + 4 * z], 'r', 4)
                            )
                        )
                if hasout:
                    for n, h in T.grid(4, 2):
                        slot = 16 * r + l // 4 + 8 * h
                        idx = Members[group * 64 + slot]
                        col = n * 8 + 2 * (l & 3)
                        token = slot % 16
                        ix = (col & 1) + ((col & 6) << 3) + ((col & 8) >> 2) + ((col & 16) >> 1) + (
                            (token & 1) << 2
                        ) + ((token & 14) << 5) + 512 * r
                        if idx >= 0:
                            Out[(idx * 16 + head) * 32 + col] = packed[warp, ix]
                            Out[(idx * 16 + head) * 32 + col + 1] = packed[warp, ix + 1]
            completion(C0, C1, n0, v0, n1, v1, bx == 0 and by == 0 and bz == 0, 128)

    return main


@T.macro
def page16(sh, A, p, sx, sy, H, W, first, l, warp, warps):
    if first:
        for u in T.serial(8):
            j = 32 * warp + l + 128 * u
            yp = 8 * sx + ((j >> 4) & 3) + 4 * ((j >> 8) & 1)
            xp = 8 * sy + ((j >> 6) & 1) + 2 * (j & 1) + 4 * ((j >> 9) & 1)
            c16 = 4 * p + ((j >> 1) & 1) + 2 * ((j >> 7) & 1)
            sh[j] = T.if_then_else(
                yp < H and xp < W,
                ld(A, 16 * ((W * c16 + xp) * H + yp) + 4 * ((l >> 2) & 3)),
                T.uint32(0)
            )
    else:
        for u in T.serial(8 // warps):
            j = warp + u * warps
            t = j // 2
            k = j & 1
            for z in T.serial(4):
                sh[(512 * j + 16 * l) // 4 + z] = T.if_then_else(
                    2 * sy + t // 2 < W // 4 and 2 * sx + (t & 1) < H // 4,
                    ld(A, rawbase(sx, sy, t, p, H) + 512 * k + 16 * l + 4 * z),
                    T.uint32(0)
                )
    T.evaluate(T.tvm_storage_sync('shared'))


@_jit
def _sixteen(sizes, H, W, variant, decoder, flags, tails):
    (n0, v0), (n1, v1) = tails
    sa, sw, ss, sr, shalf, sq, spool, spr = sizes
    hashalf, hasquant, haspool = flags
    first = variant == -2
    ffn = variant < 0
    warps = 4 if first or variant != 1 and variant != -1 else 8
    tiles = 4 if warps == 4 else 2

    @T.prim_func
    def main(
        A: T.Tensor((sa, ), 'uint8'),
        Weight: T.Tensor((sw, ), 'uint8'),
        Skip: T.Tensor((ss, ), 'uint8'),
        Raw: T.Tensor((sr, ), 'uint8'),
        Half: T.Tensor((shalf, ), 'uint32'),
        Quant: T.Tensor((sq, ), 'uint8'),
        Pool: T.Tensor((spool, ), 'uint8'),
        PoolRaw: T.Tensor((spr, ), 'uint8'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(T.ceildiv(H, 8) if ffn else T.ceildiv(H, 8) * 2,
                      T.ceildiv(W, 8),
                      2 if ffn else 1,
                      threads=32 * warps) as (bx, sy, bz):
            T.import_source(_HEADER)
            tid = T.get_thread_binding()
            l = tid % 32
            warp = tid // 32
            sx = bx if ffn else bx % T.ceildiv(H, 8)
            g = 4 * bz + (warp & 3) if ffn else 4 * (bx // T.ceildiv(H, 8)) + (warp & 3)
            t0 = 0 if warps == 4 else 2 * (warp // 4)
            sh = T.alloc_shared((1024, ), 'uint32')
            route = T.alloc_shared((warps, tiles * 1536), 'uint8')
            packed = T.alloc_shared((warps, 4096), 'uint8')
            keep = T.alloc_shared((warps, 4096 if variant == 3 else 1), 'float16')
            c = T.alloc_local((tiles, 8, 2), 'uint32')
            av = T.alloc_local((4, ), 'uint32')
            for r, n, h in T.grid(tiles, 8, 2):
                c[r, n, h] = T.uint32(0)
                if not ffn:
                    row = l // 4 + 8 * h
                    col = n * 8 + 2 * (l & 3)
                    y = 8 * sx + 4 * ((t0 + r) & 1) + row // 4
                    x = 8 * sy + 4 * ((t0 + r) // 2) + row % 4
                    v = T.alloc_var('uint32')
                    v = T.uint32(0)
                    if y < H and x < W:
                        if variant == 0:
                            ch = g * 64 + col
                            pix = (x // 4 * 4 + y % 4) * H + (y // 4 * 4 + x % 4)
                            off = pix * 16 + (ch // 16) * (H * W * 16) + (ch & 1) + (
                                (ch & 6) << 1
                            ) + ((ch & 8) >> 2)
                        else:
                            off = rawbase(sx, sy, t0 + r, g, H) + loc(row, col)
                        v = (ld(Skip, off & ~3) >> ((off & 3) * 8)) & T.uint32(65535)
                    c[r, n, h] = mul2(
                        T.call_extern('uint32', 'nr_decode2', v),
                        ld(Weight, 0x40000 + 2 * (g * 64 + col))
                    )
            for p in T.serial(8):
                page16(sh, A, p, sx, sy, H, W, first, l, warp, warps)
                for k in T.serial(2):
                    for r in T.serial(tiles):
                        for z in T.serial(4):
                            av[z] = sh[(1024 * (t0 + r) + 512 * k + 16 * l) // 4 + z]
                        for n in T.serial(8):
                            off = 2048 * g + 16384 * (2 * p + k) + 16 * l + 512 * (n //
                                                                                   2) + (n % 2) * 8
                            issue(c, r, n, av, ld(Weight, off), ld(Weight, off + 4))
                T.evaluate(T.tvm_storage_sync('shared'))
            if ffn:
                for r, n, h in T.grid(tiles, 8, 2):
                    row = l // 4 + 8 * h
                    col = n * 8 + 2 * (l & 3)
                    k = (col & ~14) | ((col & 8) >> 2) | ((col & 2) << 1) | ((col & 4) << 1)
                    off = r * 1024 + 512 * (k // 32) + akoff(row, k & 31)
                    q = qp(c[r, n, h])
                    route[warp, off] = T.cast(q, 'uint8')
                    route[warp, off + 1] = T.cast(q >> 8, 'uint8')
                T.evaluate(T.tvm_storage_sync('shared'))
                total = T.alloc_local((tiles, 8, 2), 'uint32')
                hidden = T.alloc_local((tiles, 4, 2), 'uint32')
                for r, n, h in T.grid(tiles, 8, 2):
                    total[r, n, h] = T.uint32(0)
                for i in T.serial(8):
                    for r, n, h in T.grid(tiles, 4, 2):
                        hidden[r, n, h] = T.uint32(0)
                    for k in T.serial(2):
                        for r in T.serial(tiles):
                            for z in T.serial(4):
                                av[z] = T.call_extern(
                                    'uint32',
                                    'nr_ld32',
                                    T.access_ptr(
                                        route[warp, r * 1024 + 512 * k + 16 * l + z * 4], 'r', 4
                                    )
                                )
                            for n in T.serial(4):
                                off = 0x40000 + 0x4000 * g + 0x400 * i + 0x2000 * k + 16 * l + (
                                    n // 2
                                ) * 512 + (n % 2) * 8
                                issue(hidden, r, n, av, ld(Weight, off), ld(Weight, off + 4))
                    for r, n, h in T.grid(tiles, 4, 2):
                        v = n * 8 + 2 * (l & 3)
                        col = (v & 1) + ((v & 6) << 1) + ((v & 8) >> 2) + (v & 16)
                        off = tiles * 1024 + r * 512 + akoff(l // 4 + 8 * h, col)
                        q = qp(act2(hidden[r, n, h]))
                        route[warp, off] = T.cast(q, 'uint8')
                        route[warp, off + 1] = T.cast(q >> 8, 'uint8')
                    T.evaluate(T.tvm_storage_sync('shared'))
                    for r in T.serial(tiles):
                        for z in T.serial(4):
                            av[z] = T.call_extern(
                                'uint32',
                                'nr_ld32',
                                T.access_ptr(
                                    route[warp, tiles * 1024 + r * 512 + 16 * l + z * 4], 'r', 4
                                )
                            )
                        for n in T.serial(8):
                            off = 0x60000 + 0x4000 * g + 0x800 * i + 16 * l + (n // 2) * 512 + (
                                n % 2
                            ) * 8
                            issue(total, r, n, av, ld(Weight, off), ld(Weight, off + 4))
                    T.evaluate(T.tvm_storage_sync('shared'))
                for r, n, h in T.grid(tiles, 8, 2):
                    c[r, n, h] = total[r, n, h]
            for r, n, h in T.grid(tiles, 8, 2):
                row = l // 4 + 8 * h
                col = n * 8 + 2 * (l & 3)
                off = 1024 * r + loc(row, col)
                q = qp(c[r, n, h])
                packed[warp, off] = T.cast(q, 'uint8')
                packed[warp, off + 1] = T.cast(q >> 8, 'uint8')
                y = 8 * sx + 4 * ((t0 + r) & 1) + row // 4
                x = 8 * sy + 4 * ((t0 + r) // 2) + row % 4
                if hashalf or hasquant:
                    idx = 16 * ((H // 4) * (2 * sy + (t0 + r) // 2) + 2 * sx +
                                ((t0 + r) & 1)) + tok(row) if decoder else y * W + x
                    if y < H and x < W:
                        if hashalf:
                            Half[(idx * 512 + g * 64 + col) // 2] = c[r, n, h]
                        if hasquant:
                            Quant[idx * 512 + g * 64 + col] = T.cast(q, 'uint8')
                            Quant[idx * 512 + g * 64 + col + 1] = T.cast(q >> 8, 'uint8')
                if variant == 3:
                    keep[warp, (16 * r + row) * 64 + col] = hh(c[r, n, h])
                    keep[warp, (16 * r + row) * 64 + col + 1] = hh(c[r, n, h] >> 16)
            T.evaluate(T.tvm_storage_sync('shared'))
            if variant != 4:
                for r in T.serial(tiles):
                    if 2 * sy + (t0 + r) // 2 < W // 4 and 2 * sx + ((t0 + r) & 1) < H // 4:
                        for z in T.serial(8):
                            off = 16 * l + (z // 4) * 512 + (z % 4) * 4
                            st(
                                Raw,
                                rawbase(sx, sy, t0 + r, g, H) + off,
                                T.call_extern(
                                    'uint32',
                                    'nr_ld32',
                                    T.access_ptr(packed[warp, 1024 * r + off], 'r', 4)
                                )
                            )
            else:
                for aa, h, rowhalf in T.grid(4, 4, 2):
                    xp = 8 * sy + l // 16 + 2 * h
                    yp = 8 * sx + (l // 4) % 4 + 4 * rowhalf
                    t = rowhalf + 2 * ((xp % 8) // 4)
                    row = (xp % 4) * 4 + yp % 4
                    col = aa * 16 + 2 * (l % 4)
                    lo = T.cast(
                        packed[warp, 1024 * t + loc(row, col)], 'uint32'
                    ) | (T.cast(packed[warp, 1024 * t + loc(row, col) + 1], 'uint32') << 8)
                    hi = T.cast(
                        packed[warp, 1024 * t + loc(row, col + 8)], 'uint32'
                    ) | (T.cast(packed[warp, 1024 * t + loc(row, col + 8) + 1], 'uint32') << 8)
                    if xp < W and yp < H:
                        st(
                            Raw,
                            16 * ((W * (4 * g + aa) + xp) * H + yp) + 4 * (l % 4),
                            lo | (hi << 16)
                        )
            if variant == 3:
                for p in T.serial(16):
                    py = p // 4
                    px = p % 4
                    cc = 2 * l
                    y = 2 * py
                    x = 2 * px
                    t = y // 4 + 2 * (x // 4)
                    row = (y % 4) * 4 + x % 4
                    ix = (16 * t + row) * 64 + cc
                    token = ((px >> 1)
                             & 1) + ((px & 1) << 1) + (((py >> 1) & 1) << 2) + ((py & 1) << 3)
                    off = (cc & 1) + ((cc & 6) << 3) + ((cc & 8) >> 2) + ((cc & 16) >> 1) + (
                        (cc & 32) << 4
                    ) + ((token & 1) << 2) + ((token & 14) << 5)
                    for j in T.serial(2):
                        s = add(
                            add(keep[warp, ix + j], keep[warp, ix + 64 + j]),
                            add(keep[warp, ix + 256 + j], keep[warp, ix + 320 + j])
                        )
                        q = enc(mul(s, T.float16(.25)))
                        packed[warp, off + j] = q
                        if haspool:
                            Pool[((4 * sx + py) * (T.ceildiv(W, 8) * 4) + 4 * sy + px) * 512 +
                                 64 * g + cc + j] = q
                T.evaluate(T.tvm_storage_sync('shared'))
                for z in T.serial(8):
                    off = 16 * l + (z // 4) * 512 + (z % 4) * 4
                    st(
                        PoolRaw,
                        8192 * (T.ceildiv(H, 8) * sy + sx) + 1024 * g + off,
                        T.call_extern('uint32', 'nr_ld32', T.access_ptr(packed[warp, off], 'r', 4))
                    )
            completion(C0, C1, n0, v0, n1, v1, bx == 0 and sy == 0 and bz == 0, 32 * warps)

    return main


# These names are the exact native_nr deep launch ABI, not block-name guesses.
_BRIDGES = {
    'raw_repack_completion',
    'permute',
    'publish_split',
    'merge_phase',
    'merge_phase_completion',
    'up_exit_completion'
}
_SIXTEEN = {
    'first25_completion': -2,
    'chained_completion': -1,
    'special26_completion': 0,
    'project8_completion': 1,
    'project4_completion': 2,
    'pool56': 3,
    'out132': 4
}
_NAMES = _BRIDGES | set(_SIXTEEN) | {
    'matrix_phase',
    'matrix_expand_completion',
    'matrix_contract',
    'matrix_projection',
    'joint_qkv_completion',
    'joint_attention_completion',
    'joint_local64_completion'
}


def supports(name):
    return name in _NAMES


def _build_bridge(spec, fuse_completion):
    name = spec.name
    # Unused formals alias an existing view; they are compile-time dead, never null.
    base = _view(spec, 0)
    dummy = base
    im = None
    pm = None
    gain = None
    flags = (0, 0, 0)
    index = None
    splits = 1
    activate = 0
    width = poolwidth = 1
    if name in ('raw_repack_completion', 'permute'):
        kind = 0
        count = spec.scalar(4 if name.startswith('raw') else 3)
        out = _view(spec, 1)
        im = _view(spec, 2, torch.int32)
        hasmap = name.startswith('raw') and spec.scalar(3) != 0
        pm = _view(spec, 3, torch.int32) if hasmap else im
        flags = (int(hasmap), 0, 0)
        index = 5 if name.startswith('raw') else None
    elif name == 'publish_split':
        kind = 1
        count = spec.scalar(3)
        splits = spec.scalar(4)
        out = _view(spec, 1)
        im = _view(spec, 2, torch.int32)
    elif name.startswith('merge_phase'):
        kind = 2
        count = spec.scalar(4)
        splits = spec.scalar(5)
        activate = spec.scalar(7)
        dummy = _view(spec, 1)
        im = _view(spec, 2, torch.int32)
        outputs = [i for i in (3, 6, 8) if spec.scalar(i)]
        if len(outputs) != 1:
            raise NotImplementedError('deep merge requires native runtime single output rail')
        oi = outputs[0]
        out = _view(spec, oi)
        flags = (int(oi == 3), int(oi == 6), int(oi == 8))
        pm = _view(spec, 9, torch.int32) if oi == 8 else im
        index = 10 if name.endswith('_completion') else None
    else:
        kind = 3
        count = spec.scalar(6)
        width = spec.scalar(7)
        poolwidth = spec.scalar(8)
        dummy = _view(spec, 1)
        out = _view(spec, 3)
        im = _view(spec, 4, torch.int32)
        pm = _view(spec, 5, torch.int32)
        gain = _view(spec, 2, torch.float16)
        index = 9
    if pm is None:
        pm = im
    if gain is None:
        gain = base.view(torch.float16)
    args = (base, dummy, im, out, pm, gain)
    kernel = _producer(
        spec,
        _bridge,
        (kind, tuple(x.numel() for x in args), count, splits, activate, width, poolwidth, flags),
        index,
        fuse_completion
    )
    typed = tuple(x.view(torch.float16) for x in (base, dummy, out))
    return _finish(kernel, lambda: kernel(*args, *typed, stream=0))


def build_step(spec, *, fuse_completion=True):
    if not supports(spec.name):
        raise NotImplementedError(f'TileLang deep kernel not implemented: {spec.name}')
    if spec.name in _BRIDGES:
        return _build_bridge(spec, fuse_completion)
    name = spec.name
    defs = spec.defines
    if name in _SIXTEEN:
        variant = _SIXTEEN[name]
        ffn = variant < 0
        w = _view(spec, 1)
        raw = _view(spec, 3 if ffn else 5)
        skip = w if ffn or not spec.scalar(2) else _view(spec, 2)
        half = _view(spec, 3, torch.uint32) if not ffn and spec.scalar(3) else w.view(torch.uint32)
        quant = _view(spec, 2 if ffn else 4) if spec.scalar(2 if ffn else 4) else w
        pool = _view(spec, 6) if not ffn and spec.scalar(6) else w
        poolraw = _view(spec, 7) if not ffn and spec.scalar(7) else w
        input_size = defs['DH'] * defs['DW'] * 512 if not spec.scalar(0) else _view(spec, 0).numel()
        fixed = (w, skip, raw, half, quant, pool, poolraw)
        flags = (
            int(not ffn and bool(spec.scalar(3))),
            int(bool(spec.scalar(2 if ffn else 4))),
            int(not ffn and bool(spec.scalar(6)))
        )
        fixed_sizes = [x.numel() for x in fixed]
        if not ffn and not spec.scalar(2):
            fixed_sizes[1] = input_size  # special26 skip shares the not-yet-bound input pointer.
        index = (5 if ffn else 9) if name.endswith('_completion') else None
        kernel = _producer(
            spec,
            _sixteen,
            (
                (input_size, *fixed_sizes),
                defs['DH'],
                defs['DW'],
                variant,
                spec.scalar(4 if ffn else 8),
                flags
            ),
            index,
            fuse_completion
        )

        # special26's skip is the same mutable input ctypes object as first25.
        def run_sixteen():
            live_skip = skip if ffn else _view(spec, 2)[:fixed_sizes[1]]
            kernel(
                _view(spec, 0)[:input_size],
                w,
                live_skip,
                raw,
                half,
                quant,
                pool,
                poolraw,
                stream=0
            )

        return _finish(kernel, run_sixteen)
    if name == 'joint_local64_completion':
        a = _view(spec, 0)
        plan = _view(spec, 1, torch.int32)
        w = _view(spec, 2)
        raw = _view(spec, 4)
        members = _view(spec, 6, torch.int32)
        out = _view(spec, 3) if spec.scalar(3) else raw
        args = (a, plan, w, raw, members, out)
        kernel = _producer(
            spec,
            _local, (tuple(x.numel() for x in args), *spec.grid, bool(spec.scalar(3))),
            7,
            fuse_completion
        )
        scale = w.view(torch.float32)
        return _finish(kernel, lambda: kernel(*args, scale, stream=0))
    from tilelang_nr.legacy.vit import _matrix, _qkv, _attention
    if name == 'joint_qkv_completion':
        args = (
            _view(spec, 0),
            _view(spec, 1),
            _view(spec, 6, torch.uint32),
            _view(spec, 11),
            _view(spec, 12),
            _view(spec, 13),
            _view(spec, 14, torch.float32)
        )
        kernel = _producer(
            spec,
            _qkv, (tuple(x.numel() for x in args), defs['DT'], defs['DP']),
            21,
            fuse_completion
        )
        return _finish(kernel, lambda: kernel(*args, stream=0))
    if name == 'joint_attention_completion':
        args = tuple(_view(spec, i) for i in (0, 1, 2, 4))
        kernel = _producer(
            spec, _attention, (tuple(x.numel() for x in args), spec.scalar(7)), 14, fuse_completion
        )
        return _finish(kernel, lambda: kernel(*args, stream=0))
    transition = name == 'matrix_phase'
    if transition and any(spec.scalar(i) for i in (2, 3, 4, 13, 17, 18, 19, 20, 23)):
        raise NotImplementedError('transition optional ABI rails not used by native_nr runtime')
    a = _view(spec, 0)
    w = _view(spec, 1)
    skip = _view(spec, 2) if spec.scalar(2) else a
    part = _view(spec, 5, torch.uint32) if spec.scalar(5) else w.view(torch.uint32)
    raw = _view(spec, 15) if spec.scalar(15) else a
    map_ = _view(spec, 16, torch.int32) if spec.scalar(16) else w.view(torch.int32)
    packet = _view(spec, 21, torch.int32) if transition else w.view(torch.int32)
    route = _view(spec, 22, torch.uint16) if transition else w.view(torch.uint16)
    args = (a, w, skip, part, raw, map_, packet, route, w.view(torch.float16))
    kernel = _producer(
        spec,
        _matrix,
        (
            tuple(x.numel() for x in args),
            spec.scalar(6),
            spec.scalar(7),
            spec.scalar(8),
            spec.scalar(10),
            spec.scalar(9),
            transition,
            spec.scalar(14),
            defs['PH'],
            defs['PW'], (bool(spec.scalar(2)), bool(spec.scalar(15)))
        ),
        25 if name == 'matrix_expand_completion' else None,
        fuse_completion
    )
    return _finish(kernel, lambda: kernel(*args, stream=0))
