"""ViT shared-page matrix, QKV, attention and 57/100 physical bridges.

The 797fd63 math, layouts, K32 order and launch geometry are retained.
"""
from tilelang_nr.instructions import instruction_source
from pathlib import Path
from collections import Counter
import torch
import tilelang.language as T
from .common import deep
from .common.deep import (
    _jit,
    _producer,
    _finish,
    _view,
    completion,
    rawc,
    add,
    add2,
    mul2,
    act2,
    hb,
    hh,
    dup,
    qp,
    perm,
    shfl,
    sxor,
    fma2,
    st
)

_HEADER = deep._HEADER + '\n' + instruction_source('vit_joint_primitives.cuh')
NAMES = frozenset(
    (
        'matrix_expand_completion',
        'matrix_contract',
        'matrix_projection',
        'joint_qkv_completion',
        'joint_attention_completion',
        'matrix_phase'
    )
)


def supports(name):
    return name in NAMES


@_jit
def _attention(sizes, rows, tails):
    (n0, v0), (n1, v1) = tails
    sq, sk, sv, so = sizes

    @T.prim_func
    def main(
        Q: T.Tensor((sq, ), 'uint8'),
        K: T.Tensor((sk, ), 'uint8'),
        V: T.Tensor((sv, ), 'uint8'),
        O: T.Tensor((so, ), 'uint8'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(32, T.ceildiv(rows, 256), threads=128) as (head, block):
            T.import_source(_HEADER)
            tid = T.get_thread_binding()
            l = tid % 32
            w = tid // 32
            sh = T.alloc_shared((8192, ), 'uint8')
            query = T.alloc_local((4, 4), 'uint32')
            result = T.alloc_local((4, 4, 2), 'uint32')
            den = T.alloc_local((4, 2), 'float16')
            score = T.alloc_local((4, 8, 2), 'uint32')
            av = T.alloc_local((4, ), 'uint32')
            bv = T.alloc_local((2, 4), 'uint32')
            packet = T.alloc_local((4, ), 'uint32')
            for r in T.unroll(4):
                g = block * 16 + w * 4 + r
                for z in T.unroll(4):
                    query[r, z] = T.uint32(0)
                if g < rows // 16:
                    T.evaluate(
                        mem(
                            'global4',
                            T.address_of(query[r, 0]),
                            T.address_of(Q[g * 16384 + head * 512 + l * 16])
                        )
                    )
                for n in T.unroll(4):
                    for h in T.unroll(2):
                        result[r, n, h] = T.uint32(0)
                for h in T.unroll(2):
                    den[r, h] = T.float16(0)
            stage_kv(sh, K, V, 0, 0, head, rows, tid)
            for phase in T.serial(T.ceildiv(rows, 64)):
                page = phase % 2
                T.evaluate(mem('wait0'))
                T.evaluate(T.tvm_storage_sync('shared'))
                if phase + 1 < T.ceildiv(rows, 64):
                    stage_kv(sh, K, V, 1 - page, phase + 1, head, rows, tid)
                for r in T.unroll(4):
                    for n in T.unroll(8):
                        for h in T.unroll(2):
                            score[r, n, h] = T.uint32(0)
                for pair in T.unroll(4):
                    T.evaluate(
                        mem(
                            'shared4',
                            T.address_of(bv[0, 0]),
                            T.address_of(sh[page * 4096 + pair * 512 + l * 16])
                        )
                    )
                    for r in T.unroll(4):
                        for z in T.unroll(4):
                            av[z] = query[r, z]
                        for n in T.unroll(2):
                            v = T.call_extern(
                                'uint64',
                                'nr_tl_vit_joint::mma',
                                av[0],
                                av[1],
                                av[2],
                                av[3],
                                bv[0, n * 2],
                                bv[0, n * 2 + 1],
                                score[r, pair * 2 + n, 0],
                                score[r, pair * 2 + n, 1]
                            )
                            score[r, pair * 2 + n, 0] = T.cast(v, 'uint32')
                            score[r, pair * 2 + n, 1] = T.cast(v >> 32, 'uint32')
                for r in T.unroll(4):
                    for n in T.unroll(8):
                        for h in T.unroll(2):
                            score[r, n, h] = exponent(score[r, n, h], True)
                    for h in T.unroll(2):
                        den[r, h] = add(den[r, h], denominator(score, r, h, l))
                    for kp in T.unroll(2):
                        for z in T.unroll(4):
                            n = kp * 4 + 2 * (z // 2)
                            a = qp(score[r, n, 0]) | (qp(score[r, n, 1]) << 16)
                            b = qp(score[r, n + 1, 0]) | (qp(score[r, n + 1, 1]) << 16)
                            av[z] = perm(a, b, T.if_then_else(z % 2 != 0, 0x7632, 0x5410))
                        for n in T.unroll(2):
                            T.evaluate(
                                mem(
                                    'shared4',
                                    T.address_of(bv[n, 0]),
                                    T.address_of(
                                        sh[page * 4096 + 2048 + kp * 1024 + n * 512 + l * 16]
                                    )
                                )
                            )
                        row(result, r, av, bv, 1)
                T.evaluate(T.tvm_storage_sync('shared'))
            for r in T.unroll(4):
                for h in T.unroll(2):
                    d = add(den[r, h], T.float16(-(T.ceildiv(rows, 64) * 64 - rows) * .083984375))
                    inv = T.cast(
                        T.call_extern(
                            'float32',
                            'nr_rcp',
                            T.cast(T.max(d, T.float16(6.198883056640625e-5)), 'float32')
                        ),
                        'float16'
                    )
                    for n in T.unroll(4):
                        result[r, n, h] = mul2(result[r, n, h], dup(T.cast(inv, 'float32')))
                g = block * 16 + w * 4 + r
                if g < rows // 16:
                    for n in T.unroll(2):
                        for h in T.unroll(2):
                            packet[n * 2 + h] = qp(result[r, 2 * n, h]
                                                   ) | (qp(result[r, 2 * n + 1, h]) << 16)
                    T.evaluate(
                        mem(
                            'store4',
                            T.address_of(O[g * 16384 + head * 512 + l * 16]),
                            packet[0],
                            packet[1],
                            packet[2],
                            packet[3]
                        )
                    )
            completion(C0, C1, n0, v0, n1, v1, head == 0 and block == 0, 128)

    return main


def mem(name, *args):
    return T.call_extern('handle', 'nr_tl_vit_joint::' + name, *args)


@T.macro
def issue(C, r, n, A, B, p):
    v = T.call_extern(
        'uint64',
        'nr_tl_vit_joint::mma',
        A[0],
        A[1],
        A[2],
        A[3],
        B[p, 0],
        B[p, 1],
        C[r, n, 0],
        C[r, n, 1]
    )
    C[r, n, 0] = T.cast(v, 'uint32')
    C[r, n, 1] = T.cast(v >> 32, 'uint32')


@T.macro
def row(C, r, A, B, components):
    for n in T.unroll(components * 4):
        v = T.call_extern(
            'uint64',
            'nr_tl_vit_joint::mma',
            A[0],
            A[1],
            A[2],
            A[3],
            B[n // 2, (n % 2) * 2],
            B[n // 2, (n % 2) * 2 + 1],
            C[r, n, 0],
            C[r, n, 1]
        )
        C[r, n, 0] = T.cast(v, 'uint32')
        C[r, n, 1] = T.cast(v >> 32, 'uint32')


@T.macro
def stage_matrix(Sh, A, W, Pack, mt, nt, base, page, rows, K, N, PK, PAGE, transition, tid):
    lane = tid % 32
    warp = tid // 32
    if transition:
        for j in T.unroll(8):
            i = tid + j * 128
            off = Pack[(mt * (K // 64) + base // 64) * 1024 + i]
            T.evaluate(
                mem(
                    'cp',
                    T.address_of(Sh[page * PAGE + i * 16]),
                    T.address_of(A[0]),
                    off,
                    T.cast(off >= 0, 'int32')
                )
            )
    else:
        for h in T.unroll(2):
            for st in T.unroll(PK // 32):
                tile = warp + h * 4
                off = (mt * 8 + tile) * K * 16 + (base // 32 + st) * 512 + lane * 16
                T.evaluate(
                    mem(
                        'cp',
                        T.address_of(Sh[page * PAGE + tile * PK * 16 + st * 512 + lane * 16]),
                        T.address_of(A[0]),
                        off,
                        T.cast((mt * 8 + tile) * 16 < rows, 'int32')
                    )
                )
    for j in T.unroll((128 * PK) // (16 * 128)):
        i = tid + j * 128
        T.evaluate(
            mem(
                'cp',
                T.address_of(Sh[page * PAGE + (16384 if transition else 128 * PK) + i * 16]),
                T.address_of(W[0]),
                (base // 32 + i // 256) * 32 * N + nt * 128 * 32 + (i % 256) * 16,
                1
            )
        )
    T.evaluate(mem('commit'))


@_jit
def _matrix(sizes, rows, K, N, splits, mode, transition, activate, PH, PW, flags, tails):
    (n0, v0), (n1, v1) = tails
    sa, sw, ss, sp, sr, sm, spk, srt, sg = sizes
    has_skip, has_raw = flags
    PK = 32 if not transition and K == 1024 and splits == 4 else 64
    PAGES = 3 if not transition and splits == 1 else 2
    APAGE = 16384 if transition else 128 * PK
    PAGE = APAGE + 128 * PK
    PHASES = K // splits // PK

    @T.prim_func
    def main(
        A: T.Tensor((sa, ), 'uint8'),
        W: T.Tensor((sw, ), 'uint8'),
        S: T.Tensor((ss, ), 'uint8'),
        P: T.Tensor((sp, ), 'uint32'),
        R: T.Tensor((sr, ), 'uint8'),
        M: T.Tensor((sm, ), 'int32'),
        Pack: T.Tensor((spk, ), 'int32'),
        Route: T.Tensor((srt, ), 'uint16'),
        G: T.Tensor((sg, ), 'float16'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(T.ceildiv(rows, 128), N // 128, splits, threads=128) as (mt, nt, split):
            T.import_source(_HEADER)
            tid = T.get_thread_binding()
            l = tid % 32
            warp = tid // 32
            wm = warp // 2
            wn = warp % 2
            sh = T.alloc_shared((PAGES * PAGE, ), 'uint8')
            c = T.alloc_local((4, 8, 2), 'uint32')
            av = T.alloc_local((4, ), 'uint32')
            bv = T.alloc_local((4, 4), 'uint32')
            res = T.alloc_local((4, ), 'uint32')
            for r in T.unroll(4):
                for n in T.unroll(8):
                    for h in T.unroll(2):
                        c[r, n, h] = T.uint32(0)
            if has_skip:
                if split == 0:
                    for r in T.unroll(4):
                        for g in T.unroll(2):
                            prow = mt * 128 + wm * 64 + r * 16 + l // 4
                            col = nt * 128 + wn * 64 + g * 32 + (l & 3) * 2
                            T.evaluate(
                                mem(
                                    'global4',
                                    T.address_of(res[0]),
                                    T.address_of(S[rawc(prow, col, N)])
                                )
                            )
                            for j in T.unroll(4):
                                gate = hb(G[K * N // 2 + col + j * 8]
                                          ) | (hb(G[K * N // 2 + col + j * 8 + 1]) << 16)
                                for h in T.unroll(2):
                                    bits = (res[(j // 2) * 2 + h] >>
                                            ((j % 2) * 16)) & T.uint32(65535)
                                    c[r, g * 4 + j,
                                      h] = mul2(T.call_extern('uint32', 'nr_decode2', bits), gate)
            stage_matrix(
                sh,
                A,
                W,
                Pack,
                mt,
                nt,
                split * (K // splits),
                0,
                rows,
                K,
                N,
                PK,
                PAGE,
                transition,
                tid
            )
            if PAGES == 3:
                stage_matrix(
                    sh,
                    A,
                    W,
                    Pack,
                    mt,
                    nt,
                    split * (K // splits) + PK,
                    1,
                    rows,
                    K,
                    N,
                    PK,
                    PAGE,
                    transition,
                    tid
                )
            for phase in T.serial(PHASES):
                page = phase % PAGES
                base = split * (K // splits) + phase * PK
                if PAGES == 3:
                    if phase + 1 < PHASES:
                        T.evaluate(mem('wait1'))
                    else:
                        T.evaluate(mem('wait0'))
                else:
                    T.evaluate(mem('wait0'))
                T.evaluate(T.tvm_storage_sync('shared'))
                if PAGES == 3:
                    if phase + 2 < PHASES:
                        stage_matrix(
                            sh,
                            A,
                            W,
                            Pack,
                            mt,
                            nt,
                            base + 2 * PK, (phase + 2) % PAGES,
                            rows,
                            K,
                            N,
                            PK,
                            PAGE,
                            transition,
                            tid
                        )
                else:
                    if phase + 1 < PHASES:
                        stage_matrix(
                            sh,
                            A,
                            W,
                            Pack,
                            mt,
                            nt,
                            base + PK, (phase + 1) % PAGES,
                            rows,
                            K,
                            N,
                            PK,
                            PAGE,
                            transition,
                            tid
                        )
                for st in T.unroll(PK // 32):
                    for n in T.unroll(4):
                        T.evaluate(
                            mem(
                                'shared4',
                                T.address_of(bv[n, 0]),
                                T.address_of(
                                    sh[page * PAGE + APAGE + st * 4096 + wn * 2048 + n * 512 +
                                       l * 16]
                                )
                            )
                        )
                    for r in T.unroll(4):
                        if transition:
                            for z in T.unroll(4):
                                m = wm * 64 + r * 16 + l // 4 + (z % 2) * 8
                                ar = (mt * (K // 64) + base // 64) * 8192
                                av[z] = T.uint32(0)
                                for b in T.unroll(4):
                                    if mode == 0:
                                        j = m * 64 + st * 32 + (z // 2) * 16 + (l & 3) * 4 + b
                                    else:
                                        j = m * 64 + st * 32 + (z // 2) * 16 + (l & 3) * 2 + (
                                            b % 2
                                        ) + (b // 2) * 8
                                    av[z] = av[z] | (
                                        T.cast(
                                            sh[page * PAGE + T.cast(Route[ar + j], 'int32')],
                                            'uint32'
                                        ) << (b * 8)
                                    )
                        else:
                            T.evaluate(
                                mem(
                                    'shared4',
                                    T.address_of(av[0]),
                                    T.address_of(
                                        sh[page * PAGE + (wm * 4 + r) * PK * 16 + st * 512 + l * 16]
                                    )
                                )
                            )
                        row(c, r, av, bv, 2)
                T.evaluate(T.tvm_storage_sync('shared'))
            T.evaluate(mem('wait0'))
            T.evaluate(T.tvm_storage_sync('shared'))
            if not transition and splits == 1:
                # All async groups drained; dead A pages become disjoint warp raw64 slabs.
                for r in T.unroll(4):
                    for n in T.unroll(8):
                        for h in T.unroll(2):
                            q = qp(act2(c[r, n, h]))
                            T.evaluate(
                                mem(
                                    'store2',
                                    T.address_of(
                                        sh[warp * 4096 +
                                           rawc(r * 16 + l // 4 + h * 8, n * 8 + (l & 3) * 2, 64)]
                                    ),
                                    q
                                )
                            )
                T.evaluate(T.tvm_storage_sync('shared'))
                for t in T.unroll(8):
                    off = t * 512 + l * 16
                    T.evaluate(
                        mem('shared4', T.address_of(res[0]), T.address_of(sh[warp * 4096 + off]))
                    )
                    dst = rawc(
                        mt * 128 + wm * 64 + (off // 1024) * 16, nt * 128 + wn * 64, N
                    ) + off % 1024
                    T.evaluate(mem('store4', T.address_of(R[dst]), res[0], res[1], res[2], res[3]))
            else:
                for r in T.unroll(4):
                    for n in T.unroll(8):
                        for h in T.unroll(2):
                            prow = mt * 128 + wm * 64 + r * 16 + l // 4 + h * 8
                            col = nt * 128 + wn * 64 + n * 8 + (l & 3) * 2
                            if prow < rows:
                                if splits > 1:
                                    if transition:
                                        logical = prow
                                    else:
                                        logical = (((prow % PH) & ~1) | ((prow // PH) & 1)) * PW + (
                                            ((prow // PH) & ~1) | ((prow % PH) & 1)
                                        )
                                    P[(split * rows * N + logical * N + col) // 2] = c[r, n, h]
                                else:
                                    q = qp(
                                        T.if_then_else(activate != 0, act2(c[r, n, h]), c[r, n, h])
                                    )
                                    R[M[prow * N + col]] = T.cast(q, 'uint8')
                                    R[M[prow * N + col + 1]] = T.cast(q >> 8, 'uint8')
            completion(C0, C1, n0, v0, n1, v1, mt == 0 and nt == 0 and split == 0, 128)

    return main


@T.macro
def stage_qkv(Sh, A, W, page, base, mt, headpair, tid):
    lane = tid % 32
    warp = tid // 32
    for h in T.unroll(2):
        T.evaluate(
            mem(
                'cp',
                T.address_of(Sh[page * 10240 + (warp + h * 4) * 512 + lane * 16]),
                T.address_of(A[0]),
                (mt * 8 + warp + h * 4) * 16384 + (base // 32) * 512 + lane * 16,
                1
            )
        )
    for j in T.unroll(3):
        x = tid + j * 128
        T.evaluate(
            mem(
                'cp',
                T.address_of(Sh[page * 10240 + 4096 + x * 16]),
                T.address_of(W[0]),
                0x80 + (base // 32) * 0x18000 + (headpair + x // 192) * 0xc00 + (x % 192) * 16,
                1
            )
        )
    T.evaluate(mem('commit'))


@T.macro
def stage_kv(Sh, K, V, page, phase, head, rows, tid):
    lane = tid % 32
    warp = tid // 32
    T.evaluate(
        mem(
            'cp',
            T.address_of(Sh[page * 4096 + warp * 512 + lane * 16]),
            T.address_of(K[0]), ((phase * 4 + warp) * 32 + head) * 512 + lane * 16,
            T.cast((phase * 4 + warp) * 16 < rows, 'int32')
        )
    )
    T.evaluate(
        mem(
            'cp',
            T.address_of(Sh[page * 4096 + 2048 + warp * 512 + lane * 16]),
            T.address_of(V[0]),
            phase * 65536 + (warp // 2) * 32768 + head * 1024 + (warp % 2) * 512 + lane * 16,
            1
        )
    )
    T.evaluate(mem('commit'))


@T.macro
def normalize(c, r, co, head, scale, vit):
    if co < 2:
        for h in T.unroll(2):
            a = c[r, co * 4, h]
            b = c[r, co * 4 + 1, h]
            cc = c[r, co * 4 + 2, h]
            d = c[r, co * 4 + 3, h]
            s = fma2(a, a, mul2(cc, cc))
            v = fma2(b, b, mul2(d, d))
            s0 = add2(s, v)
            s1 = add2(s0, sxor(s0, 2))
            s2 = add2(s1, sxor(s1, 1))
            den = T.max(add(hh(s2), hh(s2 >> 16)), T.float16(6.198883056640625e-5))
            inv = T.cast(T.call_extern('float32', 'nr_rsqrt', T.cast(den, 'float32')), 'float16')
            factor = dup(T.cast(inv, 'float32'))
            for n in T.unroll(4):
                z = T.alloc_var('uint32')
                z = mul2(c[r, co * 4 + n, h], factor)
                if co == 0:
                    if vit:
                        z = mul2(z, dup(5.65625))
                    z = mul2(z, dup(T.cast(T.cast(scale[head], 'float16'), 'float32')))
                c[r, co * 4 + n, h] = z


@T.macro
def exponent(a, vit):
    z = T.alloc_var('uint32')
    if vit:
        z = fma2(a, dup(.08953857421875), dup(1.708984375))
        z = T.call_extern(
            'uint32',
            'nr_min2',
            T.call_extern('uint32', 'nr_max2', z, dup(1.439453125)),
            dup(1.9775390625)
        )
        z = ((z & T.uint32(0x0fff0fff)) << 4) - T.uint32(0xc000c000)
    else:
        z = fma2(a, dup(.044921875), dup(1.30078125))
        z = T.call_extern(
            'uint32',
            'nr_min2',
            T.call_extern('uint32', 'nr_max2', z, dup(1.03125)),
            dup(1.5693359375)
        )
        z = ((z & T.uint32(0x07ff07ff)) << 5) ^ T.uint32(0x80008000)
    return z


@T.macro
def denominator(c, r, h, l):
    s = T.alloc_var('uint32')
    s = add2(c[r, 0, h], c[r, 1, h])
    for n in T.unroll(1, 4):
        s = add2(s, add2(c[r, 2 * n, h], c[r, 2 * n + 1, h]))
    total = T.alloc_var('uint32')
    total = shfl(s, l & ~3)
    for n in T.unroll(1, 4):
        total = add2(total, shfl(s, (l & ~3) + n))
    return add(hh(total), hh(total >> 16))


@_jit
def _qkv(sizes, DT, DP, tails):
    (n0, v0), (n1, v1) = tails
    sa, sw, sraw, sq, sk, sv, ss = sizes

    @T.prim_func
    def main(
        A: T.Tensor((sa, ), 'uint8'),
        W: T.Tensor((sw, ), 'uint8'),
        Raw: T.Tensor((sraw, ), 'uint32'),
        Q: T.Tensor((sq, ), 'uint8'),
        K: T.Tensor((sk, ), 'uint8'),
        V: T.Tensor((sv, ), 'uint8'),
        Scale: T.Tensor((ss, ), 'float32'),
        C0: T.Tensor((max(n0, 1), ), 'int32'),
        C1: T.Tensor((max(n1, 1), ), 'int32')
    ):
        with T.Kernel(T.ceildiv(DT, 128) * 16, threads=128) as block:
            T.import_source(_HEADER)
            tid = T.get_thread_binding()
            l = tid % 32
            wm = (tid // 32) // 2
            head = (block // T.ceildiv(DT, 128)) * 2 + (tid // 32) % 2
            mt = block % T.ceildiv(DT, 128)
            sh = T.alloc_shared((20480, ), 'uint8')
            c = T.alloc_local((4, 12, 2), 'uint32')
            av = T.alloc_local((4, ), 'uint32')
            bv = T.alloc_local((6, 4), 'uint32')
            rv = T.alloc_local((4, ), 'uint32')
            for split in T.serial(2):
                for r in T.unroll(4):
                    for n in T.unroll(12):
                        for h in T.unroll(2):
                            c[r, n, h] = T.uint32(0)
                stage_qkv(sh, A, W, 0, split * 512, mt, (block // T.ceildiv(DT, 128)) * 2, tid)
                for step in T.serial(16):
                    base = split * 512 + step * 32
                    page = step % 2
                    T.evaluate(mem('wait0'))
                    T.evaluate(T.tvm_storage_sync('shared'))
                    if step + 1 < 16:
                        stage_qkv(
                            sh,
                            A,
                            W,
                            1 - page,
                            base + 32,
                            mt, (block // T.ceildiv(DT, 128)) * 2,
                            tid
                        )
                    for n in T.unroll(6):
                        T.evaluate(
                            mem(
                                'shared4',
                                T.address_of(bv[n, 0]),
                                T.address_of(
                                    sh[page * 10240 + 4096 + (tid // 32 % 2) * 3072 + n * 512 +
                                       l * 16]
                                )
                            )
                        )
                    for r in T.unroll(4):
                        T.evaluate(
                            mem(
                                'shared4',
                                T.address_of(av[0]),
                                T.address_of(sh[page * 10240 + (wm * 4 + r) * 512 + l * 16])
                            )
                        )
                        row(c, r, av, bv, 3)
                    T.evaluate(T.tvm_storage_sync('shared'))
                for r in T.unroll(4):
                    for co in T.unroll(3):
                        g = mt * 8 + wm * 4 + r
                        for pair in T.unroll(2):
                            idx = (
                                co * DP * 1024 + g * 16384 + head * 512 + l * 8 + pair * 256
                            ) // 2
                            if split == 0:
                                T.evaluate(
                                    mem(
                                        'store4',
                                        T.address_of(Raw[idx]),
                                        c[r, co * 4 + pair * 2, 0],
                                        c[r, co * 4 + pair * 2, 1],
                                        c[r, co * 4 + pair * 2 + 1, 0],
                                        c[r, co * 4 + pair * 2 + 1, 1]
                                    )
                                )
                            else:
                                T.evaluate(
                                    mem('global4', T.address_of(rv[0]), T.address_of(Raw[idx]))
                                )
                                for j in T.unroll(4):
                                    c[r, co * 4 + pair * 2 + j // 2,
                                      j % 2] = add2(rv[j], c[r, co * 4 + pair * 2 + j // 2, j % 2])
                        if split == 1:
                            normalize(c, r, co, head, Scale, True)
                            if co < 2:
                                for n in T.unroll(2):
                                    for h in T.unroll(2):
                                        word = qp(c[r, co * 4 + 2 * n, h]
                                                  ) | (qp(c[r, co * 4 + 2 * n + 1, h]) << 16)
                                        if co == 0:
                                            st(
                                                Q,
                                                g * 16384 + head * 512 + l * 16 + n * 8 + h * 4,
                                                word
                                            )
                                        else:
                                            st(
                                                K,
                                                g * 16384 + head * 512 + l * 16 + h * 8 + n * 4,
                                                word
                                            )
                            else:
                                src = l & ~4
                                j = (l >> 2) & 1
                                off = (g // 2) * 32768 + (g % 2) * 4 + head * 1024 + (
                                    src // 8
                                ) * 16 + (l & 3) * 128 + j * 64
                                for n in T.unroll(4):
                                    e = qp(c[r, 8 + n, 0]) | (qp(c[r, 8 + n, 1]) << 16)
                                    a = shfl(e, src)
                                    b = shfl(e, src | 4)
                                    st(
                                        V,
                                        off + (n & 1) * 8 + (n // 2) * 512,
                                        perm(a, b, T.if_then_else(j != 0, 0x7351, 0x6240))
                                    )
                T.evaluate(T.tvm_storage_sync('shared'))
            completion(C0, C1, n0, v0, n1, v1, block == 0, 128)

    return main


def build_step(spec):
    if not supports(spec.name):
        raise NotImplementedError(spec.name)
    name = spec.name
    defs = spec.defines
    fuse_completion = True
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


from tilelang_nr.runtime import VitJointTileLangNR
