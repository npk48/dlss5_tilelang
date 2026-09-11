"""8H FFN, attention and physical DS/UP with corrected portrait ownership.

The 797fd63 math, layouts, K32 order and launch geometry are retained.
"""
from tilelang_nr.instructions import instruction_source
from pathlib import Path
import numpy as np
import torch
import tilelang
import tilelang.language as T
from fp8_toolchain import private_compile
from tilelang_nr.common.wide import (
    _JIT,
    _ROUTES,
    op,
    activate,
    exponent,
    permute,
    route_k,
    address48,
    shared_packed,
    local_route_word,
    _f_zero
)
from tilelang_nr.common.joint import _prmt, _PRIMITIVES
from .heads8_layout import pool_rows, up_rows

NAMES = frozenset(('full8_inpview', 'full8_chained', 'full8_outview', 'packet8_ds', 'packet8_up'))
HEADER = _PRIMITIVES + '\n' + instruction_source('eight_joint_primitives.cuh')


def supports(name):
    return name in NAMES


def mem(name, *args):
    return T.call_extern('uint32', 'nr_tl_eight_joint::' + name, *args)


@T.macro
def paired(C, a, b, m, component):
    for n in T.unroll(4):
        r = T.call_extern(
            'uint64',
            'wide_mma',
            a[0],
            a[1],
            a[2],
            a[3],
            b[n // 2, n % 2 * 2],
            b[n // 2, n % 2 * 2 + 1],
            C[m, component * 4 + n, 0],
            C[m, component * 4 + n, 1]
        )
        C[m, component * 4 + n, 0] = T.cast(r, 'uint32')
        C[m, component * 4 + n, 1] = T.cast(r >> 32, 'uint32')


@T.macro
def weighted(S, W, C, lane, first, M, components, ks, off, stride):
    a = T.alloc_local((4, 4), 'uint32')
    b = T.alloc_local((6, 4), 'uint32')
    for kp in T.serial(ks):
        for m in T.unroll(M):
            T.evaluate(
                mem(
                    'lds128',
                    T.address_of(a[m, 0]),
                    T.address_of(S[kp, (first + m) * 128 + lane * 4])
                )
            )
        for p in T.unroll(components * 2):
            T.evaluate(
                mem(
                    'ld128',
                    T.address_of(b[p, 0]),
                    T.address_of(W[(off + kp * stride) // 4 + p * 128 + lane * 4])
                )
            )
        for comp in T.unroll(components):
            for m in T.unroll(M):
                for n in T.unroll(4):
                    r = T.call_extern(
                        'uint64',
                        'wide_mma',
                        a[m, 0],
                        a[m, 1],
                        a[m, 2],
                        a[m, 3],
                        b[comp * 2 + n // 2, n % 2 * 2],
                        b[comp * 2 + n // 2, n % 2 * 2 + 1],
                        C[first + m, comp * 4 + n, 0],
                        C[first + m, comp * 4 + n, 1]
                    )
                    C[first + m, comp * 4 + n, 0] = T.cast(r, 'uint32')
                    C[first + m, comp * 4 + n, 1] = T.cast(r >> 32, 'uint32')


@T.macro
def hidden_tail(S, WX, WR, X, C, P, A, lane, off, reduceoff):
    a = T.alloc_local((4, ), 'uint32')
    eb = T.alloc_local((2, 4), 'uint32')
    rb = T.alloc_local((2, 4), 'uint32')
    for p in T.unroll(2):
        T.evaluate(
            mem(
                'ld128',
                T.address_of(eb[p, 0]),
                T.address_of(WX[(off + 7 * 4096) // 4 + p * 128 + lane * 4])
            )
        )
        T.evaluate(
            mem(
                'ld128',
                T.address_of(rb[p, 0]),
                T.address_of(WR[reduceoff // 4 + p * 128 + lane * 4])
            )
        )
    for m in T.unroll(4):
        T.evaluate(mem('lds128', T.address_of(a[0]), T.address_of(S[7, m * 128 + lane * 4])))
        paired(X, a, eb, m, 0)
        for n in T.unroll(4):
            P[m * 4 + n] = op('encode', activate(X[m, n, 0]), activate(X[m, n, 1]))
        for word in T.unroll(4):
            A[word] = local_route_word(P, m * 4 + word, lane, _ROUTES['hidden'])
        paired(C, A, rb, m, 0)


@T.macro
def initial(Residual, G, C, head, lane, off, first, M):
    for m in T.unroll(M):
        for n in T.unroll(4):
            g = G[off // 4 + head * 16 + n * 4 + lane % 4]
            C[first + m, n, 0] = op('mul', op('decode', Residual[(first + m) * 4 + n]), g)
            C[first + m, n, 1] = op('mul', op('decode', Residual[(first + m) * 4 + n] >> 16), g)


@T.macro
def collect_pool(C, Pool, Rows, tile, lane, first):
    vals = T.alloc_local((4, ), 'uint32')
    active = T.alloc_local((1, ), 'int32')
    for word in T.unroll(4):
        for pair in T.unroll(2):
            active[0] = 0
            for leaf in T.unroll(4):
                rr = T.alloc_var(
                    'int32', init=Rows[(tile * 16 + lane // 4 + (word % 2) * 8) * 4 + leaf]
                )[0]
                k = route_k(word // 2 * 16 + (lane % 4) * 4 + pair * 2)
                src = (rr & 7) * 4 + (k & 7) // 2
                vals[leaf] = 0
                # n is fixed for each word/pair; only four uniform M/half banks.
                for bank in T.unroll(4):
                    v = T.alloc_var(
                        'uint32',
                        init=op('shfl', C[first + bank // 2, word // 2 * 2 + pair, bank % 2], src)
                    )[0]
                    if (rr >= 0) & (rr // 32 == first // 2) & (((rr & 16) // 16 * 2 +
                                                                (rr & 8) // 8) == bank):
                        vals[leaf] = v
                if (rr >= 0) & (rr // 32 == first // 2):
                    active[0] = 1
            if active[0] != 0:
                avg = op(
                    'mul',
                    op('add', op('add', vals[0], vals[1]), op('add', vals[2], vals[3])),
                    op('splat', 0.25)
                )
                Pool[word] = Pool[word] | ((op('encode', avg, avg) & 65535) << (pair * 16))


@T.macro
def publish(C, P, A, R, tile, head, lane, first, height, width, gx, sx, sy, flags, out):
    for m in T.unroll(2):
        for n in T.unroll(4):
            P[m * 4 + n] = op('encode', C[first + m, n, 0], C[first + m, n, 1])
    if flags & 2:
        for word in T.unroll(8):
            A[word] = local_route_word(P, word, lane, _ROUTES['publish'])
        row = first * 16 + lane // 4 + (lane & 1) * 16 + ((lane >> 1) & 1) * 8
        for part in T.unroll(2):
            raw = T.alloc_var(
                'int32',
                init=address48(8, height, width, gx, sx, sy, 2, tile, row, head * 32 + part * 16)
            )[0]
            if (raw >= 0) & (raw <= R.shape[0] - out - 16):
                T.evaluate(
                    mem(
                        'st128',
                        T.address_of(R[out + raw]),
                        A[part * 4],
                        A[part * 4 + 1],
                        A[part * 4 + 2],
                        A[part * 4 + 3]
                    )
                )
    else:
        for parity in T.unroll(2):
            raw = T.alloc_var(
                'int32',
                init=address48(
                    8,
                    height,
                    width,
                    gx,
                    sx,
                    sy,
                    0,
                    tile,
                    first * 16 + lane // 4 + parity * 8,
                    head * 32 + (lane % 4) * 2
                )
            )[0]
            if (raw >= 0) & (raw <= R.shape[0] - out - 16):
                T.evaluate(
                    mem(
                        'st128',
                        T.address_of(R[out + raw]),
                        _prmt(P[0], P[1], 0x5410 if parity == 0 else 0x7632),
                        _prmt(P[4], P[5], 0x5410 if parity == 0 else 0x7632),
                        _prmt(P[2], P[3], 0x5410 if parity == 0 else 0x7632),
                        _prmt(P[6], P[7], 0x5410 if parity == 0 else 0x7632)
                    )
                )


@tilelang.jit(**dict(_JIT, compile_flags=['-lineinfo']))
def kernel8(
    tiles,
    height,
    width,
    gx,
    sx,
    sy,
    flags,
    rn,
    sn,
    wn,
    offsets,
    inp,
    out,
    counter,
    up,
    prn,
    gn,
    skip,
    ds,
    drn,
    pon,
    mn,
    poolout,
    ugn,
    uln,
    planar_rows
):
    ro, to, fg, qo, bo, sc, pro, ag = offsets
    heads = 8

    @T.prim_func
    def kernel(
        R: T.Tensor((rn, ), 'uint8'),
        Source: T.Tensor((sn, ), 'uint8'),
        W: T.Tensor((wn, ), 'int32'),
        Proj: T.Tensor((prn, ), 'int32'),
        Gate: T.Tensor((gn, ), 'int32'),
        Rows: T.Tensor((drn, ), 'int32'),
        PO: T.Tensor((pon, ), 'int32'),
        Matrix: T.Tensor((mn, ), 'int32'),
        UG: T.Tensor((ugn, ), 'int32'),
        UL: T.Tensor((uln, ), 'int32')
    ):
        with T.Kernel(tiles, threads=256) as tile:
            T.import_source(HEADER)
            tid = T.get_thread_binding(0)
            lane = tid % 32
            head = tid // 32
            S = T.alloc_shared((8, 512), 'int32')
            X = T.alloc_local((4, 4, 2), 'uint32')
            C = T.alloc_local((4, 4, 2), 'uint32')
            Z = T.alloc_local((4, 12, 2), 'uint32')
            P = T.alloc_local((16, ), 'uint32')
            A = T.alloc_local((16, ), 'uint32')
            Residual = T.alloc_local((16, ), 'uint32')
            Q = T.alloc_local((16, ), 'uint32')
            K = T.alloc_local((16, ), 'uint32')
            V = T.alloc_local((16, ), 'uint32')
            L = T.alloc_local((2, 8, 2), 'uint32')
            temp = T.alloc_local((4, ), 'uint32')
            acc = T.alloc_local((2, ), 'uint32')
            Pool = T.alloc_local((4, ), 'uint32')
            packetsA = T.alloc_local((4, 4), 'uint32')
            packetsB = T.alloc_local((2, 4), 'uint32')
            U = T.alloc_local((4, 2), 'uint32')
            for word in T.unroll(4):
                Pool[word] = 0
            if up:
                # Clean CUDA planar uint4 -> shared projection A map.
                for part in T.serial(2):
                    packet = tid + part * 256
                    local = packet // 32
                    group = packet % 32
                    row = UG[tile * 16 + local]
                    for j in T.unroll(4):
                        temp[j] = 0
                    inputbase = T.alloc_var(
                        'int32', init=inp + row * 16 + group * planar_rows * 16
                    )[0]
                    if (row >= 0) & (inputbase >= 0) & (inputbase <= rn - 16):
                        T.evaluate(mem('ld128', T.address_of(temp[0]), T.address_of(R[inputbase])))
                    base = (group // 2) * 128 + (local % 8) * 16 + local // 8 + (group % 2) * 2
                    for j in T.unroll(4):
                        S[(base + j * 4) // 512, (base + j * 4) % 512] = T.cast(temp[j], 'int32')
                T.sync_threads()
                for n in T.unroll(4):
                    U[n, 0] = 0
                    U[n, 1] = 0
                for kp in T.serial(16):
                    T.evaluate(
                        mem(
                            'lds128',
                            T.address_of(temp[0]),
                            T.address_of(
                                S[(kp * 128 + lane * 4) // 512, (kp * 128 + lane * 4) % 512]
                            )
                        )
                    )
                    for p in T.unroll(2):
                        T.evaluate(
                            mem(
                                'ld128',
                                T.address_of(packetsB[p, 0]),
                                T.address_of(
                                    W[0x58000 // 4 + kp * 2048 + head * 256 + p * 128 + lane * 4]
                                )
                            )
                        )
                    for n in T.unroll(4):
                        r = T.call_extern(
                            'uint64',
                            'wide_mma',
                            temp[0],
                            temp[1],
                            temp[2],
                            temp[3],
                            packetsB[n // 2, n % 2 * 2],
                            packetsB[n // 2, n % 2 * 2 + 1],
                            U[n, 0],
                            U[n, 1]
                        )
                        U[n, 0] = T.cast(r, 'uint32')
                        U[n, 1] = T.cast(r >> 32, 'uint32')
                for n in T.unroll(4):
                    for j in T.unroll(2):
                        row = UG[tile * 16 + lane // 4 + j * 8]
                        if row >= 0:
                            Proj[(row * 256 + head * 32 + n * 8 + (lane % 4) * 2) // 2
                                 ] = T.cast(U[n, j], 'int32')
                T.sync_threads()
                for m in T.unroll(4):
                    for word in T.unroll(4):
                        token = m * 16 + lane // 4 + (word % 2) * 8
                        rr = T.alloc_var('int32', init=UL[tile * 64 + token])[0]
                        temp[word] = 0
                        for pair in T.unroll(2):
                            col = route_k(word // 2 * 16 + (lane % 4) * 4 + pair * 2)
                            src = (rr & 7) * 4 + (col & 7) // 2
                            lo = T.alloc_var(
                                'uint32', init=op('shfl', U[word // 2 * 2 + pair, 0], src)
                            )[0]
                            hi = T.alloc_var(
                                'uint32', init=op('shfl', U[word // 2 * 2 + pair, 1], src)
                            )[0]
                            value = T.if_then_else((rr & 8) != 0, hi, lo)
                            raw = T.alloc_var(
                                'int32',
                                init=address48(
                                    8, height, width, gx, sx, sy, 0, tile, token, head * 32 + col
                                )
                            )[0]
                            skipbase = T.alloc_var('int32', init=skip + (raw // 4) * 4)[0]
                            if (raw >= 0) & (rr >= 0) & (skipbase >= 0) & (skipbase <= rn - 4):
                                skipword = mem('ld32', T.address_of(R[skipbase]))
                                skiphalf = op('decode', skipword >> ((raw % 4) * 8))
                                gate = Gate[(head * 32 + col) // 2]
                                quant = op(
                                    'encode', op('fma', skiphalf, gate, value), T.uint32(0)
                                ) & 65535
                                temp[word] = temp[word] | (quant << (pair * 16))
                    T.evaluate(
                        mem(
                            'sts128',
                            T.address_of(S[head, m * 128 + lane * 4]),
                            temp[0],
                            temp[1],
                            temp[2],
                            temp[3]
                        )
                    )
            elif flags & 1:
                # inpview stays in its special K4 domain.
                for m in T.unroll(4):
                    for word in T.unroll(4):
                        row = m * 16 + lane // 4 + (word % 2) * 8
                        col = route_k(head * 32 + (word // 2) * 16 + (lane % 4) * 4)
                        raw = T.alloc_var(
                            'int32',
                            init=address48(8, height, width, gx, sx, sy, 1, tile, row, col)
                        )[0]
                        temp[word] = 0
                        if (raw >= 0) & (raw <= sn - inp - 4):
                            temp[word] = mem('ld32', T.address_of(Source[inp + raw]))
                    T.evaluate(
                        mem(
                            'sts128',
                            T.address_of(S[head, m * 128 + lane * 4]),
                            temp[0],
                            temp[1],
                            temp[2],
                            temp[3]
                        )
                    )
            else:
                for slab in T.unroll(2):
                    for parity in T.unroll(2):
                        raw = T.alloc_var(
                            'int32',
                            init=address48(
                                8,
                                height,
                                width,
                                gx,
                                sx,
                                sy,
                                0,
                                tile,
                                slab * 32 + lane // 4 + parity * 8,
                                route_k(head * 32 + (lane % 4) * 4)
                            )
                        )[0]
                        for j in T.unroll(4):
                            temp[j] = 0
                        if (raw >= 0) & (raw <= sn - inp - 16):
                            T.evaluate(
                                mem(
                                    'ld128', T.address_of(temp[0]), T.address_of(Source[inp + raw])
                                )
                            )
                        for j in T.unroll(4):
                            S[head, (slab * 2 + j % 2) * 128 + lane * 4 + parity +
                              (j // 2) * 2] = T.cast(temp[j], 'int32')
            T.sync_threads()
            for word in T.unroll(16):
                Residual[word] = shared_packed(S, head, word, lane)
            _f_zero(C, 4)
            for slab in T.serial(4):
                _f_zero(X, 4)
                weighted(S, W, X, lane, 0, 4, 1, 7, head * 32768 + slab * 1024, 4096)
                hidden_tail(
                    S,
                    W,
                    W,
                    X,
                    C,
                    P,
                    A,
                    lane,
                    head * 32768 + slab * 1024,
                    ro + head * 4096 + slab * 1024
                )
            for m in T.unroll(4):
                for n in T.unroll(4):
                    P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
            T.sync_threads()
            for word in T.unroll(16):
                S[head, (word // 4) * 128 + lane * 4 +
                  word % 4] = T.cast(local_route_word(P, word, lane, _ROUTES['hidden']), 'int32')
            T.sync_threads()
            initial(Residual, W, C, head, lane, fg, 0, 4)
            weighted(S, W, C, lane, 0, 4, 1, 8, to + head * 1024, 8192)
            for m in T.unroll(4):
                for n in T.unroll(4):
                    P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
            T.sync_threads()
            for word in T.unroll(16):
                S[head, (word // 4) * 128 + lane * 4 +
                  word % 4] = T.cast(local_route_word(P, word, lane, _ROUTES['hidden']), 'int32')
            T.sync_threads()
            _f_zero(Z, 12)
            weighted(S, W, Z, lane, 0, 4, 3, 7, qo + head * 3072, 24576)
            for m in T.unroll(4):
                T.evaluate(
                    mem(
                        'lds128',
                        T.address_of(packetsA[m, 0]),
                        T.address_of(S[7, m * 128 + lane * 4])
                    )
                )
            for comp in T.unroll(3):
                for p in T.unroll(2):
                    T.evaluate(
                        mem(
                            'ld128',
                            T.address_of(packetsB[p, 0]),
                            T.address_of(
                                W[(qo + head * 3072 + 7 * 24576) // 4 + comp * 256 + p * 128 +
                                  lane * 4]
                            )
                        )
                    )
                for m in T.unroll(4):
                    for n in T.unroll(4):
                        r = T.call_extern(
                            'uint64',
                            'wide_mma',
                            packetsA[m, 0],
                            packetsA[m, 1],
                            packetsA[m, 2],
                            packetsA[m, 3],
                            packetsB[n // 2, n % 2 * 2],
                            packetsB[n // 2, n % 2 * 2 + 1],
                            Z[m, comp * 4 + n, 0],
                            Z[m, comp * 4 + n, 1]
                        )
                        Z[m, comp * 4 + n, 0] = T.cast(r, 'uint32')
                        Z[m, comp * 4 + n, 1] = T.cast(r >> 32, 'uint32')
                    if comp < 2:
                        for j in T.unroll(2):
                            a = Z[m, comp * 4, j]
                            b = Z[m, comp * 4 + 1, j]
                            c = Z[m, comp * 4 + 2, j]
                            d = Z[m, comp * 4 + 3, j]
                            s0 = op(
                                'add',
                                op('fma', a, a, op('mul', c, c)),
                                op('fma', b, b, op('mul', d, d))
                            )
                            s1 = op('add', s0, op('shfl', s0, lane ^ 2))
                            s2 = op('add', s1, op('shfl', s1, lane ^ 1))
                            total = op(
                                'max',
                                op('add', s2, (s2 >> 16) | (s2 << 16)),
                                op('splat', 0.00006198883056640625)
                            )
                            inv = op('rsqrt', total)
                            for n in T.unroll(4):
                                Z[m, comp * 4 + n, j] = op('mul', Z[m, comp * 4 + n, j], inv)
                                if comp == 0:
                                    if heads == 2:
                                        sraw = T.reinterpret(W[sc // 4 + head // 2], 'uint32')
                                        sbits = (sraw >> ((head % 2) * 16)) & 65535
                                        Z[m, n, j] = op('mul', Z[m, n, j], sbits | (sbits << 16))
                                    else:
                                        Z[m, n, j] = op(
                                            'mul',
                                            Z[m, n, j],
                                            op(
                                                'splat',
                                                T.reinterpret(W[sc // 4 + head], 'float32')
                                            )
                                        )
                    for n in T.unroll(4):
                        P[m * 4 + n] = op('encode', Z[m, comp * 4 + n, 0], Z[m, comp * 4 + n, 1])
                for word in T.unroll(16):
                    if comp == 0:
                        Q[word] = local_route_word(P, word, lane, _ROUTES['q' + str(heads)])
                    elif comp == 1:
                        K[word] = local_route_word(P, word, lane, _ROUTES['k' + str(heads)])
                    else:
                        V[word] = local_route_word(P, word, lane, _ROUTES['v' + str(heads)])
            T.sync_threads()
            for slab in T.unroll(2):
                for m in T.unroll(2):
                    for n in T.unroll(8):
                        if heads == 2:
                            wb = (
                                bo + head * 8192 + (slab * 2 + m) * 2048 +
                                (n // 2) * 512 + lane * 16 + (n % 2) * 8
                            ) // 4
                        else:
                            packet = (slab * 2 + m) * 4 + (n % 2) + (n // 4) * 2
                            wb = (
                                bo + head * 8192 + packet * 512 + lane * 16 + (n % 4) // 2 * 8
                            ) // 4
                        r = T.call_extern(
                            'uint64',
                            'wide_mma',
                            Q[(slab * 2 + m) * 4],
                            Q[(slab * 2 + m) * 4 + 1],
                            Q[(slab * 2 + m) * 4 + 2],
                            Q[(slab * 2 + m) * 4 + 3],
                            K[n * 2],
                            K[n * 2 + 1],
                            W[wb],
                            W[wb + 1]
                        )
                        L[m, n, 0] = exponent(T.cast(r, 'uint32'))
                        L[m, n, 1] = exponent(T.cast(r >> 32, 'uint32'))
                    for j in T.unroll(2):
                        if heads == 2:
                            g0 = op('add', L[m, 0, j], L[m, 1, j])
                            g1 = op('add', g0, op('add', L[m, 2, j], L[m, 3, j]))
                            g2 = op('add', g1, op('add', L[m, 4, j], L[m, 5, j]))
                            group = op('add', g2, op('add', L[m, 6, j], L[m, 7, j]))
                        else:
                            g0 = op('add', L[m, 0, j], L[m, 2, j])
                            g1 = op('add', g0, op('add', L[m, 1, j], L[m, 3, j]))
                            g2 = op('add', g1, op('add', L[m, 4, j], L[m, 6, j]))
                            group = op('add', g2, op('add', L[m, 5, j], L[m, 7, j]))
                        t0 = op('shfl', group, lane & ~3)
                        t1 = op('add', t0, op('shfl', group, (lane & ~3) + 1))
                        t2 = op('add', t1, op('shfl', group, (lane & ~3) + 2))
                        t3 = op('add', t2, op('shfl', group, (lane & ~3) + 3))
                        total = op(
                            'max',
                            op('add', t3, (t3 >> 16) | (t3 << 16)),
                            op('splat', 0.00006198883056640625)
                        )
                        inv = op('rcp', total)
                        for n in T.unroll(8):
                            L[m, n, j] = op('mul', L[m, n, j], inv)
                    for n in T.unroll(8):
                        P[m * 8 + n] = op('encode', L[m, n, 0], L[m, n, 1])
                for word in T.unroll(16):
                    A[word] = local_route_word(P, word, lane, _ROUTES['p' + str(heads)])
                for m in T.unroll(2):
                    for n in T.unroll(4):
                        acc[0] = 0
                        acc[1] = 0
                        for kp in T.unroll(2):
                            r = T.call_extern(
                                'uint64',
                                'wide_mma',
                                A[kp * 8 + m * 4],
                                A[kp * 8 + m * 4 + 1],
                                A[kp * 8 + m * 4 + 2],
                                A[kp * 8 + m * 4 + 3],
                                V[kp * 8 + n * 2],
                                V[kp * 8 + n * 2 + 1],
                                acc[0],
                                acc[1]
                            )
                            acc[0] = T.cast(r, 'uint32')
                            acc[1] = T.cast(r >> 32, 'uint32')
                        P[m * 4 + n] = op('encode', acc[0], acc[1])
                # Read FF after attention, before any warp overwrites this half-page.
                for word in T.unroll(8):
                    Residual[slab * 8 + word] = shared_packed(S, head, slab * 8 + word, lane)
                T.sync_threads()
                for word in T.unroll(8):
                    S[head, (slab * 2 + word // 4) * 128 + lane * 4 + word %
                      4] = T.cast(local_route_word(P, word, lane, _ROUTES['hidden']), 'int32')
                T.sync_threads()
                if ds or up:
                    initial(Residual, W, C, head, lane, ag, slab * 2, 2)
                    weighted(S, W, C, lane, slab * 2, 2, 1, 8, pro + head * 1024, 8192)
                    if ds:
                        collect_pool(C, Pool, Rows, tile, lane, slab * 2)
                    publish(
                        C,
                        P,
                        A,
                        R,
                        tile,
                        head,
                        lane,
                        slab * 2,
                        height,
                        width,
                        gx,
                        sx,
                        sy,
                        flags,
                        out
                    )
                    T.sync_threads()
            if not ds and not up:
                # Ordinary 8H retains the native M64 final projection.
                initial(Residual, W, C, head, lane, ag, 0, 4)
                weighted(S, W, C, lane, 0, 4, 1, 8, pro + head * 1024, 8192)
                for slab in T.unroll(2):
                    publish(
                        C,
                        P,
                        A,
                        R,
                        tile,
                        head,
                        lane,
                        slab * 2,
                        height,
                        width,
                        gx,
                        sx,
                        sy,
                        flags,
                        out
                    )
                T.sync_threads()
            if ds:
                T.evaluate(
                    mem(
                        'sts128',
                        T.address_of(
                            S[(head * 128 + lane * 4) // 512, (head * 128 + lane * 4) % 512]
                        ),
                        Pool[0],
                        Pool[1],
                        Pool[2],
                        Pool[3]
                    )
                )
                T.sync_threads()
                for group in T.serial(2):
                    for n in T.unroll(4):
                        U[n, 0] = 0
                        U[n, 1] = 0
                    for kp in T.serial(8):
                        T.evaluate(
                            mem(
                                'lds128',
                                T.address_of(temp[0]),
                                T.address_of(
                                    S[(kp * 128 + lane * 4) // 512, (kp * 128 + lane * 4) % 512]
                                )
                            )
                        )
                        for p in T.unroll(2):
                            T.evaluate(
                                mem(
                                    'ld128',
                                    T.address_of(packetsB[p, 0]),
                                    T.address_of(
                                        Matrix[kp * 4096 + (group * 8 + head) * 256 + p * 128 +
                                               lane * 4]
                                    )
                                )
                            )
                        for n in T.unroll(4):
                            r = T.call_extern(
                                'uint64',
                                'wide_mma',
                                temp[0],
                                temp[1],
                                temp[2],
                                temp[3],
                                packetsB[n // 2, n % 2 * 2],
                                packetsB[n // 2, n % 2 * 2 + 1],
                                U[n, 0],
                                U[n, 1]
                            )
                            U[n, 0] = T.cast(r, 'uint32')
                            U[n, 1] = T.cast(r >> 32, 'uint32')
                    # Pool A occupies [0,4096); both output groups [4096,12288).
                    # No reader/overwrite overlap while the second group is live.
                    for n in T.unroll(4):
                        encoded = op('encode', U[n, 0], U[n, 1])
                        for j in T.unroll(2):
                            row = lane // 4 + j * 8
                            col = (group * 8 + head) * 32 + n * 8 + (lane % 4) * 2
                            T.evaluate(
                                mem(
                                    'sts16',
                                    T.address_of(S[0, 0]),
                                    4096 + row * 512 + col,
                                    encoded >> (j * 16)
                                )
                            )
                T.sync_threads()
                for part in T.serial(2):
                    packet = tid + part * 256
                    row = packet // 32
                    group = packet % 32
                    base = 1024 + row * 128 + group * 4
                    T.evaluate(
                        mem(
                            'lds128',
                            T.address_of(packetsA[0, 0]),
                            T.address_of(S[base // 512, base % 512])
                        )
                    )
                    for word in T.unroll(4):
                        temp[word] = 0
                        for byte in T.unroll(4):
                            raw = word * 4 + byte
                            c = (raw & 1) | ((raw & 12) >> 1) | ((raw & 2) << 2)
                            temp[word] = temp[word] | (
                                ((packetsA[0, c // 4] >> ((c % 4) * 8)) & 255) << (byte * 8)
                            )
                    dest = PO[(tile * 16 + row) * 512 + group * 16]
                    if (dest >= 0) & (dest <= rn - poolout - 16):
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(R[poolout + dest]),
                                temp[0],
                                temp[1],
                                temp[2],
                                temp[3]
                            )
                        )
            T.sync_threads()
            T.evaluate(T.call_extern('int32', '__threadfence'))
            if counter >= 0:
                if tid == 0:
                    T.evaluate(mem('st32', T.address_of(R[counter + tile * 4]), T.uint32(0)))

    return kernel


def build_step(spec):
    if not supports(spec.name):
        raise NotImplementedError(spec.name)
    from native_nr import layouts
    height, width, gx, sx, sy, flags = (spec.scalar(2, k) for k in ('height', 'width', 'grid_x',
                                                                    'shift_x', 'shift_y', 'flags'))
    if tuple(spec.block) != (32, 8, 1) or spec.grid[1:] != (
            1, 1) or height % 4 or width % 4 or sx not in (0, -4) or sy not in (0, -4):
        raise NotImplementedError('Unexpected EightJoint geometry')
    up = spec.name == 'packet8_up'
    ds = spec.name == 'packet8_ds'
    r, source, w = spec.tensor(0), spec.tensor(7), spec.tensor(1, dtype=torch.int32)
    inp, out, counter = (spec.scalar(i) for i in (3, 4, 5))
    offsets = (0x40000, 0x48000, 0x58010, 0x58220, 0x88220, 0x98220, 0x98240, 0xa8240)
    proj = gate = rows = po = matrix = ug = ul = w
    owned = []
    skip = poolout = planar_rows = 0

    def device(a):
        t = torch.from_numpy(a).to(device=r.device).reshape(-1)
        owned.append(t)
        return t

    if ds:
        pi, po, matrix = (spec.tensor(8, k, torch.int32)
                          for k in ('pool_input', 'pool_output', 'matrix'))
        rows = device(pool_rows(pi.cpu().numpy()))
        poolout = spec.scalar(8, 'pool_out')
    if up:
        inv, proj, gate = (spec.tensor(8, k, torch.int32)
                           for k in ('up_inverse', 'projection', 'gate'))
        ui = spec.tensor(9, dtype=torch.int32)
        launch = dict(
            shape=(height, width),
            heads=8,
            grid=(gx, spec.grid[0] // gx, 1),
            shift=(sx, sy),
            seq=133
        )
        im, _ = layouts.wide_maps(launch)
        compact, physical, local = up_rows(ui.cpu().numpy(),
                                           inv.cpu().numpy(), im,
                                           proj.numel() * 2)
        ug, ul = device(physical), device(local)
        planar_rows = proj.numel() // 128
        skip = spec.scalar(8, 'skip')
        offsets = (*offsets[:2], 0x78000, *(v + 0x201e0 for v in offsets[3:]))
    with private_compile():
        fn = kernel8(
            spec.grid[0],
            height,
            width,
            gx,
            sx,
            sy,
            flags,
            r.numel(),
            source.numel(),
            w.numel(),
            offsets,
            inp,
            out,
            counter,
            up,
            proj.numel(),
            gate.numel(),
            skip,
            ds,
            rows.numel(),
            po.numel(),
            matrix.numel(),
            poolout,
            ug.numel(),
            ul.numel(),
            planar_rows
        )
    args = (r, source, w, proj, gate, rows, po, matrix, ug, ul)

    def launch():
        fn(*args)

    launch.kernel = fn
    launch.kernels = (fn, )
    launch.kernel_launches = launch.tilelang_launches = 1
    launch.workspace_bytes = sum(t.numel() * t.element_size() for t in owned)
    launch.scratch_bytes = 0
    launch.shared_bytes = 16384
    launch.organization = 'eight-joint-prefix-tail-M64-ordinary-M32-transition'
    return launch


from tilelang_nr.runtime import EightJointTileLangNR
