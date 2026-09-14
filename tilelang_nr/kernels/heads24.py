"""2H/4H bounded-packet FFN, attention, projection and DS/UP.

The 797fd63 math, layouts, K32 order and launch geometry are retained.
"""
from tilelang_nr.instructions import instruction_source
import torch
import tilelang
from tilelang_nr.common.runtime_jit import spatial_jit
import tilelang.language as T
from runtime.fp8_compiler import private_compile
from tilelang_nr.common import wide as qualified
from tilelang_nr.common.joint import (
    _JIT,
    _PRIMITIVES,
    _ROUTES,
    supports,
    joint_address2,
    route,
    weighted,
    finish_row,
    register_weight,
    norm_encode,
    zero,
    op,
    activate,
    exponent,
    permute,
    address48,
    route_k,
    perm2,
    PUBLISH2,
    PUBLISH150,
    shared_packed,
    local_route_word,
    _prmt
)

JOINT_SOURCE_SHA256 = '708b5656c83488dddff69dde12b6a3b34723fe2257f1bf3922487d0a946a4df2'


@spatial_jit(dynamic="tiles height width gx sx sy rn sn wns inp out counter invn prn gn skip pin pon mn poolout", **dict(_JIT, compile_flags=['-lineinfo']))
def _packet_fused(
    tiles,
    heads,
    height,
    width,
    gx,
    sx,
    sy,
    flags,
    seq,
    rn,
    sn,
    wns,
    offsets,
    inp,
    out,
    counter,
    up,
    invn,
    prn,
    gn,
    skip,
    ds,
    pin,
    pon,
    mn,
    poolout
):
    xn, redn, tn, qn, bn, scn, opn, fgn, agn = wns
    ro, to, fg, qo, bo, sc, pro, ag = offsets

    @T.prim_func
    def kernel(
        R: T.Tensor((rn, ), 'uint8'),
        Source: T.Tensor((sn, ), 'uint8'),
        RW: T.Tensor((rn // 4, ), 'int32'),
        SW: T.Tensor((sn // 4, ), 'int32'),
        WX: T.Tensor((xn, ), 'int32'),
        WR: T.Tensor((redn, ), 'int32'),
        WT: T.Tensor((tn, ), 'int32'),
        WQ: T.Tensor((qn, ), 'int32'),
        WB: T.Tensor((bn, ), 'int32'),
        WS: T.Tensor((scn, ), 'int32'),
        WP: T.Tensor((opn, ), 'int32'),
        FG: T.Tensor((fgn, ), 'int32'),
        AG: T.Tensor((agn, ), 'int32'),
        Inv: T.Tensor((invn, ), 'int32'),
        Proj: T.Tensor((prn, ), 'int32'),
        Gate: T.Tensor((gn, ), 'int32'),
        PI: T.Tensor((pin, ), 'int32'),
        PO: T.Tensor((pon, ), 'int32'),
        Matrix: T.Tensor((mn, ), 'int32')
    ):
        with T.Kernel(tiles, threads=heads * 32) as tile:
            T.import_source(_PRIMITIVES)
            tid = T.get_thread_binding(0)
            lane = tid % 32
            head = tid // 32
            S = T.alloc_shared((heads, 512), 'int32')
            D = T.alloc_shared((heads, 1024 if ds else 1), 'int32')
            X = T.alloc_local((4, 4, 2), 'uint32')
            C = T.alloc_local((4, 4, 2), 'uint32')
            F = T.alloc_local((2, 8, 2), 'uint32')
            Z = T.alloc_local((4 if heads == 4 else 2, 12, 2), 'uint32')
            P = T.alloc_local((16, ), 'uint32')
            A = T.alloc_local((16, ), 'uint32')
            Residual = T.alloc_local((16, ), 'uint32')
            Q = T.alloc_local((16, ), 'uint32')
            K = T.alloc_local((16, ), 'uint32')
            V = T.alloc_local((16, ), 'uint32')
            QP = T.alloc_local((16, ), 'uint32')
            KP = T.alloc_local((16, ), 'uint32')
            VP = T.alloc_local((16, ), 'uint32')
            L = T.alloc_local((2, 8, 2), 'uint32')
            temp = T.alloc_local((4, ), 'uint32')
            acc = T.alloc_local((2, ), 'uint32')
            vals = T.alloc_local((4, ), 'uint32')
            if up:
                # Arbitrary inverse projection/gate ownership stays per Half pair.
                # Skip bytes are extracted from aligned words, not widened across guards.
                for m in T.unroll(4):
                    for word in T.unroll(4):
                        temp[0] = 0
                        for b in T.unroll(4):
                            row = m * 16 + lane // 4 + (word % 2) * 8
                            if heads == 2:
                                col = head * 32 + (word // 2) * 16 + (lane % 4) * 4 + b
                                raw = joint_address2(height, width, gx, sx, sy, seq, tile, row, col)
                            else:
                                col = head * 32 + route_k(
                                    (word // 2) * 16 + (lane % 4) * 4 + (b // 2) * 2
                                ) + b % 2
                                raw = T.alloc_var(
                                    'int32',
                                    init=address48(
                                        heads, height, width, gx, sx, sy, 0, tile, row, col
                                    )
                                )[0]
                            if raw >= 0:
                                pi = Inv[raw * 2]
                                gi = Inv[raw * 2 + 1]
                                p = (
                                    T.reinterpret(Proj[pi // 2], 'uint32') >> ((pi % 2) * 16)
                                ) & 65535
                                g = (
                                    T.reinterpret(Gate[gi // 2], 'uint32') >> ((gi % 2) * 16)
                                ) & 65535
                                sb = (
                                    T.reinterpret(RW[(skip + raw) // 4], 'uint32') >>
                                    ((raw % 4) * 8)
                                ) & 255
                                value = op(
                                    'encode', op('fma', op('decode', sb), g, p), T.uint32(0)
                                ) & 255
                                temp[0] = temp[0] | (value << (b * 8))
                        S[head, m * 128 + lane * 4 + word] = T.reinterpret(temp[0], 'int32')
            elif heads == 2:
                if seq == 7:
                    for m in T.unroll(4):
                        row = m * 16 + lane // 2
                        col = head * 32 + (lane % 2) * 16
                        raw = joint_address2(height, width, gx, sx, sy, seq, tile, row, col)
                        for j in T.vectorized(4):
                            temp[j] = 0
                        if (raw >= 0) & (raw <= sn - inp - 16):
                            for j in T.vectorized(4):
                                temp[j] = T.cast(SW[((inp + raw) // 16) * 4 + j], 'uint32')
                        for j in T.unroll(4):
                            S[head,
                              m * 128 + ((row % 8) * 4 + j) * 4 + (row % 16) // 8 +
                              (col % 32) // 16 * 2] = T.cast(temp[j], 'int32')
                else:
                    for m in T.unroll(4):
                        row = m * 16 + lane // 4
                        col = head * 32 + (lane % 4) * 4
                        raw = joint_address2(height, width, gx, sx, sy, seq, tile, row, col)
                        for j in T.vectorized(4):
                            temp[j] = 0
                        if (raw >= 0) & (raw <= sn - inp - 16):
                            for j in T.vectorized(4):
                                temp[j] = T.cast(SW[((inp + raw) // 16) * 4 + j], 'uint32')
                        for j in T.vectorized(4):
                            S[head, m * 128 + lane * 4 + j] = T.cast(temp[j], 'int32')
            elif flags & 1:
                for m in T.unroll(4):
                    for word in T.unroll(4):
                        row = m * 16 + lane // 4 + (word % 2) * 8
                        col = route_k(head * 32 + (word // 2) * 16 + (lane % 4) * 4)
                        raw = T.alloc_var(
                            'int32',
                            init=address48(heads, height, width, gx, sx, sy, 1, tile, row, col)
                        )[0]
                        S[head, m * 128 + lane * 4 + word] = 0
                        if raw >= 0:
                            S[head, m * 128 + lane * 4 + word] = SW[(inp + raw) // 4]
            else:
                for slab in T.unroll(2):
                    for parity in T.unroll(2):
                        raw = T.alloc_var(
                            'int32',
                            init=address48(
                                heads,
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
                        for j in T.vectorized(4):
                            temp[j] = 0
                        if (raw >= 0) & (raw <= sn - inp - 16):
                            for j in T.vectorized(4):
                                temp[j] = T.cast(SW[((inp + raw) // 16) * 4 + j], 'uint32')
                        for j in T.unroll(4):
                            S[head, (slab * 2 + j % 2) * 128 + lane * 4 + parity +
                              (j // 2) * 2] = T.cast(temp[j], 'int32')
            T.sync_threads()
            if heads == 2:
                # Warp owns M32 and all N64 until the FF publication barrier.
                for channel in T.unroll(2):
                    for m in T.unroll(2):
                        for n in T.unroll(4):
                            bits = shared_packed(S, channel, (head * 2 + m) * 4 + n, lane)
                            g = FG[fg // 4 + channel * 16 + n * 4 + lane % 4]
                            F[m, channel * 4 + n, 0] = op('mul', op('decode', bits), g)
                            F[m, channel * 4 + n, 1] = op('mul', op('decode', bits >> 16), g)
                for stream in T.serial(2):
                    zero(C, 2, 4)
                    for slab in T.serial(4):
                        zero(X, 2, 4)
                        weighted(
                            S, WX, X, lane, head * 2, 2, 4, 0, 2, stream * 8192 + slab * 1024, 4096
                        )
                        for m in T.unroll(2):
                            for n in T.unroll(4):
                                P[m * 4 +
                                  n] = op('encode', activate(X[m, n, 0]), activate(X[m, n, 1]))
                        for word in T.unroll(8):
                            A[word] = local_route_word(P, word, lane, _ROUTES['hidden'])
                        register_weight(A, WR, C, lane, 2, 4, ro + stream * 4096 + slab * 1024)
                    for m in T.unroll(2):
                        for n in T.unroll(4):
                            P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
                    for word in T.unroll(8):
                        A[word] = local_route_word(P, word, lane, _ROUTES['hidden'])
                    register_weight(A, WT, F, lane, 2, 8, to + stream * 2048)
                T.sync_threads()
                for channel in T.unroll(2):
                    for m in T.unroll(2):
                        for n in T.unroll(4):
                            P[m * 4 +
                              n] = op('encode', F[m, channel * 4 + n, 0], F[m, channel * 4 + n, 1])
                    for word in T.unroll(8):
                        S[channel,
                          (head * 2 + word // 4) * 128 + lane * 4 + word % 4] = T.reinterpret(
                              local_route_word(P, word, lane, _ROUTES['hidden']), 'int32'
                          )
            else:
                for word in T.unroll(16):
                    Residual[word] = shared_packed(S, head, word, lane)
                zero(C, 4, 4)
                for slab in T.serial(4):
                    zero(X, 4, 4)
                    weighted(S, WX, X, lane, 0, 4, 4, 0, 3, head * 16384 + slab * 1024, 4096)
                    for m in T.unroll(4):
                        finish_row(S, WX, X, lane, m, 0, 3, head * 16384 + slab * 1024, 4096)
                        for n in T.unroll(4):
                            P[n] = op('encode', activate(X[m, n, 0]), activate(X[m, n, 1]))
                        for word in T.unroll(4):
                            A[word] = local_route_word(P, word, lane, _ROUTES['hidden'])
                        for n in T.unroll(4):
                            wb = (ro + head * 4096 +
                                  slab * 1024) // 4 + (n // 2) * 128 + lane * 4 + (n % 2) * 2
                            r = T.call_extern(
                                'uint64',
                                'wide_mma',
                                A[0],
                                A[1],
                                A[2],
                                A[3],
                                WR[wb],
                                WR[wb + 1],
                                C[m, n, 0],
                                C[m, n, 1]
                            )
                            C[m, n, 0] = T.cast(r, 'uint32')
                            C[m, n, 1] = T.cast(r >> 32, 'uint32')
                for m in T.unroll(4):
                    for n in T.unroll(4):
                        P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
                T.sync_threads()
                for word in T.unroll(16):
                    S[head, (word // 4) * 128 + lane * 4 + word % 4] = T.reinterpret(
                        local_route_word(P, word, lane, _ROUTES['hidden']), 'int32'
                    )
                T.sync_threads()
                qualified._f_initial(Residual, FG, C, head, lane, fg)
                weighted(S, WT, C, lane, 0, 4, 4, 0, 4, to + head * 1024, 4096)
                for m in T.unroll(4):
                    for n in T.unroll(4):
                        P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
                T.sync_threads()
                for word in T.unroll(16):
                    S[head, (word // 4) * 128 + lane * 4 + word % 4] = T.reinterpret(
                        local_route_word(P, word, lane, _ROUTES['hidden']), 'int32'
                    )
            T.sync_threads()
            # Head ownership begins here. Q/K/V wait for all encoded producers.
            if heads == 2:
                for slab in T.unroll(2):
                    zero(Z, 2, 12)
                    weighted(S, WQ, Z, lane, slab * 2, 2, 12, 0, 2, qo + head * 3072, 6144)
                    for comp in T.unroll(3):
                        for m in T.unroll(2):
                            if comp == 0:
                                norm_encode(Z, QP, WS, lane, head, heads, sc, m, comp, slab * 2 + m)
                            elif comp == 1:
                                norm_encode(Z, KP, WS, lane, head, heads, sc, m, comp, slab * 2 + m)
                            else:
                                norm_encode(Z, VP, WS, lane, head, heads, sc, m, comp, slab * 2 + m)
            else:
                zero(Z, 4, 12)
                weighted(S, WQ, Z, lane, 0, 4, 12, 0, 3, qo + head * 3072, 12288)
                for comp in T.unroll(3):
                    for m in T.unroll(4):
                        finish_row(S, WQ, Z, lane, m, comp, 3, qo + head * 3072, 12288)
                        if comp == 0:
                            norm_encode(Z, QP, WS, lane, head, heads, sc, m, comp, m)
                        elif comp == 1:
                            norm_encode(Z, KP, WS, lane, head, heads, sc, m, comp, m)
                        else:
                            norm_encode(Z, VP, WS, lane, head, heads, sc, m, comp, m)
            for word in T.unroll(16):
                Q[word] = local_route_word(QP, word, lane, _ROUTES['q' + str(heads)])
                K[word] = local_route_word(KP, word, lane, _ROUTES['k' + str(heads)])
                V[word] = local_route_word(VP, word, lane, _ROUTES['v' + str(heads)])
            T.sync_threads()
            for slab in T.serial(2):
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
                            WB[wb],
                            WB[wb + 1]
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
                # Read FF only after this M32 PV is ready, before shared overwrite.
                for word in T.unroll(8):
                    Residual[word] = shared_packed(S, head, slab * 8 + word, lane)
                T.sync_threads()
                for word in T.unroll(8):
                    S[head, (slab * 2 + word // 4) * 128 + lane * 4 + word % 4] = T.reinterpret(
                        local_route_word(P, word, lane, _ROUTES['hidden']), 'int32'
                    )
                T.sync_threads()
                for m in T.unroll(2):
                    for n in T.unroll(4):
                        g = AG[ag // 4 + head * 16 + n * 4 + lane % 4]
                        C[m, n, 0] = op('mul', op('decode', Residual[m * 4 + n]), g)
                        C[m, n, 1] = op('mul', op('decode', Residual[m * 4 + n] >> 16), g)
                weighted(S, WP, C, lane, slab * 2, 2, 4, 0, heads, pro + head * 1024, heads * 1024)
                for m in T.unroll(2):
                    for n in T.unroll(4):
                        P[m * 4 + n] = op('encode', C[m, n, 0], C[m, n, 1])
                if heads == 2:
                    # seq150 interleaves M32 slabs, so retain encoded output until
                    # both producers finish. Half accumulators are not retained.
                    for word in T.unroll(8):
                        QP[slab * 8 + word] = P[word]
                elif flags & 2:
                    for word in T.unroll(8):
                        A[word] = route(P, word, lane, _ROUTES['publish'])
                    row = slab * 32 + lane // 4 + (lane & 1) * 16 + ((lane >> 1) & 1) * 8
                    for part in T.unroll(2):
                        raw = T.alloc_var(
                            'int32',
                            init=address48(
                                heads,
                                height,
                                width,
                                gx,
                                sx,
                                sy,
                                2,
                                tile,
                                row,
                                head * 32 + part * 16
                            )
                        )[0]
                        if (raw >= 0) & (raw <= rn - out - 16):
                            for j in T.vectorized(4):
                                RW[((out + raw) // 16) * 4 + j] = T.cast(A[part * 4 + j], 'int32')
                else:
                    for parity in T.unroll(2):
                        raw = T.alloc_var(
                            'int32',
                            init=address48(
                                heads,
                                height,
                                width,
                                gx,
                                sx,
                                sy,
                                0,
                                tile,
                                slab * 32 + lane // 4 + parity * 8,
                                head * 32 + (lane % 4) * 2
                            )
                        )[0]
                        for j in T.unroll(4):
                            base = (j % 2) * 4 + (j // 2) * 2
                            temp[j] = _prmt(P[base], P[base + 1], 0x5410 if parity == 0 else 0x7632)
                        if (raw >= 0) & (raw <= rn - out - 16):
                            for j in T.vectorized(4):
                                RW[((out + raw) // 16) * 4 + j] = T.cast(temp[j], 'int32')
                if ds:
                    for m in T.unroll(2):
                        for n in T.unroll(4):
                            for j in T.unroll(2):
                                D[head, (((slab * 2 + m) * 4 + n) * 32 + lane) * 2 +
                                  j] = T.reinterpret(C[m, n, j], 'int32')
                T.sync_threads()
            if heads == 2:
                for part in T.unroll(4):
                    for j in T.unroll(4):
                        A[part * 4 + j] = route(
                            QP, part * 4 + j, lane, PUBLISH150 if seq == 150 else PUBLISH2
                        )
                    ix = T.alloc_var(
                        'int32',
                        init=permute(part * 512 + lane * 4, PUBLISH150 if seq == 150 else PUBLISH2)
                    )[0]
                    row = (ix // 512) * 16 + ((ix // 4) % 32) // 4 + (ix % 4) // 2 * 8
                    col = head * 32 + ((ix // 128) % 4) * 8 + ((ix // 4) % 4) * 2 + ix % 2
                    raw = joint_address2(
                        height, width, gx, sx, sy, seq, tile, row, perm2(col), True
                    )
                    if (raw >= 0) & (raw <= rn - out - 16):
                        for j in T.vectorized(4):
                            RW[((out + raw) // 16) * 4 + j] = T.cast(A[part * 4 + j], 'int32')
            if ds:
                for group in T.serial(2):
                    for n in T.serial(4):
                        acc[0] = 0
                        acc[1] = 0
                        for kp in T.serial(heads):
                            for word in T.unroll(4):
                                temp[word] = 0
                                for b in T.unroll(4):
                                    row = lane // 4 + (word % 2) * 8
                                    k = kp * 32 + (word // 2) * 16 + (lane % 4) * 4 + b
                                    if heads != 2:
                                        k = route_k(k)
                                    for t in T.unroll(4):
                                        ix = PI[((tile * 16 + row) * 4 + t) * (heads * 32) + k]
                                        vals[t] = 0
                                        if ix >= 0:
                                            rr = ix // (heads * 32)
                                            cc = ix % (heads * 32)
                                            raw = T.reinterpret(
                                                D[cc // 32,
                                                  (
                                                      ((rr // 16) * 4 + (cc % 32) // 8) * 32 +
                                                      (rr % 8) * 4 + (cc % 8) // 2
                                                  ) * 2 + (rr % 16) // 8],
                                                'uint32'
                                            )
                                            hv = (raw >> ((cc % 2) * 16)) & 65535
                                            vals[t] = hv | (hv << 16)
                                    avg = op(
                                        'mul',
                                        op(
                                            'add',
                                            op('add', vals[0], vals[1]),
                                            op('add', vals[2], vals[3])
                                        ),
                                        op('splat', 0.25)
                                    )
                                    temp[word] = temp[word] | (
                                        (op('encode', avg, avg) & 255) << (b * 8)
                                    )
                            wb = kp * (heads *
                                       64) * 8 + (group * heads + head
                                                  ) * 256 + (n // 2) * 128 + lane * 4 + (n % 2) * 2
                            r = T.call_extern(
                                'uint64',
                                'wide_mma',
                                temp[0],
                                temp[1],
                                temp[2],
                                temp[3],
                                Matrix[wb],
                                Matrix[wb + 1],
                                acc[0],
                                acc[1]
                            )
                            acc[0] = T.cast(r, 'uint32')
                            acc[1] = T.cast(r >> 32, 'uint32')
                        encoded = op('encode', acc[0], acc[1])
                        for b in T.unroll(4):
                            row = lane // 4 + (b // 2) * 8
                            col = (group * heads + head) * 32 + n * 8 + (lane % 4) * 2 + b % 2
                            dest = PO[(tile * 16 + row) * (heads * 64) + col]
                            if dest >= 0:
                                R[poolout + dest] = T.cast(encoded >> (b * 8), 'uint8')
            T.sync_threads()
            T.evaluate(T.call_extern('int32', '__threadfence'))
            if counter >= 0:
                if tid == 0:
                    RW[counter // 4 + tile] = 0

    return kernel


@spatial_jit(dynamic="rows rn wn imn pn inp", **dict(_JIT, compile_flags=['-lineinfo']))
def _packet_project(heads, rows, rn, wn, imn, pn, inp, woff):

    @T.prim_func
    def kernel(
        R: T.Tensor((rn // 4, ), 'int32'),
        W: T.Tensor((wn, ), 'int32'),
        IM: T.Tensor((imn, ), 'int32'),
        Proj: T.Tensor((pn, ), 'int32')
    ):
        with T.Kernel(T.ceildiv(rows, 16), heads, threads=32) as (tile, head):
            T.import_source(_PRIMITIVES)
            lane = T.get_thread_binding(0)
            a = T.alloc_local((4, ), 'uint32')
            acc = T.alloc_local((2, ), 'uint32')
            for n in T.serial(4):
                acc[0] = 0
                acc[1] = 0
                for kp in T.serial(heads * 2):
                    for word in T.unroll(4):
                        row = tile * 16 + lane // 4 + (word % 2) * 8
                        col = kp * 32 + (word // 2) * 16 + (lane % 4) * 4
                        if heads != 2:
                            col = route_k(col)
                        a[word] = 0
                        if row < rows:
                            if heads == 2:
                                base = IM[row * (heads * 64) + (col // 16) * 16]
                                extra = col % 16
                            else:
                                base = IM[row * (heads * 64) + col]
                                extra = 0
                            if base >= 0:
                                a[word] = T.reinterpret(R[(inp + base + extra) // 4], 'uint32')
                    wb = woff // 4 + kp * (heads *
                                           32) * 8 + head * 256 + (n // 2
                                                                   ) * 128 + lane * 4 + (n % 2) * 2
                    r = T.call_extern(
                        'uint64',
                        'wide_mma',
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        W[wb],
                        W[wb + 1],
                        acc[0],
                        acc[1]
                    )
                    acc[0] = T.cast(r, 'uint32')
                    acc[1] = T.cast(r >> 32, 'uint32')
                for j in T.unroll(2):
                    row = tile * 16 + lane // 4 + j * 8
                    col = head * 32 + n * 8 + (lane % 4) * 2
                    if row < rows:
                        Proj[(row * (heads * 32) + col) // 2] = T.reinterpret(acc[j], 'int32')

    return kernel


def _build_fused(spec):
    if spec.name == 'streamed_project2':
        r, proj, w, im = (spec.tensor(i, dtype=(torch.uint8 if i == 0 else torch.int32))
                          for i in range(4))
        rows = spec.scalar(5)
        with private_compile():
            fn = _packet_project(
                2, rows, r.numel(), w.numel(), im.numel(), proj.numel(), spec.scalar(4), 0
            )
        rw = r.view(torch.int32)

        def launch_project():
            fn(rw, w, im, proj)

        launch_project.kernel_launches = launch_project.tilelang_launches = 1
        launch_project.workspace_bytes = launch_project.scratch_bytes = 0
        launch_project.kernels = (fn, )
        return launch_project
    name = spec.name
    heads = 2 if name in ('streamed2',
                          'packet2_ds') else 4 if name.startswith('four_') or '4_' in name else 8
    cross = 7 if heads == 2 else 8
    up = bool(spec.scalar(cross, 'up_inverse'))
    ds = name.endswith('_ds')
    r = spec.tensor(0)
    source = r if heads == 2 else spec.tensor(7)
    if heads == 2:
        height, width, gx, sx, sy, seq = spec.tensor(2, dtype=torch.int32)[:6].cpu().tolist()
        flags = 0
        weights = tuple(
            spec.tensor(1, k, torch.int32) for k in (
                'expand',
                'reduce',
                'tail',
                'qkv',
                'bias',
                'scale',
                'projection',
                'ffn_gate',
                'attn_gate'
            )
        )
        offsets = (0, ) * 8
        inp, out, counter = (spec.scalar(i) for i in (4, 5, 6))
    else:
        height, width, gx, sx, sy, flags = (spec.scalar(2, k)
                                            for k in ('height', 'width', 'grid_x', 'shift_x',
                                                      'shift_y', 'flags'))
        seq = 0
        w = spec.tensor(1, dtype=torch.int32)
        weights = (w, ) * 9
        offsets = (0x10000, 0x14000, 0x18010, 0x18120, 0x24120, 0x2c120, 0x2c130,
                   0x30130) if heads == 4 else (
                       0x40000, 0x48000, 0x58010, 0x58220, 0x88220, 0x98220, 0x98240, 0xa8240
                   )
        if up:
            shift = 0x80e0 if heads == 4 else 0x201e0
            offsets = (
                *offsets[:2], 0x20000 if heads == 4 else 0x78000, *(v + shift for v in offsets[3:])
            )
        inp, out, counter = (spec.scalar(i) for i in (3, 4, 5))
    if tuple(spec.block) != (32, heads, 1) or spec.grid[1:] != (1, 1):
        raise NotImplementedError(f'Unexpected wide geometry: {spec.geometry}')
    if height % 4 or width % 4 or sx not in (0, -4) or sy not in (0, -4):
        raise NotImplementedError('Wide raw packet geometry outside native qualified domain')
    dummy = weights[0]
    inv, proj, gate = (tuple(
        spec.tensor(cross, k, torch.int32) for k in ('up_inverse', 'projection', 'gate')) if up else
                       (dummy, ) * 3)
    pi, po, matrix = (tuple(
        spec.tensor(cross, k, torch.int32)
        for k in ('pool_input', 'pool_output', 'matrix')) if ds else (dummy, ) * 3)
    skip = spec.scalar(cross, 'skip') if up else 0
    poolout = spec.scalar(cross, 'pool_out') if ds else 0
    prefix = []
    if up and heads != 2:
        im = spec.tensor(9, dtype=torch.int32)
        rows = im.numel() // (heads * 64)
        with private_compile():
            project = _packet_project(
                heads,
                rows,
                r.numel(),
                weights[0].numel(),
                im.numel(),
                proj.numel(),
                inp,
                0x18000 if heads == 4 else 0x58000
            )
        prefix.append((project, (r.view(torch.int32), weights[0], im, proj)))
    with private_compile():
        fn = _packet_fused(
            spec.grid[0],
            heads,
            height,
            width,
            gx,
            sx,
            sy,
            flags,
            seq,
            r.numel(),
            source.numel(),
            tuple(v.numel() for v in weights),
            offsets,
            inp,
            out,
            counter,
            up,
            inv.numel(),
            proj.numel(),
            gate.numel(),
            skip,
            ds,
            pi.numel(),
            po.numel(),
            matrix.numel(),
            poolout
        )
    args = (
        r,
        source,
        r.view(torch.int32),
        source.view(torch.int32),
        *weights,
        inv,
        proj,
        gate,
        pi,
        po,
        matrix
    )

    def launch():
        for f, a in prefix:
            f(*a)
        fn(*args)

    launch.kernel_launches = launch.tilelang_launches = 1 + len(prefix)
    launch.workspace_bytes = launch.scratch_bytes = 0
    launch.shared_bytes = heads * (6144 if ds else 2052)
    launch.kernels = tuple(f for f, a in prefix) + (fn, )
    return launch


def build_step(spec):
    if not supports(spec.name):
        raise NotImplementedError('Not a joint 2H/4H Step: ' + spec.name)
    return _build_fused(spec)
