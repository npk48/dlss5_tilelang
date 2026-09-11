"""1H shallow blocks and block0/post70 endpoints: physical M32 consumers.

The 797fd63 math, layouts, K32 order and launch geometry are retained.
"""
from tilelang_nr.instructions import instruction_source
from pathlib import Path
import torch
import tilelang
import tilelang.language as T
from fp8_toolchain import private_compile
from tilelang_nr.common.shallow import (
    NAMES,
    supports,
    nc,
    logical,
    compact,
    address,
    up_address,
    pool_local,
    activate,
    exponent,
    HEADER as MATH_HEADER
)
from tilelang_nr.common.shallow_ops import call, pack, unpack, splat, add, mul, e4four, shfl, une4, hm, ha, hf

HEADER = MATH_HEADER + '\n' + instruction_source('shallow_joint_primitives.cuh')


def mem(name, *args):
    return T.call_extern('int32', 'nr_tl_shallow_joint::' + name, *args)


@T.macro
def load_packet(q, data, base):
    for j in T.unroll(4):
        q[j] = T.uint32(0)
    if base >= 0:
        T.evaluate(mem('ld128', T.address_of(q[0]), T.address_of(data[base])))


@T.macro
def paired(c, a, weights, offset, M: T.int32, N: T.int32):
    lane = T.get_thread_binding(0)
    b = T.alloc_local((4, ), 'uint32')
    for p in T.unroll(N // 16):
        T.evaluate(
            mem(
                'ld128',
                T.address_of(b[0]),
                T.address_of(weights[offset // 4 + p * 128 + lane * 4])
            )
        )
        for m in T.unroll(M // 16):
            for n in T.unroll(2):
                T.evaluate(
                    call(
                        'mma8',
                        T.address_of(c[(m * (N // 8) + p * 2 + n) * 2]),
                        a[m * 4],
                        a[m * 4 + 1],
                        a[m * 4 + 2],
                        a[m * 4 + 3],
                        b[n * 2],
                        b[n * 2 + 1],
                        dtype='int32'
                    )
                )


@T.macro
def encode(a, c, M: T.int32):
    for m in T.unroll(M // 16):
        for p in T.unroll(2):
            for i in T.unroll(2):
                a[m * 4 + p * 2 + i] = e4four(c[m * 8 + p * 4 + i], c[m * 8 + p * 4 + 2 + i])


@T.macro
def true_half(a, ff, raw, gate):
    lane = T.get_thread_binding(0)
    encode(a, raw, 64)
    for j in T.unroll(32):
        col = j // 2 % 4 * 8 + (lane & 3) * 2
        ff[j] = mul(raw[j], pack(gate[col], gate[col + 1]))


@T.macro
def raw_word(a, ff, word, gate, m: T.int32, p: T.int32, i: T.int32):
    lane = T.get_thread_binding(0)
    canonical = T.alloc_local((1, ), 'uint32')
    canonical[0] = word
    for byte in T.unroll(4):
        v = (word >> (byte * 8)) & T.uint32(255)
        if (v & T.uint32(127)) == T.uint32(127):
            canonical[0] = (canonical[0] & (T.uint32(0xffffffff) ^
                                            (T.uint32(255) <<
                                             (byte * 8)))) | (call('e4', une4(v)) << (byte * 8))
    a[m * 4 + p * 2 + i] = canonical[0]
    for j in T.unroll(2):
        c = p * 16 + j * 8 + (lane & 3) * 2
        raw = pack(une4(word >> (j * 16)), une4(word >> (j * 16 + 8)))
        ff[m * 8 + p * 4 + j * 2 + i] = mul(raw, pack(gate[c], gate[c + 1]))


def block_address(row, c, bx, by, W):
    slot = (row & 48) | ((row & 7) << 1) | ((row & 8) >> 3)
    y, x, cc = by * 8 + slot // 8, bx * 8 + slot % 8, nc(c)
    t = (y & 7) * 8 + (x & 7)
    pp = 64 * (t // 2) + 4 * (t & 1) + 8 * (cc // 4) + (cc & 3)
    ss = pp // 512
    cy = y // 8 * 2 + T.Cast('int32', (ss == 1) | (ss == 2))
    cx = x // 8 * 2 + T.Cast('int32', (ss == 2) | (ss == 3))
    return (cy * (W // 4) + cx) * 512 + (pp & 511)


@T.macro
def consume(
    out,
    pool_a,
    rail,
    X,
    Pre,
    Readout,
    Head,
    Status,
    kind,
    seq,
    H0: T.int32,
    W0: T.int32,
    bx,
    by,
    slab: T.int32,
    out_offset: T.int32,
    has_pre
):
    lane = T.get_thread_binding(0)
    H, W = H0 // 2, W0 // 2
    q = T.alloc_local((4, ), 'uint32')
    z = T.alloc_local((4, ), 'uint32')
    b = T.alloc_local((2, ), 'uint32')
    if kind == 'post':
        # Two K16 with original Half C handoff, directly from this M32's C.
        for j in T.unroll(4):
            z[j] = T.uint32(0)
        for kp in T.unroll(2):
            for i in T.unroll(2):
                k = kp * 16 + (lane & 3) * 2 + i * 8
                b[i] = pack(Readout[k * 8 + lane // 4], Readout[(k + 1) * 8 + lane // 4])
            for m in T.unroll(2):
                T.evaluate(
                    call(
                        'mma16',
                        T.address_of(z[m * 2]),
                        out[m * 8 + kp * 4],
                        out[m * 8 + kp * 4 + 1],
                        out[m * 8 + kp * 4 + 2],
                        out[m * 8 + kp * 4 + 3],
                        b[0],
                        b[1],
                        dtype='int32'
                    )
                )
        for j in T.unroll(4):
            rg = shfl(z[j], (lane & 7) * 4)
            ba = shfl(z[j], (lane & 7) * 4 + 1)
            if lane // 8 == j:
                q[0] = rg
                q[1] = ba
        m = logical(slab * 2 + lane // 16)
        r = lane % 16
        dx = T.Cast('int32', (m == 2) | (m == 3))
        dy = T.Cast('int32', (m == 1) | (m == 2))
        x, y = bx * 8 - 4 + dx * 4 + (r & 3), by * 8 - 4 + dy * 4 + (r >> 2)
        if (x >= 0) & (x < W0) & (y >= 0) & (y < H0):
            for j in T.unroll(4):
                v = T.Cast('float32', unpack(q[j // 2], j % 2))
                if T.isinf(v) | T.isnan(v):
                    T.evaluate(call('status_or', T.address_of(Status[0]), 4, dtype='int32'))
                Head[(y * W0 + x) * 4 + j] = v
    else:
        for m in T.unroll(2):
            lm = logical(slab * 2 + m)
            if seq == 154:
                # C owners gather compact physical N16 packets before predication.
                for p in T.unroll(2):
                    lo = e4four(out[m * 8 + p * 4], out[m * 8 + p * 4 + 2])
                    hi = e4four(out[m * 8 + p * 4 + 1], out[m * 8 + p * 4 + 3])
                    for k in T.unroll(4):
                        l = shfl(lo, (lane % 8) * 4 + k)
                        h = shfl(hi, (lane % 8) * 4 + k)
                        q[k] = T.if_then_else(lane < 8, l, h)
                    dest = address(lm * 16 + lane, p * 16, bx, seq, H, W, True)
                    if (lane < 16) & (dest >= 0):
                        T.evaluate(
                            mem(
                                'st128', T.address_of(X[out_offset + dest]), q[0], q[1], q[2], q[3]
                            )
                        )
                    if has_pre:
                        for n in T.unroll(2):
                            for i in T.unroll(2):
                                d = address(
                                    lm * 16 + lane // 4 + i * 8, (p * 2 + n) * 8 + (lane & 3) * 2,
                                    bx,
                                    seq,
                                    H,
                                    W,
                                    True
                                )
                                if d >= 0:
                                    T.evaluate(
                                        mem(
                                            'st32',
                                            T.address_of(Pre[d]),
                                            out[m * 8 + p * 4 + n * 2 + i]
                                        )
                                    )
            else:
                row, c = lm * 16 + lane // 4, (lane & 3) * 2
                if kind == 'block0':
                    dest = block_address(row, c, bx, by, W0)
                else:
                    dest = address(row, c, bx, seq, H, W, True)
                if dest >= 0:
                    T.evaluate(
                        mem(
                            'st128',
                            T.address_of(X[out_offset + dest]),
                            e4four(out[m * 8], out[m * 8 + 2]),
                            e4four(out[m * 8 + 1], out[m * 8 + 3]),
                            e4four(out[m * 8 + 4], out[m * 8 + 6]),
                            e4four(out[m * 8 + 5], out[m * 8 + 7])
                        )
                    )
                    if has_pre:
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(Pre[dest]),
                                out[m * 8],
                                out[m * 8 + 2],
                                out[m * 8 + 1],
                                out[m * 8 + 3]
                            )
                        )
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(Pre[dest + 8]),
                                out[m * 8 + 4],
                                out[m * 8 + 6],
                                out[m * 8 + 5],
                                out[m * 8 + 7]
                            )
                        )
            if kind == 'block0':
                # Mandatory unquantized pool rail, independent of optional Pre.
                for n in T.unroll(4):
                    for i in T.unroll(2):
                        for j in T.unroll(2):
                            rail[lm * 16 + lane // 4 + i * 8,
                                 n * 8 + (lane & 3) * 2 + j] = unpack(out[m * 8 + n * 2 + i], j)
            if kind == 'ds':
                by_ = bx // (W // 8)
                cm, side = lm % 2, lm // 2
                if ((by_ != 0) | (cm == 1 - side)) & ((by_ != H // 8) | (cm == side)):
                    for k in T.unroll(2):
                        for i in T.unroll(4):
                            v = out[m * 8 + k * 4 + i]
                            p = add(v, shfl(v, lane ^ 4))
                            z[i] = mul(add(p, shfl(p, lane ^ 16)), splat(.25))
                        src = (lane & 3) + ((lane & 4) << 1)
                        z0, z1 = shfl(z[0], src), shfl(z[1], src)
                        z2, z3 = shfl(z[2], src), shfl(z[3], src)
                        lo = T.if_then_else((lane & 16) != 0, z1, z0)
                        hi = T.if_then_else((lane & 16) != 0, z3, z2)
                        if ((lane >> 3) & 1) == side:
                            if (by_ == 0) | (by_ == H // 8):
                                pool_a[k * 2] = pool_a[k * 2] | e4four(lo, hi)
                            else:
                                pool_a[k * 2 + (cm ^ side)] = pool_a[k * 2 +
                                                                     (cm ^ side)] | e4four(lo, hi)


@T.macro
def body(
    a,
    ff,
    W0,
    W1,
    WQ,
    WP,
    G1,
    scale,
    pool_a,
    rail,
    X,
    Pre,
    Readout,
    Head,
    Status,
    kind,
    seq,
    H0: T.int32,
    W0_: T.int32,
    bx,
    by,
    out_offset: T.int32,
    has_pre
):
    lane = T.get_thread_binding(0)
    hidden = T.alloc_local((16, ), 'uint32')
    ha_ = T.alloc_local((16, ), 'uint32')
    z = T.alloc_local((32, ), 'uint32')
    inv = T.alloc_local((8, ), 'uint32')
    qa = T.alloc_local((16, ), 'uint32')
    kb = T.alloc_local((16, ), 'uint32')
    vb = T.alloc_local((16, ), 'uint32')
    logits = T.alloc_local((32, ), 'uint32')
    pa = T.alloc_local((16, ), 'uint32')
    attended = T.alloc_local((16, ), 'uint32')
    out = T.alloc_local((16, ), 'uint32')
    seed = T.alloc_local((4, ), 'uint32')
    for part in T.serial(4):
        for pair in T.unroll(2):
            for j in T.unroll(16):
                hidden[j] = T.uint32(0)
            paired(hidden, a, W0, part * 1024 + pair * 512, 64, 16)
            for m in T.unroll(4):
                for i in T.unroll(2):
                    ha_[m * 4 + pair * 2 +
                        i] = e4four(activate(hidden[m * 4 + i]), activate(hidden[m * 4 + i + 2]))
        paired(ff, ha_, W1, part * 1024, 64, 32)
    encode(a, ff, 64)
    for component in T.unroll(3):
        for j in T.unroll(32):
            z[j] = T.uint32(0)
        paired(z, a, WQ, component * 1024, 64, 32)
        if component < 2:
            for row in T.unroll(8):
                j = row // 2 * 8 + row % 2
                s = add(
                    call('fma', z[j], z[j], mul(z[j + 4], z[j + 4])),
                    call('fma', z[j + 2], z[j + 2], mul(z[j + 6], z[j + 6]))
                )
                t = add(
                    add(shfl(s, lane & 28), shfl(s, (lane & 28) + 2)),
                    add(shfl(s, (lane & 28) + 1), shfl(s, (lane & 28) + 3))
                )
                h = T.max(ha(unpack(t, 0), unpack(t, 1)), T.float16(6.198883056640625e-05))
                inv[row] = splat(call('rsqrt', h, dtype='float32'))
            for j in T.unroll(32):
                z[j] = mul(z[j], inv[j // 8 * 2 + j % 2])
                if component == 0:
                    z[j] = mul(z[j], splat(scale))
        for m in T.unroll(4):
            for p in T.unroll(2):
                for i in T.unroll(2):
                    src = logical(m) * 8 + p * 4 + i
                    if component == 0:
                        qa[m * 4 + p * 2 + i] = e4four(z[src], z[src + 2])
                    if component == 1:
                        kb[m * 4 + i * 2 + p] = e4four(z[src], z[src + 2])
        if component == 2:
            for part in T.unroll(2):
                for n in T.unroll(4):
                    m0 = T.if_then_else(part == 0, 0, 1)
                    m1 = T.if_then_else(part == 0, 3, 2)
                    vb[part * 8 + n * 2] = e4four(
                        call('transpose', z[m0 * 8 + n * 2]),
                        call('transpose', z[m0 * 8 + n * 2 + 1])
                    )
                    vb[part * 8 + n * 2 + 1] = e4four(
                        call('transpose', z[m1 * 8 + n * 2]),
                        call('transpose', z[m1 * 8 + n * 2 + 1])
                    )
    for slab in T.unroll(2):
        for m in T.unroll(2):
            for p in T.unroll(4):
                b = 0xc00 // 4 + ((slab * 2 + m) * 4 + p) * 128 + lane * 4
                T.evaluate(mem('ld128', T.address_of(seed[0]), T.address_of(WQ[b])))
                for n in T.unroll(2):
                    dest = (m * 8 + p * 2 + n) * 2
                    logits[dest] = seed[n * 2]
                    logits[dest + 1] = seed[n * 2 + 1]
                    q = (slab * 2 + m) * 4
                    T.evaluate(
                        call(
                            'mma8',
                            T.address_of(logits[dest]),
                            qa[q],
                            qa[q + 1],
                            qa[q + 2],
                            qa[q + 3],
                            kb[p * 4 + n * 2],
                            kb[p * 4 + n * 2 + 1],
                            dtype='int32'
                        )
                    )
        for j in T.unroll(32):
            logits[j] = exponent(logits[j])
        for row in T.unroll(4):
            j = row // 2 * 16 + row % 2
            s0 = add(logits[j], logits[j + 2])
            s1 = add(s0, add(logits[j + 4], logits[j + 6]))
            s2 = add(s1, add(logits[j + 8], logits[j + 10]))
            s = add(s2, add(logits[j + 12], logits[j + 14]))
            t = add(
                add(add(shfl(s, lane & 28), shfl(s, (lane & 28) + 1)), shfl(s, (lane & 28) + 2)),
                shfl(s, (lane & 28) + 3)
            )
            h = T.max(ha(unpack(t, 0), unpack(t, 1)), T.float16(6.198883056640625e-05))
            inv[row] = splat(call('rcp', h, dtype='float32'))
        for j in T.unroll(32):
            logits[j] = mul(logits[j], inv[j // 16 * 2 + j % 2])
        for part in T.unroll(2):
            for m in T.unroll(2):
                for p in T.unroll(2):
                    for i in T.unroll(2):
                        j = m * 16 + part * 8 + p * 4 + i
                        pa[part * 8 + m * 4 + p * 2 + i] = e4four(logits[j], logits[j + 2])
        for j in T.unroll(16):
            attended[j] = T.uint32(0)
        for part in T.unroll(2):
            for n in T.unroll(4):
                for m in T.unroll(2):
                    q, b = part * 8 + m * 4, part * 8 + n * 2
                    T.evaluate(
                        call(
                            'mma8',
                            T.address_of(attended[(m * 4 + n) * 2]),
                            pa[q],
                            pa[q + 1],
                            pa[q + 2],
                            pa[q + 3],
                            vb[b],
                            vb[b + 1],
                            dtype='int32'
                        )
                    )
        encode(a, attended, 32)
        for j in T.unroll(16):
            c = j // 2 % 4 * 8 + (lane & 3) * 2
            out[j] = mul(ff[logical(slab * 2 + j // 8) * 8 + j % 8], pack(G1[c], G1[c + 1]))
        paired(out, a, WP, 0, 32, 32)
        consume(
            out,
            pool_a,
            rail,
            X,
            Pre,
            Readout,
            Head,
            Status,
            kind,
            seq,
            H0,
            W0_,
            bx,
            by,
            slab,
            out_offset,
            has_pre
        )


@tilelang.jit(
    target={
        'kind': 'cuda', 'arch': 'sm_89'
    },
    execution_backend='nvrtc',
    compile_flags=['-lineinfo'],
    pass_configs={'tl.enable_fast_math': False}
)
def kernel(
    kind,
    seq,
    H0,
    W0_,
    gx,
    gy,
    sizes,
    in_offset,
    out_offset,
    counter,
    pool_offset,
    skip_offset,
    has_mixed,
    has_pre,
    scale_bits
):
    H, W = H0 // 2, W0_ // 2

    @T.prim_func
    def main(
        Wt0: T.Tensor((sizes[0], ), 'uint32'),
        Wt1: T.Tensor((sizes[1], ), 'uint32'),
        Wq: T.Tensor((sizes[2], ), 'uint32'),
        Wp: T.Tensor((sizes[3], ), 'uint32'),
        G0: T.Tensor((sizes[4], ), 'float16'),
        G1: T.Tensor((sizes[5], ), 'float16'),
        X: T.Tensor((sizes[6], ), 'uint8'),
        Y: T.Tensor((sizes[7], ), 'uint8'),
        Features: T.Tensor((sizes[8], ), 'float32'),
        Adapter: T.Tensor((sizes[9], ), 'float16'),
        Status: T.Tensor((sizes[10], ), 'int32'),
        Mixed: T.Tensor((sizes[11], ), 'float16'),
        Pre: T.Tensor((sizes[12], ), 'float16'),
        Gate: T.Tensor((sizes[13], ), 'float16'),
        Readout: T.Tensor((sizes[14], ), 'float16'),
        Head: T.Tensor((sizes[15], ), 'float32')
    ):
        with T.Kernel(gx, gy, threads=32) as (bx, by):
            T.import_source(HEADER)
            lane = T.get_thread_binding(0)
            raw = T.alloc_local((32, ), 'uint32')
            ff = T.alloc_local((32, ), 'uint32')
            aa = T.alloc_local((16, ), 'uint32')
            bb = T.alloc_local((2, ), 'uint32')
            z = T.alloc_local((16, ), 'uint32')
            mp = T.alloc_local((4, ), 'uint32')
            sk = T.alloc_local((4, ), 'uint32')
            pool_a = T.alloc_local((4, ), 'uint32')
            packet = T.alloc_shared((64, 16), 'float16')
            rail = T.alloc_shared((64, 32), 'float16')
            for j in T.unroll(32):
                raw[j] = T.uint32(0)
            for j in T.unroll(4):
                pool_a[j] = T.uint32(0)
            if kind == 'block0':
                for i in T.unroll(2):
                    for c in T.unroll(16):
                        t = lane + 32 * i
                        v = Features[(c * H0 + by * 8 + t // 8) * W0_ + bx * 8 + t % 8]
                        hv = T.Cast('float16', v)
                        if T.isinf(v) | T.isnan(v) | T.isinf(T.Cast('float32', hv)):
                            T.evaluate(call('status_or', T.address_of(Status[0]), 1, dtype='int32'))
                        packet[t, c] = hv
                T.evaluate(call('sync', dtype='int32'))
                for m in T.unroll(4):
                    for i in T.unroll(4):
                        r = m * 16 + lane // 4 + (i % 2) * 8
                        pix = (r & 3) ^ ((r & 12) << 1) ^ ((r & 16) << 1) ^ (r &
                                                                             32) ^ ((r & 32) >> 3)
                        k = (lane & 3) * 2 + (i // 2) * 8
                        aa[i] = pack(packet[pix, k], packet[pix, k + 1])
                    for n in T.unroll(4):
                        for i in T.unroll(2):
                            k, c = (lane & 3) * 2 + i * 8, nc(n * 8 + lane // 4)
                            bb[i] = pack(Adapter[k * 32 + c], Adapter[(k + 1) * 32 + c])
                        T.evaluate(
                            call(
                                'mma16',
                                T.address_of(raw[(m * 4 + n) * 2]),
                                aa[0],
                                aa[1],
                                aa[2],
                                aa[3],
                                bb[0],
                                bb[1],
                                dtype='int32'
                            )
                        )
                true_half(aa, ff, raw, G0)
            elif kind == 'post':
                for m in T.unroll(4):
                    dx, dy = T.Cast('int32',
                                    (m == 2) | (m == 3)), T.Cast('int32', (m == 1) | (m == 2))
                    r = lane // 4
                    x, y = bx * 8 - 4 + dx * 4 + (r & 3), by * 8 - 4 + dy * 4 + (r >> 2)
                    valid = (x >= 0) & (x < W0_) & (y >= 0) & (y < H0)
                    sa = ((by * 2 - 1 + dy) * (W0_ // 4) + bx * 2 - 1 +
                          dx) * 512 + (r & 3) * 64 + (r >> 2) * 256 + (lane & 3) * 16
                    load_packet(sk, Y, T.if_then_else(valid, sa, -1))
                    for j in T.unroll(4):
                        mp[j] = T.uint32(0)
                    if valid & ((lane & 20) == 0):
                        sx, sy = bx * 4 - 2 + dx * 2 + ((lane >> 3) & 1), by * 4 - 2 + dy * 2 + (
                            (lane >> 1) & 1)
                        l = (lane & 1) * 1024 + ((sy & 7) * 8 + (sx & 7)) * 16
                        ma = (
                            ((sy >> 3) + (H0 // 16) * (l >> 10)) * (W0_ // 8) + (sx >> 5) +
                            (W0_ // 64) * ((l >> 7) & 7)
                        ) * 512 + ((sx >> 3) & 3) * 128 + (l & 127)
                        T.evaluate(mem('ld128', T.address_of(mp[0]), T.address_of(X[ma])))
                    for p in T.unroll(2):
                        for i in T.unroll(2):
                            src = (lane & 8) + p + 2 * i
                            s0, s1, s2, s3 = shfl(mp[0],
                                                  src), shfl(mp[1],
                                                             src), shfl(mp[2],
                                                                        src), shfl(mp[3], src)
                            mv = T.if_then_else(
                                lane % 4 == 0,
                                s0,
                                T.if_then_else(
                                    lane % 4 == 1, s1, T.if_then_else(lane % 4 == 2, s2, s3)
                                )
                            )
                            if valid:
                                for j in T.unroll(2):
                                    c, sv = (lane & 3) * 8 + p * 4 + j * 2, sk[p * 2 + i]
                                    raw[m * 8 + p * 4 + j * 2 + i] = pack(
                                        hf(
                                            une4(sv >> (j * 16)),
                                            Adapter[c],
                                            hm(une4(mv >> (j * 16)), Gate[c])
                                        ),
                                        hf(
                                            une4(sv >> (j * 16 + 8)),
                                            Adapter[c + 1],
                                            hm(une4(mv >> (j * 16 + 8)), Gate[c + 1])
                                        )
                                    )
                true_half(aa, ff, raw, G0)
            elif kind == 'up':
                for j in T.unroll(8):
                    z[j] = T.uint32(0)
                for kp in T.serial(2):
                    for i in T.unroll(4):
                        r = bx * 16 + lane // 4 + (i & 1) * 8
                        c = kp * 32 + (i // 2) * 16
                        for j in T.unroll(4):
                            mp[j] = T.uint32(0)
                        if (lane & 3) == 0:
                            T.evaluate(
                                mem(
                                    'ld128',
                                    T.address_of(mp[0]),
                                    T.address_of(X[in_offset + up_address(r, c, H, W)])
                                )
                            )
                        s0, s1, s2, s3 = shfl(mp[0], lane & 28), shfl(mp[1], lane & 28), shfl(
                            mp[2], lane & 28), shfl(mp[3], lane & 28)
                        aa[i] = T.if_then_else(
                            lane % 4 == 0,
                            s0,
                            T.if_then_else(
                                lane % 4 == 1, s1, T.if_then_else(lane % 4 == 2, s2, s3)
                            )
                        )
                    paired(z, aa, Wt0, 0x2000 + kp * 1024, 16, 32)
                for m in T.unroll(4):
                    base = address(m * 16 + lane // 4, (lane & 3) * 8, bx, 151, H, W)
                    load_packet(sk, X, T.if_then_else(base >= 0, skip_offset + base, -1))
                    for p in T.unroll(2):
                        for i in T.unroll(2):
                            for j in T.unroll(2):
                                c = (lane & 3) * 8 + p * 4 + j * 2
                                origin = T.if_then_else(
                                    m == 0,
                                    0,
                                    T.if_then_else(m == 1, 4, T.if_then_else(m == 2, 20, 16))
                                )
                                proj = shfl(z[p * 4 + j * 2 + i], (lane & 11) + origin)
                                if base >= 0:
                                    word = sk[p * 2 + i]
                                    raw[m * 8 + p * 4 + j * 2 + i] = pack(
                                        hf(une4(word >> (j * 16)), Gate[c], unpack(proj, 0)),
                                        hf(
                                            une4(word >> (j * 16 + 8)),
                                            Gate[c + 1],
                                            unpack(proj, 1)
                                        )
                                    )
                    if has_mixed & (base >= 0):
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(Mixed[base]),
                                raw[m * 8],
                                raw[m * 8 + 2],
                                raw[m * 8 + 1],
                                raw[m * 8 + 3]
                            )
                        )
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(Mixed[base + 8]),
                                raw[m * 8 + 4],
                                raw[m * 8 + 6],
                                raw[m * 8 + 5],
                                raw[m * 8 + 7]
                            )
                        )
                true_half(aa, ff, raw, G0)
            elif has_mixed:
                for m in T.unroll(4):
                    for n in T.unroll(4):
                        for i in T.unroll(2):
                            d = address(
                                m * 16 + lane // 4 + i * 8,
                                nc(n * 8 + (lane & 3) * 2),
                                bx,
                                seq,
                                H,
                                W
                            )
                            if d >= 0:
                                raw[(m * 4 + n) * 2 + i] = pack(Mixed[d], Mixed[d + 1])
                true_half(aa, ff, raw, G0)
            else:
                for m in T.unroll(4):
                    if seq == 3:
                        base = address(m * 16 + lane // 2, (lane & 1) * 4, bx, seq, H, W)
                    else:
                        base = address(m * 16 + lane // 4, (lane & 3) * 8, bx, seq, H, W)
                    load_packet(mp, X, T.if_then_else(base >= 0, in_offset + base, -1))
                    for p in T.unroll(2):
                        for i in T.unroll(2):
                            if seq == 3:
                                src = (lane // 4 + i * 8) * 2 + p
                                s0, s1, s2, s3 = shfl(mp[0],
                                                      src), shfl(mp[1],
                                                                 src), shfl(mp[2],
                                                                            src), shfl(mp[3], src)
                                word = T.if_then_else(
                                    lane % 4 == 0,
                                    s0,
                                    T.if_then_else(
                                        lane % 4 == 1, s1, T.if_then_else(lane % 4 == 2, s2, s3)
                                    )
                                )
                            else:
                                word = mp[p * 2 + i]
                            raw_word(aa, ff, word, G0, m, p, i)
            body(
                aa,
                ff,
                Wt0,
                Wt1,
                Wq,
                Wp,
                G1,
                T.reinterpret(T.uint16(scale_bits), 'float16'),
                pool_a,
                rail,
                X,
                Pre,
                Readout,
                Head,
                Status,
                kind,
                seq,
                H0,
                W0_,
                bx,
                by,
                out_offset,
                has_pre
            )
            if kind == 'block0':
                T.evaluate(call('sync', dtype='int32'))
                p, kh = lane // 2, lane % 2 * 16
                py, px = p // 4 * 2, p % 4 * 2
                r0, r1 = pool_local(py * 8 + px), pool_local(py * 8 + px + 1)
                r2, r3 = pool_local((py + 1) * 8 + px), pool_local((py + 1) * 8 + px + 1)
                y, x = by * 4 + p // 4, bx * 4 + p % 4
                t = (y & 7) * 8 + (x & 7)
                slot = ((t & 3) << 1) ^ ((t & 4) << 2) ^ ((t & 4) << 3) ^ (t & 8) ^ (
                    (t & 16) >> 4
                ) ^ ((t & 32) >> 1)
                base = compact(y // 8 * 8 + slot // 8, x // 8 * 8 + slot % 8, kh // 16 * 4, H, W)
                for j in T.unroll(4):
                    mp[j] = T.uint32(0)
                    for bit in T.unroll(4):
                        k = kh + j * 4 + bit
                        phys = k // 4 % 4 * 8 + k // 16 * 4 + k % 4
                        cc = nc(phys)
                        avg = hm(
                            ha(ha(rail[r0, cc], rail[r1, cc]), ha(rail[r2, cc], rail[r3, cc])),
                            T.float16(.25)
                        )
                        mp[j] = mp[j] | (call('e4', avg) << (bit * 8))
                T.evaluate(mem('st128', T.address_of(Y[base]), mp[0], mp[1], mp[2], mp[3]))
            if kind == 'ds':
                for j in T.unroll(16):
                    z[j] = T.uint32(0)
                paired(z, pool_a, Wt0, 0x50b0, 16, 64)
                for p in T.unroll(4):
                    lo, hi = e4four(z[p * 4], z[p * 4 + 2]), e4four(z[p * 4 + 1], z[p * 4 + 3])
                    for k in T.unroll(4):
                        l, h = shfl(lo, (lane % 8) * 4 + k), shfl(hi, (lane % 8) * 4 + k)
                        mp[k] = T.if_then_else(lane < 8, l, h)
                    by_ = bx // (W // 8)
                    if (lane < 16) & (((by_ != 0) & (by_ != H // 8)) | (lane < 8)):
                        row = (T.if_then_else(by_ == 0, 0, by_ * 4 - 2) +
                               lane // 4) * (W // 2) + (bx % (W // 8)) * 4 + lane % 4
                        d = row * 16 + p * (H // 2) * (W // 2) * 16
                        T.evaluate(
                            mem(
                                'st128',
                                T.address_of(X[pool_offset + d]),
                                mp[0],
                                mp[1],
                                mp[2],
                                mp[3]
                            )
                        )
            if counter >= 0:
                if lane == 0:
                    T.evaluate(mem('st32', T.address_of(X[counter + bx * 4]), T.uint32(0)))

    return main


def build_step(spec):
    if not supports(spec.name):
        raise NotImplementedError(spec.name)
    name = spec.name
    kind = 'block0' if name == 'block0_native_packet' else 'post' if name == 'post70_native_head' else 'ds' if name.endswith(
        '_ds'
    ) else 'up' if name.endswith('_up') else 'ordinary'
    seq = 6 if kind == 'ds' else 151 if kind == 'up' else int(
        name.rsplit('_', 1)[1]
    ) if kind == 'ordinary' else 0
    wi = 0 if kind in ('block0', 'post') else 1
    args = [spec.tensor(wi, f, torch.uint32) for f in ('expand', 'contract', 'qkv', 'projection')]
    args += [spec.tensor(wi, f, torch.float16) for f in ('ffn_gate', 'attn_gate')]
    half_dummy = args[4]
    byte_dummy, float_dummy, int_dummy = half_dummy.view(torch.uint8), args[0].view(
        torch.float32), args[0].view(torch.int32)
    in_offset = out_offset = pool_offset = skip_offset = 0
    counter, has_mixed, has_pre = -1, False, False
    if kind == 'block0':
        args += [
            spec.tensor(3),
            spec.tensor(4),
            None,
            spec.tensor(2, dtype=torch.float16),
            spec.tensor(5, dtype=torch.int32),
            half_dummy,
            half_dummy,
            half_dummy,
            half_dummy,
            float_dummy
        ]
    elif kind == 'post':
        args += [
            spec.tensor(1),
            spec.tensor(2),
            float_dummy,
            spec.tensor(4, dtype=torch.float16),
            spec.tensor(7, dtype=torch.int32),
            half_dummy,
            half_dummy,
            spec.tensor(3, dtype=torch.float16),
            spec.tensor(5, dtype=torch.float16),
            spec.tensor(6, dtype=torch.float32)
        ]
    else:
        in_offset, out_offset, counter = (spec.scalar(i) for i in (4, 5, 6))
        has_mixed, has_pre = bool(spec.scalar(7)), bool(spec.scalar(8))
        if kind == 'ds':
            pool_offset = spec.scalar(13)
        if kind == 'up':
            skip_offset = spec.scalar(15)
        args += [
            spec.tensor(0),
            byte_dummy,
            float_dummy,
            half_dummy,
            int_dummy,
            spec.tensor(7, dtype=torch.float16) if has_mixed else half_dummy,
            spec.tensor(8, dtype=torch.float16) if has_pre else half_dummy,
            spec.tensor(14, dtype=torch.float16) if kind == 'up' else half_dummy,
            half_dummy,
            float_dummy
        ]
    H, W = int(spec.defines['NR_H']), int(spec.defines['NR_W'])
    sizes = tuple(H * W * 16 if a is None else a.numel() for a in args)
    with private_compile():
        compiled = kernel(
            kind,
            seq,
            H,
            W,
            int(spec.grid[0]),
            int(spec.grid[1]),
            sizes,
            in_offset,
            out_offset,
            counter,
            pool_offset,
            skip_offset,
            has_mixed,
            has_pre,
            spec.scalar(wi, 'scale')
        )

    def launch():
        if kind == 'block0':
            args[8] = spec.tensor(1, dtype=torch.float32)
        compiled(*args)

    launch.kernel = compiled
    launch.kernels = [compiled]
    launch.name = name
    launch.kernel_launches = 1
    launch.workspace_bytes = 0
    launch.organization = 'shallow-joint-M32-consume'
    return launch


from tilelang_nr.runtime import ShallowJointTileLangNR
