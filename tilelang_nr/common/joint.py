"""2H/4H publication routes and fixed-order MMA fragments."""
import tilelang.language as T
from tilelang_nr.common.wide import (
    _JIT, _ROUTES, op, activate, exponent, permute, address2, address48, route_k, perm2, label2
)
from tilelang_nr.common.packet import local_route_word, shared_packed, _prmt
from tilelang_nr.instructions import instruction_source

_PRIMITIVES = instruction_source("wide_primitives.cuh") + "\n" + instruction_source("joint_primitives.cuh")
PUBLISH2 = (0, 7, 2, 3, 4, 5, 6, 1, 8, 9, 10)

PUBLISH150 = (0, 7, 4, 5, 9, 6, 1, 2, 3, 10, 8)


def joint_address2(height, width, gx, sx, sy, seq, tile, t, c, output=False):
    bind = lambda value: T.alloc_var('int32', init=value)[0]
    if seq == 7:
        z = bind(
            (t & 1) | ((t & 2) << 3) | ((t & 4) >> 1) | ((t & 8) >> 1) | ((t & 16) << 1)
            | ((t & 32) >> 2)
        )
        pixel = bind((z & 1) | ((z & 14) << 2) | ((z >> 3) & 6))
        row = bind((tile // gx * 8 + pixel // 8) * width + tile % gx * 8 + pixel % 8)
        a = bind(row * 16 + (c // 16) * height * width * 16 + c % 16)
    else:
        l = bind(label2(t, c))
        y = bind((8 * (tile // gx) + sy) // 4 + l // 2048)
        x = bind((8 * (tile % gx) + sx) // 4 + (l % 2048) // 1024)
        a = bind(
            T.if_then_else(
                (y >= 0) & (x >= 0) & (y < height // 4) & (x < width // 4),
                (y * (width // 4) + x) * 1024 + ((l // 512) & 1) * 512 + (l & 511),
                -1
            )
        )
    if output:
        if seq == 7:
            plane = bind(a // (height * width * 16))
            z = bind(a % (height * width * 16))
            y = bind(z // (width * 16))
            x = bind((z // 16) % width)
            k = bind(z % 16)
            a = bind(
                ((y // 4) * (width // 4) + x // 4) * 1024 + (k & 3) + ((x % 4 * 4 + k // 4) << 4) +
                (((y >> 1) & 1) << 2) + ((plane & 1) << 3) + ((y & 1) << 8) + ((plane >> 1) << 9)
            )
        if seq == 150:
            # Materialize the integer route DAG. Expression-only bindings here
            # duplicate nested addresses during lowering of the unrolled store.
            bind = lambda value: T.alloc_var('int32', init=value)[0]
            a0 = bind(a)
            band = bind(a0 // 512)
            z = bind(a0 % 512)
            c0 = bind(band // ((height // 8) * (width // 4)) * 16 + z % 16)
            rem = bind(band % ((height // 8) * (width // 4)))
            y = bind((rem // (width // 4)) * 8 + ((rem % 2) * 32 + z // 16) // 8)
            x = bind((rem % (width // 4) // 2) * 8 + (z // 16) % 8)
            n = bind(
                (c0 & 1) | ((x & 1) << 1) | (((x >> 1) & 1) << 2) | (((c0 >> 1) & 1) << 3)
                | (((c0 >> 3) & 1) << 4) | (((y >> 2) & 1) << 5)
            )
            low0 = bind(
                ((c0 >> 2) & 1) | (((x >> 2) & 1) << 1) | ((y & 1) << 2)
                | (((y >> 1) & 1) << 3) | (((x >> 3) & 1) << 4)
            )
            token = bind(
                low0 + 32 * (x // 16 + (width // 16) * (y // 8 + (height // 8) * (c0 // 16)))
            )
            low = bind(token & 63)
            outer = bind(token >> 6)
            row = bind((outer // (width // 16)) * 4 + ((low >> 3) & 1) + 2 * (low & 1))
            col = bind(
                (outer % (width // 16)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) + 4 *
                ((low >> 4) & 1) + 8 * ((low >> 5) & 1)
            )
            a = bind(
                T.if_then_else(
                    a0 >= 0, (row * width + col) * 16 + (n // 16) * height * width * 16 + (n & 1) +
                    ((n & 6) << 1) + ((n & 8) >> 2),
                    -1
                )
            )
    return a


def route(P, word, lane, bits):
    # All uniform register banks are shuffled by all 32 lanes, before selection.
    # Publication150 has two lane-dependent word bits, unlike hidden/QKV routes.
    banks = tuple(sorted({k - 7 for j, k in enumerate(bits) if 2 <= j < 7 and k >= 7}))
    value = T.uint32(0)
    for byte in range(4):
        ix = T.alloc_var('int32', init=permute(word * 128 + lane * 4 + byte, bits))[0]
        base = permute(word * 128 + byte, bits) // 128
        selected = T.uint32(0)
        for choice in range(1 << len(banks)):
            bank = base | sum(((choice >> i) & 1) << bit for i, bit in enumerate(banks))
            fetched = T.alloc_var('uint32', init=op('shfl', P[bank], (ix // 4) % 32))[0]
            selected = T.alloc_var(
                'uint32', init=T.if_then_else(ix // 128 == bank, fetched, selected)
            )[0]
        value = value | (((selected >> ((ix % 4) * 8)) & 255) << (byte * 8))
    return value


@T.macro
def zero(C, ms, ns):
    for m in T.unroll(ms):
        for n in T.unroll(ns):
            C[m, n, 0] = 0
            C[m, n, 1] = 0


@T.macro
def weighted(S, W, C, lane, first, ms, ns, kbegin, kend, off, stride):
    a = T.alloc_local((ms, 4), 'int32')
    b = T.alloc_local((4, ), 'int32')
    for kp in T.serial(kbegin, kend):
        for m in T.unroll(ms):
            for j in T.vectorized(4):
                a[m, j] = S[kp, (first + m) * 128 + lane * 4 + j]
        for pair in T.unroll(ns // 2):
            for j in T.vectorized(4):
                b[j] = W[(off + kp * stride) // 4 + pair * 128 + lane * 4 + j]
            for m in T.unroll(ms):
                for half in T.unroll(2):
                    r = T.call_extern(
                        'uint64',
                        'wide_mma',
                        a[m, 0],
                        a[m, 1],
                        a[m, 2],
                        a[m, 3],
                        b[half * 2],
                        b[half * 2 + 1],
                        C[m, pair * 2 + half, 0],
                        C[m, pair * 2 + half, 1]
                    )
                    C[m, pair * 2 + half, 0] = T.cast(r, 'uint32')
                    C[m, pair * 2 + half, 1] = T.cast(r >> 32, 'uint32')


@T.macro
def finish_row(S, W, C, lane, m, component, kp, off, stride):
    a = T.alloc_local((4, ), 'int32')
    b = T.alloc_local((4, ), 'int32')
    for j in T.vectorized(4):
        a[j] = S[kp, m * 128 + lane * 4 + j]
    for pair in T.unroll(2):
        for j in T.vectorized(4):
            b[j] = W[(off + kp * stride + component * 1024) // 4 + pair * 128 + lane * 4 + j]
        for half in T.unroll(2):
            n = component * 4 + pair * 2 + half
            r = T.call_extern(
                'uint64',
                'wide_mma',
                a[0],
                a[1],
                a[2],
                a[3],
                b[half * 2],
                b[half * 2 + 1],
                C[m, n, 0],
                C[m, n, 1]
            )
            C[m, n, 0] = T.cast(r, 'uint32')
            C[m, n, 1] = T.cast(r >> 32, 'uint32')


@T.macro
def register_weight(A, W, C, lane, ms, ns, off):
    b = T.alloc_local((4, ), 'int32')
    for pair in T.unroll(ns // 2):
        for j in T.vectorized(4):
            b[j] = W[off // 4 + pair * 128 + lane * 4 + j]
        for m in T.unroll(ms):
            for half in T.unroll(2):
                r = T.call_extern(
                    'uint64',
                    'wide_mma',
                    A[m * 4],
                    A[m * 4 + 1],
                    A[m * 4 + 2],
                    A[m * 4 + 3],
                    b[half * 2],
                    b[half * 2 + 1],
                    C[m, pair * 2 + half, 0],
                    C[m, pair * 2 + half, 1]
                )
                C[m, pair * 2 + half, 0] = T.cast(r, 'uint32')
                C[m, pair * 2 + half, 1] = T.cast(r >> 32, 'uint32')


@T.macro
def norm_encode(Z, P, WS, lane, head, heads, sc, m, comp, dest):
    if comp < 2:
        for j in T.unroll(2):
            a = Z[m, comp * 4, j]
            b = Z[m, comp * 4 + 1, j]
            c = Z[m, comp * 4 + 2, j]
            d = Z[m, comp * 4 + 3, j]
            s0 = op('add', op('fma', a, a, op('mul', c, c)), op('fma', b, b, op('mul', d, d)))
            s1 = op('add', s0, op('shfl', s0, lane ^ 2))
            s2 = op('add', s1, op('shfl', s1, lane ^ 1))
            total = op(
                'max', op('add', s2, (s2 >> 16) | (s2 << 16)), op('splat', 0.00006198883056640625)
            )
            inv = op('rsqrt', total)
            for n in T.unroll(4):
                Z[m, comp * 4 + n, j] = op('mul', Z[m, comp * 4 + n, j], inv)
                if comp == 0:
                    if heads == 2:
                        sraw = T.reinterpret(WS[sc // 4 + head // 2], 'uint32')
                        sbits = (sraw >> ((head % 2) * 16)) & 65535
                        Z[m, n, j] = op('mul', Z[m, n, j], sbits | (sbits << 16))
                    else:
                        Z[m, n, j] = op(
                            'mul',
                            Z[m, n, j],
                            op('splat', T.reinterpret(WS[sc // 4 + head], 'float32'))
                        )
    for n in T.unroll(4):
        P[dest * 4 + n] = op('encode', Z[m, comp * 4 + n, 0], Z[m, comp * 4 + n, 1])


_NAMES = frozenset(
    (
        'streamed2',
        'packet2_ds',
        'streamed_project2',
        'four_input',
        'four_ordinary',
        'four_output',
        'packet4_ds',
        'packet4_up'
    )
)


def supports(name):
    return name in _NAMES
