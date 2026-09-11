"""Wide Half arithmetic, physical addresses and original scalar packet routes."""
import tilelang.language as T
from runtime.device import TARGET, EXECUTION_BACKEND, CONFIG
from tilelang_nr.instructions import instruction_source

_PRIMITIVES = instruction_source("wide_primitives.cuh")
_JIT = dict(target=TARGET, execution_backend=EXECUTION_BACKEND, pass_configs=CONFIG)


def op(name, *args):
    return T.call_extern('uint32', 'wide_' + name, *args)


def activate(x):
    z = op('min', op('max', x, op('splat', -4.0)), op('splat', 4.0))
    g = op('fma', op('abs', z), op('splat', -0.055908203125), op('splat', 0.447265625))
    return op('mul', x, op('fma', z, g, op('splat', 0.89453125)))


def exponent(x):
    z = op('fma', x, op('splat', 0.044921875), op('splat', 1.30078125))
    z = op('min', op('max', z, op('splat', 1.03125)), op('splat', 1.5693359375))
    return ((z << 5) & T.uint32(0xffe0ffe0)) ^ T.uint32(0x80008000)


def permute(x, bits):
    out = T.int32(0)
    for j, k in enumerate(bits):
        out = out | (((x >> j) & 1) << k)
    return out


_ROUTES = {
    'hidden': (0, 7, 2, 3, 4, 5, 6, 1, 8, 9, 10),
    'q2': (0, 7, 2, 3, 4, 5, 6, 1, 8, 9, 10),
    'k2': (0, 7, 2, 3, 4, 5, 6, 8, 1, 9, 10),
    'v2': (4, 1, 5, 6, 0, 2, 3, 9, 7, 8, 10),
    'p2': (0, 7, 2, 3, 4, 5, 6, 1, 8, 10, 9),
    'q4': (0, 7, 2, 3, 5, 6, 4, 9, 8, 1, 10),
    'k4': (0, 7, 2, 3, 5, 6, 4, 8, 1, 9, 10),
    'v4': (5, 9, 6, 4, 0, 2, 3, 1, 7, 8, 10),
    'p4': (0, 8, 2, 3, 6, 4, 5, 10, 7, 1, 9),
    'q8': (0, 7, 2, 3, 4, 5, 1, 9, 8, 6, 10),
    'k8': (0, 7, 2, 3, 4, 5, 1, 8, 6, 9, 10),
    'v8': (4, 9, 5, 1, 0, 2, 3, 6, 7, 8, 10),
    'p8': (0, 8, 2, 3, 4, 5, 10, 6, 7, 1, 9),
    'publish': (0, 7, 9, 1, 4, 5, 6, 2, 3, 8),
}


def route_k(k):
    return (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1)


def address48(heads, height, width, gx, sx, sy, flag, tile, slot, n):
    x = (tile % gx) * 8 + (((slot >> 1) & 7) if heads == 4 else (slot & 7)) + sx
    y = (tile //
         gx) * 8 + ((((slot >> 4) & 3) * 2 + (slot & 1)) if heads == 4 else (slot // 8)) + sy
    if flag:
        nc = (n & 1) | ((n & 6) << 1) | ((n & 8) >> 2)
        addr = (y * width + x) * 16 + (n // 16) * height * width * 16 + nc
    else:
        nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4)
        st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3)
        if heads == 4:
            addr = nc + ((st & 1) << 2) + ((st & 14) <<
                                           5) + (x // 4 + (width // 4) * (y // 4)) * 2048
        else:
            st = st | ((x & 4) << 2)
            addr = nc + ((st & 1) << 2) + ((st & 14) << 5) + (
                (st & 16) << 8
            ) + (x // 8 + ((width + 7) // 8) * (y // 4)) * 8192
    return T.if_then_else((x >= 0) & (x < width) & (y >= 0) & (y < height), addr, -1)


def label2(t, c):
    return (c & 3) | ((c & 12) << 2) | ((c & 16) >> 1) | ((c & 32) << 4) | ((t & 7) << 6) | (
        (t & 8) >> 1
    ) | ((t & 48) << 6)


def perm2(c):
    return (c & 49) | ((c & 6) << 1) | ((c & 8) >> 2)


def address2(height, width, gx, sx, sy, seq, tile, t, c, output=False):
    if seq == 7:
        z = (t & 1) | ((t & 2) << 3) | ((t & 4) >> 1) | ((t & 8) >> 1) | ((t & 16) <<
                                                                          1) | ((t & 32) >> 2)
        pixel = (z & 1) | ((z & 14) << 2) | ((z >> 3) & 6)
        row = (tile // gx * 8 + pixel // 8) * width + tile % gx * 8 + pixel % 8
        a = row * 16 + (c // 16) * height * width * 16 + c % 16
    else:
        l = label2(t, c)
        y = (8 * (tile // gx) + sy) // 4 + l // 2048
        x = (8 * (tile % gx) + sx) // 4 + (l % 2048) // 1024
        a = T.if_then_else(
            (y >= 0) & (x >= 0) & (y < height // 4) & (x < width // 4),
            (y * (width // 4) + x) * 1024 + ((l // 512) & 1) * 512 + (l & 511),
            -1
        )
    if output:
        if seq == 7:
            plane = a // (height * width * 16)
            z = a % (height * width * 16)
            y = z // (width * 16)
            x = (z // 16) % width
            k = z % 16
            a = ((y // 4) * (width // 4) + x // 4) * 1024 + (k & 3) + (
                (x % 4 * 4 + k // 4) << 4
            ) + (((y >> 1) & 1) << 2) + ((plane & 1) << 3) + ((y & 1) << 8) + ((plane >> 1) << 9)
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
            low = token & 63
            outer = token >> 6
            row = bind((outer // (width // 16)) * 4 + ((low >> 3) & 1) + 2 * (low & 1))
            col = bind(
                (outer % (width // 16)) * 16 + ((low >> 1) & 1) + 2 * ((low >> 2) & 1) + 4 *
                ((low >> 4) & 1) + 8 * ((low >> 5) & 1)
            )
            a = T.if_then_else(
                a0 >= 0, (row * width + col) * 16 + (n // 16) * height * width * 16 + (n & 1) +
                ((n & 6) << 1) + ((n & 8) >> 2),
                -1
            )
    return a


def local_route_word(P, word, lane, bits):
    """Shuffle uniform source banks BEFORE selecting lane-dependent banks."""
    lane_word_bits = tuple(k - 7 for j, k in enumerate(bits) if 2 <= j < 7 and k >= 7)
    value = T.uint32(0)
    for byte in range(4):
        ix = permute(word * 128 + lane * 4 + byte, bits)
        base = permute(word * 128 + byte, bits) // 128
        if lane_word_bits:
            assert len(lane_word_bits) == 1
            # T.if_then_else would inline the shuffle into divergent C++ ?:.
            # Both bank shuffles must execute in every lane before selection.
            selected = op(
                'shfl_pair',
                P[base],
                P[base | (1 << lane_word_bits[0])], (ix // 4) % 32,
                ix // 128 != base
            )
        else:
            selected = op('shfl', P[base], (ix // 4) % 32)
        value = value | (((selected >> ((ix % 4) * 8)) & 255) << (byte * 8))
    return value


def shared_packed(S, head, word, lane):
    value = T.uint32(0)
    for byte in range(4):
        ix = permute(word * 128 + lane * 4 + byte, _ROUTES['hidden'])
        sw = ix // 128
        raw = T.reinterpret(S[head, (sw // 4) * 128 + ((ix // 4) % 32) * 4 + sw % 4], 'uint32')
        value = value | (((raw >> ((ix % 4) * 8)) & 255) << (byte * 8))
    return value


@T.macro
def _f_zero(C, ns):
    for m in T.unroll(4):
        for n in T.unroll(ns):
            C[m, n, 0] = 0
            C[m, n, 1] = 0


@T.macro
def _f_initial(P, W, C, head, lane, off):
    for m in T.unroll(4):
        for n in T.unroll(4):
            g = W[off // 4 + head * 16 + n * 4 + lane % 4]
            C[m, n, 0] = op('mul', op('decode', P[m * 4 + n]), g)
            C[m, n, 1] = op('mul', op('decode', P[m * 4 + n] >> 16), g)
