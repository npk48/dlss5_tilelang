"""1H physical addressing and packed-Half nonlinearities shared by both organizations."""
import tilelang.language as T
from tilelang_nr.instructions import instruction_source
from tilelang_nr.common.shallow_ops import call, pack, unpack, splat, add, mul, fma, e4four, shfl, une4, hm, ha, hf

HEADER = instruction_source("shallow_primitives.cuh")
NAMES = frozenset(
    {
        "block0_native_packet",
        "post70_native_head",
        "outer1_static_ds",
        "outer1_static_up",
        *(f"outer1_static_{i}" for i in (3, 4, 5, 152, 153, 154))
    }
)


def supports(name):
    return name in NAMES


def nc(c):
    return c % 8 // 2 * 8 + c // 8 * 2 + c % 2


def logical(m):
    return T.if_then_else(m == 0, 0, T.if_then_else(m == 1, 3, m - 1))


def load32(buf, a):
    return T.Cast('uint32', buf[a]) | (T.Cast('uint32', buf[
        a + 1]) << 8) | (T.Cast('uint32', buf[a + 2]) << 16) | (T.Cast('uint32', buf[a + 3]) << 24)


def compact(y, x, c, H, W):
    t = (y % 8) * 8 + x % 8
    t = ((t & 1) << 4) ^ ((t & 6) >> 1) ^ (t & 8) ^ ((t & 16) << 1) ^ (t & 32) ^ ((t & 32) >> 3)
    k = (c & 3) | ((c & 4) << 2) | ((c & 24) >> 1)
    l = (k >> 4) * 1024 + t * 16 + (k & 15)
    return (((y >> 3) + (H // 8) * (l >> 10)) * (W // 4) + (x >> 5) + (W // 32) *
            ((l >> 7) & 7)) * 512 + ((x >> 3) & 3) * 128 + (l & 127)


def address(row, c, cta, seq, H, W, output=False):
    phase = seq - 151 if seq >= 151 else seq - 3
    gx = W // 8 + int(phase in (1, 2))
    stripe = row >> 4
    cy = cta // gx * 2 - int(phase in (1, 3)) + T.Cast('int32', (stripe == 1) | (stripe == 2))
    cx = cta % gx * 2 - int(phase in (1, 2)) + T.Cast('int32', (stripe == 2) | (stripe == 3))
    if output:
        c = nc(c)
    if (seq == 3 and not output) or (seq == 154 and output):
        s = T.if_then_else((cy & 1) != 0, 1 + (cx & 1), (cx & 1) * 3)
        y = (cy >> 1) * 8 + s * 2 + ((row >> 2) & 1)
        x = (cx >> 1) * 8 + ((row & 3) << 1) + ((row >> 3) & 1)
        a = compact(y, x, c, H, W)
    else:
        a = (cy *
             (W // 4) + cx) * 512 + (row & 7) * 64 + ((row >> 3) & 1) * 4 + 8 * (c >> 2) + (c & 3)
    return T.if_then_else((cy >= 0) & (cy < H // 4) & (cx >= 0) & (cx < W // 4), a, -1)


def up_address(row, c, H, W):
    low = row & 63
    outer = row >> 6
    y = outer // (W // 32) * 4 + ((low >> 3) & 1) + 2 * (low & 1)
    x = outer % (W // 32) * 16 + (
        (low >> 1) & 1
    ) + 2 * ((low >> 2) & 1) + 4 * ((low >> 4) & 1) + 8 * ((low >> 5) & 1)
    return (y * (W // 2) + x) * 16 + (c >> 4) * (H // 2) * (W // 2) * 16 + (c & 15)


def pool_local(p):
    return (p & 3) ^ ((p & 4) << 2) ^ ((p & 4) << 3) ^ ((p & 56) >> 1)


def activate(x):
    c = call('min', splat(4), call('max', splat(-4), x))
    g = fma(call('abs', c), splat(-.055908203125), splat(.447265625))
    return mul(x, fma(c, g, splat(.89453125)))


def exponent(x):
    v = call(
        'min',
        splat(1.5693359375),
        call('max', splat(1.03125), fma(x, splat(.044921875), splat(1.30078125)))
    )
    return ((((v & T.uint32(65535)) << 5) + T.uint32(32768))
            & T.uint32(65535)) | ((((v >> 16) << 5) + T.uint32(32768)) << 16)
