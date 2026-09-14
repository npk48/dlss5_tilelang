"""Deep packed-Half operations, physical layouts and completion ABI. Default stream only."""
import torch
import tilelang
import tilelang.language as T
from runtime.fp8_compiler import private_compile
from runtime.device import TARGET, EXECUTION_BACKEND
from tilelang_nr.instructions import instruction_source

_HEADER = instruction_source("deep_primitives.h")
_CONFIG = {
    'tl.enable_fast_math': False,
    'tl.disable_thread_storage_sync': True,
    'tl.disable_loop_unswitching': True
}


def _jit(*, dynamic):
    from tilelang_nr.common.runtime_jit import spatial_jit
    return spatial_jit(
        dynamic=dynamic, target=TARGET, execution_backend=EXECUTION_BACKEND, pass_configs=_CONFIG
    )


@T.macro
def add(a, b):
    return T.cast(
        T.call_extern('float32', 'nr_add', T.cast(a, 'float32'), T.cast(b, 'float32')), 'float16'
    )


@T.macro
def mul(a, b):
    return T.cast(
        T.call_extern('float32', 'nr_mul', T.cast(a, 'float32'), T.cast(b, 'float32')), 'float16'
    )


@T.macro
def fma(a, b, c):
    return T.cast(
        T.call_extern(
            'float32', 'nr_fma', T.cast(a, 'float32'), T.cast(b, 'float32'), T.cast(c, 'float32')
        ),
        'float16'
    )


@T.macro
def dec(a):
    return T.cast(T.call_extern('float32', 'nr_dec', a), 'float16')


@T.macro
def enc(a):
    return T.call_extern('uint8', 'nr_enc', T.cast(a, 'float32'))


@T.macro
def hb(a):
    return T.call_extern('uint32', 'nr_hbits', T.cast(a, 'float32'))


@T.macro
def hh(a):
    return T.cast(T.call_extern('float32', 'nr_half', a), 'float16')


@T.macro
def dup(a):
    return T.call_extern('uint32', 'nr_dup', T.float32(a))


@T.macro
def add2(a, b):
    return T.call_extern('uint32', 'nr_add2', a, b)


@T.macro
def mul2(a, b):
    return T.call_extern('uint32', 'nr_mul2', a, b)


@T.macro
def fma2(a, b, c):
    return T.call_extern('uint32', 'nr_fma2', a, b, c)


@T.macro
def qp(a):
    return T.call_extern('uint32', 'nr_qpair', a)


@T.macro
def ld(a, i):
    return T.call_extern('uint32', 'nr_ld32', T.access_ptr(a[i], 'r', 4))


@T.macro
def st(a, i, v):
    T.evaluate(T.call_extern('handle', 'nr_st32', T.access_ptr(a[i], 'w', 4), v))


@T.macro
def shfl(a, i):
    return T.call_extern('uint32', 'nr_shuffle', a, i)


@T.macro
def sxor(a, i):
    return T.call_extern('uint32', 'nr_xor', a, i)


@T.macro
def perm(a, b, i):
    return T.call_extern('uint32', 'nr_perm', a, b, T.uint32(i))


@T.macro
def act(a):
    z = T.max(T.min(a, T.float16(4)), T.float16(-4))
    return mul(
        a,
        fma(
            z,
            fma(T.abs(z), T.float16(-.055908203125), T.float16(.447265625)),
            T.float16(.89453125)
        )
    )


@T.macro
def act2(a):
    z = T.call_extern('uint32', 'nr_min2', T.call_extern('uint32', 'nr_max2', a, dup(-4)), dup(4))
    return mul2(
        a,
        fma2(
            z, fma2(z & T.uint32(0x7fff7fff), dup(-.055908203125), dup(.447265625)), dup(.89453125)
        )
    )


@T.macro
def issue(c, r, n, a, b0, b1):
    T.evaluate(
        T.call_extern(
            'handle', 'nr_mma', T.access_ptr(c[r, n, 0], 'rw', 2), a[0], a[1], a[2], a[3], b0, b1
        )
    )


@T.macro
def rawc(p, c, N):
    return (p // 16) * 16 * N + (p & 7) * 64 + ((p >> 3) & 1) * 4 + (c & 1) + ((c & 6) << 3) + (
        (c & 8) >> 2
    ) + ((c & 16) >> 1) + (c >> 5) * 512


@T.macro
def tok(r):
    return ((r >> 3) & 1) + 2 * (r & 3) + 8 * ((r >> 2) & 1)


@T.macro
def loc(r, n):
    return (n & 1) + ((n & 6) << 3) + ((n & 8) >> 2) + ((n & 16) >> 1) + ((n & 32) << 4) + (
        (tok(r) & 1) << 2
    ) + ((tok(r) & 14) << 5)


@T.macro
def rawbase(sx, sy, t, g, H):
    return 8192 * ((H // 4) * (2 * sy + t // 2) + 2 * sx + (t & 1)) + 1024 * g


@T.macro
def akoff(r, k):
    return (r & 7) * 64 + (r // 8) * 4 + (k & 3) + ((k & 12) << 2) + ((k & 16) >> 1)


@_jit(dynamic="size value")
def _completion(size, value):

    @T.prim_func
    def main(O: T.Tensor((size, ), 'int32'), v: T.int32):
        with T.Kernel(T.ceildiv(size, 256), threads=256) as b:
            t = T.get_thread_binding()
            i = b * 256 + t
            if i < size:
                O[i] = v

    return main


@T.macro
def completion(C0, C1, n0, v0, n1, v1, first_block, threads):
    # Only CTA zero owns these independent regions. The next default-stream
    # kernel waits for the entire producer, not these stores as a publication flag.
    if first_block:
        for j in T.serial(T.ceildiv(n0, threads)):
            i = j * threads + T.get_thread_binding()
            if i < n0:
                C0[i] = v0
        for j in T.serial(T.ceildiv(n1, threads)):
            i = j * threads + T.get_thread_binding()
            if i < n1:
                C1[i] = v1


def _view(spec, i, dtype=torch.uint8):
    return spec.tensor(i, dtype=dtype)


def _producer(spec, factory, args, index, fuse_completion):
    regions = []
    if index is not None:
        publication = spec.value(index)
        for region in publication.regions[:publication.count]:
            if region.count:
                view = spec.memory.resolve(region.pointer, torch.int32).narrow(0, 0, region.count)
                regions.append((view, region.count, region.value))
    # Completion has a fixed two-region ABI. Runtime counts disable absent
    # regions; dummy views keep the buffer formals valid without specialization.
    dummy = spec.memory.buffers[0].view(torch.int32).reshape(-1)[:1]
    views = [r[0] for r in regions] if fuse_completion else []
    shape = [(r[1], r[2]) for r in regions] if fuse_completion else []
    views += [dummy] * (2 - len(views))
    shape += [(0, 0)] * (2 - len(shape))
    with private_compile():
        kernel = factory(*args, tuple(shape))
    tails = []
    if not fuse_completion:
        for view, count, value in regions:
            with private_compile():
                tail = _completion(count, value)
            tails.append((tail, view, value))

    def launch(*live, stream=0):
        kernel(*live, *views, stream=stream)
        for tail, view, value in tails:
            tail(view, value, stream=stream)

    launch.kernels = (kernel, *(tail for tail, _, _ in tails))
    return launch


def _finish(kernel, body):
    body.kernels = kernel.kernels
    body.kernel_launches = len(kernel.kernels)
    body.workspace_bytes = 0  # Existing raw/partial/aux storage only.
    return body


@T.macro
def normalize(c, r, co, head, scale, vit):
    if co < 2:
        for h in T.serial(2):
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
            for n in T.serial(4):
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
    for n in T.serial(1, 4):
        s = add2(s, add2(c[r, 2 * n, h], c[r, 2 * n + 1, h]))
    total = T.alloc_var('uint32')
    total = shfl(s, l & ~3)
    for n in T.serial(1, 4):
        total = add2(total, shfl(s, (l & ~3) + n))
    return add(hh(total), hh(total >> 16))
