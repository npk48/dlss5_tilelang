"""Real TileLang counter reset and ragged pool padding operations."""
from functools import lru_cache

import torch
import tilelang
import tilelang.language as T
from runtime.device import TARGET, CONFIG, EXECUTION_BACKEND
from runtime.fp8_compiler import private_compile
from tilelang_nr.common.runtime_jit import spatial_jit


@spatial_jit(dynamic="n", out_idx=[], target=TARGET, execution_backend=EXECUTION_BACKEND, pass_configs=CONFIG)
def _clear(n):

    @T.prim_func
    def main(counter: T.Tensor((n, ), T.int32), status: T.Tensor((1, ), T.int32)):
        with T.Kernel((n + 255) // 256, threads=256) as block:
            i = block * 256 + T.get_thread_binding(0)
            if i < n:
                counter[i] = -1
            if i == 0:
                status[0] = 0

    return main


@spatial_jit(dynamic="arena_bytes offset height width pool_height pool_width", out_idx=[], target=TARGET, execution_backend=EXECUTION_BACKEND, pass_configs=CONFIG)
def _padding(arena_bytes, offset, height, width, pool_height, pool_width):
    count = pool_height * pool_width * 512

    @T.prim_func
    def main(arena: T.Tensor((arena_bytes, ), T.uint8), start: T.int32):
        with T.Kernel((count + 255) // 256, threads=256) as block:
            i = block * 256 + T.get_thread_binding(0)
            if i < count:
                pixel = (i % (pool_height * pool_width * 16)) // 16
                y, x = pixel // pool_width, pixel % pool_width
                if y * 2 >= height or x * 2 >= width:
                    arena[start + i] = T.uint8(0)

    return main


def supports(name):
    return name in ("clear_counters", "pool8_padding")


def _bind(kernel, *args):

    def run():
        kernel(*args)

    run.kernel_launches = 1
    run.workspace_bytes = 0
    return run


def build_step(spec):
    if spec.name == "clear_counters":
        n = spec.scalar(1)
        counters = spec.tensor(0, dtype=torch.int32)[:n]
        status = spec.tensor(2, dtype=torch.int32)[:1]
        with private_compile():
            kernel = _clear(n)
        return _bind(kernel, counters, status)
    if spec.name == "pool8_padding":
        arena = spec.tensor(0)
        offset, height, width, pool_height, pool_width = [spec.scalar(i) for i in range(1, 6)]
        # Always bind the runtime padding kernel, including fully covered pools.
        # A later ragged geometry must not introduce a new compilation.
        with private_compile():
            kernel = _padding(arena.numel(), offset, height, width, pool_height, pool_width)
        return _bind(kernel, arena, offset)
    raise NotImplementedError(spec.name)
