"""Typed launch-plan boundary shared by TileLang NR kernel families.

The reference CUDA runtime supplies allocations and argument ownership only.
Factories express actual computation in TileLang and return a zero-argument
callable. They must not call the original CUDA Step or import whole CUDA bodies.
"""
from dataclasses import dataclass
import ctypes as C
from typing import Callable

import torch


class BufferViews:
    """Resolve live ctypes GPU pointers to non-copying Torch storage views."""

    def __init__(self, buffers):
        self.buffers = list(buffers)
        self.external = []
        self._views = {}
        self._input_views = {}
        self._widths = {}

    def bind_input(self, tensor):
        self.external = [tensor]
        # Input storage can be reused with a new Python owner between frames.
        self._input_views.clear()

    def resolve(self, pointer, dtype=torch.uint8):
        pointer = int(getattr(pointer, "value", pointer))
        if not pointer:
            raise ValueError("A consumed buffer pointer is null")
        key = (pointer, dtype)
        is_input = any(
            t.data_ptr() <= pointer < t.data_ptr() + t.numel() * t.element_size()
            for t in self.external
        )
        cache = self._input_views if is_input else self._views
        if key in cache:
            return cache[key]
        for tensor in self.external if is_input else self.buffers:
            base = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if base <= pointer < base + size:
                if not tensor.is_contiguous():
                    raise ValueError("Native NR storage owner must be contiguous")
                offset = pointer - base
                if dtype not in self._widths:
                    self._widths[dtype] = torch.empty((), dtype=dtype).element_size()
                width = self._widths[dtype]
                if offset % width:
                    raise ValueError("Misaligned typed buffer view")
                raw = tensor.view(torch.uint8).reshape(-1)
                length = (size - offset) // width * width
                result = raw.narrow(0, offset, length).view(dtype)
                cache[key] = result
                return result
        raise ValueError(f"Unowned native NR pointer 0x{pointer:x}")


@dataclass
class StepSpec:
    """Original launch geometry and ctypes values, with live pointer binding.

    scalar(index, field) reads a fixed scalar/offset (not tensor contents).
    tensor(index, field, dtype) resolves a pointer, including mutable packet input
    pointers. For input pointers call this in the returned callable, not during
    construction, because the original input pointer is initially null.

    Factories obtain shape constants from defines and argument structs and cache
    a tilelang.jit function. Calling that function must keep stream 0 and all
    original writes (including completion descriptors and auxiliary rails).
    """

    name: str
    source: str
    geometry: tuple
    values: tuple
    defines: dict
    memory: BufferViews

    @property
    def grid(self):
        return self.geometry[:3]

    @property
    def block(self):
        return self.geometry[3:6]

    @property
    def shared_bytes(self):
        return self.geometry[6]

    def value(self, index, field=None):
        value = self.values[index]
        if field is not None:
            for part in field.split("."):
                value = getattr(value, part)
        return value

    def scalar(self, index, field=None):
        value = self.value(index, field)
        return int(getattr(value, "value", value))

    def tensor(self, index, field=None, dtype=torch.uint8):
        return self.memory.resolve(self.scalar(index, field), dtype)


# Family implementations export supports(name) and build_step(spec).
# build_step must return a callable or raise NotImplementedError explicitly.
StepCallable = Callable[[], None]
