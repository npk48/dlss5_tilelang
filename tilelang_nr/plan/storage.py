"""Owned CUDA storage, launch arguments and frozen weight descriptors.

This is the allocation and bookkeeping half of the learned NR: arena layout,
auxiliary rails, weight uploads and per-step launch geometry/arguments. It never
compiles or binds a CUDA kernel; every step body is a TileLang kernel.
"""
import ctypes as C
import importlib
from pathlib import Path
import struct

import numpy as np
import torch

P, U, I = C.c_void_p, C.c_uint64, C.c_int32


class Unit:
    """One logical plan unit; carries its shape defines for the step specs."""

    def __init__(self, name, defines=None):
        self.name = name
        self.defines = dict(defines or {})


class PlanStep:
    """One NR launch: unit, name, geometry and typed arguments.

    ``values`` keeps the original ctypes argument objects so the TileLang
    factories can resolve pointers and scalars exactly as before.
    """

    def __init__(self, unit, name, grid, block, shared, values):
        self.unit = unit
        self.name = name
        self.geometry = (*grid, *block, shared)
        self.values = tuple(values)


class Storage:
    """Owned live storage in Torch's current primary CUDA context."""

    def __init__(self, device):
        self.device = torch.device(device)
        torch.cuda.init()
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.buffers = []
        self.closed = False
        self.compile_seconds = 0.0
        with torch.cuda.device(self.device):
            torch.empty(0, device=self.device)
            self.guard()

    def guard(self):
        if self.closed:
            raise RuntimeError('NR plan storage is closed')
        if torch.cuda.current_stream(self.device).cuda_stream != 0:
            raise RuntimeError('The TileLang NR plan requires CUDA default stream 0')

    def tensor(self, size, dtype=torch.uint8):
        value = torch.empty(size, dtype=dtype, device=self.device)
        self.buffers.append(value)
        return value

    def upload(self, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            array = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            array = np.ascontiguousarray(data)
        value = torch.from_numpy(array).to(self.device)
        self.buffers.append(value)
        return value.data_ptr()

    def unit(self, name, defines=None):
        return Unit(name, defines)

    def step(self, unit, name, grid, block, shared, values):
        return PlanStep(unit, name, grid, block, shared, values)

    def close(self):
        if self.closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
        self.buffers.clear()
        self.closed = True


# Frozen BIN records and typed weight descriptors.
class BodyWeights(C.Structure):
    _fields_ = [(x, U)
                for x in ('expand', 'contract', 'qkv', 'projection', 'bias', 'ffn_gate',
                          'attn_gate', 'cp', 'rc', 'oi', 'ai', 'pk')] + [('scale', C.c_uint16)]


class TwoWeights(C.Structure):
    _fields_ = [(x, U) for x in ('expand', 'reduce', 'tail', 'qkv', 'projection', 'bias',
                                 'ffn_gate', 'attn_gate', 'scale', 'p', 'ip', 'pk')]


class Weights:
    """Frozen BIN record ownership. No capture, reference tensor, or teacher input."""

    def __init__(self, model, path, storage):
        self.storage = storage
        source = Path(path if path is not None else model.weights_path).resolve()
        if source != Path(model.weights_path).resolve():
            raise ValueError('NR weights must match the retained frozen model')
        module = importlib.import_module(type(model).__module__)
        blob, records = module._parse_original_weights_bin(source)
        self.records = {
            str(r['name']):
            bytes(blob[int(r['payload_offset']):int(r['payload_offset']) + int(r['payload_size'])])
            for r in records
        }
        self.pointers, self.bodies = {}, {}

    def raw(self, block, layer=0):
        return self.records[f'block{block}.layer{layer}.layer']

    def ptr(self, block, layer=0):
        key = (block, layer)
        if key not in self.pointers:
            self.pointers[key] = self.storage.upload(self.raw(block, layer))
        return self.pointers[key]

    def one(self, block):
        if block in self.bodies:
            return self.bodies[block]
        raw, p = self.raw(block), self.ptr(block)
        shift = {0: 0x400, 66: 0x840, 70: 0x70}.get(block, 0)
        gate = {0: 0x2410, 66: 0x2810}.get(block, 0x2010)
        scale = int(
            np.asarray(struct.unpack_from('<f', raw, 0x4c60 + shift)[0], dtype='<f2').view('<u2'))
        result = BodyWeights(p, p + 0x1000, p + 0x2060 + shift, p + 0x4c70 + shift,
                             p + 0x2c60 + shift, p + gate, p + 0x5070 + shift, 0, 0, 0, 0, 0, scale)
        self.bodies[block] = result
        return result

    def two(self, block):
        if block in self.bodies:
            return self.bodies[block]
        raw, p = self.raw(block), self.ptr(block)
        shift = 0x2060 if block == 62 else 0
        scale = self.storage.upload(np.frombuffer(raw, '<f4', 2, 0xe0a0 + shift).astype('<f2'))
        result = TwoWeights(p, p + 0x4000, p + 0x6000, p + 0x70a0 + shift, p + 0xe0b0 + shift,
                            p + 0xa0a0 + shift, p + (0x9000 if block == 62 else 0x7010),
                            p + 0xf0b0 + shift, scale, 0, 0, 0)
        self.bodies[block] = result
        return result

    def hmma(self, block, offset, pair=0):
        lanes = np.frombuffer(self.raw(block), '<f2', 256, offset).reshape(32, 8)
        result = np.empty((16, 8), '<f2')
        for lane in range(32):
            for i in range(4):
                result[(lane % 4) * 2 + i % 2 + 8 * (i // 2), lane // 4] = lanes[lane, pair * 4 + i]
        return result

    def input_adapter(self, model):
        # Original raw adapter's canonical columns, exactly as the frozen decoder.
        return self.storage.upload(
            model.encoder.block0.input_adapter_weight.detach().cpu().numpy().T.astype('<f2'))

    def post(self, model):
        readout = np.concatenate([self.hmma(70, off) for off in (0x5130, 0x5330)])
        if np.any(readout[:, 4:]):
            raise ValueError('Nonzero unused post HMMA columns')
        main = model.post.input_scale.detach().cpu().numpy().astype('<f2')
        skip = model.post.input_sin.detach().cpu().numpy().astype('<f2')
        return self.storage.upload(main), self.storage.upload(skip), self.storage.upload(readout)
