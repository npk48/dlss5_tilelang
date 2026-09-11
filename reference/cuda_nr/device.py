"""Owned live CUDA storage and ordinary launches in Torch's current primary context."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
P, U, I = C.c_void_p, C.c_uint64, C.c_int32


def checked(code):
    if code:
        raise RuntimeError(f'Native NR CUDA driver error {code}')


class Step:

    def __init__(self, driver, function, grid, block, shared, values):
        self.driver, self.function = driver, function
        self.geometry = (*grid, *block, shared)
        self.values = tuple(values)
        self.argv = (P * len(values))(*(C.addressof(v) for v in values))

    def __call__(self):
        checked(self.driver.launch(self.function, *self.geometry, None, self.argv, None))


class Driver:

    def __init__(self, device):
        self.device = torch.device(device)
        torch.cuda.init()
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.buffers, self.modules, self.functions = [], {}, {}
        self.closed = False
        self.compile_seconds = 0.0
        self.dll = C.WinDLL('nvcuda.dll')

        def bind(name, args):
            fn = getattr(self.dll, name)
            fn.argtypes = args
            fn.restype = C.c_int
            return fn

        self.current = bind('cuCtxGetCurrent', [C.POINTER(P)])
        self.load = bind('cuModuleLoadData', [C.POINTER(P), P])
        self.unload = bind('cuModuleUnload', [P])
        self.get_function = bind('cuModuleGetFunction', [C.POINTER(P), P, C.c_char_p])
        self.launch = bind('cuLaunchKernel', [P] + [C.c_uint] * 7 + [P, C.POINTER(P), P])
        self.context = P()
        with torch.cuda.device(self.device):
            torch.empty(0, device=self.device)
            checked(self.current(C.byref(self.context)))
            if not self.context.value:
                raise RuntimeError('Torch primary CUDA context is not current')
            self.guard()

    def guard(self):
        if self.closed:
            raise RuntimeError('Native NR driver is closed')
        if torch.cuda.current_stream(self.device).cuda_stream != 0:
            raise RuntimeError('Native NR requires CUDA default stream 0')
        current = P()
        checked(self.current(C.byref(current)))
        if current.value != self.context.value:
            raise RuntimeError('Native NR cannot execute in a different CUDA context')

    def tensor(self, size, dtype=torch.uint8):
        value = torch.empty(size, dtype=dtype, device=self.device)
        self.buffers.append(value)
        return value

    def allocate(self, size):
        return self.tensor(size).data_ptr()

    def upload(self, data):
        if isinstance(data, (bytes, bytearray, memoryview)):
            array = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            array = np.ascontiguousarray(data)
        value = torch.from_numpy(array).to(self.device)
        self.buffers.append(value)
        return value.data_ptr()

    def module(self, source, defines=None):
        source = Path(source).resolve()
        defines = defines or {}
        sources = {}

        def visit(path):
            path = path.resolve()
            if path in sources:
                return
            text = path.read_bytes()
            sources[path] = text
            for inc in re.findall(rb'#include\s+"([^"]+)"', text):
                visit(path.parent / inc.decode())

        visit(source)
        options = [
            '--gpu-architecture=sm_89', '--std=c++17', '--device-as-default-execution-space',
            '-I' + str(source.parent)
        ]
        tc = Path(os.environ.get('NATIVE_NR_TOOLCHAIN',
                                 os.environ.get('DLSS5_FP8_TOOLCHAIN',
                                                str(ROOT / '.toolchains/cuda12.8')))).resolve()
        options += ['-I' + str(tc / 'runtime/include'), '-I' + str(tc / 'cccl/include')]
        options += [f'-D{k}={v}' for k, v in sorted(defines.items())]
        key = hashlib.sha256(repr(options).encode() + b''.join(sources.values())).hexdigest()
        if key in self.modules:
            return self.modules[key]
        cache = ROOT / '.cache/cuda_nr'
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (key + '.cubin')
        if not path.exists():
            tick = time.perf_counter()
            image = self._compile(source, options, tc)
            self.compile_seconds += time.perf_counter() - tick
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(image)
            temporary.replace(path)
        image = C.create_string_buffer(path.read_bytes())
        mod = P()
        checked(self.load(C.byref(mod), image))
        self.modules[key] = mod
        return mod

    @staticmethod
    def _compile(source, options, toolchain):
        location = toolchain / 'nvrtc/bin'
        with os.add_dll_directory(str(location)):
            builtins = C.WinDLL(str(location / 'nvrtc-builtins64_128.dll'))
            dll = C.WinDLL(str(location / 'nvrtc64_120_0.dll'))

            def bind(name, args):
                fn = getattr(dll, name)
                fn.argtypes = args
                fn.restype = C.c_int
                return fn

            def ok(code):
                if code:
                    raise RuntimeError(f'Native NR NVRTC error {code}')

            create = bind('nvrtcCreateProgram',
                          [C.POINTER(P), C.c_char_p, C.c_char_p, C.c_int, P, P])
            compile = bind('nvrtcCompileProgram', [P, C.c_int, C.POINTER(C.c_char_p)])
            destroy = bind('nvrtcDestroyProgram', [C.POINTER(P)])
            program = P()
            ok(create(C.byref(program), source.read_bytes(), source.name.encode(), 0, None, None))
            try:
                argv = (C.c_char_p * len(options))(*(x.encode() for x in options))
                status = compile(program, len(options), argv)
                size = C.c_size_t()
                if status:
                    ok(
                        bind('nvrtcGetProgramLogSize', [P, C.POINTER(C.c_size_t)])(program,
                                                                                   C.byref(size)))
                    log = C.create_string_buffer(size.value)
                    ok(bind('nvrtcGetProgramLog', [P, P])(program, log))
                    raise RuntimeError(f'{source}: NVRTC {status}\n{log.value.decode()}')
                ok(bind('nvrtcGetCUBINSize', [P, C.POINTER(C.c_size_t)])(program, C.byref(size)))
                image = C.create_string_buffer(size.value)
                ok(bind('nvrtcGetCUBIN', [P, P])(program, image))
                return image.raw
            finally:
                destroy(C.byref(program))

    def step(self, module, name, grid, block, shared, values):
        key = (module.value, name)
        if key not in self.functions:
            fn = P()
            checked(self.get_function(C.byref(fn), module, name.encode()))
            self.functions[key] = fn
        return Step(self, self.functions[key], grid, block, shared, values)

    def close(self):
        if self.closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            for mod in reversed(list(self.modules.values())):
                checked(self.unload(mod))
        self.modules.clear()
        self.functions.clear()
        self.buffers.clear()
        self.closed = True


# Frozen BIN records and typed weight descriptors.
"""Frozen BIN record ownership. No capture, reference tensor, or teacher input."""
import ctypes as C
import importlib
from pathlib import Path
import struct

import numpy as np


class BodyWeights(C.Structure):
    _fields_ = [(x, U)
                for x in ('expand', 'contract', 'qkv', 'projection', 'bias', 'ffn_gate',
                          'attn_gate', 'cp', 'rc', 'oi', 'ai', 'pk')] + [('scale', C.c_uint16)]


class TwoWeights(C.Structure):
    _fields_ = [(x, U) for x in ('expand', 'reduce', 'tail', 'qkv', 'projection', 'bias',
                                 'ffn_gate', 'attn_gate', 'scale', 'p', 'ip', 'pk')]


class Weights:

    def __init__(self, model, path, driver):
        self.driver = driver
        source = Path(path if path is not None else model.weights_path).resolve()
        if source != Path(model.weights_path).resolve():
            raise ValueError('Native NR weights must match the retained frozen model')
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
            self.pointers[key] = self.driver.upload(self.raw(block, layer))
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
        scale = self.driver.upload(np.frombuffer(raw, '<f4', 2, 0xe0a0 + shift).astype('<f2'))
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
        return self.driver.upload(
            model.encoder.block0.input_adapter_weight.detach().cpu().numpy().T.astype('<f2'))

    def post(self, model):
        readout = np.concatenate([self.hmma(70, off) for off in (0x5130, 0x5330)])
        if np.any(readout[:, 4:]):
            raise ValueError('Nonzero unused post HMMA columns')
        main = model.post.input_scale.detach().cpu().numpy().astype('<f2')
        skip = model.post.input_sin.detach().cpu().numpy().astype('<f2')
        return self.driver.upload(main), self.driver.upload(skip), self.driver.upload(readout)
