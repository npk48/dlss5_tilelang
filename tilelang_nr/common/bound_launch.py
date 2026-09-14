"""Bind TileLang-generated host launch metadata once per spatial workspace.

GPU code is still compiled exclusively by TileLang. This reuses CUDA argument
storage instead of rebuilding CUDA-python's scalar packing for every Step.
"""
import ctypes
import types
import threading


class _TensorSlot:
    def __init__(self, tensor, marker):
        self.tensor, self.marker = tensor, marker

    def data_ptr(self):
        return self.marker

    def __getattr__(self, name):
        return getattr(self.tensor, name)


class BoundLaunch:
    def __init__(self, adapter, tensors, scalars, stream):
        from cuda.bindings.driver import cuLaunchKernelEx, CUresult
        self.launch = cuLaunchKernelEx
        self.success = CUresult.CUDA_SUCCESS
        self.records = []
        self.lock = threading.Lock()
        markers = {0x10000 + i * 16: i for i in range(len(tensors))}

        def capture(config, kernel, parameters, extra):
            if extra != 0 or not isinstance(parameters, tuple) or len(parameters) != 2:
                raise RuntimeError('Unsupported TileLang host launch parameters')
            values, ctypes_ = parameters
            slots = []
            storage = []
            for position, (value, kind) in enumerate(zip(values, ctypes_)):
                if kind is ctypes.c_void_p:
                    if value not in markers:
                        raise RuntimeError('TileLang host launch used an unbound device pointer')
                    slots.append((position, markers[value]))
                    value = 0
                storage.append(kind(value))
            pointers = (ctypes.c_void_p * len(storage))(*(ctypes.addressof(v) for v in storage))
            self.records.append((config, kernel, storage, pointers, slots))
            return (self.success,)

        call = adapter.pymodule.call
        namespace = dict(call.__globals__)
        namespace['cuLaunchKernelEx'] = capture
        recorded_call = types.FunctionType(call.__code__, namespace, call.__name__, call.__defaults__, call.__closure__)
        proxies = [_TensorSlot(tensor, marker) for tensor, marker in zip(tensors, markers)]
        recorded_call(adapter.kernels, *proxies, *scalars, stream=stream)
        if not self.records:
            raise RuntimeError('TileLang host program did not declare a kernel launch')

    def __call__(self, tensors, stream):
        # New packets may be supplied every call. Metadata/configuration are
        # bound to this workspace, but pointers and caller stream never are.
        pointers = [tensor.data_ptr() for tensor in tensors]
        # CUDA-python can release the GIL during launch. Protect reusable host
        # argument storage until the driver has consumed it.
        with self.lock:
            for config, kernel, storage, arguments, slots in self.records:
                for position, tensor_index in slots:
                    storage[position].value = pointers[tensor_index]
                config.hStream = stream
                result = self.launch(config, kernel, ctypes.addressof(arguments), 0)[0]
                if result != self.success:
                    raise RuntimeError(f'TileLang bound kernel launch failed: {result}')
