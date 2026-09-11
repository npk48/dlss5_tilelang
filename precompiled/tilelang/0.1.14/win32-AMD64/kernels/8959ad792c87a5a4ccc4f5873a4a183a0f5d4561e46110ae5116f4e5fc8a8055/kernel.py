
from cuda.bindings.driver import (
    CUtensorMapDataType,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill,
    cuTensorMapEncodeTiled,
    cuTensorMapEncodeIm2col,
    CUresult,
    cuKernelSetAttribute,
    CUfunction_attribute,
    CUdevice,
    CUlaunchConfig,
    cuLaunchKernelEx,
    cuuint64_t,
    cuuint32_t,
    CUkernel,
    CUlaunchAttribute,
    CUlaunchAttributeID,
)
import ctypes

_function_names = ['main_kernel']

def call(kernels, Wt0, Wt1, Wq, Wp, G0, G1, X, Y, Features, Adapter, Status, Mixed, Pre, Gate, Readout, Head, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        6144,
        kernels["main_kernel"],
        CUdevice(Adapter.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 6144 for kernel main_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = 80
    config.gridDimY = 64
    config.gridDimZ = 1
    config.blockDimX = 32
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 6144
    config.hStream = stream
    

    arg_values = Adapter.data_ptr(), Features.data_ptr(), G0.data_ptr(), G1.data_ptr(), Status.data_ptr(), Wp.data_ptr(), Wq.data_ptr(), Wt0.data_ptr(), Wt1.data_ptr(), X.data_ptr(), Y.data_ptr()
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p

    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")

