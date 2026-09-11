
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

def call(kernels, Prepared, Motion, Velocity, DC, Masks, Raw, OldLock, Shade, LH, Params, Resolved, History, Locks, Luma, Up, Weight, Mean, Reactivity, HistOut, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        0,
        kernels["main_kernel"],
        CUdevice(DC.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 0 for kernel main_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = 2188
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 128
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = stream
    

    arg_values = DC.data_ptr(), HistOut.data_ptr(), History.data_ptr(), Locks.data_ptr(), Luma.data_ptr(), Masks.data_ptr(), Mean.data_ptr(), Motion.data_ptr(), Params.data_ptr(), Prepared.data_ptr(), Reactivity.data_ptr(), Resolved.data_ptr(), Shade.data_ptr(), Up.data_ptr(), Velocity.data_ptr(), Weight.data_ptr()
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p

    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")

