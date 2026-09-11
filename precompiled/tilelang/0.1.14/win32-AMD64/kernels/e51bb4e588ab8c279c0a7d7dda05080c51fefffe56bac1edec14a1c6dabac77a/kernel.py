
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

def call(kernels, D, MV, Color, Magnitude, P, Rec, Nearest, Dilated, Luma, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        0,
        kernels["main_kernel"],
        CUdevice(Color.device.index)
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
    

    arg_values = Color.data_ptr(), D.data_ptr(), Dilated.data_ptr(), Luma.data_ptr(), MV.data_ptr(), Magnitude.data_ptr(), Nearest.data_ptr(), Rec.data_ptr()
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p

    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")

