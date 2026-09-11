
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

def call(kernels, A, Plan, Weight, Raw, Members, Out, Scale, C0, C1, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        32768,
        kernels["main_kernel"],
        CUdevice(A.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 32768 for kernel main_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = 3
    config.gridDimY = 3
    config.gridDimZ = 4
    config.blockDimX = 128
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 32768
    config.hStream = stream
    

    arg_values = A.data_ptr(), C0.data_ptr(), Members.data_ptr(), Plan.data_ptr(), Raw.data_ptr(), Scale.data_ptr(), Weight.data_ptr()
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p

    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")

