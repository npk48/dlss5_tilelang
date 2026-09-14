
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

def call(kernels, A, B, M, O, P, G, AH, BH, OH, C0, C1, sizes_0, sizes_1, sizes_2, sizes_3, sizes_4, sizes_5, count, width, poolwidth, n0, v0, n1, v1, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        0,
        kernels["main_kernel"],
        CUdevice(A.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 0 for kernel main_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = (count + 255) // 256
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 256
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = stream
    

    arg_values = A.data_ptr(), C0.data_ptr(), C1.data_ptr(), M.data_ptr(), O.data_ptr(), P.data_ptr(), count, n0, n1, sizes_0, sizes_2, sizes_3, sizes_4, v0, v1
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32

    res = cuLaunchKernelEx(config, kernels["main_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel main_kernel: {res}")

