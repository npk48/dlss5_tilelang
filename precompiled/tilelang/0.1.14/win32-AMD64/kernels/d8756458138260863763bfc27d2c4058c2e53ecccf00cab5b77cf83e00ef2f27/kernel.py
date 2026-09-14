
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

_function_names = ['kernel_kernel']

def call(kernels, R, W, IM, Proj, rows, rn, wn, imn, pn, inp, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        0,
        kernels["kernel_kernel"],
        CUdevice(IM.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 0 for kernel kernel_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = (rows + 15) // 16
    config.gridDimY = 4
    config.gridDimZ = 1
    config.blockDimX = 32
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    config.hStream = stream
    

    arg_values = IM.data_ptr(), Proj.data_ptr(), R.data_ptr(), W.data_ptr(), imn, inp, pn, rn, rows, wn
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32

    res = cuLaunchKernelEx(config, kernels["kernel_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel kernel_kernel: {res}")

