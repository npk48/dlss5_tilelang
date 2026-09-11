
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

def call(kernels, R, Source, RW, SW, WX, WR, WT, WQ, WB, WS, WP, FG, AG, Inv, Proj, Gate, PI, PO, Matrix, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        8192,
        kernels["kernel_kernel"],
        CUdevice(AG.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 8192 for kernel kernel_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = 88
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 128
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 8192
    config.hStream = stream
    

    arg_values = AG.data_ptr(), FG.data_ptr(), Gate.data_ptr(), Inv.data_ptr(), Proj.data_ptr(), RW.data_ptr(), WB.data_ptr(), WP.data_ptr(), WQ.data_ptr(), WR.data_ptr(), WS.data_ptr(), WT.data_ptr(), WX.data_ptr()
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p

    res = cuLaunchKernelEx(config, kernels["kernel_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel kernel_kernel: {res}")

