
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

def call(kernels, R, Source, W, Proj, Gate, Rows, PO, Matrix, UG, UL, tiles, height, width, gx, sx, sy, rn, sn, wn, inp, out, counter, prn, gn, skip, drn, pon, mn, poolout, ugn, uln, planar_rows, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        16384,
        kernels["kernel_kernel"],
        CUdevice(R.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 16384 for kernel kernel_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = tiles
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 256
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 16384
    config.hStream = stream
    

    arg_values = R.data_ptr(), Source.data_ptr(), W.data_ptr(), counter, gx, height, inp, out, rn, sn, sx, sy, tiles, width, wn
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32

    res = cuLaunchKernelEx(config, kernels["kernel_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel kernel_kernel: {res}")

