
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

def call(kernels, R, Source, RW, SW, WX, WR, WT, WQ, WB, WS, WP, FG, AG, Inv, Proj, Gate, PI, PO, Matrix, tiles, height, width, gx, sx, sy, rn, sn, wns_0, wns_1, wns_2, wns_3, wns_4, wns_5, wns_6, wns_7, wns_8, inp, out, counter, invn, prn, gn, skip, pin, pon, mn, poolout, stream=0):
    
    res = cuKernelSetAttribute(
        CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        12288,
        kernels["kernel_kernel"],
        CUdevice(AG.device.index)
    )[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to 12288 for kernel kernel_kernel: {res}")

    config = CUlaunchConfig()
    config.gridDimX = tiles
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 64
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 12288
    config.hStream = stream
    

    arg_values = AG.data_ptr(), FG.data_ptr(), Matrix.data_ptr(), PI.data_ptr(), PO.data_ptr(), R.data_ptr(), RW.data_ptr(), SW.data_ptr(), WB.data_ptr(), WP.data_ptr(), WQ.data_ptr(), WR.data_ptr(), WS.data_ptr(), WT.data_ptr(), WX.data_ptr(), counter, gx, height, inp, mn, out, pin, pon, poolout, rn, sn, sx, sy, tiles, width, wns_0, wns_1, wns_2, wns_3, wns_4, wns_6, wns_7, wns_8
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32

    res = cuLaunchKernelEx(config, kernels["kernel_kernel"], (arg_values, arg_types), 0)[0]
    if res != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to launch kernel kernel_kernel: {res}")

