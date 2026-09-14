#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(int* __restrict__ counter, int* __restrict__ status, int n);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(int* __restrict__ counter, int* __restrict__ status, int n) {
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < n) {
    counter[((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))] = -1;
  }
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) == 0) {
    status[0] = 0;
  }
}

