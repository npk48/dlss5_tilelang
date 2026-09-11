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

extern "C" __global__ void main_kernel(const float* __restrict__ Current, half_t* __restrict__ KV, const float* __restrict__ PE, half_t* __restrict__ Q);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Current, half_t* __restrict__ KV, const float* __restrict__ PE, half_t* __restrict__ Q) {
  float v = 0x0p+0f/*0.000000e+00*/;
  v = 0x0p+0f/*0.000000e+00*/;
  v = Current[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
  half_t value = ((half_t)(v + PE[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 192)]));
  KV[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = value;
  Q[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = value;
}

