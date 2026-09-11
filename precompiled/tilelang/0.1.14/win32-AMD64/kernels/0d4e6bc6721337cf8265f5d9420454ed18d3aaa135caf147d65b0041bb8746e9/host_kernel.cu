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

extern "C" __global__ void main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, const float* __restrict__ Gaussian, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Settings);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, const float* __restrict__ Gaussian, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Settings) {
  Out[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = Gaussian[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)];
  Out[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 327680)] = Gaussian[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 2)];
  Out[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 655360)] = Gaussian[((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3))];
  Out[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 983040)] = 0x1p+0f/*1.000000e+00*/;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    half_t current = ((((half_t)Current[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)]) - half_t(0x1p-1f/*5.000000e-01*/)) * half_t(0x1p-3f/*1.250000e-01*/));
    half_t prev = ((((half_t)Previous[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)]) - half_t(0x1p-1f/*5.000000e-01*/)) * half_t(0x1p-3f/*1.250000e-01*/));
    Out[((((k * 327680) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)) + 1310720)] = ((float)current);
    float condval;
    if ((0x0p+0f/*0.000000e+00*/ < Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))])) {
      condval = ((float)prev);
    } else {
      condval = ((float)current);
    }
    Out[((((k * 327680) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)) + 2293760)] = condval;
  }
  #pragma unroll
  for (int k_1 = 0; k_1 < 5; ++k_1) {
    Out[((((k_1 * 327680) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)) + 3276800)] = Settings[k_1];
  }
  Out[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 4915200)] = 0x0p+0f/*0.000000e+00*/;
}

