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

extern "C" __global__ void main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Residual, const float* __restrict__ Sigmoid);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Residual, const float* __restrict__ Sigmoid) {
  float condval;
  if (((Sigmoid[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] * fminf(fmaxf(Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)) != (Sigmoid[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] * fminf(fmaxf(Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)))) {
    condval = (Sigmoid[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] * fminf(fmaxf(Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
  } else {
    condval = fminf(fmaxf((Sigmoid[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] * fminf(fmaxf(Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
  }
  float alpha = condval;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    float centered = ((Current[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)] * 0x1p-3f/*1.250000e-01*/) - 0x1p-4f/*6.250000e-02*/);
    float corrected = (centered + (Residual[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)] * 0x1p-5f/*3.125000e-02*/));
    float corrected_1 = fminf(fmaxf(((corrected * 0x1p+3f/*8.000000e+00*/) + 0x1p-1f/*5.000000e-01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
    Out[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)] = (corrected_1 + (alpha * (Previous[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)] - corrected_1)));
  }
}

