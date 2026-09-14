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

extern "C" __global__ void main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, const float* __restrict__ Gaussian, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Settings, int H, int W);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, const float* __restrict__ Gaussian, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Settings, int H, int W) {
  if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (H * W)) {
    int y = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / W), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / W));
    int x = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % W), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % W));
    if (0 <= x) {
      if (x < W) {
        if (0 <= y) {
          if (y < H) {
            Out[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))] = Gaussian[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + (int64_t)1)];
            Out[(((((int64_t)H) + ((int64_t)y)) * ((int64_t)W)) + ((int64_t)x))] = Gaussian[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + (int64_t)2)];
            Out[((((((int64_t)W) * ((int64_t)H)) * (int64_t)2) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = Gaussian[(((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3))];
            Out[((((((int64_t)W) * ((int64_t)H)) * (int64_t)3) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = 0x1p+0f/*1.000000e+00*/;
          }
        }
      }
    }
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      float condval;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval = Current[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      half_t current = ((((half_t)condval) - half_t(0x1p-1f/*5.000000e-01*/)) * half_t(0x1p-3f/*1.250000e-01*/));
      float condval_1;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_1 = Previous[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      half_t prev = ((((half_t)condval_1) - half_t(0x1p-1f/*5.000000e-01*/)) * half_t(0x1p-3f/*1.250000e-01*/));
      if (0 <= x) {
        if (x < W) {
          if (0 <= y) {
            if (y < H) {
              Out[(((((((int64_t)k) + (int64_t)4) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = ((float)current);
              float condval_2;
              if ((0x0p+0f/*0.000000e+00*/ < Gate[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))])) {
                condval_2 = ((float)prev);
              } else {
                condval_2 = ((float)current);
              }
              Out[(((((((int64_t)k) + (int64_t)7) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = condval_2;
            }
          }
        }
      }
    }
    #pragma unroll
    for (int k_1 = 0; k_1 < 5; ++k_1) {
      if (0 <= x) {
        if (x < W) {
          if (0 <= y) {
            if (y < H) {
              Out[(((((((int64_t)k_1) + (int64_t)10) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = Settings[k_1];
            }
          }
        }
      }
    }
    if (0 <= x) {
      if (x < W) {
        if (0 <= y) {
          if (y < H) {
            Out[((((((int64_t)W) * ((int64_t)H)) * (int64_t)15) + (((int64_t)y) * ((int64_t)W))) + ((int64_t)x))] = 0x0p+0f/*0.000000e+00*/;
          }
        }
      }
    }
  }
}

