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

extern "C" __global__ void main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Residual, const float* __restrict__ Sigmoid, int H, int W);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Current, const float* __restrict__ Gate, float* __restrict__ Out, const float* __restrict__ Previous, const float* __restrict__ Residual, const float* __restrict__ Sigmoid, int H, int W) {
  if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (H * W)) {
    int y = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / W), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / W));
    int x = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % W), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % W));
    float condval_1;
    if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
      condval_1 = Sigmoid[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float condval_2;
    if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
      condval_2 = Gate[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    float condval_3;
    if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
      condval_3 = Sigmoid[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    float condval_4;
    if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
      condval_4 = Gate[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
    } else {
      condval_4 = 0x0p+0f/*0.000000e+00*/;
    }
    float condval;
    if (((condval_1 * fminf(fmaxf(condval_2, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)) != (condval_3 * fminf(fmaxf(condval_4, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)))) {
      float condval_5;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_5 = Sigmoid[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      float condval_6;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_6 = Gate[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      condval = (condval_5 * fminf(fmaxf(condval_6, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
    } else {
      float condval_7;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_7 = Sigmoid[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
      } else {
        condval_7 = 0x0p+0f/*0.000000e+00*/;
      }
      float condval_8;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_8 = Gate[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))];
      } else {
        condval_8 = 0x0p+0f/*0.000000e+00*/;
      }
      condval = fminf(fmaxf((condval_7 * fminf(fmaxf(condval_8, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
    }
    float alpha = condval;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      float condval_9;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_9 = Current[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))];
      } else {
        condval_9 = 0x0p+0f/*0.000000e+00*/;
      }
      float centered = ((condval_9 * 0x1p-3f/*1.250000e-01*/) - 0x1p-4f/*6.250000e-02*/);
      float condval_10;
      if (((((0 <= x) && (x < W)) && (0 <= y)) && (y < H))) {
        condval_10 = Residual[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))];
      } else {
        condval_10 = 0x0p+0f/*0.000000e+00*/;
      }
      float corrected = (centered + (condval_10 * 0x1p-5f/*3.125000e-02*/));
      float corrected_1 = fminf(fmaxf(((corrected * 0x1p+3f/*8.000000e+00*/) + 0x1p-1f/*5.000000e-01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
      if (0 <= x) {
        if (x < W) {
          if (0 <= y) {
            if (y < H) {
              Out[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))] = (corrected_1 + (alpha * (Previous[((((((int64_t)y) * ((int64_t)W)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))] - corrected_1)));
            }
          }
        }
      }
    }
  }
}

