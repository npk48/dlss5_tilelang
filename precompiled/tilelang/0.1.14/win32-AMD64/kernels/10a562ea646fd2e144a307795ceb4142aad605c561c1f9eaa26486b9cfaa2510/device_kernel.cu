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

extern "C" __global__ void main_kernel(float* __restrict__ Current, float* __restrict__ Gate, float* __restrict__ MV, const float* __restrict__ Work);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ Current, float* __restrict__ Gate, float* __restrict__ MV, const float* __restrict__ Work) {
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    int condval;
    if ((((int)blockIdx.x) < 2550)) {
      condval = (((int)blockIdx.x) / 5);
    } else {
      condval = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_1;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_1 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_1 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    int condval_2;
    if ((((int)blockIdx.x) < 2550)) {
      condval_2 = (((int)blockIdx.x) / 5);
    } else {
      condval_2 = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_3;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_3 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_3 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    half_t v_ = (half_t)Work[(((k * 279990) + (max(0, min(condval_2, 509)) * 549)) + max(0, min(condval_3, 548)))];
    int condval_4;
    if ((((int)blockIdx.x) < 2550)) {
      condval_4 = (((int)blockIdx.x) / 5);
    } else {
      condval_4 = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_5;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_5 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_5 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    int condval_6;
    if ((((int)blockIdx.x) < 2550)) {
      condval_6 = (((int)blockIdx.x) / 5);
    } else {
      condval_6 = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_7;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_7 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_7 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    int condval_8;
    if ((((int)blockIdx.x) < 2550)) {
      condval_8 = (((int)blockIdx.x) / 5);
    } else {
      condval_8 = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_9;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_9 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_9 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    int condval_10;
    if ((((int)blockIdx.x) < 2550)) {
      condval_10 = (((int)blockIdx.x) / 5);
    } else {
      condval_10 = (1018 - (((int)blockIdx.x) / 5));
    }
    int condval_11;
    if (((((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x)) < 549)) {
      condval_11 = (((((int)blockIdx.x) % 5) * 128) + ((int)threadIdx.x));
    } else {
      condval_11 = ((1096 - ((int)threadIdx.x)) - ((((int)blockIdx.x) % 5) * 128));
    }
    ushort v__1 = (*(ushort *)(&(v_))) - ((ushort)(fabsf(Work[(((k * 279990) + (max(0, min(condval_6, 509)) * 549)) + max(0, min(condval_7, 548)))]) < fabsf(((float)((half_t)Work[(((k * 279990) + (max(0, min(condval_10, 509)) * 549)) + max(0, min(condval_11, 548)))])))));
    Current[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k)] = ((float)(*(half_t *)(&(v__1))));
  }
  MV[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))] = 0x0p+0f/*0.000000e+00*/;
  MV[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)] = 0x0p+0f/*0.000000e+00*/;
  Gate[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = 0x0p+0f/*0.000000e+00*/;
}

