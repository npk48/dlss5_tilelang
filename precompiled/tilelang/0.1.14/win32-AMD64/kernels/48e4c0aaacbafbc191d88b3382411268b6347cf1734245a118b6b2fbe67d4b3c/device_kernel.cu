#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/atomic.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const float* __restrict__ J, const float* __restrict__ Luma, float* __restrict__ Out);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ J, const float* __restrict__ Luma, float* __restrict__ Out) {
  int bits = 0;
  float low = 0x0p+0f/*0.000000e+00*/;
  float high = 0x0p+0f/*0.000000e+00*/;
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 139995) {
    bits = 16;
    low = 0x1.fffffep+127f/*3.402823e+38*/;
    high = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
      if (i != 4) {
        float ratio = __fdiv_rn(fmaxf(Luma[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (i / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (i % 3)) - 1), 548)))], Luma[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]), fminf(Luma[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (i / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (i % 3)) - 1), 548)))], Luma[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]));
        bool same = ((0x0p+0f/*0.000000e+00*/ < ratio) && (ratio < 0x1.0cccccccccccdp+0f/*1.050000e+00*/));
        bits = (bits | (((int)((0x0p+0f/*0.000000e+00*/ < ratio) && (ratio < 0x1.0cccccccccccdp+0f/*1.050000e+00*/))) << i));
        float condval;
        if (((0x0p+0f/*0.000000e+00*/ < ratio) && (ratio < 0x1.0cccccccccccdp+0f/*1.050000e+00*/))) {
          condval = low;
        } else {
          condval = fminf(low, Luma[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (i / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (i % 3)) - 1), 548)))]);
        }
        low = condval;
        float condval_1;
        if (((0x0p+0f/*0.000000e+00*/ < ratio) && (ratio < 0x1.0cccccccccccdp+0f/*1.050000e+00*/))) {
          condval_1 = high;
        } else {
          condval_1 = fmaxf(high, Luma[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (i / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (i % 3)) - 1), 548)))]);
        }
        high = condval_1;
      }
    }
    bool ridge = ((((((high < Luma[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]) || (Luma[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] < low)) && ((bits & 27) < 27)) && ((bits & 54) < 54)) && ((bits & 216) < 216)) && ((bits & 432) < 432));
    int qx = ((int)floorf((__fdiv_rn(((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) - J[0]), 0x1.128p+9f/*5.490000e+02*/) * 0x1.128p+9f/*5.490000e+02*/)));
    int qy = ((int)floorf((__fdiv_rn(((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) - J[1]), 0x1.fep+8f/*5.100000e+02*/) * 0x1.fep+8f/*5.100000e+02*/)));
    if ((((ridge && (0 <= qy)) && (qy < 510)) && (0 <= qx)) && (qx < 549)) {
      AtomicMax((&(Out[((qy * 549) + qx)])), 0x1p+0f/*1.000000e+00*/, 0);
    }
  }
}

