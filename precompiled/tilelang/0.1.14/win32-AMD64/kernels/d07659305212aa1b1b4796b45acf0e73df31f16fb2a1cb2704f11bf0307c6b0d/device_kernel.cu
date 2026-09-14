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

extern "C" __global__ void main_kernel(uchar* __restrict__ arena, int arena_bytes, int height, int pool_height, int pool_width, int start, int width);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(uchar* __restrict__ arena, int arena_bytes, int height, int pool_height, int pool_width, int start, int width) {
  int rmod = (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % ((pool_height * pool_width) * 16));
  int rmod_1 = (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) % ((pool_height * pool_width) * 16));
  int pixel = max((((((0 <= (pool_height * pool_width)) && (0 <= rmod)) || (((pool_height * pool_width) < 0) && (rmod <= 0))) ? rmod : (((pool_height * pool_width) * 16) + rmod)) >> 4), (((((0 <= (pool_height * pool_width)) && (0 <= rmod_1)) || (((pool_height * pool_width) < 0) && (rmod_1 <= 0))) ? rmod_1 : (((pool_height * pool_width) * 16) + rmod_1)) >> 4));
  int rmod_2 = (pixel % pool_width);
  int rdiv = (pixel / pool_width);
  int rmod_3 = (pixel % pool_width);
  int rdiv_1 = (pixel / pool_width);
  int y = max(((((0 <= pool_width) && (0 <= rmod_2)) || ((pool_width < 0) && (rmod_2 <= 0))) ? rdiv : (rdiv - 1)), ((((0 <= pool_width) && (0 <= rmod_3)) || ((pool_width < 0) && (rmod_3 <= 0))) ? rdiv_1 : (rdiv_1 - 1)));
  int rmod_4 = (pixel % pool_width);
  int rmod_5 = (pixel % pool_width);
  int x = max(((((0 <= pool_width) && (0 <= rmod_4)) || ((pool_width < 0) && (rmod_4 <= 0))) ? rmod_4 : (rmod_4 + pool_width)), ((((0 <= pool_width) && (0 <= rmod_5)) || ((pool_width < 0) && (rmod_5 <= 0))) ? rmod_5 : (rmod_5 + pool_width)));
  if ((height <= (y * 2)) || (width <= (x * 2))) {
    if (0 <= (((((int)blockIdx.x) * 256) + start) + ((int)threadIdx.x))) {
      if ((((((int)blockIdx.x) * 256) + start) + ((int)threadIdx.x)) < arena_bytes) {
        arena[(((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((int64_t)start)) + ((int64_t)((int)threadIdx.x)))] = (uchar)0;
      }
    }
  }
}

