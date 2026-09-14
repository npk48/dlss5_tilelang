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

extern "C" __global__ void main_kernel(const float* __restrict__ Confidence, float* __restrict__ Current, float* __restrict__ Gate, float* __restrict__ MV, const float* __restrict__ Motion, const float* __restrict__ Work, int Cold, int H, int InvHBits, int InvWBits, int NH, int NW, int RH, int RW, int StrengthBits, int W);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Confidence, float* __restrict__ Current, float* __restrict__ Gate, float* __restrict__ MV, const float* __restrict__ Motion, const float* __restrict__ Work, int Cold, int H, int InvHBits, int InvWBits, int NH, int NW, int RH, int RW, int StrengthBits, int W) {
  if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (NH * NW)) {
    int y = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / NW), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / NW));
    int x = max((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % NW), (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % NW));
    int condval;
    if ((y < H)) {
      condval = y;
    } else {
      condval = (((H * 2) - y) - 2);
    }
    int condval_1;
    if ((y < H)) {
      condval_1 = y;
    } else {
      condval_1 = (((H * 2) - y) - 2);
    }
    int sy = max(max(0, min(condval, (H - 1))), max(0, min(condval_1, (H - 1))));
    int condval_2;
    if ((x < W)) {
      condval_2 = x;
    } else {
      condval_2 = (((W * 2) - x) - 2);
    }
    int condval_3;
    if ((x < W)) {
      condval_3 = x;
    } else {
      condval_3 = (((W * 2) - x) - 2);
    }
    int sx = max(max(0, min(condval_2, (W - 1))), max(0, min(condval_3, (W - 1))));
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (0 <= x) {
        if (x < NW) {
          if (0 <= y) {
            if (y < NH) {
              float condval_4;
              if (((((0 <= sx) && (sx < W)) && (0 <= sy)) && (sy < H))) {
                condval_4 = Work[((((((int64_t)k) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)sy) * ((int64_t)W))) + ((int64_t)sx))];
              } else {
                condval_4 = 0x0p+0f/*0.000000e+00*/;
              }
              half_t v_ = (half_t)condval_4;
              float condval_5;
              if (((((0 <= sx) && (sx < W)) && (0 <= sy)) && (sy < H))) {
                condval_5 = Work[((((((int64_t)k) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)sy) * ((int64_t)W))) + ((int64_t)sx))];
              } else {
                condval_5 = 0x0p+0f/*0.000000e+00*/;
              }
              float condval_6;
              if (((((0 <= sx) && (sx < W)) && (0 <= sy)) && (sy < H))) {
                condval_6 = Work[((((((int64_t)k) * ((int64_t)W)) * ((int64_t)H)) + (((int64_t)sy) * ((int64_t)W))) + ((int64_t)sx))];
              } else {
                condval_6 = 0x0p+0f/*0.000000e+00*/;
              }
              ushort v__1 = (*(ushort *)(&(v_))) - ((ushort)(fabsf(condval_5) < fabsf(((float)((half_t)condval_6)))));
              Current[((((((int64_t)y) * ((int64_t)NW)) * (int64_t)3) + (((int64_t)x) * (int64_t)3)) + ((int64_t)k))] = ((float)(*(half_t *)(&(v__1))));
            }
          }
        }
      }
    }
    if (Cold == 0) {
      if ((y < H) && (x < W)) {
        uint v__2 = (uint)InvWBits;
        uint v__3 = (uint)InvHBits;
        float condval_7;
        if (((0 <= ((int)floorf((((((float)x) + 0x1p-1f/*5.000000e-01*/) * ((float)RW)) * (*(float *)(&(v__2))))))) && (0 <= ((int)floorf((((((float)y) + 0x1p-1f/*5.000000e-01*/) * ((float)RH)) * (*(float *)(&(v__3))))))))) {
          condval_7 = Motion[((min(((int64_t)floorf((((((float)x) + 0x1p-1f/*5.000000e-01*/) * ((float)RW)) * (*(float *)(&(v__2)))))), (((int64_t)RW) - (int64_t)1)) * (int64_t)2) + ((min(((int64_t)floorf((((((float)y) + 0x1p-1f/*5.000000e-01*/) * ((float)RH)) * (*(float *)(&(v__3)))))), (((int64_t)RH) - (int64_t)1)) * ((int64_t)RW)) * (int64_t)2))];
        } else {
          condval_7 = 0x0p+0f/*0.000000e+00*/;
        }
        float dx = (condval_7 * ((float)W));
        float condval_8;
        if (((0 <= ((int)floorf((((((float)x) + 0x1p-1f/*5.000000e-01*/) * ((float)RW)) * (*(float *)(&(v__2))))))) && (0 <= ((int)floorf((((((float)y) + 0x1p-1f/*5.000000e-01*/) * ((float)RH)) * (*(float *)(&(v__3))))))))) {
          condval_8 = Motion[(((min(((int64_t)floorf((((((float)x) + 0x1p-1f/*5.000000e-01*/) * ((float)RW)) * (*(float *)(&(v__2)))))), (((int64_t)RW) - (int64_t)1)) * (int64_t)2) + ((min(((int64_t)floorf((((((float)y) + 0x1p-1f/*5.000000e-01*/) * ((float)RH)) * (*(float *)(&(v__3)))))), (((int64_t)RH) - (int64_t)1)) * ((int64_t)RW)) * (int64_t)2)) + (int64_t)1)];
        } else {
          condval_8 = 0x0p+0f/*0.000000e+00*/;
        }
        float dy = (condval_8 * ((float)H));
        float px = ((((float)x) + 0x1p-1f/*5.000000e-01*/) + dx);
        float py = ((((float)y) + 0x1p-1f/*5.000000e-01*/) + dy);
        bool visible = ((((0x0p+0f/*0.000000e+00*/ <= ((((float)x) + 0x1p-1f/*5.000000e-01*/) + dx)) && (((((float)x) + 0x1p-1f/*5.000000e-01*/) + dx) <= ((float)W))) && (0x0p+0f/*0.000000e+00*/ <= ((((float)y) + 0x1p-1f/*5.000000e-01*/) + dy))) && (((((float)y) + 0x1p-1f/*5.000000e-01*/) + dy) <= ((float)H)));
        if (0 <= x) {
          if (x < NW) {
            if (0 <= y) {
              if (y < NH) {
                MV[(((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2))] = dx;
                MV[((((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2)) + (int64_t)1)] = dy;
                uint v__4 = (uint)StrengthBits;
                Gate[((((int64_t)y) * ((int64_t)NW)) + ((int64_t)x))] = ((((float)((((0x0p+0f/*0.000000e+00*/ <= ((((float)x) + 0x1p-1f/*5.000000e-01*/) + dx)) && (((((float)x) + 0x1p-1f/*5.000000e-01*/) + dx) <= ((float)W))) && (0x0p+0f/*0.000000e+00*/ <= ((((float)y) + 0x1p-1f/*5.000000e-01*/) + dy))) && (((((float)y) + 0x1p-1f/*5.000000e-01*/) + dy) <= ((float)H)))) * Confidence[((((int64_t)y) * ((int64_t)W)) + ((int64_t)x))]) * (*(float *)(&(v__4))));
              }
            }
          }
        }
      } else {
        if (0 <= x) {
          if (x < NW) {
            if (0 <= y) {
              if (y < NH) {
                MV[(((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2))] = 0x0p+0f/*0.000000e+00*/;
                MV[((((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2)) + (int64_t)1)] = 0x0p+0f/*0.000000e+00*/;
                Gate[((((int64_t)y) * ((int64_t)NW)) + ((int64_t)x))] = 0x0p+0f/*0.000000e+00*/;
              }
            }
          }
        }
      }
    } else {
      if (0 <= x) {
        if (x < NW) {
          if (0 <= y) {
            if (y < NH) {
              MV[(((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2))] = 0x0p+0f/*0.000000e+00*/;
              MV[((((((int64_t)y) * ((int64_t)NW)) * (int64_t)2) + (((int64_t)x) * (int64_t)2)) + (int64_t)1)] = 0x0p+0f/*0.000000e+00*/;
              Gate[((((int64_t)y) * ((int64_t)NW)) + ((int64_t)x))] = 0x0p+0f/*0.000000e+00*/;
            }
          }
        }
      }
    }
  }
}

