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

extern "C" __global__ void main_kernel(const float* __restrict__ DC, float* __restrict__ HistOut, float* __restrict__ History, float* __restrict__ Locks, float* __restrict__ Luma, const float* __restrict__ Masks, float* __restrict__ Mean, const float* __restrict__ Motion, const float* __restrict__ Params, const float* __restrict__ Prepared, float* __restrict__ Reactivity, float* __restrict__ Resolved, const float* __restrict__ Shade, float* __restrict__ Up, const float* __restrict__ Velocity, float* __restrict__ Weight);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ DC, float* __restrict__ HistOut, float* __restrict__ History, float* __restrict__ Locks, float* __restrict__ Luma, const float* __restrict__ Masks, float* __restrict__ Mean, const float* __restrict__ Motion, const float* __restrict__ Params, const float* __restrict__ Prepared, float* __restrict__ Reactivity, float* __restrict__ Resolved, const float* __restrict__ Shade, float* __restrict__ Up, const float* __restrict__ Velocity, float* __restrict__ Weight) {
  float hist[3];
  float weight = 0x0p+0f/*0.000000e+00*/;
  float bw = 0x0p+0f/*0.000000e+00*/;
  float up[3];
  float mean[3];
  float moment[3];
  float lo[3];
  float hi[3];
  float std[3];
  float lh[4];
  float minimum = 0x0p+0f/*0.000000e+00*/;
  signed char outside = (signed char)0;
  float accum[3];
  float blended[3];
  float rgbh[3];
  float rgbu[3];
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 139995) {
    float ux = __fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.128p+9f/*5.490000e+02*/);
    float uy = __fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.fep+8f/*5.100000e+02*/);
    float rx = (ux + Motion[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))]);
    float ry = (uy + Motion[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)]);
    bool existing = ((((0x0p+0f/*0.000000e+00*/ <= rx) && (rx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= ry)) && (ry <= 0x1p+0f/*1.000000e+00*/));
    float reactive = Masks[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))];
    float accmask = Masks[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)];
    float dc = DC[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
    float velocity = Velocity[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
    hist[0] = 0x0p+0f/*0.000000e+00*/;
    hist[1] = 0x0p+0f/*0.000000e+00*/;
    hist[2] = 0x0p+0f/*0.000000e+00*/;
    float shade = Shade[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))];
    float condval;
    if ((fmaxf(shade, shade) != 0x0p+0f/*0.000000e+00*/)) {
      condval = __fdiv_rn(fminf(shade, shade), fmaxf(shade, shade));
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float diff = (0x1p+0f/*1.000000e+00*/ - condval);
    float thisreact = fmaxf(fmaxf(reactive, 0x0p+0f/*0.000000e+00*/), fminf(fmaxf(((diff - 0x1.999999999999ap-4f/*1.000000e-01*/) * 0x1.4p+3f/*1.000000e+01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
    float condval_1;
    if ((fmaxf(shade, shade) != 0x0p+0f/*0.000000e+00*/)) {
      condval_1 = __fdiv_rn(fminf(shade, shade), fmaxf(shade, shade));
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float lockcon = fminf(fmaxf((fminf(fmaxf((fminf(fmaxf(-0x1p+0f/*-1.000000e+00*/, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * 0x1p+2f/*4.000000e+00*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * fminf(fmaxf(condval_1, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
    float sx = (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/);
    float sy = (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/);
    float jx = ((((float)((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)))) + 0x1p-1f/*5.000000e-01*/) - Params[3]);
    float jy = ((((float)((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)))) + 0x1p-1f/*5.000000e-01*/) - Params[4]);
    bool flipx = ((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx);
    bool flipy = ((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy);
    float offsetx = (jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/));
    float offsety = (jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/));
    float kr = fmaxf(thisreact, 0x1p+0f/*1.000000e+00*/);
    float bmax = (0x1p+0f/*1.000000e+00*/ - kr);
    float bmin = fmaxf(0x1p+0f/*1.000000e+00*/, ((0x1p+0f/*1.000000e+00*/ + (0x1p+0f/*1.000000e+00*/ - kr)) * 0x1.3333333333333p-2f/*3.000000e-01*/));
    float bias = ((0x1p+0f/*1.000000e+00*/ - kr) + ((bmin - (0x1p+0f/*1.000000e+00*/ - kr)) * fmaxf((0x1p-2f/*2.500000e-01*/ * dc), kr)));
    float curve = (-0x1p+1f/*-2.000000e+00*/ + (-0x1p+0f/*-1.000000e+00*/ * fminf(fmaxf((velocity * 0x1.47ae14p-6f/*2.000000e-02*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)));
    weight = 0x0p+0f/*0.000000e+00*/;
    bw = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
      up[k] = 0x0p+0f/*0.000000e+00*/;
      mean[k] = 0x0p+0f/*0.000000e+00*/;
      moment[k] = 0x0p+0f/*0.000000e+00*/;
    }
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
      int condval_2;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_2 = (1 - (j % 3));
      } else {
        condval_2 = ((j % 3) - 1);
      }
      float dx = ((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_2));
      int condval_3;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_3 = (1 - (j / 3));
      } else {
        condval_3 = ((j / 3) - 1);
      }
      float dy = ((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_3));
      int condval_4;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_4 = (1 - (j % 3));
      } else {
        condval_4 = ((j % 3) - 1);
      }
      int condval_5;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_5 = (1 - (j % 3));
      } else {
        condval_5 = ((j % 3) - 1);
      }
      int condval_6;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_6 = (1 - (j / 3));
      } else {
        condval_6 = ((j / 3) - 1);
      }
      int condval_7;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_7 = (1 - (j / 3));
      } else {
        condval_7 = ((j / 3) - 1);
      }
      float distance = ((((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_4)) * ((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_5))) + (((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_6)) * ((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_7))));
      int condval_8;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_8 = (1 - (j % 3));
      } else {
        condval_8 = ((j % 3) - 1);
      }
      int condval_9;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_9 = (1 - (j % 3));
      } else {
        condval_9 = ((j % 3) - 1);
      }
      int condval_10;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_10 = (1 - (j / 3));
      } else {
        condval_10 = ((j / 3) - 1);
      }
      int condval_11;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_11 = (1 - (j / 3));
      } else {
        condval_11 = ((j / 3) - 1);
      }
      float d = fminf(((((((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_8)) * ((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_9))) + (((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_10)) * ((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_11)))) * bias) * bias), 0x1p+2f/*4.000000e+00*/);
      float aa = ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/);
      float bb = ((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/);
      int condval_12;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_12 = (1 - (j / 3));
      } else {
        condval_12 = ((j / 3) - 1);
      }
      int condval_13;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_13 = (1 - (j / 3));
      } else {
        condval_13 = ((j / 3) - 1);
      }
      int condval_14;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_14 = (1 - (j % 3));
      } else {
        condval_14 = ((j % 3) - 1);
      }
      int condval_15;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_15 = (1 - (j % 3));
      } else {
        condval_15 = ((j % 3) - 1);
      }
      float wt = (((float)((((0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_12)) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_13) < 510)) && (0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_14))) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_15) < 549))) * ((((0x1.9p+0f/*1.562500e+00*/ * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) - 0x1.2p-1f/*5.625000e-01*/) * (((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/) * ((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/))));
      int condval_16;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_16 = (1 - (j % 3));
      } else {
        condval_16 = ((j % 3) - 1);
      }
      int condval_17;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_17 = (1 - (j % 3));
      } else {
        condval_17 = ((j % 3) - 1);
      }
      int condval_18;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_18 = (1 - (j / 3));
      } else {
        condval_18 = ((j / 3) - 1);
      }
      int condval_19;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_19 = (1 - (j / 3));
      } else {
        condval_19 = ((j / 3) - 1);
      }
      float boxweight = expf((curve * ((((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_16)) * ((jx - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_17))) + (((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_18)) * ((jy - (((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/)) + ((float)condval_19))))));
      int condval_20;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_20 = (1 - (j / 3));
      } else {
        condval_20 = ((j / 3) - 1);
      }
      int condval_21;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
        condval_21 = (1 - (j / 3));
      } else {
        condval_21 = ((j / 3) - 1);
      }
      int condval_22;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_22 = (1 - (j % 3));
      } else {
        condval_22 = ((j % 3) - 1);
      }
      int condval_23;
      if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
        condval_23 = (1 - (j % 3));
      } else {
        condval_23 = ((j % 3) - 1);
      }
      weight = (weight + (((float)((((0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_20)) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_21) < 510)) && (0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_22))) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_23) < 549))) * ((((0x1.9p+0f/*1.562500e+00*/ * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) - 0x1.2p-1f/*5.625000e-01*/) * (((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/) * ((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)))));
      bw = (bw + boxweight);
      #pragma unroll
      for (int k_1 = 0; k_1 < 3; ++k_1) {
        int condval_25;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
          condval_25 = (1 - (j / 3));
        } else {
          condval_25 = ((j / 3) - 1);
        }
        int condval_26;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
          condval_26 = (1 - (j / 3));
        } else {
          condval_26 = ((j / 3) - 1);
        }
        int condval_27;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
          condval_27 = (1 - (j % 3));
        } else {
          condval_27 = ((j % 3) - 1);
        }
        int condval_28;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
          condval_28 = (1 - (j % 3));
        } else {
          condval_28 = ((j % 3) - 1);
        }
        float condval_24;
        if (((((0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_25)) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_26) < 510)) && (0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_27))) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_28) < 549))) {
          int64_t condval_29;
          if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
            condval_29 = ((int64_t)1 - (((int64_t)j) / (int64_t)3));
          } else {
            condval_29 = ((((int64_t)j) / (int64_t)3) - (int64_t)1);
          }
          int64_t condval_30;
          if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
            condval_30 = ((int64_t)1 - (((int64_t)j) % (int64_t)3));
          } else {
            condval_30 = ((((int64_t)j) % (int64_t)3) - (int64_t)1);
          }
          int64_t condval_31;
          if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
            condval_31 = ((int64_t)1 - (((int64_t)j) / (int64_t)3));
          } else {
            condval_31 = ((((int64_t)j) / (int64_t)3) - (int64_t)1);
          }
          int64_t condval_32;
          if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
            condval_32 = ((int64_t)1 - (((int64_t)j) % (int64_t)3));
          } else {
            condval_32 = ((((int64_t)j) % (int64_t)3) - (int64_t)1);
          }
          condval_24 = Prepared[(((((((int64_t)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) * (int64_t)2196) + (condval_31 * (int64_t)2196)) + (((int64_t)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) * (int64_t)4)) + (condval_32 * (int64_t)4)) + ((int64_t)k_1))];
        } else {
          condval_24 = 0x0p+0f/*0.000000e+00*/;
        }
        float value = condval_24;
        int condval_33;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
          condval_33 = (1 - (j / 3));
        } else {
          condval_33 = ((j / 3) - 1);
        }
        int condval_34;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/) < jy)) {
          condval_34 = (1 - (j / 3));
        } else {
          condval_34 = ((j / 3) - 1);
        }
        int condval_35;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
          condval_35 = (1 - (j % 3));
        } else {
          condval_35 = ((j % 3) - 1);
        }
        int condval_36;
        if (((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/) < jx)) {
          condval_36 = (1 - (j % 3));
        } else {
          condval_36 = ((j % 3) - 1);
        }
        up[k_1] = (up[k_1] + (value * (((float)((((0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_33)) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_34) < 510)) && (0 <= (((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_35))) && ((((int)floorf((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/))) + condval_36) < 549))) * ((((0x1.9p+0f/*1.562500e+00*/ * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) * ((0x1.999999999999ap-2f/*4.000000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/)) - 0x1.2p-1f/*5.625000e-01*/) * (((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/) * ((0x1p-2f/*2.500000e-01*/ * d) - 0x1p+0f/*1.000000e+00*/))))));
        mean[k_1] = (mean[k_1] + (value * boxweight));
        moment[k_1] = (moment[k_1] + ((value * value) * boxweight));
        if (j == 0) {
          lo[k_1] = value;
          hi[k_1] = value;
        } else {
          lo[k_1] = fminf(lo[k_1], value);
          hi[k_1] = fmaxf(hi[k_1], value);
        }
      }
    }
    float condval_37;
    if ((0x1.0624dd2f1a9fcp-10f/*1.000000e-03*/ < fabsf(bw))) {
      condval_37 = bw;
    } else {
      condval_37 = 0x1p+0f/*1.000000e+00*/;
    }
    bw = condval_37;
    bool good = (0x1.0624dd2f1a9fcp-10f/*1.000000e-03*/ < weight);
    #pragma unroll
    for (int k_2 = 0; k_2 < 3; ++k_2) {
      mean[k_2] = __fdiv_rn(mean[k_2], bw);
      std[k_2] = sqrtf(fabsf((__fdiv_rn(moment[k_2], bw) - (mean[k_2] * mean[k_2]))));
      float condval_38;
      if (good) {
        condval_38 = fminf(fmaxf(__fdiv_rn(up[k_2], weight), lo[k_2]), hi[k_2]);
      } else {
        condval_38 = up[k_2];
      }
      up[k_2] = condval_38;
    }
    float condval_39;
    if (good) {
      condval_39 = (weight * 0x1.555556p-4f/*8.333334e-02*/);
    } else {
      condval_39 = 0x0p+0f/*0.000000e+00*/;
    }
    weight = condval_39;
    float current = mean[0];
    float current_1 = (nearbyintf((current * 0x1.fep+7f/*2.550000e+02*/)) * 0x1.010102p-8f/*3.921569e-03*/);
    #pragma unroll
    for (int k_3 = 0; k_3 < 4; ++k_3) {
      lh[k_3] = 0x0p+0f/*0.000000e+00*/;
    }
    float d0 = (current_1 - lh[0]);
    minimum = fabsf(d0);
    #pragma unroll
    for (int k_4 = 0; k_4 < 3; ++k_4) {
      float dd = (current_1 - lh[(k_4 + 1)]);
      float condval_41;
      if ((0x0p+0f/*0.000000e+00*/ < d0)) {
        condval_41 = 0x1p+0f/*1.000000e+00*/;
      } else {
        float condval_42;
        if ((d0 < 0x0p+0f/*0.000000e+00*/)) {
          condval_42 = -0x1p+0f/*-1.000000e+00*/;
        } else {
          condval_42 = 0x0p+0f/*0.000000e+00*/;
        }
        condval_41 = condval_42;
      }
      float condval_43;
      if ((0x0p+0f/*0.000000e+00*/ < dd)) {
        condval_43 = 0x1p+0f/*1.000000e+00*/;
      } else {
        float condval_44;
        if ((dd < 0x0p+0f/*0.000000e+00*/)) {
          condval_44 = -0x1p+0f/*-1.000000e+00*/;
        } else {
          condval_44 = 0x0p+0f/*0.000000e+00*/;
        }
        condval_43 = condval_44;
      }
      float condval_40;
      if ((condval_41 == condval_43)) {
        condval_40 = fminf(minimum, fabsf(dd));
      } else {
        condval_40 = minimum;
      }
      minimum = condval_40;
    }
    float instability = (((float)(minimum != fabsf(d0))) * powf(fminf(fmaxf((std[0] * 0x1.4p+3f/*1.000000e+01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/), 0x1.8p+2f/*6.000000e+00*/));
    float instability_1 = ((((float)(0x1.010101010101p-8f/*3.921569e-03*/ < instability)) * ((float)(0x1.010101010101p-8f/*3.921569e-03*/ <= fabsf(d0)))) * (0x1p+0f/*1.000000e+00*/ - fmaxf(accmask, powf(thisreact, 0x1.5555555555555p-3f/*1.666667e-01*/))));
    float instability_2 = (instability_1 * ((float)(lh[2] != 0x0p+0f/*0.000000e+00*/)));
    float base = ((((float)((((0x0p+0f/*0.000000e+00*/ <= rx) && (rx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= ry)) && (ry <= 0x1p+0f/*1.000000e+00*/))) * (0x1p+0f/*1.000000e+00*/ - thisreact)) * (0x1p+0f/*1.000000e+00*/ - dc));
    float base_1 = fminf(((((float)((((0x0p+0f/*0.000000e+00*/ <= rx) && (rx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= ry)) && (ry <= 0x1p+0f/*1.000000e+00*/))) * (0x1p+0f/*1.000000e+00*/ - thisreact)) * (0x1p+0f/*1.000000e+00*/ - dc)), (((((float)((((0x0p+0f/*0.000000e+00*/ <= rx) && (rx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= ry)) && (ry <= 0x1p+0f/*1.000000e+00*/))) * (0x1p+0f/*1.000000e+00*/ - thisreact)) * (0x1p+0f/*1.000000e+00*/ - dc)) + (((weight * 0x1.4p+3f/*1.000000e+01*/) - ((((float)((((0x0p+0f/*0.000000e+00*/ <= rx) && (rx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= ry)) && (ry <= 0x1p+0f/*1.000000e+00*/))) * (0x1p+0f/*1.000000e+00*/ - thisreact)) * (0x1p+0f/*1.000000e+00*/ - dc))) * fmaxf(0x0p+0f/*0.000000e+00*/, fminf(fmaxf((velocity * 0x1.4p+3f/*1.000000e+01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)))));
    float base_2 = fminf(base_1, (base_1 + ((weight - base_1) * fminf(fmaxf((velocity * 0x1.99999ap-5f/*5.000000e-02*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/))));
    outside = (signed char)0;
    #pragma unroll
    for (int k_5 = 0; k_5 < 3; ++k_5) {
      lo[k_5] = fmaxf(lo[k_5], (mean[k_5] - std[k_5]));
      hi[k_5] = fminf(hi[k_5], (mean[k_5] + std[k_5]));
      outside = ((signed char)((((bool)outside) || (hist[k_5] < lo[k_5])) || (hi[k_5] < hist[k_5])));
    }
    float contribution = fminf(fmaxf((fmaxf(instability_2, lockcon) * (0x1p+0f/*1.000000e+00*/ - sqrtf(reactive))), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
    #pragma unroll
    for (int k_6 = 0; k_6 < 3; ++k_6) {
      float condval_45;
      if (((bool)outside)) {
        condval_45 = (fminf(fmaxf(hist[k_6], lo[k_6]), hi[k_6]) + ((hist[k_6] - fminf(fmaxf(hist[k_6], lo[k_6]), hi[k_6])) * contribution));
      } else {
        condval_45 = hist[k_6];
      }
      hist[k_6] = condval_45;
      float condval_46;
      if (((bool)outside)) {
        condval_46 = (fminf(base_2, 0x1.999999999999ap-4f/*1.000000e-01*/) + ((base_2 - fminf(base_2, 0x1.999999999999ap-4f/*1.000000e-01*/)) * contribution));
      } else {
        condval_46 = base_2;
      }
      accum[k_6] = condval_46;
      HistOut[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k_6)] = hist[k_6];
      Up[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k_6)] = up[k_6];
      Mean[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k_6)] = mean[k_6];
    }
    #pragma unroll
    for (int k_7 = 0; k_7 < 3; ++k_7) {
      blended[k_7] = (hist[k_7] + ((up[k_7] - hist[k_7]) * __fdiv_rn(weight, fmaxf((accum[k_7] + weight), 0x1.0624dd2f1a9fcp-10f/*1.000000e-03*/))));
    }
    rgbh[0] = ((blended[0] + blended[1]) - blended[2]);
    rgbh[1] = (blended[0] + blended[2]);
    rgbh[2] = ((blended[0] - blended[1]) - blended[2]);
    rgbu[0] = ((up[0] + up[1]) - up[2]);
    rgbu[1] = (up[0] + up[2]);
    rgbu[2] = ((up[0] - up[1]) - up[2]);
    #pragma unroll
    for (int k_8 = 0; k_8 < 3; ++k_8) {
      float value_1 = (__fdiv_rn(rgbu[k_8], Params[0]) * Params[1]);
      Resolved[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + k_8)] = value_1;
      half_t v_ = (half_t)value_1;
      ushort v__1 = (*(ushort *)(&(v_))) - ((ushort)(fabsf(value_1) < fabsf(((float)((half_t)value_1)))));
      History[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + k_8)] = ((float)(*(half_t *)(&(v__1))));
    }
    float backx = (ux - Motion[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))]);
    float backy = (uy - Motion[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)]);
    bool validback = ((((0x0p+0f/*0.000000e+00*/ <= backx) && (backx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= backy)) && (backy <= 0x1p+0f/*1.000000e+00*/));
    float condval_47;
    if (((((0x0p+0f/*0.000000e+00*/ <= backx) && (backx <= 0x1p+0f/*1.000000e+00*/)) && (0x0p+0f/*0.000000e+00*/ <= backy)) && (backy <= 0x1p+0f/*1.000000e+00*/))) {
      condval_47 = fmaxf((0x0p+0f/*0.000000e+00*/ - (weight * Params[5])), 0x0p+0f/*0.000000e+00*/);
    } else {
      condval_47 = 0x0p+0f/*0.000000e+00*/;
    }
    float life = condval_47;
    float temporal = fminf(thisreact, 0x1.fae147ae147aep-1f/*9.900000e-01*/);
    float temporal_1 = fmaxf(temporal, (temporal + ((0x1.999999999999ap-2f/*4.000000e-01*/ - temporal) * fminf(fmaxf(velocity, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/))));
    float temporal_2 = fmaxf((temporal_1 * temporal_1), fmaxf((dc * 0x1.999999999999ap-4f/*1.000000e-01*/), reactive));
    float condval_48;
    if ((0x1p+0f/*1.000000e+00*/ <= fminf(fmaxf((velocity * 0x1.4p+3f/*1.000000e+01*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/))) {
      condval_48 = (fmaxf(0x1p+0f/*1.000000e+00*/, 0x1.0624dd2f1a9fcp-10f/*1.000000e-03*/) * -0x1p+0f/*-1.000000e+00*/);
    } else {
      condval_48 = 0x1p+0f/*1.000000e+00*/;
    }
    float temporal_3 = condval_48;
    half_t v__2 = (half_t)temporal_3;
    ushort v__3 = (*(ushort *)(&(v__2))) - ((ushort)(fabsf(temporal_3) < fabsf(((float)((half_t)temporal_3)))));
    History[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + 3)] = ((float)(*(half_t *)(&(v__3))));
    half_t v__4 = (half_t)life;
    ushort v__5 = (*(ushort *)(&(v__4))) - ((ushort)(fabsf(life) < fabsf(((float)((half_t)life)))));
    Locks[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))] = ((float)(*(half_t *)(&(v__5))));
    half_t v__6 = (half_t)shade;
    ushort v__7 = (*(ushort *)(&(v__6))) - ((ushort)(fabsf(shade) < fabsf(((float)((half_t)shade)))));
    Locks[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)] = ((float)(*(half_t *)(&(v__7))));
    Luma[((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))] = (nearbyintf((fminf(fmaxf(current_1, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * 0x1.fep+7f/*2.550000e+02*/)) * 0x1.010101010101p-8f/*3.921569e-03*/);
    #pragma unroll
    for (int k_9 = 0; k_9 < 3; ++k_9) {
      Luma[((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + k_9) + 1)] = (nearbyintf((fminf(fmaxf(lh[k_9], 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * 0x1.fep+7f/*2.550000e+02*/)) * 0x1.010101010101p-8f/*3.921569e-03*/);
    }
    Weight[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = weight;
    Reactivity[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = thisreact;
  }
}

