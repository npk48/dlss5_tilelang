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

extern "C" __global__ void main_kernel(const float* __restrict__ C, const float* __restrict__ CN, const float* __restrict__ Comp, const float* __restrict__ DD, const float* __restrict__ DI, const float* __restrict__ DM, const float* __restrict__ MV, float* __restrict__ Masks, const float* __restrict__ P, float* __restrict__ Prepared, const float* __restrict__ React, const float* __restrict__ Rec, const float* __restrict__ VM);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ C, const float* __restrict__ CN, const float* __restrict__ Comp, const float* __restrict__ DD, const float* __restrict__ DI, const float* __restrict__ DM, const float* __restrict__ MV, float* __restrict__ Masks, const float* __restrict__ P, float* __restrict__ Prepared, const float* __restrict__ React, const float* __restrict__ Rec, const float* __restrict__ VM) {
  float total = 0x0p+0f/*0.000000e+00*/;
  float ws = 0x0p+0f/*0.000000e+00*/;
  float maximum = 0x0p+0f/*0.000000e+00*/;
  float convergence = 0x0p+0f/*0.000000e+00*/;
  float mind = 0x0p+0f/*0.000000e+00*/;
  float maxd = 0x0p+0f/*0.000000e+00*/;
  signed char farfound = (signed char)0;
  float rr = 0x0p+0f/*0.000000e+00*/;
  float tc = 0x0p+0f/*0.000000e+00*/;
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 139995) {
    float condval;
    if ((0x1.47ae147ae147bp-7f/*1.000000e-02*/ < DI[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)])) {
      condval = DM[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float ux = (__fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.128p+9f/*5.490000e+02*/) + condval);
    float condval_1;
    if ((0x1.47ae147ae147bp-7f/*1.000000e-02*/ < DI[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)])) {
      condval_1 = DM[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float uy = (__fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.fep+8f/*5.100000e+02*/) + condval_1);
    float px = ((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/);
    float py = ((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/);
    float fx = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
    float fy = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
    float zz = __fdiv_rn(P[1], (DD[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] - P[0]));
    total = 0x0p+0f/*0.000000e+00*/;
    ws = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      float condval_2;
      if (((q % 2) == 0)) {
        condval_2 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_2 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_3;
      if (((q >> 1) == 0)) {
        condval_3 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_3 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float wt = (condval_2 * condval_3);
      float condval_4;
      if (((((0 <= ((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))) && (((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) < 510)) && (0 <= (((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)))) && ((((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)) < 549))) {
        condval_4 = Rec[(((((((int64_t)q) >> (int64_t)1) * (int64_t)549) + (((int64_t)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))) * (int64_t)549)) + ((int64_t)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) + (((int64_t)q) & (int64_t)1))];
      } else {
        condval_4 = 0x0p+0f/*0.000000e+00*/;
      }
      float prev = __fdiv_rn(P[1], (condval_4 - P[0]));
      float difference = (zz - prev);
      float condval_5;
      if (((q % 2) == 0)) {
        condval_5 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_5 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_6;
      if (((q >> 1) == 0)) {
        condval_6 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_6 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      bool take = ((((((0 <= ((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))) && (((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) < 510)) && (0 <= (((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)))) && ((((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)) < 549)) && (0x1.47ae147ae147bp-7f/*1.000000e-02*/ < (condval_5 * condval_6))) && (0x0p+0f/*0.000000e+00*/ < (zz - prev)));
      float sep = (P[2] * fmaxf(zz, prev));
      float condval_8;
      if (((q % 2) == 0)) {
        condval_8 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_8 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_9;
      if (((q >> 1) == 0)) {
        condval_9 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_9 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_7;
      if (((((((0 <= ((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))) && (((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) < 510)) && (0 <= (((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)))) && ((((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)) < 549)) && (0x1.47ae147ae147bp-7f/*1.000000e-02*/ < (condval_8 * condval_9))) && (0x0p+0f/*0.000000e+00*/ < (zz - prev)))) {
        float condval_10;
        if (((q % 2) == 0)) {
          condval_10 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
        } else {
          condval_10 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
        }
        float condval_11;
        if (((q >> 1) == 0)) {
          condval_11 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
        } else {
          condval_11 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
        }
        condval_7 = (powf(fminf(fmaxf(__fdiv_rn(sep, (zz - prev)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/), 0x1.ae2906p+0f/*1.680313e+00*/) * (condval_10 * condval_11));
      } else {
        condval_7 = 0x0p+0f/*0.000000e+00*/;
      }
      total = (total + condval_7);
      float condval_13;
      if (((q % 2) == 0)) {
        condval_13 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_13 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_14;
      if (((q >> 1) == 0)) {
        condval_14 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_14 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_12;
      if (((((((0 <= ((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))) && (((q >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) < 510)) && (0 <= (((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)))) && ((((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (q & 1)) < 549)) && (0x1.47ae147ae147bp-7f/*1.000000e-02*/ < (condval_13 * condval_14))) && (0x0p+0f/*0.000000e+00*/ < (zz - prev)))) {
        float condval_15;
        if (((q % 2) == 0)) {
          condval_15 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
        } else {
          condval_15 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
        }
        float condval_16;
        if (((q >> 1) == 0)) {
          condval_16 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
        } else {
          condval_16 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
        }
        condval_12 = (condval_15 * condval_16);
      } else {
        condval_12 = 0x0p+0f/*0.000000e+00*/;
      }
      ws = (ws + condval_12);
    }
    float condval_17;
    if ((0x0p+0f/*0.000000e+00*/ < ws)) {
      condval_17 = fminf(fmaxf((0x1p+0f/*1.000000e+00*/ - __fdiv_rn(total, ws)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/);
    } else {
      condval_17 = 0x0p+0f/*0.000000e+00*/;
    }
    float clip = condval_17;
    float condval_18;
    if ((549 <= ((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))) {
      condval_18 = Rec[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - 549)];
    } else {
      condval_18 = 0x0p+0f/*0.000000e+00*/;
    }
    float d0 = __fdiv_rn(P[1], (condval_18 - P[0]));
    float d1 = __fdiv_rn(P[1], (Rec[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] - P[0]));
    float condval_19;
    if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < 279441)) {
      condval_19 = Rec[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 549)];
    } else {
      condval_19 = 0x0p+0f/*0.000000e+00*/;
    }
    float d2 = __fdiv_rn(P[1], (condval_19 - P[0]));
    float clip_1 = (clip * ((float)(((d0 - d1) <= (d1 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)) || ((d1 - d2) <= (d2 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)))));
    float nx = MV[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))];
    float ny = MV[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)];
    float velocity = VM[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)];
    maximum = VM[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))];
    convergence = 0x1p+0f/*1.000000e+00*/;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
      int condval_21;
      if ((j < 3)) {
        condval_21 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_22;
        if ((5 < j)) {
          condval_22 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_22 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_21 = condval_22;
      }
      int condval_23;
      if ((j < 3)) {
        condval_23 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_24;
        if ((5 < j)) {
          condval_24 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_24 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_23 = condval_24;
      }
      int condval_25;
      if (((j % 3) < 1)) {
        condval_25 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_26;
        if ((1 < (j % 3))) {
          condval_26 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_26 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_25 = condval_26;
      }
      int condval_27;
      if (((j % 3) < 1)) {
        condval_27 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_28;
        if ((1 < (j % 3))) {
          condval_28 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_28 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_27 = condval_28;
      }
      float condval_20;
      if (((((0 <= condval_21) && (condval_23 < 510)) && (0 <= condval_25)) && (condval_27 < 549))) {
        int64_t condval_29;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_29 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_30;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_30 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_30 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_29 = condval_30;
        }
        int64_t condval_31;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_31 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_32;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_32 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_32 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_31 = condval_32;
        }
        int64_t condval_33;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_33 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_34;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_34 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_34 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_33 = condval_34;
        }
        int64_t condval_35;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_35 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_36;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_36 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_36 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_35 = condval_36;
        }
        condval_20 = MV[((condval_33 * (int64_t)1098) + (condval_35 * (int64_t)2))];
      } else {
        condval_20 = 0x0p+0f/*0.000000e+00*/;
      }
      float vx = condval_20;
      int condval_38;
      if ((j < 3)) {
        condval_38 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_39;
        if ((5 < j)) {
          condval_39 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_39 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_38 = condval_39;
      }
      int condval_40;
      if ((j < 3)) {
        condval_40 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_41;
        if ((5 < j)) {
          condval_41 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_41 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_40 = condval_41;
      }
      int condval_42;
      if (((j % 3) < 1)) {
        condval_42 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_43;
        if ((1 < (j % 3))) {
          condval_43 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_43 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_42 = condval_43;
      }
      int condval_44;
      if (((j % 3) < 1)) {
        condval_44 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_45;
        if ((1 < (j % 3))) {
          condval_45 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_45 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_44 = condval_45;
      }
      float condval_37;
      if (((((0 <= condval_38) && (condval_40 < 510)) && (0 <= condval_42)) && (condval_44 < 549))) {
        int64_t condval_46;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_46 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_47;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_47 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_47 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_46 = condval_47;
        }
        int64_t condval_48;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_48 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_49;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_49 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_49 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_48 = condval_49;
        }
        int64_t condval_50;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_50 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_51;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_51 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_51 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_50 = condval_51;
        }
        int64_t condval_52;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_52 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_53;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_53 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_53 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_52 = condval_53;
        }
        condval_37 = MV[(((condval_50 * (int64_t)1098) + (condval_52 * (int64_t)2)) + (int64_t)1)];
      } else {
        condval_37 = 0x0p+0f/*0.000000e+00*/;
      }
      float vy = condval_37;
      int condval_55;
      if ((j < 3)) {
        condval_55 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_56;
        if ((5 < j)) {
          condval_56 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_56 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_55 = condval_56;
      }
      int condval_57;
      if ((j < 3)) {
        condval_57 = max(((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) - 1), 0);
      } else {
        int condval_58;
        if ((5 < j)) {
          condval_58 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1), 509);
        } else {
          condval_58 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j / 3)) - 1);
        }
        condval_57 = condval_58;
      }
      int condval_59;
      if (((j % 3) < 1)) {
        condval_59 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_60;
        if ((1 < (j % 3))) {
          condval_60 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_60 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_59 = condval_60;
      }
      int condval_61;
      if (((j % 3) < 1)) {
        condval_61 = max((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 0);
      } else {
        int condval_62;
        if ((1 < (j % 3))) {
          condval_62 = min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1), 548);
        } else {
          condval_62 = (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j % 3)) - 1);
        }
        condval_61 = condval_62;
      }
      float condval_54;
      if (((((0 <= condval_55) && (condval_57 < 510)) && (0 <= condval_59)) && (condval_61 < 549))) {
        int64_t condval_63;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_63 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_64;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_64 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_64 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_63 = condval_64;
        }
        int64_t condval_65;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_65 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_66;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_66 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_66 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_65 = condval_66;
        }
        int64_t condval_67;
        if ((((int64_t)j) < (int64_t)3)) {
          condval_67 = max(((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_68;
          if (((int64_t)5 < ((int64_t)j))) {
            condval_68 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1), (int64_t)509);
          } else {
            condval_68 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) / (int64_t)549) + (((int64_t)j) / (int64_t)3)) - (int64_t)1);
          }
          condval_67 = condval_68;
        }
        int64_t condval_69;
        if (((((int64_t)j) % (int64_t)3) < (int64_t)1)) {
          condval_69 = max((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)0);
        } else {
          int64_t condval_70;
          if (((int64_t)1 < (((int64_t)j) % (int64_t)3))) {
            condval_70 = min((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1), (int64_t)548);
          } else {
            condval_70 = (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x))) % (int64_t)549) + (((int64_t)j) % (int64_t)3)) - (int64_t)1);
          }
          condval_69 = condval_70;
        }
        condval_54 = VM[((condval_67 * (int64_t)1098) + (condval_69 * (int64_t)2))];
      } else {
        condval_54 = 0x0p+0f/*0.000000e+00*/;
      }
      maximum = fmaxf(condval_54, maximum);
      convergence = fminf(convergence, (__fdiv_rn((__fdiv_rn(vx, maximum) * nx), maximum) + __fdiv_rn((__fdiv_rn(vy, maximum) * ny), maximum)));
    }
    float condval_71;
    if ((0x1.47ae147ae147bp-7f/*1.000000e-02*/ < velocity)) {
      condval_71 = convergence;
    } else {
      condval_71 = 0x1p+0f/*1.000000e+00*/;
    }
    convergence = condval_71;
    float divergence = (fminf(fmaxf((0x1p+0f/*1.000000e+00*/ - convergence), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * fminf(fmaxf((maximum * 0x1.9p+6f/*1.000000e+02*/), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
    float dist = DI[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)];
    float condval_72;
    if ((0x1p+0f/*1.000000e+00*/ < dist)) {
      condval_72 = ((0x1p+0f/*1.000000e+00*/ - fminf(fmaxf(__fdiv_rn(DI[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 2)], DI[((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3))]), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/)) * fminf(fmaxf((((dist * 0x1.999999999999ap-5f/*5.000000e-02*/) * (dist * 0x1.999999999999ap-5f/*5.000000e-02*/)) * (dist * 0x1.999999999999ap-5f/*5.000000e-02*/)), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
    } else {
      condval_72 = 0x0p+0f/*0.000000e+00*/;
    }
    float temporal = condval_72;
    mind = P[5];
    maxd = 0x0p+0f/*0.000000e+00*/;
    farfound = (signed char)0;
    #pragma unroll
    for (int j_1 = 0; j_1 < 9; ++j_1) {
      float condval_73;
      if (((((1 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_1 / 3))) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_1 / 3)) < 511)) && (1 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_1 % 3)))) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_1 % 3)) < 550))) {
        condval_73 = DD[((((((j_1 / 3) * 549) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)) + (j_1 % 3)) - 550)];
      } else {
        condval_73 = 0x0p+0f/*0.000000e+00*/;
      }
      float dd = ((__fdiv_rn(P[1], (condval_73 - P[0])) * P[4]) * ((float)((((1 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_1 / 3))) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_1 / 3)) < 511)) && (1 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_1 % 3)))) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_1 % 3)) < 550))));
      farfound = ((signed char)(((bool)farfound) || (dd == P[5])));
      mind = fminf(mind, dd);
      maxd = fmaxf(maxd, dd);
    }
    float depthdiv = ((0x1p+0f/*1.000000e+00*/ - __fdiv_rn(mind, maxd)) * ((float)!((bool)farfound)));
    float divergence_1 = fmaxf(divergence, fminf(fmaxf((temporal - depthdiv), 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/));
    rr = 0x0p+0f/*0.000000e+00*/;
    tc = divergence_1;
    #pragma unroll
    for (int j_2 = 0; j_2 < 9; ++j_2) {
      float r = C[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 1647) + (max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)) * 3))];
      float g = C[(((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 1647) + (max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)) * 3)) + 1)];
      float b = C[(((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 1647) + (max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)) * 3)) + 2)];
      float dot = (((C[((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3))] * r) + (C[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 2)] * b)) + (C[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)] * g));
      float similarity = __fdiv_rn(dot, fmaxf(CN[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))], CN[(((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 1098) + (max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)) * 2)) + 1)]));
      float pw = (0x1p+0f/*1.000000e+00*/ + (0x1.8p+2f/*6.000000e+00*/ - (similarity * 0x1.8p+2f/*6.000000e+00*/)));
      rr = fmaxf(rr, powf(React[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)))], (0x1p+0f/*1.000000e+00*/ + (0x1.8p+2f/*6.000000e+00*/ - (similarity * 0x1.8p+2f/*6.000000e+00*/)))));
      tc = fmaxf(tc, powf(Comp[((max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + (j_2 / 3)) - 1), 509)) * 549) + max(0, min((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549) + (j_2 % 3)) - 1), 548)))], (0x1p+0f/*1.000000e+00*/ + (0x1.8p+2f/*6.000000e+00*/ - (similarity * 0x1.8p+2f/*6.000000e+00*/)))));
    }
    Masks[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))] = (nearbyintf((fminf(fmaxf(rr, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * 0x1.fep+7f/*2.550000e+02*/)) * 0x1.010101010101p-8f/*3.921569e-03*/);
    Masks[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)] = (nearbyintf((fminf(fmaxf(tc, 0x0p+0f/*0.000000e+00*/), 0x1p+0f/*1.000000e+00*/) * 0x1.fep+7f/*2.550000e+02*/)) * 0x1.010101010101p-8f/*3.921569e-03*/);
    float r_1 = fminf(fmaxf(((fmaxf(C[((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3))], 0x0p+0f/*0.000000e+00*/) * P[6]) * P[7]), 0x0p+0f/*0.000000e+00*/), 0x1.ffcp+15f/*6.550400e+04*/);
    float g_1 = fminf(fmaxf(((fmaxf(C[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 1)], 0x0p+0f/*0.000000e+00*/) * P[6]) * P[7]), 0x0p+0f/*0.000000e+00*/), 0x1.ffcp+15f/*6.550400e+04*/);
    float b_1 = fminf(fmaxf(((fmaxf(C[(((((int)blockIdx.x) * 384) + (((int)threadIdx.x) * 3)) + 2)], 0x0p+0f/*0.000000e+00*/) * P[6]) * P[7]), 0x0p+0f/*0.000000e+00*/), 0x1.ffcp+15f/*6.550400e+04*/);
    half_t v_ = (half_t)(((0x1p-2f/*2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) + (0x1p-2f/*2.500000e-01*/ * b_1));
    ushort v__1 = (*(ushort *)(&(v_))) - ((ushort)(fabsf((((0x1p-2f/*2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) + (0x1p-2f/*2.500000e-01*/ * b_1))) < fabsf(((float)((half_t)(((0x1p-2f/*2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) + (0x1p-2f/*2.500000e-01*/ * b_1)))))));
    Prepared[((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4))] = ((float)(*(half_t *)(&(v__1))));
    half_t v__2 = (half_t)((0x1p-1f/*5.000000e-01*/ * r_1) - (0x1p-1f/*5.000000e-01*/ * b_1));
    ushort v__3 = (*(ushort *)(&(v__2))) - ((ushort)(fabsf(((0x1p-1f/*5.000000e-01*/ * r_1) - (0x1p-1f/*5.000000e-01*/ * b_1))) < fabsf(((float)((half_t)((0x1p-1f/*5.000000e-01*/ * r_1) - (0x1p-1f/*5.000000e-01*/ * b_1)))))));
    Prepared[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + 1)] = ((float)(*(half_t *)(&(v__3))));
    half_t v__4 = (half_t)(((-0x1p-2f/*-2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) - (0x1p-2f/*2.500000e-01*/ * b_1));
    ushort v__5 = (*(ushort *)(&(v__4))) - ((ushort)(fabsf((((-0x1p-2f/*-2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) - (0x1p-2f/*2.500000e-01*/ * b_1))) < fabsf(((float)((half_t)(((-0x1p-2f/*-2.500000e-01*/ * r_1) + (0x1p-1f/*5.000000e-01*/ * g_1)) - (0x1p-2f/*2.500000e-01*/ * b_1)))))));
    Prepared[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + 2)] = ((float)(*(half_t *)(&(v__5))));
    half_t v__6 = (half_t)(clip * ((float)(((d0 - d1) <= (d1 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)) || ((d1 - d2) <= (d2 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)))));
    ushort v__7 = (*(ushort *)(&(v__6))) - ((ushort)(fabsf((clip * ((float)(((d0 - d1) <= (d1 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)) || ((d1 - d2) <= (d2 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)))))) < fabsf(((float)((half_t)(clip * ((float)(((d0 - d1) <= (d1 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/)) || ((d1 - d2) <= (d2 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/))))))))));
    Prepared[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 4)) + 3)] = ((float)(*(half_t *)(&(v__7))));
  }
}

