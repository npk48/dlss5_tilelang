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

extern "C" __global__ void main_kernel(const float* __restrict__ Color, const float* __restrict__ D, float* __restrict__ Dilated, float* __restrict__ Luma, const float* __restrict__ MV, const float* __restrict__ Magnitude, float* __restrict__ Nearest, float* __restrict__ Rec);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const float* __restrict__ Color, const float* __restrict__ D, float* __restrict__ Dilated, float* __restrict__ Luma, const float* __restrict__ MV, const float* __restrict__ Magnitude, float* __restrict__ Nearest, float* __restrict__ Rec) {
  float nearest = 0x0p+0f/*0.000000e+00*/;
  int cx = 0;
  int cy = 0;
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < 139995) {
    nearest = D[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
    cx = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549);
    cy = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      int condval_1;
      if ((((i == 1) || (i == 4)) || (i == 5))) {
        condval_1 = 1;
      } else {
        int condval_2;
        if ((((i == 2) || (i == 6)) || (i == 7))) {
          condval_2 = -1;
        } else {
          condval_2 = 0;
        }
        condval_1 = condval_2;
      }
      int condval_3;
      if ((((i == 1) || (i == 4)) || (i == 5))) {
        condval_3 = 1;
      } else {
        int condval_4;
        if ((((i == 2) || (i == 6)) || (i == 7))) {
          condval_4 = -1;
        } else {
          condval_4 = 0;
        }
        condval_3 = condval_4;
      }
      int condval_5;
      if ((((i == 0) || (i == 5)) || (i == 7))) {
        condval_5 = 1;
      } else {
        int condval_6;
        if ((((i == 3) || (i == 4)) || (i == 6))) {
          condval_6 = -1;
        } else {
          condval_6 = 0;
        }
        condval_5 = condval_6;
      }
      int condval_7;
      if ((((i == 0) || (i == 5)) || (i == 7))) {
        condval_7 = 1;
      } else {
        int condval_8;
        if ((((i == 3) || (i == 4)) || (i == 6))) {
          condval_8 = -1;
        } else {
          condval_8 = 0;
        }
        condval_7 = condval_8;
      }
      float condval;
      if (((((0 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + condval_1)) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + condval_3) < 510)) && (0 <= (condval_5 + (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)))) && ((condval_7 + (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) < 549))) {
        int64_t condval_9;
        if ((((((int64_t)i) == (int64_t)1) || (((int64_t)i) == (int64_t)4)) || (((int64_t)i) == (int64_t)5))) {
          condval_9 = (int64_t)1;
        } else {
          int64_t condval_10;
          if ((((((int64_t)i) == (int64_t)2) || (((int64_t)i) == (int64_t)6)) || (((int64_t)i) == (int64_t)7))) {
            condval_10 = (int64_t)-1;
          } else {
            condval_10 = (int64_t)0;
          }
          condval_9 = condval_10;
        }
        int64_t condval_11;
        if ((((((int64_t)i) == (int64_t)0) || (((int64_t)i) == (int64_t)5)) || (((int64_t)i) == (int64_t)7))) {
          condval_11 = (int64_t)1;
        } else {
          int64_t condval_12;
          if ((((((int64_t)i) == (int64_t)3) || (((int64_t)i) == (int64_t)4)) || (((int64_t)i) == (int64_t)6))) {
            condval_12 = (int64_t)-1;
          } else {
            condval_12 = (int64_t)0;
          }
          condval_11 = condval_12;
        }
        int64_t condval_13;
        if ((((((int64_t)i) == (int64_t)1) || (((int64_t)i) == (int64_t)4)) || (((int64_t)i) == (int64_t)5))) {
          condval_13 = (int64_t)1;
        } else {
          int64_t condval_14;
          if ((((((int64_t)i) == (int64_t)2) || (((int64_t)i) == (int64_t)6)) || (((int64_t)i) == (int64_t)7))) {
            condval_14 = (int64_t)-1;
          } else {
            condval_14 = (int64_t)0;
          }
          condval_13 = condval_14;
        }
        int64_t condval_15;
        if ((((((int64_t)i) == (int64_t)0) || (((int64_t)i) == (int64_t)5)) || (((int64_t)i) == (int64_t)7))) {
          condval_15 = (int64_t)1;
        } else {
          int64_t condval_16;
          if ((((((int64_t)i) == (int64_t)3) || (((int64_t)i) == (int64_t)4)) || (((int64_t)i) == (int64_t)6))) {
            condval_16 = (int64_t)-1;
          } else {
            condval_16 = (int64_t)0;
          }
          condval_15 = condval_16;
        }
        condval = D[((((condval_13 * (int64_t)549) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x))) + condval_15)];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      float d = condval;
      int condval_17;
      if ((((i == 1) || (i == 4)) || (i == 5))) {
        condval_17 = 1;
      } else {
        int condval_18;
        if ((((i == 2) || (i == 6)) || (i == 7))) {
          condval_18 = -1;
        } else {
          condval_18 = 0;
        }
        condval_17 = condval_18;
      }
      int condval_19;
      if ((((i == 1) || (i == 4)) || (i == 5))) {
        condval_19 = 1;
      } else {
        int condval_20;
        if ((((i == 2) || (i == 6)) || (i == 7))) {
          condval_20 = -1;
        } else {
          condval_20 = 0;
        }
        condval_19 = condval_20;
      }
      int condval_21;
      if ((((i == 0) || (i == 5)) || (i == 7))) {
        condval_21 = 1;
      } else {
        int condval_22;
        if ((((i == 3) || (i == 4)) || (i == 6))) {
          condval_22 = -1;
        } else {
          condval_22 = 0;
        }
        condval_21 = condval_22;
      }
      int condval_23;
      if ((((i == 0) || (i == 5)) || (i == 7))) {
        condval_23 = 1;
      } else {
        int condval_24;
        if ((((i == 3) || (i == 4)) || (i == 6))) {
          condval_24 = -1;
        } else {
          condval_24 = 0;
        }
        condval_23 = condval_24;
      }
      bool take = (((((0 <= ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + condval_17)) && (((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + condval_19) < 510)) && (0 <= (condval_21 + (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)))) && ((condval_23 + (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) < 549)) && (d < nearest));
      int condval_25;
      if (take) {
        int condval_26;
        if ((((i == 0) || (i == 5)) || (i == 7))) {
          condval_26 = 1;
        } else {
          int condval_27;
          if ((((i == 3) || (i == 4)) || (i == 6))) {
            condval_27 = -1;
          } else {
            condval_27 = 0;
          }
          condval_26 = condval_27;
        }
        condval_25 = (condval_26 + (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549));
      } else {
        condval_25 = cx;
      }
      cx = condval_25;
      int condval_28;
      if (take) {
        int condval_29;
        if ((((i == 1) || (i == 4)) || (i == 5))) {
          condval_29 = 1;
        } else {
          int condval_30;
          if ((((i == 2) || (i == 6)) || (i == 7))) {
            condval_30 = -1;
          } else {
            condval_30 = 0;
          }
          condval_29 = condval_30;
        }
        condval_28 = ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549) + condval_29);
      } else {
        condval_28 = cy;
      }
      cy = condval_28;
      float condval_31;
      if (take) {
        condval_31 = d;
      } else {
        condval_31 = nearest;
      }
      nearest = condval_31;
    }
    int qx = cx;
    int qy = cy;
    float condval_32;
    if (((((0 <= qy) && (qy < 510)) && (0 <= qx)) && (qx < 549))) {
      condval_32 = MV[((((int64_t)qy) * (int64_t)1098) + (((int64_t)qx) * (int64_t)2))];
    } else {
      condval_32 = 0x0p+0f/*0.000000e+00*/;
    }
    float dx = condval_32;
    float condval_33;
    if (((((0 <= qy) && (qy < 510)) && (0 <= qx)) && (qx < 549))) {
      condval_33 = MV[(((((int64_t)qy) * (int64_t)1098) + (((int64_t)qx) * (int64_t)2)) + (int64_t)1)];
    } else {
      condval_33 = 0x0p+0f/*0.000000e+00*/;
    }
    float dy = condval_33;
    float condval_34;
    if (((((0 <= qy) && (qy < 510)) && (0 <= qx)) && (qx < 549))) {
      condval_34 = Magnitude[((((int64_t)qy) * (int64_t)549) + ((int64_t)qx))];
    } else {
      condval_34 = 0x0p+0f/*0.000000e+00*/;
    }
    bool moving = (0x1.999999999999ap-4f/*1.000000e-01*/ < condval_34);
    float ux = (__fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.128p+9f/*5.490000e+02*/) + (dx * ((float)moving)));
    float uy = (__fdiv_rn((((float)(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 549)) + 0x1p-1f/*5.000000e-01*/), 0x1.fep+8f/*5.100000e+02*/) + (dy * ((float)moving)));
    float px = ((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/);
    float py = ((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/);
    float fx = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
    float fy = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      float condval_35;
      if (((i_1 % 2) == 0)) {
        condval_35 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_35 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_36;
      if (((i_1 >> 1) == 0)) {
        condval_36 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_36 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float wt = (condval_35 * condval_36);
      float condval_37;
      if (((i_1 % 2) == 0)) {
        condval_37 = (0x1p+0f/*1.000000e+00*/ - (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_37 = (((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      float condval_38;
      if (((i_1 >> 1) == 0)) {
        condval_38 = (0x1p+0f/*1.000000e+00*/ - (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))));
      } else {
        condval_38 = (((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/) - ((float)((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))));
      }
      if (((((0 <= ((i_1 >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))))) && (((i_1 >> 1) + ((int)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) < 510)) && (0 <= (((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (i_1 & 1)))) && ((((int)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/))) + (i_1 & 1)) < 549)) && (0x1.47ae147ae147bp-7f/*1.000000e-02*/ < (condval_37 * condval_38))) {
        AtomicMin((&(Rec[(((((((int64_t)i_1) >> (int64_t)1) * (int64_t)549) + (((int64_t)floorf(((uy * 0x1.fep+8f/*5.100000e+02*/) - 0x1p-1f/*5.000000e-01*/))) * (int64_t)549)) + ((int64_t)floorf(((ux * 0x1.128p+9f/*5.490000e+02*/) - 0x1p-1f/*5.000000e-01*/)))) + (((int64_t)i_1) & (int64_t)1))])), nearest, 0);
      }
    }
    float lum = Color[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
    float condval_39;
    if ((lum <= 0x1.22354d28f7cd6p-7f/*8.856452e-03*/)) {
      condval_39 = (lum * 0x1.c3a5ed097b426p+9f/*9.032963e+02*/);
    } else {
      condval_39 = ((powf(lum, 0x1.5555555555555p-2f/*3.333333e-01*/) * 0x1.dp+6f/*1.160000e+02*/) - 0x1p+4f/*1.600000e+01*/);
    }
    float perceived = (condval_39 * 0x1.47ae147ae147bp-7f/*1.000000e-02*/);
    Nearest[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = nearest;
    half_t v_ = (half_t)dx;
    ushort v__1 = (*(ushort *)(&(v_))) - ((ushort)(fabsf(dx) < fabsf(((float)((half_t)dx)))));
    Dilated[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))] = ((float)(*(half_t *)(&(v__1))));
    half_t v__2 = (half_t)dy;
    ushort v__3 = (*(ushort *)(&(v__2))) - ((ushort)(fabsf(dy) < fabsf(((float)((half_t)dy)))));
    Dilated[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)) + 1)] = ((float)(*(half_t *)(&(v__3))));
    half_t v__4 = (half_t)powf(perceived, 0x1.5555555555555p-3f/*1.666667e-01*/);
    ushort v__5 = (*(ushort *)(&(v__4))) - ((ushort)(fabsf(powf(perceived, 0x1.5555555555555p-3f/*1.666667e-01*/)) < fabsf(((float)((half_t)powf(perceived, 0x1.5555555555555p-3f/*1.666667e-01*/))))));
    Luma[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((float)(*(half_t *)(&(v__5))));
  }
}

