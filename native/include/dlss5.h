#ifndef DLSS5_NATIVE_H
#define DLSS5_NATIVE_H
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
# if defined(D5_STATIC)
#  define D5_API
# elif defined(D5_BUILD)
#  define D5_API __declspec(dllexport)
# else
#  define D5_API __declspec(dllimport)
# endif
# define D5_CALL __cdecl
#else
# define D5_API
# define D5_CALL
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define D5_ABI_VERSION 1u
typedef struct D5SessionImpl* D5Session;
typedef int32_t D5Status;
enum { D5_OK=0,D5_INVALID_ARGUMENT=1,D5_UNAVAILABLE=2,D5_CUDA_ERROR=3,D5_INTERNAL_ERROR=4,D5_BUSY=5 };
enum { D5_F32=1,D5_F16=2,D5_U8=3 };
enum { D5_HWC=1,D5_CHW=2 };
enum { D5_STAGE_DEPTH=0,D5_STAGE_FLOW=1,D5_STAGE_GUIDE=2,D5_STAGE_RECONSTRUCT=3,D5_STAGE_PACKET=4,D5_STAGE_NR=5,D5_STAGE_OUTPUT=6,D5_STAGE_COUNT=7 };
enum { D5_ENABLE_DEPTH=1u<<D5_STAGE_DEPTH,D5_ENABLE_FLOW=1u<<D5_STAGE_FLOW,D5_ENABLE_GUIDE=1u<<D5_STAGE_GUIDE,D5_ENABLE_RECONSTRUCT=1u<<D5_STAGE_RECONSTRUCT,D5_ENABLE_NR=1u<<D5_STAGE_NR };
enum { D5_COLOR_SRGB=0,D5_COLOR_LINEAR709=1,D5_COLOR_LINEAR709_NITS=2,D5_COLOR_PQ2020=3 };
/* CUDA device address only. row_stride_bytes=0 means tight; plane_stride_bytes
   is used for CHW. No hidden host upload/download. All dimensions are pixels. */
typedef struct D5Tensor {
 uint32_t struct_size; uint32_t dtype; uint32_t layout; uint32_t channels;
 uint32_t width; uint32_t height; uint64_t data;
 uint64_t row_stride_bytes; uint64_t plane_stride_bytes;
} D5Tensor;
typedef struct D5NRSettings {
 uint32_t struct_size; uint32_t style; float structure; float tone;
 float skin; uint32_t automatic_mask; float temporal_strength; float intensity;
} D5NRSettings;
typedef struct D5GuideSettings {
 uint32_t struct_size; uint32_t validate,static_test,luma_test,depth_test,consistency_test;
 float static_bias,min_contrast,luma_tolerance,depth_tolerance,mv_consistency,mask_strength;
 float sign_x,sign_y,mv_scale;
} D5GuideSettings;
typedef struct D5Config {
 uint32_t struct_size; uint32_t abi_version; int32_t device; uint32_t enabled_stages;
 const char* model_directory_utf8;
 uint32_t output_width,output_height; uint32_t nr_width,nr_height;
 uint32_t flow_updates,flow_longest_side,depth_input_size;
 uint32_t input_encoding,output_encoding;
 float reference_white_nits,peak_nits,camera_fov_y,output_mix;
 uint32_t nr_pass_count; const D5NRSettings* nr_passes;
 float camera_near_m,camera_far_m; const D5GuideSettings* guide_settings;
 uint32_t reserved[12];
} D5Config;
typedef struct D5Frame {
 uint32_t struct_size; uint32_t reset; uint64_t frame_id;
 void* cuda_stream; /* CUstream; nullptr means legacy default stream */
 D5Tensor color; /* required HWC RGB or RGBA */
 D5Tensor depth; /* optional HWC1 positive meters, replaces built-in estimate */
 D5Tensor motion; /* optional HWC2 current->previous source-pixel units */
 D5Tensor confidence; /* optional HWC1 [0,1] */
 D5Tensor reactive; /* optional FSR reactive mask [0,1] on source grid */
 D5Tensor composition; /* optional FSR composition mask [0,1] on source grid */
 D5Tensor protect; /* optional output protection, NOT learned tone/structure mask */
 D5Tensor output; /* required caller-owned output HWC RGB/RGBA */
 float delta_ms; float jitter_x,jitter_y;
 uint32_t reserved[8];
} D5Frame;
/* Replacement callbacks enqueue work on cuda_stream. Buffers are valid only
   for this invocation; callback must fill provided outputs before returning
   (GPU work may remain asynchronous on that stream). Host callbacks must not
   recursively invoke a function on the same session. Nonzero aborts this frame.
   Slot meanings are documented in docs/NATIVE_SDK.md. */
typedef struct D5StageInvocation {
 uint32_t struct_size; uint32_t stage; uint64_t frame_id; uint32_t reset; uint32_t pass_index;
 void* cuda_stream; float delta_ms,jitter_x,jitter_y; uint32_t input_count,output_count;
 const D5Tensor* inputs; D5Tensor* outputs; const D5NRSettings* nr_settings;
} D5StageInvocation;
typedef D5Status (D5_CALL *D5StageCallback)(void* user,const D5StageInvocation* invocation);
typedef struct D5Plugin {
 uint32_t struct_size; uint32_t stage; D5StageCallback process; void* user;
} D5Plugin;
typedef struct D5Capabilities {
 uint32_t struct_size; uint32_t abi_version; uint32_t built_in_stages;
 uint32_t replacement_stages; uint32_t compute_major,compute_minor;
 uint32_t asynchronous; uint32_t reserved[8];
} D5Capabilities;
D5_API uint32_t D5_CALL d5_abi_version(void);
D5_API const char* D5_CALL d5_last_error(void); /* thread-local, until next API call */
D5_API void D5_CALL d5_default_config(D5Config* config);
D5_API void D5_CALL d5_default_nr_settings(D5NRSettings* settings);
D5_API void D5_CALL d5_default_guide_settings(D5GuideSettings* settings);
D5_API D5Status D5_CALL d5_create(const D5Config* config,D5Session* session);
D5_API D5Status D5_CALL d5_capabilities(D5Session session,D5Capabilities* capabilities);
D5_API D5Status D5_CALL d5_set_plugin(D5Session session,const D5Plugin* plugin);
D5_API D5Status D5_CALL d5_set_nr_settings(D5Session session,const D5NRSettings* settings,uint32_t count);
D5_API D5Status D5_CALL d5_set_output_mix(D5Session session,float mix);
/* Call before frame submission to pay shape compilation/loading up front. */
D5_API D5Status D5_CALL d5_prepare(D5Session session,uint32_t input_width,uint32_t input_height);
D5_API D5Status D5_CALL d5_process(D5Session session,const D5Frame* frame);
D5_API D5Status D5_CALL d5_prepare_packet(D5Session session,uint32_t width,uint32_t height);
/* Direct low-overhead learned network boundary, bypassing all host pipeline.
   Float32 contiguous CHW16 -> HWC4, 64-aligned dimensions, batch 1. */
D5_API D5Status D5_CALL d5_infer_packet(D5Session session,const D5Tensor* packet,const D5Tensor* head,void* cuda_stream);
/* Borrowed last-frame device view, valid until next process/prepare/reset/destroy. */
D5_API D5Status D5_CALL d5_stage_output(D5Session session,uint32_t stage,uint32_t index,D5Tensor* output);
D5_API D5Status D5_CALL d5_synchronize(D5Session session);
D5_API D5Status D5_CALL d5_reset(D5Session session);
D5_API D5Status D5_CALL d5_destroy(D5Session session);
#ifdef __cplusplus
}
#endif
#endif
