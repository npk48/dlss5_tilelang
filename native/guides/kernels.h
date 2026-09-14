#pragma once
#include <cuda_runtime_api.h>
namespace dlss5::guides::detail {
void flow_prepare(const float*, float*, int h, int w, int rh, int rw, int nh, int nw, cudaStream_t);
void depth_prepare(const float*, float*, int h, int w, int nh, int nw, cudaStream_t);
void flow_finish(const float*, float*, int h, int w, int rh, int rw, int nh, int nw, cudaStream_t);
void depth_finish(const float*, float*, int h, int w, int nh, int nw, cudaStream_t);
void cache_insert(const float*, float*, int spatial, int channels, int slot, cudaStream_t);
}
