#include "fsr2.h"
#include <exception>
#include <cuda_runtime.h>
using namespace dlss5::reconstruction;

extern "C" {
__declspec(dllexport) void *fsr_create(int w, int h, int ow, int oh, float far,
                                       int encoding, float sharpness) {
  try {
    Settings s;
    s.output_width = ow;
    s.output_height = oh;
    s.camera_far = far;
    s.encoding = encoding;
    s.sharpness = sharpness;
    return new Engine(w, h, s);
  } catch (...) {
    return nullptr;
  }
}
__declspec(dllexport) int fsr_process(void *e, const float *c, const float *d,
                                      const float *m, const float *r,
                                      const float *t, float *out,
                                      float *confidence, void *stream, float jx,
                                      float jy) {
  try {
    static_cast<Engine *>(e)->process(c, d, m, r, t, out, confidence,
                                      reinterpret_cast<CUstream>(stream), jx,
                                      jy);
    return 0;
  } catch (const std::exception &) {
    return 1;
  }
}
__declspec(dllexport) void fsr_reset(void *e) {
  static_cast<Engine *>(e)->reset();
}
__declspec(dllexport) const float *fsr_motion(void *e) {
  return static_cast<Engine *>(e)->dilated_motion();
}
__declspec(dllexport) int fsr_copy_motion(void* e,float* output,int count,void* stream){return cudaMemcpyAsync(output,static_cast<Engine*>(e)->dilated_motion(),count*2*sizeof(float),cudaMemcpyDeviceToDevice,reinterpret_cast<cudaStream_t>(stream));}
__declspec(dllexport) void fsr_normalize(const float* d,float* n,float* invalid,int w,int h,float near,float far,void* stream){normalize_depth(d,n,invalid,w,h,near,far,reinterpret_cast<CUstream>(stream));}
__declspec(dllexport) int fsr_copy_resource(void*e,unsigned slot,float*out,int count,void*stream){return cudaMemcpyAsync(out,static_cast<Engine*>(e)->test_resource(slot),count*sizeof(float),cudaMemcpyDeviceToDevice,reinterpret_cast<cudaStream_t>(stream));}
__declspec(dllexport) void fsr_destroy(void *e) {
  delete static_cast<Engine *>(e);
}
}
