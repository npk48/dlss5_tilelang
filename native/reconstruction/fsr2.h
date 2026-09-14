#pragma once
#include <cuda.h>
#include <memory>
namespace dlss5::reconstruction {
struct Settings {
  unsigned output_width = 0, output_height = 0;
  float camera_near = .1f, camera_far = 100.f, camera_fov_y = 1.0471975512f;
  unsigned encoding = 1;
  float sharpness = 0;
};
// Bounded metric depth / projection far distance; optional invalid mask is 1
// for metric values outside the camera range. All buffers have width*height
// floats.
void normalize_depth(const float *metric, float *normalized, float *invalid,
                     unsigned width, unsigned height, float near_m, float far_m,
                     CUstream stream);
// Tight device float HWC arrays. encoding 0=sRGB, 1=linear709. Depth in meters;
// motion is current->previous source pixels. Default reference variant: normal
// finite device projection, LDR accumulation, exposure/pre-exposure=1, analytic
// Lanczos. Optional RCAS sharpness [0,1]. No
// auto-exposure/HDR/display-MV/jittered-MV mode. Output uses the same encoding
// as input. No host image transfers; validation reads back only a scalar
// status. One host caller per Engine (not thread safe).
class Engine {
public:
  Engine(unsigned width, unsigned height, const Settings &settings);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  void reset();
  // Valid after successful process, source HWC2 current->previous PIXELS.
  const float *dilated_motion() const;
#ifdef D5_FSR_TEST_API
  const float *test_resource(unsigned slot) const;
#endif

  // Synchronously safe: waits only on caller stream, commits state after
  // success.
  void process(const float *color, const float *depth, const float *motion,
               const float *reactive, const float *composition, float *output,
               float *confidence, CUstream stream, float jitter_x = 0,
               float jitter_y = 0, float delta_ms = 16.666667f);

private:
  struct Impl;
  std::unique_ptr<Impl> p;
};
} // namespace dlss5::reconstruction
