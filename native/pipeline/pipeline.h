#pragma once
#include "dlss5.h"
#include <cuda.h>
#include <cstdint>
#include <memory>

namespace dlss5::pipeline {
// All image views are device F32 HWC (packet is CHW16). Row/plane strides
// are honored. Calls enqueue on stream and throw std::exception on errors.
// Caller validates finite image values (and SDR/PQ/mask ranges) on device;
// these asynchronous helpers validate metadata/settings, not resource contents.
// Caller must keep inputs alive until stream completion; one state per session,
// one ordered stream. No device image is copied to the host.
struct ColorOptions {
    uint32_t input_encoding = D5_COLOR_SRGB;
    uint32_t output_encoding = D5_COLOR_SRGB;
    float reference_white_nits = 203.f;
    float peak_nits = 1000.f;
};
enum class Reconstruction { Bilinear = 0, FSR2 = 1, EASU = 2 };
class Ops {
public:
    static bool supports(Reconstruction type) noexcept;
    // Decode source RGB(A) to same-size linear709 context RGB, preserving HDR
    // units (nits). Alpha remains in source for finish().
    static void preprocess(const D5Tensor& source, const D5Tensor& context,
                           const ColorOptions&, CUstream);
    // Linear context -> neural encoded RGB; same extents.
    static void encode(const D5Tensor& context, const D5Tensor& encoded,
                       const ColorOptions&, CUstream);
    // Explicit spatial bilinear, not FSR2. antialias implements PyTorch's
    // separable triangle downsample; disabled for masks/flow.
    static void resize(const D5Tensor& source, const D5Tensor& target,
                       CUstream, bool antialias = true);
    static void reconstruct(const D5Tensor& source, const D5Tensor& target,
                            Reconstruction, CUstream);
    // Encoded-domain NR residual transport: reference + resize(result-base).
    static void transport(const D5Tensor& reference, const D5Tensor& result,
                          const D5Tensor& base, const D5Tensor& target, CUstream);
    // Context/reference/modified use output extents. protect may be absent or
    // HWC1 at output extents; original may be source-size RGB(A), alpha is
    // bilinear resized. Output HWC RGB(A), keeps source alpha (or opaque 1).
    static void finish(const D5Tensor& context, const D5Tensor& reference,
                       const D5Tensor& modified, const D5Tensor& original,
                       const D5Tensor& protect, const D5Tensor& output,
                       const ColorOptions&, float mix, CUstream);
    static void half_rtz(const D5Tensor& source, const D5Tensor& target, CUstream);
};
class PipelineState {
public:
    PipelineState();
    ~PipelineState();
    PipelineState(const PipelineState&) = delete;
    PipelineState& operator=(const PipelineState&) = delete;
    // Allocates at prepare time, neural H=max(320,ceil(H/64)*64),
    // W=max(320,ceil((W+64)/64)*64), matching NRChain.
    void allocate(uint32_t work_width, uint32_t work_height);
    void reset() noexcept;
    // Work is encoded HWC3 at allocated work extents. Motion is HWC2
    // current->previous in pixel units of motion_units_width/height (0 means
    // motion view extent); nearest-exact sample. Confidence HWC1 is bilinear
    // sampled to work extents, absent means 1. Missing motion explicitly
    // disables temporal reuse. No history is changed until commit().
    void prepare(const D5Tensor& work, const D5Tensor& motion,
                 const D5Tensor& confidence, const D5NRSettings&, bool reset,
                 CUstream, uint32_t motion_units_width = 0,
                 uint32_t motion_units_height = 0);
    D5Tensor packet() const;
    D5Tensor current() const; // padded RTZ current (intensity reference)
    D5Tensor sampled_current() const;
    D5Tensor warped_history() const;
    D5Tensor gate() const;
    D5Tensor history() const; // committed half-RTZ history
    // HWC4 head, residual RGB + history logit. Produces padded RGB plus next
    // half-RTZ history; output() crops via a stride-aware view, no copy.
    void composite(const D5Tensor& head, CUstream);
    D5Tensor output() const;
    D5Tensor padded_output() const;
    void commit(); // call only after all frame stages have succeeded
    uint32_t neural_width() const;
    uint32_t neural_height() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
// Faithful non-geometry DLSS5-Feeder decisions; geometry fit is not supported.
struct GuideOptions {
    bool validate=true, static_test=true, luma=false, depth=true, consistency=true;
    bool geometry=false, geometry_diagnostics=false;
    float static_bias=.15f, min_contrast=.012f, luma_tolerance=.25f;
    float depth_tolerance=.10f, mv_consistency=1.4f, mask_strength=1.f;
    float sign_x=1.f, sign_y=1.f, mv_scale=1.f;
};
class GuideState {
public:
    GuideState(); ~GuideState();
    GuideState(const GuideState&)=delete;
    GuideState& operator=(const GuideState&)=delete;
    void allocate(uint32_t width,uint32_t height);
    void reset() noexcept;
    // Inputs on same color grid: encoded RGB (alpha already over black if
    // required), current->previous PIXEL flow, normalized linear depth [0,1]
    // (view-Z meters / declared camera far, NOT per-frame min/max fitting).
    // Caller is responsible for finite resource values and valid depth range.
    void prepare(const D5Tensor& color,const D5Tensor& motion,
                 const D5Tensor& normalized_depth,bool reset,CUstream,
                 const GuideOptions& options=GuideOptions{});
    D5Tensor motion() const; // FP16 RN publication, source-pixel units
    D5Tensor distrust() const; // R8_UNORM RN publication represented F32
    D5Tensor tests() const; // HWC4 luma/depth/consistency/static
    void commit(); // retain FP16 RN input luma/depth/provider UV only
private:
    struct Impl; std::unique_ptr<Impl> p_;
};
} // namespace dlss5::pipeline
