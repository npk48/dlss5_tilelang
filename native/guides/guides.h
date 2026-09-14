#pragma once
#include <cuda.h>
#include <cstdint>
#include <memory>
#include <string>

namespace dlss5::guides {
struct Config {
    bool flow = true;
    bool depth = true;
    int flow_updates = 8;
    int flow_longest_side = 512;
    int depth_input_size = 518;
};

// CUDA primary context for device; no Python, Torch, or host image staging.
// Non-copyable, not thread-safe. Calls on one Engine use the same caller stream
// until reset(). Buffers must be distinct, contiguous device float32, and remain
// valid until that stream completes. RGB is HWC3 sRGB [0,1].
// Models are loaded lazily. Missing assets/backend throw std::runtime_error;
// disabled components throw, never manufacture a guide. FP32 neural inference.
class Engine {
public:
    Engine(const std::string& model_dir, int device, Config config = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // HWC2 current -> previous motion in SOURCE-image pixels. Stateless.
    void estimate_flow(const float* current, const float* previous, int h, int w,
                       float* motion, CUstream stream);
    // HWC1 estimated metric depth (meters), full Small causal streaming model.
    // Successful calls advance this Engine's independent hidden-state history.
    // A source-size change requires reset. If inference throws, reset before reuse.
    void estimate_depth(const float* current, int h, int w, float* depth, CUstream stream);
    // Waits for in-flight work on this Engine's stream, releases history/sessions.
    // Caller stream must outlive Engine or be released only after reset().
    void reset();
    // Same shape/stream: discard temporal history but retain loaded ONNX sessions.
    void reset_history();
    std::uint64_t depth_frames() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace dlss5::guides
