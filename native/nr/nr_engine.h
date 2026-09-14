#pragma once
#include <cuda.h>
#include <filesystem>
#include <memory>

namespace dlss5::nr
{
// Windows x64, SM89 only. The caller must make device's primary CUDA context
// current for construction, prepare and infer. All storage/modules belong to it.
// model_dir contains the original weights_ht_blob.bin (no derived weight pack).
// Set NATIVE_NR_TOOLCHAIN to the CUDA 12.8 root containing nvrtc/bin,
// runtime/include and cccl/include. No Python, Torch or reference sources needed.
struct AuditStats
{
    unsigned long long nvrtc_compiles = 0, module_loads = 0, prepares = 0, kernel_pack_loads = 0;
    double last_prepare_ms = 0;
    double compile_ms = 0, module_load_ms = 0, upload_ms = 0, kernel_pack_load_ms = 0;
    double last_layout_allocation_ms = 0;
};
class Engine
{
  public:
    explicit Engine(const std::filesystem::path &model_dir, int device);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    Engine(Engine &&) = delete;
    Engine &operator=(Engine &&) = delete;
    // Synchronous preparation, one retained shape; same-shape calls are cheap.
    // Eight geometry-independent modules compile once per Engine, not per shape.
    // New geometry/table constants activate on the first infer's caller stream.
    // H and W >=64 and multiples of 64, subject to memory/int32 address limits.
    void prepare(int h, int w);
    AuditStats audit() const;
    // packet: contiguous float32 [1,16,H,W]; head: float32 [1,H,W,4].
    // Both caller-owned, >=16-byte aligned, disjoint, in the retained context.
    // Requires a matching prepare; no allocations or compilation during infer.
    // Every kernel/copy/memset uses stream (including non-default streams).
    // The original nonfinite input/head status is retained: the final four-byte
    // readback necessarily synchronizes THIS stream before return, then throws
    // on invalid data. Thus this API is not fully asynchronous/graph-capturable.
    // Calls on one Engine must be externally serialized. Separate Engines own
    // independent workspaces. Output remains valid across later calls/prepares.
    void infer(CUdeviceptr packet, CUdeviceptr head, int h, int w, CUstream stream);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace dlss5::nr
