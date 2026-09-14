# Native Windows CUDA NR

`dlss5_nr` is a C++17 static library implementing the full original 71-layer CUDA NR path. It reads `model_dir/weights_ht_blob.bin` directly. The original math, FP8 formats and MMA architecture are retained. Spatial dimensions are runtime module constants; shape-dependent routes/addresses are CUDA buffers, not generated CUDA literals. CMake embeds the generic sources for developer fallback, while releases load the eight generic kernel packs from `runtime/nr/sm89`.

## Build and integrate

Standalone, from the repository root (VS2022 x64, installed CUDA 12.2 headers/driver import library):

```powershell
cmake -S native/nr -B native/nr/build -G "Visual Studio 17 2022" -A x64 -DDLSS5_NR_BUILD_SMOKE=ON
cmake --build native/nr/build --config Release --parallel 6
$env:NATIVE_NR_TOOLCHAIN = (Resolve-Path .toolchains/cuda12.8).Path
```

SDK CMake integration:

```cmake
add_subdirectory(native/nr)
target_link_libraries(your_sdk PRIVATE dlss5_nr)
```

The target exports its include directory and `CUDA::cuda_driver`. If that CUDA target does not already exist, the subdirectory calls `find_package(CUDAToolkit REQUIRED)`. No CUDA language enablement, nvcc kernel compilation, Python or Torch is needed by this build.

```cpp
#include <nr_engine.h>

// Caller retains the selected device's primary context and makes it current.
// packet: caller-owned, contiguous float32 [1,16,H,W].
// head: caller-owned, contiguous float32 [1,H,W,4].
// Both allocations must be disjoint and at least 16-byte aligned.
dlss5::nr::Engine nr(model_dir, device_ordinal);
nr.prepare(H, W);
nr.infer(packet, head, H, W, caller_stream);
```

`prepare` accepts already-prepared H/W >=64, both multiples of 64, subject to available memory and signed 32-bit descriptor limits. It does not resize/pad the public input. The original internal DS14 axis crossing, DS22 padding and deep/ViT extent rules are retained. A single prepared shape is retained per Engine. Repeating `prepare` for the same shape is inexpensive; a different shape rebuilds runtime tables/workspace but reuses the same generic modules. There is no allowed-size catalog. **Call `prepare` explicitly before `infer`**; inference never compiles or allocates model/workspace storage.

## Deployment and execution semantics

Deploy the host executable/library, original `weights_ht_blob.bin`, MSVC runtime, NVIDIA driver, and a CUDA **12.8** compiler directory containing:

- `nvrtc/bin/nvrtc64_120_0.dll` and `nvrtc-builtins64_128.dll`;
- `runtime/include/` and `cccl/include/`, including their subordinate headers.

Set `NATIVE_NR_TOOLCHAIN` to that directory. `DLSS5_FP8_TOOLCHAIN` is a secondary override; otherwise `.toolchains/cuda12.8` is resolved relative to the process working directory. The native library dynamically loads NVRTC and verifies version 12.8. It requires no `.cu` files, Python scripts, Python DLL, Torch, model decoder cache, or reference outputs at deployment. Deploy the qualified generic packs described in `kernels/README.md` to skip NVRTC even on first preparation. Without a matching pack, each generic module is compiled once per Engine; this developer fallback is slow, but changing geometry never recompiles it. Packs contain source/payload SHA256 and are independent of H/W.

- **SM89 only**, as in the reference. Other architectures raise an error; there is no fallback.
- Construction/prepare/infer require the caller's selected **primary** CUDA context to be current. Foreign contexts, foreign buffers/streams, overlapping buffers and unmatched shapes are rejected. Do not reset the primary context while the Engine is alive.
- All kernels, completion-counter clears and the status copy use the caller's stream, including non-default streams. The original completion publications are retained in owned storage, with one slab clear per inference.
- **`infer` is not fully asynchronous.** The original input/head validation reads back a four-byte status word into pinned host memory and synchronizes the supplied stream before return. This detects nonfinite/FP16-overflowing input and nonfinite output. Graph capture is rejected rather than silently dropping this check. Failed data validation leaves the Engine reusable after a valid input is supplied.
- One Engine must be externally serialized. Engines do not share mutable weights, modules, maps, buffers or CUDA handles. Destruction releases resources in the retained owning context and restores the thread's previous context. Caller-owned heads survive later inference and preparation.

## Historical reference validation

On the available SM89 GPU, the MSVC Release build ran six full native inferences: two different packets each at **320x384, 128x192 and 384x320**. Every head was **bitwise identical** to the original Python CUDA reference (zero differing values; maximum absolute error 0). The same Engine switched shapes at runtime; repeat calls and nonfinite-input recovery also passed on a nonblocking caller stream. The C++ layout checks passed at all five sizes listed below. These are correctness runs, not a controlled performance benchmark.

## Reproduce validation

The scripts below are development tools, **not build/runtime dependencies**. They write only beneath `native/nr/validation/` and use the existing CUDA-enabled Python 3.11 to run the unmodified original reference, redirecting its generated-source/cache root into that directory.

```powershell
C:/Python311/python.exe -B native/nr/nr_validate.py layouts
C:/Python311/python.exe -B native/nr/nr_validate.py reference
C:/Python311/python.exe -B native/nr/nr_validate.py native
```

`layouts` compares the C++ integer maps, metadata and runtime route arrays with the original Python address algebra at 64x64, 128x192, 320x384, 384x320 and 512x640. `reference` creates two different packets per inference shape, runs the original 71-layer implementation and records hashes. `native` runs the standalone executable on the same files with a nonblocking caller stream, verifies repeatability/status recovery and requires bitwise-equal heads. Machine-readable evidence is in `layouts.json`, `cases.json`, `comparison.json`, `kernel_sources.json` and `native.log` under that directory.

For an existing raw float packet:

```powershell
native/nr/build/Release/nr_smoke.exe model 320 384 packet.f32 head.f32
```

`nr_generate_weight_routes.py` is an optional development-only generator for the two fixed 32-entry address permutations in `nr_weight_routes.h`. It extracts only layout codecs, not weights. Its generated header is supplied, so users do not need to run it.

## Dynamic evaluation

`native/tools/bench_nr_dynamic.py` generates shared real feature packets and compares
static/dynamic output heads and timings. `d5_nr_dynamic_bench` measures cold prepare,
new grids, switch-back and steady-state GPU span/host latency; it asserts no additional
compiles or module loads on new geometry, and tests simultaneous separate Engines
and sequential caller-stream changes. `d5_nr_sdk_bench` exercises the public DLL
image pipeline. The WebUI mixed-size test remains the final integration path.

`Engine::audit()` is an internal C++ diagnostic API (not a new public C ABI):
compile/pack/module counts, compilation/loading/upload durations and last preparation.
Runtime constant activation happens once on each new workspace's first caller stream;
infer synchronization makes later stream changes safe. No mutable module state is
shared across Engines.

### Measured runtime-spatial delivery

See `native/qualification/nr-runtime-spatial.json`. Five NR grids with two original
feature-packet variants each were bitwise identical to frozen55780eb, including
returning to a prior grid. Fresh precompiled runs reported0 NVRTC calls,8 pack loads
and8 module loads for every grid; first prepare86.9ms, later prepares75–618ms in
the recorded run. Concurrent separate Engines and caller-stream changes were
bitwise identical.

The public DLL image pipeline was compared in one process with30 alternating
blocks and300 timed frames per implementation. Medians:5.6448ms static versus
5.6526ms dynamic; median paired-block change+1.38%. Raw separate runs showed large
system/time drift, so their percentages are not isolated kernel-regression claims.
First, second and final image outputs were bitwise identical. Real full-pipeline
WebUI preparation changed from roughly245–274s to1.96s initially and0.25–0.67s on
size switches. First VDA model loading is separate from these prepare metrics.
