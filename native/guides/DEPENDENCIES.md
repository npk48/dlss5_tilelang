# Native RAFT / causal metric VDA

`dlss5_guides` is a C++17/CUDA static target. Delivery uses ONNX Runtime CUDA,
not Python, Torch, TorchScript, OpenCV, or a Python subprocess. The exporter and
reference comparator are development tools only.

## Build and delivery

Test configuration: Windows x64, VS2022 Professional MSVC 19.36, CUDA Toolkit
12.2, RTX 4080 Laptop (SM89), ONNX Runtime **1.20.1**, CUDA runtime/cuBLAS **12.4**,
cuDNN **9.1**. Helper kernels link static cudart and standalone targets use /MT;
subdirectory builds preserve the enclosing CMAKE_MSVC_RUNTIME_LIBRARY setting.
ORT 1.20's CUDA12 build requires cuDNN9, not cuDNN8. The NVIDIA
GPU driver is a system prerequisite. The parent SDK/server package also supplies
the Microsoft VC++ x64 runtime; it need not be installed separately for that package.

```powershell
cmake -S native/guides -B native/guides/build -G "Visual Studio 17 2022" -A x64 -T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
cmake --build native/guides/build --config Release
```

An enclosing SDK can `add_subdirectory(guides)` and link `dlss5_guides`. No root
CMake file is modified here. `DLSS5_ORT_ROOT` defaults to `native/third_party/ort`,
with `include/`, `lib/onnxruntime.lib`, and `runtime/*.dll`. The standalone verifier
copies the runtime DLLs beside its EXE. The enclosing SDK must deploy those same
sidecars beside its EXE/DLL (or arrange an explicit secure DLL directory before
loading). ORT API initialization is explicit and lazy (no pre-main ORT C++ static
initializer); a completely disabled Engine needs no ORT DLL, and an enabled Engine
throws a clear error if the core DLL is missing. Secure Windows default DLL search
or an already loaded module is used, not an arbitrary Python/PATH search.
ORT's CUDA provider dynamically loads its companion provider DLLs.
A self-extracting server must extract these before loading ORT; this library
does not implement EXE embedding/extraction.

Native SDK source:
https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.windows/1.20.1/microsoft.ml.onnxruntime.gpu.windows.1.20.1.nupkg

Extract `buildTransitive/native/include` into `include`, the Windows native import
library into `lib`, and `onnxruntime.dll`, `onnxruntime_providers_shared.dll`,
`onnxruntime_providers_cuda.dll` into `runtime`. Preserve ORT `LICENSE` (MIT) and
`ThirdPartyNotices.txt`. Do not deploy the unused TensorRT provider.

The staged runtime contains standalone NVIDIA redistributables: `cudart64_12`,
`cublas64_12`, `cublasLt64_12`, `cufft64_11`, all cuDNN9 components, NVRTC and its
builtins, and `zlibwapi` DLLs. For this workspace they were copied from the installed
CUDA12.4 Torch distribution, but **there is no runtime dependency on that install
or any site-packages path**. A clean deployment should source these native files
from the matching official NVIDIA CUDA/cuDNN redistributable packages. NVIDIA's
licenses, redistribution restrictions and required notices still apply:

- https://docs.nvidia.com/cuda/eula/index.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html
- https://developer.download.nvidia.com/compute/cuda/redist/
- https://developer.download.nvidia.com/compute/cudnn/redist/

The dependency manifest records file hashes and workspace provenance; copying
binaries does not grant additional redistribution rights. Do not ship the SDK
without reviewing NVIDIA and checkpoint/data redistribution terms.

## Export real checkpoints (development only)

The exporter verifies SHA-256 against `model/guide_models.json`, loads both
original networks with `strict=True`, and exports FP32 opset17 ONNX. VDA is the
complete Metric Video Depth Anything Small **experimental causal streaming**
architecture (DINOv2 + temporal DPT with eight explicit hidden-state tensors),
not offline VDA, a single-image surrogate, or a smaller replacement model.
Its output is an estimate in meters, not measured depth.

```powershell
.venv/Scripts/python.exe native/tools/export_guides.py export --height 360 --width 640
```

- RAFT is dynamic in network H/W (multiples of 8, at least 128). The number of
  recurrent updates is baked into `raft_small_u8.onnx`; export another
  `--flow-updates N` for a different config. Current image is the first RAFT input.
- VDA has exact **shape-specialized network graphs** `vda_small_HxW_init.onnx`
  and `vda_small_HxW_step.onnx`. Both must exist. DINO's reference positional
  interpolation and the temporal head use Python spatial float/int conversions;
  declaring dynamic axes there would silently freeze parts of the graph. We do
  not claim dynamic VDA network shapes. Export each desired shape offline with
  `--component depth --height H --width W --depth-input-size S`.
- Source image sizes are handled by native preprocessing and can differ from the
  export example if they map to the same VDA network size. Defaults 360x640,
  720x1280, 1080x1920 with depth input518 all map to **518x924**. Portrait and
  other aspect ratios require corresponding exports. Missing shapes throw.
- Inputs/outputs: contiguous device float RGB HWC3 [0,1], HWC2 current-to-previous
  motion in **source pixels**, HWC1 metric depth. RAFT uses bilinear resize,
  bottom/right replicate padding, [-1,1] normalization, crop and vector scaling.
  Depth uses bicubic (-0.75) and ImageNet normalization, then align-corners
  bilinear output resize. Input values must already satisfy the RGB contract.
- Config is immutable for an Engine. Calls are single-threaded and use the same
  caller CUDA primary-context stream until reset. IOBinding uses device memory
  and `user_compute_stream`; no images or caches are staged through the host.
  ORT Run is synchronizing; postprocessing is enqueued on the caller stream.
  This is not a CUDA Graph capture API.
- Each Engine owns its explicit cache history: first-frame replication to 32,
  selection `history[:2] + history[-29:]`, and second-entry eviction after the
  zero-based index exceeds10. Reset discards history and waits for outstanding
  work. A source-size change requires reset. An inference exception poisons the
  Engine until reset. Callers own output buffers and must keep all buffers and
  the stream alive through completion. Avoid all overlapping input/output ranges.
- No synthetic masks are produced. Disabled components and absent assets/backend
  fail explicitly. Main SDK component replacement is outside this target.

## Independent native/reference comparison

```powershell
.venv/Scripts/python.exe native/tools/export_guides.py validate --height 360 --width 640 --frames 13 --native-exe native/guides/build/Release/guides_verify.exe
```

This creates translating RGB fixtures, evaluates the **original** `RaftSmall`
and `MetricVideoDepth(fp32=True)` classes, then launches the standalone C++ EXE
with PATH reduced to Windows system directories. It compares source-resolution
outputs without depth scale fitting or flow alignment, writes `comparison.json`
and `native.log` below `model/native_guides/validation_*`. Thirteen frames cover
bootstrap, history growth and eviction. The C++ program additionally checks reset,
independent estimator history, dimension-change rejection, disabled components,
and missing assets, using a nonblocking nondefault CUDA stream. It prints loaded
GPU DLL paths and rejects Python/Torch DLLs. This is numerical equivalence to the
reference FP32 path, not a claim of FP16 bit equality or offline accuracy.

`validation_summary.json` records the measured 13-frame results at both default
and small work sizes. Optional `DLSS5_GUIDES_PROFILE_DIR` enables ORT profiling
into an existing directory. On the tested default path, dynamic RAFT's nine ORT
Memcpy nodes are **nine 4-byte host-to-device shape scalars**, not images; VDA
init/step graphs contain no runtime Memcpy nodes.

VDA source notices are preserved as `VDA_LICENSE` and `VDA_NOTICE`. Torchvision
source is BSD-3-Clause; pretrained RAFT checkpoint/training-data rights require a
separate review, as recorded in the existing model manifest and weight notices.
