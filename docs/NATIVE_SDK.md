# Native Windows SDK and Server

The `native/` implementation is **C++/CUDA**, not an embedded Python application.
The DLL and server do not load Python, PyTorch, or TileLang. Development-time
Python scripts only export ONNX models and compare against the reference.

## Components

| Component | Native implementation | Boundary |
|---|---|---|
| NR | Original 71-layer CUDA graph, raw BIN parsing, runtime layout/weight setup | SM89; Float32 CHW16 packet -> HWC4 head; 64-aligned axes |
| NR chain | CUDA packet/Gaussian/history warp/composition, independent state per pass | 1–30 passes; style, tone, structure, skin, automatic mask, temporal strength, intensity |
| RAFT | ONNX Runtime CUDA, RAFT-small C_T_V2, device IOBinding | Dynamic spatial dimensions; exported update count (included: 8) |
| VDA | Full Metric VDA Small FP32 init/step models, causal hidden-state history | Network spatial dimensions are exported offline; included default: 518×924 |
| Guide | Native reference static/depth/consistency tests, optional luma test | Geometry fitting is not implemented; use a replacement stage |
| FSR2 | Native default reference reconstruction, depth clip, reactive/composition, locks, accumulation | SDR only; standard finite, non-inverted depth projection |
| Color/output | sRGB/linear709/nits/PQ2020 bridge, mix/protect/alpha, residual transport | HDR+FSR unsupported without replacement; resize/HDR are close, not bitwise parity |
| HTTP/WebUI | Same SDK statically linked into native EXE; WIC image codecs, cpp-httplib | Image jobs and explicit causal streams, not an FFmpeg video player |

Enabling a component does not silently substitute a different algorithm. Missing
models and unsupported combinations return an error. With reconstruction disabled,
size changes use explicit spatial bilinear resizing, not FSR2.

## Build

Requirements: Windows x64, MSVC 2022 x64 tools, CMake/Ninja, CUDA Toolkit >=12.2,
SM89 GPU and NVIDIA driver. Node.js 22+ and npm are frontend **build-time** dependencies only. `.toolchains/cuda12.8` provides runtime NR NVRTC
compilation. The native ONNX Runtime SDK and CUDA/cuDNN DLLs are staged under
`native/third_party/ort`; see `native/guides/DEPENDENCIES.md`.

```powershell
powershell -ExecutionPolicy Bypass -File native/build.ps1
```

Outputs in `native/build`:

- `dlss5.dll` and `dlss5.lib`: shared SDK plus **import library**, not a standalone static SDK archive.
- `dlss5_server.exe`: small native API server; runtime, model and web assets are external.
- `assets/`: production React/TypeScript frontend built by Vite.
- `d5_consumer.exe`: separate consumer linked through the public DLL/import library.
- `d5_interface_bench.exe`: paired C ABI versus direct native NR timing tool.

`dlss5_static.lib` is an internal build target used by the server; distributing it
alone is insufficient because it has component-library dependencies.

## Model preparation

Place these beside the deployment, not inside the EXE:

```text
model/
  weights_ht_blob.bin
  native_guides/
    raft_small_u8.onnx
    vda_small_518x924_init.onnx
    vda_small_518x924_step.onnx
    native_guides.json
```

VDA preprocessing chooses a network grid from input aspect ratio and input-size
setting, following the reference. A different resulting grid needs its matching
init/step ONNX pair. It is **not** silently stretched to 518×924. See
`native/tools/export_guides.py --help` for offline export. The original `.pth`
files are needed for export, not for native runtime. RAFT's update count is also
part of the exported filename. Additional assets can coexist in this directory.

## C ABI

Include `native/include/dlss5.h`, link `dlss5.lib`, and deploy `dlss5.dll`. For built-in
estimators, register the shared `runtime/bin` directory using `AddDllDirectory`
before creating an estimator (or deploy its DLLs beside the host application). Set
`NATIVE_NR_TOOLCHAIN` to `runtime/cuda12.8` before NR preparation. The packaged
consumer example detects the distribution’s shared runtime automatically.

All structures have `struct_size`; configuration also has `abi_version`. Initialize
configuration and settings with the `d5_default_*` functions. NR is enabled by
default. Explicitly OR together the stage bits for the components you want:

```cpp
D5Config config;
d5_default_config(&config);
config.model_directory_utf8 = "model";
config.enabled_stages = D5_ENABLE_DEPTH | D5_ENABLE_FLOW | D5_ENABLE_GUIDE |
                        D5_ENABLE_RECONSTRUCT | D5_ENABLE_NR;
D5Session session = nullptr;
D5Status status = d5_create(&config, &session);
// Check every status; d5_last_error() is thread-local text.
```

Before processing, call `d5_prepare(session, input_width, input_height)` to allocate
shape buffers and compile/load NR. ONNX sessions are lazy-loaded on first enabled
estimator use; that first processing call includes estimator load time.

Fill `D5Frame` with **CUDA device pointers** and a caller-owned `CUstream`. Source
and output are Float32 HWC RGB/RGBA. `row_stride_bytes=0` means tight storage.
Input/output buffers belong to the caller and must stay alive until
`d5_synchronize` succeeds. No full-frame GPU/CPU staging occurs in the C ABI.

Optional resources:

- `depth`: positive meters, not hardware depth or per-frame min/max normalization;
- `motion`: current→previous **source-image pixel** units, not normalized UV;
- `confidence`: explicit NR history confidence override, not a relabeled guide mask;
- `reactive` and `composition`: FSR masks on the source grid;
- `protect`: output protection on the output grid, not a learned tone/structure mask.

Supplied depth/motion replaces its built-in estimator for that frame. Guide/FSR
require source-grid depth and motion. If reconstruction is enabled, its dilated
motion and depth-clip/reactivity confidence feed NR. The guide distrust mask is
**not** automatically treated as equivalent to FSR reactivity.

`d5_set_nr_settings` updates pass count and learned controls without reloading
weights or recompiling the same NR shape; unchanged leading-pass histories are
preserved, changed passes and their suffix reset. `d5_set_output_mix` changes only
output blending and does not reset history. These are intended for live controls.

`frame.reset` clears temporal history. Same-shape reset preserves loaded estimator
models. Use `d5_reset` before switching a session's estimator stream. A session is
single-host-caller; concurrent calls on it return `D5_BUSY`. Separate sessions own
independent state. Stop all callers before destroying a handle; the handle must not
be accessed after `d5_destroy`.

Validation errors before submission leave prior history unchanged. Errors after
image stages begin invalidate the combined session history, rather than publishing
a partially advanced pipeline.
After CUDA asynchronous device errors, recreate the session/context as required by
the CUDA error, rather than assuming recovery is possible.

### Reading component outputs

`d5_stage_output(session, stage, index, &view)` returns a borrowed GPU view from the
last successful frame without copying data. Depth/flow use index0; GUIDE exposes
corrected motion, distrust and normalized depth at indices0–2; RECONSTRUCT exposes
linear context, NR confidence and dilated pixel motion; PACKET and NR expose the
last pass's packet/head; OUTPUT exposes the caller's final output. A disabled or
unavailable result returns `D5_UNAVAILABLE`. The views remain valid only until the
next processing/preparation/reset/destruction; synchronize before reading on an
unrelated stream. Supplied resources remain caller-owned.

### Direct packet interface

```cpp
d5_prepare_packet(session, width, height);
d5_infer_packet(session, &packet_chw16, &head_hwc4, stream);
```

This bypasses estimators, guide, reconstruction, image preparation and output
processing. Buffers must be tight, disjoint, 16-byte aligned, and in the selected
CUDA primary context. `prepare_packet` takes packet dimensions directly;
`prepare` takes source-image dimensions and applies NRChain's neural-size policy.

**Synchronization is explicit and honest:** NR preserves the original 4-byte
validation readback and waits on the caller stream. FSR and ORT also have necessary
completion checks. The implementation is not a fully asynchronous/graph-capturable
submission API (`capabilities.asynchronous=0`). The SDK does not use a device-wide
synchronization to transfer images, but native preparation/allocation is blocking.

### Replacement callbacks

`d5_set_plugin` registers a callback for a stage; a null callback restores the
built-in implementation. A callback receives device views and the caller stream,
and must enqueue writes to the provided output buffers. Do not replace descriptor
pointers or retain views beyond the frame. Do not re-enter the same session.

| Stage | Input slots | Output slots |
|---|---|---|
| DEPTH | encoded source RGB | source-grid metric depth |
| FLOW | encoded current RGB, previous RGB | source-pixel motion |
| GUIDE | encoded RGB, meters depth, motion, normalized depth | corrected source-pixel motion, distrust |
| RECONSTRUCT | positive linear/reference-white RGB, meters depth, corrected motion, combined reactive, composition, distrust | temporal RGB in the same normalized units, output-grid confidence, source-grid dilated pixel motion |
| PACKET | sampled current, warped previous, visibility/confidence gate | CHW16 packet |
| NR | CHW16 packet | HWC4 head |
| OUTPUT | linear context, encoded reference, encoded modified, original RGB(A), protect | caller output RGB(A) in configured encoding |

Absent optional input slots have `data=0`. Invocation includes frame ID, reset,
pass index, timing/jitter and NR settings. Callback `user` memory belongs to the
caller and must outlive its registration and in-flight work. No callback state is
automatically freed by the SDK.

## Native Server and WebUI

Deploy the folder layout, not the EXE alone:

```text
dist/
  dlss5_server.exe              # native application code, no asset archive
  assets/                      # Vite build: index.html and hashed JS/CSS
  model/                       # NR BIN and native_guides ONNX assets
  runtime/
    bin/                       # ONNX Runtime, CUDA/cuDNN and VC++ redistributables
    cuda12.8/                  # NVRTC compiler and C++ headers
  native-sdk/
    bin/dlss5.dll
    lib/dlss5.lib
    include/dlss5.h
  licenses/
```

```powershell
.\dlss5_server.exe --port 7863
# Optional paths; defaults resolve relative to the EXE, not the current directory:
.\dlss5_server.exe --runtime D:\dlss5\runtime --model D:\models\dlss5 --assets D:\dlss5\assets
```

Open `http://127.0.0.1:7863`. The server loads native dependencies directly from
`runtime/bin` and compiles NR using `runtime/cuda12.8`. It no longer embeds or
extracts assets, and does not read a hidden LocalAppData dependency cache. Large
vendor libraries still occupy disk space; externalizing them makes the EXE small
and lets runtime, models and UI be updated independently. Node/npm is not needed
on the deployed machine.

The UI is a real npm project at `native/webui`:

```powershell
cd native/webui
npm ci
npm run dev       # Vite dev server, /v1 and /api proxied to the native server
npm run build     # dist/index.html plus production JS/CSS assets
```

`native/build.ps1` integrates the npm build; `native/package.ps1 -IncludeModels`
assembles the folder above. For frontend-only development, no C++ relink is needed:
build with npm and pass `--assets native/webui/dist` (absolute path recommended).
If assets are missing, the API still starts and `/` returns a clear 503 rather than
an embedded fallback page.

API:

- `GET /v1/info`: external asset/runtime/model paths and available model filenames;
- `GET /v1/jobs`: up to eight recent jobs, submitted config and Unix creation times;
- `POST /v1/jobs`: multipart `image` and optional `config` JSON string;
- `GET /v1/jobs/{id}`: status, error, timing and output location;
- `GET /v1/jobs/{id}/result.png`
- `GET /v1/openapi.json`

```powershell
curl.exe -F "image=@frame.png" `
  -F 'config={"structure":2,"tone":1,"depth":true,"flow":true,"guide":true,"reconstruction":true}' `
  http://127.0.0.1:7863/v1/jobs
```

Independent image jobs reset history. For ordered streaming, send the same
`stream_id`, keep configuration/dimensions fixed, use `reset=true` on the first
frame and `reset=false` thereafter. Submit frames in order; the GPU queue executes
one job at a time. Up to four stream contexts and eight recent job results are
retained. Changing shape/backend configuration recreates its SDK session; changes
to structure/tone/style/skin/pass settings or mix use the live-control API instead
of recompiling/reloading the model.

Image HTTP necessarily uploads/downloads pixels; use the DLL for GPU-native
integration. WIC handles PNG/JPEG/BMP and preserves RGBA in PNG output; only the
first frame of a multi-frame image container is decoded. Video decoding/muxing is
not built into this server. Hosts can submit decoded video frames through the API.

Default binding is loopback. Non-loopback `--host` requires `--token`; clients send
`Authorization: Bearer <token>`. Use a TLS reverse proxy on untrusted networks.
Cross-origin browser requests are rejected. The static UI remains public so its connection dialog can collect an API token;
API routes require the bearer token. The UI also fetches protected result images
with that token instead of exposing the token in image URLs.

## Qualification and unsupported variants

Component tests cover native NR arithmetic, packet/history, guides and default
FSR2 against the prior CUDA/Torch implementation. These are not proof of official
NVIDIA DLSS5 equivalence. FP32 ONNX VDA is compared to the FP32 reference, not the
prior autocast FP16 mode. FSR/HDR spatial results have measured small numerical
differences; see the component validation reports and final integration results.

Not implemented as built-ins: geometric guide fitting, EASU residual transport,
HDR FSR accumulation, auto-exposure, reversed/infinite projection variants,
learned per-pixel tone/structure masks, and optional output detail/luma/chroma guard
filters. Replacement callbacks are available; unsupported behavior is not silently
claimed. SM89/default finite projection are the qualified target, not arbitrary
GPU architectures or the paper's Blackwell 4K performance target.
