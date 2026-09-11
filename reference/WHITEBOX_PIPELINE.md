# Whitebox guided SDR sequence pipeline

## Current capability

The ordinary FSR2 v2.2.1 temporal path now runs in **PyTorch**, followed by the
existing frozen 71-block NR model and a declared SDR/history adapter. The product
path does not load or invoke the native FSR reference, NGX, RenoDX or DFC.

Implemented FSR stages: log-luminance/SPD/exposure, closest-depth/MV dilation,
previous-depth reconstruction, depth clipping and mask dilation, thin-feature
locks, Lanczos history reprojection, current-sample reconstruction/rectification,
accumulation and four-frame luminance history, plus optional RCAS. Float32 shader
equations retain Half/UNORM resource publication boundaries.

The FSR class also accepts HDR, normal/reversed/infinite depth, auto exposure,
external reactive/composition masks, display-resolution motion and jitter
cancellation. The combined guided/RGB entry now supports declared SDR and HDR,
independent multi-pass NR, work-size policies, alpha/protection and output guards.
See [P3_COLOR_AND_HISTORY.md](P3_COLOR_AND_HISTORY.md) for exact color/state
semantics; HDR is not approximated by clipping the entire image to SDR.

RGB-only optical-flow/metric-depth estimation is now implemented and tested at
the actual manifest entry; see [RGB_ESTIMATED_PIPELINE.md](RGB_ESTIMATED_PIPELINE.md)
for setup, assumed-camera/estimated-depth boundaries and full-resolution evidence.

Not yet delivered: experimental FSR opaque-only TCR autogeneration;
history-preserving dynamic render resolution;
experimental expansions listed in the plan. Streaming video/CLI and the separate
task UI are implemented; see [WHITEBOX_APP_README.md](WHITEBOX_APP_README.md)
for application use and independent package requirements. This is the guided SDR stage of the larger plan, not completion of
all Feeder/consumer features or equivalence to DLSS.

## Run the actual sequence entry

From this repository, with the existing CUDA environment:

```powershell
C:/Python311/python.exe -m whitebox_pipeline --manifest sequence.json --output results
```

The output directory must not already contain sequence output. The processor
loads the NR model once, processes frames sequentially, writes numbered PNGs
and updates `report.json`. No server is started and no model is downloaded.

Example manifest (all paths relative to this JSON):

```json
{
  "color_encoding": "linear",
  "motion_convention": "current_to_previous_pixels",
  "output_size": [192, 256],
  "fsr_settings": {
    "camera_near": 0.1,
    "camera_far": 100.0,
    "camera_fov_y": 1.0471975512
  },
  "nr_settings": {"style": 1, "structure": 2.0, "temporal_strength": 1.0},
  "frames": [
    {
      "color": "000-color.npy",
      "depth": "000-depth.npy",
      "motion": "000-motion.npy",
      "jitter_xy": [0.0, -0.1666666667],
      "delta_ms": 16.6666667,
      "reset": true
    }
  ]
}
```

- `output_size` is `[height,width]`, not `[width,height]`.
- Color `.npy`: floating HWC RGB. Opaque PNG/JPEG are accepted with explicit
  `color_encoding: "sRGB"`; the decoder then converts to linear light.
- Depth `.npy`: HW or HWC with one channel, **device depth [0,1]** in the configured
  projection convention, not a guessed grayscale/relative-depth image.
- Motion `.npy`: HWC with two channels, **current-to-previous render-grid pixels**.
- Optional `reactive` and `composition` refer to separate float HW/HWC1 mask files.
  They are not Feeder distrust or NR intensity under another name.
- Jitter must describe sampling that really occurred. FSR's convention places
  the unjittered sample at `pixel_center - jitter`; a finished unjittered image
  should declare zero, not an invented Halton offset.
- Render axes must be at least32 and no larger than output. A render-size change
  needs explicit reset. A context processes one independent sequence.
- The first NR profile is Natural/Structure2/Tone1/Skin-1, automatic mask off.
  Parameters are not fitted residual gains.

Programmatic entry:

```python
from whitebox_pipeline import GuidedSDRPipeline

pipe = GuidedSDRPipeline((output_height, output_width), device="cuda")
result = pipe.process_frame(linear_rgb_bchw, device_depth_b1hw,
                            current_to_previous_pixel_flow_b2hw,
                            jitter_xy=(jx,jy), reset=is_first_or_cut)
encoded_rgb = result["color"]          # BCHW sRGB at requested display size
linear_rgb = result["linear_color"]
```

Use explicit reset for a new sequence or camera/guide convention. NR pass/work
changes now use `configure_nr`, output changes use `set_output`, and color changes
use `set_color`, with scoped history invalidation as documented in the P3 guide.

## Explicit NR/state policy

FSR, Feeder guide history and NR history are independent. FSR never consumes NR
output. Feeder retains original input luma/depth/raw flow, not filtered flow.
NR retains its own completed, pre-output-processing NR-domain image.

NR input preparation uses standard linear-to-sRGB encoding, positive FP16
round-toward-zero, and the existing single-reflection-then-clamp padding helper.
Neural dimensions follow the tested local policy: height aligned64/min320,
width with an extra64 guard then aligned64/min320. It is not a claim of the
original DLL's size negotiation.

The NR history gate is an authored, inspectable adapter rule:

```
active-rectangle visibility
* (1 - FSR depth clip)
* (1 - FSR effective reactivity)
* user temporal strength
```

The existing learned history logit and recovered output blend are used without
changing the model. Unavailable/rejected history uses current-color source slots
in the model input; cold/reset has no history blend. This is not asserted to be
the private DFC/RenoDX consumer policy. Failed frames restore all previous state;
there is no partially updated FSR state feeding a retry after NR failure.

## Actual verification (2026-09-06)

- Complete24-frame Torch FSR sequence versus explicitly selected official FP32
  shaders: first-frame MAE `2.87e-8`, last-frame MAE `1.62e-5`; last max error
  `.00893`, with a larger sparse maximum `.03189` earlier in the sequence.
  This is not bit identity: sampler/basic arithmetic differences can cross
  temporal lock/UNORM thresholds. The entire trajectory is retained, not just
  a favorable frame.
- Eight-frame sequences covered native-size1:1 (mean error <=`4.72e-7`), odd render/output sizes, HDR+auto exposure+
  reversed depth+RCAS, reversed infinite depth, display-resolution jittered MV,
  masks and mid-sequence reset. Mean errors were ~`1e-8..5e-6` for the SDR
  contracts and <=`1.31e-4` on the HDR scene (linear values up to about6).
  Flat black/gray/white sharpening cases remained finite without invented epsilon.
- The **actual manifest entry** processed24 frames through Torch FSR2 and the real
  frozen NR with history active. Processing took46.92s after model loading.
  A deliberately failed NR call after FSR processing restored all states;
  reset reproduced the first frame exactly.
- The CLI also processed four frames using the existing user's549x510 image as
  a moving planar texture, with275x255 render input and512x640 neural buffer.
  Requested549x510 outputs were preserved; full-frame processing was about
  2.96–3.53s/frame after loading. These are real image contents with analytic
  texture-plane guides, **not real-video quality or inferred human geometry**.
- Core source/weights remain unchanged. Current `dlss5_model.py` SHA256:
  `e796339b239f432fcedbefe823680befcafaf739e20ee2e29cfbf5c568984a1b`.

Evidence and images:

- `debug-static/torch-fsr2/provided-comparison.json`
- `debug-static/fsr2-contract-modes/report.json`
- `debug-static/whitebox-sequence-verification.json`
- `debug-static/whitebox-torch-sequence/000000.png` through `000023.png`
- `debug-static/whitebox-real-image-sequence/report.json` and four PNGs

## Important reference correction

The original `bf10ec4` reference tried to disable FP16 through the SDK capability
callback. The DX12 backend independently queries hardware when selecting shaders,
so that did **not** force FP32. Its measurements were valid device-selected runs,
but their original precision label was wrong; retained reports are now marked.

The current diagnostic build makes a generated copy of the backend and changes
only permutation selection to explicitly choose the official FP32 shaders. The
cloned upstream repository and algorithm equations are unchanged. New reference
evidence lives under `debug-static/whitebox-fsr2-fp32`.

Intermediate resource readbacks also established NVIDIA D3D12 **R16 UAV stores
use toward-zero**, not PyTorch's default nearest-even conversion. Preserving that
actual publication boundary reduced first-frame error from`1.79e-4` to`2.87e-8`.
The reference captures restore the real tracked resource state; captured and
uncaptured native final outputs were SHA256-identical.

## Sources and licensing

FSR2 source revision and hashes are recorded in `reference/fsr2/sources.lock.json`;
AMD's MIT license is retained as `whitebox_pipeline/FSR2_LICENSE.txt`. Feeder's
MIT notice is retained separately in `NOTICE.txt`. These source licenses do not
relicense the existing NR weights or establish distribution/commercial rights
for future pretrained guide models.
