# CUDA NR reference

Readable CUDA implementation of the same complete 71-layer frozen NR network used by the TileLang runtime. It was imported from `C:/work/dlss5_tilelang/native_nr_dist` at source commit `b9e578f` (the distribution itself was introduced by `d791931`) and retained only as an independent implementation reference and comparison target.

## What is shared

This directory deliberately does **not** duplicate the frozen model module, vectorized BIN loader, model weights, WebUI, or generated CUBIN cache:

- model definition: `reference/dlss5_model.py`
- vectorized loader: `runtime/model_loader.py`
- all model binaries: repository root `model/`
- private CUDA 12.8 toolchain: repository root `.toolchains/cuda12.8`
- generated source/CUBIN cache: repository root `.cache/cuda_nr`

The eight files under `cuda/` are the grouped source units for shallow, 2H, 4H, 8H, deep16, ViT, bridge, and utility work. `runtime.py` preserves the original 193-step ordering, storage ownership, layouts, and prepared-packet ABI.

## Run

From the repository root:

```powershell
python reference/cuda_nr/run_packet.py --demo-size 320 384 --output outputs/cuda-reference-head.npy
```

Override the compiler toolchain only when it is not at `.toolchains/cuda12.8`:

```powershell
python reference/cuda_nr/run_packet.py --toolchain C:\path\to\cuda12.8 --demo-size 320 384 --output head.npy
```

Python API:

```python
from runtime import bootstrap
from cuda_nr import load_runtime
runtime = load_runtime()
try:
    head = runtime.infer_minimal(packet)
finally:
    runtime.close()
```

Input is contiguous CUDA Float32 `B×16×H×W`; output is owned Float32 `B×H×W×4`. The qualified platform remains Windows x64, CUDA 12.8 NVRTC, default CUDA stream, and SM89. There is no fallback.
