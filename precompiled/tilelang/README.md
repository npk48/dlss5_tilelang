# Qualified TileLang cache bundle

This is a compact, tracked seed for the actual WebUI workload that exposed the cold-compile failure. It is **not** the old 776 MB development cache.

Qualified environment and workload:

- TileLang `0.1.14`
- Windows `win32-AMD64`
- CUDA target `sm_89`
- RGB-estimated WebUI output `510×549`
- NR neural shape `512×640`
- one NR pass

`bundle.json` records the exact bundle ID, file count, and byte size. `runtime/bootstrap.py` copies this read-only bundle once into the writable, ignored `.cache/tilelang` directory. New shapes compile into `.cache`; they never modify this tracked directory.

TileLang's disk loader verifies each entry through its own `manifest.json`. The generated `host_kernel.cu` and `device_kernel.cu` files look redundant, but TileLang 0.1.14 requires them when reconstructing a cached `JITKernel`, so they cannot be removed independently.

A cache-only replay was run with `DLSS5_FP8_TOOLCHAIN` intentionally pointed at a missing directory. The workload completed, proving that this bundle contains all required FP8 NR entries for the qualified shape. Other output/neural shapes may still compile on first use.
