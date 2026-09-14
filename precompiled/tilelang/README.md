# Runtime-spatial TileLang cache bundle

This is a compact, tracked seed for TileLang **0.1.14**, Windows **win32-AMD64**, CUDA **SM89**. It is not a general binary distribution and not a resolution catalog.

The NR entries contain runtime H/W, launch grids, buffer extents, offsets and routing geometry. Model/channel/MMA/FP8 structure remains specialized. There are 59 architecture bindings across 13 core factories plus three NR-chain factories; identical generated programs can share a disk entry. Unrelated auxiliary entries from the previous bundle are retained, while the 74 obsolete static-core cache entries were replaced.

`bundle.json` records the exact bundle ID, file count and bytes. `runtime/bootstrap.py` seeds this directory into writable `.cache/tilelang`; setting `TILELANG_CACHE_DIR` explicitly disables automatic seeding. Files in this tracked directory are never modified by inference.

The full NR was compared bit-exactly with the frozen pre-change TileLang implementation on 320×384, 384×512 and 512×640, including switchback. A fresh writable cache seeded **only from this bundle** also ran the real `Engine.run` path over ten mixed-size/continuous frames and 1→2→1 NR-pass changes without adding NR compilation. See `native/qualification/tilelang-runtime-spatial.json`.

Versions, model structure, GPU architecture or source changes may need new compilation. First-ever uncached compilation remains expensive. Non-NR pipeline kernels retain their own compilation behavior. Tensor allocation and layout/routing preparation still occur when geometry changes; steady-state inference is not claimed to match the static implementation's speed.

Keep the generated host/device sources and per-entry manifests: TileLang's disk loader requires them when rebuilding `JITKernel` objects.
