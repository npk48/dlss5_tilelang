# Model assets

All runtime model binaries live in this directory:

- `weights_ht_blob.bin` — frozen 71-layer NR weights
- `raft_small_C_T_V2-01064c6d.pth` — RGB motion estimator
- `metric_video_depth_anything_vits.pth` — RGB metric-depth estimator

`guide_models.json` pins filenames, sizes, hashes, upstream revisions, and URLs for the guide checkpoints. `WEIGHT_RIGHTS.md` records redistribution boundaries.

The binary files are tracked through Git LFS. Run `git lfs pull` after cloning. Runtime code never downloads model files during inference.

Native consumers use `weights_ht_blob.bin` and the exported ONNX files in
`native_guides/` (also Git LFS). The original PTH checkpoints are needed for
offline export, not native runtime. Included production VDA grid: 518x924.
See `../docs/NATIVE_SDK.md` for other aspect ratios and exported update counts.
