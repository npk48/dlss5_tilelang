# Model assets

All runtime model binaries live in this directory:

- `weights_ht_blob.bin` — frozen 71-layer NR weights
- `raft_small_C_T_V2-01064c6d.pth` — RGB motion estimator
- `metric_video_depth_anything_vits.pth` — RGB metric-depth estimator

`guide_models.json` pins filenames, sizes, hashes, upstream revisions, and URLs for the guide checkpoints. `WEIGHT_RIGHTS.md` records redistribution boundaries.

The binary files are intentionally ignored by Git and must be supplied with a local checkout or distribution. Runtime code never downloads them during inference.
