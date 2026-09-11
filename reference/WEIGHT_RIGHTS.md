# Source and weight rights

This local research package is not a blanket license grant for the complete pipeline.

- AMD FSR algorithms: MIT, see whitebox_pipeline/FSR2_LICENSE.txt.
- Feeder guide code: MIT attribution is retained with the package's source notices.
- Video Depth Anything / DINOv2 code: original Apache-2.0 / attribution material is retained in whitebox_pipeline/_vda/. Metric Small checkpoint's official model card declares Apache-2.0; the fixed revision and hash are in guide_models/manifest.json. Base/Large NC checkpoints are not included.
- torchvision/RAFT implementation is an installed dependency with its own BSD source license. That source license alone does not settle every training-data/checkpoint commercial or redistribution condition. Review the original weights and dataset terms for the intended use.
- The existing NR weights originate from the user's supplied NVIDIA asset. This reconstruction/package does not create a new grant to publicly redistribute or commercially exploit those weights, nor does it claim endorsement by NVIDIA.
- FFmpeg is an external installed dependency, not included. Its build configuration may be GPL/LGPL and codecs can have separate patent/licensing considerations.
- Test photographs/videos and native SDK/compiler/runtime binaries are not shipped.

The builder's --include-weights option produces a local offline research bundle with weights as separate files. Use the code-only form or obtain the relevant permissions before sharing outside your authorized environment. Source provenance, fixed hashes and technical reconstruction evidence are not substitutes for legal rights.
