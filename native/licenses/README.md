# Native dependency notices

This directory preserves the ONNX Runtime MIT license, third-party notices and a
hash/provenance manifest for the native runtime DLLs used by this build.

Additional applicable licenses:

- CUDA Toolkit / NVRTC / cuBLAS / cuFFT:
  https://docs.nvidia.com/cuda/eula/index.html
- cuDNN:
  https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html
- Microsoft Visual C++ 2022 runtime (VC143, 14.36.32532), copied from the licensed
  Visual Studio redistributable directory. Distribution is governed by the Visual
  Studio license and REDIST list:
  https://learn.microsoft.com/visualstudio/releases/2022/redistribution
- WIC, BCrypt and Compression API are Windows system components, not copied into
  the package.

cpp-httplib and nlohmann/json licenses reside beside their vendored headers.
FSR2's license resides in native/reconstruction/FSR2_LICENSE.txt. VDA and
Torchvision notices are in native/guides. Pretrained weights retain the restrictions
in model/WEIGHT_RIGHTS.md; converting a checkpoint to ONNX does not relicense it.

The local package is not a representation that every asset is cleared for public
commercial redistribution. Review the applicable model, data and vendor terms
before publishing a binary distribution.
