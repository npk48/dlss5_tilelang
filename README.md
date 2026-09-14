# DLSS5 白箱管线 · TileLang NR

这是一个面向研究与本地验证的完整图像/视频处理管线。默认执行路径以 **TileLang** 实现冻结的 71 层 NR 网络，并保留 Torch 与可读 CUDA 实现作为数值参考。

项目不调用 NGX，也不是游戏注入插件。它接收显式 guide，或从 RGB 估计 depth / optical flow，再经过 NR chain 与 FSR 时域重建输出图像序列。

## 原生 Windows SDK / 单文件 Server

新增 `native/` 提供不依赖 Python 的 C ABI：`dlss5.dll` + 导入库 `dlss5.lib`，以及静态链接同一 SDK 的 `dlss5_server.exe`。基于 CUDA NR，原生接入可选 VDA、RAFT、guide 与默认 SDR FSR2；支持调用方 GPU buffer/stream 和逐阶段替换回调。

构建、模型导出、API、同步语义与当前限制见 **[docs/NATIVE_SDK.md](docs/NATIVE_SDK.md)**。单 EXE 内嵌 web assets 和压缩原生依赖，模型外置 `./model`；首次启动会将依赖解包到用户本地缓存。原有 Python/TileLang 路径继续保留用于研究和对照。

## 执行路径

| 路径 | 用途 | 入口 |
| --- | --- | --- |
| **Native CUDA SDK / Server** | 第三方 C ABI 集成；可组合的原生图像管线；无需 Python | `native/include/dlss5.h` / `dlss5_server.exe` |
| **TileLang（Python 默认）** | 完整白箱管线；71 层 NR 共 193 个逻辑 Step 由 TileLang 执行 | `run.py --backend tilelang` / WebUI |
| **Torch reference** | 同一完整管线的冻结 Torch 数值参考 | `run.py --backend torch` / WebUI |
| **CUDA NR reference** | 独立验证 NR prepared packet；不包含 depth、flow、FSR 或视频调度 | `reference/cuda_nr/run_packet.py` |

“纯 TileLang NR”指默认产品路径不依赖 CUDA reference 执行。`reference/cuda_nr/` 只作为独立实现参考和逐位对照保留，不会被默认 backend 静默调用。

## 项目结构

```text
dlss5_remake/
├── run.py                         # manifest CLI；创建 Engine 并选择 TileLang/Torch
├── whitebox_app.py                # 本地 WebUI 服务、任务队列和状态 API
├── app/
│   ├── backend.py                 # scoped backend dispatch；只在激活区间替换计算路径
│   └── execution.py               # manifest、帧循环、输入输出与进度事件
├── pipeline/
│   ├── guides/
│   │   ├── depth.py               # DINO/VDA depth attention 加速
│   │   ├── flow.py                # RAFT-small correlation/index 路径
│   │   └── temporal.py            # 32 帧时域输入与 packed K/V
│   ├── fsr/                       # reconstruct、locks、depth clip、history、accumulate
│   └── nr_chain.py                # prepared packet、历史与 NR 输出混合
├── tilelang_nr/
│   ├── runtime.py                 # TileLang NR runtime、shape cache 与执行报告
│   ├── plan/                      # arena、布局、权重和 193 Step 的纯 Python 计划
│   ├── kernels/                   # utility / shallow / 2H-4H / 8H / 16H / ViT bridge
│   ├── common/                    # 各 kernel family 共用的 TileLang 构件
│   └── instructions/              # FP8/MMA/布局相关设备端 primitives
├── runtime/
│   ├── bootstrap.py               # reference/model 路径与预编译 cache seed
│   ├── model_loader.py            # 冻结 BIN 的向量化加载器
│   ├── fp8_compiler.py            # 为 FP8 kernel 挂载私有 CUDA 12.8 工具链
│   └── device.py                  # SM89 target 与 TileLang 配置
├── native/                        # 原生 C ABI SDK、CUDA/ORT 组件与单 EXE Server
├── reference/
│   ├── dlss5_model.py             # 冻结模型定义、BIN 解码与 packet 语义
│   ├── whitebox_pipeline/         # Torch 参考管线及第三方许可/来源
│   └── cuda_nr/                   # 可读 CUDA 71 层 NR reference、API 与 packet CLI
├── model/
│   ├── weights_ht_blob.bin        # NR 权重（Git LFS）
│   ├── raft_small_*.pth           # RAFT-small checkpoint（Git LFS）
│   ├── metric_video_*.pth         # Metric Video Depth Anything Small（Git LFS）
│   ├── guide_models.json          # guide checkpoint hash、来源和固定版本
│   └── WEIGHT_RIGHTS.md           # 权重与再分发边界
├── precompiled/tilelang/          # 有明确平台/shape 边界的 tracked cache seed
├── webui/                         # 精简的浏览器前端
├── docs/WEBUI.md                  # WebUI、Engine 和 HTTP API 说明
└── requirements.txt
```

更细的 NR 文件到 Step 映射见 [`tilelang_nr/README.md`](tilelang_nr/README.md)。CUDA reference 的边界和调用方式见 [`reference/cuda_nr/README.md`](reference/cuda_nr/README.md)。

## 环境要求

当前验证环境：

- Windows x64
- Python 3.11
- NVIDIA SM89（RTX 40 系）
- Torch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- TileLang `0.1.14`
- CUDA Python bindings
- 默认 CUDA stream

安装 Python 依赖：

```powershell
cd C:\work\dlss5_remake
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需视频输入或 MP4 预览，还需要本机可用的 `ffmpeg` 与 `ffprobe`。

## 模型文件与 Git LFS

模型二进制统一位于 `./model`，并由 Git LFS 跟踪：

```text
model/weights_ht_blob.bin                         147,695,410 bytes
model/metric_video_depth_anything_vits.pth       116,444,063 bytes
model/raft_small_C_T_V2-01064c6d.pth               4,006,189 bytes
```

克隆后执行：

```powershell
git lfs install
git lfs pull
```

运行时不会联网下载模型。Guide checkpoint 的 SHA256、来源 URL 和固定 revision 位于 `model/guide_models.json`；权重使用与再分发边界见 `model/WEIGHT_RIGHTS.md`。

## CUDA 12.8 私有工具链

仓库不跟踪 `.toolchains/`。TileLang FP8 kernel 与 CUDA NR reference 的首次编译需要：

```text
.toolchains/cuda12.8/
├── nvrtc/bin/nvrtc64_120_0.dll
├── nvrtc/bin/nvrtc-builtins64_128.dll
├── runtime/include/
└── cccl/include/
```

默认位置就是上述目录。也可以显式设置：

```powershell
$env:DLSS5_FP8_TOOLCHAIN = "C:\path\to\cuda12.8"
$env:NATIVE_NR_TOOLCHAIN = "C:\path\to\cuda12.8"   # 仅 CUDA reference
```

缺少工具链时，已经命中预编译 cache 的 shape 仍可运行；出现新 shape、cache miss 或源码变化时会明确编译失败，不会退回其他 NR backend。

## WebUI

启动：

```powershell
.venv\Scripts\python.exe whitebox_app.py
```

打开 <http://127.0.0.1:7861>。服务只监听 loopback，不应反向代理到外网。

首页只保留计算路线、输入类型、文件和输出尺寸；编码、NR work size、pass、FOV 与 manifest 位于“高级设置”。任务状态分成两个真实阶段：

1. **加载模型与 kernels**：按实际 `neural_size` 加载/编译并执行一次无状态 NR 预热；
2. **推理与输出**：正式处理图片/序列/视频。

预热不会消费 NRChain、FSR、depth 或 flow 历史。页面分别显示 setup 时间、正式 NR 调用数和逐帧耗时。完整行为与 HTTP API 见 [`docs/WEBUI.md`](docs/WEBUI.md)。

## CLI

### RGB-estimated 模式

最小 manifest：

```json
{
  "mode": "rgb_estimated",
  "color_encoding": "sRGB",
  "output_size": [540, 960],
  "frames": [
    {"color": "frame000.png", "reset": true},
    {"color": "frame001.png"}
  ]
}
```

`output_size`、`nr_work_size` 和所有 shape 均使用 `[height, width]`。输入路径相对于 manifest 所在目录。

运行 TileLang：

```powershell
.venv\Scripts\python.exe run.py `
  --backend tilelang `
  --manifest C:\path\to\manifest.json `
  --output C:\path\to\empty-output
```

运行 Torch reference：

```powershell
.venv\Scripts\python.exe run.py `
  --backend torch `
  --manifest C:\path\to\manifest.json `
  --output C:\path\to\empty-output
```

输出目录必须为空。每帧会写 PNG 预览；HDR、linear 或 `save_float` 会另写 NPY。目录中还包含：

- `run-manifest.json`：实际运行配置；
- `frames.jsonl`：逐帧尺寸、guide、backend 和耗时；
- `report.json`：任务结果；
- `backend.json`：模型加载、实际 NR backend、调用和 shape 统计。

### Guided 模式

Guided manifest 必须显式提供 depth、motion 和 `motion_convention: "current_to_previous_pixels"`。Motion 单位是 source pixels；管线不会猜测或自动翻转方向。

## 独立 CUDA NR reference

CUDA reference 接收已经构造好的 Float32 CUDA `B×16×H×W` prepared packet，返回独立拥有的 Float32 `B×H×W×4` head；它不执行完整应用管线。

确定性 demo：

```powershell
.venv\Scripts\python.exe reference\cuda_nr\run_packet.py `
  --demo-size 320 384 `
  --output outputs\cuda-reference-head.npy
```

现有 packet：

```powershell
.venv\Scripts\python.exe reference\cuda_nr\run_packet.py `
  --input packet.npy `
  --output head.npy
```

CUDA reference 与 TileLang NR 已在同一 320×384 packet 上验证逐位一致。

## TileLang 预编译缓存

`precompiled/tilelang/` 不是通用二进制发行版，也不是旧开发仓库的完整 cache。当前 bundle 只明确覆盖：

- TileLang `0.1.14`
- `win32-AMD64`
- CUDA target `sm_89`
- WebUI output `510×549`
- NR neural shape `512×640`
- RGB-estimated、1 个 NR pass

默认启动会把 tracked bundle 一次性复制到可写、Git 忽略的 `.cache/tilelang/`。显式设置 `TILELANG_CACHE_DIR` 时不会自动 seed。其他输出尺寸、NR work size、版本、平台或源码变化仍可能首次编译。

Bundle 的 ID、文件数和体积见 `precompiled/tilelang/bundle.json`。它保留 `host_kernel.cu` 与 `device_kernel.cu`，因为 TileLang 0.1.14 从磁盘重建 `JITKernel` 时会校验并读取这些文件。

## 已验证事实与边界

- 默认 TileLang NR：完整 71 层、193 个逻辑 Step；不存在 CUDA NR 静默 fallback。
- CUDA reference、原 CUDA distribution 与 TileLang NR 在确定性 320×384 packet 上逐位一致。
- RGB-estimated 路径使用 RAFT-small 与 Metric Video Depth Anything Small；估计 guide 不是真实游戏 depth/motion。
- 当前仓库只验证 SM89 与默认 CUDA stream；其他 GPU 架构和 graph/non-default stream 不自动兼容。
- 预编译 cache 只对其 manifest 声明的环境与 shape 作保证；工具链仍是新 shape 的必要依赖。
- 本仓库不会因代码重构自动获得模型权重、训练数据或第三方组件的新许可。
