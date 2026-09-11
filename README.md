# DLSS5 白箱管线：纯 TileLang NR

这是从 `dlss5_tilelang` 抽取整理后的**纯 TileLang DLSS5 管线**：整条 NR 图（每帧 193 个逻辑 Step）
由 TileLang 实现执行，外围的 FSR / 光流 / 度量深度 / 时域 / NR-chain 阶段仍在冻结的 Torch 参考实现上加速。

## 入口

```text
# CLI：manifest 进，帧序列出
.venv/Scripts/python.exe run.py --manifest <manifest.json> --output <新目录>

# WebUI
.venv/Scripts/python.exe whitebox_app.py
```

`run.py` 的 `--backend tilelang`（默认）走完整白箱管线并调用 TileLang NR；
`--backend torch` 用冻结的 Torch 参考跑同一条管线，用作数值对照。

## 计算组织

| 层 | 位置 | 说明 |
| --- | --- | --- |
| NR 计算 | `tilelang_nr/` | `VitJointTileLangNR`（算法基准 797fd63）：utility / shallow / 2H-4H / 8H / 16H / ViT-物理桥，共 193 个逻辑 Step |
| NR 计划 | `native_nr/` | 只提供 packet / layout 计划、launch 几何与 buffer 归属（`cuda/*.cu` 在 `_prepare` 时编译，仅用于取计划元数据，不执行任何原版数学 Step） |
| Host 管线 | `fsr_*.py`、`flow_pipeline.py`、`depth_attention.py`、`temporal_inputs.py`、`nr_chain_pipeline.py` | FSR reconstruct/locks、accumulate、depth-clip、RAFT-small 适配、DINO 概率发布、时域输入准备、NR-chain 输入/输出 |
| 冻结参考 | `reference/` | 逐字节不变的模型、权重与 `whitebox_pipeline` 包 |
| 调度 | `backend.py`、`run.py`、`execution.py` | 线程局部 dispatch：替换 NR 图 + 加速外围阶段 |

`tilelang_nr/README.md` 给出从公开入口追到计算的文件导览；`native_nr/README.md` 说明计划层。

## 运行准备

1. Python 3.11 venv，安装 `reference/requirements-whitebox.txt` 与 `requirements-accelerated.txt`
   （已验证 Torch 2.5.1+cu124 / torchvision 0.20.1 / TileLang 0.1.14 / CUDA 12.9 bindings）。
2. `reference/weights_ht_blob.bin` 与 `reference/guide_models/*.pth`：冻结权重，随工作树提供，不入 Git。
3. `.toolchains/cuda12.8/`：项目私有的 CUDA 12.8 NVRTC + CCCL + runtime，用来编译 `native_nr/cuda/*.cu`
   计划模块。缺它则 `_prepare` 直接失败（`add_dll_directory` 找不到 `nvrtc/bin`）。同样不入 Git。
4. GPU 需 SM89（RTX 40 系），且 NR 只在默认 CUDA stream 上验证。

## 抽取边界

本仓库只保留唯一最终管线：TileLang NR + 外围管线设施 + 冻结参考。相对源仓库已剔除：

- 原版 CUDA NR 的分发产物与替代后端：`native_nr_dist/`（含 ZIP）、`cuda_native/`、`cuda` 默认后端；
- TileLang 的中间组织与未采纳候选：旧 `tilelang_nr/legacy/`、endpoint / serial / parallel split、
  wide_packet、旧 joint/UP8 及 `experiments/archive/`；
- NR 侧被 TileLang 取代的 Torch 融合层（`mega_*`、`compact_one_*`、`cooperative_*`、`layout_*`、
  `fp8_one/encoder`、`decoder2h`、`four/eight/one_head` 等）——运行证据显示它们在 tilelang-vit 路径下
  一次也不会被调用；
- 全部诊断与证据脚手架：`check_*` / `benchmark_*` / `analyze_*` / `profile_*` / `diagnose_*`、
  `analysis/`、`evidence/`、`outputs/`、`experiments/`、`gpu_experiments/`、`tools/`。
