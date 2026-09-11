# DLSS5 白箱管线：纯 TileLang NR

整条 NR 图（每帧 193 个逻辑 Step）由 TileLang 实现执行。外围的 FSR / 光流 / 度量深度 /
时域 / NR-chain 阶段仍在冻结的 Torch 参考实现上加速。

NR 的计划（arena 布局、buffer 归属、每步 launch 几何与参数）由 `tilelang_nr/plan/` 用纯 Python/Torch
构建，不编译、不绑定任何 CUDA kernel。

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
| NR 计算 | `tilelang_nr/kernels/` | `VitJointTileLangNR`（算法基准 797fd63）：utility / shallow / 2H-4H / 8H / 16H / ViT-物理桥，共 193 个逻辑 Step |
| NR 计划 | `tilelang_nr/plan/` | 纯 Python/Torch：arena 与 aux 布局、buffer 归属、每步几何与 typed 参数；无编译、无 CUDA kernel |
| FSR | `pipeline/fsr/` | reconstruct/locks、depth-clip、history sample、accumulate |
| Guides | `pipeline/guides/` | RAFT-small 光流、DINO 度量深度、时域输入 |
| NR chain | `pipeline/nr_chain.py` | NR 输入、历史与输出混合 |
| 应用调度 | `app/` | scoped backend dispatch 与 manifest 执行 |
| 运行设施 | `runtime/` | reference bootstrap、设备策略、模型加载、FP8 编译器 |
| 冻结参考 | `reference/` | 逐字节不变的模型、权重与 `whitebox_pipeline` 包 |
| WebUI | `whitebox_app.py`、`webui/` | 唯一服务入口与静态前端 |

`tilelang_nr/README.md` 给出从公开入口追到计算的文件导览。

## 运行准备

1. Python 3.11 venv，安装 `requirements.txt`（已验证 Torch 2.5.1+cu124 / torchvision 0.20.1 /
   TileLang 0.1.14 / CUDA 12.9 bindings）。
2. `reference/weights_ht_blob.bin` 与 `reference/guide_models/*.pth`：冻结权重，随工作树提供，不入 Git。
3. `.toolchains/cuda12.8/`：项目私有的 CUDA 12.8 NVRTC + CCCL + runtime。TileLang 的 FP8 内核
   （E4M3 转换与 F16 累加）需要 NVRTC ≥ 12.8，由 `runtime.fp8_compiler.private_compile` 只在编译这些内核时
   挂载，系统环境不变。缺它则内核编译直接报错。同样不入 Git。
4. GPU 需 SM89（RTX 40 系），且 NR 只在默认 CUDA stream 上验证。

## 抽取边界

本仓库只保留唯一最终管线：TileLang NR + 外围管线设施 + 冻结参考。相对源仓库已剔除：

- 原版 CUDA NR 的全部实现与分发产物：`native_nr/`（含 `cuda/*.cu`）、`native_nr_dist/`（含 ZIP）、
  `cuda_native/`、`cuda` 默认后端；NR 计划已改为纯 Python，不再需要 NVRTC 编译计划模块；
- TileLang 的中间组织与未采纳候选：旧 `tilelang_nr/legacy/`、endpoint / serial / parallel split、
  wide_packet、旧 joint/UP8 及 `experiments/archive/`；
- NR 侧被 TileLang 取代的 Torch 融合层（`mega_*`、`compact_one_*`、`cooperative_*`、`layout_*`、
  `fp8_one/encoder`、`decoder2h`、`four/eight/one_head` 等）——运行证据显示它们在 tilelang-vit 路径下
  一次也不会被调用；
- 全部诊断与证据脚手架：`check_*` / `benchmark_*` / `analyze_*` / `profile_*` / `diagnose_*`、
  `analysis/`、`evidence/`、`outputs/`、`experiments/`、`gpu_experiments/`、`tools/`。
