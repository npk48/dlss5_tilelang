# TileLang NR 源码导览

按实际计算功能组织。算法组合以 797fd63 的 `VitJointTileLangNR` 为准，公开入口经
`run.Engine` → `app.backend.Backend` → `runtime.VitJointTileLangNR` → `registry.resolve(organization='vit')`，
每个计划为 Step 绑定空间数据，编译产物则按模型/硬件结构共享。

## 从公开入口追到计算

| 活跃文件 | 负责的计算 |
| --- | --- |
| `runtime.py` | Step 替换、buffer view 生命周期、覆盖统计 |
| `registry.py` | 唯一的 `vit` 组织：按名字直接选定工厂 |
| `spec.py` | `StepSpec` / `BufferViews`：把计划里的指针解析成非拷贝 Torch 视图 |
| `plan/` | 纯 Python 计划：arena 布局、aux 轨道、buffer 归属、每步几何与 typed 参数 |
| `kernels/shallow_endpoints.py` | 1H 浅层、block0 输入/池化、post70 输出；10 个逻辑 Step |
| `kernels/heads24.py` | 2H/4H FFN、attention、projection、DS/UP；21 个逻辑 Step |
| `kernels/heads8.py`、`kernels/heads8_layout.py` | 8H 五种角色、DS 行归属和 portrait UP 的物理投影；16 个逻辑 Step |
| `kernels/heads16_bridge.py` | 16H、local64、raw/partial 发布、merge、repack、UP 出口；102 个逻辑 Step |
| `kernels/vit_bridge.py` | 八组 ViT 的矩阵/QKV/attention，以及 57/100 物理桥；42 个逻辑 Step |
| `kernels/utility.py` | 原计数器/状态清理和 pool padding |
| `common/` | 共享 Half 运算、MMA fragment、物理地址/路由、completion ABI 和默认 stream 的调用封装 |
| `instructions/` | 小 PTX、packed-Half、访存指令适配；计算循环与布局仍在真实 TileLang DSL 中 |

循环中的 `K/N/M`、`PK`、`C/A/B` 保留矩阵/分片含义；`raw` 是物理发布布局，`partial` 是按既定顺序归约的
Half 分片，不能当作普通行主序中间 tensor。共享 helper 不合并"看起来相似"但归约次序、
`serial/unroll`、字节路由或舍入边界不同的实现。

## 计划与执行

`plan/runtime.py` 构建整条 NR 的计划：arena 与全部 skip/aux/weight buffer 的分配、
每步的 launch 几何和 typed 参数。它**不编译也不绑定任何 CUDA kernel**，只产出
`PlanStep(name, geometry, values)` 与 `Storage` 里的实际 buffer。

`runtime._prepare` 逐条取出这些 Step，把其中**每一个逻辑 Step 的 function 换成 TileLang 工厂产物**
（`PortedStep`）；`spec.tensor(i)` 再用 `BufferViews` 把参数里的指针解析成真实的 Torch 视图。
执行期没有任何原版数学 Step，也没有回退路径——工厂缺失或编译/数值失败都直接抛错。

## 覆盖

`report()` 里的 `coverage[].selected_factories` 给出最终选中的工厂分布，`tilelang_steps` 计真正执行过的
逻辑 Step。193 = utility 2 + shallow 10 + 2H/4H 21 + 8H 16 + 16H 102 + ViT 42。

## 运行时空间几何

`common/runtime_jit.py` 的 `spatial_jit` 只把结构参数放入编译缓存键。空间 H/W、grid、buffer 长度、offset、completion 和路由参数成为显式 int32 PrimFunc 参数；新计划仅建立 `BoundKernel`，即使旧 plan 被驱逐也不会因网格变化重新编译。原有输入对齐、尺寸和显存限制仍有效。

运行时整数地址表达式使用选择性的 opaque binding 保留 DAG，防止 TVM 把它们反复展开。仅处理纯整数运算，不隐藏 load/external call，也不关闭正常 Simplify 或安全访存。GPU 编译器将 `max(x,x)` 化简为原值。通道、MMA、FP8 及浮点计算顺序未改变。

`common/bound_launch.py` 绑定 TileLang 生成的 host launch 配置，并复用 CUDA 参数存储，减少逐 Step 参数封装。它不编译另一套 GPU 代码：仍执行原 TileLang CUBIN。每次调用刷新 tensor 指针和当前 stream；参数存储在 driver 消费期间受锁保护。核心 NR 本身仍遵守原默认-stream/单 workspace 使用边界。

报告中的 `runtime_compilation` 是进程级结构缓存统计；`coverage[].runtime_geometry` 展示当前 arena/grid/deep geometry。`jit_builds` 包括从磁盘重建的结构绑定，不能把它等同于 NVRTC 次数。所有 pool geometry 都绑定 padding kernel，避免首次遇到需要 padding 的网格才补编译。

## 复现验证

`native/tools/bench_nr_dynamic.py packets <directory>` 生成确定性真实模型输入 packet；`bench_tilelang_dynamic.py` 和 `run_tilelang_pair.py` 分别运行完整 NR 与冻结版本交替配对。`check_tilelang_dynamic_pipeline.py --generate` 生成实际 Engine.run 输入；随后运行该脚本验证混合尺寸、连续帧、pass/settings 改变和切回。

`tilelang_nr/tests/test_runtime_jit.py` 覆盖空间整数绑定规则；`tests/check_bound_stream.py` 在 GPU 上验证准备式 launch 的新指针、两个非默认 stream 和默认 stream。最终证据见 `native/qualification/tilelang-runtime-spatial.json`；性能仍存在已量化的回退，不是零开销动态化。
