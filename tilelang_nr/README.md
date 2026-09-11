# TileLang NR 源码导览

按实际计算功能组织。算法组合以 797fd63 的 `VitJointTileLangNR` 为准，公开入口经
`run.Engine` → `backend.Backend` → `runtime.VitJointTileLangNR` → `registry.resolve(organization='vit')`，
**每个被选中的 Step 只构建一次**。

## 从公开入口追到计算

| 活跃文件 | 负责的计算 |
| --- | --- |
| `runtime.py` | Step 替换、buffer view 生命周期、覆盖统计 |
| `registry.py` | 唯一的 `vit` 组织：按名字直接选定工厂 |
| `spec.py` | `StepSpec` / `BufferViews`：把计划里的指针解析成非拷贝 Torch 视图 |
| `plan/` | 纯 Python 计划：arena 布局、aux 轨道、buffer 归属、每步几何与 typed 参数 |
| `shallow_endpoints.py` | 1H 浅层、block0 输入/池化、post70 输出；10 个逻辑 Step |
| `heads24.py` | 2H/4H FFN、attention、projection、DS/UP；21 个逻辑 Step |
| `heads8.py`、`heads8_layout.py` | 8H 五种角色、DS 行归属和 portrait UP 的物理投影；16 个逻辑 Step |
| `heads16_bridge.py` | 16H、local64、raw/partial 发布、merge、repack、UP 出口；102 个逻辑 Step |
| `vit_bridge.py` | 八组 ViT 的矩阵/QKV/attention，以及 57/100 物理桥；42 个逻辑 Step |
| `utility.py` | 原计数器/状态清理和 pool padding |
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
