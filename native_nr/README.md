# NR 计划层（native_nr）

在本仓库里 `native_nr` **不再作为运行后端**，它只为 TileLang NR 提供计划元数据：
packet / layout 计划、launch 几何与 buffer 归属。原版 CUDA 数学 Step 不会被执行。

## 结构

```text
native_nr/
├── __init__.py
├── runtime.py       # NativeNR、OuterRuntime、DeepRuntime：71 层调用组织与计划构建
├── device.py        # CUDA/NVRTC、Step 参数所有权、BIN 及 typed 权重参数
├── layouts.py       # 尺寸与浅层/过渡/深层地址公式
├── source.py        # 功能编译单元选择与三份动态 ownership 表
├── cuda/            # 8 个编译单元：shallow / heads2 / heads4 / heads8 / deep16 / vit / bridge / utility
└── runtime_manifest.json
```

## 在管线中的位置

```text
run.Engine / whitebox_app
  -> 原完整白箱前后处理与 NRChain
  -> Backend.checked_infer
  -> tilelang_nr.runtime.VitJointTileLangNR
       -> native_nr.NativeNR._prepare   # 只取计划：Step 列表、几何、buffer 归属
       -> 每个 Step 换成 TileLang 工厂产物
  -> outer encoder -> deep -> outer decoder/head （全部由 TileLang 执行）
  -> 原混合、每 pass 历史与输出
```

- `cuda/*.cu` 在 `_prepare` 时被编译，仅用于取得计划元数据；TL 路径不调用其中任何 kernel。
- 编译需要项目私有工具链 `.toolchains/cuda12.8`（NVRTC/CCCL/runtime），可用
  `NATIVE_NR_TOOLCHAIN` 覆盖；缓存写在 `.cache/native_nr/`。
- `source.py` 会按 shape 生成 ownership 表并拷贝编译单元到 `.cache/native_nr/shapes/<H>x<W>/`。
- 输入仍是真实 CUDA contiguous Float32 B×16×H×W packet；返回独立拥有的 Float32 B×H×W×4 head。
- 原 neural_size 策略、2-shape 缓存、batch、异常恢复、默认 CUDA stream0、SM89 限制不变。
