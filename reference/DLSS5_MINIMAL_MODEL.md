# DLSS5 最小神经模型定义与完成边界

## 结论

当前工程针对本地 `nvngx_dlssnr.dll` 实际提供的唯一学习模型：Preset #1、153 条权重记录、block0–70。全部学习块已接入恢复后的真实权重、物理布局与量化公式；旧 `ArchiveFusedSwin` 等六个结构代理类、切换开关及 fallback 构造均已删除，不再是“默认绕过、仍可切回”的状态。

当前固定权重图已取得完整端到端的**算法对齐证据**，但**标准Torch逐bit数值等价不作承诺**。本机正常 Feature18 的逐层及原始纹理 oracle 已建立。2026-09-06最新统一交付验证中，原人物图 Natural + LocalStructure2（全71层自馈）记录 head 相关性 0.991973、Torch/DLL head 绝对值均值 0.138340/0.137036、原始 RGB MAE 0.00435551。320×320神经尺寸的对应对照为head相关性0.969353、输出MAE0.00607497；448×512为0.996653/0.00197551。小图相较此前9f62fc3的结果变差，未隐瞒或据此回退已独立证明的算法。后续无中间注入的完整自馈诊断进一步确认：仅在debug中匹配矩阵算术与同一Box–Muller公式的基本函数实现，small/原人物/448三实图的block0 compact、full skip、block70输入全部逐值一致；计入已观察到的FP16输出转换后，小图与448图RGB全等，原人物仅一个分量差最小FP16次正规数约5.96e-8。首包18个adapter差异已明确归因到基本函数实现精度，不是新的网络图缺口。结合独立源码、地址和控制实验，当前固定权重最小模型已取得完整端到端算法对齐证据。交付仍是标准Torch，上述纯Torch测量值没有改变，也不宣称跨设备逐bit等价。完整架构与验证范围见[DETAILED_MODEL_ARCHITECTURE.md](DETAILED_MODEL_ARCHITECTURE.md)；开发仓库中的原始取证记录为debug-static/END_TO_END_ARITHMETIC_ATTRIBUTION_20260906.md（不随分发包提供）。此前幅度严重偏弱的问题不再复现；指标和输出接近仍不能替代完整算法证明。

1H已补回两条被误放的量化边界：FFN残差进入第一次W2累加，attention残差保留FFN的FP16输出，只有QKV读FP8副本；P×V第一K32覆盖keys0–15与48–63。独立诊断按真实操作数顺序使用本机算术后，完整block1在自然图、gradient及随机E4M3三组输入上，每组2,621,440元素均与DLL全等。交付路径仍为纯Torch，不包含该诊断PTX。其他家族不能照搬1H：独立字节来源追踪证明2H/4H/8H/split16H/ViT的attention残差确实经FP8发布/重读。完整8H block15也已用当前图直接替换矩阵算术核验：真实20×20×256输入上，纯Torch MAE为0.074243，而native-arithmetic诊断的102,400个输出全部相同；没有替换路由或scalar公式。这确认了该代表块的算术误差来源，但不能直接把其他整图差异都归为硬件。

ViT已改回真实的 `E4M3(exp) → FP16 P×V → half reciprocal → E4M3`，而非先量化归一化概率；其K64归约、虚拟key先加后扣及logical/fragment token映射均有SASS和原CUBIN验证。8H/16H的learned scale也在乘法前显式转Half。五个实际神经尺寸512×640、512×576、384×640、320×320、448×512均已跑完71层，并确认原始BIN单文件版与模块版完整head逐值相同；这仍不是全图DLL逐值等价声明。

“完成”只在网络图、真实权重语义、tensor布局、FP16/E4M3边界、学习算子、skip/resize和四通道head都进入Torch路径，并由同层NVIDIA oracle验证后成立。当前固定权重图已取得上文的完整自馈证据；这不是所有输入、设备及宿主组合的形式化证明，也不以 finite、shape 正确或输出幅度接近替代算法等价。

## 最小模型契约

```text
输入  prepared_features: [B,16,H,W]

  0..3    g1, g2, g0, 1（坐标/frame PCG hash 的 Box-Muller companion）
  4..7    current.RGB, source.R
  8..15   source.GB, style/128, tone, structure, skin, auto, 0

  current/source RGB 均为 FP16 中的 (value-0.5)/8；source 在
  PrevOutput 与 MVec 同时绑定时才切换为正向 warp 的五 tap history。

学习模型

  block0 FP16 HMMA K16xN32 adapter + full-resolution physical 1H
  block0 output0 2x compact chain + output1 full-resolution post skip
  blocks1..22 encoder
  blocks23..38 bottleneck
  blocks39..69 decoder
  block70 main/skip gate + physical 1H + FP16 4x32 head

输出  head: [B,H,W,4]

  head[...,0:3]  learned RGB residual
  head[...,3:4]  learned history logit
```

Color/PrevOutput/MVec如何生成16通道pre-HMMA packet，以及head如何与Color/history/Backbuffer合成，属于模型接口边界；packet中的Gaussian与conditioning slot公式已由block0 SASS恢复，不能再用旧七通道近似替代。

## IDA依据：哪些值会改变模型

### 唯一weight/graph选择：Preset

`CG2RFindWeightByPreset`位于`sub_180023A40`。descriptor表为：

```text
0x1800B0D80 .. 0x1800B1008
```

该DLL构建只有一个descriptor，preset ID为1。请求不可用ID时明确回退shipping default Preset #1。因此当前`weights_ht_blob.bin`已经覆盖这版DLL所有实际可用的学习权重；不存在尚未导出的Preset #2/#3网络。

### 其余设置只绑定pre/post接口

`sub_18003F490`遍历网络layer，但只对以下类型执行runtime setter：

```text
CCTinlayoutFusedPreBlockSwin1HLayer
CCTinlayoutFusedPostBlockSwin1HLayer
```

它不替换blocks1–69、不修改weight record、不改变graph。Style、LocalTone、LocalStructure、SkinStructure、AutoMask、UI/UIAlpha、Depth、MVecScale、Intensity和ControlMask均属于以下一种：

- 16通道pre-HMMA packet构造；
- 外部resource/subrect绑定；
- history生命周期；
- block70之后的确定性颜色合成；
- pre/post kernel variant选择。

它们不是新的学习权重或隐藏网络层，因此不列入最小模型完成条件。

## Torch接口

模块化维护版：

```python
import torch
from torch_rebuild import DLSS5Reconstruction

model = DLSS5Reconstruction().eval()
prepared = torch.zeros(1, 16, 512, 640)  # 仅演示接口；实际需构造16通道packet

with torch.inference_mode():
    result = model.forward_minimal(prepared, collect_trace=False)

head4 = result.head       # [1,512,640,4]
latent = result.latent    # block70 32-channel latent
```

单文件版：

```python
import torch
from dlss5_model import load_model

model = load_model("weights_ht_blob.bin", device="cpu")
prepared = torch.zeros(1, 16, 512, 640)  # 仅演示接口；实际需构造16通道packet
head4 = model.infer_minimal(prepared)
```

如需trace：

```python
result = model.infer_minimal(
    prepared,
    collect_trace=True,
    return_result=True,
)
assert result.head.shape[-1] == 4
assert len(result.trace) == 71
```

## 与完整帧便利接口的关系

`FrameInputs`和`infer_frame()`仍保留，负责演示Color/PrevOutput/MVec到16通道packet以及head4到RGB的已恢复默认映射。它们是便利封装，不再作为最小模型完备性的判断依据：

```python
rgb = model.infer_frame(frame_inputs)
```

未实现某个UI/HDR参数不等于缺少学习权重；但任何会改变16通道packet或head合成的参数，都必须按已恢复的pre/post公式映射，不能从模型契约中静默省略。

## 文件组织

```text
torch_rebuild.py                 模块化模型主实现（维护真源）
physical_1h_ffn.py               1H物理FFN
physical_1h_block.py             1H attention/window
physical_2h_block.py             2H physical block
block70_color.py                 默认外部颜色合成便利层
dlssnr_inputs.py                 已恢复的16通道packet构造便利层
build_single_file_model.py       生成单文件

dlss5_model.py                  单Python文件交付实现
weights_ht_blob.bin              原始153-record权重archive
DLSS5_MINIMAL_MODEL.md           最小模型边界与完成定义
DLSS5_MODEL_API.md               完整API说明
```

算法修改应落在模块化维护源码，再运行：

```bash
python build_single_file_model.py
```

重新生成`dlss5_model.py`，避免模块版与单文件版漂移。
