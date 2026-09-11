# `dlss5_model.py` API

`dlss5_model.py` 是固定当前DLL Preset #1的单Python文件PyTorch实现。运行时只需要：

```text
dlss5_model.py
weights_ht_blob.bin
```

它直接解析原始BIN的153条权重记录，并内置网络图、物理layout和routing资料，不需要预导出的`.npy/.npz/JSON`。

最小模型的边界见[DLSS5_MINIMAL_MODEL.md](DLSS5_MINIMAL_MODEL.md)，详细架构见[DETAILED_MODEL_ARCHITECTURE.md](DETAILED_MODEL_ARCHITECTURE.md)。分发包同时附带独立WebUI入口；解压、安装和启动方法见[README_DISTRIBUTION.md](README_DISTRIBUTION.md)。WebUI不是上述两个文件的模型API依赖。

## 安装与加载

```bash
pip install numpy torch
```

```python
import torch
from dlss5_model import load_model

model = load_model(
    "weights_ht_blob.bin",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

加载器验证BIN总长度、153条record framing、payload/trailer、名称唯一性与物理矩阵解码。模型实例持有私有临时解码缓存；长期服务应复用同一实例。

## 最小模型接口（推荐）

```python
prepared = torch.zeros(B, 16, H, W)
head4 = model.infer_minimal(prepared)
```

输入：

```text
prepared: float tensor [B,16,H,W]
```

输出：

```text
head4: float tensor [B,H,W,4]
head4[...,0:3] = learned RGB residual
head4[...,3:4] = learned history logit
```

`infer_minimal()`只执行固定权重的学习模型。Color/MVec资源映射、Style/LocalTone等输入策略、history ownership、HDR/UI以及最终颜色合成均留给调用者。

这里的 `H/W` 是**神经 buffer 的尺寸，不是原 PNG 尺寸**。本机正常 Feature18 已确认：128×128 图片实际使用320×320神经 buffer，ViT descriptor为8×8（64 tokens）。不能拿128×128直接生成的神经tensor或人工16/32-token kernel调用，替代这个真实小图路径。调用者应按实际神经尺寸构造16通道输入，再按输出图尺寸裁剪/合成。

### Color补边：一次反射，再钳制

准备宿主已协商尺寸的Color资源时，不要直接使用`np.pad(..., mode="reflect")`。DLL在超出原尺寸时只做一次`2*size-2-coordinate`反射，随后由采样器钳制；它不是无限周期反射。原图128×128而神经buffer为320×320时，这个差别会明显改变模型输入。

```python
from dlss5_model import pad_color_for_neural_buffer

# rgba是宿主实际Color，BHWC；不是在这里改变sRGB/linear或HDR颜色域。
color = pad_color_for_neural_buffer(rgba, neural_height, neural_width)
```

该工具保持设备和dtype，只处理右侧/底部扩展，不决定神经尺寸、不执行resize或颜色转换。之后再按已有输入策略构造16通道packet，或将它交给`FrameInputs`资源适配器。

获取latent和trace：

```python
result = model.infer_minimal(
    prepared,
    collect_trace=True,
    return_result=True,
)

print(result.head.shape)    # [B,H,W,4]
print(result.latent.shape)  # [B,H,W,32]
print(len(result.trace))    # 71
```

`H/W` 是神经缓冲区尺寸，不是显示图像尺寸。已从完整入口实测 `(H,W)=(512,640)、(512,576)、(384,640)、(320,320)、(448,512)`；五种尺寸均跑过全部71层，单文件完整 head 与模块版逐值相同。320/448的半组地址、4H publication和8H进出桥另有本机原生执行证据。不能把“16的倍数”当作任意尺寸均已验证的保证，也不能为了绕过错误而擅自 pad 到某个参考尺寸。真实图片须按实际宿主协商结果构造 packet。

## 完整帧便利接口

默认Color/PrevOutput/MVec映射仍作为可选便利层保留：

```python
from dlss5_model import FrameInputs

frame = FrameInputs(
    color=color,                    # [B,H,W,3/4]
    prev_output=previous,           # [B,H,W,3/4]
    mvec=mvec,                      # [B,H,W,2]
    mvec_scale_xy=torch.tensor([1.0, 1.0]),
    output_dimensions_wh=torch.tensor([W, H]),
    local_structure_strength=1.0,
    style=0,
)

rgb = model.infer_frame(frame)
```

默认映射：

```text
model input = {
  g1, g2, g0, 1,
  centered current RGB, centered selected-source R,
  centered selected-source G/B,
  style/128, tone, structure, skin, auto, 0
}

selected-source = current Color unless PrevOutput and MVec are both bound
centered color = FP16(value - 0.5) * FP16(0.125)

current = saturate(Color + head4.rgb/4)
history = CatmullRom(PrevOutput, uv + MVec*scale)
alpha = saturate(blend_scale)*sigmoid(head4.a)
rgb = current + alpha*(history-current)
```

这个便利接口不是最小学习模型契约。LocalTone、SkinStructure、AutoMask、UI/HDR等宿主策略即使未封装，也不表示模型缺失权重或学习层。

## Output composite便利层

如需已恢复的Backbuffer/ControlMask合成：

```python
frame = FrameInputs(
    color=color,
    prev_output=previous,
    mvec=mvec,
    mvec_scale_xy=torch.tensor([1.0, 1.0]),
    output_dimensions_wh=torch.tensor([W, H]),
    intensity=0.7,
    backbuffer=backbuffer,
    control_mask=control_mask,
)
```

```text
strength = intensity
strength *= saturate(ControlMask)  # mask存在时
final = saturate(Backbuffer + strength*(rgb-Backbuffer))
```

这里的`control_mask`是`DLSSNR.ControlMask`，不是标准DLSS的`BiasCurrentColorMask`。

## 直接使用nn.Module接口

```python
# 最小模型
minimal = model.forward_minimal(prepared, collect_trace=False)

# 带外部映射的完整便利路径
full = model(
    prepared,
    block70_runtime=frame,
    collect_trace=False,
)

print(full.output)
print(full.head)  # 始终保留完整四通道学习head
```

## 完成与证据边界

当前单文件包含本地DLL唯一可用Preset #1的完整71-block学习模型和153条权重。其他DLSSNR参数经IDA确认只设置pre/post资源和控制字段，不改变内部graph或weights，故不属于最小模型完成条件。

实现使用 PyTorch 运算表达恢复后的 FP16/E4M3 边界、归约树、残差初值和 ViT split-K，不要求复刻 QMMA 内部算术或 GPU 原子归约的到达顺序。交付模型不依赖 CUDA 自定义核，诊断脚本中的本机 PTX oracle 不会打包进模型。

本机已建立独立的正常 Feature18 逐层、受控权重和原始纹理 oracle。原人物图的完整 head 相关性约 0.9547、原始 RGB MAE 约 0.01083；仍未证明完整数值等价。单文件包含固定模型的全部实际学习路径，旧结构代理类与 fallback 已删除；`runtime_complete=True` 仅表示已提供颜色合成参数，不能解释为“与 DLL 完全相同”。

## 维护

`torch_rebuild.py`及相关模块是维护真源。修改后运行：

```bash
python build_single_file_model.py
```

重新生成`dlss5_model.py`，不要单独手改生成物。
