# RGB-only估计入口（SDR）

普通RGB序列现在可以从真实入口运行：

**RGB → RAFT-small光流 + Metric VDA Small流式深度 → 显式相机投影 → Feeder guides → Torch FSR2 → 原NR及独立history → RGB输出。**

不需要用户提供depth/MV文件。推理不联网、不调用native FSR参考、不启动服务；NR模型本体和权重未改。它不等价于游戏引擎的真实几何，也不声称等价于DLSS/DFC/RenoDX。

## 用法

本机已经具备匹配的torch2.5.1+cu124/torchvision0.20.1+cu124和einops0.8.0，**没有重装Torch或安装上游旧requirements**。

新环境需要的依赖单列在`requirements-whitebox.txt`。不要把VDA仓库的Torch2.1.1/xformers整套requirements装进现有CUDA环境。

权重需显式准备一次，推理入口不会自动下载：

```powershell
C:/Python311/python.exe tools/fetch_rgb_guide_weights.py --proxy http://127.0.0.1:7890
```

运行：

```powershell
C:/Python311/python.exe -m whitebox_pipeline --manifest rgb-sequence.json --output output-rgb
```

最小manifest：

```json
{
  "mode": "rgb_estimated",
  "color_encoding": "sRGB",
  "output_size": [540, 960],
  "fsr_settings": {
    "camera_near": 0.1,
    "camera_far": 1000.0,
    "camera_fov_y": 1.0471975511965976
  },
  "frames": [
    {"color": "000001.png", "delta_ms": 41.6666667, "reset": true},
    {"color": "000002.png", "delta_ms": 41.6666667}
  ]
}
```

- 尺寸是`[height,width]`。可显式传`render_size`，必须>=32且不超过output；默认取source/output各轴的较小值。
- 图像用opaque8bit RGB文件，或float HWC `.npy`。支持显式`linear`或`sRGB`，当前SDR值域必须[0,1]，不会静默吞掉负数/HDR。
- RGB模式不接受depth/MV文件，也不允许非零jitter。已经完成的RGB图像没有真实渲染jitter，不能补写Halton。
- 可选`reactive`/`composition`仍为独立float HW/HWC1文件；在RGB模式中是source网格，不是render网格。
- 切镜/新序列传`reset:true`。目前是**显式reset，不是自动切镜检测**。改变source尺寸也必须reset。
- `estimator_settings`可指定`flow_longest_side`（默认512）、`flow_updates`（默认12）、`depth_input_size`（默认518）、`depth_fp32`（默认false）和`model_dir`。

Python入口：

```python
from whitebox_pipeline import RGBSequencePipeline
pipe = RGBSequencePipeline((540,960), device="cuda")
result = pipe.process_frame(source_srgb_bchw, reset=is_cut)
encoded_rgb = result["color"]
provenance = result["info"]["estimation"]
```

## 几何、尺寸和状态的真实含义

**Depth不是ground truth。** 使用真正的Metric Small checkpoint，将其输出解释为估计view-Z米；FOV和near/far是显式默认或用户假设，未自动标定。使用固定投影公式转换device depth，绝不逐帧min/max归一化或用常量代替估计器。超近/超远值按固定范围截断并进入明确的FSR reactive保护；整帧没有有效范围内深度会报错，不退化为常量边界深度继续冒充时域处理。

RAFT调用顺序固定`current, previous`，输出current→previous像素。source、flow工作网格、depth工作网格、render、display、neural buffer分别记录。RAFT缩小后8对齐/最小128并在右下复制边缘，返回时按真实resize比例换回source像素；送render前再按各轴比例换算，不能只resize数值而忘记向量单位。

VDA保留上游**实验性因果streaming**的32位置attention、首帧hidden-state复制初始化和最长42项cache选择规则。复制cache不表示已经观察32帧，报告另记真实帧数。上游明确指出该模式相对offline有精度下降；本实现不声称offline质量。

Torch预处理复现相同aspect/multiple14策略与bicubic(-0.75)数学，不声称与任意OpenCV版本逐bit一致。没有OpenCV、xformers或easydict运行依赖：只vendor神经网络部分，使用上游Torch attention分支，EasyDict仅换成普通dict构造kwargs。

估计器先生成proposal，只有FSR和NR都成功后才提交RGB和depth缓存。失败会保留上次完成帧的estimator、guide、FSR及NR状态。NR输出从不回到估计器或FSR；原始source RGB才是下一帧RAFT参照。所有状态均按序列隔离。

## 已实际验证

1. 用户原图549×510的独立已知4px平移：真实RAFT测得中位数`(-4.0430,-0.0266)`px，内部区域**分量MAE0.05667px**（不是Euclidean EPE）。Metric Small本例输出0.485–41.635估计米，未做范围拟合；没有真实深度用来证明米制准确性。
2. 上游真实视频48帧、source960×540，无depth/MV文件，经公开`run_manifest`完整运行；render/output224×126用于长序列/cache验证。最长cache42项、显式reset、NR后段失败后的全组件状态恢复、reset首帧逐值重现全部通过。模型加载后共102.0秒，峰值约1965MiB。该clip没有transfer标签，测试显式按SDR sRGB解释，不当成颜色标定结果。
3. 另通过实际`python -m whitebox_pipeline`处理同一视频4帧，**source/render/output均960×540，neural576×1024**。不是只用低分辨率输出代表整条能力；后续帧约4.86–5.40秒/帧，非实时。
4. RGB模式拒绝混入提供的guides、虚构jitter、缺失权重、错误SHA；加载或帧处理失败写出`failed`报告，不静默替代。

结果与图片在：

- `debug-static/rgb-estimators/report.json`
- `debug-static/rgb-sequence-verification.json`
- `debug-static/rgb-real-video-output/`（48帧）
- `debug-static/rgb-full-size-output/`（960×540）

这些验证证明实际执行、尺度/方向和状态行为；**没有证明该真实视频的深度准确性或所有场景的主观质量**。

## 固定资产与许可

- torchvision0.20.1，RAFT `Raft_Small_Weights.C_T_V2`，4,006,189 bytes。
- Metric VDA Small，116,444,063 bytes；HF revision `273d090f2ce17df50c2872d82c8322c45da5b4dd`，官方LFS SHA256 `3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33`。
- VDA源码revision `4f5ae23172ba60fd7bc11ef671cca678842c7072`；vendor范围、原始/修改后hash和修改说明在`whitebox_pipeline/_vda/SOURCES.json`。
- `guide_models/manifest.json`保存完整权重SHA。Metric Small模型卡明确Apache-2.0；Base/Large的NC权重未取用。VDA/Meta来源声明和Apache全文保留在vendor目录。
- RAFT的torchvision源码BSD-3-Clause不自动解决训练数据/权重的所有商用问题；现有NR权重许可单列。未把整体宣称为MIT/Apache，也未把测试视频纳入分发。

HDR→NR颜色桥、多pass独立history、尺寸/alpha/保护与输出策略已扩展到同一入口，详见`P3_COLOR_AND_HISTORY.md`。本文保留SDR默认用法与P2证据；视频codec、任务UI与独立分发用法见`WHITEBOX_APP_README.md`。
