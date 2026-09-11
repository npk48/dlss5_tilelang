# P3：HDR、多pass、工作尺寸与输出保护

当前共同入口支持guided和RGB-estimated的SDR/HDR序列。NR学习图、固定权重和既有量化算法不变。这里定义的是**自有白箱宿主策略**，不是DFC/RenoDX的codec、Native Look或Clean Fry复刻。

## 颜色输入与输出

`color_encoding`必须显式指定：

|编码|输入内容|文件|
|---|---|---|
|`sRGB`|[0,1] SDR编码RGB|8bit图片或float HWC npy|
|`linear`|[0,1]相对线性BT.709|图片需明确自认linear，或float npy|
|`linear709_nits`|有符号线性BT.709，单位nit|float HWC npy|
|`PQ2020`|[0,1] ST.2084/PQ、BT.2020 RGB|float HWC npy|

HDR转换采用标准ST.2084常数及D65 BT.2020/709矩阵；BT.709工作输入限制为±20000nit，包含10,000nit PQ色域转换后的分量。负分量不能直接送有界NR/FSR，因此它们不进入正色视图，但保留在原始工作context中。PQ输出必须截断其不能表达的负BT.2020分量；线性输出保留有符号结果。

`color_settings`包含：

- `reference_white_nits`：默认203，是声明的自有归一化参考，不冒称原宿主值。
- `peak_nits`：默认1000、允许到10000，**限制逆codec的差分解码范围**，不是强制把整个原始context截到该峰值的mastering tone mapper。
- `output_encoding`：SDR可选`sRGB`/`linear`；HDR可选`linear709_nits`/`PQ2020`。

HDR供FSR的线性正色按reference white归一化。原始负分量以当前帧的display网格context保留。供估计器和guide验证的SDR视图明确定义为同样的正色压缩+sRGB，alpha存在时在该视图上合成到黑色；这些都不等于原场景标定。

NR编码只包住整条pass链一次：

```
u = max(linear709_nits, 0) / reference_white_nits
t = u / (1 + maxRGB(u))
encoded = sRGB(t)
```

逆变换保留色比并限制到声明的逆变换peak。最终不是直接丢弃原图做一次有损roundtrip，而是：

```
output_linear = original_FSR_context
              + decode(modified_encoded) - decode(reference_encoded)
```

因此全局`output_settings.mix=0`、完全保护的像素、或无其他效果的零NR差分，保留同一**FSR输出线性context**。这不意味着跳过FSR，也不保证PQ编码字节经过标准浮点转换后逐bit不变。没有LDR→真实HDR恢复功能。

## 多pass与参数变更

- `nr_passes`是1–30项`NRSettings`；不同时传`nr_settings`。30是控制上限，不是显存/性能承诺。
- 每pass都有自己的完成图像、帧编号和历史有效性。后pass消费前pass结果，但不共享history。
- `style/structure/tone/skin/automatic_mask/temporal_strength`沿用既有模型与宿主接口。
- 新`intensity`是每pass的[0,1]输出mix，**不是learned head gain**。原默认值1保持既有路径。
- `configure_nr(passes, work_size=None)`保留共同且未改变的上游前缀；从首次变化的pass开始重新初始化，包括下游。增删pass同理。
- 原完整SDR默认路径已与`0cfcca0`的实际两帧、同一真实NR模型输出逐值比较，`torch.equal`。

## 工作尺寸、EASU与保护

`nr_work_size: [height,width]`独立于source/render/display以及内部64对齐neural buffer。工作尺寸改变只重置NR链，不把NR输出倒灌FSR。尺寸变化不是对含global ViT的NR做tile后声称等价。

较小工作图的结果通过**明确的encoded差分传送**回完整display context：

- `output_settings.nr_resample="bilinear"`：上采样`processed_work - original_work`。
- `"easu"`：`FSR1_EASU(processed_work) - FSR1_EASU(original_work)`；完整12tap、梯度/各向异性/近邻deringing和原近似倒数都在Torch实现。它不是bilinear换名。
- EASU只允许放大；更大NR work向下采样用bilinear，冲突明确报错。不将该差分策略说成全尺寸NR等价。

输出效果全部在NR历史保存之后：

- `mix`：全局NR/输出效果mix，[0,1]。
- `sharpness`：`None`关闭；[0,1]启用唯一一个最终RCAS，作用于有界encoded RGB。FSR内部RCAS在此共同入口固定关闭，避免叠加锐化。
- `detail_only`：NR差分减去复制边界的3×3均值，仅保留差分高频。
- `max_chroma_delta`：YCoCg中Co/Cg差分限幅；HDR按reference white归一化。
- `max_luma_stops`：BT.709亮度的log2变化限幅，黑电平floor明确为reference white/65536，通过中性RGB修正亮度。

这些是自有可审计输出算法，不用于掩盖NR计算错误。更改输出mix、锐化、guard、输出传递函数或逆变换peak**不重置history**。更改HDR/SDR工作模式或reference white会重置所有受影响状态，包括RGB估计器。

`protect`是**display网格**float HW/HWC1，1表示保留FSR context、不受NR/锐化/guard影响。它不是Feeder distrust、Feature1 reactive，也不是模型内部automatic mask。

## Alpha、reset、取消和诊断

- RGB/RGBA都可输入；straight alpha为默认。
- `alpha_mode="premultiplied_linear"`只接受`linear`或`linear709_nits`。明确拒绝把PQ/sRGB编码域当作线性预乘来解。
- 解预乘后以straight工作；透明及alpha变化进入声明的reactive保护。alpha在render/display之间做线性resize，零alpha位置不应用NR效果。
- 输出可选`output_alpha_mode="premultiplied_linear"`，但仅在线性输出编码下。零alpha的预乘RGB保持零。
- `resize(output_size)`明确重建受影响的FSR/guide/NR状态；不把它称为保留history的DRS。
- `cancel`回调在估计后、FSR后、每个NR pass之前及最终提交前检查；当前GPU调用不会被强行中断。取消或异常均回滚整帧状态，不能提交已完成的第一pass而遗失失败的第二pass。
- 配置事件发生在帧之间，立即应用其失效规则；若下一帧失败，新配置仍在，旧且不适配的history不被复活。
- `guide_settings.geometry=true`运行既有920样本两轮拟合和选择。`geometry_diagnostics=true`可只计算诊断、保持原motion/mask不变。`save_guides=true`输出motion、distrust、四项tests、geometry decision的NPZ，标量fit/测试均值进入JSON。

## 示例

```json
{
  "mode": "rgb_estimated",
  "color_encoding": "linear709_nits",
  "output_size": [540, 960],
  "nr_work_size": [270, 480],
  "color_settings": {"reference_white_nits":203, "peak_nits":4000, "output_encoding":"PQ2020"},
  "nr_passes": [{"structure":2}, {"structure":1}],
  "output_settings": {"nr_resample":"easu", "sharpness":0.35, "max_luma_stops":0.5},
  "frames": [
    {"color":"first.npy", "reset":true},
    {"color":"second.npy", "nr_passes":[{"structure":2},{"structure":1,"tone":0.8}]}
  ]
}
```

同一`python -m whitebox_pipeline --manifest ... --output ...`入口。帧级可更新`nr_passes/nr_work_size/output_size/color_settings/output_settings`。HDR写float NPY真输出及**另附SDR PNG预览**；PNG不是HDR交付物。

## 运行证据与边界

- `tests/check_p3_pipeline.py`使用真实固定NR：旧SDR两帧equal、HDR双pass、保护/mix0精确context、下游reset、工作尺寸变更、第二pass异常回滚、pass间取消、target resize，以及实际RGB-estimated HDR→PQ/RGBA manifest均通过。
- `debug-static/p3-full-output/report.json`：真实960×540图像内容构造的明确HDR辐亮度（不是原生HDR摄像数据），原尺寸运行两pass；首帧NR work540×960/neural576×1024，之后切270×480/neural320×576并实际走EASU+RCAS+guard，输出保持960×540。首帧10.58s，后两帧5.55/5.32s。geometry fit和guide sidecar均产生，历史在尺寸切换后重新生效。
- 同网格alpha逐值保留，透明预乘RGB为零；线性HDR输出范围最高约1572nit，没有被clamp为SDR。
- EASU对独立标量12tap方程最大误差`9.35e-7`，黑/灰/白常量exact；这是数学移植验证，**不是新增native EASU GPU oracle**。
- `p3-math-verification.json`、`p3-controls-verification.json`记录PQ基准、inverse peak、色域与预乘检查、配置失效范围和无作用diagnostics。
- `debug-static/p3-pq-output/report.json`记录另一次实际公开CLI：PQ2020/RGBA浮点输入→RGB估计→FSR/NR→PQ2020浮点输出完成；不是只对转换函数做孤立测试。

NR核心、权重和旧分发包没有更改。仍不承诺相机/估计深度准确性、全场景主观质量、原生HDR capture等价或实时性能。P4任务/codec/UI已接入，使用与分发说明见`WHITEBOX_APP_README.md`；实验FSR opaque-only TCR、保留history的DRS、4K→8K TextureBoost仍不是当前声明的能力。
