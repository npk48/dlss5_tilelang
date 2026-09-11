# 完整白箱图像处理管线：调研结论与实施计划

日期：2026-09-06。状态：**P1–P4软件能力已接入实际入口；CLI/UI用法见WHITEBOX_APP_README.md，独立包以运行验证报告为准**。

使用与证据见 [WHITEBOX_PIPELINE.md](WHITEBOX_PIPELINE.md)。常规FSR2时域阶段已移植到Torch，实际manifest入口通过24帧及用户原图549×510场景；独立native runner仅是参考，不是产品后端。原参考FP32标签错误已纠正，新参考明确选择官方FP32 shader。

本文承接 `FEEDER_IMAGE_PIPELINE_GAPS.md`。用户认可“用开源时域算法替代DLSS Feature 1，接已恢复NR核心”的方向，但将其明确称为暂时想法；下面的具体版本、模型、默认值和交付阶段都是**建议**，不冒充已经确认或已经完成。

## 1. 目标、非目标与完成标准

### 1.1 目标

构建一套可独立处理单图和连续帧的白箱系统：

```text
图像/图像序列 + 已提供或估计的guides
→ guide处理、几何与采样规范
→ 开源时域抗锯齿/重建/超分
→ 明确的NR颜色与参数接口
→ 现有已恢复的71-block NR核心
→ 颜色恢复、必要的输出缩放/锐化/保护
→ 最终图像/图像序列
```

主图像计算使用可阅读的PyTorch实现；不在产品内部调用`nvngx_dlss.dll`、`nvngx_dlssnr.dll`或closed RenoDX/DFC来完成缺失阶段。文件/视频编解码可用开源库，不要求用Torch重写PNG、EXR或视频编码器。

“功能等效”指具备Feeder生态相应类别的图像处理能力，**不承诺DLSS、RenoDX、DFC的逐像素或算法等价**。对实际选择并移植的FSR/Feeder算法，则应保留其算法与数据语义，不能用插值、任意混合或fitted gain占位。

### 1.2 已确认与未确认

| 项目 | 状态 |
|---|---|
|NR核心继续复用，不重做学习图|已确认方向|
|允许开源时域算法替代Feature 1，不再逆向DLSS为必要前置|已确认方向|
|排除游戏集成、注入、Frame Generation与跨API transport|已确认范围|
|用户已批准实施；公开源码可经127.0.0.1:7890 clone到C:/work，仍禁SSH和主动启WebUI|当前授权与约束|
|FSR2 v2.2.1作为首选基准|本计划建议，待采纳|
|RAFT-small、视频深度Small作为RGB-only候选|本计划建议，待质量与许可检查|
|第一版先以离线正确性和有界内存为目标|本计划建议；尚未承诺实时帧率|
|HDR支持范围、发布/商用、目标最大分辨率、多pass上限|待明确，不默认已决定|

### 1.3 非目标

- 不继续追求Feature 1原始网络、DLSS E/F/J/K各preset等价。
- 不把NR核心数值差异当成外围调gain的理由。
- 不复刻closed consumer的每个私有实现细节；需要某项功能时优先采用明确的白箱设计，并标出差异。
- 不宣称能够从单张RGB恢复真实几何、真实运动或未知HDR场景辐亮度。
- 不先承诺4K/8K实时；现有NR本机在约512级神经尺寸是秒级，而不是毫秒级。

### 1.4 怎样才算完成

完整输入路径必须真实经过所选算法与现有NR模型，输出保持声明的显示尺寸、颜色域和时间顺序。连续处理具有独立且正确的history、reset、错误行为；发布包不依赖开发目录或原NGX/consumer DLL。各子算法有独立参考，完整序列有画质、时域和资源数据。

只出现一个FSR类、几个guide helper、有限输出或单帧截图，不代表该目标完成。

## 2. 调研后最重要的结论

1. **源码基础足够，但不是全部已有即插即用Torch包。** FSR2主体是HLSL/GLSL/宿主C++，需要完整移植；本次没有找到可确认直接复用的完整PyTorch FSR2实现。
2. **FSR是为渲染输入设计的，不天然等于普通视频增强器。** jitter、depth、MV、透明度mask的质量决定时域效果。
3. **RGB-only可以做，但必须明确是估计guide模式。** 单目深度的尺度、相机模型与时序稳定性可能比网络能否运行更重要。
4. **颜色与state接口决定系统能否组合。** FSR history、NR history、guide history不能混用；FSR处理域也不能直接等同NR已验证SDR输入域。
5. **白箱、开源和可再分发是三件不同的事。** 外围MIT/BSD/Apache许可不会自动改变现有NR权重的许可状态。

## 3. 组件候选、版本与许可

### 3.1 时域重建

| 候选 | 已确认事实 | 适用性与建议 |
|---|---|---|
|**FSR2 v2.2.1**|独立GPUOpen仓库、MIT许可证、完整shader和宿主实现；release tag `v2.2.1`，页面显示commit前缀`1680d1e`。[S1–S4]|首选基准：范围集中、无需训练权重、便于逐阶段参考。不是“最新FSR”，也不是已完成Torch端移植。实施前解析完整commit及文件hash。|
|**FSR3.1.3 Upscaler**|FidelityFX SDK `v1.1.3`，release显示`54fbaaf`；已读到MIT许可的实际upscaler accumulate源码。[S5]|有质量改进，可作为候补。只考虑upscaler，不带Frame Generation；不通过预编译AMD DLL冒充Torch实现。若早期比较显示明显收益，可在大规模移植前决定改选，而不是两套都先实现。|
|FSR1 EASU/RCAS|本地Feeder已有完整派生HLSL及MIT说明。[L2]|适合空间恢复和可选锐化。不是时域重建，也不能替代FSR2历史处理。|
|普通插值/TAA混合|可公开实现|只作为明确的单帧/对照路径，不能用来填充尚未移植的FSR阶段。|

GPUOpen在线手册当前页标题为FSR2.3.3，而独立FSR2仓库的固定release为2.2.1。**不得把不同版本的文档、shader、资源格式和常数混成一个实现。** 本计划用2.2.1的tag/API作为建议冻结点；在线手册用于理解，精确执行以冻结源码为准。

FSR2算法至少包括：

- luminance pyramid与可选曝光；
- previous-depth重建、depth/MV dilation；
- depth clip/disocclusion；
- lock创建与维护；
- history重投影、上采样、rectification、权重与累积；
- 可选RCAS；
- 可选reactive/composition自动生成及其所需输入。

“六个阶段”的概览不等于六个随便实现的helper。实际dispatch顺序、所有持久资源、清空与读写角色、LUT、Half/UNORM格式边界须从同一tag的host代码和shader恢复。

### 3.2 光流

**建议首个可选估计器：Torchvision RAFT-small。**

- 原始RAFT代码为BSD-3-Clause；有预训练模型和标准PyTorch实现。[S6]
- Torchvision已有`raft_small`/`raft_large`，不需要另发明网络。[S7]
- 现有Torch2.5环境应选匹配的Torchvision0.20系列，而不是盲装最新版本；建议先核对2.5.1/0.20.1 wheel组合。[S8]
- 候选权重明确记为`Raft_Small_Weights.C_T_V2`，不要用随未来版本变化的`DEFAULT`作为模型身份。该模型约0.99M参数、权重文件约3.8MB；权重来源和再分发条款仍要单独记录。[S7]
- 输入RGB按官方变换归一化到[-1,1]，几何尺寸满足8倍数及相关金字塔的最小尺寸要求。
- 输出是**第一张到第二张的像素位移**。本管线要current→previous时，应明确使用该方向的调用与验证，不能事后靠随意取负号修正。[S9]

限制：RAFT-small参数少，不代表全分辨率correlation便宜。all-pairs的主要内存按`(H*W/64)^2`增长；4K直接计算并不适合本机12GB。规划中必须有独立的flow工作尺寸，并在恢复flow分辨率时同步缩放X/Y位移。

所有预训练权重先作为本地研究候选。源码许可、checkpoint说明及训练数据条款分别记录；不根据BSD代码许可证直接承诺所有权重可任意商用/再分发。

### 3.3 深度

**优先级：外部准确guide > 明确估计guide > 无guide时拒绝相应严格模式。**

| 来源 | 已查信息 | 用途与边界 |
|---|---|---|
|外部device depth或metric view-Z|无需增加学习模型，但必须有depth convention/投影或转换参数|严格guided路径的首选；不是任意灰度图都可当深度|
|Depth Anything V2 Small|24.8M参数，相对深度；Small为Apache-2.0，Base/Large/Giant为CC-BY-NC-4.0。[S10]|单图/逐帧诊断候选；不能承诺时序一致或真实尺度|
|**Video Depth Anything Small**|28.4M参数；项目提供视频与逐帧streaming入口、relative/metric模型；Small声明Apache-2.0，Base/Large为CC-BY-NC-4.0。[S11–S12]|视频估计候选优先评估。应显式选择Small，避免上游脚本默认Large；streaming缓存、reset、延迟及实际checkpoint仍需核实|

关键问题不是“能不能得到一张depth图”，而是能否转换为FSR使用的depth语义。

FSR2 API需要camera near/far、vertical FOV、view-space到米的scale等参数。[S4]
相对单目深度不能通过每帧min-max归一化就变成可信device depth；这种做法会改变每帧的遮挡阈值和几何比例。

估计模式必须声明：

- 它是relative、inverse-relative，还是估计metric depth；
- 固定或可追踪的尺度/偏移与相机假设；
- 是否使用未来帧，多少延迟；
- 相机/场景切换如何重置；
- 其遮挡判断是启发式，不是等同真实渲染guide。

若某种估计方式不能提供稳定FSR输入，不得偷偷填常量深度并继续称为完整深度辅助FSR。应更换明确的估计/校准策略，或者提供另一个明确命名的模式并披露能力降低。

### 3.4 Feeder自有guide处理和输出算法

本地 `DLSS5-Feeder` 为MIT；FSR1派生部分另保留AMD声明。[L1–L2]

直接移植范围沿用已有审计：

- provider采样与单位转换；
- static/luma/depth/MV一致性分工及软mask；
- guide history；
- 920点、9项basis、两轮求解的可选geometry；
- 工作比例、偶数尺寸、同步采样与Halton；
- EASU、RCAS和调试图。

它们不负责从RGB产生真正的光流/深度，也不实现FSR2。Feeder的`BiasCurrentColorMask`不能直接改名为FSR reactive mask。

### 3.5 颜色、I/O及许可建议

- 颜色数学在Torch侧明确实现；sRGB、PQ、BT.709/BT.2020及亮度单位都进入输入描述。OpenColorIO可作独立CPU对照，项目为BSD-3-Clause，并须留意第三方与config许可。[S13]
- 初始I/O以无损图片序列和独立guide文件为参考；视频容器随后接入。否则codec压缩、limited/full range、YUV矩阵错误可能被误判成算法误差。
- FFmpeg可作开源视频I/O候选，但默认主体LGPL、启用特定组件可变为GPL；例如libx264/libx265不能假设仍是同一许可。[S14]
- RAFT/深度/NR权重保持外置，有来源、hash、许可文件和可选性。推理过程中不自动下载未知checkpoint。
- 当前NR实现可读，不代表原始NR权重获得MIT/Apache再许可。若目标是公开分发或商业使用，这项必须单独解决；本计划不把法律结论伪装成技术完成。

## 4. 输入模式：明确区分，不靠假数据补空缺

### 模式A：Guided Sequence

输入真实连续Color、depth、current→previous MV、时间戳、jitter/相机信息，以及可用的reactive/composition/保护mask。

用途：验证算法、处理具备guide的数据、建立可靠参考。所有guide有单位/方向/分辨率/active rect。缺必需字段时返回明确错误，或由用户显式选择估计模式。

### 模式B：RGB Estimated Sequence

仅给RGB序列时，由选定的开源光流/视频深度模型估计guides。输出附带`estimated_guides=true`、模型版本、depth假设、工作尺寸和reset信息。

用途：普通视频便利入口。质量、时间一致性和相机假设单独验收，不把它与模式A的参考误差混为一谈。

### 模式C：Single Frame

没有前帧时不存在真实时域信息。明确采用选定的空间处理与NR路径，不伪造一段重复history宣称已做时域超分。

是否执行空间EASU、目标尺寸与NR工作尺寸必须显式。可复用同一管线公共颜色和输出模块，但这不是FSR2全时域结果。

### 模式D：Diagnostic / Bypass

允许明确观察：原图、仅空间恢复、仅FSR、仅NR、完整链、各类guide和mask。
`bypass`必须标清旁路的是哪一段；不得用内部自动旁路让本应失败的FSR/NR看起来成功。

## 5. jitter与普通视频：必须先回答的实验问题

FSR2期望Color/depth等渲染分辨率输入对应已实际施加的jitter；MV通常应不含jitter，除非选择明确的取消机制。[S2–S4]

不能只给已拍摄视频填一串Halton偏移，就说视频现在具有那些采样信息。

建议比较三个明确模式：

1. **Provided jitter**：数据实际按该jitter生成，使用给定值；参考路径。
2. **Unjittered existing video**：已有视频无渲染jitter，默认声明0；其时域/1:1效果必须实际评估。
3. **Synthetic work sampling**：当源图比work尺寸大时，按真实偏移重新采样Color和guides，再把同一偏移交给时域算法。这类似Feeder的性能旋钮，但不会创造超出源图的信息。

FSR2固定quality枚举是1.5/1.7/2/3倍；1:1应按自定义尺寸路径核对，不凭空叫作DLAA，也不能仅用更新版本手册推断旧tag全支持。[S3–S4]

在大规模移植前，用完整官方参考链对20–60帧的两个输入集回答：

- 在我们的已有视频输入上，零jitter或synthetic work哪种才有合理效果？
- FSR2.2.1与候补FSR3.1.x是否有足以改变移植选择的差距？
- 估计depth能否通过稳定性要求？若不能，如何界定RGB-only模式？

这是会改变架构的有限实验，不是无止境逐算子微调。本轮只列实验，不运行。

## 6. 拟定的整体架构

```text
InputDecoder / SequenceReader
    │   颜色标签、alpha、时间戳、frame id
    ▼
FrameGeometry + ColorNormalization
    ├────────────── 原始输入/颜色context留存 ──────────────────────┐
    ▼                                                            │
GuideSource                                                       │
    ├─ provided depth/flow                                       │
    └─ RAFT / video depth adapter（显式estimated）                  │
    ▼                                                            │
GuideProcessor（Feeder逻辑、验证、geometry、独立guide history）       │
    ▼                                                            │
WorkSampler（同一采样网格、MV尺度、真实jitter）                       │
    ▼                                                            │
TemporalReconstructor（完整FSR，独立history与曝光状态）               │
    ▼                                                            │
NRColorEncoder / NRGeometryPlanner                                 │
    ▼                                                            │
NRChain                                                          │
    ├─ Pass0: packet → frozen core → NR域合成 → own history        │
    ├─ Pass1..N: 同一guide，独立参数与history                       │
    └─ 不将NR结果倒灌FSR history                                  │
    ▼                                                            │
NRColorDecoder / ImageProtection ◄────────────────────────────────┘
    ▼
OutputResize / OptionalSharpen / OptionalDetailGuard
    ▼
ImageSequenceWriter / VideoEncoder / Diagnostics
```

源input、temporal工作/输出、NR工作、NR内部buffer以及估计器工作尺寸是不同层级。
一个全局`resize()`不能管理全部坐标域。

### 建议模块划分

```text
whitebox_pipeline/
  pipeline.py          统一process_frame/process_sequence及state提交
  contracts.py         带单位、颜色域、尺寸与来源的数据定义
  guides.py            Feeder可读guide算法
  estimators/          光流/深度输入适配，不隐藏权重下载
  temporal/            FSR完整移植及资源/state
  color.py             编码/解码/颜色context
  nr_adapter.py        接现有dlss5_model，不改核心
  output.py            空间恢复、锐化、保护
  io.py                图片/序列读写
```

这是维护组织，不要求按文件分批交付。每次交付以实际输入到输出能力为单位。

## 7. 统一接口和数据语义

以下为拟定schema，实施时形成可检查的类型，不是已存在API。

### 7.1 FrameInput

| 字段 | 语义 |
|---|---|
|`color`|推荐计算侧BCHW float32；显式RGB/alpha及premultiplied状态|
|`color_encoding`|transfer、primaries、range、scene/display-referred、亮度单位|
|`frame_id`, `pts`, `delta_ms`|单调帧身份和时间；FSR使用毫秒，不把秒直接传进去|
|`source_size`, `active_rect`, `target_size`|真实可见像素与目标显示尺寸|
|`motion`|约定current→previous、XY、像素单位；附所属网格尺寸与是否含jitter|
|`depth`|device depth或metric view-Z，附near/far/FOV、normal/reversed/infinite配置|
|`guide_provenance`|provided/estimated、模型与版本、相机假设、可用性|
|`jitter`|实际采样偏移，明确所在像素网格|
|`reactive_mask`, `composition_mask`|FSR专用语义，不与Feeder或NR mask混用|
|`nr_control_mask`, `protection_mask`|NR合成及用户保护区域，语义独立|
|`reset`, `settings_generation`|显式reset请求及配置代际|

B=1作为首个序列state实现范围。批量独立序列须有独立state；不能把不同视频放进一个共享history。

### 7.2 尺寸与坐标规则

- 像素中心采用`(x+.5)/W,(y+.5)/H`，XY顺序统一。
- flow缩放到新尺寸时，同时乘X/Y比例；改变sampling UV与改变位移单位是不同动作。
- 光流估计器的8对齐、depth模型的patch对齐、FSR render/display尺寸、NR64级buffer约束分别处理。
- NR内部补边复用已恢复的一次反射后clamp，不改成无限reflect/edge；输出裁回实际NR工作域后再走颜色恢复。
- 各shader的point/linear、clamp/zero border、half-texel与load坐标单独对应。不能把现有`bilinear_texture`默认zero边界无条件用于FSR/Feeder的clamp采样。
- 分辨率不支持时，明确错误或显式选择较低工作尺寸；不得偷偷改变用户目标尺寸。
- Feeder检查使用的归一化linear depth与FSR device depth、metric view-Z是不同表示，不能把“米”直接套进Feeder的`.999`天空阈值。若采用原ReShade规则，须取得确切`ReShade.fxh`版本；本地Feeder仓库未附带该文件。
- Feeder Luma/PatchError应声明所采样的颜色域。SDR参考用原FX输入域，不把图像先线性化后原封不动沿用其阈值却宣称等价。

### 7.3 masks必须分开

| Mask/信号 | 用途 |
|---|---|
|Feeder distrust|光流/重投影的验证结果|
|FSR reactive|减少该像素对history的依赖|
|FSR transparency/composition|控制相应累积/lock行为|
|NR Auto/Character条件|已恢复NR输入/对应宿主策略|
|NR ControlMask|NR颜色合成的强度控制|
|用户保护mask/alpha|保持指定内容或合成语义|

FSR自动reactive/composition生成需要opaque-only Color等信息。[S2–S4] 普通RGB视频没有这个输入，不能默认开启官方autogen再填一张假的opaque图。

可接受：缺省mask按所选源码的明确默认行为处理；或增加有名称、有公式、有测试的估计mask策略。不能将Feeder distrust直接改名后声称是官方FSR mask。

### 7.4 输出和可观察性

`FrameResult`包含显示尺寸Color、颜色标签、各阶段有效设置、guide来源、reset原因、耗时/峰值内存和可选诊断图。长期history不作为无边界日志保留。

建议接口：

```python
result, next_state = pipeline.process_frame(frame, state, settings)
# 或由明确拥有state的SequenceProcessor迭代
for result in processor.process_sequence(frames):
    writer.write(result)
```

相同输入、版本、设置、初始state应可重放；不要让未声明的全局RNG或全局history决定结果。

## 8. 时域状态和reset规范

至少四套独立状态：

1. **Estimator state**：视频深度缓存、可能的flow缓存；RAFT基本双帧输入本身也需明确上帧是哪一帧。
2. **Guide state**：输入luma、linear depth、raw provider flow；沿用Feeder语义。
3. **FSR state**：history color、reconstructed/dilated depth与MV、locks、luminance/exposure、mask相关持久资源；以固定tag清单为准。
4. **NR state数组**：每pass独立的NR颜色域历史、参数代际和有效性。

禁止：

- 把已NR增强的帧喂回FSR history；
- 用最终锐化/输出tone后的图覆盖NR内部history；
- 用validated/scaled MV覆盖guide所需的raw flow历史；
- 用一个history tensor服务所有pass或所有视频。

建议reset规则：

| 事件 | 建议行为 |
|---|---|
|首帧|state显式无效，使用所选算法的首帧规则，不随机填充|
|用户reset/明确scene cut|相关层全部reset；scene-cut detector是可选白箱策略，不冒称Feeder已有|
|source/target/NR尺寸或颜色域变化|按受影响状态完整重建；支持DRS时以冻结FSR源码规则处理|
|MV方向/单位、depth convention、jitter策略或估计器改变|重置相关estimator、guide、FSR、NR状态|
|某NR pass参数改变|先以该pass及下游reset为保守明确规则，再依据具体语义缩小|
|pass数变化|保留不受影响的上游，新增/受影响下游重新初始化|
|仅最终预览/锐化显示改变|不重置不相关上游history|
|取消、OOM或中途失败|不发布部分state；不能回滚的估计器cache标无效并reset，禁止继续使用混合代际|

处理状态应在一帧成功完成后统一提交。离线任务中断后可从明确检查点/重置边界重放，不靠重用半帧缓存恢复。

## 9. 颜色与NR衔接

### 9.1 SDR首版建议

```text
输入编码RGB
→ 依据标签解码到线性工作域
→ FSR工作域处理
→ 编码到已验证NR输入域
→ FP16 RTZ Color / packet / frozen NR
→ NR域合成
→ 颜色解码与目标输出编码
```

NR默认从已验证SDR准备方式起步，参数profile明确记录Natural、Structure2、Tone1、Skin-1、Auto Mask关闭等；这只是首个测试profile，不限制最终只能支持它。

不同公开算法的输入域不同，不能把RAFT的[-1,1]、FSR线性Color、NR的编码Color当成同一tensor复用。guide验证、估计器、FSR、NR分别持有明确的颜色视图；HDR供光流/深度估计器使用的SDR映射也必须显式定义，不能将高光任意clip后忽略其对guide的影响。

### 9.2 HDR计划

HDR不靠`clamp(0,1)`实现。实施前冻结：

- SDR/sRGB、scene-linear BT.709、PQ/BT.2020各自的输入与输出；
- reference white/nits、exposure/pre-exposure和有效范围；
- NR编码压缩与解码、超范围/负值/高光策略；
- 原始颜色context如何保留；
- zero-effect/identity是否回到原始context，而不是被codec额外改色；
- alpha与premultiplication，用户保护区域如何合成。

建议以标准颜色转换加明确的自有NR颜色桥实现功能，不默认逆向closed consumer全部codec。若需要某个RenoDX/DFC特有效果，再单独锁定版本恢复；不把概念相似的实现称作相同算法。

没有原始scene-linear信息的LDR视频，不承诺恢复真实HDR或反演未知tone mapper。

### 9.3 多pass、锐化与反馈

NR多pass共享guide，但各自拥有NR历史和参数；颜色编码/解码原则上只包住整条NR链，不每pass重复tone转换。

默认只选择一个明确的最终锐化位置。不要同时打开FSR RCAS、Feeder RCAS和额外unsharp后又把过锐伪影归因NR。

若提供detail-only/halo guard等功能，应定义自己的数学与名字；不根据DFC文档标题就声称复刻Native Look/Clean Fry。

## 10. PyTorch移植策略

### 10.1 保留什么

- 固定源码版本下的完整数据流和state更新。
- 各stage颜色域、depth方向、MV/jitter/subrect语义。
- 必要的Half/UNORM/整数重解释及资源发布边界。
- LUT、重建核、锁定/拒绝策略、动态尺寸/首帧/reset逻辑。

### 10.2 允许怎样改变执行方式

- shader线程组换成批量tensor操作。
- 纹理采样换成明确语义的gather/grid_sample。
- depth重建中的原子min/max换成等价scatter-reduce，保留normal/inverted规则。
- wave级归约可以用数学等价的Torch实现；对影响阈值/量化的数值边界单独验证。
- 先FP32可审计实现，再依据source格式保留必要量化；不要求原硬件指令逐bit，但不能删掉算法阶段。

禁止用预编译FSR DLL或GPU shader运行库作为最终产品的“PyTorch backend”。官方shader只作为本地离线reference；最终功能不能依赖它。

`torch.compile`可在正确性稳定后考虑。若性能最终需要额外自定义kernel，应保留完整源码并另作明确决定，不在本计划中默认获得该实现授权。

### 10.3 上游来源冻结

实施开始时为每个依赖记录：URL、完整commit/tag、源码hash、license、checkpoint URL/hash、预处理和版本兼容。
本轮仅核实公开资料，不把还未下载校验的checkpoint写成已有本地资产。

## 11. 分阶段交付：每阶段都产出可运行结果

### P0：选型锁定与关键不确定性消除

**目的**：决定FSR版本、输入模式和颜色/state边界，避免移植中途重选整套算法。

产出：

- dependency/source lock与许可证清单；
- Frame/State/Color/Geometry规范；
- 固定FSR版本的本地离线参考入口；
- 两组20–60帧实验：正确guide/jitter与普通视频估计guide；
- FSR2/候补FSR3.1、zero/provided/synthetic jitter、depth适配的明确取舍；
- 小规模本机峰值内存与耗时，用来确定可承诺分辨率，而非借用AMD shader宣传数字。

通过条件：能说明选定算法在我们的输入条件下如何工作，关键假设都有可观察结果。必要资料缺失或候选失败时，明确调整方案，不把问题埋到最终集成。

**首次计划编制时尚未执行。当前已完成合成序列官方参考和NR衔接，详见文首执行记录；真实视频、估计深度及候补FSR3对比仍未完成。**

### P1：Guided SDR完整序列管线

输入：有明确depth/MV/jitter的SDR序列。
输出：经完整开源时域阶段、NR单pass和颜色恢复的显示尺寸图像序列。

一轮组合完成：Feeder guide处理、work sampler、完整FSR移植、state/reset、SDR NR bridge、frozen core衔接、必要输出恢复、基础CLI/序列I/O。

通过条件：真实入口处理连续帧；所有选定FSR stage参与；逐stage对官方reference、最终对组合reference；尺寸/颜色/时间正确；无原NGX/consumer DLL依赖。没有交付几个helper后称为“FSR已完成”。

### P2：RGB-only可用入口

在P1接口上接入所选RAFT/视频深度适配，形成直接读RGB序列、明确标注estimated guides的结果。

同时实现估计器工作尺寸、位移恢复、depth尺度/相机假设、首帧/cache reset、失败行为及诊断。

通过条件：真实短视频无未来信息泄漏（若选择因果模式），能解释ghosting/闪烁来源；估计模式与guided模式分别报告，不用正确guide的指标为RGB-only背书。

### P3：完整颜色/state与功能覆盖扩展

在同一端到端入口扩展：

- 声明支持的HDR/格式与alpha/保护路径；
- NR参数、多pass与下游reset；
- Feeder实验性geometry和全部guide诊断选项；
- 工作比例/NR工作尺寸、不同target和动态尺寸；
- EASU/RCAS/输出detail guard的明确组合；
- 场景切换、任务取消、OOM后的正确状态。

这些是关联的完整图像能力，不按文件或角色机械切里程碑。P2与P3在接口冻结后可分开实现；本机GPU参考测试应串行安排，避免多进程同时占用导致错误归因。

### P4：独立可用的应用与分发

- 批量图片/视频或帧序列CLI；
- WebUI显示加载、任务进度、分辨率、guide来源、耗时、有效设置，支持取消和显式reset；
- 流式处理，不把全视频解码或所有中间tensor一直驻留内存；
- 模型权重外置、依赖清单和许可证、运行manifest；
- 从临时解压目录在隔离开发路径的环境执行完整序列并输出可读取结果；
- 完整用户说明：不是DLSS等价，哪些模式用估计guide，已测分辨率/颜色域/硬件范围。

完成后才称“独立白箱图像处理管线已可用”。本轮不启动/重启WebUI；未来部署动作仍按用户授权执行。

### 扩展研究项，不混进首版完成声明

- Texture Boost式4K→8K中间处理或任意NR supersampling。
- 高pass数量压力场景。
- 更大光流/深度模型、更广HDR/宽色域和实时优化。

这些在完整计划中保留，但在资源/质量未验证前不承诺首版支持。NR含全局ViT，简单切tile会改变图的全局交互；不能把tile化结果当成原核心等价执行。

## 12. 验证体系

### 12.1 三层参考，职责不同

| 参考 | 用途 | 不证明什么 |
|---|---|---|
|固定版本FSR/Feeder shader离线执行|验证Torch移植的算法和资源语义|不证明与DLSS相同|
|现有NR单文件及已保存证据|确保接入未改学习图、权重和输入准备|不证明新的HDR/history适配全部正确|
|完整真实序列与独立gt/观测|判断时域稳定、实际画质、资源和可用性|没有GT时不伪造PSNR/绝对质量结论|

允许本地reference程序用于验证，不要求产品依赖该程序。验证过程不涉及游戏或SSH；本轮不编译/运行这些reference。

### 12.2 必要场景

- 静止纹理/细线、亮度闪烁、平坦移动表面；
- 正确方向与反方向MV、亚像素平移、快速移动、独立运动前景；
- disocclusion、屏幕边界、天空、normal/reversed/infinite depth；
- 透明/粒子类区域，提供mask和无mask情况分开；
- 1/8 flow升采样、奇数尺寸、多种比例、带active rect；
- SDR/BGRA/10bit/FP16/PQ及alpha，codec zero-effect；
- 首帧、硬切、帧跳过、variable delta_ms、reset、尺寸/参数/pass变化；
- 长序列内存、取消、异常后重放与并行独立任务隔离。

先用可解析ground truth的合成变换定位坐标/深度错误，再用授权的真实视频验收。不能只在合成平移或单张图片上宣布整条链完成。

### 12.3 指标与门槛的制定

- Tensor：同域误差、max/MAE、NaN/Inf、关键离散决策的一致性与差异位置。
- 图像：有GT时PSNR/SSIM等；无GT用有标签的对照，不把锐化后的分数提升当普遍画质改进。
- 时域：遮挡区域排除后的重投影误差、静止区闪烁、ghost残留、细节保持。用gt或独立flow评估，避免用同一估计器自证正确。
- State：reset后等价于新state、两序列不串、失败不污染下一帧、配置代际正确。
- 资源：cold load、warm处理P50/P95、峰值显存、主机内存、长序列增长、吞吐。

具体数值门槛在P0参考数据上冻结。无需硬编码一个尚无基准的“相关性>.99”；不得测试后调低门槛来隐藏失败。算法语义已证、仅标准算术微差时按误差归因处理，不追ISA内部逐bit。

## 13. 性能、内存与实施风险

本机已知NR测试在约512级neural尺寸需要约2–5秒/图，加载约一分钟。完整新管线还要加FSR及估计器，**不能据上游FSR的原生shader毫秒表承诺全Torch实时**。

| 风险 | 应对 |
|---|---|
|普通视频没有真实jitter/depth/mask|先固定输入模式；P0比较，不伪造元数据|
|单目depth漂移破坏FSR disocclusion|时序模型、固定尺度/相机假设、独立指标；必要时调整RGB-only方案|
|FSR手工Torch移植范围大|冻结版本，完整stage/state inventory，参考捕获覆盖；不同时追多个版本|
|双重时域反馈造成ghosting|FSR与各NR history隔离，逐层旁路诊断，明确reset|
|色域/曝光重复处理|全链颜色标签和原始context；zero-effect/round-trip测试|
|RAFT correlation或NR全局attention爆显存|独立工作尺寸、生命周期管理、容量测量；不默默tiling改变算法|
|多pass线性增时/增state|按预算明确上限，取消/失败事务化；不复用同一个history|
|源码可读但权重许可不清|外置权重与manifest；公开发布前单独核实|
|新依赖破坏现有CUDA环境|新环境或锁定匹配版本；保留现有模型验证环境和回退点|

建议先离线正确、再缩短处理时间。性能调整必须从实际入口测量，不能只报告局部kernel计数。没有明确目标分辨率/FPS前不写确定工期或实时承诺。

## 14. 需要设计师确认的少量问题

这份计划可以作为后续实施依据，但下面事项还未定：

1. **首要输入**：普通RGB视频是第一优先级，还是可以先用带depth/MV的数据建立可靠管线？建议二者都支持，但先验证guided基准。
2. **性能目标**：先接受离线秒级，还是从第一版就有明确分辨率/FPS要求？后者会显著影响算法和工作尺寸。
3. **颜色范围**：首版SDR先可用，再补HDR，是否符合优先级？完整目标中的HDR并未因此删除。
4. **用途/分发**：仅本地研究还是计划公开/商业分发？这决定权重与媒体依赖许可门槛。

若暂不决定，本计划只保留建议默认，不把“暂时想法”写成已获授权的实现规格。具体checkpoint、FSR版本与高级效果不需要现在逐个问；可在P0形成依据后一次确认。

## 15. 本轮交付与后续维护

最初的调研轮只交付计划；用户随后已批准实施及经指定HTTP代理获取公开源码。当前仍不修改已验证NR模型或旧ZIP，不主动启动WebUI；执行进展在文首链接单独记录。

实施时维护一份精简状态：当前目标、已定接口、源码/权重版本、实际可用入口、阻塞和对应证据。保留当前单文件模型与分发包作为回退资产。

不按每个算子刷新大计划、不默认独立review、不用重复全模型测试代替外围进展。只有新语义、新失败或新相关风险才扩大验证。

## 16. 来源与证据索引

公开资料查阅时间：2026-09-06。移动分支/在线手册只作为调研证据；实施依赖需冻结完整commit与hash。

- **[L1]** 本地 `C:/work/DLSS5-Feeder`，commit `03da7d9ca36c92f462ed936f5a629116b6b8ee5b`；`LICENSE`为MIT，`shaders/DLSS5_Feed.fx`、`src/dlss5-feed.cpp`。具体语义及行号见 `FEEDER_IMAGE_PIPELINE_GAPS.md`。
- **[L2]** 同仓库 `src/feed_fsr1.h`，AMD FSR1派生HLSL，包含MIT版权声明。
- **[S1]** FSR2固定版本与release：<https://github.com/GPUOpen-Effects/FidelityFX-FSR2/releases/tag/v2.2.1>
- **[S2]** FSR2 v2.2.1集成说明：<https://github.com/GPUOpen-Effects/FidelityFX-FSR2/tree/v2.2.1>
- **[S3]** FSR2公开手册：<https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/>。当前标题2.3.3，不能与旧tag无条件混用。
- **[S4]** FSR2 v2.2.1 API源码：<https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/v2.2.1/src/ffx-fsr2-api/ffx_fsr2.h>；shader目录 <https://github.com/GPUOpen-Effects/FidelityFX-FSR2/tree/master/src/ffx-fsr2-api/shaders>；许可 <https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/master/LICENSE.txt>。
- **[S5]** FSR3.1.3对应SDK release：<https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v1.1.3>；实际MIT shader：<https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.3/sdk/include/FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_accumulate.h>。
- **[S6]** RAFT原始实现：<https://github.com/princeton-vl/RAFT>。
- **[S7]** Torchvision0.20 RAFT-small与明确weights枚举：<https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.optical_flow.raft_small.html>。
- **[S8]** Torch/Torchvision版本匹配表：<https://github.com/pytorch/vision>。
- **[S9]** RAFT输入/输出与坐标说明：<https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html>；实施时以选定0.20源码预处理为准。
- **[S10]** Depth Anything V2模型及区别许可：<https://github.com/DepthAnything/Depth-Anything-V2>。
- **[S11]** Video Depth Anything模型及区别许可：<https://github.com/DepthAnything/Video-Depth-Anything>。
- **[S12]** Video Depth Anything逐帧入口：<https://github.com/DepthAnything/Video-Depth-Anything/blob/main/run_streaming.py>。已确认有逐帧调用，不据此省略内部cache/因果性审查。
- **[S13]** OpenColorIO与许可：<https://github.com/AcademySoftwareFoundation/OpenColorIO>。
- **[S14]** FFmpeg许可与可选GPL组件：<https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md>。

现有核心与分发依据：`DETAILED_MODEL_ARCHITECTURE.md`、`DLSS5_MODEL_API.md`、`README_DISTRIBUTION.md`、`debug-static/distribution-package-verification.json`。本轮没有重复运行已有效的核心测试。
