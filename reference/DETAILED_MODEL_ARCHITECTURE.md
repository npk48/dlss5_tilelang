# DLSS5 Reconstructed Model — Detailed Model Architecture

> 当前DLL可用的固定 Preset #1 权重配置；完整 block0–70 学习模型。
> 本文根据当前可执行实现、原始权重解码、SASS/地址推导和本机运行证据整理。
> 模型本体版本：`6f6b5cb`；完整自馈归因证据：`a44d8d4`。
> 后续WebUI与分发打包修改没有改变该学习模型或权重。

## 目录

1. 模型范围与阅读约定
2. 总体拓扑与71个block
3. 空间尺寸、布局与skip连接
4. 原始权重、矩阵规模与加载方式
5. 统一数值原语
6. block0输入适配与双输出
7. 1H/32局部注意力块
8. 2H/64局部注意力块
9. 4H/128局部注意力块
10. 8H/256局部注意力块
11. 16H/512分解式局部注意力块
12. 编码器下采样与block30 terminal
13. blocks31–38全局ViT
14. 解码器与所有上采样连接
15. block70学习输出头
16. 物理布局与动态尺寸
17. 最小API、外部输入和WebUI
18. 验证结论、数值差异与边界
19. 实现索引与常见误读

---

## 1. 模型范围与阅读约定

### 1.1 本文描述什么

这是从目标DLL实际提供的固定学习配置恢复出的模型，而不是根据网络名字
重新搭建的标准Swin、普通U-Net或近似ViT。它包含：

- 原始BIN中的真实权重及其物理存储解释；
- 71个顶层block，覆盖输入adapter、编码器、bottleneck、解码器和学习head；
- 各族不同的FFN结构、真实attention头数、32维head和相应量化边界；
- 真实tensor布局、窗口相位、compact/repack与跨层skip连接；
- FP16/E4M3舍入边界、分段累加、归约树和近似激活/指数公式。

本文不将DLL宿主的资源管理、全部预设选择策略、历史帧生命周期、HDR/UI覆盖、
所有ControlMask分支或完整颜色合成选项冒充为已经完备的学习模型部分。

当前DLL实际提供的学习权重配置只有Preset #1。Style等输入条件不会替换
blocks1–69的权重或图；它们可以改变输入packet，从而改变神经网络结果。
“固定权重”并不等于“所有输入条件下结果相同”。

### 1.2 记号

- `B`：batch大小；本机完整图像对照以B=1为主，不把它外推成所有batch已验证。
- `H,W`：神经buffer尺寸，不一定等于PNG显示尺寸。
- `C`：当前特征通道数。
- `N`：全局ViT的token数，等于其descriptor的高×宽。
- `h`：attention heads数量；本模型各head的维度均为32。
- `H16(x)`：对x执行一次FP16舍入。
- `Q8(x)`：先FP16舍入，再按有限E4M3值域量化。
- `M(A,W;C0)`：具有明确FP16累加边界、可带初始残差C0的矩阵乘法。
- `R(x)`：已恢复的索引/通道/窗口重排；R不是一个可学习层。
- `G(x)`：第5节定义的实际激活函数，不是普通GELU或精确SiLU。

文中的矩阵形状统一使用逻辑 `K×N` 约定，即 `[...,K] @ [K,N]`。
源码有时以转置形式存储buffer，调用时再`.T`；这不是架构矛盾。

### 1.3 “HWC”不是普通图片坐标的保证

API以BCHW输入16通道packet，以BHWC输出head4。中间实现常用BHWC形状作为
可操作视图，但其中token、channel、quarter-plane、lane和矩阵M/N坐标
可能已经按物理地址重排。**不能看到BHWC就直接用普通图像roll/pool/reshape替换。**

以下表格同时给出逻辑层次和当前实现的代表性tensor尺寸，并明确何处是物理视图。

---

## 2. 总体拓扑与71个block

### 2.1 总览

整体具有多尺度编码器/解码器和长skip连接，中间串联局部16H块与全局ViT。
同一局部块中是 **FFN在前、attention在后**，不是默认Transformer的另一种排列。

```text
prepared [B,16,H,W]
    │
    ▼
block0: 16→32 FP16 adapter → 1H physical body
    ├──────────────────────────── skip0 [full H,W,32] ──────────────────────────┐
    └─ avg2x2 → main32                                                        │
         │                                                                    │
blocks1–4: 1H/32; block4末端avg2x2 + 32→64                                    │
    ├──────────────────── skip4 ────────────────────────────┐                 │
    ▼                                                       │                 │
blocks5–8: 2H/64; block8末端avg2x2 + 64→128                  │                 │
    ├──────────────── skip8 ────────────────────┐           │                 │
    ▼                                           │           │                 │
blocks9–14: 4H/128; block14末端avg2x2 + 128→256   │           │                 │
    ├──────────── skip14 ──────────────┐         │           │                 │
    ▼                                 │         │           │                 │
blocks15–22: 8H/256; block22末端avg2x2 + 256→512 │           │                 │
    ├──────── skip22 ────────┐         │         │           │                 │
    ▼                       │         │         │           │                 │
blocks23–30: split16H/512    │         │         │           │                 │
    ├──── skip30 ────┐      │         │         │           │                 │
    └─ block30 avg2x2 + 512→1024       │         │           │                 │
         │            │      │         │         │           │                 │
     repack58/首ViT输入route │         │         │           │                 │
         ▼            │      │         │         │           │                 │
blocks31–38: 8个global ViT/1024, 32 heads×32     │           │                 │
         │            │      │         │         │           │                 │
     repack99/consumer坐标桥 │         │         │           │                 │
         ▼            ▼      │         │         │           │                 │
block39: 2×主分支 + 1024→512 + gated skip30      │           │                 │
         ▼                   │         │         │           │                 │
blocks40–47: split16H/512     │         │         │           │                 │
         ▼                   ▼         │         │           │                 │
block48: 512→256 + gated skip22 + 8H body        │           │                 │
blocks49–55: 8H/256                     │         │           │                 │
         ▼                             ▼         │           │                 │
block56: 256→128 + gated skip14 + 4H body        │           │                 │
blocks57–61: 4H/128                               │           │                 │
         ▼                                       ▼           │                 │
block62: 128→64 + gated skip8 + 2H body                        │                 │
blocks63–65: decoder2H/64                                     │                 │
         ▼                                                   ▼                 │
block66: 64→32 + gated skip4 + 1H body                                          │
blocks67–69: 1H/32                                                             │
         ▼                                                                    ▼
block70: 主分支真实像素2×复制 + main/skip双门 → 1H body → FP16 32→4
         │
         ▼
head4 [B,H,W,4] = RGB residual + history logit
```

`repack58/99`是原生launch/布局操作的名字，不是额外的block58或block99。
本模型的block58是解码器4H学习块；两套编号不得混淆。

### 2.2 完整block表

下面使用第3节的空间级别S0–S6。`64`表示每个局部attention窗口的slot容量；
边界实际有效slot可能不足64，不能理解成把无效key全部屏蔽掉。

| Block | 当前角色 | 输入→输出通道 | 空间级别 | Attention | 长连接/备注 |
|---:|---|---:|---|---|---|
|0|输入adapter、1H body、双输出|16→32|S0→S1|1×32，64 slots|发布full-res skip0；main单独池化|
|1|1H body|32→32|S1|1×32，64|普通相位|
|2|1H body|32→32|S1|1×32，64|shifted物理窗口|
|3|1H body|32→32|S1|1×32，64|shifted物理窗口|
|4|1H body + downsample|32→64|S1→S2|1×32，64|发布skip4|
|5|首2H inpview body|64→64|S2|2×32，64|compact consumer与独立publication|
|6|2H body|64→64|S2|2×32，64|真实跨CTA窗口gather/scatter|
|7|2H body|64→64|S2|2×32，64|真实跨CTA窗口gather/scatter|
|8|2H body + downsample|64→128|S2→S3|2×32，64|发布skip8|
|9|4H body|128→128|S3|4×32，64|quarter-plane视图|
|10|4H body|128→128|S3|4×32，64|物理XY相位|
|11|4H body|128→128|S3|4×32，64|物理X相位|
|12|4H body|128→128|S3|4×32，64|物理Y相位|
|13|4H body|128→128|S3|4×32，64|普通相位|
|14|4H body + downsample|128→256|S3→S4|4×32，64|skip14；纵向Half pair池化|
|15|8H body|256→256|S4|8×32，64|8H物理轴与ragged布局|
|16|8H body|256→256|S4|8×32，64|物理窗口相位|
|17|8H body|256→256|S4|8×32，64|物理窗口相位|
|18|8H body|256→256|S4|8×32，64|物理窗口相位|
|19|8H body|256→256|S4|8×32，64|物理窗口相位|
|20|8H body|256→256|S4|8×32，64|物理窗口相位|
|21|8H body|256→256|S4|8×32，64|物理窗口相位|
|22|8H body + downsample|256→512|S4→S5|8×32，64|skip22；padding slots参与compact布局|
|23|split16H body|512→512|S5|16×32，64|factorized FFN + local attention|
|24|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|25|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|26|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|27|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|28|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|29|split16H body|512→512|S5|16×32，64|同族不同窗口相位|
|30|split16H + terminal downsample|512→1024|S5→S6|16×32，64|skip30；保留projection Half供池化|
|31|global ViT|1024→1024|S6|32×32，N个keys|FFN1024→4096→1024|
|32|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|33|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|34|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|35|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|36|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|37|global ViT|1024→1024|S6|32×32，N|同构，不共享权重|
|38|global ViT|1024→1024|S6|32×32，N|末端publication供repack/decoder|
|39|decoder输入上采样|1024→512|S6→S5|无attention body|融合skip30；K256×4|
|40|decoder split16H|512→512|S5|16×32，64|decoder物理窗口视图|
|41|decoder split16H|512→512|S5|16×32，64|同族相位|
|42|decoder split16H|512→512|S5|16×32，64|同族相位|
|43|decoder split16H|512→512|S5|16×32，64|同族相位|
|44|decoder split16H|512→512|S5|16×32，64|同族相位|
|45|decoder split16H|512→512|S5|16×32，64|同族相位|
|46|decoder split16H|512→512|S5|16×32，64|同族相位|
|47|decoder split16H|512→512|S5|16×32，64|terminal publication为planar N16|
|48|mixed upsample + 8H|512→256|S5→S4|8×32，64|融合skip22|
|49|8H body|256→256|S4|8×32，64|decoder相位|
|50|8H body|256→256|S4|8×32，64|decoder相位|
|51|8H body|256→256|S4|8×32，64|decoder相位|
|52|8H body|256→256|S4|8×32，64|decoder相位|
|53|8H body|256→256|S4|8×32，64|decoder相位|
|54|8H body|256→256|S4|8×32，64|decoder相位|
|55|8H body|256→256|S4|8×32，64|decoder相位|
|56|mixed upsample + 4H|256→128|S4→S3|4×32，64|融合skip14；处理quarter-plane桥|
|57|4H body|128→128|S3|4×32，64|布局对应encoder12|
|58|4H body|128→128|S3|4×32，64|布局对应encoder13|
|59|4H body|128→128|S3|4×32，64|布局对应encoder14|
|60|4H body|128→128|S3|4×32，64|布局对应encoder11|
|61|4H body|128→128|S3|4×32，64|布局对应encoder12|
|62|mixed upsample + decoder2H|128→64|S3→S2|2×32，64|融合skip8|
|63|decoder2H body|64→64|S2|2×32，64|不能套encoder普通HWC窗口|
|64|decoder2H body|64→64|S2|2×32，64|decoder专用布局|
|65|decoder2H body|64→64|S2|2×32，64|decoder专用布局|
|66|mixed upsample + 1H|64→32|S2→S1|1×32，64|融合skip4，保留Half到首残差|
|67|1H body|32→32|S1|1×32，64|物理XY相位|
|68|1H body|32→32|S1|1×32，64|物理X相位|
|69|1H body|32→32|S1|1×32，64|物理Y相位|
|70|dual gate + 1H + readout|32/32→32→4|S1/S0→S0|1×32，64|融合skip0；输出真实像素head4|

因此71个block并非71个相同Transformer。按学习body统计：1H有10个，2H有8个，
4H有12个，8H有16个，split16H有16个，全局ViT有8个；另外block39只有上采样连接。

---

## 3. 空间尺寸、布局与skip连接

### 3.1 当前实现使用的级别

对本文已验证的64对齐神经buffer，设 `A4(x)=4*ceil(x/4)`：

| 级别 | 当前实现的tensor空间轴 | 通道 | H=512,W=640例子 |
|---|---|---:|---|
|S0|`(H,W)`|32，最终head为4|512×640|
|S1|`(H/2,W/2)`|32|256×320|
|S2|`(H/4,W/4)`|64|128×160|
|S3|`(H/8,W/8)`|128|64×80|
|S4|`(W/16,H/16)`的8H物理视图|256|40×32|
|S5|`(4*ceil(S4.h/8),4*ceil(S4.w/8))`|512|20×16|
|S6|`(A4(S5.h/2),A4(S5.w/2))`|1024|12×8，即N=96|

S4的轴交换是布局表达，不表示输出图像被旋转。S5/S6中对齐产生的slots是
实际descriptor的一部分，不能根据名义的1/32或1/64尺寸裁掉。

例如320×320神经buffer：

```text
S0 320×320×32
S1 160×160×32
S2  80×80×64
S3  40×40×128
S4  20×20×256
S5  12×12×512      不是10×10
S6   8×8×1024     N=64，不是16/32个token
```

对于512×576、384×640等尺寸，8H的ragged列和half-band也必须保持真实布局。
把输入强行扩大到512×640仅为绕过reshape错误，不是等价实现。

### 3.2 六条长skip

| 保存点 | 保存内容 | 消费点 | 合并方法 |
|---:|---|---:|---|
|block0|全分辨率32通道body的E4M3 publication|70|两路学习门，main真实像素2×复制|
|block4|下采样前32通道body的E4M3 publication|66|64→32 transition + sin门|
|block8|下采样前64通道body的E4M3 publication|62|128→64 transition + sin门|
|block14|下采样前128通道body的E4M3 publication|56|256→128 transition + sin门|
|block22|下采样前256通道body的E4M3 publication|48|512→256 transition + sin门|
|block30|末端512通道projection的E4M3 publication|39|1024→512 transition + sin门|

长skip与块内FFN/attention残差不是同一类连接。名为`sin`的权重是保存的学习
缩放向量，不代表此处执行三角函数。

---

## 4. 原始权重、矩阵规模与加载方式

### 4.1 153条record不等于153个普通线性层

`weights_ht_blob.bin` 总长147,695,410字节，包含153条命名记录。按当前使用关系：

| 范围 | 记录组织 | 数量 |
|---|---|---:|
|blocks0–22|各一个layer0；terminal中同时打包transition|23|
|blocks23–29|每块4个split阶段记录|28|
|block30|4个split阶段加terminal投影|5|
|blocks31–38|每块5个ViT记录|40|
|block39|投影及skip门记录|1|
|blocks40–47|每块4个split阶段记录|32|
|blocks48–69|每块一个普通或mixed记录|22|
|block70|layer0和独立blend_scale|2|
|合计||153|

记录中既有E4M3 packed矩阵，也有FP16标量/向量、FP32 scale以及布局/marker。
例如ViT attention的一个记录只有一个Half大小的marker，不能因为记录存在就
臆造一个学习矩阵。反过来，一个fused记录可包含多路FFN、QKV、bias和projection。

### 4.2 为什么不能对BIN直接reshape为FP16矩阵

若干历史容器把原始payload装进`<f2`数组，但这只是保持字节序列的载体；其中
多数矩阵实际是每元素一个字节的E4M3。正确加载步骤包括：

1. 验证原BIN framing、名称和payload边界，读取对应record。
2. 按K32平面、成对N8 fragment及head/stream地址公式提取字节。
3. 解码E4M3值，恢复逻辑K×N矩阵。
4. 单独读取正确偏移、dtype和顺序的FP16/FP32 gate、scale、bias。
5. 恢复hidden-N→consumer-K、output-N→canonical等路由。
6. 以固定buffer保存解码结果，供推理反复复用。

`dlss5_model.py`已经内置这些加载和路由资料，运行不依赖仓库中的辅助模块。
临时解码缓存是加载过程的内部实现，不是用户必须预先提供的模型文件。

### 4.3 逻辑矩阵规模

下表按已恢复的逻辑矩阵形状相乘，不把bias、gate、scale或存储padding算进来。
它用于理解模型容量；不是通过文件大小猜出的参数量，也不是框架
`model.parameters()`的计数——本固定推理实现主要使用`register_buffer`。

| Body族 | 每个body的矩阵系数数 | Body数量 |
|---|---:|---:|
|1H/32|12,288|10|
|2H/64|45,056|8|
|4H/128|163,840|12|
|8H/256|622,592|16|
|split16H/512|1,835,008|16|
|globalViT/1024|12,582,912|8|

这些body合计142,434,304个矩阵系数；五级down与对应up的通道投影另外共
1,396,736个E4M3系数。FP16输入adapter有16×32=512个有效系数，readout有
32×4=128个有效系数。合计逻辑矩阵规模143,831,680，约143.83M；
**不含**学习bias/gate/scale、也不将未使用的硬件输出列填充视为有效权重。

---

## 5. 统一数值原语

### 5.1 E4M3量化

当前路径采用有限E4M3数值：最大有限绝对值448，最小normal值2^-6，
subnormal步长2^-9。对有限输入可概括为：

```text
v = H16(x)
a = min(abs(v), 448)
normal_step = 2^(clamp(floor(log2(a)), -6, 8) - 3)
normal = round_to_even(a / normal_step) * normal_step
subnormal = round_to_even(a / 2^-9) * 2^-9
q = sign(v) * min(select(a < 2^-6, subnormal, normal), 448)
```

必须保留入口的FP16舍入。直接从FP32量化到E4M3不是同一边界。实现可以用
FP32 tensor保存已量化数值；tensor的容器dtype不代表逻辑上没有FP8边界。

### 5.2 实际激活：clamped gate × original value

对FP16输入z：

```text
u = clamp(z, -4, 4)
a = H16(abs(u) * (-0.055908203125) + 0.447265625)
b = H16(u * a + 0.89453125)
G(z) = H16(z * b)
```

两次乘加各只舍入一次，对应已恢复的HFMA边界。最后乘的是原始z，不是u。
这套激活在fused Swin和ViT FFN中使用；把它替成`gelu(z)`、`silu(z)`或把
整个多项式只在FP32计算后统一量化，都改变了实际数值语义。

### 5.3 32维Q/K归一化

这里不是通常的LayerNorm：没有减均值，也不能臆造LayerNorm的affine参数。
对每个32维向量，按恢复出的half平方/乘加树求平方和，再做rsqrt归一化。

每个i=0..7先形成：

```text
a[i] = H16(x[i]^2 + H16(x[i+16]^2))
b[i] = H16(x[i+8]^2 + H16(x[i+24]^2))
c[i] = H16(a[i] + b[i])
u[0:2] = H16(H16(c[0:2]+c[4:6]) + H16(c[2:4]+c[6:8]))
norm2 = max(H16(u[0]+u[1]), 6.198883056640625e-5)
inv = H16(rsqrt(float32(norm2)))
normalized = H16(x * inv)
```

Q随后乘相应学习scale并量化；K归一化后量化；V按自己的投影边界量化。
ViT的Q还具有第13节说明的显式sqrt(32)半精度因子。

### 5.4 矩阵乘法与C初值

大部分FP8矩阵乘法按K32切片暴露累加器FP16边界：

```text
C = H16(C0)                              若存在初始残差
for K32 slice in recovered_order:
    partial = float32(A_slice) @ float32(W_slice)
    C = H16(partial)                     首段无C0时
        或 H16(float32(C) + partial)     有前序C时
```

C0通常是Half缩放后的残差。把C0放到所有K段结束之后再加，可能改变结果。
某些短矩阵调用使用Torch直接乘法，再由其消费者暴露相应Half边界；不要根据
helper名字推断每一行Python都强制新建一个FP16 tensor。

这恢复了算法上的分段与舍入位置，但没有模拟NVIDIA指令内部每个乘积的归约顺序。
PyTorch matmul与原生矩阵指令可在这一级保留数值差异。

### 5.5 Split-K不是一条长GEMM

ViT的部分投影和up39采用独立K分区：每段单独计算Half结果，再Half合并。
存在残差时只有第0个分区持有C0，不能每段都重复加入残差。

- ViT FFN contract：4×K1024。
- ViT QKV：2×K512。
- ViT output projection：4×K256。
- up39：4×K256，原生tilesync按分区先后顺序合并。

Torch按分区index顺序执行。ViT原生原子到达的调度顺序不是此纯Torch实现
所承诺模拟的指令级细节。

### 5.6 Swin局部attention指数链

局部各族都有64-slot bias，每head逻辑bias为64×64，而不是假定标准Swin的
某个小relative-position表。QK累加以Half bias作为C初值。

```text
L = M(Q, K^T; bias)
t = clamp(H16(H16(L) * 0.044921875 + 1.30078125),
          1.03125, 1.5693359375)
u16 = bit_pattern_of_half(t)
E = reinterpret_half(((u16 << 5) + 0x8000) & 0xffff)
D = family_specific_half_sum64(E)
R = H16(reciprocal(float32(max(D, epsilon))))
P = Q8(H16(E * R))
A = Q8(M(P, V))
Y = projection(A, initial=Half-scaled attention residual)
```

它不是调用标准`softmax`，也不是`exp(L-max(L))`的任意数值稳定改写。
指数近似、bias初值、归一化、量化及PV次序必须一起保持。

### 5.7 分母的Half归约树

不同族的slot→归约次序不同。2H/4H/8H/16H/ViT在相应64-key permutation后
使用pair、四项串行和、再次四项和、两半相加的树。1H是另一棵已恢复的树。

不能用“数学上都是sum”替换为任意FP32 `torch.sum` 后一次转Half。
这些次序来自独立bias地址/SASS数据来源追踪，不是按图片相关性挑选。

---

## 6. block0：输入适配、全分辨率body和双输出

### 6.1 16通道packet

最小输入是`[B,16,H,W]`，通道顺序如下：

| 通道 | 含义 |
|---:|---|
|0|Gaussian g1|
|1|Gaussian g2|
|2|Gaussian g0|
|3|常量1|
|4–6|centered current RGB|
|7|centered selected-source R|
|8–9|centered selected-source G、B|
|10|style/128|
|11|tone condition|
|12|structure condition|
|13|skin condition|
|14|auto-mask condition|
|15|常量0|

current/source采用已恢复的Half居中和缩放：`H16(H16(value-0.5)*H16(0.125))`。
selected-source的默认来源是current Color；绑定history与motion时由可选外部
适配器产生正向warp的五tap历史采样。它不是额外学习层。

### 6.2 确定性Gaussian

Gaussian来自坐标/frame驱动的PCG风格整数hash，再经过Box–Muller。
整数部分按32位环绕计算，四个uniform由恢复的状态变换产生；主要公式为：

```text
r0 = sqrt(-2 * log(u0)); angle0 = 2π*u1
r1 = sqrt(-2 * log(u2)); angle1 = 2π*u3
g0 = H16(r0*cos(angle0))
g1 = H16(r1*cos(angle1))
g2 = H16(r1*sin(angle1))
```

不是每次调用从Torch全局RNG抽新的噪声。frame变化应通过接口的frame条件进入
坐标hash。标准Torch基本函数与MUFU近似实现可在最终Half舍入前不同；第18节
说明它如何影响最终图像。

### 6.3 输入adapter和两个发布结果

1. 每个真实像素的16维packet经FP16 16→32投影。
2. adapter结果通过真实像素→1H canonical route送入full-resolution 1H body。
3. body最终Half结果的Q8副本作为全分辨率`skip0`。
4. 将Half结果还原到真实像素网格，以两个横向Half pair、pair sum、×1/4池化。
5. 池化后Q8并映回compact consumer视图，成为block1输入。

主分支和skip不能由同一个已量化tensor随意resize得到。保留prequant Half
对于后续池化和块内残差都具有实际意义。

---

## 7. 1H/32局部注意力块

### 7.1 FFN是两路32→64→32

每路有独立W1、W2，共两路，不共享权重：

```text
hidden_s = Q8(G(M(R(input), W1_s)))        W1_s:32×64, s∈{0,1}
output = W2 contraction + learned residual W2_s:64×32
```

W1输出N顺序必须重排为W2输入K顺序。W2两个stream及其K32切片以恢复的次序
进入同一个Half残差累加链。残差属于第一次W2累加的C，而非整个FFN末尾的
附加FP32项。

### 7.2 1H的特殊prequant残差

1H FFN输出的Half值被保留给attention残差；QKV读取它的E4M3副本。
其他更宽族不能直接套用这条规则。

```text
F_half = FFN_with_initial_residual(input)
Q,K,V = projections(Q8(F_half))
attention_output = project(attention(Q,K,V), C0=H16(F_half * attn_skip))
```

1H局部Q/K/V各32维，projection为32→32。PV的K32分组并非简单前32/后32：
第一组对应canonical keys0–15与48–63，第二组对应16–47。

### 7.3 物理相位

以tensor `(y,x)` 原点偏移记号：

| Block | 原点偏移 |
|---|---|
|1、66|(0,0)|
|2、67、70|(-4,-4)|
|3、68|(0,-4)|
|4、69|(-4,0)|

512-byte cell实际是2×8 token stripe，不是普通4×4图片patch。四条stripe按
恢复的TL/BL/BR/TR cell关系组合。外侧cells补零，只有有效query发布结果。

block0使用其专用输入/池化布局，不能仅凭block编号套这个普通相位表。

---

## 8. 2H/64局部注意力块

### 8.1 FFN：两路带32维瓶颈的分解

每一路是：

```text
64 → 128 → 32 → 64
     G,Q8   Q8
```

即逻辑矩阵W1=64×128、W2_main=128×32、W2_tail=32×64。
中间128维和32维均有已恢复的consumer重排。两条tail依次累加到Half缩放残差，
然后发布E4M3 FFN输出。旧的容器形状`64→96→64`不是实际算法。

### 8.2 Attention

Q/K/V各由两个head组成，每head32维，总宽64；output projection为64→64。
FFN publication经FP8读取，是attention的输入及残差来源。

### 8.3 首block5与普通encoder6–8不能混为一谈

- block5使用原生`inpview`入口，消费block4的compact线性像素分组。
- 它必须恢复attention所需的真实空间窗口，再处理其独立全局publication。
- blocks6–8在其FFN物理行视图中重新gather真实CTA窗口，投影后再scatter。

encoder6/7/8的 `(sx,sy)` 原点分别是 `(-4,-4),(-4,0),(0,-4)`。
80×80真实输入上的CTA数分别为11×11、11×10、10×11，而不是始终100个
互不相交的canonical8×8块。有效rows构成全局双射；虚拟rows的Q/K/V为零。

通过4096-byte完整FFN→QKV来源追踪，native QKV M位对应packet byte位
`(6,7,8,2,10,11)`。在这个正确的native行视图中，独立key/query标签证明
query与key的token route为identity。不能把这条结论错误地套到旧canonical视图。

### 8.4 Decoder2H

blocks62–65使用专门的decoder输入、窗口和publication布局。矩阵族与2H一致，
但布局不是将encoder特征简单roll后复用。实现通过`TorchPhysicalDecoder2HBlock`
及decoder布局函数执行，不保留一个“尺寸不合适就退回encoder”的代理分支。

---

## 9. 4H/128局部注意力块

### 9.1 四路分解式FFN

```text
for s in 0..3:
    h_s = Q8(G(M(R(X), W1_s)))           W1_s:128×128
    a_s = M(R(h_s), W2a_s)              W2a_s:128×32
middle = Q8(R(concat(a_0,a_1,a_2,a_3)))  合并宽度128
F = Q8(M(middle,W2b;H16(X*ffn_skip)))    W2b:128×128
```

它不是单个128→512→128稠密FFN。四个branch各自的激活位置、32维瓶颈、
concat后的量化以及W2b都属于实际网络。

### 9.2 Attention与bias

四个head分别使用128→32的Q/K/V矩阵；concatenate后为128通道，再经128→128
projection。bias是4×64×64个Half，但原始存储不是直接row-major。

已恢复的bias地址位包含query位 `(7,5,6,10,1,11)` 与key位
`(4,0,3,8,2,9)`；head提供对应slab。物理字节位置与逻辑(q,k)必须按该关系解释。

### 9.3 quarter-plane窗口

4H的窗口成员从4×4 canonical tile、half-band、side和slot位组合。
奇数half-band的进位会同时影响window坐标和slot bit。按单独轴截断再拼接
会在320/448这类真实尺寸上出错。

decoder57–61的当前layout映射是57→12、58→13、59→14、60→11、61→12。
这是复用已经恢复的encoder物理公式，不是宣称decoder的图像轴与encoder相同。
block56是mixed入口，由其自己的桥接规则处理。

---

## 10. 8H/256局部注意力块

### 10.1 八组256→128→32，之后256投影

```text
for g in 0..7:
    h_g = Q8(G(M(R(X), expansion_g)))      256×128
    r_g = Q8(M(h_g, reduce_g))            128×32
reduced = concat(r_0,...,r_7)             宽度256
F = Q8(M(reduced, output;H16(X*skip)))     256×256
```

reduce/output矩阵的raw地址包含K/N相关的XOR/bit route。
加载器先恢复对应有效矩阵，执行时不能再把同一route重复应用一遍。

### 10.2 Attention

八个head的Q/K/V矩阵各为256×32，concat维度256，projection为256×256。
8个learned head scale以FP32字节存储，但进入对应乘法时显式转Half。
bias总数8×4096个Half，有独立head/query/key布局。

### 10.3 ragged有效位置

这里的本地轴会与外层tile轴交叉。有效token的compact plane顺序通过native
地址的rank构造；不能只用`height//8,width//8`丢掉不足8的尾部。

- body只发布有效位置，虚拟slot作为零Q/K/V进入相应窗口。
- block22的池化需要扩展到native对齐slots；pool消费prequant投影Half。
- 这种对齐影响后续真实descriptor尺寸，不是为了让某个测试通过随意pad。

---

## 11. 16H/512分解式局部注意力块

### 11.1 每block有四个物理阶段

blocks23–30及40–47各自包含：

1. factorized FFN expansion；
2. FFN projection及学习残差；
3. QKV、local attention、bias和head scale；
4. attention projection及学习残差。

block30另外有terminal512→1024投影。四阶段记录不是四个额外顶层block。

### 11.2 真实factorized FFN

```text
P = Q8(M(R(X), expand_input))             512×512
P_groups = reshape/reorder(P, 8,64)
C = zeros(Half, ...,8,64)
for rep in 0..3:
    H_rep = Q8(G(group_M(P_groups, factor_in[rep])))
    C = group_M(R(H_rep), factor_out[rep]; C)
expanded = Q8(concat_groups(C))
F = Q8(M(R(expanded), ffn_projection; H16(X*ffn_skip)))
```

每个rep有8组64×64 factor_in和8组64×64 factor_out。
前一个rep的输出作为后一个rep的累加器初值，不是先各自产生独立Half结果再任意求和。

本体不是512→2048→512 dense FFN，也不是将记录硬拆成两个512宽矩阵后构造
SwiGLU。当前活跃路径使用真实group factor结构。

### 11.3 16-head窗口注意力

总QKV矩阵为512×1536，拆成16个32维head的Q/K/V。
每head有64×64 bias，scale存储为16个FP32值；output projection为512×512。
归一化、Q8边界、近似指数与归约遵循本族规则。

encoder和decoder分别通过`_attention_hwc`、`_attention_physical`处理其实际视图，
不是互为无条件别名。对齐后的S5空间中，padding与输出有效范围都要保留。

---

## 12. 编码器下采样与block30 terminal

### 12.1 共同结构与关键差异

每个terminal body同时产生：

- Q8后的skip publication，留给解码器；
- 未量化的最终projection Half，供池化和transition使用。

| Terminal | 池化的数据/分组 | 后续投影 |
|---:|---|---|
|0|真实像素上的横向Half pair→pair sum→×1/4|无通道扩张，32保持32|
|4|恢复真实像素后，横向Half pair→pair sum→×1/4|32→64|
|8|fused M位0/1定义的四项组，按恢复的pair次序|64→128|
|14|**纵向**Half pair→pair sum→×1/4|128→256|
|22|8H有效/补齐slot中的四项组，按实际坐标取pair|256→512|
|30|projection Half上的横向pair→pair sum→×1/4，descriptor四对齐|512→1024|

pool结果进入对应Q8及投影边界。不得从已经量化过的skip重建这些Half值。

### 12.2 DS14为什么不能换成普通平均池化

在2×2组中布置`(+64,-64,1/64,1/64)`可以严格区分Half加法树：
先抵消±64的pair保留两项小值，其它pair次序会因Half舍入丢失小值。
原生identity body/transition控制实验明确支持纵向pair；它不是依据自然图的
最高相关性选出来的公式。

### 12.3 block30、全局token和repack

block30 attention projection的Half结果被拆为Q8 skip30及Half池化来源。
池化后descriptor进行4对齐，再投影为1024维全局token。

真实小图中S5=12×12，池化6×6，descriptor变为8×8；这些64个空间token进入
后续ViT。descriptor padding与ViT K64归约所需的虚拟key不是同一件事。

---

## 13. blocks31–38：全局ViT

### 13.1 每个ViT块的学习结构

八个块具有相同结构、独立权重：

```text
X [B,N,1024]
  │  FFN expand 1024→4096, G, Q8
  │  FFN contract 4096→1024, split-K4 + FFN residual, Q8
  ▼
F [B,N,1024]
  │  Q/K/V三投影1024→1024，每个split-K2
  │  reshape成32 heads × 32 dimensions
  │  Q/K归一化；Q额外sqrt32与learned scale；Q/K/V量化
  ▼
全局attention，对N个真实descriptor tokens
  │  concatenate32 heads
  │  output projection1024→1024，split-K4 + attention residual
  ▼
Q8 output；除最后一块外执行next-input通道路由
```

attention residual使用FFN之后的F，不是最初的block输入X。
FFN residual则使用对应raw/canonical通道关系下的block输入。

### 13.2 Q的缩放

```text
Q = Q8(H16(H16(normalize32(Q_half) * 5.65625) * H16(learned_scale)))
K = Q8(normalize32(K_half))
V = Q8(V_half)
```

5.65625是这里显式使用的Half sqrt(32)值。不能同时再无证添加标准
`QK / sqrt(32)`因子。各族scale和指数仿射链已经确定了数值尺度。

### 13.3 全局指数链不同于局部Swin

ViT没有把局部Swin的64×64 bias复制成global bias，当前全局链为：

```text
L = Q @ K^T
T = clamp(H16(H16(L)*0.08953857421875 + 1.708984375),
          1.439453125, 1.9775390625)
E = reinterpret_half(((half_bits(T) << 4) + 0x4000) & 0xffff)
D = native_order_half_denominator(E)
U = M(Q8(E), V)
A = Q8(H16(U * H16(reciprocal(float32(max(D,epsilon))))))
```

关键是 **先量化未归一化的指数E，做PV，再乘Half reciprocal**。
`Q8(E/D) @ V`虽然看似softmax写法，但不是同一个实际算法。

### 13.4 N不是64倍数时

K/V按已经恢复的column-major/low-XY关系组织为native fragment顺序。
分母按K64块归约；超出真实N的虚拟key具有QK=0，指数为43/512。
原生过程先把这些项加入Half归约，再扣除其总贡献，不能直接缩短归约长度。
虚拟key的V=0，其最终PV零贡献可以省略。

例如N=96时，有128个分母循环slot，其中32个虚拟key被按真实边界加后扣除。
这与S6里原本就存在的空间padding token完全不同。

### 13.5 repack与token语义

模型保留逻辑二维token视图，native pointwise矩阵行则由
`e16(logical_to_fragment(H,W))`给出：

```text
e16(m) = 16*floor(m/16) + 2*(m mod 8) + floor((m mod 16)/8)
```

repack58与repack99是无学习权重的数据重排。必须将它们和后续consumer的地址
公式组合，而不是对两个不同顺序的tensor直接计算相关性。此前错误的直接HWC
比较曾制造“ViT输出崩坏”的假象，修正view之后实际链可以逐值对齐。

---

## 14. 解码器与所有上采样连接

### 14.1 block39：没有额外Swin body的连接

block39将ViT1024通道主分支按真实输出像素需求严格2×复制，裁到skip30有效范围。
它不是依据两个tensor大小比例做任意interpolate：source坐标是整数`output//2`。

```text
main = Q8(repeat2_and_crop(main))
projected = splitK4(main @ W1024x512)      四段各K256
fused = Q8(Half projection + skip30 * sin)
output = encoder512_to_decoder512_view(fused)
```

四个原生K256分区各自从零C开始，前序分区的Half部分和按tilesync顺序传递，
完成后才做skip融合。单条K1024累加不能替代。

### 14.2 mixed48/56/62/66的共通原则

这些块把上采样projection、学习skip门和一个真实局部body打包在一起。
虽然名字里包含upsample，不能只做resize+add而漏掉body。

| Block | transition矩阵 | 融合skip | fused body |
|---:|---|---|---|
|48|512×256|skip22，256通道|8H/256|
|56|256×128|skip14，128通道|4H/128|
|62|128×64|skip8，64通道|decoder2H/64|
|66|64×32|skip4，32通道|1H/32|

主分支投影保持Half进入skip融合。不能在transition刚算完时就先Q8，再与skip相加。
实际source选择通过对应的索引公式完成，不是对所有family使用相同HWC nearest。

### 14.3 mixed块的残差边界并不统一

- 48/56/62：融合值进入对应body的FP8发布/重读路径，FFN残差也来自该Q8值。
- 66：融合Half保留给1H首W2残差；W1才读取Q8副本。

这是通过producer/consumer数据来源恢复的区别。将1H的prequant策略无条件推广
到全部wide族，同样会导致错误。

### 14.4 block47的特殊publication

普通decoder512 body输出与block47的terminal输出不是同一种raw存储。
后者为planar N16，供up48消费。当前canonical decoder view由以下组合恢复：

```text
encoder-first512 raw codec
    → _encoder_to_decoder512_pixels
    → decoder current view
```

该组合曾对完整163,840个已恢复地址验证，而非用最近数值匹配“猜”一个排列。

---

## 15. block70：双门、1H本体和四通道读出

### 15.1 双输入融合

输入main来自block69，空间级别S1；skip0是S0全分辨率32通道。
main先通过真实像素坐标2×复制，再回到body所需布局。两个32维学习门分别
缩放main与skip，且存储顺序为projection-N，不能按QKV-K顺序直接读取。

```text
m = H16(main_upsampled * main_gate)
x = H16(skip0 * skip_gate + m)
```

这里先舍入main乘积，再对skip乘加一次舍入；提前单独舍入skip乘积会改变边界。

### 15.2 学习body与latent

融合Half送入1H/32 FFN与attention，block70使用物理XY相位。
最终body保留未Q8的32通道Half latent，供FP16读出。

### 15.3 FP16 32→4读出

readout权重由两个真实K16的HMMA-B fragment解码；每个fragment物理上有8个
输出列，其中后4列确认为零，逻辑上只有4个有效输出。不能把字节连续reshape
为任意4×32矩阵。

```text
C = H16(latent_N[...,0:16] @ W0[16,4])
head_N = H16(C + latent_N[...,16:32] @ W1[16,4])
head = physical_pixel_and_channel_route(head_N)
```

输出为`[B,H,W,4]`：前三通道是learned RGB residual，第四通道是history logit。
在常用单帧观察路径中：

```text
RGB = clamp(Color + head[...,0:3]/4, 0, 1)
```

这是head之后的颜色解释，不是网络内部又增加一个可学习层。
第四通道也不是alpha颜色本身：宿主可通过sigmoid及blend_scale等参与历史合成。

---

## 16. 物理布局与动态尺寸

### 16.1 必须区分四种坐标

1. 真实显示像素 `(x,y)`；
2. 模型维护的canonical tensor token/channel；
3. 矩阵fragment的M/K/N坐标；
4. native buffer中的字节地址、plane、lane和slot。

`QKV-K residual`和`projection-N residual`即使shape相同也不能直接逐项相加。
后者的学习缩放向量也可能按N序存储。索引变换是在恢复同一张图的数据流，
不是额外的学习能力。

### 16.2 关键布局关系

| 边界 | 必须保留的事实 |
|---|---|
|block0→1|full skip与compact main不同发布；首1H输入使用compact inpview codec|
|block4→5|compact线性分组、首2H真实inpview入口以及block5独立publication|
|blocks6–8|phase-specific FFN M/N视图与全局CTA attention gather/scatter|
|block14→15|四个pooled quarter-plane进入8H，交错取决于真实window_columns|
|block22→23|ragged有效地址rank与compact descriptor，不可丢尾列|
|block30→31|prequant池化、descriptor对齐、首ViT输入K route|
|block38→39|global token行序、repack99与真实2×consumer地址组合|
|block47→48|terminal planar N16，不是普通decoder512 raw|
|block56/62/66|各自mixed输入与body布局，不能用统一resize替代|
|block69→70|真实像素2×复制与main/skip N序学习门|

### 16.3 已验证尺寸与非保证范围

完整单文件/模块head对照已覆盖：

```text
H×W = 512×640, 512×576, 384×640, 320×320, 448×512
```

这些尺寸跑过全部71个block。它们证明实际非参考尺寸路径，而不是只跑一个
孤立算子fixture。它们不代表任意整数H/W、任意HDR资源、全部batch或设备都已穷举。
真实神经尺寸应来自调用者的有效几何约束或宿主协商；WebUI使用的自动尺寸
只是显式标注的本地测试策略。

---

## 17. 最小API、外部输入与WebUI

### 17.1 最小学习接口

```python
from dlss5_model import load_model

model = load_model('weights_ht_blob.bin', device='cuda')
# prepared: 按第6节构造的[B,16,H,W] tensor
result = model.infer_minimal(prepared, collect_trace=True, return_result=True)
head4 = result.head       # [B,H,W,4]
latent = result.latent    # [B,H,W,32]
trace = result.trace     # 71个顶层block的观测
```

需要低开销推理时关闭trace并复用同一model实例。临时解码及GPU传输是启动成本，
不要每张图片重新加载147MB BIN。

### 17.2 哪些属于外部适配

| 内容 | 模型接口关系 | 本文的边界 |
|---|---|---|
|PNG解码、resize、EXIF、颜色域|产生实际Color资源|不是学习图层|
|Gaussian/conditioning packet|成为16维输入|公式与slot恢复；基本函数实现精度另述|
|Style/Tone/Structure/Skin/Auto|改变输入condition|不代表切换隐藏权重集合|
|MVec、history采样|改变selected-source及合成|不等于已复现完整history生命周期|
|Intensity、Backbuffer、ControlMask|head之后的宿主合成|不能用增益补偿错误网络|
|RGBA16F surface格式转换|改变最终纹理舍入|不属于额外学习参数|

`FrameInputs`/`infer_frame`提供部分已经恢复的默认pre/post便利层；最小学习接口
仍是packet→head4。调用者不能把RGB三通道图片直接冒充16通道packet。

### 17.3 WebUI如何使用模型

WebUI加载同一`dlss5_model.py`和原始BIN。后台加载线程不会改变模型：它只是
让页面/状态接口先可访问，模型ready后再允许推理。

- 输入RGBA8 `/255 → FP16 toward-zero Color`；
- 使用一次反射后钳制的补边，不是周期`np.pad(reflect)`或edge padding；
- 默认Natural、Structure2、Tone1、Skin-1、Auto Mask关闭；
- 单帧显示真实head/4，差分倍率仅影响观察图；
- 不提供Residual Gain；时域合成明确标为实验性宿主适配。

安装、后台加载、显式神经尺寸与本地测试策略见
[README_DISTRIBUTION.md](README_DISTRIBUTION.md)。

---

## 18. 验证结论、数值差异与边界

### 18.1 证据分层

本项目没有把“shape对”“finite”或“相关性高”单独当作算法完成标准。
证据包括：

1. SASS中的真实操作数、量化/归约/残差边界；
2. 原始权重字节与矩阵/scale/bias解码；
3. 受控identity、单分支、地址签名和严格可区分的数值输入；
4. native同帧中间buffer与真实纹理oracle；
5. 当前图的完整自馈执行，而不是永久替换中间结果来掩盖错误。

对于已恢复的分段或舍入边界，即使某张图片相关性暂时下降，也不以此为理由
回退正确公式或拟合gain。

### 18.2 三张实际输入的完整自馈归因

仅在诊断中匹配原生矩阵及同一Box–Muller基本函数的算术实现，不改变学习图、
权重、slot公式、量化位置或随机seed，也不注入任何中间结果：

| 实际神经尺寸 | block0 compact | block0 full skip | block70 main | 最终RGB |
|---|---|---|---|---|
|320×320|全等|全等|全等|49,152个RGB分量全等|
|512×640|全等|全等|全等|841,500个中仅一个差5.96e-8|
|448×512|全等|全等|全等|504,000个RGB分量全等|

最终RGB比较计入`clamp(Color+head/4)`与已观察到的FP16 toward-zero surface转换。
原人物唯一差异位于一个次正规数，大小为最小正FP16 subnormal；这是尚未完全
刻画的surface转换细节，不应据此给学习图加入特判。

### 18.3 18个adapter差异的来源

当只匹配矩阵算术时，小图adapter还有18/3,276,800个FP8数值不同。
进一步只匹配Box–Muller基本函数实现后，这18项全部消失，block0双输出也全等。
对应Gaussian Half有599/307,200项发生变化。

这证明同一公式的`log/sqrt/sin/cos`近似实现及其舍入能通过量化产生稀疏变化。
后续量化网络会放大某些微差，最终图像差异不一定也很小。

### 18.4 分发模型没有使用上述native诊断实现

交付仍是标准Torch。其当前真实自馈测量为：

| Case | head correlation | 可选完整颜色路径output MAE |
|---|---:|---:|
|原人物Natural+Structure2|0.99197343|0.00435551|
|small320|0.96935346|0.00607497|
|448 case|0.99665327|0.00197551|

这里的native head对照由Color/Output纹理推导，包含surface量化；output MAE是
可选帧合成路径对DLL输出的误差，不是WebUI显示的“输出相对输入”的delta MAE。
这些数值没有因为归因结论而被改写，也没有通过显示增益隐藏。
“算法和结构对齐”与“标准Torch逐bit复现DLL”是两个不同结论。

目前的独立推导和真实执行共同支持当前固定配置学习图的完整算法对齐；不将
这三张图片的验证外推成所有输入、设备、宿主组合的形式化证明。

---

## 19. 实现索引与常见误读

### 19.1 在单文件中寻找实现

分发包不需要下面这些开发模块；相应类和函数已内联到`dlss5_model.py`。
开发仓库名称仅用于源码导航与证据追溯：

| 对象 | 单文件中的主要类/函数 | 开发模块来源 |
|---|---|---|
|原BIN加载|`DLSS5Model`, `load_model`|单文件构建器与权重导出/解码工具|
|整体图|`DLSS5Reconstruction`|`torch_rebuild.py`|
|编码器|`DLSS5Encoder`, `InputBlock`, `PackedEncoderDownsample`|`torch_rebuild.py`|
|1H body|`TorchPhysical1HBlock`|`physical_1h_block.py`|
|2H body|`TorchPhysical2HBlock`, `TorchPhysicalDecoder2HBlock`|`physical_2h_block.py`|
|4H body|`TorchPhysical4HBlock`|`physical_4h_block.py`|
|8H body|`TorchPhysical8HFFN`, `TorchPhysical8HBlock`|`physical_8h_block.py`|
|split16H|`TorchPhysicalSplitSwin16H`|`physical_split_swin.py`|
|globalViT|`Vit1DWeights`, `StructuralVit1D`, `_vit_attention`|`torch_rebuild.py`|
|mixed upsample|`TorchPhysicalUpsample48/56/62/66`|`torch_rebuild.py`|
|学习输出头|`StructuralPostBlock.forward_head`|`torch_rebuild.py`|
|公共数值原语|`matmul_fp16_accumulate`, `sum_fp16_key64`等|`physical_2h_block.py`及1H实现|
|输入packet|`build_preblock_features`, `gaussian_dither`|`dlssnr_inputs.py`|
|可选资源/颜色适配|`FrameInputs`, `reconstruct_block70_color`|`block70_runtime.py`, `block70_color.py`|

某些历史类名仍含`Structural`，不代表当前活跃实现仍是标准Swin placeholder。
判断应看其实际forward、权重解码、布局和验证证据；旧六个proxy类及fallback构造
已经从活跃模型中移除。

### 19.2 不应再作的简化

- 不把153个record理解成153个dense层。
- 不把每族FFN统一替成`C→4C→C`。
- 不把32维Q/K归一化写成LayerNorm。
- 不替换实际激活、bit指数和Half归约为通用GELU/softmax。
- 不把所有残差都放到最后相加，也不把1H prequant规则套到所有wide族。
- 不在下采样前把唯一的Half池化来源先Q8。
- 不把ViT的指数Q8→PV→reciprocal改成归一化概率Q8→PV。
- 不把普通HWC roll当作物理shifted窗口。
- 不丢弃真实descriptor padding或为绕错强行换参考分辨率。
- 不以近似图片相关性代替地址、数值边界和完整图的证据。
- 不把native算术诊断结果冒充为标准Torch分发包的逐bit承诺。

进一步的接口示例见[DLSS5_MODEL_API.md](DLSS5_MODEL_API.md)，范围定义见
[DLSS5_MINIMAL_MODEL.md](DLSS5_MINIMAL_MODEL.md)。
