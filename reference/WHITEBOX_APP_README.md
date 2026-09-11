# 独立白箱图像处理应用

全Torch图像链：输入 →（RGB模式的RAFT/Metric VDA估计guides）→ Feeder公开guide算法 → Torch FSR2 → 固定NR多pass → 显式颜色/保护策略 → 图片、浮点HDR与可选视频预览。

这是开源时域重建接已恢复NR的白箱路线，不是DLSS DLAA/SR或DFC/RenoDX等价产品。没有代理网络或输出gain来代替未恢复的NR算法。

## 环境与权重

已验证Windows、Python3.11、Torch2.5.1+cu124、torchvision0.20.1+cu124、einops0.8.0，RTX4080 Laptop 12GB。依赖见`requirements-whitebox.txt`。请选择匹配的Torch/torchvision CUDA构建；不要覆盖正常运行的现有Torch或安装VDA上游旧requirements。

视频额外需要本机`ffmpeg`和`ffprobe`，放入PATH；也可设置`WHITEBOX_FFMPEG`与`WHITEBOX_FFPROBE`。没有把FFmpeg可执行文件、SDK、游戏DLL或本地测试视频放入此包。

权重以**独立文件**加载，不嵌入Python：

- 根目录`weights_ht_blob.bin`：原固定NR权重。
- `guide_models/`：RAFT-small和Metric VDA Small，文件名与SHA见其manifest。

带权重的本地研究包已经包含这些独立文件；code-only包必须由使用者提供NR权重，并显式运行`tools/fetch_rgb_guide_weights.py`准备公开guide权重。推理或UI任务不会联网补下模型。缺失或hash错误会显示失败，不换成常量depth/假motion。

## CLI

完整manifest入口（兼容既有guided、SDR/HDR、RGBA、逐帧参数）：

```powershell
python -m whitebox_pipeline --manifest sequence.json --output new-output
```

直接读视频，未知标签必须明确解释；下面是**明确假设BT.709**，不是自动推断：

```powershell
python -m whitebox_pipeline --video movie.mp4 --transfer BT709 --matrix bt709 --mp4 --output video-output
```

短段试用及独立NR工作尺寸（尺寸顺序是height width）：

```powershell
python -m whitebox_pipeline --video movie.mp4 --transfer BT709 --matrix bt709 --max-frames 8 --size 540 960 --nr-work-size 270 480 --mp4 --output short-output
```

批量图片默认逐张reset；`--sequence`才把目录当连续序列：

```powershell
python -m whitebox_pipeline --input-dir photos --output batch-output
python -m whitebox_pipeline --input-dir frames --sequence --fps 24 --mp4 --output sequence-output
```

未指定target时，独立照片按各自尺寸处理。相邻source尺寸变化会显式reset。目录按文件名排序，建议使用零补齐序号。

`--config settings.json`可给便利入口提供NR passes、颜色、guides和输出策略；完整原始inputs/帧级配置用`--manifest`。例：

```json
{
  "mode":"rgb_estimated",
  "color_encoding":"sRGB",
  "output_size":[540,960],
  "video":{"file":"movie.mp4","transfer":"BT709","matrix":"bt709","max_frames":8},
  "nr_passes":[{"structure":2},{"structure":1}],
  "nr_work_size":[270,480],
  "output_settings":{"nr_resample":"easu","sharpness":0.35},
  "video_output":{"file":"preview.mp4"},
  "frame_events":{"4":{"reset":true}}
}
```

## UI（需使用者手动启动）

```powershell
python whitebox_app.py
```

仅监听`http://127.0.0.1:7861`。可用`--workspace`、`--device`、`--weights`、`--model-dir`和`--port`指定本机配置。它与旧单图WebUI/7860及旧分发包分开；不自动打开浏览器。

导入app和读取首页/status不会加载Torch。第一项任务在唯一后台GPU worker中加载NR（本机通常约一分钟），状态页持续响应；后续任务共享固定模型权重，但每项任务创建自己的估计/guide/FSR/NR历史。

UI支持：上传视频或多张图片、生成或编辑完整manifest、排队、查看模型加载/任务进度/实际尺寸/估计来源/有效NR参数、下载PNG/NPY/报告、取消和下一帧显式reset。guided任务可上传对应的float depth/MV并在高级manifest引用文件名。上传文件是平铺目录，名称需唯一；不解压用户提供的ZIP。

任务目录默认`~/.dlss5-whitebox/jobs/<id>/`，包含inputs、output与失败时的error.txt。输出是用户数据，不会自动清理或覆盖已有目录。这个界面是受信任的本机工具，不是多租户网络服务；拒绝非loopback Host和跨Origin控制请求，不要将端口代理给外部网络。

## 输出与视频语义

- PNG：SDR图像或明确的HDR压缩预览，RGBA会保留alpha。
- HDR/PQ/linear或`save_float=true`：NPY真实浮点输出，编码/单位记录在报告。
- `preview.mp4`：**静音、不含alpha的SDR预览**，固定target尺寸且宽高均为偶数。HDR master仍是NPY；有alpha或动态target的任务请导出帧，不会静默丢alpha/改尺寸。
- `report.json`：状态、已完成总帧数及最近100帧。
- `frames.jsonl`：每帧完整记录，流式追加。长视频不会把全视频图像或所有中间tensor留在内存，也不会反复重写不断增长的全轨迹JSON。
- `run-manifest.json`：本次有效输入配置副本。`save_guides=true`另输出guide NPZ。

视频用本地FFmpeg进行codec和YUV矩阵转换；RGB图像计算由Torch承担。支持声明的BT.709/sRGB SDR以及PQ/BT.2020视频；未知transfer/matrix报错，可显式指定。BT.709编码先按标准解码再转sRGB估计视图，不冒充sRGB同一曲线。HDR通过浮点planar RGB读出，不先降成8bit SDR。

视频按声明/探测FPS通过FFmpeg fps filter生成明确的恒定帧率序列；VFR输入会重采样，不声称保留其原始逐帧PTS。帧数是预计值，实际完成数以报告为准。未实现音频保留、HLG或任意SDR宽色域变换；alpha视频须转RGBA帧序列。

取消在安全帧/pass边界生效，不强行中断正在执行的CUDA kernel。已写完的PNG/NPY前缀保留；不完整MP4移除，不冒充完成。失败显示具体error；后续任务仍可继续，新任务不复用失败任务的历史。

## 参数和已验证范围

颜色、alpha、mask分类、HDR context、多pass下游reset、EASU差分策略详见`P3_COLOR_AND_HISTORY.md`。估计器及固定相机假设详见`RGB_ESTIMATED_PIPELINE.md`；FSR源码来源及参考误差见`WHITEBOX_PIPELINE.md`。详细固定模型架构见`DETAILED_MODEL_ARCHITECTURE.md`。

此前真实960×540 RGB入口、HDR双pass、工作尺寸切换、取消回滚都已运行；P4另测试实际视频解码→模型→PNG/可播放MP4与UI任务API。测试使用Flask test_client，不监听端口、不打开浏览器；没有把浏览器自动化或原生HDR拍摄质量说成已验证。

这是离线处理工具，不承诺实时或任意4K/8K输入适合12GB显存。较小NR work是明确的差分传送策略，不等价于full-size NR，也不是对含global ViT网络做tiling等价。

## 许可与分发

`PACKAGE_MANIFEST.json`列出包内容、hash、源码commit及是否包含权重。`WEIGHT_RIGHTS.md`区分源码许可与权重/数据权利。FSR/Feeder/VDA来源声明随包保留。**整体包没有被自动重新许可为MIT或Apache；本地研究打包不等于获得公开或商用再分发授权。**
