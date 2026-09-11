# WebUI

入口是项目根目录 `whitebox_app.py`，静态文件是 `webui/`。任务经根目录 `run.Engine` 执行，
NR 由 TileLang 完成；旧参考 WebUI/CLI 不再保留，只有根目录这一套应用入口。导入与状态查询不加载
Torch / NR / TileLang，也不自动监听或打开浏览器。

## 使用

只有使用者明确手动启动时才运行：

```powershell
cd c:\work\dlss5_remake
.venv\Scripts\python.exe whitebox_app.py
```

默认仅监听 `127.0.0.1:7861`，不暴露外部 host 参数。支持 `--port`、`--workspace`、`--model-dir`。
默认任务目录为 `~/.dlss5-tilelang-whitebox/jobs/<id>/`。此工具仅适合受信任的本机使用，不要反向代理到外网；
拒绝非 loopback Host 与跨 Origin 请求，单请求上限 1 GiB。

1. 选择视频、批量图片或连续序列并上传。支持 JPEG/PNG、浮点 NPY、视频以及 guided manifest；
   视频编解码依赖已安装的 FFmpeg/FFprobe。
2. 计算路线只有两条：`tilelang-vit`（默认，TileLang NR）与 `torch`（全 Torch 参考，不经过 NR dispatch）。
   不自动切换、不静默回退。
3. 输出尺寸与可选 NR work size 原样传给 Engine。UI 不强制 1088×1920，不另做缩放或 padding；
   后端继续原 `neural_size` 策略，状态显示真实 `last_frame.neural_hw`。首次 shape 准备/编译可能较慢。
4. 点击加入队列。只有一个 worker 和一个成功加载的 Engine，最多 16 个活跃/排队任务；
   独立 pipeline 历史由 `Engine.run` 每次创建。
5. 取消通过真实 cancel 回调在安全边界生效，已完成的 PNG/NPY 保留；下一帧 reset 通过真实
   `before_frame` 回调消费一次。

HDR 真输出仍是 float NPY；PNG/MP4 为 SDR 预览。完整配置、每帧记录、报告与后端记录可下载。
上传为平铺目录：manifest 输入字段使用已上传文件的安全 basename，不允许绝对路径或目录穿越。
估计器与 NR 模型目录统一为仓库根 `model/`；Engine 只从本地加载，推理不联网下载。

## Engine 接口

```python
engine = Engine(fast_load=True)
proc, report, stats = engine.run(
    manifest, output, backend='tilelang',   # 'tilelang' 或 'torch'
    quiet=True, cancel=cancel_event.is_set,
    progress=progress, before_frame=before_frame,
)
```

- `progress`：stage、message、processed_frames、total_frames_estimate、last_frame、seconds。
- `before_frame(index)`：有 reset 请求时返回 `{'reset': True, 'reset_reason': 'UI explicit reset'}`，消费后返回 None。
- `report`：status、processed_frames、seconds、frames、video_output；可选 `stage_timings`。
- `stats.nr`：`selected_backend`、`actual_backend`、`calls`、`fallback_calls`、
  `shapes: [{height,width}]`、`prepare_seconds`、`compile_seconds`。`calls` 是本任务新 NR 的实际调用，
  不是 Engine 生命周期累计。
- 实际 neural size 来自 `last_frame.neural_hw`。缺失的调用数、shape 准备、编译与 stage timing 显示“未测”，
  不根据选择或 pass 数推断为 0。

## HTTP（适用于 test_client）

- `GET /`、`GET /ui/app.js`、`GET /api/status`
- `POST /api/jobs`：multipart `manifest`（JSON 字符串）、`files`（可重复）、`nr_backend`
  （`tilelang-vit` / `torch`，默认 `tilelang-vit`）；也接受 JSON `{manifest: {...}, nr_backend: "..."}`。
- `GET /api/jobs/<id>`、`POST /api/jobs/<id>/cancel`、`POST /api/jobs/<id>/reset`
- `GET /api/jobs/<id>/outputs?offset=0`：每页 100 个文件，包含总数。
- `GET /api/jobs/<id>/files/<name>?download=1`：限制在该任务 output 内，支持 Range 与 PNG/NPY/MP4/JSON/JSONL 下载。
