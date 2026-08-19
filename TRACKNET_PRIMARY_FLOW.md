# TrackNetV3 主羽毛球检测流程（开发分支）

分支 `feature-tracknet-primary-flow` 将 TrackNetV3 的**原始**
`Frame,Visibility,X,Y` 结果作为一场新分析的主羽毛球来源。它的目标是让
操作者直接查看带人物、骨架、TrackNet 球点与轨迹的整场标注视频，而不是先在
WebUI 中执行 A/B 报告。

## 行为边界

- 选择 `TrackNetV3 原始轨迹（GPU 主流程）` 时，服务器先完整运行 TrackNet，
  再开始人物、球场、轨迹、球路和回合主流程。
- 标注视频右上角显示 `Shuttle: TrackNetV3 raw`；结果页提供原始 CSV 下载。
- `detections.jsonl.shuttlecock` 会标记
  `source=tracknet_v3_raw`、`measurement_kind=temporal_heatmap` 和
  `confidence_status=uncalibrated_binary_visibility_threshold_0.5`。
- 轨迹预测仍显示为 `predicted`，且 `accepted=false`；不会冒充 TrackNet 的真实点。
- TrackNet 失败时，本分支会明确失败，**不会静默回退到 YOLO**。需要旧路径时，在
  WebUI 下拉框明确选择 `YOLO 羽毛球检测（旧路径）`。
- 此分支用于直观验收，不代表 TrackNet 已通过正式模型替换或比赛判分验收。

## GPU 服务一次性配置

TrackNet 源码和 `TrackNet_best.pt` 已存在于 GPU 持久目录时，将以下内容加入
`/root/good-badminton-gpu-api-state/.gpu-api.env`：

```bash
GOOD_BADMINTON_TRACKNET_ROOT=/root/good-badminton-gpu-api-state/models/tracknetv3/source
GOOD_BADMINTON_TRACKNET_CHECKPOINT=/root/good-badminton-gpu-api-state/models/tracknetv3/ckpts/TrackNet_best.pt
GOOD_BADMINTON_TRACKNET_PYTHON=/root/miniconda3/bin/python3
GOOD_BADMINTON_TRACKNET_BATCH_SIZE=16
GOOD_BADMINTON_TRACKNET_BACKGROUND_SAMPLES=120
```

代码包仍不包含 TrackNetV3 上游源码和权重；它们由上述持久目录保存。更新应用代码后，
重启 API 即可读取这些变量。若其中任一路径无效，任务会在开始时以明确错误失败。

## 发布与本地预览

本地打包时，主流程所需的 TrackNet 适配器会自动包含：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\package_gpu_api.ps1
```

上传固定包并在 GPU 服务器执行：

```bash
bash /root/good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh
```

然后重启本地 WebUI，选择 `TrackNetV3 原始轨迹（GPU 主流程）` 并运行一段视频。
远端任务完成后，WebUI 自动下载标注视频、`detections.jsonl`、`metadata.json` 和
`tracknet_raw_csv` 到本地 `outputs/remote_jobs/`；无需再手工从服务器取 CSV。

Windows 本机当前没有可用 CUDA/TrackNet 权重时，不应选择本地 fallback。WebUI 会保留
远端失败原因，而不是用 YOLO 输出伪装成 TrackNet 结果。
