# TrackNetV3 羽毛球轨迹 A/B 基准

这个实验把当前 YOLO 路径（A）与 TrackNetV3 原始热图路径（B）放在同一套、人工复核的逐帧真值下比较。它不会替换生产模型，不会修改既有 `detections.jsonl`，也不会将 TrackNet 的修复轨迹冒充为真实球点。

## 比较对象

| 组别 | 输入 | 可进入轨迹/回合证据 | 说明 |
|---|---|---|---|
| A | 当前 `detections.jsonl.shuttlecock` | 仅既有 `status=detected` 且 `accepted=true` | `predicted` 不计作检测成功 |
| B | TrackNetV3 原始 `Frame,Visibility,X,Y` CSV | `Visibility=1` 的时序热图球点 | CSV 未提供校准置信度，报告会明确标记 |
| B* | TrackNetV3 + InpaintNet CSV | 否 | 只供展示和人工复核，状态为 `rectified_tracknet` |

TrackNetV3 的上游代码与权重保持在本仓库外。这样不会污染项目的 Python/PyTorch 依赖，也避免将外部权重提交到 Git。上游的 `predict.py` 直接调用 CUDA，因此请使用其可运行的 GPU Python 环境。

## 1. 创建人工复核集

先选固定机位、球场稳定、且已经用当前系统跑完的短片段。优先涵盖背景混淆、短遮挡、高速模糊、真实出画/落地和清晰对照五类情况。标注帧必须落在当前分析输出的 `detections.jsonl` 中已有的 `frame`，否则基准会拒绝比较。

```powershell
.\.venv\Scripts\python.exe evaluation\shuttle_tracknet_ab\init_annotation_set.py `
  --video "<原视频路径>" `
  --output-dir evaluation\shuttle_tracknet_ab\datasets\match_a_segment_01 `
  --start-sec 60 `
  --duration-sec 8 `
  --sample-fps 25
```

逐帧编辑 `annotations.jsonl`：

```json
{
  "frame_index": 1800,
  "time_sec": 60.0,
  "image": "frames/frame_00001800.jpg",
  "label_status": "complete",
  "shuttle": {
    "visibility": "visible",
    "image_xy": [426.5, 188.0]
  }
}
```

- `visible`：人眼能确定球中心，填写**原视频像素**坐标。
- `not_visible`：球确实不可见、被遮挡或出画，`image_xy` 必须为 `null`；该类用于计算背景假阳性。
- `ambiguous`：人眼也无法判断，保留但不进入指标分母。

全部复核后，把 `metadata.json` 改为：

```json
{
  "ground_truth_source": "human_manual",
  "reviewer": "<复核人>",
  "review_status": "complete"
}
```

SHA-256 会阻止把标注误用到另一份视频；未完成、代理复核或模型生成的标签会被基准阻塞。

## 2. 在独立 TrackNetV3 环境跑 B 与 B*

将官方仓库和检查点放在本机可用的 GPU 环境中。先只跑原始 TrackNet；确认 B 的原始召回和误检值得后，再加 InpaintNet 做 B*。

```powershell
.\.venv\Scripts\python.exe evaluation\shuttle_tracknet_ab\run_tracknet_v3.py `
  --video "<原视频路径>" `
  --tracknet-root "C:\Code\TrackNetV3" `
  --tracknet-python "C:\Code\TrackNetV3\.venv\Scripts\python.exe" `
  --tracknet-checkpoint "C:\Code\TrackNetV3\ckpts\TrackNet_best.pt" `
  --inpaint-checkpoint "C:\Code\TrackNetV3\ckpts\InpaintNet_best.pt" `
  --output-dir outputs\tracknet_ab\match_a_segment_01
```

脚本会运行两次上游 `predict.py`：

- `tracknet_raw\<video>_ball.csv`：B，原始热图检测；
- `tracknet_rectified\<video>_ball.csv`：B*，修复后的轨迹。

原始 B 的 Good-Badminton 适配器默认以 96 帧为有界解码/预处理块；可传
`--chunk-frames 64` 降低峰值内存，代价是更多分块切换。该参数不改变模型的逐帧检测
频率或时间坐标。

## 3. 运行 A/B 评估

```powershell
.\.venv\Scripts\python.exe evaluation\shuttle_tracknet_ab\run_ab_benchmark.py `
  --video "<原视频路径>" `
  --baseline-detections "outputs\<当前分析目录>\detections.jsonl" `
  --tracknet-raw-csv "outputs\tracknet_ab\match_a_segment_01\tracknet_raw\<video>_ball.csv" `
  --tracknet-rectified-csv "outputs\tracknet_ab\match_a_segment_01\tracknet_rectified\<video>_ball.csv" `
  --annotations evaluation\shuttle_tracknet_ab\datasets\match_a_segment_01\annotations.jsonl `
  --metadata evaluation\shuttle_tracknet_ab\datasets\match_a_segment_01\metadata.json `
  --reference-terminals "outputs\<当前分析目录>\evaluation\rally_boundary_reference_user_review.jsonl" `
  --output-dir outputs\tracknet_ab\match_a_segment_01\benchmark
```

`--reference-terminals` 是可选的评估真值，永远不会回写为模型预测。

## 输出与通过条件

- `report.json`：原始检测召回、`not_visible` 假阳性、定位误差、最长连续原始漏检，以及可选的回合终止对比。
- `summary.csv`：A、B、B* 的核心横向指标。
- `a_current_yolo/`、`b_tracknet_raw/`：两组隔离的 `detections.jsonl` 副本和派生产物；源分析目录不被改写。
- `b_star_tracknet_rectified/`：B* 的只读复核证据，不运行得分/回合派生。
- `blocked.json`：标注、视频指纹或帧契约不满足时的明确原因。

首轮仅当 B 的**原始**检测在锁定测试片段上提升可见球召回、`not_visible` 假阳性没有明显变差、最长原始漏检缩短，且回合提前截断没有增加，才考虑接入主链。B* 的连续展示效果不构成接入证据。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests\test_tracknet_ab_benchmark.py -v
```
