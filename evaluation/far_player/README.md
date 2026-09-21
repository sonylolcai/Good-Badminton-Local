# 固定机位远端球员评测基线

这个目录独立于主检测流程，用人工复核的 ground truth 对三种方案做同口径比较：

1. `full_640`：全画面，`imgsz=640`；
2. `full_1280`：全画面，`imgsz=1280`；
3. `full_640+far_roi_640`：全画面 640，加远端 ROI 二次 640，随后跨来源 NMS 合并。

评测不会用模型自身输出直接充当真值。标注未完成时，benchmark 会退出并生成 `blocked.json`，不会生成伪造的召回率。代理逐帧视觉复核的模型候选框可以作为 `preliminary_reviewed` 初步基线，但报告会保留来源，且不能通过正式 ByteTrack 接入门槛；人工复核后才成为 `human_reviewed`。

## 1. 发现候选视频

```powershell
.venv\Scripts\python.exe evaluation\far_player\discover_videos.py --root . --output outputs\far_player_candidates.json
```

输出会标出 852×480 候选和高于 852×480 的候选，但 `ground_truth_status` 始终要求人工确认。

## 2. 初始化人工标注集

分别为低分辨率视频和高分辨率对照视频运行：

```powershell
.venv\Scripts\python.exe evaluation\far_player\init_annotation_set.py `
  --video "<852x480视频路径>" `
  --output-dir evaluation\far_player\datasets\current_852x480 `
  --start-sec 60 `
  --duration-sec 2 `
  --sample-fps 10 `
  --max-samples 20 `
  --far-roi 0.0,0.0,1.0,0.55

.venv\Scripts\python.exe evaluation\far_player\init_annotation_set.py `
  --video videos\demo.mp4 `
  --output-dir evaluation\far_player\datasets\higher_resolution_control `
  --start-sec 2 `
  --duration-sec 2 `
  --sample-fps 10 `
  --max-samples 20 `
  --far-roi 0.0,0.0,1.0,0.55
```

命令提取复核帧，并生成 `metadata.json` 与故意保持 `unlabeled` 的 `annotations.jsonl`。
连续的 20 个 10Hz 样本覆盖约 2 秒，既便于人工复核，也能让“最长连续漏检”有明确时间意义。若希望检查全场稀疏分布，也可用 `--sample-fps 1 --max-samples 12`，或通过 `--frame-indices 0,30,60,...` 明确指定源帧。

## 3. 标注格式

每行对应一个抽样帧：

```json
{
  "frame_index": 0,
  "time_sec": 0.0,
  "image": "frames/frame_00000000.jpg",
  "label_status": "complete",
  "people_annotation_complete": true,
  "people": [
    {
      "id": "far_player_1",
      "role": "far_player",
      "visibility": "visible",
      "bbox_xyxy": [390, 118, 445, 245]
    },
    {
      "id": "near_player_1",
      "role": "other_player",
      "visibility": "visible",
      "bbox_xyxy": [382, 280, 490, 470]
    }
  ],
  "ignore_regions": []
}
```

规则：

- `bbox_xyxy` 使用原视频像素坐标，不是归一化坐标。
- `role` 可为 `far_player`、`other_player`、`non_player_person`。
- `non_player_person` 表示裁判、场边人员等干扰项；检测框落在这些人身上仍按“球员误检”统计。
- `visibility` 可为 `visible`、`partial`、`not_visible`；前两者进入召回率分母。
- 只有画面中所有人都已标注，才能设置 `people_annotation_complete=true`。否则误检数量不可计算。
- 看台、裁判或无法可靠判定的区域可加入 `ignore_regions`，中心落入其中的未匹配检测不计误检。
- 复核完成后才把 `label_status` 改为 `complete`。
- `annotation_template.example.jsonl` 只是格式示例，不是任何视频的真实标注。
- Benchmark 会校验同目录 `metadata.json` 中的视频 SHA-256；标注不能挪用到另一个视频。
- `metadata.json` 必须记录 `ground_truth_source`、`reviewer`、`review_status`。人工复核使用 `ground_truth_source=human_manual`；代理视觉复核的模型候选框须明确使用 `agent_visual_review_with_model_proposals`，不能伪装成人工真值。

校验：

```powershell
.venv\Scripts\python.exe evaluation\far_player\validate_annotations.py `
  --annotations evaluation\far_player\datasets\current_852x480\annotations.jsonl `
  --metadata evaluation\far_player\datasets\current_852x480\metadata.json
```

## 4. 运行三方案评测

```powershell
.venv\Scripts\python.exe evaluation\far_player\benchmark_far_player.py `
  --video "<852x480视频路径>" `
  --annotations evaluation\far_player\datasets\current_852x480\annotations.jsonl `
  --output-dir outputs\far_player_baseline\current_852x480 `
  --model weights\yolo11n-pose.pt `
  --confidence 0.25 `
  --inference-iou 0.7 `
  --matching-iou 0.3 `
  --far-roi 0.0,0.0,1.0,0.55
```

同样命令再对高分辨率视频运行一次。设备可用 `--device cpu` 或 `--device 0` 显式固定。

两个视频均完成后，生成默认方案与 ByteTrack 接入门槛结论：

```powershell
.venv\Scripts\python.exe evaluation\far_player\compare_baselines.py `
  --reports `
    outputs\far_player_baseline\current_852x480\report.json `
    outputs\far_player_baseline\higher_resolution_control\report.json `
  --output outputs\far_player_baseline\baseline_summary.json
```

默认方案先比较汇总召回率；差距不超过 1 个百分点时，选择平均耗时较低者，再以误检数打破平局。ByteTrack 前置门槛要求：两段不同视频、同时覆盖 852×480 和更高分辨率、每段至少 20 个远端真值框、每段召回率不低于 95%、最长漏检不超过 0.30 秒、误检可评估，并完成明确的人工复核。通过只表示“可以开始单独评测 ByteTrack”，不表示跟踪效果已经合格。

## 输出

- `report.json`：完整可复核结果；
- `summary.csv`：三方案横向表；
- `predictions_*.jsonl`：逐帧检测框、置信度、来源、关键点、实际 `imgsz` 和 ROI；
- `false_positives/<method>/*.jpg`：人工真值、ROI、检测框和误检框可视化；
- `blocked.json`：人工标注缺失或无效时的阻塞原因。

核心指标：远端召回率、漏检率、最长连续漏检抽样数/秒数、平均/中位/P95 单帧耗时、误检数及误检样本。报告同时记录视频 SHA-256、视频信息、标注 SHA-256、模型权重 SHA-256、Ultralytics 版本、阈值、输入尺寸、设备和 ROI。

计时前会分别预热 640、1280 和远端 ROI 路径；预热不计入单帧耗时。

最长连续漏检按连续“人工复核抽样帧”计算，同时换算为秒；它不是未经标注的每个源视频帧。

## 当前 preliminary 基线（2026-08-12）

两段视频各使用 12 个逐帧视觉复核样本；模型框只作为标注候选，不直接充当真值。由于尚未人工二次复核且每段少于 20 个样本，结果只用于选择下一步工程方案。

| 方案 | 852×480 召回/CPU均值 | 1280×720 召回/CPU均值 | 汇总召回/CPU均值 |
|---|---:|---:|---:|
| 全场 640 | 0% / 60.8ms | 0% / 59.3ms | 0% / 60.0ms |
| 全场 1280 | 91.7% / 136.3ms | 100% / 135.8ms | 95.8% / 136.0ms |
| 全场640 + 紧致远端ROI 640 | 75% / 115.6ms | 100% / 122.0ms | 87.5% / 118.8ms |

当前默认推荐 `full_1280`。`full_640+far_roi_640` 已证明可以独立运行，但当前低清召回率和误检情况仍不如全场1280；后续应从固定机位球场标定生成紧致ROI并继续调合并规则，不应凭单次观感替换模型。

ByteTrack gate 当前为 `false`：每个视频不足20个远端真值框、低清召回不足95%、最长抽样漏检为5秒、且尚未人工二次复核。完整结果见 `outputs/far_player_baseline/baseline_summary.json`。

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s evaluation\far_player\tests -v
```
