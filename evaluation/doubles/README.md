# 固定机位双打追踪评测

此目录评测四项可验证能力：四人检测召回、`track_id` 切换、连续轨迹中断/遮挡恢复，以及击球归属。真值必须由人工按稳定 `person_id` 标注；预测只读取 `detections.jsonl` 的 `spatial.tracks` 和 `spatial.hit_events`。

`predicted` 和 `missing` 位置不会得到检测召回、遮挡恢复或击球归属的分数，避免把预测轨迹包装成检测事实。

标注 JSONL 的每行对应一个抽样源帧：

```json
{"frame":120,"view_id":"rear","players":[{"person_id":"p1","court_xy_m":[1.2,2.1]},{"person_id":"p2","court_xy_m":[4.8,2.0]},{"person_id":"p3","court_xy_m":[1.1,11.6]},{"person_id":"p4","court_xy_m":[4.9,11.4]}],"occluded_person_ids":[],"hits":[{"person_id":"p3"}]}
```

至少建立正后方与侧后方/斜后方两套**真实固定机位双打视频**标注，每套包含交叉、同队前后换位、网前遮挡、短暂漏检与恢复。仓库当前只有合成单元测试夹具，不能把它当作真实验收结论。

运行：

```powershell
.venv\Scripts\python.exe evaluation\doubles\run_doubles_evaluation.py `
  --annotations evaluation\doubles\datasets\rear\annotations.jsonl `
  --detections outputs\your_match\detections.jsonl `
  --output outputs\your_match\doubles_tracking_report.json
```
