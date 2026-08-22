# 性能预算门禁

这套工具把“视频分析应该够快”变成版本化、可审计的约束。它不执行模型推理；它读取
GPU 服务生成的不可变 `performance_trace.json`，验证生产参数并比较阶段回退。

当前生产参数锁定 `YOLO Pose imgsz=960`、统一分析频率 `10 Hz` 与 `YOLO` 羽毛球。
统一频率会同时约束 Pose、YOLO 球、持续 Track ID/roster、回合派生和 JSONL；不再只检查
Pose 的采样率。trace 内的 `execution.analysis_metrics.components` 还会给出这些组件实际
调用次数和耗时，便于在每次迭代后定位回退来自 GPU 推理、CPU 跟踪还是写入/后处理。

## 日常使用

完成一次远端 GPU 任务后，在服务器仓库内运行：

```bash
bash deploy/run_performance_gate.sh \
  /root/good-badminton-gpu-api-state/api_data/jobs/<job_id>/output/performance_trace.json
```

结果会写在同目录的 `performance_gate.json`。

- `pass`：参数、回退比较和（若提供）流式回放指标均通过。
- `warn`：结果可用于观察，但不足以证明流式 SLA；批处理 trace 默认属于这一类。
- `fail`：生产参数漂移、同源阶段性能回退过大，或流式指标超出预算。

同一个输入视频、同一参数、同一 GPU 上的上一份 trace 可作为基线：

```bash
bash deploy/run_performance_gate.sh current/performance_trace.json baseline/performance_trace.json
```

## 流式证明的额外要求

完整文件批处理不能证明流式能力。等 `stream-sessions` 实现后，压测器应额外写入
`good_badminton_stream_replay_benchmark` JSON，并作为第三个参数传入。它至少包含：

```json
{
  "kind": "good_badminton_stream_replay_benchmark",
  "segments": {"p95_end_to_end_seconds": 1.2},
  "queue": {"backlog_seconds_at_video_end": 120},
  "completion": {"finalize_seconds": 50, "final_tail_seconds": 180},
  "llm": {"request_count": 1, "elapsed_seconds": 12}
}
```

只有这类 1--2 秒分片回放通过时，门禁报告才会把 `streaming_slo_proven` 设为 `true`。
这样不会把短视频批处理或静态日志误当成 15 分钟比赛的流式能力证明。
