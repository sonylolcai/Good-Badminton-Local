# 固定机位比赛流式分析：10 分钟结果目标

## 目标与完成定义

对于一场最长 **15 分钟** 的固定机位比赛，最后一个视频分片被服务端确认接收后
10 分钟内，必须生成：

1. 不可变的球员/羽毛球原始检测、持续 `track_id`、明确的 `detected` / `predicted` /
   `missing` 状态；
2. 可信度明确的回合、击球和球路候选；
3. 每个视觉 `track_id` 的运动表现证据包，以及一次有状态的大模型报告请求；
4. 可供小程序业务服务保存、认领和展示的结果清单。

只有报告服务返回成功时，任务才可标记为“含 LLM 报告完成”。报告服务未配置、超时或
失败必须明确显示，不能用规则文本冒充大模型结论。

## 采样预算

| 数据 | 固定目标 | 15 分钟上限 | 原则 |
|---|---:|---:|---|
| 统一分析采样 | 10 / 15 / 30 Hz，默认 10 Hz | 10 Hz 时 9,000 个分析时刻 | YOLO Pose、YOLO 羽毛球、Track/roster、轨迹/回合派生和 JSONL 共用一个时间桶。30 FPS 的 10 Hz 为每 3 帧一次；60 FPS 为每 6 帧一次。 |
| TrackNetV3（可选） | 全时序模型 | 仅人工主动启用 | 它仍需连续 8 帧窗口才能保持已验证的模型语义；当前不假装它能靠后处理降采样获得同等速度。其输出只在统一分析时间桶进入主数据链路。 |
| 球场校验 | 开场标定 + 2 Hz 轻量健康检查 | 不重复全片模板匹配 | 固定机位失效时进入 `camera_invalid`，不继续解释运动数据。 |
| 报告 | 整场一次 LLM 请求 | 75 秒默认超时 | 一次请求包含全部 Track ID，不能按四名球员串行请求。 |

跳过的原始帧不会调用人物/球模型、不会更新 roster、不会写入 JSONL，也不会被误记为
“零人”或“球丢失”。关节关键点只记录在真正的统一分析测量帧，并保存其
`measurement_frame`。这使后续肢体分析不会把旧骨架错误视为当前动作。

`TrackNetV3` 是唯一明确列出的模型级例外：为保持官方 8 帧时序窗口，它在被主动启用时
仍进行连续时序推理；它绝不会被静默当成 10 Hz 快模型。生产默认 `YOLO` 或“不检测球”，
两者都完全遵守统一采样频率。

## 流式会话契约

当前 `/api/v1/jobs` 是“完整文件接收后再分析”的兼容接口，**不是**实时推理接口。
新流式路径应独立实现，避免破坏已有上传分析：

```text
POST /api/v1/stream-sessions
  -> session_id、接收确认、初始状态

POST /api/v1/stream-sessions/{session_id}/segments/{segment_index}
  -> 1–2 秒 fMP4 分片、source_start_time_sec、sha256
  -> 持久化接收确认；异步检测状态

GET /api/v1/stream-sessions/{session_id}
  -> 已接收/排队/已处理秒数、积压、Track ID 候选、各阶段耗时

POST /api/v1/stream-sessions/{session_id}/complete
  -> 封口、处理最后窗口、生成派生数据与大模型报告
```

相邻 TrackNet 分片需要保留 7 帧重叠（模型序列长度为 8），输出时只写入新分片拥有的
帧，防止重复球点。人体追踪器状态跨分片保存；服务重启后必须从已持久化的状态恢复或
显式标记 `interrupted_needs_rebuild`，绝不能静默换一个新的 Track ID。

## 赛前登记与赛后认领

小程序赛前可向业务服务登记两/四个 `check_id`。这些 ID 是**参赛资格和业务记录**，
不是视觉识别线索，GPU 服务不能用它们猜测画面人物。

1. 赛后端只保存 `match_registration.check_ids` 与比赛模式；
2. GPU 服务在开场获得稳定、完整的场上人数后锁定视觉 `track_001...`；
3. `GET session` 在分析仍继续时返回候选 Track ID、状态和置信度；
4. 比赛结束后用户选择“我是哪个 Track ID”；业务服务写入独立的人工绑定记录
   `check_id -> track_id`；
5. 不改写原始检测、不把当前半场/前后站位当作身份。

这样用户可在报告仍在生成时看到并认领候选轨迹，但个人统计只在绑定和覆盖率检查后
写入业务库。

## GPU 调度与验收

单张 3090 第一阶段将 Pose 与 TrackNet 按小批次交替运行，避免两个模型长期争用
24 GB 显存；视频解码、JSON 落盘和报告调用与 GPU 推理解耦。人体标注视频导出不应
阻塞数据完成：优先原视频 + 前端数据叠加，离线导出时使用 NVENC 分片编码。

| 指标 | 3090 验收目标 |
|---|---:|
| 2 秒分片端到端 P95 | <= 1.5 秒 |
| 开场 Track ID 候选出现 | 完整人数稳定后 <= 30 秒 |
| 比赛结束时待处理积压 | <= 360 秒 |
| 封口后的派生数据汇总 | <= 120 秒 |
| 一次 LLM 报告调用 | <= 75 秒或明确失败 |
| 最终尾部延迟 | <= 600 秒 |

用同一套协议先做“15 分钟录像按 2 秒分片实时投送”的压测，再接真实 RTSP/摄像头。
必须分别测试 1080p 30 FPS 与 1080p 60 FPS，并记录：每阶段 P50/P95、GPU 利用率、
显存、队列积压、漏检率、Track ID 中断和最终完成时刻。

如果单卡 3090 未通过，不应盲目降低所有数据频率。优先保持人体 10 Hz、球 30 Hz，
再升级到两张 4090：一张固定给 TrackNet，一张固定给 YOLO Pose/导出；5090 也沿用
相同会话协议和基准报告。只有基准显示人体覆盖率仍达标时，才允许再降低 Pose 频率。

## 批处理阶段计时契约

批处理接口的每个任务在 `GET /api/v1/jobs/{job_id}` 返回 `timing`。它不是根据总耗时
倒推的估算，而是由实际阶段开始、阶段心跳和阶段结束时刻组成的可持久化记录：

```text
queue_wait → analysis_bootstrap → preparing.*
  → tracknet.launch / checkpoint_load / video_decode / frame_preprocess
  → tracknet.model_initialize / inference（批次心跳）/ csv_export
  → human_tracking.* → post_processing.* → succeeded | failed | cancelled
```

每个阶段包含 `started_at`、`last_heartbeat_at`、结束后的 `elapsed_seconds`，TrackNet
推理还包含已完成/总批数和窗口数。轮询响应同时返回当前 `stage`、`stage_detail` 和完整
`timing`；WebUI 因而能区分“正在处理但尚未生成人体帧进度”与真正没有心跳的故障。

为避免监控反过来拖慢推理，人体逐帧回调不会再每帧重写完整任务 JSON；进度最多每
0.5 秒持久化一次（结束帧立即写入）。旧任务不具备此计时记录，不能将历史总时长
事后精确归因。

完成任务的 `performance_trace.json.execution.analysis_metrics` 还会记录每个组件的
`calls` 与 `elapsed_seconds`：视频解码、球场健康检查、Pose、YOLO 球、球轨迹、空间
Track/roster、旧数据兼容 PlayerTracker、JSONL 写入、可选绘制/编码，以及赛后派生。
业务端生成的 `end_to_end_trace.json.timeline.remote_gpu.component_metrics` 原样带回这份
证据；由此可区分 GPU 推理、CPU 跟踪、磁盘写入和后处理，而不是只看到一个总耗时。

第一次带此计时记录的 1080p 30 FPS 和 60 FPS 实测，才是调整 Pose 采样率、TrackNet
批量、FP16/TensorRT 或视频导出方案的基准；在此之前不得把当前完整文件批处理速度
当作流式 SLA 的证明。

### 端到端链路追踪

在业务端任务进入 `succeeded`、`failed`、`cancelled` 或 `interrupted_unconfirmed` 后，系统还会
写入不可变的 `end_to_end_trace.json`。它同时保存到：

```text
outputs/business_tasks/end_to_end_traces/<business_task_id>.json   # 耐久审计副本
outputs/remote_jobs/<run>/end_to_end_trace.json                    # 本次结果目录副本
```

它将业务端 `submission_started`、上传字节完成、远端接收回执、轮询、逐个产物下载和本地
元数据写入，与远端 `performance_trace.json` 的队列和模型阶段合并。每次分析都能直接回答：

| 当前实际关系 | 是否并行 | 对总耗时的意义 |
|---|---|---|
| 完整 MP4 上传 → GPU 入队 | 否 | 服务端收完整容器后才开始分析，不能边传边推理。 |
| GPU 队列/TrackNet/Pose/导出等顶层阶段 | 否 | 当前单 worker 以记录顺序执行，是主要计算关键路径。 |
| 浏览器/业务端状态轮询 | 是，但仅控制面 | 可显示进度，不缩短 GPU 推理。 |
| 远端完成 → 逐个产物下载 → 本地元数据写入 | 否 | 当前按 manifest 顺序下载，可从单个产物耗时判断是否值得并行。 |

`execution_topology` 只描述实际已执行的关系；其中的 `optimization_candidates` 只标记可测的
候选改造，绝不把尚未实现的分片并行或多 GPU 流水线写成已有能力。跨主机的绝对时差可能受
时钟误差影响，因此同一 GPU 主机内的阶段耗时才是模型优化的权威依据；端到端时长用于定位
上传、轮询、下载和落盘的链路瓶颈。

## 当前实现状态

- 已完成：批处理默认改为 Pose `960 + 统一 10 Hz`；YOLO 球、Track ID/roster、回合派生
  和 JSONL 与该频率同步；跳帧不会重置开场 roster；关键点与真实测量帧持久化到
  `spatial.tracks`；任务轮询可返回开场锁定的候选 Track ID；赛后生成可审计的
  表现证据包和一次受限 LLM 报告调用接口。
- 未完成：上述 `stream-sessions` 分片接收/跨分片模型状态、TrackNet 连续推理、15 分钟
  压测和服务器实测。因此当前批处理 API 不得被宣传为满足 10 分钟流式 SLA。

## 版本迭代性能门禁

每次修改模型、采样率、推理流程、视频导出、CUDA/TensorRT 或 GPU 部署脚本后，都必须
对一个完成任务的 `performance_trace.json` 运行版本化门禁：

```bash
bash deploy/run_performance_gate.sh /path/to/performance_trace.json
```

门禁锁定生产参数：Pose `960 + 统一分析 10 Hz`、YOLO 球，以及整场最多一次 LLM 请求；它还能
与同源视频的上一个 trace 比较每阶段回退。工具和报告格式见
[`evaluation/performance/README.md`](evaluation/performance/README.md)。

批处理 trace 只能给出真实阶段耗时，结果会明确标为 `warn`，不能证明流式 SLO。等分片
接口实现后，必须额外提供 1--2 秒分片回放报告；只有该回放通过 P95 分片、队列积压、封口
汇总、LLM 和最终尾部延迟预算时，报告才会标记 `streaming_slo_proven=true`。
