# 流式双服务生产集成与兼容说明

## 当前结论

任务 F 已把任务 0～E 的独立模块接入同一条正式链路：

```text
本地录像/未来边缘摄像头
  → 业务侧 1～2 秒可独立解码分片
  → StreamSessionClient（回执、重试、ledger）
  → GPU API /api/v1/stream-sessions
  → OpenCVSegmentDecoder
  → AnalysisEngine（跨分片采样时钟/checkpoint）
  → person_only + 可选球检测
  → events.jsonl/checkpoint/end_to_end_trace
  → 业务侧轮询、拉事件并在业务数据库解释
```

旧 `/api/v1/jobs`、`AnalysisJobManager`、WebUI 完整上传与旧
`detections.jsonl` 均保留。流式会话是新增路径，不会强制替换开发和人工复核流程。

## 正式 API

所有路由复用 `GOOD_BADMINTON_API_KEY`，通过 `X-API-Key` 认证：

- `POST /api/v1/stream-sessions`：创建匿名视觉会话；
- `POST /api/v1/stream-sessions/{id}/segments/{index}`：上传分片并取得持久回执；
- `GET /api/v1/stream-sessions/{id}`：读取接收、队列、处理和候选轨迹状态；
- `GET /api/v1/stream-sessions/{id}/events`：按 `(source_time_sec,event_id)` 游标读取事件；
- `GET /api/v1/stream-sessions/{id}/trace`：读取 GPU 端阶段耗时；
- `POST /api/v1/stream-sessions/{id}/complete`：封口并等待/触发最终汇总；
- `DELETE /api/v1/stream-sessions/{id}`：取消；迟到的模型结果不能复活会话。

上传成功只表示分片已可靠持久化，不表示推理完成。业务侧必须保留
`delivery-ledger.json` 并轮询状态。

## GPU 运行配置

参考根目录 `.gpu-api.env.example`：

```dotenv
GOOD_BADMINTON_API_KEY=replace-with-secret
GOOD_BADMINTON_API_DATA_DIR=/root/good-badminton-gpu-api-state/api_data
GOOD_BADMINTON_STREAM_RETENTION_HOURS=72
GOOD_BADMINTON_STREAM_SEGMENT_TIMEOUT_SECONDS=10
GOOD_BADMINTON_STREAM_POSE_MODEL=/root/good-badminton-gpu-api-state/weights/yolo11n-pose.pt
GOOD_BADMINTON_STREAM_BALL_MODEL=/root/good-badminton-gpu-api-state/weights/yolo11s-ball.pt
GOOD_BADMINTON_STREAM_DEVICE=auto
GOOD_BADMINTON_STREAM_POSE_CONF=0.15
PORT=8080
```

`GOOD_BADMINTON_STREAM_SEGMENT_TIMEOUT_SECONDS` 是单个 GPU 分片的硬上限，
不是整场超时。默认流式分片为 2 秒，默认上限为 10 秒。普通解码、模型或追踪异常会被记录为该分片的失败证据，后续
分片继续处理；遇到无法从 Python 线程安全中断的 CUDA/OpenCV 卡死时，API
会先持久化失败记录并以受监督的退出码重启，再从下一分片和新的连续性 epoch
恢复。生产环境必须通过 `deploy/start_gpu_api_container.sh`（或同等的进程
监督器）启动，不能直接裸跑 uvicorn。

GPU 不保存、注册或自动检测球场标定。业务服务的相机配置保存
`camera_id`、`calibration_id`、`court_corners` 和配置版本；创建流会话时
携带四角，**每一个分片元数据也重复携带相同四角**。GPU 会拒绝与会话
四角不一致的分片。它仅把会话输入副本放进该会话 manifest/checkpoint，
用于恢复，不形成可被其他会话查询的全局相机记录。

本地 WebUI 的“GPU 服务地址（开发用）”可临时覆盖本次完整上传或模拟流式
回放的 GPU URL，例如 `http://xn-g.suanjiayun.com:55606`。模拟流式回放会将
该地址随业务任务持久化，状态轮询和后续分片仍访问同一地址；API Key 不进入
浏览器，仍由 WebUI/业务网关本机配置提供。生产环境不得把 GPU 地址开放给终端
用户填写，应由业务服务的部署配置或 allowlist 固定。

开发模式将“每次上传”视为一个临时 camera profile，由 WebUI 人工标四角；
生产模式按摄像头 ID 从业务数据库查出已验收的四角。摄像头移动、焦距、
裁切或输入分辨率变化时，业务侧创建新配置版本，后续会话带新四角。

业务服务使用：

```dotenv
GPU_ANALYSIS_BASE_URL=https://configured-gpu-service
GPU_ANALYSIS_API_KEY=the-same-secret
GPU_ANALYSIS_TIMEOUT_SECONDS=30
GPU_ANALYSIS_MAX_ATTEMPTS=5
GPU_ANALYSIS_INITIAL_BACKOFF_SECONDS=0.25
GPU_ANALYSIS_MAX_BACKOFF_SECONDS=5
```

不在代码中写死公网地址、映射端口或 `localhost:8080`。

## 三种球检测配置

- `shuttle_detector=none`：只运行人物姿态/跑位，是当前低延迟首选；
- `shuttle_detector=yolo`：在同一个 10/15/30Hz 测量帧运行 YOLO 球检测；
- `shuttle_detector=tracknet_v3`：只接受已注入的“有界状态、逐分片”的 temporal processor。

仓库现有 TrackNetV3 官方 runner 仍是完整文件批处理实现，不能冒充生产流式插件。
若未配置真实插件，API 在创建会话时明确拒绝；不会静默改用 YOLO，也不会捏造球数据。

流式 `generate_annotated_video=true` 同样会被明确拒绝。生产流式路径只保存数据，旧 WebUI 仍可使用完整上传生成标注视频。

## Track ID 与恢复边界

每个流会话在 `configuration` 中显式携带 `tracker_backend`、
`lock_match_roster` 和 `roster_stable_frames`，而不是读取 GPU 进程环境变量。业务端可选
`court_association` 或 `bytetrack`；在不允许比赛中途换人的场景中，`lock_match_roster=true`
会在连续稳定的在场人数后自动锁定匿名 roster。它不要求业务传单打/双打，也不把轨迹
强行绑定成用户、队伍或画面上下方身份；锁定后多出的检测作为未分配证据保留，不能静默
生成新球员 ID。

默认 `court_association` 追踪器可随 checkpoint 跨 API 进程恢复。ByteTrack 可显式开启，
并可跨正常分片持续工作；但当前 Ultralytics ByteTrack 没有稳定公开的完整状态序列化接口。因此：

- 同一 API 进程内：ByteTrack 正常跨分片；
- API 进程重启：会话明确进入 `interrupted_needs_rebuild`，不把新 ID 冒充旧 ID；
- 在 ByteTrack 跨进程恢复适配器和真实 ID-switch 门禁通过前，不把它设为默认。

## 数据共存与责任边界

完整上传模式继续产生原有 `metadata.json`、`detections.jsonl`、热力图及可选标注视频。
流式模式每场会话位于：

```text
api_data/stream_sessions/<analysis_session_id>/
  manifest.json
  segments/*.bin
  events.jsonl
  checkpoint.json
  end_to_end_trace.json
```

新事件仅包含匿名 `track_id`、球场坐标、姿态/球观测、置信度和
`detected/predicted/missing` 证据状态。`check_id`、用户、队伍、比分、胜负、认领和商业报告只进入业务服务器；GPU 请求遇到这些额外字段会直接返回 422，而不是静默丢弃。

旧 `detections.jsonl` 的只读证据解析已移到中立的
`good_badminton_contracts` 包；GPU 旧可视化与业务指标都依赖该包，二者不再互相导入。
`badminton_analysis.analysis.movement_metrics/performance_report` 只保留旧公开导入兼容层，GPU 主流程不会调用它们；拆分部署后业务代码应直接使用 `business_gateway`。

业务长期保存原始比赛索引、用户认领和商业派生结果。GPU 只短期保存匿名分片与视觉证据。
TTL 由 `GOOD_BADMINTON_STREAM_RETENTION_HOURS` 配置。清理只允许处理已到终态且
`updated_at` 超过 TTL 的单个会话目录；`StreamSessionManager.cleanup_expired_terminal_sessions()`
默认是 dry-run，运维确认候选后才可传 `dry_run=False`。运行中/排队会话永不作为候选。

## 耗时证据与串并行关系

GPU `end_to_end_trace.json` 记录：接收、队列等待、打开片段、组合的
解码/采样/模型/追踪、事件与 checkpoint 写入、finalize。业务回放目录中的同名文件再记录：
创建、等待/切片、上传、封口、首次状态回传，并嵌入 GPU trace。

当前执行关系：

- 单 GPU worker 对片段串行处理，保证事件/track 时钟连续；
- 同一片段内解码、采样、姿态、可选球检测与追踪目前串行；
- 业务上传片段 N+1 可与 GPU 处理片段 N 重叠；
- LLM/业务报告不在视频分析服务链路内。

录像回放必须保留 `streaming_slo_proven=false`。只有真实摄像头/边缘网关 → 业务服务 → GPU 的 15 分钟生产配置测试通过门禁后才能改为 true。

## 联调与验收

本地正式路由/传输兼容：

```powershell
& .\.venv\Scripts\python.exe -m unittest `
  tests.test_stream_production_routes `
  tests.test_stream_client_api_compat `
  tests.test_stream_sessions `
  tests.test_stream_runtime -v
```

录像分片回放：

```powershell
& .\.venv\Scripts\python.exe -m business_gateway.streaming.replay_cli `
  --video S:\path\match.mp4 `
  --create-request docs\contracts\stream_session_v1\examples\create_session_request.json `
  --work-dir .\stream_replay_data\match-001 `
  --idempotency-key match-20260823-000001 `
  --segment-seconds 2
```

任务 D 门禁：

```powershell
& .\.venv\Scripts\python.exe -m evaluation.streaming.baseline `
  --benchmark path\to\benchmark.json `
  --output path\to\gate.json
```

部署前仍必须补一轮真实 GPU/真实模型与至少 15 分钟输入的吞吐门禁；本地合成测试只能证明接口、状态、恢复和证据语义，不能证明生产速度。
