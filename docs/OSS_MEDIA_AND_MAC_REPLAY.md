# 分片完整性、OSS 归档与 Mac 连续回放

2026-10-23。本文记录本地补充的实现与部署步骤。真实 CPU 与 Apple MPS 已完成短合成视频的模型功能对照；不能作为远端部署、真实 OSS、CUDA 或真实比赛质量验收证明。

## 本轮审核结论和修复

原流程已有可靠接收、分片回执和流式引擎，但没有独立的录像完整性判定、GPU 成片和 OSS 归档，也没有 Mac 按 OSS 清单连续运行同一引擎的入口。

另外修复两处会阻碍新流程的问题：

- `StreamSessionManager.complete` 的立即结束分支会调用重新读盘的 finalizer，却未先保存封存状态，导致部分路径丢失 `sealed`、最终索引和 `allow_partial`。现在先持久化输入边界再结束。
- 场馆网关只在 TrackNet 模式探测和上传源帧身份与逐帧时间。根目录和 Windows 入口现在在所有检测模式执行此探测。探测失败仍报告未知，绝不推算成已验证；旧会话不伪造历史帧编号。Windows、Linux、macOS 分发包由同一源重新生成。

## 两条独立的流水线

```text
场馆分片 -> GPU 原始字节与回执持久化
                   |
                   +-> 流式引擎：analysis 或 decode_smoke
                   |
                   +-> 封存后媒体 worker：逐片验证 -> 无损拼接 -> 成片验证 -> 私有 OSS
                                                                               |
                                                                               +-> Mac 按清单连续回放
```

媒体 worker 不等待模型推理终态。即使推理失败或积压，也可以独立保存完整录像；媒体就绪与分析终态分别查询。现有业务端仍按原有分析完成流程回收结果，本轮没有把运营 Web 改成独立的录像交付界面。

首次测试建议在独立数据目录部署 `decode_smoke`：消费真实解码帧，运行现有 AnalysisEngine 的跨片生命周期与 checkpoint，不加载模型、不产生假人物或球观测。健康与运行记录明确标记此模式。它只能证明解码/引擎流程，不能证明 YOLO 或 TrackNet 推理。`analysis` 模式才使用原有真实模型。

## 云端验证和归档规则

媒体 worker 自动扫描已封存会话，逐场串行工作；当前部署要求一个 Uvicorn worker/进程、一个持久数据目录，不能让多个归档进程同时写同一目录。

1. 按 `expected_last_segment_index` 清点从 0 开始的所有索引，包含头片和尾片；空会话或任何缺片不发布完整录像。
2. 校验持久字节的长度与 SHA-256；ffprobe 检查独立起始关键帧、逐帧 PTS 和流参数，FFmpeg 全量解码视频并生成帧摘要，有音频时也执行音频解码。
3. 核对声明的源帧数、源帧起点、逐帧时间；跨片帧编号与源时间必须连续。源时间与媒体时间误差最多 2 ms。内部大于标称帧间隔 1.5 倍的跳变拒绝作为完整输入；当前严格验收面向连续固定帧率摄像头，任意可变帧率素材需另外评估。
4. 核对流结构、编码、分辨率、像素格式、time base 与编码附加数据。有漂移就报错，保留原字节，不自动转码掩盖问题。
5. FFmpeg concat + `-c copy` 生成 `master.mp4`，再全量解码；逐帧比较它与有序原分片的画面摘要，检查逐帧时间及总时长。允许正常短尾片。
6. 上传原分片、完整录像和完整性报告，最后上传 `manifest.json`。清单是提交标记；前面任一步失败都不能发布 ready 清单。

这是严格“完整录像”通道：`partial` 分析状态本身不阻止媒体归档，但实际缺片、未知时间或损坏字节会令媒体状态为 `invalid`。当前不生成带缺口的伪完整成片，也没有新增封存后修复/替换能力。缺片需使用现有上传/修复协议在封存前解决；损坏替换的授权与场馆自动修复执行器仍是后续工作，不能把本轮的拒绝机制写成已自动补齐。

旧版 Aliyun/无媒体身份的本地测试目录为 `<prefix>/<sport_id>/<session_id>/<输入版本SHA256>/`。新版业务会话和多吉云采用 [业务媒体目录规范](MEDIA_STORAGE_CONTRACT.md) 的 `testVideo/球馆/摄像头/日期/场次_媒体UUID/`。二者均包含：

- `segments/00000000.mp4` 等：原始分片，字节不变。
- `master.mp4`：完整录像。
- `integrity.json`：分片解码、连续性和成片核验报告。
- `manifest.json`：源时间、原始帧身份、标定、配置、对象 key、SHA-256、完整性报告引用。

相较原来的“只存成片”设计，这个验证阶段同时保留原始分片，为 Mac/服务器分片等价回放提供证据，视频存储量约增加一份；后续根据留存期限配置 OSS 生命周期。并未自动建立 Bucket 或修改其生命周期。

对象使用私有权限，SDK V4 环境凭证、CRC、断点上传/下载；上传后核对 HEAD 长度及摘要元数据，Mac 下载后再次计算 SHA-256。清单不含 URL 凭证或密钥，不通过 OSS 列表顺序处理分片。

失败后持久保留本地证据并自动重试网络错误；服务重启后继续扫描。`invalid` 不反复重跑，管理员可明确请求重试。启用归档时，TTL 清理不会删除还没有 ready 回执的媒体；原有 `persistent_media` 策略仍禁止 TTL 整体清理；后来新增的已验证视频副本释放、单场上限及磁盘保护见 [半小时缓存规则](MEDIA_STORAGE_CONTRACT.md)。

## 云端配置

先在现有 CUDA Python 环境安装 `python -m pip install -r requirements-media.txt`，系统安装 FFmpeg/ffprobe。不要用本仓原来的 CPU 固定版 requirements 覆盖 CUDA/Mac 的 PyTorch。

在 GPU 服务的私有环境文件配置（示例见 `.gpu-api.env.example`）：

```text
GOOD_BADMINTON_STREAM_PROCESSING_MODE=decode_smoke
GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED=1
GOOD_BADMINTON_API_DATA_DIR=/持久卷/独立的媒体测试目录
GOOD_BADMINTON_STREAM_MIN_FREE_BYTES=2147483648
GOOD_BADMINTON_OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
GOOD_BADMINTON_OSS_REGION=cn-hangzhou
GOOD_BADMINTON_OSS_BUCKET=实际私有Bucket
GOOD_BADMINTON_OSS_PREFIX=match-media
```

SDK 从环境读取 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`，STS 时再配置 `OSS_SESSION_TOKEN`。GPU 账号需要指定前缀的上传、分片续传、对象私有 ACL 与读取 HEAD 权限；Mac 使用该前缀的只读凭证。SDK 用法参照[阿里云 V4 初始化说明](https://www.alibabacloud.com/help/en/oss/initialization-2)与[断点续传说明](https://www.alibabacloud.com/help/en/oss/developer-reference/resumable-upload-1)。

`analysis` 与 `decode_smoke` 模式不可在活动会话中切换；改模式使用新的测试数据目录与新会话，防止错误恢复旧 checkpoint。启动脚本在 smoke 模式不要求姿态权重。健康应显示 `stream_processing_mode=decode_smoke`、`media_archive_enabled=true` 和 `media_archive_running=true`。

沿用原设备上传与 complete 合同，新增三个管理端接口，均要求 `X-API-Key`，上传 token 不具有读取权限：

```text
GET  /api/v1/stream-sessions/{id}/media
POST /api/v1/stream-sessions/{id}/media/prepare
GET  /api/v1/stream-sessions/{id}/media/manifest
```

`prepare` 只排队，不在 HTTP 请求内拼接；未封存为 409。状态为 `awaiting_seal / pending / checking / assembling / uploading / ready / invalid / failed`。状态 ready 后返回 `manifest_object_key` 与 `manifest_sha256`，提供给 Mac。完整清单尚未就绪时 manifest 接口为 409。

首次上线顺序：GPU/OSS 测试配置 -> 同步更新场馆网关 -> 一路新采集会话 -> 停止采集并完成封存 -> 查媒体 ready -> Mac 下载回放。直传开关、业务服务版本和网络前提仍按原直传设计核对；本轮没有直接开启远端直传或部署服务。

## Mac 配置与运行

在原生 Python 环境安装项目所需的 OpenCV/PyTorch/Ultralytics 等与 `requirements-media.txt`。用 `torch.backends.mps.is_available()` 检测本机 MPS，模型权重单独配置；`auto` 可以选择 CPU，因此运行记录要核对实际后端。明确 `--device mps` 可防止把 CPU 运行误认作 Mac GPU 验收。

设置同一套 OSS endpoint、region、Bucket 和只读环境凭证。以下命令均从本仓根目录运行，替换 key、hash、输出目录：

```sh
# 先仅验证下载、真实解码和共享引擎生命周期。
python -m api.oss_replay --manifest-key '实际清单对象key' \
  --manifest-sha256 '管理接口返回的64位摘要' \
  --output '/本地/新建的smoke运行目录' --processing-mode decode_smoke

# 人物/球模型使用 manifest 中冻结的配置，不在 Mac 偷换成另一套批处理算法。
python -m api.oss_replay --manifest-key '实际清单对象key' \
  --manifest-sha256 '管理接口返回的64位摘要' \
  --output '/本地/新建的分析运行目录' --device mps --source fragments

# 从同一完整录像读取，仍按原分片边界与源时间喂给同一引擎，作一致性对照。
python -m api.oss_replay --manifest-key '实际清单对象key' \
  --manifest-sha256 '管理接口返回的64位摘要' \
  --output '/本地/另一新建运行目录' --device mps --source master
```

默认分片模式按清单下载/预取一片、核验后分析，跨片保留一个模型和跟踪器；不因下载速度改变采样。完整录像模式下载 master，用原始清单恢复每帧源时间和分片边界，无再次编码。支持 `--cache-dir` 复用校验通过的输入；共享缓存不要多进程同时写同一目标。每轮输出新目录，禁止覆盖旧实验。断点下载可复用缓存和 SDK checkpoint；算法运行中断当前要求新建一次完整重放，不冒充已实现推理断点续跑。

输出包括 `manifest.json`、`events.jsonl`、每片 `checkpoint.json`、`run.json`。运行记录包含配置、代码内容摘要、依赖版本、权重摘要、设备与是否实际执行模型。模型证据仅包含视觉检测/轨迹；本轮没有把候选击球或完整教练报告新增为 GPU 的承诺。

## 本地验证和边界

```sh
.venv/bin/python -m unittest tests.test_media_archive tests.test_media_storage -v
.venv/bin/python -m unittest tests.test_gpu_api tests.test_gpu_sport_profiles tests.test_gpu_pure_launchers tests.test_stream_sessions tests.test_stream_repair_window tests.test_stream_production_routes tests.test_stream_runtime tests.test_gpu_direct_upload -v
```

真实 H.264 的 3 片（10、10、3 帧，含短尾片）经过现有接收管理器、FFmpeg 完整验证、无损拼接与显式本地对象存储，覆盖乱序、缺片、坏字节、不可解码、参数漂移、上传失败/重启重试、鉴权、缓存校验与 TTL 保护。比较 Mac 完整录像/分片及服务端引擎的语义证据时，只忽略实际事件产生墙钟 `emitted_at`。

另外分别在 CPU 与真实 Apple MPS 上运行了 YOLO Pose + TrackNetV3：每个后端上，同一合成输入的 master/fragment 两条路径输出语义事件一致。证据分别在工作区 `outputs/media-archive-20261023/real-models/verification.json` 与 `outputs/media-archive-20261023/mps/verification.json`。它证明模型入口可执行与这一个输入在各自后端内的分组一致性，不证明真实比赛准确率或跨硬件一致性。沙箱内的 MPS 探测为 false，获得执行环境许可后在沙箱外完成实际 Apple GPU 测试；另一个仅装 PyTorch 的 `.venv-mps312` 缺少 OpenCV，最终使用依赖完整的项目 `.venv` 完成验证。

本轮新增媒体/存储 20 项通过，GPU 原有相关回归 105 项通过；场馆根测试 28 项和 Windows 共享逻辑/传输测试 18 项通过，后者在 Mac 上执行，不代表 Windows 现场安装完成。

尚需远端验收：真实私有 OSS 往返与凭证权限、RTX 4090 CUDA 和 Mac MPS 对照、真实摄像头 30 分钟视频、并发吞吐与磁盘预算。OSS SDK 已在项目 `.venv` 安装并完成离线接口契约测试，不替代 OSS 在线验收。默认归档开关关闭。
