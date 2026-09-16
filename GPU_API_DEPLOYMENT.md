# GPU 视频分析 API 部署与联调

## 项目进度（2026-08-13）

### 已完成

- GPU 分析 API 已实现：受 API Key 保护的异步单队列任务、任务查询、结果 JSON 与分析产物下载均已具备。
- 固定机位单打的球场坐标分析已进入当前代码分支；服务可接收人工球场四角并输出分析产物。
- 面向受限公网 GPU 容器的部署路径已验证：可复用镜像自带 CUDA PyTorch、通过上传源码包和模型权重离线部署、以 `uvicorn` 后台进程运行。
- API 容器监听端口已统一为 `8080`；部署文档记录的算家云实例曾通过健康检查和公网映射验证。
- WebUI 采用远端 GPU 优先：将视频以受限内存的 multipart 分块写入远端 API，轮询任务并下载产物；远端提交、轮询或下载失败时才本地分析，并在 `metadata.json.execution` 标记 `local_fallback` 和失败原因。
- 当前分支包含 GPU API、离线权重供应与部署脚本修复；当前提交为 `a922066`。

### 正在进行 / P0

- 固定机位下的远端球员稳定检测与持续追踪。当前它是后续双打、姿态分析、跑位统计、击球归属和能力画像的前置阻塞项。
- 需要先完成固定测试集上的远端 ROI 二次检测、输入尺寸对比、人体降级定位、漏检/中断/误检指标与验收门槛；在此之前不训练或替换模型。

### 尚未开始或未完成

- 持久 `track_id` 驱动的通用球员模型，以及未来单双打可扩展的身份、队伍和分区模型。
- 羽毛球近似三维轨迹、击球事件、回合状态机和高置信得分判断；低置信回合必须保持 `unknown`，不得进入胜负或能力统计。
- 业务服务器联调：比赛、用户/球员认领、确认赛果、社区互动、数据库和对象存储仍不属于 GPU API 的实现范围。
- 远端生产实例是否仍在线、模型权重是否已上传、以及最新提交是否已部署，尚未在本次更新中重新连线验证；上线前必须重新执行健康检查和一次真实任务验收。

### 下一步

1. 在固定机位测试集上完成远端球员 P0 基线与二次检测方案的量化评测。
2. 仅当远端人体检测达到验收门槛后，接入持久 `track_id` 跟踪并验证单打身份连续性。
3. 将已验证代码和权重以带提交号的源码包发布到 GPU 实例，在临时端口完成真实任务验收后再切换正式服务。

### 发布纪律

- GPU 部署只发布已提交的源码；当前工作区未跟踪的社区策略文档不属于 GPU API 发布内容。
- 每次部署同时记录代码提交、权重文件名与 SHA-256、运行配置和验收结果。
- 业务结果不得覆盖原始 `detections.jsonl`；由业务服务保存任务关联、用户确认和可展示的聚合数据。

## 服务边界

GPU 服务器仅负责：接收视频/球场模板、运行模型、保存分析产物并返回任务 JSON。用户、SSO、比赛业务记录、运动员档案、复核数据库和查询接口仍由业务服务器负责。

```text
业务服务器 / WebUI → GPU API（提交任务、轮询、取结果）
                         └→ 本地磁盘：输入、视频、detections.jsonl、metadata、汇总 JSON
```

API 是异步单队列：一张 24 GB 显卡一次只处理一场比赛，避免多任务同时加载姿态/球模型导致显存耗尽。它不提供公网匿名计算；除健康检查外所有接口必须携带 `X-API-Key`。

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 服务存活、鉴权配置、工作线程状态 |
| `POST` | `/api/v1/jobs` | 上传视频、球场模板、人工四角，创建分析任务 |
| `GET` | `/api/v1/jobs/{job_id}` | 查询排队/运行进度/错误 |
| `GET` | `/api/v1/jobs/{job_id}/result` | 已成功任务的 JSON 结果与产物 URL |
| `GET` | `/api/v1/jobs/{job_id}/artifacts/{name}` | 下载标注视频、元数据、检测 JSONL、热力图等 |

### 可靠提交与状态账本

不能把视频请求发出后就假定 GPU 已收到。业务端必须先创建自己的
`business_task_id`，并把它放在 `X-Idempotency-Key` 请求头中。GPU 只有在
上传文件已落盘、`job.json` 已持久化并已进入单 GPU 队列后，才返回 HTTP 202 和
接收回执：

```json
{
  "job_id": "…",
  "status": "queued",
  "receipt": {
    "accepted": true,
    "accepted_at": "…",
    "reused": false,
    "status_url": "/api/v1/jobs/…",
    "result_url": "/api/v1/jobs/…/result",
    "poll_after_seconds": 2
  }
}
```

只有收到该回执，业务端才可将任务标为 `accepted`；网络超时重试相同
`X-Idempotency-Key` 会返回同一个 GPU `job_id`，不会重复执行。GPU 的 `job.json`
会保存 `accepted → queued → running → succeeded/failed` 的 `state_history`。

当前本地 WebUI 作为业务端验证实现，会在 `outputs/business_tasks/<business_task_id>.json`
保存：发送开始、上传进度、GPU 接收回执、每次轮询的帧进度、下载结果、远端错误与
本地兜底。该账本不含 API Key。正式业务服务器应使用同一结构写入其数据库，并在服务
重启后按已保存的 `job_id` 继续轮询；前端只读取业务服务器记录，不直接信任 GPU 状态。
若业务进程在收到回执前中断，则用
`GET /api/v1/jobs/by-idempotency/{business_task_id}` 恢复同一个任务，绝不盲目重传视频。
本地验证可执行 `python -m webui.reconcile_remote_tasks`；它只轮询并下载已完成产物，可由
业务服务器的定时任务安全地每 2 秒调用一次。

`POST /api/v1/jobs` 使用 `multipart/form-data`：

- `video`：MP4、MOV、MKV、AVI 或 WebM；当前上限 10 GiB。
- `template`：同机位的球场模板图片。
- `court_corners`：例如 `[[120,210],[1035,209],[1150,700],[35,700]]`。
- `options_json`：可选 JSON；支持推理尺寸、人体置信度、远端 ROI、匿名骨架视频等，不接受用户传入任意模型路径。

请求示例：

```bash
curl -X POST http://GPU_HOST:PUBLIC_PORT/api/v1/jobs \
  -H "X-API-Key: $GOOD_BADMINTON_API_KEY" \
  -F "video=@match.mp4" \
  -F "template=@court.png" \
  -F 'court_corners=[[120,210],[1035,209],[1150,700],[35,700]]' \
  -F 'options_json={"pose_imgsz":1280,"generate_annotated_video":false,"browser_video_reencode":false,"audio":false}'
```

成功后轮询 `/api/v1/jobs/{job_id}`。`succeeded` 后调用 `/result`，始终返回 `metadata`、`detections` 和（生成时）`spatial_match_summary` 的受保护下载 URL；只有显式传入 `generate_annotated_video:true` 时才会附带 `annotated_video`。`browser_video_reencode` 默认为 `false`，且仅在已生成标注视频时生效。

## WebUI 远端优先模式

WebUI 本身不暴露 API Key 给浏览器。应在运行 WebUI 的后端进程环境中设置：

```bash
GOOD_BADMINTON_GPU_API_URL=http://xn-g.suanjiayun.com:52028
GOOD_BADMINTON_GPU_API_KEY=<从 GPU 实例 .gpu-api.env 读取的密钥>
```

也可将这些值（尤其是 API Key）写到 WebUI 项目根目录的 gitignored 文件 `.webui-remote-gpu.env`；以 `.webui-remote-gpu.env.example` 为模板。显式环境变量优先于该文件。不要把这个文件上传到 GPU 实例或提交 Git。

点击“解析视频”后，WebUI 先以 HTTP multipart 将本地文件持续分块写入 GPU API，随后轮询远端任务并下载标注视频、JSONL、元数据与热力图到本机 `outputs/remote_jobs/`。远端任一步骤异常时，才自动切换到已有的本地 `run_analysis`，并写入：

```json
{
  "execution": {
    "mode": "local_fallback",
    "fallback_used": true,
    "remote_failure": "..."
  }
}
```

远端成功时该字段为 `mode: remote_gpu`，含远端 `job_id`。这使结果可追溯，避免把本地兜底误认为 GPU 分析。

### 流式传输与边传边解析的边界

现在已经是**分块传输**：客户端不会把多 GB 视频整体读进 WebUI 内存再上传。但 GPU API 仍需接收完整的 MP4/MOV/MKV 等容器文件后才能创建分析任务；因此当前不是“上传第 1 秒视频时就解析第 1 秒”。

真正的边传边解析需要独立的实时会话协议：摄像头产生可独立解码的短 GOP 分片（推荐 1–2 秒 fMP4 或 WebRTC）、服务器维护跨分片的追踪状态/时间戳、迟到分片处理和最终回合汇总。它不能只靠当前完整文件 API 改成 chunked upload 完成。首版批处理仍保留完整视频上传，以保证标定、轨迹和热力图汇总一致；后续可在不破坏 `/jobs` 的前提下新增 `/stream-sessions`。

## 服务器安装

前置条件：NVIDIA 驱动、Python 3、Git、FFmpeg、可使用 `sudo` 的 Linux 用户。安装脚本会按驱动版本自动安装 PyTorch CUDA 12.1（驱动 535+）或 CUDA 12.4（驱动 550+）wheel；不要求镜像工具包标签恰好为 CUDA 12.4。先将当前分支推送到你的 fork：

```powershell
git push -u origin fixed-camera-singles-spatial-tracking
```

在 GPU 实例内执行：

```bash
git clone --branch fixed-camera-singles-spatial-tracking https://github.com/sonylolcai/Good-Badminton-Local.git ~/good-badminton
cd ~/good-badminton
chmod +x deploy/install_gpu_api.sh
./deploy/install_gpu_api.sh ~/good-badminton fixed-camera-singles-spatial-tracking
```

### GitHub HTTPS 受限时：官方源码包兜底

若实例无法连接 `github.com:443`，先测试官方下载域名：

```bash
curl -IL --connect-timeout 10 --max-time 20 \
  https://codeload.github.com/sonylolcai/Good-Badminton-Local/zip/refs/heads/fixed-camera-singles-spatial-tracking
```

能返回 `200` 或 `302` 时，使用同一分支的源码包部署（不需要 Git，也不会跳转到任何第三方镜像）：

```bash
test ! -e ~/good-badminton-source || { echo "~/good-badminton-source already exists; choose a new empty directory."; exit 1; }
curl -fL --retry 2 \
  https://codeload.github.com/sonylolcai/Good-Badminton-Local/zip/refs/heads/fixed-camera-singles-spatial-tracking \
  -o /tmp/good-badminton-source.zip
unzip -q /tmp/good-badminton-source.zip -d ~
mv ~/Good-Badminton-Local-fixed-camera-singles-spatial-tracking ~/good-badminton-source
cd ~/good-badminton-source
chmod +x deploy/install_gpu_api.sh
GOOD_BADMINTON_SKIP_GIT_SYNC=1 ./deploy/install_gpu_api.sh ~/good-badminton-source fixed-camera-singles-spatial-tracking
```

若 `codeload.github.com` 同样无法访问，不要使用不受控的第三方 GitHub 镜像。请通过云平台的文件上传功能上传本分支的源码包，或为实例配置平台提供的 HTTP/HTTPS 代理，然后重复该流程。

脚本会安装与驱动兼容的 CUDA PyTorch、其他项目依赖、创建只允许当前用户读取的 `.gpu-api.env` 并运行 API 测试。存在可用 systemd 时，它会配置 systemd 服务；多数租赁 GPU 容器没有 systemd 时，则自动以后台 `uvicorn` 进程启动，并在项目目录记录 `.gpu-api.pid` 和 `gpu-api.log`。密钥只存在 `.gpu-api.env`，不要提交、截图或发到聊天中。

### 无公网但镜像已自带 GPU PyTorch

一些租赁实例禁止访问 `download.pytorch.org`，却已有可用的 CUDA PyTorch。先确认 `python3 -c 'import torch; print(torch.cuda.is_available())'` 输出 `True`。将其余 Linux x86_64 / Python 3.12 依赖 wheel 上传到一个目录（例如 `/root/good-badminton-wheelhouse`）后，执行：

```bash
GOOD_BADMINTON_SKIP_GIT_SYNC=1 \
GOOD_BADMINTON_USE_SYSTEM_TORCH=1 \
GOOD_BADMINTON_WHEELHOUSE=/root/good-badminton-wheelhouse \
./deploy/install_gpu_api.sh /root/good-badminton-source fixed-camera-singles-spatial-tracking
```

此模式不会创建隔离 venv 或下载/覆盖镜像的 PyTorch；其他依赖仅从上传的 wheel 目录安装，缺包会明确失败而不会访问公网。

启动后在实例内检查：

```bash
curl http://127.0.0.1:8080/api/v1/health
sudo journalctl -u good-badminton-gpu-api -f
```

容器模式请改为：

```bash
tail -f /root/good-badminton-source/gpu-api.log
```

## 业务服务器联调

业务服务器不写 GPU 服务器的数据库。它保存 `job_id`、自己业务侧的比赛 ID、任务状态和 API Key；处理完成后下载 JSON/视频到对象存储或自身存储，再入库业务索引。`detections.jsonl` 是原始模型输出，不可被业务层覆盖。

公开端口前至少完成其一：只允许业务服务器 IP 访问映射后的 API 端口，或让 API 仅绑定 `127.0.0.1` 并经 SSH/VPN/反向代理访问。API Key 是访问控制，不是 HTTPS；跨公网调用应由反向代理提供 TLS。

## 已验证的算家云部署记录（2026-08-13）

| 项目 | 已验证配置 |
| --- | --- |
| 实例 | RTX 3090 24 GiB，NVIDIA 驱动 535.154.05 |
| 系统 Python | Conda Python 3.12.7；自带 `torch 2.5.1`、CUDA 12.4，`torch.cuda.is_available()` 为 `True` |
| API 监听 | 容器内 `0.0.0.0:8080`，由 `python3 -m uvicorn api.app:app` 启动 |
| 公网映射 | `http://xn-g.suanjiayun.com:52028` → 容器 `8080/TCP` |
| 公网验收 | `GET /api/v1/health` 返回 HTTP 200 和 `status: ok`；根路径 `/` 返回 404 属于预期 |
| 鉴权 | 除健康检查外，接口均需要 `.gpu-api.env` 中 API Key 对应的 `X-API-Key` |
| 网络限制 | 实例无法访问 GitHub、codeload、PyTorch wheel 官方站和 APT 源；普通 pip 依赖可通过镜像安装 |

### 当前环境的安全措施

将公网 `52028/TCP` 的入站来源限制为业务服务器的公网 IP。不要把 `.gpu-api.env` 上传、提交或发送到聊天中。每次修改端口映射后，从业务服务器复验：

```bash
curl -fsS http://xn-g.suanjiayun.com:52028/api/v1/health
```

返回 `status: ok` 说明公网 `52028` 已转发到容器内 `8080`。启动命令和 `.gpu-api.env` 的 `PORT` 必须保持为 `8080`；不要将程序改回 `8001`。

### 固定目录的一键覆盖升级（当前标准）

该实例无法从 GitHub 拉取分支，因此不能直接 `git pull`。从现在起不再保留一串
`good-badminton-source-<commit>` 目录，也不在服务器手工打补丁。统一使用下面三个
固定路径：

| 路径 | 作用 | 覆盖升级时是否删除 |
| --- | --- | --- |
| `/root/good-badminton-gpu-api-upload.zip` | 每次从 Windows 上传的固定包名 | 被新上传文件替换 |
| `/root/good-badminton-gpu-api` | 当前运行的应用代码、日志和 PID | 会整体删除后重新解压 |
| `/root/good-badminton-gpu-api-state` | API Key、`api_data/` 任务与产物、`weights/` 权重 | **绝不删除** |

本地先从当前工作区制作上传包。该命令会包含所有已跟踪代码的当前修改（即使还没有
提交），但不会把 `.gpu-api.env`、模型权重、虚拟环境、任务数据或未跟踪实验文件放进包：

```powershell
cd 'S:\Code Base\MiniProgram\bdTeach\Good-Badminton'
powershell -ExecutionPolicy Bypass -File .\deploy\package_gpu_api.ps1
```

把生成的 `deploy\good-badminton-gpu-api-upload.zip` 上传到服务器固定位置：

```text
/root/good-badminton-gpu-api-upload.zip
```

然后在服务器运行唯一的升级命令：

```bash
bash /root/good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh
```

若这是第一次切换到固定目录（即 `/root/good-badminton-gpu-api` 还不存在），从固定
ZIP 中直接执行同一脚本即可，不需要另上传 `.sh` 文件：

```bash
unzip -p /root/good-badminton-gpu-api-upload.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | bash
```

该脚本会先检查 ZIP 路径安全、Python 依赖、CUDA、应用导入和模型路径；随后将候选代码
切换到固定应用目录，并把上一版保留在 state 目录。候选版本启动失败或 health 没有同时
声明羽毛球和网球时，脚本会自动恢复上一版。密钥、权重和任务目录始终位于持久 state 目录；
首次运行时会创建持久 `.gpu-api.env`，但不会在终端输出 API Key。

若服务器目前仍运行旧式目录（例如 `/root/good-badminton-source-2ff758c`），首次切换
时在运行脚本前设置它。脚本会先迁移旧目录中的 `.gpu-api.env`、`api_data/` 和 `weights/`，
健康检查通过后再删除旧目录：

```bash
unzip -p /root/good-badminton-gpu-api-upload.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh > /tmp/good-badminton-refresh.sh
GOOD_BADMINTON_LEGACY_APP_DIR=/root/good-badminton-source-2ff758c \
GOOD_BADMINTON_REMOVE_LEGACY_APP=1 \
  bash /tmp/good-badminton-refresh.sh
rm -f /tmp/good-badminton-refresh.sh
```

将示例中的旧目录替换为实际 `uvicorn` 进程的工作目录；不要猜路径。若旧进程没有
`.gpu-api.pid`，先手动停止已确认属于 Good-Badminton 的旧 `uvicorn` 进程，再执行迁移，
避免两个程序争抢 `8080` 端口。

首次切换还会校验姿态与羽毛球模型在固定持久路径中存在：

```text
/root/good-badminton-gpu-api-state/weights/yolo11n-pose.pt
/root/good-badminton-gpu-api-state/weights/yolo11s-ball.pt
```

旧目录里已有 `weights/` 时脚本会自动迁移；否则只需手工上传一次到上面的路径。脚本会在
切换代码之前因缺权重而失败，不会生成“健康但实际无法分析视频”的假部署。

云平台的开机启动命令也固定为：

```bash
GOOD_BADMINTON_ENV_FILE=/root/good-badminton-gpu-api-state/.gpu-api.env \
  bash /root/good-badminton-gpu-api/deploy/start_gpu_api_container.sh \
  /root/good-badminton-gpu-api
```

这套流程适合现有已安装依赖的 GPU 实例。若更换了 Python 大版本、CUDA 运行时或新增
Python 依赖，先单独完成一次依赖安装和小视频验收，再使用固定目录流程日常升级。

### TrackNetV3 A/B：独立候选模型，不替换 YOLO

TrackNetV3 不能填入 WebUI 的 `yolo11s-ball.pt` 输入框；它是独立的连续帧轨迹程序。
首次 A/B 时，在 Windows 下载官方 [TrackNetV3 源码](https://github.com/qaz812345/TrackNetV3)
的 Code ZIP，并下载该仓库 README 链接的 `TrackNetV3_ckpts.zip`（内含
`TrackNet_best.pt`，以及通常包含的 `InpaintNet_best.pt`）。上游 README 指定将检查点
放到 `ckpts/` 后以 `predict.py` 推理；其公布的开发环境较旧，因此必须以服务器实际
GPU 小视频验收为准，不能假定与当前 Python/Torch 一定兼容。

当前共享 GPU 包刻意不包含 TrackNetV3 A/B 工具；不要向
`package_gpu_api.ps1` 传已删除的 `-IncludeTrackNetABTools` 参数。若要恢复该候选实验，
应先单独审批并交付独立工具包。然后把下面两份文件上传到服务器：

```text
/root/good-badminton-tracknet-upload/TrackNetV3-source.zip
/root/good-badminton-tracknet-upload/TrackNetV3_ckpts.zip
```

在服务器进行一次安装和预检（不会停止或修改现有 GPU API）：

```bash
bash /root/good-badminton-gpu-api/deploy/setup_tracknet_v3_ab.sh
```

`run_tracknet_v3_ab.sh` 对普通的 2–5 分钟视频使用官方 TrackNetV3 权重与连续
`weight` 集成。适配器先顺序抽样 120 帧生成模型尺寸的背景中值，再顺序解码、预处理和推理
固定大小的分块（默认每块 96 帧）；它不会把整段原始视频或全部预处理帧保存在内存中。CSV
同目录的 `tracknet_execution.json` 会记录背景、分块和内存策略。首轮只产生原始 TrackNet
候选 B；默认不跑 InpaintNet B*，避免将轨迹补全误读为额外真实检测。

可按显存和视频稳定性调整批量、背景样本数或内存上限：

```bash
TRACKNET_BATCH_SIZE=8 TRACKNET_BACKGROUND_SAMPLE_COUNT=180 TRACKNET_CHUNK_FRAMES=64 \
  bash /root/good-badminton-gpu-api/deploy/run_tracknet_v3_ab.sh <video> <detections.jsonl>
```

只有原始 B 在人工标签上通过 A/B 门槛后，才按需运行更慢的 B*：

```bash
TRACKNET_RECTIFICATION=1 \
  bash /root/good-badminton-gpu-api/deploy/run_tracknet_v3_ab.sh <video> <detections.jsonl>
```

该 B* 调用官方 `predict.py`，会重新执行原始处理，且只可用于人工复核；A/B 报告必须记录
批量、背景样本数和是否启用轨迹修复，不能把 B 与 B* 混为相同类型的检测证据。

完成后，使用与 YOLO 完全相同的原视频及其对应的 `detections.jsonl` 跑 TrackNet 原始
结果。第三个参数只在人工标注完成后提供；没有人工真值时脚本只产出可复核的原始 CSV，
不会宣称 TrackNet 更好：

```bash
bash /root/good-badminton-gpu-api/deploy/run_tracknet_v3_ab.sh \
  /absolute/path/to/original.mp4 \
  /absolute/path/to/detections.jsonl
```

TrackNet 源码与权重会固定保存在
`/root/good-badminton-gpu-api-state/models/tracknetv3/`；每次 A/B 输出写到
`/root/good-badminton-gpu-api-state/tracknet_ab/`。原始 YOLO 输出、正式 API 和 WebUI
模型路径均不会被覆盖。

### 可选：Huji 兼容比赛进行中证据

Huji 的动作分类器只区分画面是否处于打球过程，不能提供羽毛球位置、击球、落点或得分。
因此它在本项目中仅用于一个保守用途：当“羽毛球丢失后的方向反转”可能误切回合、而场景模型
仍显示比赛持续时，将该候选降为人工复核；它不会覆盖落地、出界、人工边界或自动计分。

Huji 的公开仓库配置引用了单打/双打 `best.pt` 分类权重路径，但项目不能假设这些训练权重已经
公开可用或适合本机位。只有已取得、已审核且类别包含 `play_ball` 的 Ultralytics 分类模型时才配置：

```bash
set -a; source /root/good-badminton-gpu-api-state/.gpu-api.env; set +a
export GOOD_BADMINTON_HUJI_ACTION_MODEL=/root/good-badminton-gpu-api-state/models/huji/badminton_scene_best.pt
export GOOD_BADMINTON_HUJI_SAMPLE_HZ=6
```

将这两个变量写入持久 `.gpu-api.env` 后重启 GPU API。每次完成分析会额外写出
`derived/huji_play_state_v1.jsonl`，并在 `metadata.json -> derived.play_state` 中记录是否可用。
未配置模型时正常分析不会变慢，也不会产生任何 Huji 推断结果。

### 模型权重属于独立发布物

`weights/yolo11n-pose.pt` 与 `weights/yolo11s-ball.pt` 不应依赖运行时自动下载：租赁实例无公网时，首次分析会因此失败。每个新实例首次部署时，将经校验的模型权重包上传并解压到应用目录的 `weights/`；代码升级时仅当权重版本变更才重新上传。记录权重文件名、SHA-256、来源和对应代码提交，避免代码与模型不匹配。

## 本次部署经验与后续运行准则

### 结论

GPU 推理服务已经可运行：容器内健康检查和已验证的公网映射都返回 `status: ok`。GPU 服务的职责只限模型和视频处理；比赛业务、账户/SSO、数据库、查询和前端应留在业务服务器。服务采用异步单队列，一次处理一场比赛，避免 24 GiB 显存上的多个姿态/球模型任务互相争抢资源。

### 已踩到的问题与处理方式

| 问题 | 根因 | 已采用处理方式 |
| --- | --- | --- |
| 通过网关 SSH 不能自动选择实例 | 平台登录后是交互菜单，不是稳定的实例 SSH 端点 | 通过平台 Web 终端进入实例；未来如需自动开关机，必须使用平台 OpenAPI/CLI，而不是模拟菜单输入 |
| 脚本提示 `bash\r` 不存在 | Windows 打包后的 shell 脚本为 CRLF 换行 | 实例内使用 `sed -i 's/\r$//' deploy/install_gpu_api.sh`；后续发布包应在 Linux/CI 中校验 shell 脚本行尾 |
| GitHub、codeload、PyTorch 官方站和 APT 访问失败 | 租赁容器限制公网出口 | 本地按提交号打包源码和权重，经平台上传；普通 pip 依赖使用镜像中可用的 pip 源 |
| GPU PyTorch 下载失败 | 脚本新建 venv 后无法访问官方下载站 | 复用镜像自带的 CUDA PyTorch；已确认 `torch.cuda.is_available()` 为 `True` |
| 没有 systemd | GPU 镜像运行在容器中 | 使用 `nohup python3 -m uvicorn ...` 启动，记录 `.gpu-api.pid`，查看 `gpu-api.log` |
| 根路径返回 404 | API 未定义 `/` 路由 | 以 `/api/v1/health` 的 HTTP 200 和 `status: ok` 作为唯一健康验收标准 |

### 日常操作

启动或重启 API 前先确认端口和旧进程：

```bash
cd /root/good-badminton-source
if [ -f .gpu-api.pid ] && kill -0 "$(cat .gpu-api.pid)" 2>/dev/null; then
  kill "$(cat .gpu-api.pid)"
fi
set -a; source .gpu-api.env; set +a
nohup python3 -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8080}" > gpu-api.log 2>&1 &
echo $! > .gpu-api.pid
sleep 3
curl -fsS "http://127.0.0.1:${PORT:-8080}/api/v1/health"
```

### 容器开机自启动

`nohup` 只能脱离当前终端运行；容器/实例重启后不会自动恢复。项目提供了容器启动脚本 `deploy/start_gpu_api_container.sh`。在算家云控制台的“启动命令 / 开机脚本 / 容器启动命令”字段配置：

```bash
bash /root/good-badminton-source-<commit>/deploy/start_gpu_api_container.sh /root/good-badminton-source-<commit>
```

脚本读取同目录 `.gpu-api.env` 的 `PORT=8080`、写入 `.gpu-api.pid` 与 `gpu-api.log`，并在 10 秒内验证本机健康检查。重复执行时，如果已有健康 API，会安全退出而不创建重复进程。停止命令为：

```bash
bash /root/good-badminton-source-<commit>/deploy/stop_gpu_api_container.sh /root/good-badminton-source-<commit>
```

若云平台没有启动命令能力，不能依赖 crontab 或容器内 systemd；每次开机后必须手动执行该启动脚本，或向平台申请 OpenAPI/CLI/启动钩子。

分析完成、确认没有 `queued` 或 `running` 任务后再停止 API 并从平台关机。GPU 实例已关机时不能自唤醒；要实现“有任务才开机、完成就关机”，由业务服务器调用算家云 OpenAPI 或官方 CLI 执行启动、等待健康检查、提交任务、取回结果、停机的生命周期。

### 升级和回滚

每次发布记录代码提交号、权重版本/SHA-256 和 API 端口映射。将新源码上传并解压到新目录，先用临时端口做健康检查；验证成功后才停止旧 API 并切到正式端口。旧目录和旧 PID 在新版本首次真实任务完成前保留，以便快速回滚。纯 Python 单文件修复可上传补丁，但任何涉及依赖、模型或多个文件的变更均发布完整源码包。

## 停止与关机

处理完成后先确认没有 `queued` 或 `running` 任务，再停止服务并关机：

```bash
sudo systemctl stop good-badminton-gpu-api
sudo shutdown -h now
```

容器模式先执行 `kill "$(cat /root/good-badminton-source/.gpu-api.pid)"`，再从平台页面执行关机/释放实例；不要在任务运行时直接关机。

不要在任务运行时直接关机；任务状态会保留为中断失败，已上传文件和已有产物不会自动删除。
