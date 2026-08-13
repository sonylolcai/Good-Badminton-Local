# GPU 视频分析 API 部署与联调

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
| `GET` | `/api/v1/jobs/{job_id}/artifacts/{name}` | 下载标注视频、元数据、检测 JSONL 等 |

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
  -F 'options_json={"pose_imgsz":1280,"output_video_style":"skeleton","audio":false}'
```

成功后轮询 `/api/v1/jobs/{job_id}`。`succeeded` 后调用 `/result`，返回 `annotated_video`、`metadata`、`detections` 和（生成时）`spatial_match_summary` 的受保护下载 URL。

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

### 后续升级方式

该实例无法从 GitHub 拉取分支，因此不能直接 `git pull`。每次升级建议由本地从指定提交打包源码 zip，经平台上传后解压到新目录（例如 `/root/good-badminton-source-<commit>`）；在新目录安装/复用依赖、启动到临时端口并执行健康检查后，再停止旧 PID 并切换正式端口。这保留可回退的旧目录和旧进程。

只有纯 Python 代码的小修复，也可以上传补丁文件并在实例内 `patch -p1`；但涉及依赖、模型、部署脚本或多个文件时，一律上传完整、带提交号的源码包，减少版本漂移。

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
