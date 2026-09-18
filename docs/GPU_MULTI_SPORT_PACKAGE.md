# 共享双运动 GPU 包运行手册

默认产物是源码包 `good-badminton-gpu-api-upload.zip`，入口为
`api.app:app`。同一进程按请求的 allow-listed `sport_id` 选择羽毛球或网球
视觉 profile；源码包不包含模型、任务数据、业务服务、评测平台或密钥。

需要完整发布时，显式构建
`good-badminton-gpu-api-full-release.zip`。它额外包含且只包含以下运行模型
及其 SHA-256 清单：

```text
weights/yolo11n-pose.pt
weights/yolo11s-ball.pt
weights/tennis-ball.pt
```

两种包都不会包含 API key、`.gpu-api.env` 或 `api_data`。

## 本地构建

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gpu_package tests.test_local_docker -v
powershell -ExecutionPolicy Bypass -File .\deploy\package_gpu_api.ps1
Get-FileHash .\deploy\good-badminton-gpu-api-upload.zip -Algorithm SHA256
```

完整代码和模型发布包：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\package_gpu_api.ps1 -IncludeWeights
Get-FileHash .\deploy\good-badminton-gpu-api-full-release.zip -Algorithm SHA256
```

该命令会在打包前要求本地已经有上述三份模型；若任一文件缺失，打包直接失败。

本地 Docker CPU 验证使用 [LOCAL_GPU_API_DOCKER.md](LOCAL_GPU_API_DOCKER.md)。它只验证接口和状态契约，不代表 GPU 性能。

## 已授权的 GPU 主机刷新

日常源码更新上传 ZIP 到 `/root/good-badminton-gpu-api-upload.zip` 后，执行：

```bash
unzip -p /root/good-badminton-gpu-api-upload.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | bash
```

源码更新前，持久目录必须存在经过校验的：

```text
/root/good-badminton-gpu-api-state/weights/yolo11n-pose.pt
/root/good-badminton-gpu-api-state/weights/yolo11s-ball.pt
```

### 完整代码和模型发布

上传完整包到 `/root/good-badminton-gpu-api-full-release.zip` 后，执行：

```bash
unzip -p /root/good-badminton-gpu-api-full-release.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | \
  bash -s -- /root/good-badminton-gpu-api-full-release.zip
```

脚本会先校验 ZIP、CUDA Python 运行时、依赖、应用导入及三份模型的 SHA-256，
再切换代码和 `state/weights`。上一版应用与模型目录分别保存在
`/root/good-badminton-gpu-api-state/previous-app` 和
`/root/good-badminton-gpu-api-state/previous-weights`；候选版本启动失败或
health 未同时声明两项运动时会一起恢复。`.gpu-api.env`、API key 和任务数据
始终留在 state 目录，不会被包覆盖。

完整包在未配置 `GOOD_TENNIS_STREAM_BALL_MODEL` 时，会安全地补入
`tennis-ball.pt` 的默认路径；已有自定义模型路径不会被改写，且必须仍指向
新的 `weights` 目录中存在的文件。不要直接删除或 `unzip -o` 覆盖
`/root/good-badminton-gpu-api-state`。

### 全新 GPU 服务器：仅通过文件上传首次部署

不从 GPU 服务器拉取 Git。选择具备可用 CUDA PyTorch 的 Linux 镜像后，通过
云平台上传完整包到 `/root/good-badminton-gpu-api-full-release.zip`，再执行：

```bash
unzip -q /root/good-badminton-gpu-api-full-release.zip -d /root
bash /root/good-badminton-gpu-api/deploy/install_gpu_api.sh
unzip -p /root/good-badminton-gpu-api-full-release.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | \
  bash -s -- /root/good-badminton-gpu-api-full-release.zip
```

首次安装脚本只从已解压的上传包读取代码；它安装运行时依赖并创建持久 API
密钥。若镜像不能访问依赖源，先上传匹配 Linux/Python 的 wheelhouse，并设置
`GOOD_BADMINTON_USE_SYSTEM_TORCH=1`、`GOOD_BADMINTON_PYTHON_BIN` 与
`GOOD_BADMINTON_WHEELHOUSE` 后再运行该脚本。

正常刷新后，确认：

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health
```

响应必须包含 `status="ok"` 与
`supported_sport_ids=["badminton","tennis"]`。随后才可在真实 GPU 上执行 T-MGP-10 的最小羽毛球、网球请求验收。

`deploy/install_gpu_api.sh` 仅用于首次从已解压完整上传包安装依赖；日常更新只使用 ZIP 刷新脚本。
