# 共享双运动 GPU 包运行手册

本产物只有一个源码包 `good-badminton-gpu-api-upload.zip` 和一个入口
`api.app:app`。同一进程按请求的 allow-listed `sport_id` 选择羽毛球或网球
视觉 profile；它不包含模型、任务数据、业务服务、评测平台或密钥。

## 本地构建

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gpu_package tests.test_local_docker -v
powershell -ExecutionPolicy Bypass -File .\deploy\package_gpu_api.ps1
Get-FileHash .\deploy\good-badminton-gpu-api-upload.zip -Algorithm SHA256
```

本地 Docker CPU 验证使用 [LOCAL_GPU_API_DOCKER.md](LOCAL_GPU_API_DOCKER.md)。它只验证接口和状态契约，不代表 GPU 性能。

## 已授权的 GPU 主机刷新

上传 ZIP 到 `/root/good-badminton-gpu-api-upload.zip` 后，在 GPU 主机执行：

```bash
unzip -p /root/good-badminton-gpu-api-upload.zip \
  good-badminton-gpu-api/deploy/refresh_gpu_api_from_zip.sh | bash
```

刷新前，持久目录必须存在经过校验的：

```text
/root/good-badminton-gpu-api-state/weights/yolo11n-pose.pt
/root/good-badminton-gpu-api-state/weights/yolo11s-ball.pt
```

脚本先校验 ZIP、CUDA Python 运行时、依赖、应用导入和模型路径，再切换代码。
上一版应用目录保存在 `/root/good-badminton-gpu-api-state/previous-app`；候选版本启动失败或 health 未同时声明两项运动时会自动恢复它。模型、API key 和任务数据始终留在 state 目录。

正常刷新后，确认：

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health
```

响应必须包含 `status="ok"` 与
`supported_sport_ids=["badminton","tennis"]`。随后才可在真实 GPU 上执行 T-MGP-10 的最小羽毛球、网球请求验收。

`deploy/install_gpu_api.sh` 仅用于首次从 Git 或已解压源码安装依赖；Git 模式必须显式提供已审核的共享 GPU 分支。日常更新只使用 ZIP 刷新脚本。
