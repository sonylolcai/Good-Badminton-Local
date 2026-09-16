# 本地 Docker GPU API

此 Compose 只启动本地 CPU 版 `api.app:app`，用于接口、任务状态和多运动契约验证；不代表本地实时 GPU 推理能力，也不连接或控制远端 GPU。

在仓库根目录执行：

```powershell
Copy-Item .gpu-api.local.env.example .gpu-api.local.env
# 编辑 .gpu-api.local.env，替换 GOOD_BADMINTON_API_KEY
docker compose --env-file .gpu-api.local.env -f docker-compose.local.yml up --build
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8080/api/v1/health
```

若 `8080` 已被占用，在 `.gpu-api.local.env` 中将
`GOOD_BADMINTON_LOCAL_PORT` 改为另一个空闲 loopback 端口，例如 `28080`，并按该端口访问 health。

任务状态放在 Docker named volume，`docker compose ... down` 不会删除它。模型不进入镜像；如需本地推理，将权重放在被 Git 忽略的 `weights/`，Compose 会以只读方式挂载到应用的标准路径 `/app/weights`。这同时覆盖完整视频任务、羽毛球分片和网球 YOLO 的默认权重回退路径。停止服务：

```powershell
docker compose --env-file .gpu-api.local.env -f docker-compose.local.yml down
```

不要加 `-v`，除非明确要删除本地 API 任务状态。

网球完整视频任务必须在 `options_json` 中显式传入
`"session_mode":"singles_match"`；单人训练应使用 `/api/v1/stream-sessions`。
