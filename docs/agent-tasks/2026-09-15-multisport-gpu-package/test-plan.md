# 测试方案：多运动 GPU 单包

## 测试边界

不下载模型、不执行真实 GPU 推理、不上传或重启远端服务器。本地测试以既有 `unittest`、FastAPI `TestClient` 与临时目录为夹具；模型构造均使用现有 fake/stub seam。所有改动先以 RED 证明旧实现不满足新契约，再在实现后转绿。

| Test ID | Acceptance ID | 层级 | 场景 | 预期结果 | 命令/环境 | 编码前证据 | 最终证据 |
|---|---|---|---|---|---|---|---|
| T-MGP-01 | AC-01 | API 单元 | 未鉴权 health 请求单个 GPU 进程 | 返回 `supported_sport_ids=["badminton","tennis"]`；不以单一固定 profile 作为服务身份 | `python -m unittest tests.test_gpu_api -v`，项目 `.venv` | RED：`KeyError: supported_sport_ids` | 32 项 MGP-01 定向测试通过 |
| T-MGP-02 | AC-02 | API/运行时单元 | 羽毛球、网球分别创建 stream session | 两个请求均成功；会话清单持久化各自 `sport_id` 与 allow-list 派生 profile | `python -m unittest tests.test_gpu_api tests.test_gpu_sport_profiles -v` | RED：网球创建返回 422，固定为羽毛球 profile | 32 项 MGP-01 定向测试通过 |
| T-MGP-03 | AC-02 | API 边界 | 不支持的 `sport_id`、运动与 mode/人数冲突 | 返回 422，且不会创建会话目录或持久化清单 | 同 T-MGP-02 | RED：unknown sport 得到固定-profile mismatch；网球 roster 冲突未进入目标校验 | 32 项 MGP-01 定向测试通过 |
| T-MGP-04 | AC-03 | API 单元 | 完整视频创建任务传入 `sport_id` | `sport_id` 写入任务请求/清单；创建、查询、取消、产物路径保持原有契约 | `python -m unittest tests.test_gpu_api -v` | RED：任务清单缺少 `input.sport_id`；`squash` 请求被接受（202） | 32 项 MGP-01 定向测试通过 |
| T-MGP-05 | AC-03 | 回归 | 完整视频和 2 秒分片接口的认证、创建、查询、取消、产物读取 | 所有既有 API 测试保持绿色 | `python -m unittest tests.test_gpu_api tests.test_gpu_code_boundary -v` | 已有基线：47 项通过；RED 运行中其余旧 API 断言仍通过 | 32 项 MGP-01 定向测试通过 |
| T-MGP-06 | AC-04 | 业务服务单元 | 共享 GPU health 包含/不包含目标运动；业务服务创建任务时显式传 `sport_id` | 前者允许，后者在提交前拒绝；旧按运动配置仍可回退 | `python -m unittest tests.test_remote_gpu -v` | RED：共享配置未优先，仍选用 `GOOD_TENNIS_GPU_API_URL` | 15 项业务远端测试通过 |
| T-MGP-07 | AC-04 | 评测服务单元 | 评测 `GpuClient.verify` 面对共享 health | 目标运动在 `supported_sport_ids` 中则通过；不在则抛出 `GpuClientError` | `..\\Good-Badminton-Evaluation-Platform\\backend\\.venv\\Scripts\\python.exe -m unittest tests.test_gpu_client -v` | RED：共享 health 成功用例报 `GPU identity mismatch` | 7 项评测 client/service 测试通过，含完整视频和流式 `sport_id` |
| T-MGP-08 | AC-05 | 打包集成 | 不带运动参数执行 PowerShell 打包脚本，检查 ZIP 清单 | 生成唯一 `good-badminton-gpu-api-upload.zip`；含 `api/app.py`；不含 `evaluation/`、`webui/`、`tests/`、权重、数据、`.env` | `python -m unittest tests.test_gpu_package -v` | RED：脚本引用不存在的 `deploy/run_performance_gate.sh`，无法出包 | 2026-09-15：2 项 ZIP/入口测试通过 |
| T-MGP-09 | AC-05, AC-06 | 部署脚本静态 | 刷新、启动、systemd 脚本指向唯一 `api.app:app` | 不含固定运动入口或 `start_{sport}` 分派；保留外部状态/权重目录约束 | `python -m unittest tests.test_gpu_package -v` | RED：`-Sport` 参数及启动脚本仍指向固定运动 entrypoint | 2026-09-15：2 项 ZIP/入口测试通过 |
| T-MGP-10 | AC-06 | 人工/环境 | 后续授权的 GPU 主机受控部署 | 上传单一 ZIP 后 health 同时报两项运动；两类最小请求成功，权重/状态仍持久化 | 后续由运维在真实 GPU 执行，记录 health、日志、请求响应 | 本任务不执行（无部署授权） | 未执行，不能以本地测试替代 |
| T-MGP-11 | AC-07 | Docker 集成 | 用 Compose 构建并启动本地多运动 API | health 为 `ok` 且含两项运动；容器可见 `/app/weights` | `docker compose -f docker-compose.local.yml build && up -d` | RED：Compose 将权重映射到不存在的 `/models` | 2026-09-16：镜像构建成功；临时 `127.0.0.1:28080` health 返回两项运动，`docker compose ps` 为 `healthy`，容器内 `weights-mount-ok`。本机 8080/18080 已被占用，未停止既有进程。 |
| T-MGP-12 | AC-03, AC-04 | API/评测集成 | 评测完整视频提交与网球完整视频 profile | 评测元数据不进入 GPU `options_json`；网球仅接受完整视频的 `singles_match`，冲突在持久化前拒绝 | 主仓及评测后端 `unittest` | RED：GPU 收到不支持的 `court_corners`、`sport_id`、`template_sha256`；完整网球任务绕过 profile | 2026-09-16：主仓 66 项、评测后端 12 项定向回归通过。 |

## 预期新增或修改的测试代码

| 文件 | 动作 | 覆盖 |
|---|---|---|
| `tests/test_gpu_api.py` | 修改 | T-MGP-01、02、03、04、05 |
| `tests/test_gpu_sport_profiles.py` | 修改 | T-MGP-02、03 |
| `tests/test_gpu_pure_launchers.py` | 修改或替换为单包启动契约 | T-MGP-01、09 |
| `tests/test_remote_gpu.py` | 修改 | T-MGP-06 |
| `tests/test_gpu_package.py` | 新增 | T-MGP-08、09 |
| `tests/test_local_docker.py` | 新增 | T-MGP-11 |
| `backend/tests/test_gpu_client.py` | 修改（评测项目） | T-MGP-07 |

## 回归范围

实现后运行：

```powershell
.venv\\Scripts\\python.exe -m unittest tests.test_gpu_api tests.test_gpu_code_boundary tests.test_gpu_pure_launchers tests.test_gpu_sport_profiles tests.test_remote_gpu tests.test_gpu_package -v
..\\Good-Badminton-Evaluation-Platform\\backend\\.venv\\Scripts\\python.exe -m unittest tests.test_gpu_client -v
```

不将这组本地 mock/API 测试解释为模型精度、GPU 显存、实时吞吐或远端可用性证据；这些仅由 T-MGP-10 的后续真实环境验收补齐。
