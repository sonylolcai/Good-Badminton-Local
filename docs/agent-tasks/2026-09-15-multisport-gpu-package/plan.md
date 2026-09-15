# 实施计划

## 唯一交付决策

本任务只生成 `good-badminton-gpu-api-upload.zip`：它同时包含羽毛球和网球的运行代码，由每个请求携带的 `sport_id` 选择既有视觉 profile。不会生成、交付或部署任何按运动拆分的 ZIP。

## 待替换现状与目标流

```text
当前：调用方 -> 固定羽毛球或网球 GPU URL -> 固定 profile 进程
目标：业务后端 / 评测后端 -> 一个 GPU URL -> sport_id -> session/job profile -> GPU artifacts
```

GPU 仍只产出匿名视觉观测、状态、性能追踪及文件产物；评测在独立评测服务执行，业务解释在业务服务执行。

## 文件清单

| 路径 | 动作 | 用途 | 任务 |
|---|---|---|---|
| `api/app.py` | 修改 | 作为唯一多运动完整视频与流式 API 组合根；health 公开支持列表。 | MGP-01 |
| `api/stream_runtime.py`、`api/mode_sync.py` | 修改 | 按请求解析 profile，持久化派生配置，构建正确的运行时。 | MGP-01 |
| `api/gpu_stream_app.py`、`apps/badminton_gpu/app.py`、`apps/tennis_gpu/app.py` | 删除 | 去除与单包目标冲突的固定入口。 | MGP-01 |
| `webui/remote_gpu.py` | 修改 | 共享地址/密钥优先，health 检查目标是否在支持列表，羽毛球也显式传 `sport_id`。 | MGP-02 |
| `deploy/package_gpu_api.ps1` | 修改 | 仅打入 GPU 运行时白名单，输出唯一的多运动 ZIP，不再引用删除的评测文件。 | MGP-03 |
| `deploy/refresh_gpu_api_from_zip.sh`、`deploy/start_gpu_api_container.sh`、`deploy/install_gpu_api.sh`、`deploy/good-badminton-gpu-api.service` | 修改 | 启动/刷新唯一 `api.app:app`，保留持久化密钥、权重与任务状态。 | MGP-03 |
| `deploy/start_sport_gpu_container.sh`、`deploy/start_badminton_gpu_container.sh`、`deploy/start_tennis_gpu_container.sh` | 删除 | 移除固定入口部署歧义。 | MGP-03 |
| `tests/test_gpu_pure_launchers.py`、`tests/test_gpu_sport_profiles.py`、`tests/test_remote_gpu.py` | 修改 | 将固定进程断言替换为多运动会话和共享 health 断言。 | MGP-01, MGP-02 |
| `tests/test_gpu_package.py` | 新增 | 运行打包脚本并检查 ZIP 内容和入口。 | MGP-03 |
| `docs/GPU_MULTI_SPORT_PACKAGE.md` | 新增 | 单包构建、配置、上传和后续受控部署说明。 | MGP-03 |
| `docs/GPU_MULTI_SPORT_ITERATION_PLAN.md` | 修改 | 标注旧“两包/固定入口”方案已被本次用户决策替代。 | MGP-03 |
| `backend/evaluation_service/gpu_client.py` | 修改 | 接受共享 health 的 `supported_sport_ids` 并验证目标运动。 | MGP-02 |
| `backend/tests/test_gpu_client.py` | 修改 | 覆盖共享 health 成功与不支持运动失败。 | MGP-02 |

## 任务与依赖

| 任务 | 依赖 | 产出 | 验收 | 回滚 |
|---|---|---|---|---|
| MGP-01 多运动 GPU API | 无 | 单一 API 按 session/job 解析 profile | AC-01–03 的单元与 API 测试 | 回退到 `fd305b3` 的固定入口包 |
| MGP-02 调用方兼容 | MGP-01 | 业务/评测后端按支持列表校验共享实例 | AC-04 定向测试 | 保留旧配置回退，回退调用方提交 |
| MGP-03 单包构建 | MGP-01 | 白名单 ZIP、刷新/启动脚本、运行说明 | AC-05–06 的构建检查 | 旧远端目录与状态不自动变更 |

MGP-02 与 MGP-03 在 MGP-01 通过后可并行；本次由同一执行者串行完成并逐任务提交。

## 风险与控制

| 风险 | 控制 |
|---|---|
| 混用场地或人数规则 | 仅从 allow-list profile 推导规则；`sport_id`/mode 冲突在写入前拒绝。 |
| 旧调用方未传运动 | 服务端仅兼容性默认羽毛球；受控调用方更新为显式发送。 |
| 完整视频接口在纯流入口中丢失 | 唯一部署入口切到现有 `api.app`，API 测试覆盖 `/jobs` 与 `/stream-sessions`。 |
| ZIP 泄露或混入评测/业务代码 | 源码白名单 + 自动 ZIP 内容测试；权重、数据和 `.env` 继续不入包。 |
| 远端状态或权重损失 | 刷新仅替换应用目录；状态/权重目录为外部持久目录；本任务不执行远端刷新。 |
| GPU 显存/吞吐未验证 | 明确列为后续真实 GPU 验收，不把本地测试当作硬件结论。 |

## 验证矩阵

| 验收 | 证据 |
|---|---|
| AC-01–03 | FastAPI `TestClient`：两种 stream request、错误组合、完整视频任务的持久化与取消。 |
| AC-04 | `webui.remote_gpu` 与评测 `GpuClient` 使用模拟 health 的成功/失败测试。 |
| AC-05 | PowerShell 打包至临时 ZIP；验证 POSIX 路径、唯一入口和禁止路径。 |
| AC-06 | 构建脚本静态检查及运行手册；远端部署单列为未执行。 |

## 计划审批

已批准：用户于 2026-09-15 确认“只交付一个 GPU 包；按 `sport_id` 在同一 GPU 进程中选择运动 profile”的方案。下一关为测试方案审批；在获批前不修改生产运行代码。
