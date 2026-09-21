# 运营后台设计（MVP-01）

## 目标与范围

本后台服务于平台管理员、球馆负责人和运营人员；它不是球友使用的微信小程序，也不是 GPU 服务本身。现有 Gradio 视频分析、球路复核和任务历史必须完整保留，并统一收纳到“分析工作台”页签。

首版提供：

1. **总览**：业务 API、PostgreSQL、GPU API、当前本地分析任务和最近任务账本的状态。
2. **场馆管理**：租户、场馆、场地的查询与新增/编辑。摄像头、标定和二维码保留为下一小版，因为它们依赖设备接入和安全的二维码生成器。
3. **人员管理**：用户、场馆成员角色（owner/operator/viewer）查询、创建/更新与授权；不显示微信身份、手机号或 subject hash。
4. **GPU 服务**：GPU 地址和 API Key 的服务端配置、健康检查、受控的启动/关闭请求和操作审计。
5. **数据查询**：业务侧分析账本、数据库 analysis_jobs 与其状态。
6. **任务控制**：本机 WebUI 任务中断、已受理远端 GPU Job 的主动取消，以及可验证的取消状态。
7. **分析工作台**：原视频分析、球路复核、任务历史，功能不删减。

## 信息架构

```text
运营后台
├─ 总览
├─ 场馆管理
├─ 人员管理
├─ GPU 服务
├─ 数据查询
├─ 任务控制
└─ 分析工作台
   ├─ 视频分析
   ├─ 球路复核
   └─ 分析任务历史
```

## 权限与数据边界

| 角色 | 可见范围 | 可变更范围 |
| --- | --- | --- |
| platform_admin | 跨租户、服务与全量运营数据 | 全部，含 GPU 配置/启动关闭 |
| venue_owner | 所属球馆、成员与运营数据 | 本球馆、成员角色、场地 |
| venue_operator | 所属球馆任务与设备状态 | 场地状态、任务中断 |
| analyst | 已授权的分析任务 | 提交/复核/中断自己任务 |
| viewer | 已授权查询 | 无 |

当前 Gradio 是本机开发运营模式，尚未接入登录/会话身份。因此所有写操作都会写入业务 `audit_events`，且页面显式标识为“本机开发运营模式”。上线前必须接入服务端鉴权并将上述矩阵落实到 API/RLS，不能把该页面直接暴露公网。

GPU 边界不变：GPU 只收到匿名录像、标定、匿名 `track_id` 与分析配置；它不得接收用户、昵称、比分、队伍、积分或排行榜。人员和赛果只保留在业务侧。

## GPU 自动化策略

GPU 健康检查和已受理 Job 取消是真实 API 动作：

- 健康：业务服务使用服务端 API Key 调用 `/api/v1/health`。
- 取消：业务服务使用服务端 API Key 调用 `DELETE /api/v1/jobs/{job_id}`。
- 若 DELETE 请求没有得到可验证响应，任务只能标为 `interrupted_unconfirmed`，不能声称已停止。

“启动/关闭”不能把公网地址当作可控制实例。首版采用受控适配器：只有服务进程预先配置 `GOOD_BADMINTON_GPU_CONTROL_COMMAND` 并显式设置 `GOOD_BADMINTON_ENABLE_GPU_AUTOMATION=1` 时，后台才能以固定命令尾随 `start` 或 `stop` 执行。页面不接收任意 shell 命令、SSH 私钥或云厂商令牌。

未配置时显示“外部托管，仅健康检查”，按钮返回可操作说明而不是假成功。后续接入云厂商/CompShare 时，新增 `GpuControlAdapter` 实现，并要求：每个操作有 idempotency key、provider receipt 和审计记录；start/stop 需轮询中间状态；凭据只进入密钥管理或服务端环境，永不回传浏览器。

## 数据访问与运行配置

- 业务数据库：`GOOD_BADMINTON_BUSINESS_DATABASE_URL`，仅服务端环境变量。后台通过 `psycopg` 访问现有 `business` schema。
- GPU 配置：`.webui-remote-gpu.env`（已 gitignore），UI 可更新 URL；API Key 仅接受新值且绝不读回/显示。
- 任务账本：`outputs/business_tasks`，仅用于本地业务侧任务追踪，远端任务仍以 GPU API 为准。
- GPU 控制审计：本机开发阶段落入 `outputs/operator_backoffice/operations.jsonl`；接入数据库 Worker 后迁移为 `business.compute_operations` + `audit_events`。

## 验收标准

1. 旧视频分析、球路复核、任务历史均可在“分析工作台”内打开。
2. 未配置数据库时，场馆/人员/数据库查询明确提示，不创建影子数据。
3. 配置数据库后，场馆、场地、用户和成员角色的新增/更新写入既有表，写操作同时产生不含敏感字段的审计记录。
4. GPU 地址保存后不显示 API Key；健康检查的失败信息可见但不泄露密钥。
5. 远端 Job 取消只接受业务侧账本中已有的 `remote.job_id`；无法确认时保留 `interrupted_unconfirmed`。
6. 没有预配置自动化适配器时，启动/关闭不可执行且有明确解释；不能从 UI 输入 shell 命令。
