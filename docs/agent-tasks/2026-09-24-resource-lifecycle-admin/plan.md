# 实施计划

## 结果

交付一个最小但完整的控制面：业务后台负责管理员、权限、球员、业务视频、手动解析和全局保留策略；评测平台继续拥有自己的 Run 与资源；GPU API 只执行受认证的物理删除。三处以资源账本和幂等删除状态连接，不共享数据库表或磁盘目录。

## 基线

| 仓库 | 分支 / HEAD | 工作区 | 基线检查 |
| --- | --- | --- | --- |
| `Good-Badminton` | `feat/match-clip-delivery` / `74b5fe3` | 发现时干净 | 后端 28 项通过；Web lint 通过；build 因 Google Fonts 网络请求失败 |
| `Good-Badminton-Evaluation-Platform` | `feat/evaluation-backend-separation` / `dfd3af5` | 4 个既有任务文件未提交 | 后端 16 项、lint、边界检查通过 |

## 目标数据流

```text
管理员浏览器
  -> operator-api 会话认证 + RBAC
  -> business.managed_media_resources / managed_media_resource_locations / analysis_jobs
  -> 业务独立媒体根目录
  -> 共享 GPU API（任务 id + API key）

评测浏览器
  -> evaluation API
  -> evaluation.resources / resource_locations / run_resources
  -> 评测独立 artifact root
  -> 共享 GPU API（任务/流会话 id + API key）

retention-worker（单实例）
  -> 读取 business.video_retention_policy
  -> 清理业务视频位置
  -> 调用受内部密钥保护的 evaluation cleanup API
  -> GPU API 仅删视频；各平台保留 JSON 与逻辑记录
```

## SQL 设计

### 业务库 migration `0010`

- `business.admin_accounts`：用户名、scrypt 参数/盐/摘要、状态、必须改密、微信扩展字段、时间戳。
- `business.admin_role_assignments`：固定角色 `platform_admin|venue_admin`，平台角色不带球馆，球馆角色必须带 `venue_id`。
- `business.admin_sessions`：会话令牌摘要、过期、撤销、最近使用时间。
- `business.managed_media_resources`：后台资源生命周期账本，关联 tenant/venue/player/match/analysis job，类型、媒体类型、上传成功时间、保留状态；与既有交付表 `business.media_assets` 分离。
- `business.managed_media_resource_locations`：每个本地/GPU/未来对象存储位置一行，含 backend、location、object key、sha256、size、删除状态/错误/时间。
- `business.video_retention_policy`：单行配置，默认 `enabled=false`、`retention_days=7`、时区与每日执行时间。
- 扩展 `business.analysis_jobs`：输入资源、请求管理员和手动触发信息。
- 复用 `business.users` 作为球员资产，复用 `business.venue_memberships` 表示球员与球馆关系；后台登录身份不写入此表。

### 评测库 migration `0003`

- `evaluation.resources`：逻辑资源、media type、uploaded_at、sha256、size、状态。
- `evaluation.resource_locations`：本地、GPU 或未来对象存储位置及逐位置删除状态。
- `evaluation.run_resources`：Run 到资源的多对多关系与用途，支持内容哈希复用与引用计数。
- 旧 Run 不自动猜测远程资源；访问时可安全识别的本地路径按需登记，未登记资源不进入自动清理。

## 控制规则

- RBAC 权限码在服务端固定映射，数据库只保存账号到角色/球馆的分配，避免为两个角色建设可编辑权限设计器。
- `platform_admin`：全局管理、管理员创建、保留设置、两种删除。
- `venue_admin`：限定球馆的球员/视频/解析管理和“仅删除资源”。
- 账号密码首版完成；微信字段仅保留可空身份映射，不实现 OAuth 流程。
- 存储后端采用小型枚举分发函数，首版只实现 `local_disk` 与 `gpu_http`，不引入对象存储 SDK 或单实现接口层。
- 自动清理 worker 为单独 Compose 服务，不嵌入两个 uvicorn worker，避免重复执行；同一清理批次使用数据库锁和幂等状态。

## 预计文件

### `Good-Badminton`

| 文件 | 动作 | 用途 |
| --- | --- | --- |
| `business_gateway/migrations/0010_admin_auth_media_resources.sql` | 新增 | 认证、RBAC、资源、保留策略表 |
| `operator_api/services/auth.py` | 新增 | 密码、会话、角色与球馆授权 |
| `operator_api/services/resource_lifecycle.py` | 新增 | 资源登记、两类删除、保留清理 |
| `operator_api/services/operator_backoffice.py` | 修改 | 复用现有业务查询并接入球员/任务操作 |
| `operator_api/main.py` | 修改 | 登录、管理员、球员、资源、解析与设置 API |
| `business_gateway/edge_api.py` | 修改 | 受保护的录像资源删除能力 |
| `api/jobs.py` | 修改 | 终态全视频任务仅删视频/删除全部数据 |
| `api/app.py` | 修改 | GPU 资源删除 API，保留原取消语义 |
| `api/stream_sessions.py` | 修改 | 7 天按上传时间仅删视频，保留 JSON |
| `deploy/business-server/retention_worker.py` | 新增 | 单实例每日协调清理 |
| `deploy/business-server/docker-compose.stone.yml` | 修改 | worker、业务媒体卷与环境配置 |
| `deploy/business-server/business.env.example` | 修改 | bootstrap、会话、评测内部 API 与保留配置 |
| `webui-next/src/lib/operator-api.ts` | 修改 | 带 Cookie 的认证/资源请求 |
| `webui-next/src/components/AuthGate.tsx` | 新增 | 登录态保护和首次改密 |
| `webui-next/src/components/Sidebar.tsx` | 修改 | 用户、资源、设置入口 |
| `webui-next/src/app/layout.tsx` | 修改 | 后台认证边界 |
| `webui-next/src/app/login/page.tsx` | 新增 | 账号密码登录 |
| `webui-next/src/app/users/page.tsx` | 新增 | 管理员和球员资产管理 |
| `webui-next/src/app/resources/page.tsx` | 新增 | 视频、关联、解析和删除 |
| `webui-next/src/app/settings/page.tsx` | 新增 | 7 天清理开关与状态 |
| `tests/test_operator_auth.py` | 新增 | bootstrap、登录、改密、会话、RBAC |
| `tests/test_resource_lifecycle.py` | 新增 | 两类删除、部分失败、清理阈值 |
| `tests/test_operator_api.py` | 修改 | 管理 API 与球馆范围 |
| `tests/test_edge_ingest.py` | 修改 | 录像删除边界 |
| `tests/test_gpu_api.py` | 修改 | 取消和物理删除语义分离 |
| `tests/test_stream_sessions.py` | 修改 | 视频清理保留 JSON |

### `Good-Badminton-Evaluation-Platform`

| 文件 | 动作 | 用途 |
| --- | --- | --- |
| `backend/migrations/0003_resource_lifecycle.sql` | 新增 | 评测资源及位置账本 |
| `backend/evaluation_service/resources.py` | 新增 | 登记、引用、逐位置幂等删除 |
| `backend/evaluation_service/repository.py` | 修改 | Run/资源事务与全删除保护 |
| `backend/evaluation_service/service.py` | 修改 | 上传、GPU 任务和产物登记 |
| `backend/evaluation_service/gpu_client.py` | 修改 | 调用远程物理删除接口 |
| `backend/evaluation_service/api.py` | 修改 | 两类删除与内部保留清理 API |
| `src/app/history/HistoryClient.tsx` | 修改 | 拆分“仅删除资源/删除全部数据” |
| `backend/tests/test_registry_api.py` | 修改 | 替换数据库-only 删除测试 |
| `backend/tests/test_resources.py` | 新增 | 引用计数、路径安全、失败重试、TTL |
| `backend/tests/test_service.py` | 修改 | 上传和下载产物登记 |
| `.env.example` | 修改 | 独立资源根目录、内部清理密钥 |

### 文档

- 两仓任务记录、主仓部署手册和 Obsidian 既有评测/运维索引在实现收敛时更新；不创建第二套长期文档体系。

## 任务顺序与提交边界

1. 数据库与领域状态机。
2. GPU 物理资源删除。
3. 评测平台两种删除与资源账本。
4. 业务认证/RBAC 与管理员/球员 API。
5. 业务视频管理和手动解析。
6. 中央保留配置与每日 worker。
7. 管理页面与浏览器验收。
8. 回归、文档和收敛。

每步独立测试并使用精确文件暂存形成一个绿色本地提交；不推送。

## 测试策略

- SQL 静态迁移约束与已有数据兼容检查。
- 临时目录模拟本地/GPU 资源，证明只删视频、JSON 保留、全删和失败重试。
- FastAPI TestClient 验证 401/403、角色、球馆隔离、首次改密和删除权限。
- 固定上传时间验证 6天23小时不删、7天及以上删除；默认关闭不删。
- 前端 lint、build 和真实浏览器验证登录、用户、资源、设置、确认弹窗和错误反馈。
- GPU/远程存储只做契约测试；生产主机、真实对象存储和真实微信登录另列环境验证。

## 验收与证据矩阵

| 验收 | 主要证据 |
| --- | --- |
| AC-01 | 评测资源 API 测试 + 临时本地/GPU 视频删除测试 + JSON 比较回归 |
| AC-02 | 平台管理员全删除 API 测试 + Run/复核/资源 404 验证 |
| AC-03 | 一处成功一处失败 fixture + 持久化失败状态 + 幂等重试测试 |
| AC-04 | 固定时钟边界测试 + 默认关闭测试 + worker 单实例锁测试 |
| AC-05 | 空库 bootstrap、重复启动、缺失环境密码和首次改密测试 |
| AC-06 | 平台/球馆管理员权限矩阵与跨球馆 403 测试 |
| AC-07 | 管理员/球员 API 测试 + 用户管理页面浏览器验证 |
| AC-08 | 视频上传、关联、手动解析契约测试 + 资源页面浏览器验证 |
| AC-09 | 两库迁移检查、独立根目录检查、网络边界脚本 |
| AC-10 | 登录、设置、解析和删除后的审计事件断言 |
| AC-11 | 既有业务/评测回归套件 + 未登记旧资源不进入清理的测试 |

## 风险与回滚

| 风险 | 控制 | 回滚 |
| --- | --- | --- |
| 误删或跨目录删除 | 资源账本、根目录校验、终态校验、确认提示 | 默认关闭 worker；回滚服务版本；不自动恢复已删视频 |
| 本地删成功、远程失败 | 逐位置状态、幂等重试、仅全部成功才完成 | 保留逻辑记录并重试失败位置 |
| 两个 API worker 重复清理 | 独立单实例 worker + DB 锁 | 停止 worker 服务 |
| 管理员越权 | 服务端 RBAC + venue scope + 审计 | 禁用账号/撤销会话，回滚 API |
| 迁移影响旧数据 | 仅新增表/可空列，默认关闭 | 回滚代码；保留新增表，不做破坏性 down migration |
| 构建依赖 Google Fonts | 记录为当前基线；实施时改为本地/系统字体可单独修复 | 恢复原字体配置 |

## 非本地验证

- 真实 GPU 主机文件删除和失败恢复。
- 生产 Compose 单实例 worker 的 24 小时调度。
- 真实大视频上传、对象存储、吞吐和 7 天实际时间流逝。
- 微信开放平台扫码登录（本轮不实现）。

## PLAN APPROVAL

状态：2026-09-24 已批准并完成本地实施。批准未授权生产迁移、真实数据删除、部署、推送或发布；本次均未执行。
