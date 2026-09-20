# 服务与仓库拆分

## 目标

把目前混合在 `Good-Badminton` 的生产职责，整理为五个有明确所有权、部署边界和最小权限的正式代码仓库：

1. `bdteach-venue-gateway`：场馆 Windows/macOS/Linux 中转插件；
2. `bdteach-business-api`：业务控制面和数据库迁移；
3. `bdteach-clients`：球友小程序与场馆运营 Web；
4. 既有 GitHub 仓 `Good-Badminton-Local`：多运动 GPU 视觉 API；
5. `Good-Badminton-Evaluation-Platform`：继续作为独立评测仓库。

这不是模型重训、业务功能扩张或生产切换任务，而是保证既有职责不被混放的代码与发布边界整理。

## 已确认事实

- `Good-Badminton` 当前同时包含 GPU API、视觉流水线、场馆网关、业务网关、运营 API 和 Next.js 运营页。
- `customer-inter` 与工作区根目录 `business-db` 有代码/迁移，但各自不是 Git 仓库。
- 评测平台已经是独立 Git 仓库，包含自身的 Next.js 前端与 FastAPI 后端；它不承载场馆业务。
- `Good-Badminton-Tennis-Refactor` 与 `Good-Tennis` 是实验/独立项目，不在本任务迁移范围。
- 主仓和评测仓当前都有用户未提交改动，必须原样保留，不能被移动、覆盖、提交或丢弃。

## 范围

- 从记录的基线提交抽取业务/网关/客户端代码，并以一次只读快照纳入当前尚未 Git 管理的 `customer-inter` 与工作区根目录 `business-db`。GPU 代码在既有 `Good-Badminton-Local` Git 历史中继续演进，使用新的干净本地 worktree 避开当前脏工作区；评测仓保留原仓。
- 建立服务 README、环境样例、忽略规则、版本/来源清单和服务间合同说明。
- 将已有测试随其唯一服务移动或复制为该服务的回归测试；修复 GPU 对 WebUI 的直接导入，使 GPU 服务只输出视觉证据。
- 让评测平台以稳定 GPU HTTP 合同访问 GPU 服务，不读取其源码或业务数据。
- 把业务数据库迁移统一为业务 API 的版本化迁移来源；比较迁移脚本，不自动运行任何数据库变更。
- 为旧主仓建立迁移映射和弃用说明。

## 非范围

- 不部署、不迁移生产数据库、不删除视频/模型/状态、不改动远端 GPU 或场馆设备。
- 不训练、替换或发布模型，不宣称性能、推理质量或流式 SLO。
- 不把 `Good-Badminton-Tennis-Refactor` 或 `Good-Tennis` 合并进生产仓。
- 不推送代码、创建 PR、合并、改写或删除现有远端仓库；GitHub 新仓仅在批准后按私有仓创建，代码推送仍需单独明确授权。

## 目标控制流

```text
bdteach-venue-gateway -> bdteach-business-api -> good-sports-gpu-api
                                  ^                      |
                                  |                      v
                         bdteach-clients        anonymous visual events

Good-Badminton-Evaluation-Platform ------------> good-sports-gpu-api
```

- 场馆网关仅保存场馆设备凭据、RTSP 和本地缓冲，通过签名 HTTPS 调用业务 API；它不连接 GPU。
- 业务 API 是用户、场馆、赛果、权益、设备授权、对象存储签名和任务编排的唯一控制面。
- GPU API 仅处理视频、姿态、球、轨迹、匿名 `track_id`、质量和任务事件；它不读取身份、赛果、积分或运营 UI。
- 客户端只调用业务 API；浏览器和小程序不得获得 GPU、RTSP、对象存储或数据库密钥。
- 评测平台只能调用 GPU 的版本化 HTTP 合同，不能写生产业务数据或把评测结果升格为生产事实。

## 兼容性与安全约束

- 保留 `edge-ingest.v1`、`stream-session.v1`、GPU health 的 `supported_sport_ids` 与已有业务 API 路径，除非后续专项版本化迁移明确废弃。
- 主仓代码提取以基线提交为源；`customer-inter` 与工作区根目录 `business-db` 是非 Git 源，使用已记录 SHA-256 的只读文件快照。GPU worktree 也固定在该基线。已有脏工作区仍留在原目录；在独立的“未提交改动整合”任务通过 diff 审核前，不把这些改动悄悄带入新仓或 GPU worktree。
- 不提交 `.env`、密钥、RTSP、对象存储键、视频、模型权重、运行状态、`node_modules`、`.venv`、`.next` 或打包 ZIP。

## 验收标准

| ID | 结果 | 证据 |
| --- | --- | --- |
| AC-01 | 五个正式职责都有唯一代码仓与 README，评测仓保持独立。 | 本地 Git 根、来源清单、README。 |
| AC-02 | 场馆插件只访问业务 API，不包含 GPU 调用、业务数据库或前端。 | 静态边界测试、网关包测试。 |
| AC-03 | 业务 API 是业务迁移的唯一来源，并不将迁移自动应用到任何数据库。 | 迁移清单校验、无数据库执行记录。 |
| AC-04 | GPU API 没有 `webui`、`business_gateway` 或评测源码依赖；既有 GPU 合同测试通过。 | GPU 边界测试和 API/配置回归。 |
| AC-05 | 小程序与运营 Web 不含 GPU/RTSP/数据库密钥，且可分别构建。 | 搜索门禁、前端构建/类型检查。 |
| AC-06 | 评测平台仍能以 HTTP 合同校验 GPU，且不依赖新仓的本地路径。 | 现有边界检查及后端 GPU client 测试。 |
| AC-07 | 原主仓的用户未提交文件未被修改、暂存、提交、删除或纳入新仓。 | 初末 `git status --short` 对比。 |
| AC-08 | 无服务发布或生产数据变更；每项交付可通过本地 Git commit 回滚。 | 任务记录、提交哈希、无部署记录。 |

## 未决但已给出默认值

- GitHub 所有者：当前已认证账户 `sonylolcai`；只新建 `bdteach-venue-gateway`、`bdteach-business-api` 与 `bdteach-clients` 三个私有仓库。
- GPU 服务复用既有 `Good-Badminton-Local` GitHub 仓；不新建替代远端。一个误建且未推送的私有空 `good-sports-gpu-api` 仓等待具备 `delete_repo` 权限后删除。
- 旧 `Good-Badminton` 在首次迁移中保留为只读迁移源，直至用户确认所有脏改动已逐项迁入；不会在本任务中删除其源文件。
