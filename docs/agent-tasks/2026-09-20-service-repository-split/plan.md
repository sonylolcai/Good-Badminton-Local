# PLAN APPROVAL — 服务与仓库拆分

## 基线与当前风险

| 位置 | 基线 | 状态 | 处理 |
| --- | --- | --- | --- |
| `Good-Badminton` | `932a0d2` / `feat/match-clip-delivery` | 15 项未提交改动 | 不触碰、不暂存；所有新仓从基线提取。 |
| `Good-Badminton-Evaluation-Platform` | `1a7a235` / `feat/evaluation-backend-separation` | 19 项未提交/生成改动 | 不触碰、不暂存。 |
| `Good-Badminton-Tennis-Refactor` | `e07b169` | 实验仓有未提交改动 | 不在范围。 |
| `Good-Tennis` | `88e4fda` | 独立仓有未提交改动 | 不在范围。 |

基线检查：评测仓 `npm run check:boundaries` 通过。主仓的 `tests.test_gpu_code_boundary` 在基线即失败，原因是 `badminton_analysis/pipeline.py` 从 `webui.player_results` 导入；这正是拆分必须消除的现有耦合，不计为本任务引入的失败。

## 新仓与来源清单

| 新仓 | 本地目录 | 来源（只取被 Git 跟踪的源码/测试/文档） | 明确不带入 |
| --- | --- | --- | --- |
| `bdteach-venue-gateway` | `S:\Code Base\MiniProgram\bdTeach\bdteach-venue-gateway` | `deploy/venue-gateway/`、`deploy/venue-gateway-windows/`、`deploy/venue-gateway-macos/` 的源码、安装脚本、测试和文档 | `dist/` ZIP、缓存、现场配置与密钥。 |
| `bdteach-business-api` | `S:\Code Base\MiniProgram\bdTeach\bdteach-business-api` | `business_gateway/`、`operator_api/`、业务 API 对应测试、`deploy/business-server/`；工作区根目录 `business-db/` 的版本化迁移 | 视频、对象存储内容、`.env`、本机 Docker volume。 |
| `bdteach-clients` | `S:\Code Base\MiniProgram\bdTeach\bdteach-clients` | 非 Git 的 `customer-inter/` 只读快照作为 `miniprogram/`；`webui-next/` 基线源码作为 `operator-web/` | `node_modules/`、`.next/`、本地环境文件、旧 Gradio `webui/`。 |
| 既有 `Good-Badminton-Local` | `S:\Code Base\MiniProgram\bdTeach\Good-Badminton-GPU-Service`（干净 worktree） | `api/`、`apps/`、`badminton_analysis/`、`good_badminton_contracts/`、GPU 启动/打包/测试与必要的依赖声明 | `webui/`、业务 API、业务迁移、评测源码、`api_data/`、权重、模型/视频产物。 |
| `Good-Badminton-Evaluation-Platform` | 保持现目录 | 前端、`backend/evaluation_service/`、`backend/evaluation_suites/`、测试 | 场馆业务、业务 DB、GPU 源码。 |

`webui/` 是遗留的本地 Gradio 操作页，不符合“GPU 无 UI”目标。首次拆分不迁入任一正式服务；待评测平台工作台覆盖其必要功能后，再以独立弃用任务删除或归档，避免改变现有操作流程。

## 文件动作清单

| 路径 | 动作 | 目的 | 任务 |
| --- | --- | --- | --- |
| `docs/agent-tasks/2026-09-20-service-repository-split/{state,spec,plan,tasks}.md` | 新增 | 本迁移的协调记录 | T0 |
| `bdteach-venue-gateway/{agent.py,business_gateway/edge_contract.py,linux/,windows/,macos/,tests/,README.md,requirements*.txt,.gitignore}` | 新增 | 网关唯一源码与三平台包 | T2 |
| `bdteach-business-api/{business_gateway/,operator_api/,migrations/,deploy/,tests/,README.md,requirements*.txt,.gitignore}` | 新增 | 业务控制面与唯一迁移来源 | T3 |
| `bdteach-clients/{miniprogram/,operator-web/,README.md,.gitignore}` | 新增 | 球友端和运营端前端 | T4 |
| `Good-Badminton-GPU-Service/{api/,apps/,badminton_analysis/,good_sports_contracts/,deploy/,tests/,README.md,requirements*.txt,.gitignore}` | 在既有远端的干净 worktree 修改 | 无 UI/无业务依赖的 GPU 服务 | T5 |
| `Good-Badminton-Evaluation-Platform/{README.md,AGENTS.md,backend/evaluation_service/gpu_client.py,backend/tests/test_gpu_client.py}` | 按需修改 | 固定 HTTP 合同和仓库边界说明 | T6 |
| `Good-Badminton/{README.md,MIGRATION.md}` | 仅最后新增/修改 | 旧仓迁移说明；不删除源码 | T7 |

每个目录内的文件以相应来源路径在记录的基线提交中的受 Git 跟踪文件为准。T1 会把逐文件 SHA-256 与来源提交写入各新仓的 `MIGRATION_MANIFEST.md`；生成物和受忽略文件不在清单内。

## 任务顺序、提交和回滚

| 任务 | 内容 | 依赖 | 通过条件 | 提交/回滚 |
| --- | --- | --- | --- | --- |
| T0 | 冻结逐文件迁移清单、创建测试计划与特征测试。 | 计划批准 | 清单可复现，脏文件排除。 | 协调文档提交；反向提交即可回滚。 |
| T1 | 新建三个本地 Git 仓，添加私有远端；为既有 GPU 远端建立干净本地 worktree；不推送。 | T0、测试批准 | 三个新仓有 README、忽略规则和来源清单；GPU worktree 指向既有远端。 | 三个新仓 bootstrap 提交；GPU worktree 不改写原工作区。 |
| T2 | 提取场馆网关并验证三平台包一致性。 | T1 | 网关包测试与禁止 GPU 连接测试通过。 | 网关仓单独提交。 |
| T3 | 提取业务 API、运营 API 和唯一数据库迁移来源。 | T1 | 迁移清单唯一且不执行；业务 API 测试通过。 | 业务仓单独提交。 |
| T4 | 提取小程序与运营 Web，清除密钥/本机依赖。 | T1、T3 | 两前端构建/检查通过，只指向业务 API。 | 客户端仓单独提交。 |
| T5 | 在 GPU worktree 消除 WebUI/业务/评测导入。 | T1 | GPU API、sport profile、启动器和边界测试通过。 | 既有 GPU 仓的独立提交。 |
| T6 | 验证评测仓只经 HTTP 调用 GPU。 | T5 | 评测前端边界与后端 GPU client 回归通过。 | 评测仓单独提交，若需要。 |
| T7 | 文档收敛、跨仓冒烟、旧仓迁移说明。 | T2–T6 | AC-01 至 AC-08 均有证据。 | 文档提交；旧源码不删除。 |

## 验证策略

- 网关：既有 `test_package_parity.py`、Windows transport/portability 测试，加一条“无 GPU URL/客户端导入”边界测试。
- 业务：业务 API 的既有单测，迁移来源/禁止自动执行检查，客户端不得传递 GPU 密钥的合同测试。
- GPU：`test_gpu_api`、`test_gpu_sport_profiles`、`test_gpu_pure_launchers`、`test_gpu_code_boundary`；新边界测试必须先 RED 后绿。
- 客户端：小程序现有校验与 `operator-web` 的 lint/build；静态搜索拒绝 GPU、RTSP、数据库凭据。
- 评测：`npm run check:boundaries`、`npm run lint`、`npm run build` 与后端 GPU client/service 测试。
- 集成：使用固定 JSON 请求/响应样本做合同兼容性检查；不调用真实 RTSP、GPU、数据库或生产服务。

## 风险与控制

| 风险 | 控制与回滚 |
| --- | --- |
| 脏工作区被混入/覆盖 | 只从基线提取，初末状态逐项比对；发现重叠即停止。 |
| 代码复制造成永久分叉 | 每个新仓都有来源清单；旧仓只保留迁移说明，后续变更只进入新仓。 |
| 网关/业务/GPU 合同被破坏 | 先写合同特征测试，保留既有版本化路径，不做静默重命名。 |
| 误泄露密钥、视频、权重或生成包 | `.gitignore`、暂存前扫描与逐文件 manifest；不提交生成物。 |
| 远端仓被意外公开或覆写 | 仅为三个新服务创建私有空仓；GPU 复用既有私有远端；本任务不推送。 |
| 数据库迁移被误执行 | 只复制与校验；本任务中不运行 Docker/psql/部署脚本。 |

## 批准记录

用户已确认初始计划（“批准”）及修订方案（“ok”）：复用既有 `Good-Badminton-Local` 作为 GPU 远端，保持 `Good-Badminton-Evaluation-Platform` 不变；只在 `sonylolcai` 名下新建场馆网关、业务 API、客户端三个私有仓。首次迁移只提取基线代码，用户现有未提交改动保持在原仓，之后另行做逐文件整合。空的 `good-sports-gpu-api` 远端待有 `delete_repo` 权限时删除，不影响迁移。
