# TEST APPROVAL — 服务与仓库拆分

## 测试环境与固定输入

- 协调测试在 `Good-Badminton` 的项目虚拟环境运行；不调用摄像头、数据库、真实 GPU 或网络服务。
- 初始迁移源固定为 `932a0d2a54d4bc8e5a9d7341118c8d6ea8f0e854`；未提交工作区文件不是迁移输入。
- 评测平台基线固定为 `1a7a235e9b8b10a37ceb9eae419d36454b3dcb9e`。
- 每项服务提取后先跑本仓测试，再跑合同和跨仓静态检查。

| Test ID | Acceptance ID | Level | 场景 | 期望结果 | 命令/环境 | Pre-code evidence | Final evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TS-01 | AC-01 | 静态 | 三个新服务仓与既有 GPU worktree 存在，评测仓未被复制。 | 三个新本地 Git 根、既有 GPU 远端的干净 worktree、现有评测仓唯一存在。 | `python -m unittest tests.test_service_repository_layout` | RED：业务/客户端/GPU worktree 未就绪。 | 绿；各仓来源 manifest 完整。 |
| TS-02 | AC-02 | 静态/单测 | 网关源码只含业务 API 通讯边界。 | 无 GPU URL、GPU 客户端或业务 DB/前端依赖。 | 网关仓 `unittest` + layout test | RED：网关仓不存在。 | 包一致性与边界测试绿。 |
| TS-03 | AC-03 | 静态/单测 | 业务迁移由业务 API 唯一拥有。 | 迁移在业务仓且只做清单比对，不执行 Docker/psql。 | 业务仓 migration test | RED：业务仓不存在。 | 迁移重复/缺失检查绿。 |
| TS-04 | AC-04 | 单测 | 既有 GPU 仓的干净 worktree 无 UI、业务或评测源码导入。 | `api/`、视觉流水线无禁用导入；既有 GPU 合同仍通过。 | GPU worktree `unittest` | RED：当前基线 `pipeline.py` 导入 `webui.player_results`，GPU worktree 尚未建立。 | API/profile/launcher/boundary 全绿。 |
| TS-05 | AC-05 | 静态/build | 客户端仅使用业务 API，且无 GPU、RTSP、DB 凭据。 | 小程序校验、运营 Web lint/build 均通过。 | 客户端 npm scripts + secret scan | RED：客户端仓不存在。 | 两应用构建绿，扫描无命中。 |
| TS-06 | AC-06 | 单测/build | 评测前后端只按 HTTP 合同访问 GPU。 | 评测前端边界、后端 client/service 测试通过。 | 评测仓 npm + unittest | 基线 `check:boundaries` 绿。 | 相关检查保持绿。 |
| TS-07 | AC-07 | Git 比对 | 原仓用户改动未受迁移影响。 | 初末 `git status --porcelain` 的既有路径/内容相同。 | Git path/status/hash manifest 对比 | 当前脏路径已记录。 | 路径与哈希一致。 |
| TS-08 | AC-08 | 审计 | 没有发布、推送、数据库执行或远端 GPU 调用。 | 任务状态和 Git/命令记录没有这些操作。 | 审计任务记录 | 不适用。 | 审计绿。 |

## 新测试文件

`tests/test_service_repository_layout.py` 是协调层的最小特征测试。它目前只验证：

1. 三个新目标本地仓及既有 GPU worktree 的关键服务目录必须存在；
2. `Good-Badminton-GPU-Service` 内不能出现 UI、业务或评测源码导入；
3. 场馆网关不能直接连接 GPU；
4. 客户端仓不能出现 GPU/RTSP/数据库敏感配置名。

它不模拟视频、不探测服务端口、不读取 `.env`，也不把目录存在误当作部署成功。

## 基线与 RED

- 已通过：评测仓 `npm run check:boundaries`。
- 已失败：主仓现有 GPU 边界测试因 `badminton_analysis/pipeline.py` 直接导入 `webui.player_results` 失败，且该导入存在于 `HEAD`，不是本任务造成。
- 待执行：修订后的协调测试应因业务/客户端/GPU worktree 尚未就绪、现有 GPU/UI 耦合尚未消除而失败。该失败将直接证明修订迁移目标尚未实现。

## 后续回归范围

- 网关：三平台 package parity、portability、edge transport；
- 业务：运营 API、edge contract、迁移清单；
- GPU：API、sport profiles、纯启动器、代码边界；
- 客户端：小程序现有校验、运营 Web lint/build；
- 评测：网络边界、lint/build、GPU client/service 单测。

任何需要真实摄像头、远端 GPU、外部对象存储、生产数据库或 GitHub 推送的验证，均标为环境验证，不能以本地测试替代。
