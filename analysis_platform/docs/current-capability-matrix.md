# 三模块边界、权威仓库与现有能力基线

记录时间：2026-09-14

本文记录执行清单中的拆分基线与当前实现状态。本轮不改变任何业务功能、GPU 算法或现有接口。

## S-00：冻结结论

### 唯一权威代码源

| 项目 | 决定 | 现场证据 |
|---|---|---|
| `S:\Code Base\MiniProgram\bdTeach\Good-Badminton` | 业务、GPU 与 Python 分析引擎的权威仓库 | 远端为 `sonylolcai/Good-Badminton-Local.git`；当前工作分支 `feat/tennis-gpu-modes`；基线提交 `7dbc65031a33871839c3dd74429d274e25908eb3` |
| `S:\Code Base\MiniProgram\bdTeach\Good-Badminton-Evaluation-Platform` | 独立 Next.js 评测项目 | 自有 `package.json`、构建命令、3100 端口与页面入口；不属于业务 `webui-next` 路由树 |
| `S:\Code Base\MiniProgram\bdTeach\Good-Badminton-Tennis-Refactor` | 只读实验 worktree，不接受本轮提交 | 同一远端；分支 `feat/tennis-vision-contract`；存在未提交改动，不能作为可复现基线 |
| `S:\Code Base\MiniProgram\bdTeach\Good-Tennis` | 独立网球项目，只读参考 | 独立远端 `yo-WASSUP/Good-Tennis.git`，不属于本轮羽毛球平台拆分 |

`origin/main` 比当前权威工作分支少 63 个提交，不能直接作为 Python 引擎改动的开发基线。后续 Agent 不得从两个只读实验目录复制未审查代码。

### 三模块硬边界

| 模块 | 归属 | 本轮改动规则 |
|---|---|---|
| 数据分析评测平台 | 独立 Next.js 项目 `Good-Badminton-Evaluation-Platform/`；经 `analysis_platform/api.py` 调用现有 Python 引擎与 `evaluation/` | 唯一主动建设模块；处理匿名 Case、Dataset、Run、指标、对比和人工真值版本 |
| GPU 服务 | `api/`、推理与流式运行模块 | 冻结模型、参数、状态语义和部署；后续只允许导入路径/合同适配及回归测试 |
| 业务服务 | `webui-next/`、`operator_api/`、`business_gateway/`；旧 Gradio 业务页在兼容期保留 | 冻结现有功能；身份、场馆、对局、交付、教练档案不得进入分析平台 |

## S-01：现有能力留存矩阵

### 旧 Gradio WebUI

| 当前入口/能力 | 当前实现 | 最终归属 | 迁移处理 | 验收方式 |
|---|---|---|---|---|
| 经营看板 | `webui/operator_backoffice.py` | 业务服务 | 保留，不改功能 | 页面可读取既有运营指标和状态 |
| 场馆、场地、二维码 | `webui/operator_backoffice.py` | 业务服务 | 保留；长期由 `webui-next` + `operator_api` 承载 | 场馆/场地新增编辑、二维码生命周期原 smoke 通过 |
| 活动与社区动态 | `webui/operator_backoffice.py` | 业务服务 | 保留，不迁入分析平台 | 活动、动态列表和保存能力不变 |
| 比赛与认领 | `webui/operator_backoffice.py` | 业务服务 | 保留，不迁入分析平台 | 最近比赛、认领/交付相关能力不变 |
| 视频业务任务与 GPU 运维 | `webui/operator_backoffice.py` | 业务服务 | 保留当前运营控制；不作为实验参数台 | GPU 状态、任务账本、取消与受控启停行为不变 |
| 权限与审计 | `webui/operator_backoffice.py` | 业务服务 | 保留，不迁入分析平台 | 人员、场馆角色、审计读取/保存行为不变 |
| 视频/模板上传与球场自动/手动标定 | `webui/app.py` | 数据分析评测平台 | 迁入 Case/Run 工作台；先保持交互 | 同一视频、模板和四角输入可创建分析 Run |
| 运动模式、Pose 尺寸、采样率、球检测器、追踪器、远端球员增强 | `webui/app.py` | 数据分析评测平台 | 作为 Experiment/Run 参数保存并计算参数指纹 | 历史 Run 可还原完整参数，不能静默使用页面默认值 |
| 本地 CPU、整视频远端 GPU、2 秒直连分片 | `webui/app.py` | 数据分析评测平台（研究入口）/GPU 服务（执行） | 平台通过公开合同调用；不复制 GPU 内部逻辑 | 三种既有开发路径的成功/错误语义可对照复现 |
| GPU 运动身份校验、运行、中断、结果恢复 | `webui/app.py` | 数据分析评测平台（研究任务控制） | 迁入 Run 操作；生产 GPU 运维仍留业务侧 | 不重复提交；中断和恢复保留持久任务身份 |
| 标注视频、截图、逐 Track 证据、metadata、detections、TrackNet CSV | `webui/app.py` | 数据分析评测平台 | 迁入 Artifact 查看/下载 | 每项产物关联 `run_id`、路径和 SHA-256 |
| 球路复核、逐帧跳转、击球增删改、回合终止人工确认 | `webui/app.py`、`webui/shot_review.py` | 数据分析评测平台 | 迁入 Annotation Version/Failure Case；禁止覆盖原始检测 | 人工修改产生新版本；原始 `detections.jsonl` 不变 |
| 赛后 Track ID 合并、`person_id`、`team_id` 绑定 | `webui/app.py` | 业务服务 | 保留业务侧；平台只使用匿名 `track_id` | 身份数据不出现在分析任务合同中 |
| 身高体重、能量估算、教练档案、纵向反馈 | `webui/app.py` | 业务服务 | 原功能保留；不进入评测平台 | 原数据与页面行为不变，GPU 不接收身份/身体数据 |
| AI 宣传视频与业务表现报告 | `webui/app.py`、`business_gateway/post_match.py` | 业务服务 | 继续消费已完成分析产物；不触发重复推理 | 生成失败不改变 GPU 分析成功状态 |
| “分析任务历史”中的业务账本与远端对账 | `webui/app.py` | 业务服务 | 保留为业务任务历史；新平台另建 Experiment/Run 历史 | WebUI 重启后不重复上传，既有任务可对账 |
| 后台日志控制台 | `webui/app.py` | 各服务自己的运维视图 | 不整体迁移；平台只显示本 Run 日志 | 不泄露其他服务密钥或无关日志 |

### 当前业务服务能力

| 边界 | 已存在能力 | 本轮处理 |
|---|---|---|
| `webui-next/` | Dashboard、Venues & Courts、GPU Services 三个页面入口 | 冻结 UI，不重做页面 |
| `operator_api/main.py` | readiness、dashboard、租户/场馆/场地、球场运行、标定、采集、GPU 转发、GPU 状态/配置/操作 | 冻结 API 语义 |
| `business_gateway/dev_api.py` | 固定视频、候场、比赛开始/结束、交付窗口、流式回放、认领、个人汇总、候选照片 | 冻结现有开发业务闭环 |
| `business_gateway/edge_api.py` | 边缘设备心跳、会话、顺序分片、完成、状态、预览、GPU 事件 | 冻结边缘数据面与控制面语义 |
| `business_gateway/post_match.py` | 从匿名分析产物生成业务解释 | 保留为分析到业务的发布后接口；正式 Run 不允许其回写原始产物 |

### 当前 GPU 服务能力

| 边界 | 已存在能力 | 本轮处理 |
|---|---|---|
| `api/gpu_stream_app.py` | 独立、运动类型固定、匿名视觉观测的纯 GPU 组合入口 | 作为目标 GPU 边界，不改功能 |
| `api/stream_routes.py` | health、创建会话、顺序提交分片、状态、事件、trace、候选照片、完成、取消 | 冻结 `stream-session.v1` |
| `api/app.py` | 整视频任务提交、幂等恢复、状态、取消、结果、性能 trace、产物下载；同时挂载流式路由 | 兼容保留，后续仅处理对 `webui.pipeline` 的反向依赖 |
| `api/jobs.py` | 整视频 Worker 当前导入 `webui.pipeline.run_analysis` | 已确认拆分接缝；本阶段只记录，不修改 |

### 已存在评测能力

| 套件 | 入口 | 当前可复用内容 |
|---|---|---|
| 远端球员 | `evaluation/far_player/` | 视频发现、标注、检测运行、召回/误检/连续漏检比较 |
| 双打 | `evaluation/doubles/` | 四人召回、ID、轨迹中断、遮挡恢复、击球归属评测 |
| 羽毛球 A/B | `evaluation/shuttle_tracknet_ab/` | 人工真值、YOLO/TrackNet 输入适配、定位与连续漏检指标 |
| 性能/流式 | `evaluation/performance/`、`evaluation/streaming/` | 性能基线、门禁、trace、可靠性和流式 benchmark |

## 黄金样本与指纹

这些对象只作为拆分前基线，不自动代表人工真值或生产精度已经验收。

| 类型 | 路径/ID | SHA-256 或状态 | 用途 |
|---|---|---|---|
| 可重复视频输入 | `videos/demo.mp4` | `b60cc98c5c5066e31eed777c6cf92bc2d4899d210a5f7285d7358ee7c63d7b26` | 本地 smoke 与标注流程输入；14.2 秒、1280×720、30 FPS |
| 小型合成视频 | `samples/simulated_rally_10s/stick_match_10s.mp4` | `f63f3b9b8fc4cc2d34f753211ba326768ddc3e296ecaf40bac4fc9d565bf121c` | 不依赖真实比赛的流程 smoke |
| 最近完整成功视频与产物 | 视频见该产物 `metadata.json`；产物为 `outputs/remote_jobs/20260910_154638_588604/` | 视频 SHA-256 `279a7967b789c60f2f52a8ebbbadb112f686baca54fc039049cc9b8e8e8e0394`；业务任务 `cdf84b0dfb3349a68c287ed1f51a2106`，状态 `succeeded` | 原始视频当前仍在 Gradio 临时目录，可作本次人工对照；尚未纳入权威数据集目录，不能当作长期可复现 Run |
| 已有性能基线 | `evaluation/performance/rtx_3090_production_v1.json` | `021ef3fb6ca1b67c1cc8a40f586d58298ff81a35e91d421976ded8103a0f3f59` | 评测报告/门禁 fixture |
| 失败任务 | `outputs/business_tasks/231f091674b44843ad7e54e588164483.json` | `failed`；SHA-256 `ceafe835f4fe992cd59ea2693d9c9c6e449c042f235e4933e3f7edce2518aeac` | 验证失败 Run 和 TrackNet 错误不会变成伪成功 |
| 取消任务 | `outputs/business_tasks/78c7c90ed3d947aa93e54a32ce5f4dfc.json` | `cancelled`；SHA-256 `c9b315f684814140d49229be44d8fd81f69844a59b7f8bcb99d61d8163151d6c` | 验证取消不是失败或成功，且保留状态历史与错误原因 |

平台只能登记当前可读取且可哈希的对象，不能凭历史列表补造黄金样本。

## S-02 合同冻结摘要

权威实现：`analysis_platform/contracts.py`；合法/非法样例在 `tests/fixtures/analysis_contract_*.json`。

三个合同类型：

1. `analysis_task`：不可变研究输入，携带 Case/Dataset/Run、代码/模型/参数指纹、输入 SHA-256、相对路径和匿名参数。
2. `gpu_analysis_result`：终态 GPU 结果，携带相同输入身份、明确状态、错误和带 SHA-256 的相对产物路径。
3. `published_result_ref`：业务服务可读取的正式结果引用，携带结果 manifest、源 GPU 结果和发布时间；不携带业务身份。

共同约束：

- Schema 固定为 `analysis-boundary.v1`，未知字段拒绝。
- `input_sha256`、模型/参数/产物指纹使用小写 SHA-256；代码指纹使用 Git 提交，并可附加 dirty patch 指纹。
- 路径必须相对且不能包含 `..`，防止跨 Run 读取或覆盖文件。
- 参数递归拒绝 `person_id`、`team_id`、`score`、`venue_id` 等业务字段。
- 成功结果必须有产物且 `error=null`；失败/取消/部分结果必须保留结构化错误。
- `contract_sha256()` 是冻结身份；P-B1 的存储实现必须以它执行写后校验和防覆盖。

## 后续任务入口

`S-00`、`S-01`、`S-02` 完成后，才允许并行启动 `P-A1`～`P-E1`。GPU 和业务冻结回归属于 `P-E1`，本轮没有借前置任务修改这些服务。

## 验证记录

```powershell
.venv\Scripts\python.exe -m unittest tests.test_analysis_platform_contracts tests.test_stream_session_contract tests.test_gpu_api tests.test_operator_api tests.test_operator_backoffice
.venv\Scripts\python.exe -m compileall -q analysis_platform tests\test_analysis_platform_contracts.py
git diff --check
```

2026-09-14 结果：42 个测试通过，编译检查和差异格式检查通过。该结果证明最小合同与现有边界测试兼容，不代表真实摄像机、远端 GPU 或完整业务端到端已经重新验收。

## 当前实现进度（2026-09-14）

- `analysis_platform/runner.py` 已承接无 UI 分析执行器；`webui.pipeline` 为同模块兼容别名。
- `analysis_platform/gpu_client.py`、`analysis_platform/review.py` 和 `analysis_platform/stream_speed_summary.py` 已承接研究 GPU 客户端、人工复核与流式汇总核心；旧 `webui` 路径保持模块级兼容别名。
- 本地 JSON/JSONL 存储已支持 Dataset、Dataset Version、Case、Annotation Version、Experiment、不可变 Run、Metric Result、Artifact 和 Baseline Alias。
- 四套现有评测器已统一适配；比较只允许相同数据版本、指标版本和单位，性能指标额外要求硬件与执行模式一致。
- Python CLI 入口为 `.venv\Scripts\python.exe -m analysis_platform`；评测 API 入口为 `.venv\Scripts\python.exe -m analysis_platform.api`，默认监听 `127.0.0.1:8010`。
- 独立 Next.js 项目位于 `S:\Code Base\MiniProgram\bdTeach\Good-Badminton-Evaluation-Platform`，默认监听 `127.0.0.1:3100`。它拥有独立 `package.json`、构建、导航和部署入口，不是业务 `webui-next` 的页面。
- Next.js 平台已有视频分析（本地、整视频远程 GPU、2 秒直连分片）、数据/标注、Experiment/Run、对比门禁、球路人工复核和研究 Run 历史页面；自动球场角点提取、GPU 身份校验、运行中断、分片恢复和产物下载校验已通过评测 API 接入。

`I-04` 已完成默认入口切流：旧 `webui` 本身就是业务服务，身高体重、能量估算、赛后身份绑定、教练档案、纵向反馈、宣传视频和业务任务历史没有跨服务迁移，只是在默认页面中从原先混合的“分析工作台”重新组合为业务页面。默认业务 WebUI 不再挂载分析运行、球路复核或研究分析历史；这些入口只由独立数据分析评测平台提供。旧混合页面实现暂留为未挂载的回退代码，待 `S-04` 清理，不影响当前运行入口。

最新验证：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.venv\Scripts\python.exe -m compileall -q analysis_platform webui api
git diff --check
```

结果：345 个 Python 测试全部通过；独立 Next.js 项目 `npm run lint`、`npm run build` 通过，评测 API、Next.js 总览、工作台、数据和复核页面真实启动烟测均返回 HTTP 200。该验证未实际提交远端 GPU 视频，也不替代真实数据集准确率验收。
