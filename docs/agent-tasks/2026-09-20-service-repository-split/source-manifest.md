# 提取源清单

## Git 基线

| 服务 | 来源提交 | 来源路径 | 跟踪文件数 | 排除 |
| --- | --- | --- | ---: | --- |
| 场馆网关 | `932a0d2a54d4bc8e5a9d7341118c8d6ea8f0e854` | `deploy/venue-gateway*` | 26 | `dist/` 生成 ZIP、缓存与配置。 |
| 业务 API | 同上 | `business_gateway/`、`operator_api/`、`deploy/business-server/` | 52（不含独立 DB 快照） | 运行数据、密钥、视频。 |
| 客户端运营 Web | 同上 | `webui-next/` | 32 | `.next/`、`node_modules/`、`.env.remote.local`。 |
| GPU API（复用既有远端） | 同上 | `api/`、`apps/`、`badminton_analysis/`、`good_badminton_contracts/` | 59（不含部署/测试，后续按测试依赖补入） | `webui/`、`api_data/`、权重、视频、产物。 |
| 评测平台 | `1a7a235e9b8b10a37ceb9eae419d36454b3dcb9e` | 保持原仓 | 不复制 | 不变。 |

Git 源中的逐文件 blob ID 可由下列命令再生，不读取工作区脏文件：

```powershell
git ls-tree -r 932a0d2a54d4bc8e5a9d7341118c8d6ea8f0e854 -- <source path>
```

三个新仓在 T1 创建时会写入 `MIGRATION_MANIFEST.md`；GPU 则在 `Good-Badminton-GPU-Service` 干净 worktree 中记录同样的来源信息：源相对路径、源提交或快照、源 SHA-256、目标相对路径和目标 SHA-256。只有双端摘要相等的文件才会进入首个提取提交。

## 非 Git 只读快照

| 服务 | 来源路径 | 文件数 | 排除 |
| --- | --- | ---: | --- |
| 业务 API 数据库迁移 | `S:\Code Base\MiniProgram\bdTeach\business-db` | 8 | `.env`、Docker volumes。 |
| 球友小程序 | `S:\Code Base\MiniProgram\bdTeach\customer-inter` | 66 | `node_modules/`、`release/`、`miniprogram_npm/`。 |

这些目录没有 Git 基线。T1 将先计算并记录每个被提取文件的 SHA-256，再复制；原文件不会修改、移动、暂存或删除。

## 原仓保护快照

主仓在提取开始前的用户文件：`api/app.py`、`badminton_analysis/pipeline.py`、`badminton_analysis/system.py`、`badminton_analysis/tracking/player.py`、`docs/agent-tasks/2026-09-18-match-clip-analysis/*`、`good_badminton_contracts/detections_reader.py`、`tests/test_annotation_sampling.py`、`tests/test_gpu_api.py`、`tests/test_performance_slo.py`、`tests/test_spatial_player_positions.py`、`tests/test_webui_player_results.py`、`webui/player_results.py`、未跟踪 `AGENTS.md`。

本任务新增的协调文件仅为 `docs/agent-tasks/2026-09-20-service-repository-split/*` 与 `tests/test_service_repository_layout.py`；它们不与上述用户文件重叠。
