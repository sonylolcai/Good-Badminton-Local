# Good-Badminton Agent Guide

## 目标与职责

这是固定机位羽毛球赛后复盘的主仓，同时承载共享多运动 GPU API。首要产物是可追溯的视觉证据和赛后资料，不是实时裁判系统。

- 人体、球、球拍和轨迹属于视觉分析；用户身份、确认赛果、积分、报告和场馆运营属于业务侧。
- 击球/技术标签应以球的位置、轨迹和落点为主，Pose 只提供球员上下文和候选窗口。
- 不足以判断时输出 `unknown`；`predicted` 羽毛球轨迹只能用于连续展示，不能当作击球或失误事实。

## 代码地图

- `main.py`：离线 CLI 入口；`badminton_analysis/`：视频、球场、检测、追踪、数据和可视化主流程。
- `api/`：GPU-only FastAPI 边界和流式会话；`apps/`：运动专属 GPU 入口；`good_badminton_contracts/`：共享数据合同。
- `business_gateway/`、`business-db/`、`operator_api/`：业务、数据和运营端边界；`webui/`、`webui-next/`：本地操作界面。
- `tests/`：按功能拆分的 `unittest` 回归；`deploy/`：源码包发布、启动和回滚脚本。

## 开发与验证

- 优先用项目 `.venv\\Scripts\\python.exe` 运行针对性 `unittest`，例如：
  ` .\\.venv\\Scripts\\python.exe -m unittest tests.test_gpu_api tests.test_gpu_sport_profiles tests.test_gpu_pure_launchers tests.test_gpu_code_boundary -v`
- GPU 调用方先确认 `/api/v1/health` 的 `supported_sport_ids` 包含目标 `sport_id`；旧的单一 `sport_id` 字段仅作兼容信息。
- 评估真实视频、固定机位、远端球员、流式会话或性能时，保留任务账本、事件和阶段 trace；监听端口成功不等于任务成功。

## 发布与安全边界

- 远端 GPU 更新默认通过上传完整 ZIP，不依赖服务器拉取 Git；发布包包含代码、允许的模型和 manifest，不含密钥、`api_data` 或运行时状态。
- 发布或刷新前后验证哈希与健康接口；保留 `/root/good-badminton-gpu-api-state`，支持回滚。
- 本地测试或包构建不代表 GPU、吞吐、20 分钟稳定性或 `streaming_slo_proven` 已验证。

## Windows 开发文档

- 总索引：`S:\Software Tool\Obsidian\AI羽毛球赛后复盘与训练助手\总索引.md`
- 记录需求、讨论和结果时更新既有分类与索引；不要创建并行文档集。Mac 路由待补充。
