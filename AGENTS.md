# Good-Badminton-Tennis-Refactor Agent Guide

## 定位

这是共享多运动 GPU 与网球适配的重构实验仓。它与主仓结构相近，但任何修复或功能都必须单独验证、评审并明确迁移，不能默认回流 `Good-Badminton`。

## 代码地图

- `api/`：多运动 FastAPI、流式运行时和会话边界；`apps/badminton_gpu/`、`apps/tennis_gpu/`：运动专属 GPU 启动入口。
- `badminton_analysis/`：固定机位视觉主流程；`business_gateway/`、`operator_api/`：业务/运营接口；`webui/`、`webui-next/`：本地操作界面。
- `tests/`：重构验证；`deploy/`：GPU 发布和启动脚本；`docs/`：本仓实验与迁移资料。

## 不可突破的边界

- 所有请求、manifest 与结果必须保留允许列表内的 `sport_id`；调用方只在健康接口声明支持目标运动时发起分析。
- 人体/球/轨迹是视觉证据，身份、赛果和经营解释仍在业务层；没有证据必须保留 `unknown`。
- 本仓的本地、CPU 或接口测试不代表远端 GPU 推理、模型质量、长视频稳定性或生产部署已完成。

## 开发与验证

- 在已激活的项目 Python 环境中运行最小相关回归；GPU 合同变更至少覆盖：
  `python -m unittest tests.test_gpu_api tests.test_gpu_sport_profiles tests.test_gpu_pure_launchers -v`
- 模型权重与密钥不进 Git。远端发布使用带 manifest 与 SHA-256 的上传 ZIP，并保留运行状态与上一个可回滚版本。

## Windows 开发文档

- 总索引：`S:\Software Tool\Obsidian\AI羽毛球赛后复盘与训练助手\总索引.md`
- 记录本仓的验证结论和迁移判断；不要把实验结果表述为主仓或生产能力。Mac 路由待补充。
