# Good-Badminton-Tennis-Refactor Agent Guide

## 定位

这是共享多运动 GPU 与网球适配的重构实验仓。它与主仓结构相近，但任何修复或功能都必须单独验证、评审并明确迁移，不能默认回流 `Good-Badminton`。

## 羽毛球解析默认与一致性

- 新建羽毛球解析默认使用优化版 `tracknet_v3`；只有明确关闭球检测或做模型对照时才选择 `none` / `yolo`。不要从旧成功任务继承 YOLO 后当作当前默认。
- 模型路径与默认值以 `badminton_analysis/inference_policy.py` 为准。平台调用标准 GPU API；版本、权重、执行流程或回执不符时失败，禁止静默回退。
- `feat/shuttle-accuracy-iteration` 分支中的球体实验采用沿四角左右边线外扩的空中走廊：本机当前每侧为近端场宽的 20%，上方保留高球区域，实际比例以私密 GPU 配置为准。不要只裁地面梯形，也不要恢复成按近端宽度向上覆盖的大矩形。参数 `GOOD_BADMINTON_SHUTTLE_LATERAL_MARGIN_RATIO` 由私密 GPU 配置管理，须在执行流程签名和产物中保留。
- 球体准确率实验已停放在上述专属分支；当前工作区保持优化 TrackNet 基线，不启动准确率迭代。
- 整段与流式有独立执行流程签名，不能把两者的窗口策略和覆盖率混作同一口径。运行与核验见 `docs/GPU_INFERENCE_PARITY.md`。

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

## Local configuration and credentials

- Keep real configuration and credentials under this project's ignored `.secrets/` directory. Follow `docs/LOCAL_CONFIGURATION.md` and `config/local-secrets.json`; edit canonical private files, not duplicate copies.
- Public examples contain placeholders only. Never put permanent secrets in source, frontend/public environment variables, logs, Git, Docker contexts or source release archives. Do not print credential values while inspecting or verifying configuration.
- Run `python3 tools/local_secrets.py check` before delivery. The local pre-commit hook checks actual Git index blobs. Preserve existing user changes.
- Server credentials are deployed separately after the user authorizes the target connection. Preserve stable master/hash keys, back up remote configuration, validate after reload, and retain rollback. Local templates alone do not mean a service consumes their variables.
