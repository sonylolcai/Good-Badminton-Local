# 任务清单

| 任务 | 状态 | 依赖 | 文件 | 验证 | 提交 |
|---|---|---|---|---|---|
| MGP-01 | passed | plan/test approval | 多运动 API、profile、GPU API 测试 | 双运动会话、错误 profile、完整视频与取消 | `a87addd` |
| MGP-02 | passed | MGP-01 | `webui/remote_gpu.py`、评测 GPU client 与测试 | 共享 health 的目标运动校验 | `de063f9`、`17e64b5` |
| MGP-03 | passed | MGP-01 | 打包/刷新/启动脚本、ZIP/Docker 测试、运行文档 | 白名单 ZIP、候选预检/自动回滚、本地 Docker health | uncommitted |

## 当前门禁

测试方案已批准；MGP-01、MGP-02、MGP-03（含 AC-07 本地 Docker 容器验收）的本地门禁已通过。T-MGP-10 仍需单独授权真实 GPU 部署。
