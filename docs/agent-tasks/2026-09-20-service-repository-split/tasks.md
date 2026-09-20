# 任务清单

| Task ID | Status | Depends on | Owned files | Expected change | Verification | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | passed | PLAN APPROVAL | 本任务目录、迁移 manifest、特征测试 | 已冻结 Git 与非 Git 源、保护脏改动并取得 RED | `tests.test_service_repository_layout` RED-01；`source-manifest.md` | 待与首个协调提交一起记录 |
| T1 | in_progress | T0, TEST APPROVAL | 三个新仓根目录、GPU worktree | 初始化私有远端与干净 GPU worktree | 远端/README/ignore/worktree 检查 | 新仓各自 bootstrap |
| T2 | pending | T1 | `bdteach-venue-gateway/**` | 提取三平台网关 | package parity、边界测试 | 网关提交 |
| T3 | pending | T1 | `bdteach-business-api/**` | 提取业务控制面与迁移 | 业务/迁移合同测试 | 业务提交 |
| T4 | pending | T1, T3 | `bdteach-clients/**` | 提取小程序与运营 Web | 小程序校验、lint/build、密钥门禁 | 客户端提交 |
| T5 | pending | T1 | `Good-Badminton-GPU-Service/**` | 在既有 GPU 仓 worktree 断开 UI/业务依赖 | GPU API/profile/launcher/boundary 测试 | GPU 提交 |
| T6 | pending | T5 | 评测仓 GPU client/测试/说明 | 保持纯 HTTP GPU 调用 | boundary、GPU client/service 测试 | 评测提交（如有） |
| T7 | pending | T2, T3, T4, T5, T6 | 迁移说明、结果记录 | 跨仓收敛与旧仓路由 | AC 证据矩阵、状态对比 | 文档提交 |
