# 任务清单

| Task ID | Status | Depends on | Owned files | Expected change | Verification | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | passed | PLAN APPROVAL | 本任务目录、迁移 manifest、特征测试 | 已冻结 Git 与非 Git 源、保护脏改动并取得 RED | `tests.test_service_repository_layout` RED-01；`source-manifest.md` | 待与首个协调提交一起记录 |
| T1 | passed | T0, TEST APPROVAL | 三个新仓根目录、GPU worktree | 初始化私有远端与干净 GPU worktree | 远端、README、ignore 与 worktree 已核验 | `64c8bcf`、`e98c15e`、`549bd94`；GPU 为既有远端 worktree |
| T2 | passed | T1 | `bdteach-venue-gateway/**` | 提取三平台网关 | 2 package parity + 8 Windows transport/portability 通过 | `ca8aff0` |
| T3 | passed | T1 | `bdteach-business-api/**` | 提取业务控制面与迁移 | 28 API/backoffice/edge-ingest 测试、编译和边界扫描通过 | `05175b2`、`f976505` |
| T4 | passed | T1, T3 | `bdteach-clients/**` | 提取小程序与运营 Web | 小程序 typecheck、运营 Web lint/build、客户端密钥边界通过 | `2bb075b` |
| T5 | passed | T1 | `Good-Badminton-GPU-Service/**` | 在既有 GPU 仓 worktree 断开 UI/业务依赖 | 42 GPU 检查通过，完整权重包因输入缺失跳过 | `f0399e2`、`6f41cb7` |
| T6 | skipped | T5 | 评测仓 GPU client/测试/说明 | 保持纯 HTTP GPU 调用 | 未改评测源码；基线边界检查已通过 | 不适用 |
| T7 | passed | T2, T3, T4, T5, T6 | 迁移说明、结果记录 | 跨仓收敛与旧仓路由 | 评测边界与四仓协调检查通过 | 待本次文档提交 |
