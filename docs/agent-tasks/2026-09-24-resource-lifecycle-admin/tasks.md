# 任务清单

| Task ID | Status | Depends on | Owned files | Expected change | Verification | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | passed | - | 两仓 migrations、resource modules | 建立资源、位置、认证、RBAC 和保留策略数据模型 | 迁移测试 + 状态机单测 | `35d4597`, `5708407` |
| T02 | passed | T01 | `api/jobs.py`, `api/app.py`, `api/stream_sessions.py` | GPU 取消与资源删除分离；视频-only 与 full delete | GPU/stream 单测 | `e8a1d15` |
| T03 | passed | T01,T02 | 评测 backend、HistoryClient、tests | 评测两种删除、远程同步、引用计数和重试 | 评测资源/API/服务测试 + lint/build | `93e6395` |
| T04 | passed | T01 | `operator_api/services/auth.py`, `operator_api/main.py`, auth tests | bootstrap、账号密码、改密、会话、RBAC | auth/API 单测 | `23dea8a` |
| T05 | passed | T01,T02,T04 | business resource service/API、edge API、tests | 业务视频上传、关联、手动解析和两种删除 | lifecycle/API/edge tests | `2b29511` |
| T06 | passed | T03,T05 | retention worker、Compose、env、settings API | 默认关闭的中央 7 天每日视频清理 | 时钟边界/锁/跨服务契约测试 | `8d9b52f`, `22ce89c` |
| T07 | passed | T04,T05,T06 | Next 登录、用户、资源、设置页面 | 管理员可见且按权限收敛的 UI | lint/build + 浏览器登录门禁 | `c4c350d`, `b3f4be5` |
| T08 | passed | T01-T07 | docs、state、result | 全回归、验收矩阵、部署说明与 Obsidian 路由更新 | 两仓回归与差异审查 | 本提交 |

## 当前门禁

- `PLAN APPROVAL` 已于 2026-09-24 获得（用户原文：`批准`）。
- `test-plan.md` 已完成；业务仓 8 项、评测仓 4 项有效 RED 已记录。
- `TEST APPROVAL` 已于 2026-09-24 获得（用户原文：`批准`）。
- T01-T08 已完成本地实现和验证；没有推送、部署、生产迁移或真实数据删除。
- 业务仓全量 362 项中 361 项通过；唯一失败是基线已有的 `badminton_analysis/pipeline.py -> webui.player_results` 边界依赖，不在本任务差异中。
- 评测仓全量 80 项、两套前端 lint/build、评测边界检查均通过；浏览器验证登录页可访问且未登录访问后台会跳转登录页。
