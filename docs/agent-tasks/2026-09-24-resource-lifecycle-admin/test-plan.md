# 测试计划

## 门禁和基线

- PLAN APPROVAL：2026-09-24，用户原文 `批准`。
- 本阶段只允许修改测试和任务文档；生产代码保持不变。
- 业务后端基线：28 个相关测试通过。
- 业务前端基线：lint 通过；build 仅因执行环境无法访问 Google Fonts 失败。
- 评测后端基线：16 个相关测试通过。
- 评测前端基线：lint 与边界检查通过。

## 自动化用例

| ID | 验收条件 | 用例 | 期望 |
| --- | --- | --- | --- |
| RT-01 | AC-05, AC-06 | 未登录访问现有业务数据 | 401，不泄露球馆数据 |
| RT-02 | AC-05 | 错误账号密码登录 | 401，路由存在且不开放注册 |
| RT-03 | AC-06, AC-08 | 未登录访问资源清单 | 401 |
| RT-04 | AC-04, AC-06 | 未登录访问保留策略 | 401 |
| RT-05 | AC-05, AC-06, AC-09 | 扫描业务迁移 SQL | 认证、RBAC、资源位置和保留策略表均存在 |
| RT-06 | AC-01 | 评测 Run 仅删除资源 | Run 契约保留，返回 `mode=resources` |
| RT-07 | AC-03 | 本地或远程部分删除失败 | 返回 409、`partial`、`retryable=true` |
| RT-08 | AC-04, AC-09 | 评测定时清理入口无内部密钥 | 401 |
| RT-09 | AC-09, AC-11 | 扫描评测迁移 SQL | 独立资源、位置、Run 关联表均存在 |
| RT-10 | AC-01, AC-04 | GPU 视频资源删除 | MP4 删除，job.json 与分析 JSON 保留 |
| RT-11 | AC-01, AC-04 | 流式会话到期清理 | 只删分片视频，manifest/JSON 和记录保留；默认 dry-run |

## 后续切片测试

- T01：迁移可重复执行、约束和资源状态机单测。
- T02：运行中拒绝删除、终态视频-only、full-delete、幂等重试。
- T03：共享输入引用计数、评测两种删除、本地/远程部分失败、旧记录兼容。
- T04：首次管理员 bootstrap、强制改密、会话过期、密码散列、平台/球馆 RBAC。
- T05：业务上传成功时间、球馆归属、手动解析、球员资产关联、越权拒绝。
- T06：开关默认关闭、7 天边界、每天一次、并发锁、只删视频、失败审计。
- T07：登录、用户、资源、设置页面的 lint/build 和浏览器角色验收。
- T08：两仓全回归、迁移演练和验收矩阵。

## RED 证据

2026-09-24 已得到 12 个有效 RED，均为“期望的新行为尚未存在”，没有导入、语法或测试夹具错误：

- 业务仓 8 项失败：现有球馆接口未鉴权（实际 200）；登录、资源清单、保留策略和 GPU 资源删除路由尚不存在（实际 404）；认证/RBAC/资源/保留策略表尚不存在；流式清理会连同 manifest 和记录一起删除。
- 评测仓 4 项失败：仅删除资源、部分失败报告、内部定时清理路由尚不存在（实际 404）；独立资源/位置/Run 关联表尚不存在。

执行命令：

```powershell
# Good-Badminton
.\.venv\Scripts\python.exe -m unittest tests.test_admin_resource_api tests.test_admin_resource_migration tests.test_gpu_api.GpuApiTests.test_video_resource_deletion_keeps_job_and_json_evidence tests.test_stream_sessions.StreamSessionManagerTests.test_retention_cleanup_removes_only_video_bytes_and_is_dry_run_by_default -v

# Good-Badminton-Evaluation-Platform/backend
.\.venv\Scripts\python.exe -m unittest tests.test_resource_lifecycle_api tests.test_resource_migration -v
```

结果：业务仓 `8/8 RED`，评测仓 `4/4 RED`。其中 GPU 幂等键和测试类名在首次执行发现为夹具错误，修正后已重跑为期望的行为失败，不计入 RED 证据。

## 最终 GREEN 证据

- 业务新增相关测试：42 项通过；管理员状态补充回归：26 项通过。
- 评测后端全量：80 项通过。
- 业务前端：lint、Next.js 生产构建通过。
- 评测前端：lint、网络边界检查、Next.js 生产构建通过。
- Playwright：登录页表单可访问；未登录访问 `/` 自动跳转 `/login`。本地未启动 operator API，因此控制台仅出现预期的 `localhost:8000` 连接拒绝。
- 业务后端全量：362 项中 361 项通过；唯一失败来自基线已有 `badminton_analysis/pipeline.py` 对 `webui.player_results` 的依赖，基线至本任务 HEAD 对该文件无差异。
