# 任务分解

| Task ID | Status | Depends on | Owned files | 验证 |
|---|---|---|---|---|
| T1 | pending | — | `agent.py`、`edge_api.py`、存储配置、边缘测试 | 同编码双轨输出、母版断点重传与分片回归。 |
| T2 | pending | T1 | `match_delivery.py`、网关/API 测试 | 母版/分片 PTS 覆盖、fallback 与完整性状态。 |
| T3 | pending | T2 | 后处理/GPU 适配、流测试 | 严格 repair 门控、窗口限定、无正常全片重跑。 |
| T4 | pending | T2,T3 | API、前端、前端测试 | 既有采集控制和交付状态可见。 |
