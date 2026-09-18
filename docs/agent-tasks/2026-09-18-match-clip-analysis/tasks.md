# 任务分解

| Task ID | Status | Depends on | Owned files | Expected change | Verification | Commit |
|---|---|---|---|---|---|---|
| T1 | pending | — | `edge_api.py`、`test_edge_ingest.py`、`test_match_delivery.py` | 当前单局 session 到 PTS 交付的完整性合同 | 连续/缺段/重传测试 | — |
| T2 | pending | T1 | `match_delivery.py`、`post_match.py`、流测试 | 结构化证据融合、局部补算选择、VLM 证据包 | 不重跑整局与不捏造数据测试 | — |
| T3 | pending | T1,T2 | `main.py`、`operator_backoffice.py`、前端/API 测试 | 既有采集控制旁展示本局交付状态 | API/UI 合同与人工验收清单 | — |
