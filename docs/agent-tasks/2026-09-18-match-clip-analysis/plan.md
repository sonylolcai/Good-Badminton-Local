# PLAN APPROVAL — 单局两秒采集与最终交付

## 用户可见结果

用户点击现有“开始采集”，系统开始一局；点击“停止视频”，系统封存该局完整原始 MP4。GPU 在采集期间持续分析两秒片段，但最终报告、标注时间线和用户成片融合的是已经积累的结构化结果，不会重跑正常完整视频。完整录像还能生成供 VLM/LLM 异步复盘的可审计证据包。

## 目标数据流

```text
开始采集 -> edge_ingest_session（即对局 ID） -> 连续两秒 MP4 -> edge-gateway 持久化
                                                       |-> 已有 GPU 顺序会话/检查点 -> PTS 证据
停止视频 -> 完整性检查 -> recording.mp4（流复制，无二次编码）
                            + 已有 PTS 证据 -> 对局融合器 -> 标注成片 / 统计 / VLM 证据包
```

完整录像不需要在停止后再传一遍：每两秒文件已经上传并保存在业务服务器；停止动作只是验证序号、封存为一个文件，并完成 GPU 会话。若某个外部 VLM 必须读取视频，由业务服务器以受控内部地址提供完整录像或 PTS 选择的证据窗口。

## 文件范围

| 路径 | 动作 | 用途 |
|---|---|---|
| `business_gateway/edge_api.py` | 修改 | 在现有 session 封存时记录完整性/交付状态；暴露受控的最终结果和 VLM 证据包内部入口。 |
| `business_gateway/match_delivery.py` | 新增 | 对局 PTS 结果融合、异常边界筛选、原始/标注交付清单；不运行整局二次推理。 |
| `operator_api/services/operator_backoffice.py` | 修改 | 将现有 capture case 投影为单局交付状态。 |
| `operator_api/main.py` | 修改 | 查询该局完整性、最终结果、原始录像、成片和异步复盘任务的 API。 |
| `webui-next/src/app/venues/[id]/courts/CourtManagementClient.tsx` | 修改 | 保持既有开始/停止按钮，显示本局封存、融合和交付状态。 |
| `webui-next/src/lib/operator-api.ts` | 修改 | 最终交付 API 的类型化客户端。 |
| `badminton_analysis/streaming/engine.py` 或既有 GPU 适配层 | 修改 | 将已有 segment checkpoint/PTS 证据完整投影到本局最终融合器。 |
| `badminton_analysis/post_match.py` | 新增 | 只做证据汇总、质量门控和 VLM PTS 证据包，不重跑视频模型。 |
| `tests/test_edge_ingest.py`、`tests/test_operator_api.py` | 修改 | 单局开始/结束、完整性和交付授权。 |
| `tests/test_stream_sessions.py`、`tests/test_streaming_analysis_engine.py` | 修改 | 跨段状态、重传、恢复和局部补算选择。 |
| `tests/test_match_delivery.py` | 新增 | PTS 融合、缺段/低置信边界、无全局复跑的规则。 |
| `webui-next/src/app/venues/[id]/courts/*.test.tsx` 或现有前端测试位置 | 新增/修改 | 既有控制与交付状态呈现。 |

不计划修改 `deploy/venue-gateway/agent.py`，除非测试发现其现有 `source_start_time_sec` 不能作为跨片连续时间线；也不计划新增数据库迁移。先复用现有 `edge_ingest_sessions`、片段表、GPU 事件和持久化录像目录；若实施中证明交付列表无法可靠持久化，才在下一审批包提出最小迁移。

## 任务顺序与回滚

| 任务 | 依赖 | 交付 | 回滚点 |
|---|---|---|---|
| `T1` 结果合同 | 无 | session/PTS 到最终交付的只读数据合同和测试。 | 不启用新 API，不影响采集。 |
| `T2` 融合与质量门控 | T1 | 结构化证据融合、局部补算标记、VLM 证据包。 | 保留原始录像和现有 GPU 结果。 |
| `T3` 业务 API/UI | T1,T2 | 现有采集页面展示封存/交付状态。 | 隐藏新读接口，不改变开始/停止。 |

## 验证策略

- 合成 H.264 样本：验证连续段流复制封存、缺段拒绝完整状态和最终 PTS 时间线。
- 复用现有计数处理器：验证检查点跨片连续、重传不重复、结束后只做融合而非重跑。
- 端到端假 GPU：一次“开始采集 -> 两秒片段 -> 停止 -> 原始录像 -> 融合交付”。
- 真实场馆：20 分钟单局验证连续播放、边界标注、停止封存、GPU 状态和 VLM 证据来源；不会以本地测试替代。

## 授权边界

本计划不授权生产部署、远端数据迁移、真实场馆采集、外部 VLM/LLM 调用、提交或推送。计划获批后先设计测试并提交 TEST APPROVAL，再开始代码修改。
