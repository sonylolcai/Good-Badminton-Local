# PLAN APPROVAL — 单局双轨录像与局部修复

## 数据流

```text
开始采集 -> session / 对局 ID
  -> 同一次 H.264 编码
       -> 两秒分片 -> 业务服务器 -> GPU 顺序分析与 PTS 证据
       -> 场馆本地 master.mp4
停止视频 -> 上传 master.mp4 + 哈希/时长/PTS 清单
  -> 校验母版与分片覆盖
  -> 融合实时 GPU 证据
  -> [仅 repair ticket] 母版小窗口复算
  -> 母版 + 统一标注一次渲染 -> 用户成片 / VLM 证据包
```

完整母版不是停止后再交给 GPU 正常重跑，而是独立权威录像。两秒分片已经完成实时 GPU 推理；完整母版只在严格质量门控触发时提供少量窗口复算。

## 改动范围

| 路径 | 动作 | 用途 |
|---|---|---|
| `deploy/venue-gateway/agent.py` | 修改 | 一次编码同时输出两秒分片与本地 `master.mp4`；停止后以可重试上传发送母版和清单。 |
| `business_gateway/edge_api.py` | 修改 | 接收/验证母版、保存清单、比较覆盖、选择母版或 fallback，并创建受控 repair ticket。 |
| `business_gateway/match_delivery.py` | 新增 | PTS 证据融合、严格质量门控、局部窗口选择和最终交付清单。 |
| `operator_api/services/operator_backoffice.py`、`operator_api/main.py` | 修改 | 查询一局的母版、fallback、修复和交付状态；不新增采集控制。 |
| `webui-next/src/app/venues/[id]/courts/CourtManagementClient.tsx`、`webui-next/src/lib/operator-api.ts` | 修改 | 保持现有开始/停止按钮，展示母版上传、完整性、修复与交付状态。 |
| `badminton_analysis/post_match.py`、现有 GPU 适配层 | 新增/修改 | 实时证据融合、repair 窗口运行和 VLM PTS 证据包；禁止正常全片重跑。 |
| `deploy/business-server/docker-compose.stone.yml`、环境示例 | 修改 | 增加可持久化的母版存储目录和限制配置。 |
| `tests/test_edge_ingest.py`、`tests/test_stream_sessions.py`、`tests/test_streaming_analysis_engine.py`、`tests/test_match_delivery.py`、`tests/test_operator_api.py` | 新增/修改 | 双轨采集、母版校验、严格门控、局部修复、授权与回归。 |

## 任务与回滚

| 任务 | 依赖 | 交付 | 回滚 |
|---|---|---|---|
| `T1` 双轨采集合同 | — | agent 母版输出/上传和 edge 母版接收；分片原链路不变。 | 关闭母版开关，继续既有分片录像。 |
| `T2` 校验与选择 | T1 | 母版/分片覆盖清单、fallback 标记、用户原始录像选择。 | 所有交付仍回退既有 `recording.mp4`。 |
| `T3` 证据融合与 repair | T2 | 严格门控、窗口复算、审计记录和 VLM 证据包。 | 不启用 repair，保留实时结果。 |
| `T4` API/UI | T2,T3 | 既有控制旁的只读交付状态。 | 隐藏新状态，不影响采集。 |

## 验证

- 合成 H.264 夹具：单次编码的母版/片段时长和抽样帧一致；缺段或母版不一致不可称完整。
- 容错：母版上传中断后重试；母版失败时 fallback 可用且来源明确。
- GPU：顺序 checkpoint、重传、恢复仍绿色；正常结束无整局母版重跑。
- 质量门控：正常低频遮挡不触发；结构失败和连续严重低证据触发有限 PTS 窗口。
- 真实场馆：20 分钟单局的母版连续性、停机封装、上传、对齐、磁盘占用和局部修复证据。

## 授权边界

本计划不授权部署、真实场馆采集、外部 VLM/LLM 调用、提交或推送。计划获批后先出 TEST APPROVAL，再写生产代码。
