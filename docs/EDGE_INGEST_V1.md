# 场馆终端接入协议 v1

本协议用于首轮“终端 → 业务服务器 → GPU”直连试点。它不含 OSS；业务服务器仅把未取得 GPU 回执的分片暂存在受保护的临时目录，取得回执后立刻删除。PostgreSQL 只保存运行状态、哈希、幂等键和 GPU 回执，不保存视频字节。

终端必须先由统一运营后台创建并绑定：`venue_id`、`court_id`、`device_id`、`camera_id` 和 `credential_version`。终端请求里可以带 `device_id`、`camera_id`，但**不能自行声明或修改场馆/场地归属**；业务服务器总是按数据库绑定关系解析归属。

## 认证

每台终端使用一次性展示的派生密钥签名。业务服务器只保存根密钥 `GOOD_BADMINTON_EDGE_MASTER_KEY`（至少 32 字节），密钥为：

`base64url(HMAC-SHA256(master, "good-badminton-edge.v1:{device_id}:{credential_version}"))`

所有请求都有下列 Headers，`timestamp` 与 JSON 内同名字段必须相同：

| Header | 值 |
| --- | --- |
| `X-Edge-Timestamp` | RFC3339 UTC 时间，例如 `2026-08-31T10:00:00Z` |
| `X-Edge-Nonce` | 16–128 位 URL-safe 随机字符串；同一设备十分钟内不能重复 |
| `X-Edge-Payload-SHA256` | 小写 SHA-256 摘要 |
| `X-Edge-Signature` | 小写 HMAC-SHA256 签名 |

签名输入是五段 UTF-8 文本，以换行分隔：`METHOD\nPATH\nTIMESTAMP\nNONCE\nPAYLOAD_SHA256`。服务端仅接受前后 120 秒内的请求，拒绝签名、时间或 nonce 失败的请求。

JSON 请求的 payload 摘要为 key 排序、无空格的 UTF-8 JSON SHA-256。分片上传因为 multipart boundary 不稳定，摘要为 `canonical_metadata + "\\n" + 原始视频字节` 的 SHA-256。

## 心跳

`POST /api/v1/edge/devices/{device_id}/heartbeats`

```json
{
  "schema_version": "edge-ingest.v1",
  "device_id": "设备 UUID",
  "camera_id": "摄像头 UUID",
  "timestamp": "2026-08-31T10:00:00Z",
  "nonce": "每次请求新的随机值",
  "agent_version": "camera-agent/0.1.0",
  "disk_free_bytes": 123456789,
  "capture_state": "idle",
  "active_session_id": null,
  "last_segment_index": null
}
```

服务端更新设备、摄像头最近心跳；后台按场馆分组展示在线状态。`capture_state` 只能是 `idle`、`capturing` 或 `error`。

心跳响应还会包含业务服务器下发的控制指令：

```json
{
  "status": "accepted",
  "capture_control": { "mode": "idle", "revision": 3, "updated_at": "2026-09-01T10:00:00Z" }
}
```

`mode` 只能为 `idle`、`preview` 或 `record`。终端只在 `preview` / `record` 时创建会话并启动 FFmpeg；`idle` 时只发送心跳，不能上传视频。该状态只能由运营后台修改，终端也不能绕过 `idle` 自行创建会话。这样球馆 Mac 只需一次安装和常驻运行，现场人员不参与每天的启停。

## 开始解析

`POST /api/v1/edge/devices/{device_id}/sessions`

请求带 `schema_version`、`device_id`、`camera_id`、`timestamp`、`nonce` 和匿名 `configuration`。`configuration` 必须完整符合现有 `stream-session.v1` 的匿名配置（至少 `analysis_sample_hz`、`pose_imgsz`、`shuttle_detector`、`generate_annotated_video`）；服务端会在创建会话时验证，避免首个分片才失败。禁止 `user_id`、`match_id`、比分、队伍、昵称等业务身份字段。服务端读取该摄像头最新的 `validated` 标定，并返回业务侧 `edge_ingest_session_id`。同一摄像头不能同时有两个活跃会话。业务服务器未下发 `preview` 或 `record` 时，此接口返回 `409`。

## 上传分片

`POST /api/v1/edge/sessions/{edge_ingest_session_id}/segments/{segment_index}`，`multipart/form-data`：

- `segment`：独立可解码的 MP4 或 ISO segment，时长大于 0 且不超过 10 秒；
- `metadata`：JSON 字符串，包含下列 envelope：

```json
{
  "schema_version": "edge-ingest.v1",
  "device_id": "设备 UUID",
  "camera_id": "摄像头 UUID",
  "timestamp": "2026-08-31T10:00:02Z",
  "nonce": "新的随机值",
  "segment": {
    "schema_version": "stream-session.v1",
    "segment_index": 0,
    "source_start_time_sec": 0,
    "duration_sec": 2,
    "sha256": "视频字节 SHA-256",
    "idempotency_key": "至少16位的稳定键",
    "content_type": "video/mp4",
    "content_length_bytes": 123456,
    "court_corners": [[0,0],[1,0],[1,1],[0,1]]
  }
}
```

`court_corners` 由业务服务器对照已验证标定校验，不接受终端覆盖。相同 `segment_index + sha256 + idempotency_key` 的重试返回同一回执；冲突哈希返回 `409`。若业务服务器未获得 GPU 回执，终端不得把分片当作成功，应以同一幂等键重试。

## 结束解析

终端停止采集、且最后一个分片已得到 GPU 回执后，调用 `POST /api/v1/edge/sessions/{edge_ingest_session_id}/complete`。签名方式不变，JSON 包含 `schema_version`、`device_id`、`camera_id`、`timestamp`、`nonce`、`expected_last_segment_index` 与布尔值 `allow_partial`。业务服务器将其转为 GPU 的完成请求，并只在 GPU 返回状态后更新为 `succeeded`、`partial`、`failed` 或 `cancelled`；网络异常时终端必须用**同一完成请求**重试。

## 状态语义

后台对每片场地显示：设备/摄像头最近心跳、采集状态、活跃业务会话、已收/已转发分片数、GPU 会话 ID、GPU 状态和最近错误。会话状态从 `requested → receiving → relaying → processing → succeeded|partial|failed|cancelled` 流转；没有 GPU 回执时不能显示为“已解析完成”。

GPU 启动/关闭仍通过受控 `GpuControlAdapter` 实现；运营后台只能配置地址、检查健康、请求启停和取消，不能接收任意 shell 命令或云令牌。
