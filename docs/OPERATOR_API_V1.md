# 运营后台 API v1

运营后台使用 `operator_api.main:app`。本地开发地址为 `http://127.0.0.1:8000`，OpenAPI 文档为 `/docs`，OpenAPI JSON 为 `/openapi.json`。

这套 API 面向运营后台和受控内部工具，**不是摄像头终端协议**。终端心跳、开始解析、分片与完成会话必须使用 [EDGE_INGEST_V1.md](EDGE_INGEST_V1.md) 中签名的 `/api/v1/edge/*` 路由。

## 注册边界

注册必须形成完整链路：`tenant → venue → one or more courts`。

- 传入 `tenant_id`：注册到一个已有且 `active` 的租户；
- 传入 `tenant_name`：原子创建新租户；
- 两者必须且只能传一个；
- `courts` 至少一条；同一场馆内的场地编码不能重复；
- 任一步骤失败将整体回滚，不留下半注册场馆。

## 核心接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/tenants` | 获取已有租户选择项 |
| `GET` | `/api/v1/venues` | 获取场馆列表（对象结构，不使用数组下标） |
| `POST` | `/api/v1/venue-registrations` | 原子注册租户/场馆/首批场地 |
| `GET` | `/api/v1/venues/{venue_id}/courts` | 获取单馆场地 |
| `POST` | `/api/v1/venues/{venue_id}/courts` | 新增一个场地 |
| `PATCH` | `/api/v1/venues/{venue_id}/courts/{court_id}/status` | 设置为 `active`、`maintenance` 或 `inactive` |
| `GET` | `/api/v1/venues/{venue_id}/operations` | 获取每个场地的摄像头连接、短时预览、当前 case 与 GPU 状态 |
| `POST` | `/api/v1/venues/{venue_id}/courts/{court_id}/capture` | 下发 `idle`、`preview` 或 `record`；Mac 在下次心跳执行 |
| `POST` | `/api/v1/venues/{venue_id}/courts/{court_id}/case/gpu-forwarding` | 为当前实时 case 开启或暂停 GPU 推送 |
| `GET` | `/api/v1/cases/{case_id}/gpu-events` | 获取指定 case 的 GPU 执行事件与安全回执 |
| `POST` | `/api/v1/venues/{venue_id}/courts/{court_id}/edge-bindings` | 创建终端/摄像头绑定，并**仅本次**返回终端密钥 |
| `GET` | `/api/v1/gpu/status` | GPU 服务健康状态 |
| `POST` | `/api/v1/gpu/config` | 保存 GPU 地址和可选的新 API Key |
| `POST` | `/api/v1/gpu/operate` | 请求 `start` 或 `stop`（需服务端受控适配器） |

## 注册请求示例

```json
{
  "tenant_name": "好雨时节体育",
  "venue_code": "haoyushijie-01",
  "venue_name": "好雨时节球馆",
  "timezone": "Asia/Shanghai",
  "address": "上海市徐汇区",
  "courts": [
    { "code": "court-01", "name": "一号场", "sort_order": 0, "status": "active" },
    { "code": "court-02", "name": "二号场", "sort_order": 1, "status": "maintenance" }
  ]
}
```

成功时返回 `201`，结构如下：

```json
{
  "registration": {
    "tenant": { "id": "uuid", "name": "好雨时节体育", "created": true },
    "venue": { "id": "uuid", "tenant_id": "uuid", "code": "haoyushijie-01", "name": "好雨时节球馆", "timezone": "Asia/Shanghai", "address": "上海市徐汇区", "status": "active" },
    "courts": [{ "id": "uuid", "code": "court-01", "name": "一号场", "sort_order": 0, "status": "active" }]
  },
  "message": "场馆与场地已注册。"
}
```

业务规则失败统一返回 `422`：

```json
{
  "error": {
    "code": "business_validation_failed",
    "message": "同一场馆内的场地编码不能重复。"
  }
}
```

## 实时场地控制

`GET /api/v1/venues/{venue_id}/operations` 的每一项都代表一个场地，而不是一张技术 ID 表。`camera.connected` 只有在终端和摄像头都是活动状态、且二者心跳都在最近 90 秒内时才为 `true`。

```json
{
  "court": { "id": "uuid", "name": "一号场", "code": "court-01", "status": "active" },
  "camera": { "connected": true, "camera_code": "cam-01", "calibration_status": "validated" },
  "case": {
    "id": "business-edge-case-uuid",
    "status": "receiving",
    "preview_available": true,
    "preview_url": "http://business-gateway/api/v1/edge/sessions/.../preview/latest.mp4",
    "gpu_forwarding_enabled": false,
    "gpu_analysis_session_id": null,
    "received_segment_count": 3,
    "forwarded_segment_count": 0
  }
}
```

先由业务后台控制视频采集；球馆 Mac 不需要人工操作：

```json
{ "mode": "preview" }
```

- `idle`：Mac 只保留签名心跳，不读取 RTSP、不上传视频；
- `preview`：网关在下一次心跳后开启短预览。若该摄像头尚未有已验证标定，业务服务器仍可接收并展示预览，但该会话永不推 GPU；运营人员据此保存四角后，停止预览并重新开始采集。
- `record`：要求摄像头已有已验证标定；开始采集后 GPU 仍须单独开启。

停止时发送 `{ "mode": "idle" }`。Mac 会关闭 FFmpeg、完成当前 case，然后回到心跳模式。不会建立从业务服务器到球馆网络的公网入站连接。

在已有实时 case 后，通过下列请求开启或暂停 GPU 推送：

```json
{ "enabled": true }
```

暂停时，业务服务器继续保留一个很短的可播放 MP4 预览窗口，但不会向 GPU 转发新片段；再次开启只处理开启后的片段，不补推暂停期间的视频。`case_id` 始终是业务服务器生成的 `edge_ingest_session_id`，并通过 GPU 返回的 `gpu_analysis_session_id` 关联执行事件。网页展示的是结构化 GPU 事件/回执，不能展示原始终端 shell、RTSP 地址或密钥。

部署前先执行 `business_gateway/migrations/0003_business_tenants.sql` 至 `0007_edge_capture_control.sql`。边缘网关与运营 API 必须配置同一份强度足够的 `GOOD_BADMINTON_EDGE_MASTER_KEY`；缺少它时不要启动摄像头接入服务。生产环境还应分别设置浏览器可访问的 `GOOD_BADMINTON_EDGE_GATEWAY_PUBLIC_URL=https://api.example.com` 与服务器内部的 `GOOD_BADMINTON_EDGE_GATEWAY_INTERNAL_URL=http://127.0.0.1:18080`。

## 安全与部署

当前 API 是本机开发模式，CORS 默认只允许 `localhost:3000` 和 `127.0.0.1:3000`。生产环境必须设置 `GOOD_BADMINTON_OPERATOR_API_ALLOWED_ORIGINS`，并在 API 前加入登录、角色校验与租户数据隔离后才可对公网暴露。

终端绑定接口会返回一次性 `device_secret`。它只应由有权限的运营人员复制到受控终端，不能记录到日志、截图、浏览器存储或工单中。GPU API Key 同样只允许写入，不能从 API 读回。
