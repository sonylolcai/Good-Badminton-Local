# 视频目录、身份台账与访问边界

2026-10-23。本规范适用于新版业务端创建的 GPU 推送会话。尚未部署到远程业务/GPU；旧测试 Excel 仍保留原路径，不自动搬迁、不冒充某一场比赛。

## 对象目录

存储桶与对象前缀是两个概念。沿用已配置的存储桶，测试对象统一放在其 `testVideo/` 前缀，正式环境使用 `productionVideo/`：

```text
testVideo/
  haoyu/                           # 球馆固定 storage_slug
    <camera_uuid>/                 # 业务端摄像头 UUID，不用 RTSP 地址、IP 或显示名称
      2026-10-23/                  # 首次推送时按球馆 timezone 冻结的日期
        000001_<media_uuid>/        # 此摄像头当天第 1 场；媒体 UUID 保证目录唯一
          segments/00000000.mp4
          segments/00000001.mp4
          master.mp4
          integrity.json
          manifest.json
```

“第几场”指此摄像头当天第几次新建 GPU 推送会话，不把算法识别的一局/回合当作录像边界。预览不会分配场次；首次真正向 GPU 创建会话时分配。跨零点、重试和服务重启不更改已分配日期、序号、球馆目录和媒体 ID。断开的新采集会话得到新媒体 ID；不会凭相同名称拼接不同摄像头或不同场次。

已知试点球馆 UUID `77777777-7777-7777-7777-777777777777`、code `haoyushijie-venue-01` 映射至现有 `haoyu` 文件夹。其他球馆默认使用全局唯一的固定目录标识，注册后目录不会随球馆显示名称改变。目录不含手机号、姓名、摄像头账号或密钥。

## 身份与状态

迁移 `business_gateway/migrations/0033_media_archive_registry.sql` 增加：

- `venues.storage_slug`：全局唯一球馆目录标识。
- `media_daily_counters`：`environment + camera_id + capture_date` 的数据库原子计数器。
- `media_archives`：媒体 UUID、租户/球馆/场地/摄像头/网关/标定/业务采集会话 ID、可选的比赛和录制尝试 ID、GPU 会话 ID、日期序号、固定目录、输入版本、清单摘要、状态和过期时间。
- `media_objects`：每个分片、完整录像、完整性报告与清单都有独立对象 UUID，保留角色、GPU 分片序号、key、长度及 SHA-256。
- `media_download_audit`：签发下载链接的用户、媒体、对象与时间，记录签发，不冒充浏览器已下载完成。

业务分片索引 = GPU 分片索引 + `source_segment_offset`，该偏移也存入台账并在查询中返回，能追溯先预览再推送的来源。业务端原有 `edge_ingest_segments` 保留接收及修复代次身份。

分配时锁定业务采集会话，数据库检查摄像头、场地、网关、球馆和标定的归属；同一业务会话只有一个媒体台账。并发计数采用 UPSERT，失败事务不会留下半条台账。比赛/录制尝试已绑定时冻结其 ID，纯运营推送允许两项为空；不会编造比赛身份。

`media_identity` 是新增的可选服务间合同，GPU 验证 UUID、日期、目录字符及摄像头/标定/业务引用的一致性，并持久化到会话及不可变输入版本。匿名视觉推理不使用这些 ID 解释人物身份、赛果或评分。创建会话幂等判断包含媒体身份，改目录或序号后复用幂等键会被拒绝。

媒体归档状态独立于分析结果。业务 collector 即使已存好分析结果或推理失败，也会继续恢复媒体台账。GPU 仍需实际完成输入封存：仅有推理失败、但场馆未封存的录像不能自动宣称完整。

GPU 验证原片并拼接后，先上传分片、报告及 `master.mp4`，最后提交 `manifest.json`。业务端检查返回的身份、GPU 会话、固定目录、文件清单和清单摘要，再在事务中登记每个对象。网络错误只记录异常类型并重试，不能把失败当作 ready。DogeCloud 模式拒绝没有业务媒体身份的旧会话，旧 Aliyun/显式本地测试目录仍保持兼容；不悄悄迁移活动中的旧 GPU 会话。

## 访问与凭据

DogeCloud 适配器读取本项目 `.secrets/dogecloud.env`，或 `DOGECLOUD_ENV_FILE`。配置只在服务端读入，环境变量优先，不进入 Git、前端或媒体清单。S3 临时凭证按指定桶获取并短暂缓存，端点必须是无账号、查询串的 HTTPS 地址。SDK 包括 boto3、requests。

当前 GPU 和业务端分别持有各自本地维护的服务端凭据。它们使用桶范围的 `OSS_FULL` 临时权限；应用内前缀限制不等于云厂商 IAM 隔离。正式上线应使用专用私有媒体桶和独立受限服务账号；需要进一步缩小 GPU 权限时应增加业务端临时授权代理，不应把主密钥发给场馆或 Web。

适配器检查桶 ACL，上传指定 `private`，并检查对象 ACL、HEAD 长度和摘要元数据。重试只复用摘要和长度一致的已有对象，不覆盖不同字节；当前 GPU 媒体 worker 仍要求单进程、单持久目录，不声明实现跨 GPU 实例的分布式上传锁。测试与生产前缀严格隔离；显式使用错误环境的目录会失败，不能回退到另一个桶或本地目录。

运营接口：

```text
GET  /api/v1/cases/{edge_session_id}/media
POST /api/v1/cases/{edge_session_id}/media/{media_object_uuid}/download
```

每次先通过既有登录和球馆权限检查；POST 继续要求 CSRF。调用方只提交业务对象 UUID，不能提交任意 key/URL。仅已提交、未过期且属于此采集会话的对象可签发 HTTPS 下载链接，默认 60 秒，临近录像过期时进一步缩短。响应禁止缓存、下载签发留审计。权限撤销会阻止新签发；已签发链接在短期 TTL 内仍有效，不声明即时撤销已经外发的签名 URL。

本轮直接使用私有 S3 签名下载，不依赖 CDN 地址。CDN 是另一个访问入口：如果以后配置自定义/测试 CDN 域名，必须单独开启 URL 鉴权并验证未签名访问被拒绝；桶 ACL 私有不能证明 CDN 也是私有。没有启用 CDN 鉴权前，不把比赛视频发布为永久公开链接。本轮没有修改多吉云的 CDN/生命周期配置。

## 留存及发布

业务过期时间为首次推送分配时起最多 7 天，查询标记过期、下载接口拒绝新签发；这不是云端自动删除。还需在多吉云对两个前缀分别配置 7 天生命周期，删除原分片、成片、清单、失败上传残留及未完成的 multipart；云厂商按其执行时刻回收，不能承诺到秒。GPU 本地原片/成片的安全释放规则见下文“半小时场次与小磁盘保护”；`persistent_media` 仍保护会话和未归档证据。原分片加成片通常约占两份视频容量，应按两份核算留存费用。

发布顺序：

1. 先更新 GPU 代码/依赖以接受可选媒体身份，暂不启用归档；旧业务合同继续工作。
2. 更新业务代码，运行现有 `deploy/business-server/apply_migrations.py` 应用 0033，再启动业务进程。新会话才会带业务媒体身份；已运行的旧会话不重命名。
3. 在两端各自私密配置中选择 `GOOD_BADMINTON_MEDIA_STORAGE_PROVIDER=dogecloud`，维护对应 DogeCloud 三项凭据/桶名，两端显式配置 `DOGECLOUD_MEDIA_ENVIRONMENT=test` 用于当前联调，正式录像改为 `production`；分别生成 testVideo、productionVideo。未配置此项时按 `GOOD_BADMINTON_APP_ENV` 推导，不能通过请求覆盖。运营服务本身的生产鉴权配置不会因媒体测试环境而降级。
4. GPU 设置 `GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED=1`，FFmpeg/ffprobe 可用，并使用单 Uvicorn worker 与持久数据目录。业务自动结果 collector 必须运行。
5. 一路新摄像头测试：预览→推送→停止并封存→分析及媒体分别就绪→查询对象 UUID→签名下载及校验。

只填写配置文件不会重启当前服务，也不会自动修改云端权限或删除旧文件。远端部署、现场摄像头长视频和 CDN 鉴权还需实机验收。本次没有修改 Windows 场馆协议，它仍只上传分片到 GPU，收不到对象存储密钥。

## 验证

```sh
.venv/bin/python -m unittest tests.test_media_registry tests.test_edge_results tests.test_edge_ingest tests.test_operator_route_guards -v
.venv/bin/python tools/verify_media_registry.py
```

第二条只连接本地 PostgreSQL，创建并删除隔离测试库，验证迁移重复执行、12 个并发场次、24 次同场重试、重启身份不变、推理失败后的对象登记、幂等、下载审计与过期拦截，不写入真实业务库。GPU 对应测试为 `tests.test_media_identity_archive`、`tests.test_dogecloud_storage` 及既有媒体/GPU 合同回归。

## 半小时场次与小磁盘保护（2026-10-23）

正常使用为每约 30 分钟由业务端结束/封存一场，GPU 随即验证、拼接并归档；不等待当天累计四小时。20 Mbps 下单场原片约 4.5 GB，成片再占约 4.5 GB。60 GB 如果是整个磁盘，能否运行取决于安装后的实际余量、并发路数和归档速度；不再按一个四小时大文件估算，但也不承诺带宽未知时可以无限连续采集。

GPU 新增应用级安全阈值，正常场次留有余量，配置应写在 GPU 项目私密运行环境（例如 `.secrets/gpu-api.env`）：

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `GOOD_BADMINTON_STREAM_MAX_SOURCE_SECONDS` | 2700 | 单场最大源视频跨度 45 分钟；长时间预览产生的绝对时间偏移不算录制时长 |
| `GOOD_BADMINTON_STREAM_MAX_SESSION_BYTES` | 8589934592 | 单场原片最多 8 GiB |
| `GOOD_BADMINTON_STREAM_MAX_SEGMENTS` | 4096 | 限制异常产生海量小分片 |
| `GOOD_BADMINTON_STREAM_MAX_REQUEST_BYTES` | 35651584 | 单次完整上传请求最大 34 MiB（含 multipart 元数据） |
| `GOOD_BADMINTON_STREAM_MAX_UPLOAD_REQUESTS` | 4 | 单进程最多四个并发上传，约束解析器临时文件占用 |
| `GOOD_BADMINTON_STREAM_MAX_CACHE_BYTES` | 25769803776 | API 数据目录总预算 24 GiB，检查时还计入待拼接空间预留 |
| `GOOD_BADMINTON_STREAM_MIN_FREE_BYTES` | 10737418240 | 磁盘余量阈值 10 GiB；直接上传授权仍要求显式配置此项 |
| `GOOD_BADMINTON_MEDIA_LOCAL_CLEANUP_ENABLED` | 1 | 启用已验证视频副本清理；设 0 可关闭 |
| `GOOD_BADMINTON_MEDIA_LOCAL_CACHE_GRACE_SECONDS` | 300 | 云端 ready 后至少保留五分钟，并且必须等 GPU 处理结束 |

所有新增分片写盘都执行限制，包括服务密钥转发、场馆 token 直推、修复和缺片重传。通过入口临时空间检查后，幂等重试已经存在的相同文件无需再持久写盘，仍返回原回执；入口不足时仍可通过既有受保护的回执查询确认已接收的数据。创建新会话也检查容量；已存在会话的创建重试不会因磁盘不足而丢失原 ID。配置为 0、负数或非整数不能绕过限制。

预留按照所有未归档会话的原片计算待拼接容量，包含 10% 封装余量和每场 1 MiB 固定余量，避免只够收分片却不够拼接的死锁。FFmpeg 成片还设置输出体积上限，截断或不完整的输出会被完整性核验拒绝，不发布 ready。

单场超限返回 HTTP 413（不自动截断成假完整视频，也不偷偷创建下一场）；空间/缓存不足返回 HTTP 507，可在缓存释放后重试。已经接收的数据保留，仍可按已接收的边界完成封存。业务转发保留这两类容量错误，不能统一伪装为网络 502。健康接口的 `media_limits` 返回当前阈值和磁盘余量，供现场核对；Web 的阈值告警展示仍需前端接入。

归档线程在后台持续尝试清理，但必须同时满足：已封存、GPU 已进入终态、没有正在执行该会话的分片、云端归档 ready、输入版本及本地清单摘要匹配、重新 HEAD 核对云端提交清单的摘要和长度、未超过保守的云端留存期限、已过缓存宽限时间。归档失败或云端确认失败保留本地输入，靠拒收/重试形成背压，不删除唯一副本来腾空间。

清理只删除该场已归档的原始 `.bin` 分片和对应 `master.mp4`。会话、事件、checkpoint、照片、清单、完整性报告和 ID 记录继续保留，业务端可以在清理后回收分析结果和对象台账。清理前先持久化云端回执，再删除重复字节；中断可恢复，不能跟随符号链接删除其他位置。已释放分片的管理端回执会附 `durable_location=object_storage`，以校验通过且未过期的云端提交记录确认可靠接收，不伪装成本地文件仍在。

`persistent_media` 仍禁止 TTL 整体删除会话；本次新增的是已归档重复视频字节的单独清理，未放宽未归档证据保护。云端七天生命周期仍需单独配置。事件/结果的长期归档与最终清理、系统日志轮转和其他进程磁盘使用不由这套视频缓存规则替代；多进程写同一数据目录也不在当前部署保证内。

上传入口在 multipart 解析之前检查 Content-Length，实际流式字节也计数，不能通过省略或伪造长度绕过限制。并发和临时目录空间分别检查，避免文件先落到系统临时目录才被分片校验拒绝。并发已满返回 429 并允许重试。部署使用自有 Nginx 或平台入口时，也要配置匹配的 34 MiB 请求上限；自有 Nginx 可设 `client_max_body_size 34m; proxy_request_buffering off;`，防止前置代理先缓存整份异常请求。应用内限制不替代代理配置。

对应回归：GPU `tests.test_media_limits`、`tests.test_media_release`、`tests.test_media_upload_guard`，业务 `tests.test_edge_ingest` 的容量错误转发。使用真实短 H.264 验证归档、删除、云端回执和恢复；不代表新 4090 上已完成半小时实拍或长时间并发验收。

## 本次代码审查后的部署前提（2026-10-23）

- `deploy/install_gpu_api.sh` 仍只安装基础依赖，已有环境的 ZIP 刷新也不会自动补齐媒体 SDK。启用归档前，使用服务实际使用的 Python 单独安装 `requirements-media.txt`，并检查 `ffmpeg`、`ffprobe` 可用；不要通过直接安装含 CPU torch 固定版本的完整 requirements 来替换 CUDA 环境。
- GPU 的 `deploy/refresh_gpu_api_from_zip.sh` 会替换应用目录，目前只自动保留既有 API 环境、数据和权重。部署前将对象存储私密配置放到应用目录外的持久路径，在持久 API 环境中配置 `DOGECLOUD_ENV_FILE` 指向它；不能让应用目录内的 `.secrets/` 成为服务器唯一配置副本。代码包不能携带密钥。
- 业务 Docker Compose 不挂载开发机的 `.secrets/`。需通过受保护的服务器 `business.env` 向 edge-gateway、operator-api 注入 DogeCloud 配置，或给两个容器挂载独立只读配置文件并设置容器内的 `DOGECLOUD_ENV_FILE`。宿主机文件存在不代表容器可以读取。
- 迁移 GPU 地址前，排空旧服务器的分析、归档和业务台账同步；当前配置未将 GPU 地址按会话固化。Web 的 OSS 归档就绪、下载和阈值告警界面尚需接入；后端接口测试通过不代表这些界面已经完成。
- TTL 清理已补充跨重启保护：即使重启后关闭归档开关，只要已有媒体归档目录且未 ready，也不能整体删除该场证据。该保护不等同于云端七天生命周期配置。
