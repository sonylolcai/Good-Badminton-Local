# GPU 连续片段测试

这个测试把一段本地比赛视频切成独立可解码的短片，并作为**同一场比赛**依次上传。
它不是浏览器播放测试；目标是验证 GPU 的跨片段人物 ID、羽毛球时序、回合状态和动作事件。

## 归属与隔离

业务侧测试清单 `business_session_manifest.json` 保存以下映射：

```text
venue_id -> court_id -> match_id -> business_session_id -> gpu_analysis_session_id
```

`venue_id`、`court_id`、`match_id` 不会发送到 GPU。GPU 只收到 `camera_id`、
标定版本、四角、匿名 `client_reference`，并创建一个 `analysis_session_id`。这样既可
追溯“哪个场地的哪场比赛”，也不会把业务身份和赛果规则带入视觉服务。

## 顺序合同

同一 `gpu_analysis_session_id` 内：

1. `segment_index` 从 0 开始且必须连续递增；
2. 每段携带源视频时间轴的 `source_start_time_sec` 和真实时长；
3. GPU 先持久化字节、SHA-256、索引和幂等键，再按缺失前驱阻塞/顺序处理；
4. `expected_last_segment_index` 封口后仍有缺段，即不能被表述为完整连续比赛。

分片只是传输单位，人物轨迹、羽毛球时序和回合状态应由同一 GPU 会话持续保存。

## 本次视频的执行方式

先打开 GPU 到本机的隧道，并在本机设置 GPU 的 API 地址和密钥。不要把密钥写入命令
历史或测试清单。

```powershell
$env:GPU_ANALYSIS_BASE_URL = 'http://127.0.0.1:8080'
$env:GPU_ANALYSIS_API_KEY = '<仅在当前 PowerShell 会话设置>'
$env:GOOD_BADMINTON_FFMPEG = 'S:\Software Tool\bilibili-download\DownKyi-1.6.1\ffmpeg.exe'

& .\.venv\Scripts\python.exe -m business_gateway.streaming.continuity_replay_cli `
  --video 'S:\Software Tool\bilibili-download\DownKyi-1.6.1\Media\edited\【4K 50帧】神龙！2024校运会 安塞龙vs森-00.00.09.264-00.02.43.237-seg1.mp4' `
  --work-dir .\outputs\continuity-test-001 `
  --venue-id venue_test_001 `
  --court-id court_test_001 `
  --match-id match_test_001 `
  --camera-id camera_test_001 `
  --calibration-id calibration_test_001 `
  --court-corners '[[100,80],[1820,80],[1900,1020],[40,1020]]' `
  --segment-seconds 2 `
  --encoding-mode h264 `
  --tracker-backend court_association `
  --shuttle-detector yolo
```

四角必须先根据该比赛画面替换；文中的四角只是命令格式示例，不能当作有效标定。
完成后检查测试目录中的 `business_session_manifest.json`、`delivery-ledger.json`、
`end_to_end_trace.json` 和 GPU 会话事件。

## 验收

- `business_session_manifest.json` 同时拥有业务侧的场馆/场地/比赛映射和 GPU 会话 ID；
- `delivery-ledger.json` 中每一个已确认分片的索引连续，没有重复或冲突哈希；
- GPU 状态的 `missing_segment_indexes`、`failed_segment_indexes` 均为空；
- 任意跨片段的人物 ID 变化、羽毛球中断、回合/动作异常必须逐条抽查，不能被“已上传”掩盖；
- GPU 重启或故意漏段时，必须出现连续性 epoch 或明确的 partial/interrupted 状态，不能伪造连续运动统计。
