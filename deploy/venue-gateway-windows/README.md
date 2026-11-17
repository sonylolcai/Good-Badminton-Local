# 球馆 Windows 中转主机安装包

本程序运行在球馆的 64 位 Windows 电脑上。它从球馆局域网读取摄像头 RTSP，将视频重新编码为独立可播放的 2 秒 MP4 分片，再通过带 HMAC-SHA256 签名的 HTTPS 请求上传到业务服务器。

摄像头地址、账号、密码和原始 RTSP 只保存在球馆 Windows 主机。程序不监听公网端口，也不直接连接 GPU。业务后台通过每 10 秒一次的签名心跳控制预览和采集：服务端返回 `idle` 时只发送心跳，返回 `preview` 或 `record` 后才读取和上传视频。

当前单球馆阶段，设备 ID、摄像头 ID 和密钥仅作为固定传输凭据使用，不参与球馆端的用户、比赛或场馆状态管理。球馆端运行状态被收敛为 `idle` 与 `pushing`：`preview` 和 `record` 对球馆端都表示开始推送，是否保存、播放或分送 GPU 由业务服务器决定。活动会话及最后成功分片序号会原子保存到 `spool\active-session.json`，进程重启后从下一分片继续。心跳控制面与视频数据面彼此隔离；服务器 409、FFmpeg 退出或分片上传失败时，视频侧进入 `error` 并退避重试，心跳线程继续运行，不再让整个网关随采集错误退出。

当业务服务器在心跳响应中声明 `capabilities.edge_error_reports=true` 后，网关会自动在后续心跳附加最近一条结构化错误报告（错误代码、脱敏摘要、发生时间、累计次数）。旧服务器不会收到新增字段，因此仍与当前 `edge-ingest.v1` 兼容。

## 一、运行前必须准备的环境

### 1. Windows 电脑

- Windows 10/11 64 位，或 Windows Server 2019 及以上（Windows on ARM 不在本版本验证范围内）。
- 至少 4 核 CPU、8 GB 内存；1080p 实时转码建议使用较新的 6 核以上 CPU。
- 系统盘至少保留 5 GB 可用空间。
- 关闭自动睡眠和休眠；显示器可以关闭。
- 使用有线网口，不要用 Wi-Fi 承载摄像头视频。
- 安装和注册开机任务时需要本机管理员权限。
- 首次安装虚拟环境时需要通过 HTTPS 访问 PyPI 下载 `requests` 及其依赖；安装完成后的日常运行不依赖 PyPI。

### 2. Python 3.11 x64

从 Python 官方网站安装 64 位 Python 3.11。安装时勾选 `Install launcher for all users`，并建议勾选 `Add python.exe to PATH`。验证：

```powershell
py -3.11 --version
```

安装脚本会在 `%ProgramData%\GoodBadminton\venue-gateway\.venv` 创建独立虚拟环境，不会把依赖装入系统 Python。

### 3. FFmpeg x64

安装包含 `libx264` 编码器的 64 位 FFmpeg，并把 FFmpeg 的 `bin` 目录加入系统 `PATH`。重新打开 PowerShell 后验证：

```powershell
ffmpeg -version
ffmpeg -hide_banner -encoders | Select-String libx264
ffprobe -version
```

三条命令均需成功。安装程序会把 `ffmpeg.exe` 的绝对路径写入本地配置，使开机任务不依赖交互用户的 PATH。仅安装 Python 或 OpenCV 并不能替代 FFmpeg。

### 4. 摄像头与局域网

- Windows、摄像头、路由器和 PoE 交换机必须在同一可互通局域网。
- 摄像头应使用固定地址或路由器 DHCP 保留地址。
- Windows 必须能访问摄像头 RTSP TCP 554 端口。
- 大华摄像头主码流常见示例：

```text
rtsp://USER:PASSWORD@192.168.3.13:554/cam/realmonitor?channel=1&subtype=0
```

账号或密码中的 `@`、`#`、`%`、`:` 等字符必须进行 URL 编码。不要把真实 RTSP 地址发送到聊天、Git 或截图中。

### 5. 业务服务器与互联网

- Windows 必须能出站访问 `https://for-one-dream.cloud/badminton-edge` 的 TCP 443。
- 不需要开放任何入站公网端口。
- 防火墙或杀毒软件不能阻止 Python/FFmpeg 的局域网 RTSP 和公网 HTTPS 出站连接。
- 业务服务器必须已创建场馆、场地、摄像头和边缘设备，并导出：`EDGE_DEVICE_ID`、`EDGE_CAMERA_ID`、`EDGE_CREDENTIAL_VERSION`、`EDGE_DEVICE_SECRET` 和已验证的 `COURT_CORNERS_JSON`。
- Windows 时间必须自动同步；签名协议只允许约 120 秒时钟偏差。

## 二、安装

将 ZIP 解压到普通目录，例如 `C:\GoodBadmintonSetup`。以管理员身份打开 Windows PowerShell：

```powershell
cd C:\GoodBadmintonSetup\good-badminton-venue-gateway-windows
Set-ExecutionPolicy -Scope Process Bypass
.\install-windows.ps1
```

安装程序将：

1. 检查 Python 3.11 和 FFmpeg/libx264；
2. 复制程序到 `%ProgramData%\GoodBadminton\venue-gateway`；
3. 创建 Python 虚拟环境并安装 `requests`；
4. 创建 `%ProgramData%\GoodBadminton\venue-gateway.env`；
5. 使用 ACL 将配置限制为 SYSTEM、Administrators 和安装用户；
6. 注册 `\GoodBadminton\GoodBadmintonVenueGateway` 开机计划任务；
7. 配置未完成时保持任务禁用，避免携带占位符启动。

## 三、填写配置

继续在管理员 PowerShell 中打开配置：

```powershell
notepad "$env:ProgramData\GoodBadminton\venue-gateway.env"
```

必须填写或核对：

```text
EDGE_GATEWAY_URL=https://for-one-dream.cloud/badminton-edge
EDGE_DEVICE_ID=由业务服务器提供
EDGE_CAMERA_ID=由业务服务器提供
EDGE_CREDENTIAL_VERSION=v1
EDGE_DEVICE_SECRET=由一次性配对结果提供
CAMERA_RTSP_URL=球馆内网真实RTSP地址
FFMPEG_BIN=安装程序自动填写
SEGMENT_SECONDS=2
SPOOL_DIR=安装程序自动填写
HEARTBEAT_SECONDS=10
COURT_CORNERS_JSON=与服务器标定完全一致的四点JSON
```

环境文件不是 PowerShell 脚本，不要给值添加引号，也不要在等号两侧添加额外空格。RTSP URL 中的 `&` 可以直接保留。

## 四、启动前检查与启动

```powershell
& "$env:ProgramData\GoodBadminton\venue-gateway\test-windows.ps1" -ProbeCamera
```

它会检查 Python、配置、FFmpeg/ffprobe、磁盘、摄像头 RTSP、业务服务器 TCP 443和 Windows 时间服务。全部通过后启动：

```powershell
& "$env:ProgramData\GoodBadminton\venue-gateway\start-gateway.ps1"
```

查询状态：

```powershell
Get-ScheduledTask -TaskPath "\GoodBadminton\" -TaskName "GoodBadmintonVenueGateway"
Get-ScheduledTaskInfo -TaskPath "\GoodBadminton\" -TaskName "GoodBadmintonVenueGateway"
```

查看实时错误日志：

```powershell
Get-Content -Wait "$env:ProgramData\GoodBadminton\venue-gateway\logs\venue-gateway.stderr.log"
```

查看建会话和分片已被业务服务器接受的运行日志：

```powershell
Get-Content -Wait "$env:ProgramData\GoodBadminton\venue-gateway\logs\venue-gateway.stdout.log"
```

成功时会出现 `session accepted: ...`，随后出现 `segment accepted: ... index=0`。只有两者都出现，才表示业务服务器已接收到首个视频分片。

业务后台应先显示终端在线。后台点击“开启预览”或“开始采集”后，Windows 会在下一次心跳内启动 FFmpeg 和上传。进程启动不代表上传完成，还应核对后台预览、case ID、分片序号和服务器日志。

## 五、停止与卸载

临时停止（下次开机仍自动启动）：

```powershell
& "$env:ProgramData\GoodBadminton\venue-gateway\stop-gateway.ps1"
```

移除开机任务：

```powershell
& "$env:ProgramData\GoodBadminton\venue-gateway\uninstall-windows.ps1"
```

卸载脚本只移除计划任务，会保留程序目录、密钥配置、日志以及当时仍存在的 spool 文件。不要把 spool 当作长期录像归档；正常切换采集会话时，程序会清理不属于新会话的残留分片。

## 六、故障排查

| 现象 | 检查 |
| --- | --- |
| 找不到 Python 3.11 | 安装 x64 Python，并启用 Python Launcher |
| 找不到 FFmpeg | 将 FFmpeg `bin` 加入系统 PATH，重新打开管理员 PowerShell |
| RTSP 探测失败 | 检查摄像头 IP、554 端口、账号密码、主码流路径和防火墙 |
| 心跳签名失败 | 检查设备 ID、凭据版本、设备密钥和 Windows 时间同步 |
| 服务器 401/403 | 核对服务器端设备绑定；不要反复更换密钥 |
| 服务器 409 | 服务器仍有活动会话；新版会尝试采用响应中返回的会话 ID，否则保持在线并退避重试，等待服务器结束遗留会话 |
| FFmpeg CPU 过高 | 将摄像头稳定设置为 1080p/25fps，或使用性能更好的主机 |
| 任务立即停止 | 查看 stderr 日志及计划任务的 `LastTaskResult` |
| 磁盘持续增长 | 检查公网上传与服务端响应；停止采集但不要直接删除 spool |

## 安全边界

- 不要提交或传播 `venue-gateway.env`。
- 不要在命令行参数中填写设备密钥。
- 不要把 RTSP、密码或密钥上传到 GPU、GitHub 或聊天记录。
- 不需要公网端口映射、DMZ 或公网暴露远程桌面。
- 更新程序前先停止计划任务，并保留配置和 spool。
