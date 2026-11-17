# 球馆 Mac 中转主机安装包

此安装包运行在球馆的 Mac mini / Mac 上。它从球馆局域网读取 RTSP，切成 2 秒 MP4 片段，并通过 HTTPS 签名请求上传至业务服务器。摄像头地址、账号、密码和原始 RTSP 始终只保留在球馆内网；Mac 不监听公网端口，也不会直接连接 GPU。

Mac 只需一次安装后常驻运行。它每 10 秒向业务服务器发送签名心跳并领取采集指令；运营人员只在业务后台开启预览、开始/停止采集和开关 GPU，不需要到球馆操作 Mac。

## 安装前准备

- 使用有线以太网连接 Mac、路由器和 PoE 交换机；不要用球馆 Wi-Fi 承载摄像头流。
- 确认 Mac 能访问摄像头的 RTSP 地址，且可出站访问 `https://for-one-dream.cloud/badminton-edge`。
- 在业务服务器完成场馆、场地、摄像头注册和真实球场四角标定；未标定的摄像头会被服务端拒绝创建解析会话。
- 将业务服务器导出的环境文件通过安全渠道复制到 Mac。本文件含 `EDGE_DEVICE_SECRET`，不得提交 Git、发到聊天或截图。

## 安装与启动

解压后，在终端执行：

```zsh
cd good-badminton-venue-gateway-macos
chmod +x install-macos.sh
./install-macos.sh
```

安装脚本会在当前 macOS 用户的 `~/Library/Application Support/GoodBadminton/venue-gateway` 建立运行目录与 Python 虚拟环境，并在首次缺少工具时提示安装 Homebrew、`python@3.11` 与 `ffmpeg`。它不会替你写入任何密钥；配置仍为占位符时不会启动中转服务。

把受保护的业务服务器环境文件保存为：

```text
~/Library/Application Support/GoodBadminton/venue-gateway.env
```

并编辑两项现场信息：

- `CAMERA_RTSP_URL`：球馆 LAN 内的 RTSP 地址；
- `COURT_CORNERS_JSON`：真实画面四个球场角点，顺序必须和服务器已验证的标定一致。

填写配置后加载后台常驻任务：

```zsh
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.goodbadminton.venue-gateway.plist"
tail -f "$HOME/Library/Logs/GoodBadminton/venue-gateway.stderr.log"
```

停止中转程序：

```zsh
launchctl bootout gui/$(id -u)/com.goodbadminton.venue-gateway
```

## 现场验证

先在 Mac 上运行一次前台程序，确认 RTSP 与签名上传都正常：

```zsh
GOOD_BADMINTON_VENUE_ENV_FILE="$HOME/Library/Application Support/GoodBadminton/venue-gateway.env" \
  "$HOME/Library/Application Support/GoodBadminton/venue-gateway/.venv/bin/python" \
  "$HOME/Library/Application Support/GoodBadminton/venue-gateway/agent.py"
```

业务后台应先看到：终端在线、摄像头已连接。点击该场地的“开启预览”或“开始采集”后，Mac 会在下一次心跳内自动开始上传，随后出现预览和 case ID。确认 GPU 服务在线后，再从该场地开启“推送到 GPU”；关闭推送会停止新的 GPU 分发，不会删除已产生的 SQL 解析结果。点击“停止视频”后，Mac 回到仅心跳模式。
