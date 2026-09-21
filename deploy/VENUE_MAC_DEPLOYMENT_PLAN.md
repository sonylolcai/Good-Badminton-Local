# 好雨时节球馆：一号场现场部署方案（macOS）

## 已分配的稳定标识

数据库主键均为不含业务含义的 UUIDv4；运营人员使用场馆/场地 `code` 和名称识别对象。

| 对象 | 名称/编码 | UUID |
| --- | --- | --- |
| 租户 | 好雨时节试点 | `64c28b72-1bd7-4c09-b73a-5cc8c43d33a3` |
| 场馆 | 好雨时节球馆 / `haoyushijie-venue-01` | `d73a28e0-f4cb-4cfc-9ece-9424d252c10d` |
| 场地 | 一号场 / `court-01` | `84b70d2e-8e5e-49bd-a6e7-a506aa68c7cc` |
| Mac 中转终端 | `haoyushijie-gateway-01` | `d958591c-ddcd-46ec-a47a-dd449cb9c9eb` |
| 摄像头 | `haoyushijie-cam-01` | `2e5b3f68-01d1-4168-9b06-2440aeb68b4e` |

设备密钥不记录在本文档。业务服务器已将其写入仅 root 可读的 `/etc/good-badminton/good-yushi-venue-gateway.env`。

## 首次试点拓扑

```text
固定 PoE 网络摄像头 --Cat6--> PoE 交换机 --LAN--> 路由器 --5 GHz 主 Wi-Fi--> Mac 中转主机
                                            |
                                     路由器 WAN / Internet
                                            |
                                    仅 HTTPS 出站 443
                                            |
                         业务服务器（预览、会话、GPU 分发开关）
                                            |
                              GPU 服务（仅业务服务器调用）
```

RTSP 原始视频只在球馆 LAN 内从摄像头经交换机、路由器流向 Mac 的主 Wi-Fi。Mac 将签名的 2 秒 MP4 片段发送到业务服务器；业务服务器可显示临时预览，并在后台操作员点击“开始推送到 GPU”后才向 GPU 转发。试点阶段不接入 OSS。

## 一片场地的硬件清单

| 设备 | 最低要求 | 现场建议 |
| --- | --- | --- |
| 摄像头 | 固定式 PoE IP Camera，RTSP，H.264/H.265，1080p/25fps | 不用 PTZ、自动变焦或自动巡航；先配置 H.264、CBR 4–6 Mbps、GOP 2 秒 |
| 摄像机位 | 看见完整双打场地和边界 | 后场中线后方、离地约 5.5–7 m；横向画面；避开灯具直射与人体频繁遮挡 |
| PoE 交换机 | 千兆、802.3at PoE+ | 8 口以上；PoE 预算至少为摄像头标称功耗总和的 1.3 倍（四机位建议 120 W） |
| 中转主机 | macOS、16 GB 内存、512 GB SSD、稳定 5 GHz Wi-Fi | Mac mini M2/M4 或更高；本试点可通过 Wi-Fi 接入，保留 USB-C 千兆网卡作为现场故障回退 |
| 网关 | 千兆 NAT、可设 VLAN / ACL | 关闭 UPnP；不做任何公网端口映射 |
| 供电 | 稳定市电 | 摄像头 PoE 交换机、路由器、Mac 接同一台 1000 VA 级 UPS |

Mac 不做 GPU 推理，也不做长期视频存储；SSD 用于短暂分段缓冲。若断网，当前试点中转程序会重试和由 launchd 拉起，但并非离线录像归档器，因此应优先保证上行稳定。

## 网络配置

本次试点不要求 VLAN：让摄像头、路由器 LAN 和 Mac 主 Wi-Fi 位于同一个受控私网即可（例如均为 `192.168.50.0/24`）。路由器的 **WAN/Internet** 口只接光猫/上级网络；从任意一个空闲 **LAN** 口接一根网线到 PoE 交换机的 Uplink 口（若交换机无 Uplink 标识，则接任意普通千兆口）。摄像头接 PoE 口。

路由器已有三个“正常网口”时，通常它们都是可互换的 LAN 口，任选一个接交换机即可；不要把交换机接到标有 `WAN`、`Internet`、地球图标或上行标识的端口。若路由器已能上网和提供 Wi-Fi，现有上网线不要移动。

若以后场地增多，再由现场网络管理员创建三个网段（地址仅为例）：

| 网段 | 用途 | 示例 | 规则 |
| --- | --- | --- | --- |
| VLAN 20 | 摄像头 | `10.20.0.0/24` | 摄像头固定 IP；禁止直接访问公网 |
| VLAN 30 | Mac 中转主机 | `10.30.0.0/24` | 仅允许访问摄像头 RTSP（TCP 554）和公网 TCP 443、DNS、NTP |
| VLAN 10 | 员工/访客网络 | 独立网段 | 不允许直接访问摄像头；管理摄像头仅用受控管理员设备 |

Mac 必须接“主 Wi-Fi”，不要接访客 Wi-Fi；关闭访客网络隔离、AP/client isolation，确保 Mac 能访问摄像头 TCP 554。路由器上需要允许 **Mac → 摄像头 TCP 554**，以及 **Mac → `for-one-dream.cloud:443`**。不需要、也不应配置任何从公网进入 Mac 或摄像头的端口转发。摄像头用 DHCP 保留或静态地址，地址、管理员账号只记录在球馆设备台账中。

单场地试点把摄像头主码流固定在 1080p/25fps、H.264 CBR 4–6 Mbps。公共上行带宽至少 50 Mbps；多场地按每路实际码率加 30% 余量估算，四场地建议稳定上行 100–200 Mbps。Mac 的 5 GHz Wi-Fi 到路由器需要在摄像头码流运行时稳定承载至少 10 Mbps，信号强度建议优于 -65 dBm；若丢帧、断连或距离过远，立即改用 USB-C 千兆网卡有线连接。

## 明天现场执行顺序

1. 断电布线：摄像头至 PoE 交换机使用独立 Cat6，贴上“场馆/场地/摄像头”标签；Mac、交换机、路由器均接 UPS。
2. 固定机位：锁死俯仰、焦距与曝光策略；录一段无人和有人画面，确认整片场地四角可见。机位移动、变焦或大幅裁剪后必须重新标定。
3. 配网络：给摄像头分配固定 IP；在 Mac 上验证 RTSP，但不要把 RTSP URL 发到业务服务器。
4. 装 Mac 包：解压 `good-badminton-venue-gateway-macos.zip`，运行 `./install-macos.sh`。它安装 `ffmpeg`、Python 虚拟环境和当前登录用户的 launchd 常驻任务。
5. 安全导入中转配置：把服务器的受保护环境文件经 SSH 直接重定向到 Mac 的 `~/Library/Application Support/GoodBadminton/venue-gateway.env`，设为 `chmod 600`；补齐 RTSP 地址。
6. 标定：截取实际画面，读取四个球场角点像素坐标，写入业务服务器的标定 SQL 模板并标记 `validated`。不得用示例坐标代替。
7. 先前台试跑：先确认业务后台的终端在线/摄像头已连接；再从后台点击“开启预览”，观察 Mac 日志、预览和 case ID。确认后执行 `launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/com.goodbadminton.venue-gateway.plist"` 让后台常驻。此后每日启停和 GPU 控制都在后台完成。
8. GPU 联调：先确认业务后台显示 GPU 已连接，再在“一号场”点击“开始推送到 GPU”；检查会话 case ID、GPU 状态和事件日志。点击“关闭推送”验证停止新的 GPU 分发。

## macOS 获取与密钥导入

安装包已经放在业务服务器的 `/home/ubuntu/good-badminton-venue-gateway-macos.zip`，且不含密钥。Mac 取得包后：

```zsh
cd ~/Downloads
unzip good-badminton-venue-gateway-macos.zip
cd good-badminton-venue-gateway-macos
chmod +x install-macos.sh
./install-macos.sh
```

如果 Mac 可以 SSH 到业务服务器，使用下面方式导入密钥文件；终端不会回显文件内容：

```zsh
mkdir -p "$HOME/Library/Application Support/GoodBadminton"
ssh stone 'sudo cat /etc/good-badminton/good-yushi-venue-gateway.env' > \
  "$HOME/Library/Application Support/GoodBadminton/venue-gateway.env"
chmod 600 "$HOME/Library/Application Support/GoodBadminton/venue-gateway.env"
```

随后只在 Mac 编辑 `CAMERA_RTSP_URL` 与真实的 `COURT_CORNERS_JSON`。该 Mac 包用的是当前登录用户的 LaunchAgent，适合明天联调；若后续要无人值守重启恢复，需要将其升级为受控专用 macOS 账户的 LaunchDaemon。
