# 球馆中转主机安装包

本程序只在球馆现场的中转主机运行：它本地读取摄像头 RTSP，切成 2 秒 MP4 片段并以签名请求上传到业务服务器。RTSP 地址、摄像头密码和原始视频不会发送给业务服务器或 GPU。

## 安装

以 Ubuntu 为例，解压安装包后：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3 python3-venv
sudo useradd --system --create-home --shell /usr/sbin/nologin goodbadminton
sudo install -d -o goodbadminton -g goodbadminton -m 0750 /opt/good-badminton-venue-gateway
sudo cp -R . /opt/good-badminton-venue-gateway/
sudo -u goodbadminton python3 -m venv /opt/good-badminton-venue-gateway/.venv
sudo -u goodbadminton /opt/good-badminton-venue-gateway/.venv/bin/pip install -r /opt/good-badminton-venue-gateway/requirements.txt
```

将业务服务器上仅 `ubuntu` 可读的配置文件安全复制到球馆主机，并保存为 `/etc/good-badminton-venue-gateway.env`、权限 `600`。不要把其中的 `EDGE_DEVICE_SECRET` 发到聊天、Git 或截图。

在该文件只补两项现场信息：

- `CAMERA_RTSP_URL`：摄像头在球馆 LAN 上的 RTSP 地址；
- `COURT_CORNERS_JSON`：实际画面中球场四个角的像素坐标，顺序必须与业务服务器审核通过的标定一致。

复制 `good-badminton-venue-gateway.service` 至 `/etc/systemd/system/` 后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now good-badminton-venue-gateway
sudo journalctl -u good-badminton-venue-gateway -f
```

首次安装前，业务服务器必须把这组真实角点保存并标记为 `validated`；可复制 `deploy/business-server/good_yushi_calibration.sql.template`，替换四组实际像素坐标后导入。程序不会用示例的 `[[0,0],[1,0],[1,1],[0,1]]` 伪造标定。未完成标定时，业务服务器会拒绝开始解析会话，这属于预期保护。

## 维护发布包

`deploy/venue-gateway/agent.py` 和 `business_gateway/edge_contract.py` 是 Linux、Windows 和 macOS 网关唯一的共享运行时源码。修改它们后，在仓库根目录执行：

```bash
python3 deploy/venue-gateway/build_packages.py
python3 -m unittest discover -s deploy/venue-gateway/tests -p 'test_package_parity.py'
```

构建脚本会重新生成 Linux、Windows 和 macOS 的嵌入式代理文件及 ZIP 包；一致性测试会验证三个平台的代理和签名协议字节级相同。平台专属的安装脚本、服务定义和操作说明仍保留在各自目录中。
