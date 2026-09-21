# 业务服务器部署

此目录部署的是业务控制面：PostgreSQL 迁移、签名摄像头接入网关、运营 API 和 Next.js SaaS 后台。GPU CV 服务继续部署在独立 GPU 主机；业务服务器只通过受保护的 `GOOD_BADMINTON_GPU_API_URL` / `KEY` 调用它。

## 上线前条件

- Ubuntu/Debian + systemd；已安装 `python3`、`python3-venv`、`nginx`、`curl`、Git、Node.js 20+ 与 npm；
- 代码已经通过只读 deploy key 拉取到业务服务器，例如 `/home/ubuntu/apps/good-badminton`；
- PostgreSQL 已创建一个空业务库；脚本会创建 `business` schema、基础表和后续迁移；
- `operator` 与 `api` 两个 DNS 已解析，且对应 TLS 证书已经签发；
- 运营域名位于 VPN / SSO / 访问网关之后。当前代码尚未实现登录与租户角色认证，不能把运营域名公开暴露；
- GPU 主机仅允许业务服务器访问其非健康检查接口，并要求 `X-API-Key`。

## 1. 创建受保护配置

```bash
sudo install -d -m 0750 /etc/good-badminton
sudo cp deploy/business-server/business.env.example /etc/good-badminton/business.env
sudo chmod 640 /etc/good-badminton/business.env
sudoedit /etc/good-badminton/business.env
```

设置 `GOOD_BADMINTON_EDGE_MASTER_KEY` 时只生成一次，例如 `openssl rand -base64 48`。该值由边缘网关与运营 API 共用，不能写入 Git、终端镜像、浏览器或截图。

`GOOD_BADMINTON_EDGE_GATEWAY_PUBLIC_URL` 是浏览器获得的 HTTPS 地址；`GOOD_BADMINTON_EDGE_GATEWAY_INTERNAL_URL` 保持为业务服务器的回环地址。这防止把 `127.0.0.1:18080` 错误发送给运营人员的浏览器。

## 2. 部署或升级

先以拥有 deploy key 的服务器用户更新代码；脚本不自行生成密钥，也不尝试改写远端 Git 状态。

```bash
cd /home/ubuntu/apps/good-badminton
git pull --ff-only origin main

sudo bash deploy/business-server/deploy-business-server.sh \
  --app-dir "$PWD" \
  --run-user ubuntu \
  --operator-host operator.example.com \
  --api-host api.example.com
```

脚本会：

1. 创建或复用独立的 `.venv-business`，只安装业务控制面依赖；
2. 执行尚未记录的 `business_gateway/migrations/*.sql`；
3. 用生产环境的 `NEXT_PUBLIC_OPERATOR_API_BASE_URL` 构建 Next.js；
4. 安装并重启三个 systemd 服务；
5. 写入 Nginx HTTPS 反向代理并检查配置；
6. 在回环地址验证三个服务健康状态。

若 Nginx/TLS 已由其他网关管理，使用 `--skip-nginx`，但仍必须让 `operator` 域名的 `/api/*` 指向 `127.0.0.1:8000`、让 `api` 域名的 `/api/v1/edge/*` 指向 `127.0.0.1:18080`。

## 3. 日常检查与日志

```bash
sudo bash deploy/business-server/status-business-server.sh
sudo journalctl -u good-badminton-edge-gateway -u good-badminton-operator-api -u good-badminton-operator-web -f
```

公开入口只应开放 443。端口 `18080`、`8000`、`3000` 均监听在 `127.0.0.1`，不要直接在安全组或防火墙中放行。
