# 开发 / 生产环境配置

所有环境差异都由变量或被忽略的环境文件承载；业务代码、小程序页面和 WebUI 不再为上线修改地址。

## 业务服务与 WebUI

1. 开发：复制 `.env.development.example` 为 `.env.development`，默认业务服务为 `http://127.0.0.1:8080`，WebUI 为 `http://127.0.0.1:7861`。
2. 生产：在部署平台注入变量，或在受保护的位置保存 `.env.production` 并以 `GOOD_BADMINTON_ENV_FILE` 指向它。必须设置 `GOOD_BADMINTON_BUSINESS_API_URL=https://实际业务域名`。
3. 用 `python -m business_gateway.dev_api` 启动业务服务、`python -m webui.app` 启动 WebUI。二者会在启动时读取该配置；变更变量后重启相应进程即可。

生产模式拒绝空地址、`localhost`/`127.0.0.1` 和非 HTTPS 的业务 API 地址，防止上线服务或运营 WebUI 误连本机。

| 变量 | 开发默认 | 生产要求 |
| --- | --- | --- |
| `GOOD_BADMINTON_APP_ENV` | `development` | `production` |
| `GOOD_BADMINTON_BUSINESS_API_URL` | `http://127.0.0.1:8080` | 非本机 HTTPS 业务域名 |
| `GOOD_BADMINTON_BUSINESS_HOST` / `PORT` | `127.0.0.1` / `8080` | 按反向代理/容器设置 |
| `GOOD_BADMINTON_WEBUI_HOST` / `PORT` | `127.0.0.1` / `7861` | 私网或反向代理监听地址 |

`GOOD_BADMINTON_GPU_API_URL`、`GOOD_BADMINTON_GPU_API_KEY` 和数据库 URL 是服务端私密变量，不写入小程序构建文件。

## 小程序

在 `customer-inter` 目录运行：

```powershell
# 本机开发，生成 http://127.0.0.1:8080 配置
npm run env:dev

# 构建生产包前由 CI/部署环境注入真实 HTTPS 域名
$env:BDTEACH_BUSINESS_API_BASE_URL = 'https://api.example.com'
npm run env:prod
```

命令生成被 git 忽略的 `miniprogram/config/runtime.generated.ts`。该文件只包含环境名和业务 API 地址；不含 GPU、数据库或任何密钥。生产生成命令在未提供 HTTPS 地址时会失败。
