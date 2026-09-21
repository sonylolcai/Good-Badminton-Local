# 开发 / 生产环境配置

所有环境差异都由变量或被忽略的环境文件承载；业务代码、小程序页面和 Next.js 运营后台不再为上线修改地址。

## 业务服务、运营 API 与 SaaS 后台

1. 开发：复制 `.env.development.example` 为 `.env.development`。业务服务默认是 `http://127.0.0.1:8080`；运营 API 默认可运行在 `http://127.0.0.1:8000`，Next.js 后台默认是 `http://127.0.0.1:3000`。
2. 生产：在部署平台注入变量，或在受保护的位置保存 `.env.production` 并以 `GOOD_BADMINTON_ENV_FILE` 指向它。必须设置 `GOOD_BADMINTON_BUSINESS_API_URL=https://实际业务域名`。
3. 用 `python -m business_gateway.dev_api` 启动业务服务、`python -m operator_api.main` 启动运营 API；在 `webui-next` 目录运行 `npm run dev` 启动 SaaS 后台。变更变量后重启相应进程即可。

生产模式拒绝空地址、`localhost`/`127.0.0.1` 和非 HTTPS 的业务 API 地址，防止上线服务或运营 WebUI 误连本机。

| 变量 | 开发默认 | 生产要求 |
| --- | --- | --- |
| `GOOD_BADMINTON_APP_ENV` | `development` | `production` |
| `GOOD_BADMINTON_BUSINESS_API_URL` | `http://127.0.0.1:8080` | 非本机 HTTPS 业务域名 |
| `GOOD_BADMINTON_BUSINESS_HOST` / `PORT` | `127.0.0.1` / `8080` | 按反向代理/容器设置 |
| `GOOD_BADMINTON_OPERATOR_API_ALLOWED_ORIGINS` | 本地 Next.js 地址 | 精确的运营后台 HTTPS 域名 |
| `NEXT_PUBLIC_OPERATOR_API_BASE_URL` | `http://127.0.0.1:8000` | 运营 API HTTPS 域名，不含任何密钥 |

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
