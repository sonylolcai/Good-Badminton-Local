# 本地配置与密钥维护

真实配置集中保存在本项目根目录的 `.secrets/`。它不进入 Git、Docker 镜像或代码发布包。目录权限为 0700，文件为 0600；Windows 使用服务账号对应的 ACL。

## 初始化与检查

```bash
python3 tools/local_secrets.py init
python3 tools/local_secrets.py check
git config --local core.hooksPath .githooks
```

`init` 无损复制现有配置并核对内容，再用相对符号链接保留旧路径。没有旧文件时只创建私密模板，默认不激活；两份内容不同时停止，不覆盖。以后只编辑 `.secrets/` 下的文件。重新执行可收拢旧路径中新生成的配置。Windows 没有符号链接权限时，显式把启动器配置路径指向私密文件，不要把本机链接当成部署文件。

公开的 `config/*.example` 和已有 `*.env.example` 只保留无密钥模板。未接入运行时的文件（例如部署清单）只是保管位置，不代表服务已经读取它们。前端仅配置公开接口地址，不能存放 AppSecret、数据库密码、对象存储永久密钥或 GPU Key。

需要保管 SSH 私钥时，放在 `.secrets/ssh/`，文件设为 0600，目录设为 0700。线上管理员密码不能写入公开文档。

## 提交与打包

提交前检查读取实际暂存内容；禁止私密配置路径、私钥、常见访问令牌，以及本项目私密 env 中已知的非占位密钥值。只输出文件路径和行号，不回显值。新电脑上需要再次安装钩子。不要通过 `git add -f` 或 `--no-verify` 绕过保护。

检查不能保证识别所有未知格式的密钥，也不追溯已发布的 Git 历史。`.gitignore` 不会自动移除已跟踪文件。Docker 和发布包也必须独立排除私密目录。

## 后续服务器替换

1. 在 `.secrets/deployment.env` 维护已确认的主机、用户、端口、私钥路径、部署目录与远端配置路径；这个文件本身不会连接或部署。
2. 获得连接目标服务器的授权后，先备份远端配置并核对状态，再单独传输需要的配置；代码包不携带 `.secrets/`。
3. Linux 运行时可继续使用 `/etc/good-badminton/` 等服务专用目录；Windows 球馆继续使用安装器保护的 ProgramData 配置目录，与本地清单对应。
4. 重载后验证健康和权限，失败则恢复原配置。长期主密钥、身份哈希密钥和数据库密码不能随意重新生成。
5. 本地 `.secrets/` 不受 GitHub 备份保护，需要独立的加密备份。

## 本项目文件清单

以下文件由 `config/local-secrets.json` 定义；配置值只存在本机。

| `.secrets/` file | Compatibility path |
| --- | --- |
| `development.env` | `.env.development` |
| `production.env` | `.env.production` |
| `gpu-api.env` | `.gpu-api.env` |
| `remote-gpu.env` | `.webui-remote-gpu.env` |
| `operator.env` | `.webui-operator.env` |
| `dogecloud.env` | DogeCloud 服务端媒体适配器读取；参见 `MEDIA_STORAGE_CONTRACT.md` |
| `deployment.env` | manual / inactive |

多吉云媒体适配器读取本项目 `.secrets/dogecloud.env`，或显式的 `DOGECLOUD_ENV_FILE`。已注入的进程环境变量优先。业务侧新增媒体台账需先执行数据库迁移；GPU 归档开关仍由运行环境显式控制，详情见 [媒体目录和权限规范](MEDIA_STORAGE_CONTRACT.md)。
