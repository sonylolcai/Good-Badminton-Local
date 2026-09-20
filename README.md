# Good Sports GPU API

`Good-Badminton-Local` 的 GPU 服务工作树：负责固定机位视频解码、匿名人体/球/轨迹证据、多运动视觉 profile、分片会话及模型发布。

它不包含场馆终端、用户/场馆/积分业务、数据库迁移、运营 Web、评测控制台、RTSP 凭据或模型权重。业务系统通过受保护的 HTTP API 调用它；视觉证据不足时必须保持 `unknown`，不产生裁判级赛果。

## 本地检查

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gpu_code_boundary tests.test_candidate_photos tests.test_gpu_api tests.test_gpu_sport_profiles tests.test_gpu_pure_launchers -v
```

## 运行与发布

- API：`api/app.py`；健康检查不需密钥，其他端点使用 `X-API-Key`。
- 配置：从 `.gpu-api.env.example` 复制到 Git 忽略的运行时文件；绝不提交密钥、任务数据或权重。
- GPU 发布：使用 `deploy/package_gpu_api.ps1` 创建源码包，再按 `deploy/refresh_gpu_api_from_zip.sh` 的哈希校验与回滚流程部署。

远程 GPU、真实视频吞吐与模型质量须在目标 CUDA 环境单独验证；本地单元测试不构成这些结论。
