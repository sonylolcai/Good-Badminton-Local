# S-03 三模块拆分验收记录

记录时间：2026-09-14

## 已通过

| 边界 | 证据 | 结论 |
|---|---|---|
| 业务 WebUI | 默认 `build_ui()` 只挂载经营、场馆、活动、比赛、视频/GPU 运维、权限、赛后业务、教练档案和业务任务历史；没有分析运行或球路复核回调 | 业务功能原地保留，分析入口已切走 |
| 数据分析评测平台 | 独立 Next.js 项目 `Good-Badminton-Evaluation-Platform`；Python 只提供 `python -m analysis_platform.api` 评测 API | 不依赖业务 WebUI 启动，不复用业务项目路由或侧边栏 |
| GPU 服务 | `api/` 静态扫描没有 `webui` 导入；整视频 Worker 使用 `analysis_platform.runner` | GPU 后端不再反向依赖 UI 包 |
| 业务边界 | `business_gateway/`、`operator_api/`、`webui/` 没有导入 `analysis_platform.ui` | 业务端不依赖评测 UI |
| 回归 | `.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'` | 345 项通过 |
| Next.js | `npm run lint`、`npm run build` | 通过；7 个评测页面生成成功 |
| 编译与差异 | `compileall`、`git diff --check` | 通过 |
| 启动烟测 | 评测 API `:8010`；Next.js 总览、工作台、数据和复核页面 `:3100` | HTTP 200 |

身体参数、能量估算、赛后身份绑定、教练档案、纵向反馈、宣传视频和业务任务历史始终属于业务服务。本次只改变默认页面装配，没有迁移这些业务数据，也没有修改 GPU 算法、模型或合同语义。

## 尚未通过

- 未向真实远端 GPU 提交固定真实视频重新跑完整链路。
- 未用已发布的固定真实数据集重新计算球检测召回率、连续漏检和失败案例分布。
- 因此本记录只能证明代码拆分和本地回归通过，不能证明球检测成功率已经提高，也不能将 `S-03` 标记为最终完成。

## 最终放行条件

在同一视频、同一模型、同一参数和同一硬件条件下完成一次远端 Run，并将产物 SHA-256、GPU 身份、指标报告和门禁结果登记到平台；业务服务再读取该已完成结果，验证身份绑定、身体参数、教练跟进和宣传视频不会触发重复推理。
