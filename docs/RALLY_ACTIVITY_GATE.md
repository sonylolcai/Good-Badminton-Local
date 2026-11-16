# 本机回合活动门禁 v1

2026-10-30，按用户“优先优化回合开始/结束”要求，本机校准场地离线派生版本2.6接入 `badminton_analysis/analysis/rally_activity.py`。保留原始候选供复核，回合结束后进入等待重新开球；相邻异侧人员候选和连续实测球飞行可重启候选活动区间，局间候选不贡献个人击球汇总。不推断得分或确认触地；首段可能在录像开始前已开球。

独立迁移/验证：评测副本 `backend/evaluation_suites/shuttle_tracknet_ab/rally_activity.py` 与本文件对应模块内容一致，但分别在各项目环境执行测试和保存检测的离线重建，未复用评测输出冒充本机结果。两个环境在192723录像独立派生均产生首个自动结束4.852673秒、63条原始候选中55条活动/8条等待；不覆盖保存的90条人工复核集合。未重新运行TrackNet、未改变标定、未接触远程GPU。

本机74个相关测试通过，包含GPU API、sport profiles、pure launchers、四角门禁和新增回合状态测试。原始证据、人工数据、当前可见结果、参数、漏检与误开限制及恢复方法见工作区 `docs/personal-performance-v1/rally-activity-gate.md`；内容指纹和两个环境独立产物见 `outputs/rally-activity-20261030/`。
