# 本地羽毛球击球人物范围门禁

2026-10-30 按用户要求，羽毛球击球候选只允许当前人物脚点在标定四角内触发，边界包含，不使用人体跟踪的场外裕量。球与手可以位于四角外；球飞行检测区域维持既定 TrackNet 配置。

在线入口 `fixed_camera_match.py`，离线入口 `offline_shot_reconstruction.py`（派生版本 2.5），共享纯几何 `court_contact_gate.py`。最近手来自场外人物时不改归给较远场内人物。缺脚点/低位置置信度不触发；场内捡球/准备动作仍可能成为候选，不能把范围门禁当作击球分类器。

此处代码单独通过 67 项相关测试，包含 GPU API/运动隔离/启动入口契约测试；评测平台的独立门禁副本另有 48 项相关回归。经空闲和 drain 核验后已重载本机 8082，沿用原私密配置、MPS 与 TrackNet；未重新推理视频，也未部署远程。完整契约和可追溯证据见工作区 `docs/personal-performance-v1/court-contact-scope.md` 与 `outputs/court-contact-scope-20261030/`。这些结果不代表远端或新视频准确率已验证。
