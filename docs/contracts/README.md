# 独立运动服务的统一视觉协议

羽毛球和网球视觉服务可以使用完全不同的模型、跟踪器和发布节奏，但进入业务层前统一为 [`visual-observation.v1`](visual_observation_v1.schema.json)。业务端据此复用速度、距离、覆盖区域、时间轴对齐、宣传视频合成和视觉大模型取帧逻辑。

边界约定：

- GPU 视觉服务只输出人物、Pose、球和证据状态；
- 判分、回合识别、动作评价、技战术结论属于业务层；
- `detected` 是本帧模型实测，`predicted` 是跟踪/缓存结果，`missing` 是没有可靠证据；
- 人物框检测成功但 Pose 失败时，人物记录仍然有效；
- 每条记录必须使用原视频帧号和媒体时间，不使用网络上传或任务处理时间。

当前羽毛球 `stream-session.v1` 事件可通过 `business_gateway.visual_observation.from_stream_event` 转换。网球服务直接生成同版本 JSONL，因此两个服务无需合并为一个运行时，也无需通过 `sport_id` 在同一 GPU 进程中切换模型。
