# 固定机位单打：空间追踪、近似三维与回合数据契约

## 状态与边界

- 状态：第一版数据基础已接入；真实比赛的模型质量与计分准确率仍需固定机位评测验证。
- 输入：完整、无剪辑的单打比赛；人工球场四角；可选人工球网两端像素坐标。
- 当前不支持：移动机位、镜头切换、集锦、自动人名识别、记分牌识别、双打业务规则或强制判分。
- 兼容性：`detections.jsonl.players.upper/lower` 暂时保留给旧 UI。新功能只能读取 `spatial`，不得把旧字段作为身份、队伍或空间分区依据。

## 基本规则

```text
固定机位标定 → 标准球场坐标(米) → track_id 持续关联 → zone_id 瞬时空间归属
                                     └→ 羽毛球单目近似3D → 击球候选 → 回合状态机
```

- `track_id` 是唯一持久个人键，如 `track_001`；其跨越任何空间区仍不改变。
- `person_id` 在赛后由人工认领到 `track_id`，不由画面上/下、左/右、衣服颜色或队伍推测。
- `zone_id` 是 `rear|mid|front × left|center|right` 的标准球场 3×3 分区，只描述某一帧的位置。
- 关联距离、速度门限和轨迹均以球场米制坐标计算，因此正后方、侧后方、斜后方固定机位使用同一规则。

## 运行与标定

首次运行仍以四个角点标定完整双打场地（6.10 m × 13.40 m）。单打有效边线属于规则层，不重新标定摄像头。球网可由四角推导，也可通过 CLI 明确提供：

```powershell
python main.py --video-path .\videos\match.mp4 --template-path .\templates\court.png --net-image-line 120,310,980,308
```

`--net-image-line` 格式为 `x1,y1,x2,y2`，仅用于记录/校验人工球网线；不传时由球场坐标 `y=6.70m` 反投影得到。固定场馆后续应把四角、球网、ROI、参考宽高比、校验阈值保存为命名机位配置；本分支没有把自动重新标定作为默认行为。

## detections.jsonl v2 片段

```json
{
  "schema_version": "2.0",
  "players": {"upper": {"...": "legacy compatibility"}},
  "spatial": {
    "coordinate_system": "standard_badminton_court_m",
    "tracks": [{
      "track_id": "track_001",
      "person_id": null,
      "court_xy_m": [2.31, 10.08],
      "zone_id": "front_center",
      "status": "detected",
      "confidence": 0.84,
      "location_evidence": {"method": "ankles_midpoint", "source": "far_roi"}
    }],
    "shuttlecock_3d": {
      "status": "approximate",
      "xyz_m": [3.11, 6.52, 1.47],
      "confidence": 0.31,
      "source": "single_view_physics_fit"
    },
    "rally": {"state": "active", "rally_id": 4, "score_status": "unknown"}
  }
}
```

人体 `status` 为 `detected` 或短时 `predicted`。`location_evidence` 会保留双脚踝中点、单脚或框底中心等降级来源与置信度。`predicted` 不得伪装为检测真值。

## 羽毛球、击球与得分证据

固定单目视频仅可靠提供经透视变换的平面近似。`shuttlecock_3d.xyz_m[2]` 是受物理约束的高度代理，`source=single_view_physics_fit` 且置信度上限为 0.45；它不是标定相机三维测量。缺失或预测球点不会进入击球/失误证据。

`hit_events` 当前是“球与真实检测球员在球场坐标中接近”的低置信候选，目的在于保留后续击球序列训练/规则的输入，并不构成得分依据。

回合状态机只在两名真实球员和连续真实球证据存在时开启；球证据中断或视频结束时结束。默认分数为：

```json
{"status":"unknown","included_in_player_statistics":false}
```

只有未来引入并验证了明确终止证据（落点/出界/网前/记分确认）后，高置信回合才能写入胜负、关键分、挑战、排行榜和能力统计。`unknown` 回合仍保留跑位、姿态和击球候选数据。

## 输出与可视化

处理结束生成 `spatial_match_summary.json`：

- `identity_claims`：赛后 `track_id → person_id` 的可选映射；
- `player_style_inputs`：检测帧数、预测帧数、移动距离、9 区覆盖；其范围严格为 `movement_and_space_only`；
- `rallies`：起止帧、击球候选、回合置信度和可拒绝的得分状态。

现有热力图仍使用旧兼容字段；后续应优先迁移为按 `track_id` 聚合 `spatial.tracks`，从而可同时支持单打与四人双打。不要将本版本的移动/覆盖输入宣传为综合羽毛球能力、失误率或胜率评分。

## 验证与下一步

- 单元测试覆盖正后方与斜后方球场映射、跨任意 zone 的 ID 不变、单目三维低置信标记、以及没有终止证据时的 `unknown` 分数。
- 已有 `evaluation/far_player/` 只衡量远端人体检测。接入正式 ByteTrack 前，仍需每视频至少 20 个连续人工复核样本，并量化 ID switch、遮挡恢复、最长可信预测长度。
- 先以真实 1080P/4K 固定机位比较全画面 1280、全场 640 + 紧致远端 ROI；不以一次观感替换或训练模型。
- 双打在视觉层复用本模块的多轨集合；另加模式规则、队伍归属与四人赛后认领，不能把单打 `upper/lower` 逻辑复制过去。
