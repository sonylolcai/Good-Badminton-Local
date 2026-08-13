# 10秒单打对抗模拟数据

这是一套合成数据，用来检查现有和未来分析系统是否拥有足够字段支持运动表现与失误分析。它不是模型跑出来的真实结果。

## 场景

- 标准球场：宽6.10米、长13.40米，球网位于 y=6.70米。
- player_a 位于下半场，player_b 位于上半场。
- 回合持续10秒，共12次击球。
- player_b 在9.78秒尝试直线吊球，10.00秒球触网；模拟标注为非受迫性下网失误，player_a赢得回合。

## 文件

- player_tracking_10hz.csv：两名球员每0.1秒的位置、速度、加速度、方向、场区、阶段和跟踪质量。
- shuttlecock_tracking_30hz.csv：羽毛球每1/30秒的三维位置、速度、轨迹阶段和跟踪质量。
- events.jsonl：12次击球事件，一行一个JSON对象。
- rally_summary.json：派生统计、示例评分及样本有效性。
- stick_match_10s.mp4：由上述数据驱动的10秒火柴人对打回放，包含羽毛球和事件面板。
- render_stick_match.py：重新生成火柴人回放的脚本。
- ACTUAL_VS_TARGET_DATA.md：当前项目真实输出与这套目标数据合同的逐项对照。

重新生成视频：

```powershell
.\.venv\Scripts\python.exe .\samples\simulated_rally_10s\render_stick_match.py
```

## 重要字段

- tracking_status=detected：模型真实检测。
- tracking_status=predicted：跟踪器或插值生成，不能冒充真实检测。
- confidence：本条观测的可信度。
- stroke_quality_score：模拟的击球质量字段；当前项目尚不能从球员位置单独得到它。
- forcedness：受迫/非受迫失误归因；需要球路、对手压力和回合上下文。

## 评分解释

rally_summary.json中的0～100分只是评分数据合同演示。单回合只能描述这一回合，不能代表运动员稳定水平。正式评分需要至少20个有效回合、每人至少50次有效击球、检测完整率不低于90%，并使用同水平对手和教练标签建立基准。

## 能力边界

仅靠player_tracking_10hz.csv可以较可靠计算跑位、速度、覆盖、回位和站位倾向。要计算击球质量、球路、落点和失误率，还必须使用连续球轨迹、击球事件、回合结局及其置信度。
