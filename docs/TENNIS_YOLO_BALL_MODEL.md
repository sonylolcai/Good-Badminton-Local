# Tennis YOLO Ball Model Contract

The tennis GPU service can start in pose-only mode. In WebUI, select **only
person pose** when no ball model is required. Selecting YOLO ball detection
uses one of the following explicitly separated modes:

1. **Dedicated tennis mode**: configure a dedicated checkpoint only on the
   tennis GPU service:

```bash
GOOD_TENNIS_STREAM_BALL_MODEL=/root/good-tennis-gpu-api-state/weights/tennis-ball-yolo.pt
```

2. **Current-checkpoint experiment**: build the tennis package with the
   existing `yolo11s-ball.pt` included:

```powershell
.\deploy\package_gpu_api.ps1 -Sport tennis -IncludeExperimentalTennisBallModel
```

It creates `deploy/good-tennis-gpu-api-upload.zip` and places the current Pose
and experimental ball checkpoints into persistent `weights/` during refresh. The fixed tennis launcher
already selects the sport; `GOOD_SPORT_VISION_PROFILE` is optional and must
only be absent or equal to `tennis`.

The dedicated checkpoint's `names` metadata must contain the exact label
`tennis_ball`; detections use `ball_kind=tennis_ball`. The optional experiment
accepts only the existing `badminton` label and emits
`ball_kind=experimental_badminton_ball_candidate` plus `experimental=true`.
It is not a tennis accuracy claim. The WebUI reports detected/missing counts
and detection rate; precision and recall require manually labelled tennis-ball
positions from the same video.

The GPU emits `ball_observation` as raw image-space evidence only. It does not
derive shots, hits, rallies, scores, player ownership, or a filled-in ball
trajectory. If the ball is missed, the event is recorded as `missing`.

Before production deployment, validate the chosen weights on held-out fixed
camera singles and near-side training clips, recording precision/recall by
distance, lighting and motion blur. The initial confidence/gating defaults are
deployment starting points, not a quality guarantee.
