# Tennis YOLO Ball Model Contract

The shared GPU service can run tennis sessions in pose-only mode. In WebUI,
select **only person pose** when no ball model is required. Selecting YOLO ball
detection uses one of the following explicitly separated modes:

1. **Dedicated tennis mode**: configure a dedicated checkpoint on the shared
   GPU server:

```bash
GOOD_TENNIS_STREAM_BALL_MODEL=/root/good-badminton-gpu-api-state/weights/tennis-ball.pt
```

2. **Current-checkpoint experiment**: place the existing `yolo11s-ball.pt`
   in the persistent shared weights directory and configure:

```bash
GOOD_TENNIS_EXPERIMENTAL_BALL_MODEL=/root/good-badminton-gpu-api-state/weights/yolo11s-ball.pt
```

For a source-only update, build with `powershell -File deploy/package_gpu_api.ps1`;
it contains no checkpoints and preserves the shared persistent `weights/`
directory. For a complete code-and-model release, use
`powershell -File deploy/package_gpu_api.ps1 -IncludeWeights`; it contains the
dedicated `tennis-ball.pt` together with the two badminton runtime checkpoints
and a SHA-256 manifest. Submit these sessions with `sport_id=tennis`.

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
