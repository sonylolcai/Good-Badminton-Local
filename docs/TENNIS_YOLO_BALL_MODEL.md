# Tennis YOLO Ball Model Contract

The tennis GPU image must contain a dedicated YOLO checkpoint.  Configure it
only on the tennis GPU service:

```bash
GOOD_TENNIS_STREAM_BALL_MODEL=/opt/good-tennis/weights/tennis-ball-yolo.pt
```

Build the separately deployable archive on Windows with:

```powershell
.\deploy\package_gpu_api.ps1 -Sport tennis
```

It creates `deploy/good-tennis-gpu-api-upload.zip`. Upload that file to
`/root/good-tennis-gpu-api-upload.zip`; before refreshing, configure the same
checkpoint path and `GOOD_SPORT_VISION_PROFILE=tennis` in the tennis server's
`/root/good-tennis-gpu-api-state/.gpu-api.env`.

The checkpoint's `names` metadata must contain the exact label
`tennis_ball`. The runtime rejects a badminton-only checkpoint, a generic
model without labels, and detections from all non-`tennis_ball` classes.
`deploy/start_tennis_gpu_container.sh` also refuses to start until the configured
checkpoint path exists, so a tennis server cannot silently run with only the
bundled badminton weights.

The GPU emits `ball_observation` as raw image-space evidence only. It does not
derive shots, hits, rallies, scores, player ownership, or a filled-in ball
trajectory. If the ball is missed, the event is recorded as `missing`.

Before production deployment, validate the chosen weights on held-out fixed
camera singles and near-side training clips, recording precision/recall by
distance, lighting and motion blur. The initial confidence/gating defaults are
deployment starting points, not a quality guarantee.
