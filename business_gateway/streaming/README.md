# Business-side stream delivery

This package implements task C without importing the GPU API or computer-vision
modules. It owns independently decodable segment creation, durable delivery
receipts and retry/restart behavior for `stream-session.v1`.

## Configuration

- `GPU_ANALYSIS_BASE_URL`: configured GPU API base URL; required in new deployments.
- `GPU_ANALYSIS_API_KEY`: API credential; required in new deployments.
- `GOOD_BADMINTON_STREAM_API_URL` / `GOOD_BADMINTON_STREAM_API_KEY`: migration aliases.
- `GOOD_BADMINTON_GPU_API_URL` / `GOOD_BADMINTON_GPU_API_KEY`: legacy complete-file aliases.
- `GPU_ANALYSIS_TIMEOUT_SECONDS`: one HTTP attempt timeout, default 30.
- `GPU_ANALYSIS_MAX_ATTEMPTS`: bounded attempts per operation, default 5.
- `GPU_ANALYSIS_INITIAL_BACKOFF_SECONDS`: initial exponential delay, default 0.25.
- `GPU_ANALYSIS_MAX_BACKOFF_SECONDS`: maximum delay, default 5.
- The corresponding `GOOD_BADMINTON_STREAM_*` timeout/retry names remain migration aliases.
- `GOOD_BADMINTON_FFMPEG` / `GOOD_BADMINTON_FFPROBE`: optional executable overrides.

No public hostname or localhost default is embedded in the client.

## FFmpeg strategy

`GrowingVideoSegmenter.iter_input()` starts FFmpeg's segment muxer and publishes a
fragment only after the next fragment has appeared (or FFmpeg exits). Therefore
the newest file still being written is never uploaded.

- `encoding_mode=copy` is the default. It remuxes at existing keyframes without
  quality loss or GPU/CPU encoding cost. Camera/edge GOP should be 1–2 seconds.
- `encoding_mode=h264` is an explicit fallback for an unsuitable/irregular GOP.
  It inserts aligned keyframes with libx264 and records the command/mode in
  `segment_manifest.json`; this adds latency and is not silently enabled.
- A copy-mode fragment longer than the contract's 10-second maximum is rejected
  with an actionable error instead of publishing misleading source timing.

For a camera or edge gateway that already emits closed fMP4/HLS fragments, those
files may be registered directly with `SegmentMetadata.from_file()` and submitted
through `StreamSessionClient`; no second encoding step is required.

## Durable ledger

`delivery-ledger.json` atomically records:

- the create request and create idempotency key;
- the returned anonymous `analysis_session_id`;
- each segment path, index, source time, duration, SHA-256, byte length and stable
  idempotency key;
- attempts, pending/accepted/failed state, original GPU receipt and last error;
- the final completion request/response.

Only a validated durable GPU receipt moves a segment to `accepted`. If the server
accepted bytes but the response was lost, the same index/hash/key is retried and
the reused receipt closes the uncertainty. On process restart,
`deliver_pending()` skips accepted segments and retries every unconfirmed record.

## Local recording replay

With the GPU stream routes installed by task F:

```powershell
$env:GOOD_BADMINTON_STREAM_API_URL = "https://configured-gpu-service"
$env:GOOD_BADMINTON_STREAM_API_KEY = "configured-secret"

.\.venv\Scripts\python.exe -m business_gateway.streaming.replay_cli `
  --video S:\path\match.mp4 `
  --create-request docs\contracts\stream_session_v1\examples\create_session_request.json `
  --work-dir .\stream_replay_data\match-001 `
  --idempotency-key match-20260823-000001 `
  --segment-seconds 2
```

Add `--realtime` to pace a local recording like a live source. The replay still
uses the same create/upload/receipt/complete client path as production.

The replay writes two durable business-side files under `--work-dir`:

- `delivery-ledger.json`: every request, retry and accepted segment receipt;
- `end_to_end_trace.json`: segmentation/waiting, upload, sealing, first result
  return, plus the GPU-side trace fetched from the session trace endpoint.

Because the source is still a recording, this file always keeps
`streaming_slo_proven=false`. A real camera/edge-to-business-to-GPU run is the
only accepted proof of the live SLO.
