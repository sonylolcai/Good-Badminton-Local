"""Command-line replay of a local recording through the stream-session API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import StreamSessionClient
from .models import DeliveryLedger, StreamClientConfig
from .replay import replay_video
from .segmenter import GrowingVideoSegmenter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--create-request", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--encoding-mode", choices=("copy", "h264"), default="copy")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--overwrite-segments", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.create_request.read_text(encoding="utf-8"))
    config = StreamClientConfig.from_environment()
    client = StreamSessionClient(
        config,
        DeliveryLedger(args.work_dir / "delivery-ledger.json"),
    )
    segmenter = GrowingVideoSegmenter(
        args.work_dir / "segments",
        segment_duration_sec=args.segment_seconds,
        encoding_mode=args.encoding_mode,
        preserve_audio=bool(request.get("configuration", {}).get("preserve_audio", False)),
    )
    result = replay_video(
        args.video,
        client=client,
        segmenter=segmenter,
        create_request=request,
        create_idempotency_key=args.idempotency_key,
        realtime=args.realtime,
        overwrite_segments=args.overwrite_segments,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
