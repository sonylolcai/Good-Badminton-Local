"""Replay one test match through the GPU stream API with business-side linkage.

This is intentionally a business-side test adapter.  The local manifest keeps
the venue, court and match identifiers, while the GPU receives only its opaque
``client_reference`` and returns an ``analysis_session_id``.  Every fragment in
one invocation belongs to that one GPU analysis session and is uploaded by
``replay_video`` in increasing ``segment_index`` order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .client import StreamSessionClient
from .models import DeliveryLedger, StreamClientConfig
from .replay import replay_video
from .segmenter import GrowingVideoSegmenter


MANIFEST_SCHEMA_VERSION = "business-stream-continuity-test.v1"


def build_business_session_manifest(
    *,
    venue_id: str,
    court_id: str,
    match_id: str,
    camera_id: str,
    calibration_id: str,
    video_path: Path,
    video_sha256: str,
    opaque_client_reference: str,
    create_idempotency_key: str,
    segment_seconds: float,
) -> dict[str, Any]:
    """Build the local-only mapping needed to audit one replay.

    The ``business_identity`` object must never be copied into the GPU request.
    The returned manifest makes this separation inspectable after a test run.
    """

    identifiers = {
        "venue_id": venue_id,
        "court_id": court_id,
        "match_id": match_id,
        "camera_id": camera_id,
        "calibration_id": calibration_id,
    }
    for name, value in identifiers.items():
        if not str(value).strip():
            raise ValueError(f"{name} is required")
    if len(video_sha256) != 64:
        raise ValueError("video_sha256 must be a SHA-256 hex digest")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "business_identity": {
            "venue_id": str(venue_id),
            "court_id": str(court_id),
            "match_id": str(match_id),
        },
        "camera_binding": {
            "camera_id": str(camera_id),
            "calibration_id": str(calibration_id),
        },
        "input": {
            "video_path": str(Path(video_path).resolve()),
            "video_sha256": video_sha256,
        },
        "transport": {
            "business_session_id": f"business_test_{uuid.uuid4().hex}",
            "gpu_client_reference": opaque_client_reference,
            "create_idempotency_key": create_idempotency_key,
            "gpu_analysis_session_id": None,
        },
        "stream_order_contract": {
            "scope": "one_gpu_analysis_session_id",
            "segment_index_start": 0,
            "requires_contiguous_indexes": True,
            "source_time_basis": "source_match_timeline_seconds",
            "segment_seconds": float(segment_seconds),
        },
        "result": None,
    }


def build_gpu_create_request(
    *,
    camera_id: str,
    calibration_id: str,
    court_corners: list[list[float]],
    opaque_client_reference: str,
    analysis_sample_hz: int,
    pose_imgsz: int,
    shuttle_detector: str,
    tracker_backend: str,
) -> dict[str, Any]:
    """Build the anonymous request accepted by ``stream-session.v1``."""

    request = {
        "schema_version": "stream-session.v1",
        "camera_id": camera_id,
        "calibration_id": calibration_id,
        "court_corners": court_corners,
        "analysis_mode": "person_only",
        "client_reference": opaque_client_reference,
        "configuration": {
            "analysis_sample_hz": analysis_sample_hz,
            "pose_imgsz": pose_imgsz,
            "shuttle_detector": shuttle_detector,
            "generate_annotated_video": False,
            "tracker_backend": tracker_backend,
            "lock_match_roster": True,
            "roster_stable_frames": 3,
            "preserve_audio": False,
            "court_health_check_hz": 2,
        },
    }
    forbidden = {"venue_id", "court_id", "match_id"}
    if forbidden.intersection(request):
        raise AssertionError("business identifiers must not be sent to GPU")
    return request


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_court_corners(value: str) -> list[list[float]]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("court-corners must be JSON") from exc
    if not isinstance(raw, list) or len(raw) != 4:
        raise argparse.ArgumentTypeError("court-corners must contain exactly four [x, y] points")
    corners: list[list[float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise argparse.ArgumentTypeError("each court corner must be [x, y]")
        corners.append([float(point[0]), float(point[1])])
    return corners


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--venue-id", required=True)
    parser.add_argument("--court-id", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--court-corners", required=True, type=_parse_court_corners)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--encoding-mode", choices=("copy", "h264"), default="h264")
    parser.add_argument("--analysis-sample-hz", type=int, choices=(10, 15, 30), default=10)
    parser.add_argument("--pose-imgsz", type=int, choices=(640, 960, 1280), default=960)
    parser.add_argument("--shuttle-detector", choices=("none", "yolo", "tracknet_v3"), default="yolo")
    parser.add_argument("--tracker-backend", choices=("court_association", "bytetrack"), default="court_association")
    parser.add_argument("--ffmpeg-bin", default="")
    parser.add_argument("--ffprobe-bin", default="")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not 0.5 <= args.segment_seconds <= 10:
        parser.error("segment-seconds must be between 0.5 and 10")
    manifest_path = args.work_dir / "business_session_manifest.json"
    if manifest_path.exists():
        parser.error(f"work-dir already has a manifest: {manifest_path}; use a new work directory")

    opaque_client_reference = f"continuity_test_{uuid.uuid4().hex}"
    create_idempotency_key = f"continuity_{uuid.uuid4().hex}"
    request = build_gpu_create_request(
        camera_id=args.camera_id,
        calibration_id=args.calibration_id,
        court_corners=args.court_corners,
        opaque_client_reference=opaque_client_reference,
        analysis_sample_hz=args.analysis_sample_hz,
        pose_imgsz=args.pose_imgsz,
        shuttle_detector=args.shuttle_detector,
        tracker_backend=args.tracker_backend,
    )
    manifest = build_business_session_manifest(
        venue_id=args.venue_id,
        court_id=args.court_id,
        match_id=args.match_id,
        camera_id=args.camera_id,
        calibration_id=args.calibration_id,
        video_path=args.video,
        video_sha256=_sha256_file(args.video),
        opaque_client_reference=opaque_client_reference,
        create_idempotency_key=create_idempotency_key,
        segment_seconds=args.segment_seconds,
    )
    _write_json(manifest_path, manifest)

    client = StreamSessionClient(
        StreamClientConfig.from_environment(),
        DeliveryLedger(args.work_dir / "delivery-ledger.json"),
    )
    segmenter = GrowingVideoSegmenter(
        args.work_dir / "segments",
        segment_duration_sec=args.segment_seconds,
        encoding_mode=args.encoding_mode,
        preserve_audio=False,
        declare_frame_sequence=args.shuttle_detector == "tracknet_v3",
        ffmpeg_path=args.ffmpeg_bin or None,
        ffprobe_path=args.ffprobe_bin or None,
    )
    try:
        result = replay_video(
            args.video,
            client=client,
            segmenter=segmenter,
            create_request=request,
            create_idempotency_key=create_idempotency_key,
            realtime=args.realtime,
        )
    except Exception as exc:
        manifest["result"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(manifest_path, manifest)
        raise
    manifest["transport"]["gpu_analysis_session_id"] = result["analysis_session_id"]
    manifest["result"] = {
        "status": "submitted",
        "last_segment_index": result["last_segment_index"],
        "gpu_status_after_seal": result["status"].get("status"),
        "end_to_end_trace": result["end_to_end_trace"],
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), **manifest["transport"], "result": manifest["result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
