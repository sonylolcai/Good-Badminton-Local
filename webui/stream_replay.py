"""WebUI client for the local business-to-GPU replay demonstration.

Unlike :mod:`webui.remote_gpu`, this client does not submit a complete video
to the GPU API.  It submits the file to the local business gateway, which
creates short MP4 fragments and delivers them through ``stream-session.v1``.
The module is intentionally development-only; a real deployment receives an
already-growing camera stream at the business ingest service.
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from webui.remote_gpu import _load_local_config_file
from runtime_config import business_api_base_url, load_runtime_environment


class StreamReplayError(RuntimeError):
    """The local business replay gateway did not accept or complete a task."""


def business_stream_config() -> dict[str, Any]:
    load_runtime_environment()
    _load_local_config_file()
    return {
        "base_url": business_api_base_url(),
        "timeout_seconds": float(os.environ.get("GOOD_BADMINTON_BUSINESS_STREAM_TIMEOUT", "30")),
        "poll_seconds": float(os.environ.get("GOOD_BADMINTON_BUSINESS_STREAM_POLL_SECONDS", "1")),
    }


def start_local_stream_replay(
    video_path: str,
    corners: list[list[float]] | list[tuple[float, float]],
    *,
    analysis_sample_hz: int,
    pose_imgsz: int,
    shuttle_detector: str,
    tracker_backend: str,
    generate_annotated_video: bool,
    far_player_enhancement: bool = False,
    realtime: bool = False,
    gpu_base_url: str | None = None,
) -> dict[str, Any]:
    """Send an anonymous local replay with its business-owned court corners."""

    if not video_path or not Path(video_path).is_file():
        raise StreamReplayError("请先上传可读取的视频文件")
    if not corners or len(corners) != 4:
        raise StreamReplayError("请先确认四个球场角点")
    config = business_stream_config()
    # Development treats every upload as a camera profile. Production instead
    # looks this ID and corners up from its business-owned camera registry.
    normalized_corners = [[float(x), float(y)] for x, y in corners]
    calibration_id = f"local-development-upload-{uuid.uuid4().hex}"
    payload = _multipart_submit(
        config,
        "/api/v1/development/stream-replays",
        Path(video_path),
        {
            "calibration_id": calibration_id,
            "court_corners": json.dumps(normalized_corners, ensure_ascii=False),
            "camera_id": "local-development-camera",
            # The development business gateway persists this destination with
            # the replay task, so later polling cannot drift back to an
            # environment-configured GPU address.
            "gpu_base_url": str(gpu_base_url or "").strip(),
            "analysis_sample_hz": str(int(analysis_sample_hz)),
            "pose_imgsz": str(int(pose_imgsz)),
            "shuttle_detector": str(shuttle_detector),
            "tracker_backend": str(tracker_backend),
            # A confirmed player roster is fixed for this match.  It is
            # inferred from stable on-court detections, so the UI never asks
            # whether this is singles, doubles, or an informal game.
            "lock_match_roster": "true",
            "roster_stable_frames": "3",
            "max_roster_count": "4",
            "roster_discovery_seconds": "8.0",
            # The GPU derives a tight far-half ROI from these same court
            # corners only when the operator explicitly enables it.
            "far_player_enhancement": "true" if far_player_enhancement else "false",
            # The stream contract has no annotated-video encoder. The legacy
            # checkbox is ignored here rather than producing a late GPU error.
            "generate_annotated_video": "false",
            "segment_seconds": "2.0",
            # File replay runs as fast as local segment production allows. It
            # demonstrates the segment contract, not a wall-clock camera SLO.
            "realtime": "false" if not realtime else "true",
        },
    )
    payload["calibration_id"] = calibration_id
    payload["mode"] = "local_business_to_gpu_segment_replay"
    return payload


def poll_local_stream_replay(task_id: str) -> dict[str, Any]:
    config = business_stream_config()
    payload = _json_request(config, "GET", f"/api/v1/development/stream-replays/{task_id}")
    payload["mode"] = "local_business_to_gpu_segment_replay"
    return payload


def iter_local_stream_replay(initial: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield durable business/GPU state until the local replay task ends."""

    task_id = str(initial["business_task_id"])
    yield initial
    config = business_stream_config()
    terminal = {"completed", "failed"}
    while True:
        time.sleep(max(0.25, config["poll_seconds"]))
        status = poll_local_stream_replay(task_id)
        yield status
        if status.get("status") in terminal:
            return


def _json_request(config: dict[str, Any], method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = Request(
        config["base_url"] + path,
        method=method,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers={"Accept": "application/json", "Content-Type": "application/json"} if body is not None else {"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=config["timeout_seconds"]) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise StreamReplayError(f"本地业务流服务不可用：{exc}") from exc


def _multipart_submit(config: dict[str, Any], path: str, video_path: Path, fields: dict[str, str]) -> dict[str, Any]:
    parsed = urlparse(config["base_url"])
    boundary = f"----GoodBadmintonLocalStream{uuid.uuid4().hex}"
    field_bytes = b"".join(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode("utf-8")
        for name, value in fields.items()
    )
    content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    file_prefix = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"video\"; filename=\"{video_path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=config["timeout_seconds"])
    endpoint = parsed.path.rstrip("/") + path
    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Accept", "application/json")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(len(field_bytes) + len(file_prefix) + video_path.stat().st_size + len(suffix)))
        connection.endheaders()
        connection.send(field_bytes)
        connection.send(file_prefix)
        with video_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read()
        if not 200 <= response.status < 300:
            raise StreamReplayError(f"本地业务流服务返回 HTTP {response.status}: {body.decode('utf-8', 'replace')}")
        return json.loads(body.decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        raise StreamReplayError(f"无法提交本地流回放：{exc}") from exc
    finally:
        connection.close()
