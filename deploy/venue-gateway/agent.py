"""Outbound-only camera relay for one venue court.

The relay reads its RTSP URL locally, produces independently playable MP4
segments, signs every request to the business server and never contacts GPU.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import requests

from business_gateway.edge_contract import (
    EDGE_NONCE_HEADER, EDGE_PAYLOAD_SHA256_HEADER, EDGE_SCHEMA_VERSION,
    EDGE_SIGNATURE_HEADER, EDGE_TIMESTAMP_HEADER, payload_sha256, sign_request,
)


def load_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nonce() -> str:
    return base64.urlsafe_b64encode(os.urandom(24)).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class Settings:
    base_url: str
    device_id: str
    camera_id: str
    credential_version: str
    device_secret: str
    rtsp_url: str
    spool_dir: Path
    segment_seconds: int
    heartbeat_seconds: int
    corners: list[list[float]]

    @classmethod
    def from_environment(cls) -> "Settings":
        env_path = Path(os.environ.get("GOOD_BADMINTON_VENUE_ENV_FILE", "/etc/good-badminton-venue-gateway.env"))
        if env_path.is_file():
            load_env_file(env_path)
        required = ["EDGE_GATEWAY_URL", "EDGE_DEVICE_ID", "EDGE_CAMERA_ID", "EDGE_DEVICE_SECRET", "CAMERA_RTSP_URL"]
        missing = [key for key in required if not os.environ.get(key, "").strip()]
        if missing:
            raise ValueError("Missing venue relay settings: " + ", ".join(missing))
        corners = json.loads(os.environ.get("COURT_CORNERS_JSON", ""))
        if not isinstance(corners, list) or len(corners) != 4:
            raise ValueError("COURT_CORNERS_JSON must contain four [x,y] points")
        return cls(
            base_url=os.environ["EDGE_GATEWAY_URL"].rstrip("/"), device_id=os.environ["EDGE_DEVICE_ID"],
            camera_id=os.environ["EDGE_CAMERA_ID"], credential_version=os.environ.get("EDGE_CREDENTIAL_VERSION", "v1"),
            device_secret=os.environ["EDGE_DEVICE_SECRET"], rtsp_url=os.environ["CAMERA_RTSP_URL"],
            spool_dir=Path(os.environ.get("SPOOL_DIR", "/var/lib/good-badminton-venue-gateway")),
            segment_seconds=max(1, min(10, int(os.environ.get("SEGMENT_SECONDS", "2")))),
            heartbeat_seconds=max(5, int(os.environ.get("HEARTBEAT_SECONDS", "10"))), corners=corners,
        )


class EdgeClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()

    def _headers(self, method: str, path: str, body: dict[str, Any], segment: bytes | None = None) -> dict[str, str]:
        digest = payload_sha256(body, segment)
        timestamp = body["timestamp"]
        token = body["nonce"]
        return {
            EDGE_TIMESTAMP_HEADER: timestamp, EDGE_NONCE_HEADER: token, EDGE_PAYLOAD_SHA256_HEADER: digest,
            EDGE_SIGNATURE_HEADER: sign_request(self.settings.device_secret, method, path, timestamp, token, digest),
        }

    def _envelope(self, **values: Any) -> dict[str, Any]:
        return {"schema_version": EDGE_SCHEMA_VERSION, "device_id": self.settings.device_id,
                "camera_id": self.settings.camera_id, "timestamp": now(), "nonce": nonce(), **values}

    def heartbeat(self, capture_state: str, active_session_id: str | None, last_index: int | None) -> dict[str, Any]:
        path = f"/api/v1/edge/devices/{self.settings.device_id}/heartbeats"
        body = self._envelope(agent_version="venue-gateway/0.1.0", disk_free_bytes=shutil.disk_usage(self.settings.spool_dir).free,
                              capture_state=capture_state, active_session_id=active_session_id, last_segment_index=last_index)
        response = self.session.post(self.settings.base_url + path, json=body, headers=self._headers("POST", path, body), timeout=15)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("business gateway returned an invalid heartbeat response")
        return payload

    def start_case(self) -> str:
        path = f"/api/v1/edge/devices/{self.settings.device_id}/sessions"
        body = self._envelope(configuration={"analysis_sample_hz": 10, "pose_imgsz": 960,
                              "shuttle_detector": "yolo", "generate_annotated_video": False})
        response = self.session.post(self.settings.base_url + path, json=body, headers=self._headers("POST", path, body), timeout=20)
        response.raise_for_status()
        return str(response.json()["edge_ingest_session_id"])

    def upload(self, case_id: str, index: int, video_path: Path) -> None:
        video = video_path.read_bytes()
        digest = hashlib.sha256(video).hexdigest()
        segment = {"schema_version": "stream-session.v1", "segment_index": index,
                   "source_start_time_sec": index * self.settings.segment_seconds, "duration_sec": self.settings.segment_seconds,
                   "sha256": digest, "idempotency_key": f"seg_{hashlib.sha256((case_id + ':' + str(index)).encode()).hexdigest()[:32]}",
                   "content_type": "video/mp4", "content_length_bytes": len(video), "court_corners": self.settings.corners}
        body = self._envelope(segment=segment)
        path = f"/api/v1/edge/sessions/{case_id}/segments/{index}"
        response = self.session.post(self.settings.base_url + path, data={"metadata": json.dumps(body, separators=(",", ":"))},
            files={"segment": (video_path.name, video, "video/mp4")}, headers=self._headers("POST", path, body, video), timeout=180)
        response.raise_for_status()

    def complete(self, case_id: str, last_index: int) -> None:
        """Close the signed ingest session so a relay restart is recoverable."""
        path = f"/api/v1/edge/sessions/{case_id}/complete"
        body = self._envelope(expected_last_segment_index=last_index, allow_partial=True)
        response = self.session.post(self.settings.base_url + path, json=body,
                                     headers=self._headers("POST", path, body), timeout=30)
        response.raise_for_status()


def ffmpeg_command(settings: Settings, output_pattern: Path) -> list[str]:
    # Re-encode ensures each segment is independently browser-playable. The
    # source camera must still be configured for a stable fixed full-court view.
    fps = 25
    gop = fps * settings.segment_seconds
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp", "-i", settings.rtsp_url,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0", "-force_key_frames", f"expr:gte(t,n_forced*{settings.segment_seconds})", "-f", "segment",
            "-segment_time", str(settings.segment_seconds), "-reset_timestamps", "1", "-movflags", "+faststart", str(output_pattern)]


def main() -> None:
    settings = Settings.from_environment()
    settings.spool_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = settings.spool_dir / "segments"
    segment_dir.mkdir(exist_ok=True)
    client = EdgeClient(settings)
    stopped = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    case_id: str | None = None
    last_index: int | None = None
    process: subprocess.Popen[bytes] | None = None
    control_lock = Lock()
    control: dict[str, Any] = {"mode": "idle", "revision": 0}

    def desired_mode() -> str:
        with control_lock:
            return str(control.get("mode") or "idle")

    def stop_capture() -> None:
        nonlocal case_id, last_index, process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            process = None
        if case_id is not None and last_index is not None:
            try:
                client.complete(case_id, last_index)
            except requests.RequestException as error:
                # The business command is still authoritative.  A later
                # operator decision can inspect/reconcile the incomplete case.
                print(f"session completion failed: {error}", flush=True)
        # Do not accidentally attach an unfinished segment from a stopped
        # command to the next, separately auditable business case.
        for pending in segment_dir.glob("*.mp4"):
            pending.unlink(missing_ok=True)
        case_id = None
        last_index = None

    def heartbeats() -> None:
        while not stopped.wait(settings.heartbeat_seconds):
            try:
                response = client.heartbeat("capturing" if process else "idle", case_id, last_index)
                received = response.get("capture_control")
                mode = received.get("mode") if isinstance(received, dict) else None
                if mode not in {"idle", "preview", "record"}:
                    raise ValueError("business gateway returned an invalid capture mode")
                with control_lock:
                    control.update(received)
            except requests.RequestException as error:
                print(f"heartbeat failed: {error}", flush=True)
                # Fail closed: do not continue collecting video when the
                # business server cannot confirm the desired capture state.
                with control_lock:
                    control["mode"] = "idle"
            except ValueError as error:
                print(f"heartbeat control failed: {error}", flush=True)
                with control_lock:
                    control["mode"] = "idle"

    initial = client.heartbeat("idle", None, None)
    initial_control = initial.get("capture_control")
    if not isinstance(initial_control, dict) or initial_control.get("mode") not in {"idle", "preview", "record"}:
        raise ValueError("business gateway did not return a valid capture control")
    with control_lock:
        control.update(initial_control)
    Thread(target=heartbeats, daemon=True).start()
    try:
        while not stopped.wait(0.5):
            if desired_mode() == "idle":
                if process is not None or case_id is not None:
                    stop_capture()
                continue
            if process is None:
                # The case is created only after an explicit business command.
                # A GPU forwarding decision remains separate and defaults off.
                for pending in segment_dir.glob("*.mp4"):
                    pending.unlink(missing_ok=True)
                case_id = client.start_case()
                last_index = None
                process = subprocess.Popen(ffmpeg_command(settings, segment_dir / "%08d.mp4"))
            if process.poll() is not None:
                raise RuntimeError(f"ffmpeg exited with {process.returncode}")
            for path in sorted(segment_dir.glob("*.mp4")):
                # A recently modified file may still be written by ffmpeg.
                if time.time() - path.stat().st_mtime < 1.0:
                    continue
                index = int(path.stem)
                client.upload(case_id, index, path)
                path.unlink()
                last_index = index
    finally:
        stopped.set()
        stop_capture()


if __name__ == "__main__":
    main()
