"""Outbound-only camera relay for one venue court.

The relay reads its RTSP URL locally, produces independently playable MP4
segments, signs every request to the business server and never contacts GPU.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
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
    # Windows PowerShell 5.1 writes a UTF-8 BOM by default.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nonce() -> str:
    return base64.urlsafe_b64encode(os.urandom(24)).decode("ascii").rstrip("=")


def load_active_session(path: Path) -> tuple[str | None, int | None]:
    """Recover the one transport session owned by this single-camera relay."""

    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        session_id = str(payload.get("edge_ingest_session_id") or "").strip()
        last_index = payload.get("last_segment_index")
        if not session_id or (
            last_index is not None
            and (isinstance(last_index, bool) or not isinstance(last_index, int) or last_index < 0)
        ):
            raise ValueError("invalid active session state")
        return session_id, last_index
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ignored invalid active session state: {error}", flush=True)
        return None, None


def save_active_session(path: Path, session_id: str, last_index: int | None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"edge_ingest_session_id": session_id, "last_segment_index": last_index},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _find_existing_session_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("edge_ingest_session_id", "active_session_id", "session_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _find_existing_session_id(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_existing_session_id(value)
            if found:
                return found
    return None


def sanitize_error_message(error: Exception | str) -> str:
    """Keep operator diagnostics useful without leaking RTSP credentials."""

    message = " ".join(str(error).split())
    message = re.sub(r"rtsp://[^@\s]+@", "rtsp://***@", message, flags=re.IGNORECASE)
    return message[:240] or "unspecified error"


class SessionConflict(RuntimeError):
    """The server still owns a session but did not identify it to the relay."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    device_id: str
    camera_id: str
    credential_version: str
    device_secret: str
    rtsp_url: str
    ffmpeg_bin: str
    spool_dir: Path
    segment_seconds: int
    heartbeat_seconds: int
    corners: list[list[float]]

    @classmethod
    def from_environment(cls) -> "Settings":
        if os.name == "nt":
            default_root = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "GoodBadminton"
            default_env_path = default_root / "venue-gateway.env"
            default_spool_dir = default_root / "venue-gateway" / "spool"
        else:
            default_env_path = Path("/etc/good-badminton-venue-gateway.env")
            default_spool_dir = Path("/var/lib/good-badminton-venue-gateway")
        env_path = Path(os.environ.get("GOOD_BADMINTON_VENUE_ENV_FILE", str(default_env_path)))
        if env_path.is_file():
            load_env_file(env_path)
        required = ["EDGE_GATEWAY_URL", "EDGE_DEVICE_ID", "EDGE_CAMERA_ID", "EDGE_DEVICE_SECRET", "CAMERA_RTSP_URL", "COURT_CORNERS_JSON"]
        missing = [key for key in required if not os.environ.get(key, "").strip()]
        if missing:
            raise ValueError("Missing venue relay settings: " + ", ".join(missing))
        corners = json.loads(os.environ.get("COURT_CORNERS_JSON", ""))
        if (
            not isinstance(corners, list)
            or len(corners) != 4
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
                for point in corners
            )
        ):
            raise ValueError("COURT_CORNERS_JSON must contain four [x,y] points")
        return cls(
            base_url=os.environ["EDGE_GATEWAY_URL"].rstrip("/"), device_id=os.environ["EDGE_DEVICE_ID"],
            camera_id=os.environ["EDGE_CAMERA_ID"], credential_version=os.environ.get("EDGE_CREDENTIAL_VERSION", "v1"),
            device_secret=os.environ["EDGE_DEVICE_SECRET"], rtsp_url=os.environ["CAMERA_RTSP_URL"],
            ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg",
            spool_dir=Path(os.environ.get("SPOOL_DIR", str(default_spool_dir))),
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

    def heartbeat(
        self,
        capture_state: str,
        active_session_id: str | None,
        last_index: int | None,
        error_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = f"/api/v1/edge/devices/{self.settings.device_id}/heartbeats"
        values: dict[str, Any] = {
            "agent_version": "venue-gateway/0.2.3-windows.6",
            "disk_free_bytes": shutil.disk_usage(self.settings.spool_dir).free,
            "capture_state": capture_state,
            "active_session_id": active_session_id,
            "last_segment_index": last_index,
        }
        if error_report is not None:
            values["error_report"] = error_report
        body = self._envelope(**values)
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
        if response.status_code == 409:
            try:
                conflict_payload = response.json()
            except ValueError:
                conflict_payload = None
            existing_session_id = _find_existing_session_id(conflict_payload)
            if existing_session_id:
                print(f"resuming server-reported active session {existing_session_id}", flush=True)
                return existing_session_id
            detail = " ".join(response.text.split())[:500] or "no response detail"
            raise SessionConflict(f"business server has an unidentified active session: {detail}")
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


def ffmpeg_command(settings: Settings, output_pattern: Path, *, start_number: int = 0) -> list[str]:
    # Re-encode ensures each segment is independently browser-playable. The
    # source camera must still be configured for a stable fixed full-court view.
    fps = 25
    gop = fps * settings.segment_seconds
    return [settings.ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp", "-i", settings.rtsp_url,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0", "-force_key_frames", f"expr:gte(t,n_forced*{settings.segment_seconds})", "-f", "segment",
            "-segment_time", str(settings.segment_seconds), "-segment_start_number", str(start_number),
            "-reset_timestamps", "1", "-movflags", "+faststart", str(output_pattern)]


def main() -> None:
    settings = Settings.from_environment()
    if shutil.which(settings.ffmpeg_bin) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {settings.ffmpeg_bin}")
    settings.spool_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = settings.spool_dir / "segments"
    segment_dir.mkdir(exist_ok=True)
    active_session_path = settings.spool_dir / "active-session.json"
    client = EdgeClient(settings)
    stopped = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    case_id, last_index = load_active_session(active_session_path)
    process: subprocess.Popen[bytes] | None = None
    next_start_attempt_at = 0.0
    next_complete_attempt_at = 0.0
    capture_faulted = False
    supports_error_reports = False
    error_lock = Lock()
    last_error: dict[str, Any] | None = None
    control_lock = Lock()
    control: dict[str, Any] = {"mode": "idle", "revision": 0}

    def desired_mode() -> str:
        with control_lock:
            return str(control.get("mode") or "idle")

    def record_error(code: str, error: Exception | str) -> None:
        nonlocal last_error
        message = sanitize_error_message(error)
        with error_lock:
            if last_error and last_error["code"] == code and last_error["message"] == message:
                last_error["count"] += 1
                last_error["occurred_at"] = now()
            else:
                last_error = {"code": code, "message": message, "occurred_at": now(), "count": 1}

    def current_error_report() -> dict[str, Any] | None:
        with error_lock:
            return dict(last_error) if last_error else None

    def learn_server_capabilities(response: dict[str, Any]) -> None:
        nonlocal supports_error_reports
        capabilities = response.get("capabilities")
        if isinstance(capabilities, dict) and capabilities.get("edge_error_reports") is True:
            supports_error_reports = True

    def stop_process() -> None:
        nonlocal process
        if process is None:
            return
        target = process
        process = None
        try:
            if target.poll() is None:
                target.terminate()
                try:
                    target.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait(timeout=5)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"ffmpeg stop failed: {error}", flush=True)

    def defer_capture(error: Exception, *, code: str) -> None:
        """Keep the control plane alive while the video data plane recovers."""

        nonlocal capture_faulted, next_start_attempt_at
        print(f"capture deferred: {type(error).__name__}: {error}", flush=True)
        record_error(code, error)
        stop_process()
        # Live preview favors bounded disk and a clean keyframe boundary over
        # replaying stale fragments after a transport outage.
        for pending in segment_dir.glob("*.mp4"):
            pending.unlink(missing_ok=True)
        capture_faulted = True
        next_start_attempt_at = time.monotonic() + settings.heartbeat_seconds

    def stop_capture() -> bool:
        nonlocal case_id, last_index, capture_faulted
        stop_process()
        completion_succeeded = True
        if case_id is not None:
            try:
                # An empty session can otherwise remain active forever if
                # FFmpeg fails before producing segment zero.
                client.complete(case_id, last_index if last_index is not None else 0)
            except requests.RequestException as error:
                # The business command is still authoritative.  A later
                # operator decision can inspect/reconcile the incomplete case.
                print(f"session completion failed: {error}", flush=True)
                completion_succeeded = False
        # Do not accidentally attach an unfinished segment from a stopped
        # command to the next, separately auditable business case.
        for pending in segment_dir.glob("*.mp4"):
            pending.unlink(missing_ok=True)
        if completion_succeeded:
            active_session_path.unlink(missing_ok=True)
            case_id = None
            last_index = None
            capture_faulted = False
        return completion_succeeded

    def heartbeats() -> None:
        while not stopped.wait(settings.heartbeat_seconds):
            try:
                capture_state = "capturing" if process else ("error" if capture_faulted else "idle")
                response = client.heartbeat(
                    capture_state,
                    case_id,
                    last_index,
                    current_error_report() if supports_error_reports else None,
                )
                learn_server_capabilities(response)
                received = response.get("capture_control")
                mode = received.get("mode") if isinstance(received, dict) else None
                if mode not in {"idle", "preview", "record"}:
                    raise ValueError("business gateway returned an invalid capture mode")
                with control_lock:
                    control.update(received)
            except requests.RequestException as error:
                print(f"heartbeat failed: {error}", flush=True)
                record_error("business_unreachable", error)
                # Fail closed: do not continue collecting video when the
                # business server cannot confirm the desired capture state.
                with control_lock:
                    control["mode"] = "idle"
            except ValueError as error:
                print(f"heartbeat control failed: {error}", flush=True)
                record_error("heartbeat_rejected", error)
                with control_lock:
                    control["mode"] = "idle"
            except Exception as error:
                # A malformed local state or unexpected response must not kill
                # the only control channel to the venue machine.
                print(f"heartbeat unexpected failure: {type(error).__name__}: {error}", flush=True)
                record_error("heartbeat_unexpected_failure", error)
                with control_lock:
                    control["mode"] = "idle"

    while not stopped.is_set():
        try:
            initial = client.heartbeat("idle", case_id, last_index)
            learn_server_capabilities(initial)
            initial_control = initial.get("capture_control")
            if not isinstance(initial_control, dict) or initial_control.get("mode") not in {"idle", "preview", "record"}:
                raise ValueError("business gateway did not return a valid capture control")
            break
        except (requests.RequestException, ValueError) as error:
            print(f"initial heartbeat deferred: {error}", flush=True)
            record_error("initial_heartbeat_failed", error)
            if stopped.wait(settings.heartbeat_seconds):
                return
    with control_lock:
        control.update(initial_control)
    Thread(target=heartbeats, daemon=True).start()
    try:
        while not stopped.wait(0.5):
            if desired_mode() == "idle":
                if process is not None or (
                    case_id is not None and time.monotonic() >= next_complete_attempt_at
                ):
                    if not stop_capture():
                        next_complete_attempt_at = time.monotonic() + settings.heartbeat_seconds
                continue
            if process is None:
                if time.monotonic() < next_start_attempt_at:
                    continue
                # The case is created only after an explicit business command.
                # A GPU forwarding decision remains separate and defaults off.
                for pending in segment_dir.glob("*.mp4"):
                    pending.unlink(missing_ok=True)
                if case_id is None:
                    try:
                        case_id = client.start_case()
                        last_index = None
                        save_active_session(active_session_path, case_id, last_index)
                        print(f"session accepted: {case_id}", flush=True)
                    except (requests.RequestException, SessionConflict) as error:
                        print(f"session start deferred: {error}", flush=True)
                        record_error(
                            "session_rejected" if isinstance(error, SessionConflict) else "session_start_failed",
                            error,
                        )
                        capture_faulted = True
                        next_start_attempt_at = time.monotonic() + settings.heartbeat_seconds
                        continue
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                try:
                    process = subprocess.Popen(
                        ffmpeg_command(
                            settings,
                            segment_dir / "%08d.mp4",
                            start_number=(last_index + 1) if last_index is not None else 0,
                        ),
                        creationflags=creationflags,
                    )
                    capture_faulted = False
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    defer_capture(error, code="ffmpeg_start_failed")
                    continue
            if process.poll() is not None:
                return_code = process.returncode
                process = None
                defer_capture(RuntimeError(f"ffmpeg exited with {return_code}"), code="ffmpeg_exited")
                continue
            for path in sorted(segment_dir.glob("*.mp4")):
                try:
                    # A recently modified file may still be written by ffmpeg.
                    if time.time() - path.stat().st_mtime < 1.0:
                        continue
                    index = int(path.stem)
                    client.upload(case_id, index, path)
                    print(f"segment accepted: session={case_id} index={index} bytes={path.stat().st_size}", flush=True)
                    path.unlink()
                    last_index = index
                    save_active_session(active_session_path, case_id, last_index)
                except (requests.RequestException, OSError, ValueError) as error:
                    defer_capture(error, code="segment_upload_failed")
                    break
    finally:
        stopped.set()
        stop_capture()


if __name__ == "__main__":
    main()
