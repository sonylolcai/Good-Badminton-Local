"""Local-only business gateway for replaying a recording as stream segments.

This is deliberately a development adapter, not the production business
service.  A browser uploads one completed recording to this gateway; the
gateway then uses the same ``stream-session.v1`` client used by a future
camera-ingest service to create 2-second, independently decodable segments and
deliver them to the GPU API.  The GPU API never receives user identities,
names, teams, or check IDs.

The business gateway owns calibration values.  It forwards the current
camera's four corners with every stream session and every segment; the GPU API
does not maintain a camera-calibration registry.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from runtime_config import business_service_listener, load_runtime_environment

from .court_match import CourtMatchManager
from .fixed_video_catalog import get_fixed_video, list_public_fixed_videos
from .metrics.movement import generate_movement_metrics, write_body_profiles
from .streaming.client import StreamSessionClient
from .streaming.models import DeliveryLedger, StreamClientConfig
from .streaming.replay import replay_video
from .streaming.segmenter import GrowingVideoSegmenter


MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_TERMINAL = {"finalized", "partial", "failed", "cancelled", "interrupted_needs_rebuild"}
_LOCAL_GPU_ENV_KEYS = {
    "GPU_ANALYSIS_BASE_URL",
    "GOOD_BADMINTON_STREAM_API_URL",
    "GOOD_BADMINTON_GPU_API_URL",
    "GOOD_BADMINTON_GPU_API_KEY",
    "GOOD_BADMINTON_GPU_API_TIMEOUT",
    "GOOD_BADMINTON_GPU_API_POLL_SECONDS",
    "GOOD_BADMINTON_GPU_JOB_TIMEOUT",
}


def _load_local_gpu_env() -> None:
    """Load the uncommitted local GPU settings for the development gateway.

    Production injects ``GPU_ANALYSIS_*`` through its service runtime.  The
    local WebUI already reads ``.webui-remote-gpu.env``; doing the same here
    prevents the separately started 8081 gateway from silently losing its
    GPU URL/API key after a restart.  Existing environment variables always
    win and secrets are never logged.
    """

    config_path = Path(__file__).resolve().parents[1] / ".webui-remote-gpu.env"
    if not config_path.is_file():
        return
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in _LOCAL_GPU_ENV_KEYS:
            os.environ.setdefault(key, value.strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class LocalReplayManager:
    """Durable local task manager around the already-tested replay client."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        config = StreamClientConfig.from_environment()
        return {
            "status": "ok",
            "service": "good-badminton-business-dev-gateway",
            "gpu_base_url": config.base_url,
            "calibration_transport": "business_owned_per_session_and_segment",
        }

    def submit(
        self,
        upload: UploadFile | None,
        *,
        calibration_id: str,
        court_corners: list[list[float]],
        camera_id: str,
        analysis_sample_hz: int,
        pose_imgsz: int,
        shuttle_detector: str,
        generate_annotated_video: bool,
        tracker_backend: str,
        lock_match_roster: bool,
        roster_stable_frames: int,
        max_roster_count: int,
        expected_player_count: int | None,
        roster_discovery_seconds: float,
        far_player_enhancement: bool,
        far_pose_roi: list[float] | None,
        segment_seconds: float,
        realtime: bool,
        gpu_base_url: str | None = None,
        source_video_path: Path | None = None,
        fixed_video_id: str | None = None,
    ) -> dict[str, Any]:
        task_id = f"bstr_{uuid.uuid4().hex}"
        task_dir = self.root / task_id
        if source_video_path is not None:
            video_path = Path(source_video_path).resolve()
            if not video_path.is_file():
                raise ValueError("configured fixed video source does not exist")
            written = video_path.stat().st_size
            input_filename = video_path.name
            input_kind = "fixed_catalog"
        else:
            if upload is None:
                raise ValueError("video upload is required when no fixed source is selected")
            input_dir = task_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
            video_path = input_dir / f"video{suffix}"
            written = 0
            with video_path.open("wb") as destination:
                while chunk := upload.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        destination.close()
                        video_path.unlink(missing_ok=True)
                        raise ValueError("video exceeds local development upload limit")
                    destination.write(chunk)
            input_filename = upload.filename or video_path.name
            input_kind = "uploaded_file"
        if written <= 0:
            raise ValueError("video is empty")
        if written > MAX_UPLOAD_BYTES:
            raise ValueError("video exceeds local development upload limit")
        if not 0.5 <= float(segment_seconds) <= 10.0:
            raise ValueError("segment_seconds must be between 0.5 and 10")
        if analysis_sample_hz not in {10, 15, 30}:
            raise ValueError("analysis_sample_hz must be 10, 15, or 30")
        if pose_imgsz not in {640, 960, 1280}:
            raise ValueError("pose_imgsz must be 640, 960, or 1280")
        if shuttle_detector not in {"none", "yolo", "tracknet_v3"}:
            raise ValueError("unsupported shuttle_detector")
        if tracker_backend not in {"court_association", "bytetrack"}:
            raise ValueError("tracker_backend must be court_association or bytetrack")
        if not 1 <= int(roster_stable_frames) <= 10:
            raise ValueError("roster_stable_frames must be between 1 and 10")
        if not 2 <= int(max_roster_count) <= 4:
            raise ValueError("max_roster_count must be between 2 and 4")
        if expected_player_count is not None and int(expected_player_count) not in {2, 4}:
            raise ValueError("expected_player_count must be 2 or 4 when provided")
        if expected_player_count is not None and int(max_roster_count) != int(expected_player_count):
            raise ValueError("max_roster_count must equal the registered expected_player_count")
        if not 1.0 <= float(roster_discovery_seconds) <= 15.0:
            raise ValueError("roster_discovery_seconds must be between 1 and 15")
        if far_pose_roi is not None and (
            len(far_pose_roi) != 4
            or any(not 0.0 <= float(value) <= 1.0 for value in far_pose_roi)
            or float(far_pose_roi[0]) >= float(far_pose_roi[2])
            or float(far_pose_roi[1]) >= float(far_pose_roi[3])
        ):
            raise ValueError("far_pose_roi must be an ordered normalized rectangle")
        if not isinstance(court_corners, list) or len(court_corners) != 4:
            raise ValueError("court_corners must contain exactly four [x, y] points")
        normalized_corners = []
        for point in court_corners:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each court corner must be [x, y]")
            normalized_corners.append([float(point[0]), float(point[1])])

        request = {
            "schema_version": "stream-session.v1",
            "camera_id": str(camera_id),
            "calibration_id": str(calibration_id),
            "court_corners": normalized_corners,
            "analysis_mode": "person_only",
            "client_reference": task_id,
            "configuration": {
                "analysis_sample_hz": int(analysis_sample_hz),
                "pose_imgsz": int(pose_imgsz),
                "shuttle_detector": shuttle_detector,
                # Stream sessions preserve measurement events and checkpoints;
                # they do not encode an annotated video. Never submit an
                # unsupported setting just because the legacy WebUI checkbox
                # happened to be selected.
                "generate_annotated_video": False,
                "tracker_backend": tracker_backend,
                "lock_match_roster": bool(lock_match_roster),
                "roster_stable_frames": int(roster_stable_frames),
                "max_roster_count": int(max_roster_count),
                # Fixed MVP roster policy: only two/four registrations are
                # allowed and the tracker locks to exactly that roster size.
                "expected_player_count": int(expected_player_count) if expected_player_count is not None else None,
                "roster_discovery_seconds": float(roster_discovery_seconds),
                "far_player_enhancement": bool(far_player_enhancement),
            },
        }
        if far_pose_roi is not None:
            request["configuration"]["far_pose_roi"] = [
                float(value) for value in far_pose_roi
            ]
        # Development-only override: persist the selected endpoint on this
        # task rather than mutating process-wide configuration. Production
        # uses a deployment-owned allowlisted GPU endpoint instead.
        config = self._stream_config_for_task({}, gpu_base_url=gpu_base_url)
        task = {
            "schema_version": "business-stream-replay.v1",
            "business_task_id": task_id,
            "status": "queued",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "input": {
                "filename": input_filename,
                "path": str(video_path),
                "bytes": written,
                "kind": input_kind,
                "fixed_video_id": fixed_video_id,
            },
            "stream_request": request,
            "replay": {
                "segment_seconds": float(segment_seconds),
                "realtime": bool(realtime),
                "encoding_mode": "h264",
                "kind": "development_file_replay",
                "annotated_video_requested": bool(generate_annotated_video),
                "annotated_video_effective": False,
            },
            "analysis_session_id": None,
            "gpu_base_url": config.base_url,
            "error": None,
        }
        self._write_task(task_dir, task)
        thread = threading.Thread(target=self._run, args=(task_id,), daemon=True, name=f"stream-replay-{task_id[-8:]}")
        with self._lock:
            self._threads[task_id] = thread
        thread.start()
        return self.status(task_id)

    def submit_fixed_video(self, fixed_video_id: str) -> dict[str, Any]:
        """Submit a business-owned local fixture without exposing its path."""

        try:
            fixed_video = get_fixed_video(fixed_video_id)
        except KeyError as exc:
            raise ValueError("fixed_video_id does not exist") from exc
        analysis = fixed_video["analysis"]
        return self.submit(
            None,
            calibration_id=str(fixed_video["calibration_id"]),
            court_corners=fixed_video["court_corners"],
            camera_id=f"fixed-video-{fixed_video['id']}",
            analysis_sample_hz=int(analysis["analysis_sample_hz"]),
            pose_imgsz=int(analysis["pose_imgsz"]),
            shuttle_detector=str(analysis["shuttle_detector"]),
            generate_annotated_video=bool(analysis["generate_annotated_video"]),
            tracker_backend="bytetrack",
            lock_match_roster=True,
            roster_stable_frames=3,
            max_roster_count=int(analysis["expected_player_count"]),
            expected_player_count=int(analysis["expected_player_count"]),
            roster_discovery_seconds=8.0,
            far_player_enhancement=False,
            far_pose_roi=None,
            segment_seconds=2.0,
            realtime=False,
            source_video_path=Path(str(fixed_video["source_path"])),
            fixed_video_id=str(fixed_video["id"]),
        )

    def status(self, task_id: str) -> dict[str, Any]:
        task_dir = self.root / task_id
        task = self._read_task(task_dir)
        if task is None:
            raise KeyError(task_id)
        session_id = task.get("analysis_session_id")
        if session_id and task.get("status") not in {"completed", "failed"}:
            try:
                ledger = DeliveryLedger(task_dir / "delivery-ledger.json")
                status = StreamSessionClient(self._stream_config_for_task(task), ledger).get_status()
                task["gpu_status"] = status
                # Do not overwrite the worker-owned task file from a polling
                # request. The worker persists receipts/progress; the response
                # exposes the latest remote state as a transient view.
                if status.get("status") in _TERMINAL and task.get("status") in {"running", "submitted"}:
                    task["status"] = "completed"
            except Exception as exc:
                task["gpu_status_error"] = f"{type(exc).__name__}: {exc}"
        return self._public_task(task)

    def candidate_photo(self, task_id: str, track_id: str) -> tuple[bytes, str]:
        """Proxy one verified candidate crop without exposing GPU credentials."""

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(track_id)):
            raise ValueError("invalid track_id")
        task_dir = self.root / task_id
        task = self._read_task(task_dir)
        if task is None:
            raise KeyError(task_id)
        session_id = str(task.get("analysis_session_id") or "")
        if not session_id:
            raise FileNotFoundError("analysis session is not available")
        terminal = ((task.get("result") or {}).get("status") or task.get("gpu_status") or {})
        candidates = terminal.get("track_candidates") if isinstance(terminal, dict) else None
        known_tracks = {
            str(candidate.get("track_id"))
            for candidate in (candidates or [])
            if isinstance(candidate, dict) and candidate.get("track_id")
        }
        if track_id not in known_tracks:
            raise FileNotFoundError("candidate track is not available for this task")
        config = self._stream_config_for_task(task)
        request = Request(
            f"{config.base_url}/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}",
            headers={"X-API-Key": config.api_key},
            method="GET",
        )
        try:
            with urlopen(request, timeout=config.timeout_seconds) as remote:
                media_type = remote.headers.get_content_type() or "image/jpeg"
                return remote.read(), media_type
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError("candidate photo is not available") from exc
            raise RuntimeError(f"GPU candidate photo request failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("GPU candidate photo request failed") from exc

    def claim_track(self, task_id: str, track_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        """Bind one anonymous track to the local user's body profile.

        This development endpoint is deliberately business-side only. It never
        forwards age, gender, height, weight, score, or a user identifier to
        the GPU; it only recomputes the locally derived energy estimate for the
        claimed anonymous track.
        """

        task_dir = self.root / task_id
        task = self._read_task(task_dir)
        if task is None:
            raise KeyError(task_id)
        self._require_known_track(task, track_id)
        normalized_profile = self._normalize_profile(profile)
        derivation = ((task.get("result") or {}).get("business_derivation") or {})
        profiles_path = write_body_profiles(
            task_dir,
            [{"track_id": track_id, **normalized_profile}],
            consent=True,
        )
        derivation["body_profiles_path"] = profiles_path
        task.setdefault("result", {})["business_derivation"] = derivation
        # This reference is not sent to GPU and lets the dev gateway retain the
        # latest local claim without persisting a name or WeChat identity.
        task["claimed_track_id"] = track_id
        self._write_task(task_dir, task)
        self._refresh_movement_metrics(task_dir, task)
        return self.personal_summary(task_id, track_id)

    def personal_summary(self, task_id: str, track_id: str) -> dict[str, Any]:
        """Return only the real, claimed-track movement evidence for Summary."""

        task_dir = self.root / task_id
        task = self._read_task(task_dir)
        if task is None:
            raise KeyError(task_id)
        self._require_known_track(task, track_id)
        derivation = ((task.get("result") or {}).get("business_derivation") or {})
        metrics = derivation.get("metrics") or {}
        player = next(
            (
                item
                for item in (metrics.get("players") or [])
                if isinstance(item, dict) and str(item.get("track_id")) == track_id
            ),
            None,
        )
        # Completed sessions created before a new business-side metric schema
        # are refreshed on demand.  This keeps historic demo sessions useful
        # without asking the GPU to rerun or sending it any profile data.
        if player is not None and (
            not isinstance(player.get("ability_scores"), dict)
            or not isinstance(player.get("match_load"), dict)
            or float(
                ((player.get("movement") or {}).get("minimum_acceleration_sprint_distance_m"))
                or 0.0
            ) != 2.0
            or float(
                ((player.get("movement") or {}).get("acceleration_sprint_window_seconds"))
                or 0.0
            ) != 0.5
            or float(
                (((player.get("movement") or {}).get("agility_movement") or {}).get(
                    "maximum_seconds_each_leg"
                ))
                or 0.0
            ) != 0.5
        ):
            metrics = self._refresh_movement_metrics(task_dir, task)
            player = next(
                (
                    item
                    for item in (metrics.get("players") or [])
                    if isinstance(item, dict) and str(item.get("track_id")) == track_id
                ),
                None,
            )
        if player is None:
            raise ValueError("personal summary is not ready; claimed-track metrics are unavailable")
        terminal = ((task.get("result") or {}).get("status") or {})
        duration_sec = float(
            ((terminal.get("progress") or {}).get("processed_source_time_sec")) or 0.0
        )
        return {
            "schema_version": "business-personal-summary.v1",
            "business_task_id": task_id,
            "analysis_session_id": task.get("analysis_session_id"),
            "track_id": track_id,
            "match": {
                "duration_sec": round(duration_sec, 3),
                "analysis_status": terminal.get("status"),
            },
            "measurement_coverage": player.get("measurement_coverage") or {},
            "movement": player.get("movement") or {},
            "ability_scores": player.get("ability_scores") or {},
            "match_load": player.get("match_load") or {},
            "energy_estimate": player.get("energy_estimate") or {},
            "quality": player.get("quality") or {},
        }

    def _refresh_movement_metrics(self, task_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
        """Rebuild business-only metrics from materialized anonymous tracks."""

        derivation = ((task.get("result") or {}).get("business_derivation") or {})
        required_paths = {
            key: derivation.get(key)
            for key in ("materialized_detections_path", "spatial_summary_path", "metadata_path")
        }
        if not all(required_paths.values()):
            raise ValueError("personal summary is not ready; movement derivation is unavailable")
        metrics = generate_movement_metrics(
            output_dir=task_dir,
            detections_path=required_paths["materialized_detections_path"],
            spatial_summary_path=required_paths["spatial_summary_path"],
            metadata_path=required_paths["metadata_path"],
            body_profiles_path=derivation.get("body_profiles_path") or None,
        )
        derivation["metrics"] = metrics
        task.setdefault("result", {})["business_derivation"] = derivation
        self._write_task(task_dir, task)
        return metrics

    @staticmethod
    def _require_known_track(task: dict[str, Any], track_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(track_id)):
            raise ValueError("invalid track_id")
        terminal = ((task.get("result") or {}).get("status") or task.get("gpu_status") or {})
        known_tracks = {
            str(candidate.get("track_id"))
            for candidate in (terminal.get("track_candidates") or [])
            if isinstance(candidate, dict) and candidate.get("track_id")
        }
        if track_id not in known_tracks:
            raise ValueError("claimed track is not available for this task")

    @staticmethod
    def _normalize_profile(profile: dict[str, Any]) -> dict[str, float]:
        if not isinstance(profile, dict):
            raise ValueError("profile is required for the energy estimate")
        try:
            height_cm = float(profile.get("height_cm", profile.get("heightCm")))
            weight_kg = float(profile.get("weight_kg", profile.get("weightKg")))
        except (TypeError, ValueError) as exc:
            raise ValueError("height_cm and weight_kg are required for the energy estimate") from exc
        if not 80.0 <= height_cm <= 260.0 or not 20.0 <= weight_kg <= 300.0:
            raise ValueError("height_cm or weight_kg is outside the supported range")
        return {"height_cm": height_cm, "weight_kg": weight_kg}

    def _run(self, task_id: str) -> None:
        task_dir = self.root / task_id
        task = self._read_task(task_dir)
        if task is None:
            return
        task["status"] = "running"
        task["started_at"] = _utc_now()
        self._write_task(task_dir, task)
        try:
            config = self._stream_config_for_task(task)
            ledger = DeliveryLedger(task_dir / "delivery-ledger.json")
            client = StreamSessionClient(config, ledger)
            replay = task["replay"]

            def persist_progress(event: dict[str, Any]) -> None:
                # A GPU create receipt is written immediately, before the
                # first segment is cut. Pollers can therefore prove that the
                # remote side received the task and query its actual state.
                latest = self._read_task(task_dir)
                if latest is None:
                    return
                session_id = event.get("analysis_session_id")
                if session_id:
                    latest["analysis_session_id"] = session_id
                latest["progress"] = event
                self._write_task(task_dir, latest)

            result = replay_video(
                Path(task["input"]["path"]),
                client=client,
                segmenter=GrowingVideoSegmenter(
                    task_dir / "segments",
                    segment_duration_sec=float(replay["segment_seconds"]),
                    encoding_mode=str(replay["encoding_mode"]),
                    preserve_audio=False,
                ),
                create_request=task["stream_request"],
                create_idempotency_key=f"{task_id}_create",
                realtime=bool(replay["realtime"]),
                overwrite_segments=False,
                progress_callback=persist_progress,
            )
            # Upload/seal only proves durable delivery.  Do not mark a local
            # replay complete until the GPU has finished the queued segments
            # and the business service has converted anonymous event evidence
            # into the movement metrics shown by the WebUI.
            terminal_status = client.wait_for_terminal(
                poll_interval_seconds=1.0,
                timeout_seconds=3600.0,
            )
            result["status"] = terminal_status
            if terminal_status.get("status") in {"finalized", "partial"}:
                from .streaming.derivation import derive_stream_movement_metrics

                result["business_derivation"] = derive_stream_movement_metrics(
                    task_dir,
                    client=client,
                    terminal_status=terminal_status,
                    create_request=task["stream_request"],
                )
            task["analysis_session_id"] = result["analysis_session_id"]
            task["result"] = result
            task["status"] = "completed" if result["status"].get("status") in _TERMINAL else "submitted"
            task["finished_at"] = _utc_now()
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = {"type": type(exc).__name__, "message": str(exc)}
            task["finished_at"] = _utc_now()
        self._write_task(task_dir, task)

    @staticmethod
    def _public_task(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "business_task_id": task["business_task_id"],
            "status": task["status"],
            "created_at": task["created_at"],
            "updated_at": task.get("updated_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "analysis_session_id": task.get("analysis_session_id"),
            "gpu_base_url": task.get("gpu_base_url"),
            "gpu_status": task.get("gpu_status"),
            "gpu_status_error": task.get("gpu_status_error"),
            "progress": task.get("progress"),
            "error": task.get("error"),
            "input": {
                key: value
                for key, value in (task.get("input") or {}).items()
                if key != "path"
            },
            "replay": task.get("replay"),
            "result": task.get("result"),
            "status_url": f"/api/v1/development/stream-replays/{task['business_task_id']}",
        }

    @staticmethod
    def _read_task(task_dir: Path) -> dict[str, Any] | None:
        path = task_dir / "task.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    @staticmethod
    def _write_task(task_dir: Path, task: dict[str, Any]) -> None:
        task["updated_at"] = _utc_now()
        _atomic_json(task_dir / "task.json", task)

    @staticmethod
    def _stream_config_for_task(task: dict[str, Any], *, gpu_base_url: str | None = None) -> StreamClientConfig:
        """Return the immutable destination saved for one development replay.

        The API key and retry policy always come from gateway configuration;
        only a validated base URL may be supplied by the local operator.
        """
        configured = StreamClientConfig.from_environment()
        selected = str(
            gpu_base_url if gpu_base_url is not None else task.get("gpu_base_url") or ""
        ).strip()
        if not selected:
            return configured
        parsed = urlparse(selected)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("gpu_base_url must be an http(s) base URL without credentials or query parameters")
        return replace(configured, base_url=selected.rstrip("/"))


def create_app(root: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("GOOD_BADMINTON_BUSINESS_DATA_DIR", "outputs/business_stream_replays"))
    manager = LocalReplayManager(root)
    court_matches = CourtMatchManager(manager)
    app = FastAPI(title="Good-Badminton Local Business Stream Gateway", version="0.1.0")
    app.state.replay_manager = manager
    app.state.court_match_manager = court_matches

    @app.get("/api/v1/health")
    def health():
        try:
            return manager.health()
        except Exception as exc:
            return JSONResponse(status_code=503, content={"status": "misconfigured", "error": str(exc)})

    @app.get("/api/v1/development/fixed-videos")
    def fixed_videos():
        """Expose selector metadata without revealing local asset locations."""
        try:
            return {"items": list_public_fixed_videos()}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail=f"fixed video catalogue unavailable: {exc}") from exc

    @app.get("/api/v1/development/courts/{court_id}/active-match")
    def active_court_match(court_id: str, viewer_id: str = ""):
        """Read the only waiting/playing match for one court, if present."""
        return {"match": court_matches.active(court_id, viewer_id)}

    @app.post("/api/v1/development/courts/{court_id}/queue/join")
    def join_court_match(court_id: str, body: dict = Body(...)):
        try:
            return {
                "match": court_matches.join(
                    court_id,
                    str(body.get("actor_id") or ""),
                    str(body.get("slot_id") or ""),
                )
            }
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/development/court-matches/{match_id}")
    def court_match_status(match_id: str, viewer_id: str = ""):
        try:
            return {"match": court_matches.status(match_id, viewer_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="court match not found") from exc

    @app.post("/api/v1/development/court-matches/{match_id}/leave")
    def leave_court_match(match_id: str, body: dict = Body(...)):
        try:
            return {"match": court_matches.leave(match_id, str(body.get("actor_id") or ""))}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="court match not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/development/court-matches/{match_id}/start", status_code=202)
    def start_court_match(match_id: str, body: dict = Body(...)):
        try:
            return {
                "match": court_matches.start(
                    match_id,
                    str(body.get("actor_id") or ""),
                    str(body.get("fixed_video_id") or ""),
                )
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="court match not found") from exc
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/development/court-matches/{match_id}/end")
    def end_court_match(match_id: str, body: dict = Body(...)):
        try:
            return {"match": court_matches.end(match_id, str(body.get("actor_id") or ""))}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="court match not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/development/court-matches/{match_id}/delivery-window")
    def begin_match_delivery(match_id: str, body: dict = Body(...)):
        try:
            return {"match": court_matches.begin_delivery(match_id, str(body.get("actor_id") or ""))}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="court match not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/development/stream-replays", status_code=202)
    def submit_replay(
        video: UploadFile = File(...),
        calibration_id: str = Form(...),
        court_corners: str = Form(...),
        camera_id: str = Form("local-development-camera"),
        analysis_sample_hz: int = Form(10),
        pose_imgsz: int = Form(960),
        shuttle_detector: str = Form("yolo"),
        generate_annotated_video: bool = Form(False),
        tracker_backend: str = Form("bytetrack"),
        lock_match_roster: bool = Form(True),
        roster_stable_frames: int = Form(3),
        max_roster_count: int = Form(4),
        expected_player_count: int | None = Form(None),
        roster_discovery_seconds: float = Form(8.0),
        far_player_enhancement: bool = Form(False),
        far_pose_roi: str = Form(""),
        segment_seconds: float = Form(2.0),
        realtime: bool = Form(False),
        gpu_base_url: str = Form(""),
    ):
        try:
            parsed_corners = json.loads(court_corners)
            parsed_far_roi = json.loads(far_pose_roi) if far_pose_roi.strip() else None
            return manager.submit(
                video,
                calibration_id=calibration_id,
                court_corners=parsed_corners,
                camera_id=camera_id,
                analysis_sample_hz=analysis_sample_hz,
                pose_imgsz=pose_imgsz,
                shuttle_detector=shuttle_detector,
                generate_annotated_video=generate_annotated_video,
                tracker_backend=tracker_backend,
                lock_match_roster=lock_match_roster,
                roster_stable_frames=roster_stable_frames,
                max_roster_count=max_roster_count,
                expected_player_count=expected_player_count,
                roster_discovery_seconds=roster_discovery_seconds,
                far_player_enhancement=far_player_enhancement,
                far_pose_roi=parsed_far_roi,
                segment_seconds=segment_seconds,
                realtime=realtime,
                gpu_base_url=gpu_base_url,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/development/fixed-video-replays", status_code=202)
    def submit_fixed_video_replay(fixed_video_id: str = Body(..., embed=True)):
        """Create a replay from a local, business-owned fixed video asset."""
        try:
            return manager.submit_fixed_video(fixed_video_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/development/stream-replays/{task_id}")
    def replay_status(task_id: str):
        try:
            return manager.status(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="stream replay task not found") from exc

    @app.post("/api/v1/development/stream-replays/{task_id}/claims")
    def claim_replay_track(task_id: str, body: dict = Body(...)):
        try:
            return manager.claim_track(
                task_id,
                str(body.get("track_id") or ""),
                body.get("profile") if isinstance(body.get("profile"), dict) else {},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="stream replay task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/development/stream-replays/{task_id}/personal-summary/{track_id}")
    def replay_personal_summary(task_id: str, track_id: str):
        try:
            return manager.personal_summary(task_id, track_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="stream replay task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/development/stream-replays/{task_id}/candidate-photos/{track_id}")
    def replay_candidate_photo(task_id: str, track_id: str):
        try:
            content, media_type = manager.candidate_photo(task_id, track_id)
            return Response(
                content=content,
                media_type=media_type,
                headers={"Cache-Control": "private, max-age=60"},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="stream replay task not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


load_runtime_environment()
_load_local_gpu_env()
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host, port = business_service_listener()
    uvicorn.run(app, host=host, port=port)
