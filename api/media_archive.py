"""Decode, verify, assemble and archive sealed input independently of inference.

The final OSS manifest is the commit marker. Original segments and master are
published first under an immutable input revision; no bucket listing is needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

from .media_storage import oss_from_environment, safe_key, sha256_file
from .media_identity import media_prefix, validate_media_identity
from .stream_sessions import _durable_replace
from .stream_models import TERMINAL_SESSION_STATES
from .vision_profiles import get_vision_profile


class MediaValidationError(ValueError):
    pass


def write_json(path, value):
    _durable_replace(Path(path), (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


class MediaInspector:
    def __init__(self, *, ffmpeg="ffmpeg", ffprobe="ffprobe", timeout=1200):
        self.ffmpeg, self.ffprobe, self.timeout = ffmpeg, ffprobe, float(timeout)

    def _run(self, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout, check=False)
        if result.returncode or result.stderr.strip():
            # ffmpeg runs at error log level: even recovered decode errors fail QA.
            raise MediaValidationError("media decode/probe failed: " + result.stderr[-800:])
        return result.stdout

    def inspect(self, path):
        path = Path(path).resolve()
        probe = json.loads(self._run([
            self.ffprobe, "-v", "error", "-show_streams", "-show_data", "-of", "json", str(path),
        ]))
        streams = probe.get("streams", [])
        videos = [stream for stream in streams if stream["codec_type"] == "video"]
        if len(videos) != 1 or any(s["codec_type"] not in {"video", "audio"} for s in streams):
            raise MediaValidationError("exactly one video stream and optional audio are supported")
        video = videos[0]
        rate = float(Fraction(video.get("avg_frame_rate", "0/1")))
        if not math.isfinite(rate) or rate <= 0:
            raise MediaValidationError("video frame rate is unavailable")
        frame_data = json.loads(self._run([
            self.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time,key_frame",
            "-of", "json", str(path),
        ])).get("frames", [])
        if not frame_data or not frame_data[0].get("key_frame"):
            raise MediaValidationError("segment must start with an independently decodable keyframe")
        pts = [float(frame["best_effort_timestamp_time"]) for frame in frame_data]
        if any(not math.isfinite(t) for t in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
            raise MediaValidationError("frame timestamps must be finite and strictly increasing")
        last = frame_data[-1]
        tail = float(last.get("duration_time", last.get("pkt_duration_time", 1 / rate)))
        if not math.isfinite(tail) or tail <= 0:
            tail = 1 / rate
        hashes = self._run([
            self.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
            "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
            "-fps_mode", "passthrough", "-f", "framemd5", "-",
        ])
        frame_hashes = [line.rsplit(",", 1)[1].strip() for line in hashes.splitlines()
                        if line.strip() and not line.startswith("#")]
        if len(frame_hashes) != len(pts):
            raise MediaValidationError("probe and full decode frame counts disagree")
        if any(s["codec_type"] == "audio" for s in streams):
            self._run([self.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
                       "-i", str(path), "-map", "0:a", "-f", "null", "-"])
        signature_fields = ("codec_type", "codec_name", "codec_tag_string", "width", "height", "pix_fmt",
                            "time_base", "sample_rate", "channels", "channel_layout", "extradata")
        return {
            "frame_count": len(pts), "frame_offsets_sec": [t - pts[0] for t in pts],
            "duration_sec": pts[-1] - pts[0] + tail, "frame_period_sec": 1 / rate,
            "stream_signature": [{field: s.get(field) for field in signature_fields} for s in streams],
            "decoded_frame_hashes": frame_hashes,
        }


def input_snapshot(session):
    configuration = dict(session["configuration"])
    sport_id = configuration.get("sport_id", "badminton")
    get_vision_profile(sport_id)  # Never archive an unrecognised sport identity.
    configuration.setdefault("sport_id", sport_id)
    fields = ("segment_index", "source_start_time_sec", "duration_sec", "sha256", "idempotency_key",
              "content_type", "content_length_bytes", "source_frame_start_index", "source_frame_count",
              "source_frame_time_offsets_sec")
    snapshot = {
        "schema_version": "media-archive.v1", "analysis_session_id": session["analysis_session_id"],
        "sport_id": sport_id, "camera_id": session["camera_id"], "calibration_id": session["calibration_id"],
        "court_corners": session["court_corners"], "configuration": configuration,
        "expected_last_segment_index": session["expected_last_segment_index"],
        "segments": [{field: entry.get(field) for field in fields}
                     for entry in sorted(session["segments"].values(), key=lambda e: e["segment_index"])],
    }

    if session.get("media_identity") is not None:
        snapshot["media_identity"] = validate_media_identity(session["media_identity"],
            camera_id=session["camera_id"], client_reference=session.get("client_reference"),
            calibration_id=session["calibration_id"])
    return snapshot


def revision_for(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_segments(snapshot, segment_path, inspector):
    """Return checked evidence or reject unknown continuity; never infer identities."""
    entries = snapshot["segments"]
    expected = snapshot["expected_last_segment_index"]
    if type(expected) is not int or expected < 0 or expected > 1_000_000:
        raise MediaValidationError("a non-empty final segment range is required")
    indexes = [entry["segment_index"] for entry in entries]
    if indexes != list(range(expected + 1)):
        raise MediaValidationError("missing, duplicate or out-of-range segment indexes")
    checked, previous = [], None
    for entry in entries:
        index = entry["segment_index"]
        path = Path(segment_path(index))
        if not path.is_file() or path.stat().st_size != entry["content_length_bytes"] or sha256_file(path) != entry["sha256"]:
            raise MediaValidationError(f"segment {index}: bytes do not match durable receipt")
        evidence = inspector.inspect(path)
        count, start = entry.get("source_frame_count"), entry.get("source_frame_start_index")
        offsets = entry.get("source_frame_time_offsets_sec")
        if type(start) is not int or start < 0 or type(count) is not int or count != evidence["frame_count"]:
            raise MediaValidationError(f"segment {index}: source frame identity missing or count mismatch")
        if (not isinstance(offsets, list) or len(offsets) != count
                or any(not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) for t in offsets)
                or any(abs(a - b) > 0.002 for a, b in zip(offsets, evidence["frame_offsets_sec"]))):
            raise MediaValidationError(f"segment {index}: source timestamps missing or disagree with media")
        period = evidence["frame_period_sec"]
        if any(b - a > period * 1.5 for a, b in zip(offsets, offsets[1:])):
            raise MediaValidationError(f"segment {index}: internal source timestamp gap")
        if abs(entry["duration_sec"] - evidence["duration_sec"]) > 0.002:
            raise MediaValidationError(f"segment {index}: declared and decoded durations disagree")
        if previous:
            prior_entry, prior_evidence = previous
            if start != prior_entry["source_frame_start_index"] + prior_entry["source_frame_count"]:
                raise MediaValidationError(f"segment {index}: source frame gap or overlap")
            if abs(entry["source_start_time_sec"] - prior_entry["source_start_time_sec"] - prior_entry["duration_sec"]) > 0.002:
                raise MediaValidationError(f"segment {index}: source time gap or overlap")
            if evidence["stream_signature"] != prior_evidence["stream_signature"]:
                raise MediaValidationError(f"segment {index}: stream parameters changed")
        checked.append(evidence)
        previous = entry, evidence
    return checked


class StreamMediaArchive:
    def __init__(self, manager, storage, *, prefix="match-media", inspector=None, retry_seconds=60):
        self.manager, self.storage = manager, storage
        manager.media_archive_enabled = True
        self.prefix = safe_key(prefix.strip("/"))
        self.inspector = inspector or MediaInspector(
            ffmpeg=os.environ.get("FFMPEG_BIN", "ffmpeg"), ffprobe=os.environ.get("FFPROBE_BIN", "ffprobe"),
        )
        self.retry_seconds = retry_seconds
        self._run_lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None

    def _directory(self, session_id):
        return self.manager._session_dir(session_id) / "media"

    def status(self, session_id):
        session = self.manager.get_session(session_id)
        if session is None:
            raise ValueError("session not found")
        if not session.get('sealed') and session.get('status') in {'failed', 'cancelled', 'interrupted_needs_rebuild'}:
            return {'status': 'failed', 'error': 'analysis_ended_before_seal'}
        path = self._directory(session_id) / "status.json"
        return json.loads(path.read_text()) if path.is_file() else {
            "status": "pending" if session.get("sealed") else "awaiting_seal",
        }

    def manifest(self, session_id):
        state = self.status(session_id)
        if state["status"] != "ready":
            raise MediaValidationError("media archive is not ready")
        return json.loads((self._directory(session_id) / state["revision"] / "manifest.json").read_text())

    def process_session(self, session_id):
        # Serializes automatic and manual work without holding the ingest lock
        # across decoding, assembly or network I/O. Input is immutable after seal.
        with self._run_lock:
            with self.manager._lock:
                session = deepcopy(self.manager.get_session(session_id))
            if session is None or not session.get("sealed"):
                raise MediaValidationError("seal input before preparing a media archive")
            snapshot = input_snapshot(session)
            revision = revision_for(snapshot)
            old = self.status(session_id)
            if old.get("status") == "ready" and old.get("revision") == revision:
                return old
            root = self._directory(session_id)
            root.mkdir(parents=True, exist_ok=True)
            state = {"status": "checking", "revision": revision, "attempted_at": time.time(),
                     "analysis_quality_verified": False, "attempts": int(old.get("attempts", 0)) + 1}
            write_json(root / "status.json", state)
            try:
                if (getattr(self.storage, "allowed_root", None)
                        or getattr(self.storage, "requires_media_identity", False)):
                    if not snapshot.get("media_identity"):
                        raise MediaValidationError("DogeCloud archive requires a business-issued media identity")
                    try:
                        self.storage._key(media_prefix(snapshot["media_identity"]) + "/manifest.json")
                    except ValueError:
                        raise MediaValidationError("media identity belongs to a different storage environment")
                result = self._prepare(snapshot, revision, root, state)
            except Exception as error:
                state["status"] = "invalid" if isinstance(error, MediaValidationError) else ("failed" if state["attempts"] >= 5 else "retrying")
                # SDK exception messages can contain request URLs/credential IDs.
                state["error"] = str(error)[:1000] if isinstance(error, MediaValidationError) else type(error).__name__
                write_json(root / "status.json", state)
                return state
            write_json(root / "status.json", result)
            return result

    def _prepare(self, snapshot, revision, root, state):
        session_id = snapshot["analysis_session_id"]
        path_for = lambda i: self.manager._segment_path(session_id, i)
        checked = verify_segments(snapshot, path_for, self.inspector)
        directory = root / revision
        directory.mkdir(parents=True, exist_ok=True)
        concat = directory / "concat.txt"
        concat.write_text("".join("file '" + str(path_for(e["segment_index"]).resolve()).replace("'", "'\\''") + "'\n"
                                  for e in snapshot["segments"]))
        state["status"] = "assembling"
        write_json(root / "status.json", state)
        master = directory / "master.mp4"
        expected_hashes = [digest for evidence in checked for digest in evidence["decoded_frame_hashes"]]
        master_evidence = None
        if master.is_file():
            try:
                cached = self.inspector.inspect(master)
                if cached["decoded_frame_hashes"] == expected_hashes:
                    master_evidence = cached
            except (MediaValidationError, ValueError, subprocess.SubprocessError):
                pass
        if master_evidence is None:
            master.unlink(missing_ok=True)
            from .media_limits import ensure_capacity
            required = sum(entry["content_length_bytes"] for entry in snapshot["segments"])
            with self.manager._lock:
                ensure_capacity(self.manager, 0, session_id=session_id)
            # Bound even a malformed/buggy concat output. QA rejects a truncated result.
            maximum_output = int(required * 1.1) + 1024**2
            self.inspector._run([
                self.inspector.ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat), "-map", "0", "-c", "copy", "-movflags", "+faststart",
                "-fs", str(maximum_output), str(master),
            ])
            master_evidence = self.inspector.inspect(master)
        if master_evidence["decoded_frame_hashes"] != expected_hashes:
            raise MediaValidationError("assembled video frames differ from ordered original segments")
        expected_duration = sum(e["duration_sec"] for e in checked)
        if abs(master_evidence["duration_sec"] - expected_duration) > max(0.002, checked[0]["frame_period_sec"] / 2):
            raise MediaValidationError("assembled video duration differs from source timeline")
        # Compare every decoded PTS, not only total duration: concat must not hide a boundary jump.
        expected_offsets, cursor = [], 0.0
        for evidence in checked:
            expected_offsets.extend(cursor + offset for offset in evidence["frame_offsets_sec"])
            cursor += evidence["duration_sec"]
        if any(abs(a - b) > 0.002 for a, b in zip(master_evidence["frame_offsets_sec"], expected_offsets)):
            raise MediaValidationError("assembled frame timestamps differ from source timeline")
        base = (media_prefix(snapshot["media_identity"]) if snapshot.get("media_identity") else
                f"{self.prefix}/{snapshot['sport_id']}/{session_id}/{revision}")
        manifest = deepcopy(snapshot)
        manifest.update({"revision": revision, "status": "ready", "analysis_quality_verified": False,
                         "validation": "full_decode_frame_hash_and_source_timeline",
                         "total_source_frames": len(expected_hashes)})
        state["status"] = "uploading"
        write_json(root / "status.json", state)
        for entry, evidence in zip(manifest["segments"], checked):
            entry["object_key"] = f"{base}/segments/{entry['segment_index']:08d}.mp4"
            entry["decoded_frame_count"] = evidence["frame_count"]
            self.storage.upload(entry["object_key"], path_for(entry["segment_index"]))
        integrity = {
            "schema_version": "media-integrity.v1", "revision": revision,
            "sport_id": snapshot["sport_id"], "status": "complete",
            "expected_last_segment_index": snapshot["expected_last_segment_index"],
            "source_frame_identity_verified": True, "source_timeline_verified": True,
            "master_decoded_frames_match_segments": True,
            "model_accuracy_verified": False,
            "segments": [{"segment_index": entry["segment_index"], "sha256": entry["sha256"],
                          "decoded_frame_count": evidence["frame_count"], "decoded_duration_sec": evidence["duration_sec"],
                          "stream_signature": evidence["stream_signature"]}
                         for entry, evidence in zip(snapshot["segments"], checked)],
        }
        integrity_path = directory / "integrity.json"
        write_json(integrity_path, integrity)
        manifest["integrity"] = {"object_key": f"{base}/integrity.json", "sha256": sha256_file(integrity_path),
                                 "content_length_bytes": integrity_path.stat().st_size}
        self.storage.upload(manifest["integrity"]["object_key"], integrity_path)
        manifest["master"] = {"object_key": f"{base}/master.mp4", "sha256": sha256_file(master),
                              "content_length_bytes": master.stat().st_size, "duration_sec": expected_duration}
        self.storage.upload(manifest["master"]["object_key"], master)
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        key = f"{base}/manifest.json"
        self.storage.upload(key, manifest_path)  # Commit last; failures never publish a ready manifest.
        return {**state, "status": "ready", "ready_at": time.time(), "manifest_object_key": key,
                "manifest_sha256": sha256_file(manifest_path), "total_source_frames": len(expected_hashes),
                "analysis_at_archive": session_analysis_status(self.manager.get_session(session_id))}

    def release_verified_media(self, session_id, *, now=None):
        """Release duplicate video bytes only; preserve receipts, events and archive JSON.

        Ready is a cloud commit marker, not an inference completion marker. Both
        must hold before releasing the local inputs. Failures retain all inputs.
        """
        if os.environ.get("GOOD_BADMINTON_MEDIA_LOCAL_CLEANUP_ENABLED", "1").lower() not in {"1", "true"}:
            return False
        now = time.time() if now is None else float(now)
        grace = int(os.environ.get("GOOD_BADMINTON_MEDIA_LOCAL_CACHE_GRACE_SECONDS", "300"))
        if grace < 0:
            raise ValueError("media cache grace period must be nonnegative")
        with self._run_lock:
            state = self.status(session_id)
            if state.get("status") != "ready" or state.get("local_cache_status") == "released":
                return False
            if now - state.get("ready_at", state.get("attempted_at", now)) < grace:
                return False
            with self.manager._lock:
                session = deepcopy(self.manager.get_session(session_id))
                active = self.manager._active_segment
                if active and active.get("session_id") == session_id:
                    return False
            if (not session.get("sealed") or session.get("status") not in TERMINAL_SESSION_STATES
                    or session.get("repair", {}).get("window_open")):
                return False
            revision = revision_for(input_snapshot(session))
            if revision != state.get("revision"):
                return False
            directory = self._directory(session_id) / revision
            manifest_path = directory / "manifest.json"
            if sha256_file(manifest_path) != state.get("manifest_sha256"):
                return False
            # Recheck the remote commit marker before deleting any duplicate bytes.
            try:
                remote = self.storage.head(state["manifest_object_key"])
                if (remote.get("sha256") != state["manifest_sha256"]
                        or remote.get("size_bytes") != manifest_path.stat().st_size):
                    raise IOError("remote archive commit identity mismatch")
            except Exception as error:
                write_json(self._directory(session_id) / "status.json", {
                    **state, "cleanup_error_type": type(error).__name__,
                })
                return False
            days = int(os.environ.get("DOGECLOUD_RETENTION_DAYS", "7"))
            if not 1 <= days <= 7:
                raise ValueError("media retention must be 1-7 days")
            expiry = datetime.fromisoformat(session["created_at"].replace("Z", "+00:00")) + timedelta(days=days)
            if expiry.timestamp() <= now:
                return False  # The remote lifecycle may already have expired the only other copy.
            with self.manager._lock:
                session = self.manager.get_session(session_id)
                active = self.manager._active_segment
                if (not session.get("sealed") or session.get("status") not in TERMINAL_SESSION_STATES
                        or (active and active.get("session_id") == session_id)):
                    return False
                base = self.manager._session_dir(session_id).resolve()
                targets = [self.manager._segment_path(session_id, int(i)) for i in session["segments"]]
                targets.append(directory / "master.mp4")
                for target in targets:
                    if (target.is_symlink() or target.parent.is_symlink()
                            or not target.resolve().is_relative_to(base)):
                        raise ValueError("media release target escapes its session")
                session["local_media_release"] = {
                    "phase": "releasing", "revision": revision,
                    "manifest_sha256": state["manifest_sha256"],
                    "manifest_object_key": state["manifest_object_key"],
                    "expires_at": expiry.isoformat(), "cloud_verified_at": now,
                }
                # Persist the remote receipt before unlink; a crash is safely retryable.
                self.manager._save(session)
                released = 0
                for target in targets:
                    if target.is_file():
                        released += target.stat().st_size
                        target.unlink()
                session["local_media_release"]["phase"] = "released"
                self.manager._save(session)
                write_json(self._directory(session_id) / "status.json", {
                    **state, "local_cache_status": "released", "local_cache_released_at": now,
                    "released_bytes": released, "cleanup_error_type": None,
                })
            return True

    def run_once(self):
        for session in self.manager._list_sessions():
            if not session.get("sealed"):
                continue
            session_id = session["analysis_session_id"]
            try:
                state = self.status(session_id)
                if state["status"] == "ready":
                    self.release_verified_media(session_id)
                    continue
                if state["status"] in {"invalid", "failed"}:
                    continue
                if time.time() - state.get("attempted_at", 0) < self.retry_seconds:
                    continue
                self.process_session(session_id)
            except Exception as error:
                # A malformed older session cannot starve other cameras.
                import logging
                logging.getLogger(__name__).error("Media scheduling failed for %s (%s)", session_id, type(error).__name__)

    def start(self):
        if self._thread is not None:
            return
        def loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:
                    # A bad session does not kill recovery for every other camera.
                    pass
                self._wake.wait(5)
                self._wake.clear()
        self._thread = threading.Thread(target=loop, name="stream-media-archive", daemon=True)
        self._thread.start()

    def wake(self):
        self._wake.set()

    def request_retry(self, session_id):
        state = self.status(session_id)
        if state["status"] in {"invalid", "failed"}:
            write_json(self._directory(session_id) / "status.json", {**state, "status": "pending", "attempted_at": 0})
        self.wake()

    def stop(self):
        self._stop.set()
        self.wake()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()


def session_analysis_status(session):
    return {"mode": session.get("processing_mode", "analysis"), "status": session.get("status")}


def archive_from_environment(manager):
    if os.environ.get("GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED", "0").lower() not in {"1", "true"}:
        return None
    return StreamMediaArchive(
        manager, oss_from_environment(manager.data_dir / "oss-checkpoints", allow_environment_routing=True),
        prefix=os.environ.get("GOOD_BADMINTON_OSS_PREFIX", "match-media"),
    )
