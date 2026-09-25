"""Durable models owned by the business-side streaming client.

The ledger contains only transport identifiers and anonymous stream metadata.
User IDs, check IDs, player names and team information deliberately never cross
this package's GPU-service boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


STREAM_SCHEMA_VERSION = "stream-session.v1"
LEDGER_VERSION = "stream-delivery-ledger.v1"
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def validate_idempotency_key(value: str, field_name: str = "idempotency_key") -> str:
    """Apply the same transport profile enforced by the GPU stream API."""

    text = str(value or "")
    if not IDEMPOTENCY_KEY_PATTERN.fullmatch(text):
        raise ValueError(f"{field_name} must contain 16 to 128 safe characters")
    return text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class StreamClientConfig:
    """Runtime configuration for the business-to-GPU connection."""

    base_url: str
    api_key: str
    timeout_seconds: float = 30.0
    max_attempts: int = 5
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 5.0
    upload_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute http(s) URL")
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("retry backoff values must be non-negative")
        if self.upload_chunk_bytes < 64 * 1024:
            raise ValueError("upload_chunk_bytes must be at least 64 KiB")

    @classmethod
    def from_environment(cls) -> "StreamClientConfig":
        """Read configuration without embedding a deployment hostname in code."""

        # GPU_ANALYSIS_* is the service-neutral production name. Keep both
        # Good-Badminton aliases during migration so existing deployments do
        # not need an atomic environment-variable rename.
        base_url = (
            os.environ.get("GPU_ANALYSIS_BASE_URL", "").strip()
            or os.environ.get("GOOD_BADMINTON_STREAM_API_URL", "").strip()
            or os.environ.get("GOOD_BADMINTON_GPU_API_URL", "").strip()
        )
        api_key = (
            os.environ.get("GPU_ANALYSIS_API_KEY", "").strip()
            or os.environ.get("GOOD_BADMINTON_STREAM_API_KEY", "").strip()
            or os.environ.get("GOOD_BADMINTON_GPU_API_KEY", "").strip()
        )
        if not base_url:
            raise ValueError("GPU_ANALYSIS_BASE_URL (or a supported legacy alias) is required")
        if not api_key:
            raise ValueError("GPU_ANALYSIS_API_KEY (or a supported legacy alias) is required")
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=float(
                os.environ.get(
                    "GPU_ANALYSIS_TIMEOUT_SECONDS",
                    os.environ.get("GOOD_BADMINTON_STREAM_TIMEOUT_SECONDS", "30"),
                )
            ),
            max_attempts=int(
                os.environ.get(
                    "GPU_ANALYSIS_MAX_ATTEMPTS",
                    os.environ.get("GOOD_BADMINTON_STREAM_MAX_ATTEMPTS", "5"),
                )
            ),
            initial_backoff_seconds=float(
                os.environ.get(
                    "GPU_ANALYSIS_INITIAL_BACKOFF_SECONDS",
                    os.environ.get("GOOD_BADMINTON_STREAM_INITIAL_BACKOFF_SECONDS", "0.25"),
                )
            ),
            max_backoff_seconds=float(
                os.environ.get(
                    "GPU_ANALYSIS_MAX_BACKOFF_SECONDS",
                    os.environ.get("GOOD_BADMINTON_STREAM_MAX_BACKOFF_SECONDS", "5"),
                )
            ),
        )


@dataclass(frozen=True)
class SegmentMetadata:
    segment_index: int
    source_start_time_sec: float
    duration_sec: float
    sha256: str
    idempotency_key: str
    content_type: str
    content_length_bytes: int
    court_corners: list[list[float]]
    schema_version: str = STREAM_SCHEMA_VERSION
    source_frame_start_index: int | None = None
    source_frame_count: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != STREAM_SCHEMA_VERSION:
            raise ValueError("unsupported stream schema version")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if self.source_start_time_sec < 0:
            raise ValueError("source_start_time_sec must be non-negative")
        if not 0 < self.duration_sec <= 10:
            raise ValueError("duration_sec must be greater than 0 and no more than 10")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be lowercase hexadecimal")
        validate_idempotency_key(self.idempotency_key)
        if self.content_type not in {"video/mp4", "video/iso.segment"}:
            raise ValueError("content_type must be video/mp4 or video/iso.segment")
        if self.content_length_bytes <= 0:
            raise ValueError("content_length_bytes must be positive")
        if not isinstance(self.court_corners, list) or len(self.court_corners) != 4:
            raise ValueError("court_corners must contain exactly four [x, y] points")
        for point in self.court_corners:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each court corner must be [x, y]")
        if (self.source_frame_start_index is None) != (self.source_frame_count is None):
            raise ValueError("source frame start and count must be provided together")
        if self.source_frame_start_index is not None:
            if not isinstance(self.source_frame_start_index, int) or isinstance(self.source_frame_start_index, bool) or self.source_frame_start_index < 0:
                raise ValueError("source_frame_start_index must be non-negative")
            if not isinstance(self.source_frame_count, int) or isinstance(self.source_frame_count, bool) or self.source_frame_count <= 0:
                raise ValueError("source_frame_count must be positive")

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        segment_index: int,
        source_start_time_sec: float,
        duration_sec: float,
        idempotency_prefix: str,
        content_type: str = "video/mp4",
        court_corners: list[list[float]] | None = None,
        source_frame_start_index: int | None = None,
        source_frame_count: int | None = None,
    ) -> "SegmentMetadata":
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return cls(
            segment_index=int(segment_index),
            source_start_time_sec=round(float(source_start_time_sec), 6),
            duration_sec=round(float(duration_sec), 6),
            sha256=digest.hexdigest(),
            idempotency_key=f"{idempotency_prefix}-segment-{segment_index:06d}",
            content_type=content_type,
            content_length_bytes=path.stat().st_size,
            court_corners=[[float(x), float(y)] for x, y in (court_corners or [])],
            source_frame_start_index=source_frame_start_index,
            source_frame_count=source_frame_count,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.source_frame_start_index is None:
            payload.pop("source_frame_start_index")
            payload.pop("source_frame_count")
        return payload


class DeliveryLedger:
    """JSON ledger that survives business-process restarts.

    Every mutation is written through a temporary file and atomically replaced.
    A segment becomes ``accepted`` only after a valid GPU receipt is observed.
    Unknown outcomes remain pending and are retried with the same idempotency key.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "ledger_version": LEDGER_VERSION,
                "schema_version": STREAM_SCHEMA_VERSION,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "create": None,
                "analysis_session_id": None,
                "segments": {},
                "completion": None,
            }
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("ledger_version") != LEDGER_VERSION:
            raise ValueError("unsupported delivery ledger version")
        return data

    def _save(self) -> None:
        self._data["updated_at"] = utc_now()
        # A fixed ``.tmp`` name is unsafe on Windows when a status request,
        # antivirus scanner, or a second recovery worker briefly observes the
        # ledger.  Use a per-save sibling and retry the atomic replacement for
        # a short bounded period.  The durable ledger remains all-or-nothing:
        # a failed replacement never turns an unconfirmed segment into an
        # accepted one.
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        try:
            for attempt in range(8):
                try:
                    os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            # After a successful replace this path no longer exists.  On an
            # exhausted retry, remove only our unique, uncommitted temp file.
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data, ensure_ascii=False))

    @property
    def analysis_session_id(self) -> Optional[str]:
        return self._data.get("analysis_session_id")

    def remember_create(self, request: dict[str, Any], idempotency_key: str) -> None:
        with self._lock:
            existing = self._data.get("create")
            candidate = {"request": request, "idempotency_key": idempotency_key}
            if existing and existing != candidate:
                raise ValueError("ledger already belongs to a different stream session request")
            self._data["create"] = candidate
            self._save()

    def remember_remote_session(self, response: dict[str, Any]) -> None:
        session_id = str(response.get("analysis_session_id") or "")
        if not session_id:
            raise ValueError("create response has no analysis_session_id")
        with self._lock:
            existing = self._data.get("analysis_session_id")
            if existing and existing != session_id:
                raise ValueError("idempotent create returned a different analysis_session_id")
            self._data["analysis_session_id"] = session_id
            self._data["create_response"] = response
            self._save()

    def register_segment(self, path: Path, metadata: SegmentMetadata) -> dict[str, Any]:
        path = Path(path).resolve()
        key = str(metadata.segment_index)
        candidate = {
            "path": str(path),
            "metadata": metadata.to_payload(),
            "status": "pending",
            "attempts": 0,
            "receipt": None,
            "last_error": None,
            "updated_at": utc_now(),
        }
        with self._lock:
            existing = self._data["segments"].get(key)
            if existing:
                old = existing["metadata"]
                if old["sha256"] != metadata.sha256 or old["content_length_bytes"] != metadata.content_length_bytes:
                    raise ValueError(f"segment {metadata.segment_index} conflicts with the durable ledger")
                return existing
            self._data["segments"][key] = candidate
            self._save()
            return candidate

    def mark_attempt(self, segment_index: int) -> None:
        with self._lock:
            record = self._data["segments"][str(segment_index)]
            record["attempts"] = int(record.get("attempts", 0)) + 1
            record["updated_at"] = utc_now()
            self._save()

    def mark_accepted(self, segment_index: int, response: dict[str, Any]) -> None:
        with self._lock:
            record = self._data["segments"][str(segment_index)]
            record["status"] = "accepted"
            record["receipt"] = response
            record["last_error"] = None
            record["updated_at"] = utc_now()
            self._save()

    def mark_failed(
        self,
        segment_index: int,
        message: str,
        *,
        terminal: bool,
        retryable: bool,
    ) -> None:
        with self._lock:
            record = self._data["segments"][str(segment_index)]
            if not terminal:
                record["status"] = "pending"
            elif retryable:
                # The configured retry window ended, but a later operator or
                # restarted worker may safely retry the same idempotent bytes.
                record["status"] = "retry_exhausted"
            else:
                record["status"] = "failed"
            record["last_error"] = str(message)
            record["updated_at"] = utc_now()
            self._save()

    def segment(self, segment_index: int) -> Optional[dict[str, Any]]:
        return self._data["segments"].get(str(segment_index))

    def pending_segments(self) -> list[dict[str, Any]]:
        with self._lock:
            records = [
                record
                for record in self._data["segments"].values()
                if record.get("status") in {"pending", "retry_exhausted"}
            ]
            return sorted(records, key=lambda record: int(record["metadata"]["segment_index"]))

    def remember_completion(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        with self._lock:
            self._data["completion"] = {
                "request": request,
                "response": response,
                "updated_at": utc_now(),
            }
            self._save()
