"""Reliable client for the frozen ``stream-session.v1`` GPU API."""

from __future__ import annotations

import hashlib
import http.client
import json
import mimetypes
import ssl
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .models import (
    DeliveryLedger,
    SegmentMetadata,
    StreamClientConfig,
    validate_idempotency_key,
)


class StreamAPIError(RuntimeError):
    """Structured transport or contract error returned by the GPU service."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: str = "transport_error",
        retryable: bool = False,
        response: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = bool(retryable)
        self.response = response or {}


class StreamTransport(Protocol):
    def request_json(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]: ...

    def upload_segment(
        self,
        session_id: str,
        segment_path: Path,
        metadata: SegmentMetadata,
    ) -> dict[str, Any]: ...


class UrllibStreamTransport:
    """Small stdlib HTTP transport with bounded-memory multipart uploads."""

    def __init__(self, config: StreamClientConfig) -> None:
        self.config = config
        self._parsed = urlparse(config.base_url.rstrip("/"))

    def _url(self, path: str) -> str:
        prefix = self._parsed.path.rstrip("/")
        return f"{self._parsed.scheme}://{self._parsed.netloc}{prefix}{path}"

    def request_json(self, method, path, *, body=None, headers=None):
        encoded = None
        request_headers = {
            "Accept": "application/json",
            "X-API-Key": self.config.api_key,
        }
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json; charset=utf-8"
        request_headers.update(headers or {})
        request = Request(self._url(path), data=encoded, headers=request_headers, method=method)
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                return _parse_response(response.status, response.read())
        except HTTPError as exc:
            raise _http_error(exc.code, exc.read()) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise StreamAPIError(str(exc), retryable=True) from exc

    def upload_segment(self, session_id, segment_path, metadata):
        segment_path = Path(segment_path)
        boundary = f"----GoodBadmintonStream{uuid.uuid4().hex}"
        metadata_bytes = json.dumps(
            metadata.to_payload(), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        prefix = _multipart_prefix(boundary, segment_path.name, metadata_bytes)
        suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
        content_length = len(prefix) + segment_path.stat().st_size + len(suffix)

        connection_cls = (
            http.client.HTTPSConnection
            if self._parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        kwargs: dict[str, Any] = {"timeout": self.config.timeout_seconds}
        if self._parsed.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        connection = connection_cls(self._parsed.hostname, self._parsed.port, **kwargs)
        endpoint = (
            self._parsed.path.rstrip("/")
            + f"/api/v1/stream-sessions/{session_id}/segments/{metadata.segment_index}"
        )
        try:
            connection.putrequest("POST", endpoint)
            connection.putheader("Accept", "application/json")
            connection.putheader("X-API-Key", self.config.api_key)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            connection.send(prefix)
            with segment_path.open("rb") as source:
                while chunk := source.read(self.config.upload_chunk_bytes):
                    connection.send(chunk)
            connection.send(suffix)
            response = connection.getresponse()
            payload = response.read()
            if not 200 <= response.status < 300:
                raise _http_error(response.status, payload)
            return _parse_response(response.status, payload)
        except StreamAPIError:
            raise
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            # The server may already have persisted the segment.  The caller
            # retries with the same index/hash/idempotency key and receives the
            # original receipt instead of guessing whether the upload landed.
            raise StreamAPIError(str(exc), retryable=True) from exc
        finally:
            connection.close()


class StreamSessionClient:
    """Business-side session lifecycle with durable receipt-aware retries."""

    def __init__(
        self,
        config: StreamClientConfig,
        ledger: DeliveryLedger,
        *,
        transport: Optional[StreamTransport] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.transport = transport or UrllibStreamTransport(config)
        self.sleep_fn = sleep_fn

    def create_session(
        self,
        request: dict[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        _assert_anonymous_create_request(request)
        if request.get("schema_version") != "stream-session.v1":
            raise ValueError("create request must use stream-session.v1")
        idempotency_key = validate_idempotency_key(
            idempotency_key,
            "create idempotency_key",
        )
        self.ledger.remember_create(request, idempotency_key)
        if self.ledger.analysis_session_id:
            snapshot = self.ledger.snapshot()
            return snapshot.get("create_response") or {
                "analysis_session_id": self.ledger.analysis_session_id,
                "recovered_from_ledger": True,
            }

        response = self._retry(
            lambda: self.transport.request_json(
                "POST",
                "/api/v1/stream-sessions",
                body=request,
                headers={"X-Idempotency-Key": idempotency_key},
            )
        )
        self.ledger.remember_remote_session(response)
        return response

    def submit_segment(
        self,
        segment_path: Path,
        metadata: SegmentMetadata,
    ) -> dict[str, Any]:
        session_id = self._session_id()
        segment_path = Path(segment_path)
        if not segment_path.is_file():
            raise FileNotFoundError(segment_path)
        _validate_segment_file(segment_path, metadata)
        record = self.ledger.register_segment(segment_path, metadata)
        if record.get("status") == "accepted":
            return record["receipt"]

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_attempts):
            self.ledger.mark_attempt(metadata.segment_index)
            try:
                response = self.transport.upload_segment(session_id, segment_path, metadata)
                _validate_receipt(session_id, metadata, response)
                self.ledger.mark_accepted(metadata.segment_index, response)
                return response
            except Exception as exc:  # retry policy is centralized below
                last_error = exc
                retryable = _is_retryable(exc)
                terminal = (not retryable) or attempt + 1 >= self.config.max_attempts
                self.ledger.mark_failed(
                    metadata.segment_index,
                    str(exc),
                    terminal=terminal,
                    retryable=retryable,
                )
                if terminal:
                    raise
                self.sleep_fn(self._backoff(attempt))
        raise RuntimeError("segment retry loop exited unexpectedly") from last_error

    def deliver_pending(self) -> list[dict[str, Any]]:
        receipts = []
        for record in self.ledger.pending_segments():
            metadata = SegmentMetadata(**record["metadata"])
            receipts.append(self.submit_segment(Path(record["path"]), metadata))
        return receipts

    def get_status(self) -> dict[str, Any]:
        return self._retry(
            lambda: self.transport.request_json(
                "GET", f"/api/v1/stream-sessions/{self._session_id()}"
            )
        )

    def get_trace(self) -> dict[str, Any]:
        """Fetch the GPU-side durable execution trace for this session.

        The trace is deliberately a separate resource from the status payload:
        status remains small enough for frequent polling, while the trace keeps
        per-segment timing and execution-topology evidence for performance gates.
        """

        return self._retry(
            lambda: self.transport.request_json(
                "GET", f"/api/v1/stream-sessions/{self._session_id()}/trace"
            )
        )

    def wait_for_terminal(
        self,
        *,
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Poll durable GPU status without hiding failure or rebuild states."""

        terminal = {
            "finalized",
            "partial",
            "failed",
            "cancelled",
            "interrupted_needs_rebuild",
        }
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            status = self.get_status()
            if status.get("status") in terminal:
                return status
            self.sleep_fn(max(0.01, float(poll_interval_seconds)))
        raise TimeoutError(
            f"stream session {self._session_id()} did not reach a terminal state "
            f"within {timeout_seconds:g} seconds"
        )

    def read_events(self, *, cursor: Optional[str] = None, limit: int = 100) -> dict[str, Any]:
        query = urlencode({key: value for key, value in {"cursor": cursor, "limit": limit}.items() if value is not None})
        path = f"/api/v1/stream-sessions/{self._session_id()}/events?{query}"
        return self._retry(lambda: self.transport.request_json("GET", path))

    def complete(self, expected_last_segment_index: int, *, allow_partial: bool = False) -> dict[str, Any]:
        request = {
            "schema_version": "stream-session.v1",
            "expected_last_segment_index": int(expected_last_segment_index),
            "allow_partial": bool(allow_partial),
        }
        response = self._retry(
            lambda: self.transport.request_json(
                "POST",
                f"/api/v1/stream-sessions/{self._session_id()}/complete",
                body=request,
            )
        )
        self.ledger.remember_completion(request, response)
        return response

    def cancel(self) -> dict[str, Any]:
        return self._retry(
            lambda: self.transport.request_json(
                "DELETE", f"/api/v1/stream-sessions/{self._session_id()}"
            )
        )

    def _session_id(self) -> str:
        session_id = self.ledger.analysis_session_id
        if not session_id:
            raise RuntimeError("create_session must succeed before this operation")
        return session_id

    def _retry(self, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_attempts):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt + 1 >= self.config.max_attempts:
                    raise
                self.sleep_fn(self._backoff(attempt))
        raise RuntimeError("request retry loop exited unexpectedly") from last_error

    def _backoff(self, attempt: int) -> float:
        return min(
            self.config.initial_backoff_seconds * (2**attempt),
            self.config.max_backoff_seconds,
        )


def _assert_anonymous_create_request(request: dict[str, Any]) -> None:
    forbidden = {
        "user",
        "user_id",
        "check_id",
        "player",
        "player_id",
        "player_name",
        "team",
        "team_id",
        "score",
        "winner",
        "ranking",
    }

    def visit(value: Any, path: str = "request") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    raise ValueError(f"business identity field must not be sent to GPU service: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(request)


def _validate_segment_file(path: Path, metadata: SegmentMetadata) -> None:
    if path.stat().st_size != metadata.content_length_bytes:
        raise ValueError("segment byte length changed after metadata was created")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != metadata.sha256:
        raise ValueError("segment content changed after metadata was created")


def _validate_receipt(session_id: str, metadata: SegmentMetadata, response: dict[str, Any]) -> None:
    receipt = response.get("receipt") or {}
    if response.get("analysis_session_id") != session_id:
        raise StreamAPIError("segment receipt belongs to another session", code="invalid_receipt")
    if int(response.get("segment_index", -1)) != metadata.segment_index:
        raise StreamAPIError("segment receipt index does not match", code="invalid_receipt")
    if not receipt.get("accepted"):
        raise StreamAPIError("segment receipt did not confirm durable acceptance", code="invalid_receipt")
    if receipt.get("sha256") != metadata.sha256:
        raise StreamAPIError("segment receipt hash does not match", code="invalid_receipt")
    if int(receipt.get("content_length_bytes", -1)) != metadata.content_length_bytes:
        raise StreamAPIError("segment receipt byte length does not match", code="invalid_receipt")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, StreamAPIError):
        return exc.retryable or exc.status_code in {408, 425, 429} or (
            exc.status_code is not None and exc.status_code >= 500
        )
    return isinstance(exc, (TimeoutError, OSError, http.client.HTTPException))


def _multipart_prefix(boundary: str, filename: str, metadata_bytes: bytes) -> bytes:
    media_type = mimetypes.guess_type(filename)[0] or "video/mp4"
    return b"".join(
        [
            f"--{boundary}\r\n".encode("ascii"),
            b'Content-Disposition: form-data; name="metadata"\r\n',
            b"Content-Type: application/json; charset=utf-8\r\n\r\n",
            metadata_bytes,
            b"\r\n",
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="segment"; filename="{Path(filename).name}"\r\n'.encode(
                "utf-8"
            ),
            f"Content-Type: {media_type}\r\n\r\n".encode("ascii"),
        ]
    )


def _parse_response(status: int, payload: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8")) if payload else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StreamAPIError(
            f"GPU service returned invalid JSON with HTTP {status}",
            status_code=status,
            retryable=status >= 500,
        ) from exc
    if not isinstance(parsed, dict):
        raise StreamAPIError("GPU service response must be a JSON object", status_code=status)
    return parsed


def _http_error(status: int, payload: bytes) -> StreamAPIError:
    try:
        response = json.loads(payload.decode("utf-8")) if payload else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        response = {}
    error = response.get("error") if isinstance(response, dict) else {}
    error = error if isinstance(error, dict) else {}
    return StreamAPIError(
        error.get("message") or f"GPU stream API returned HTTP {status}",
        status_code=status,
        code=error.get("code") or "http_error",
        retryable=bool(error.get("retryable")) or status >= 500,
        response=response if isinstance(response, dict) else {},
    )
