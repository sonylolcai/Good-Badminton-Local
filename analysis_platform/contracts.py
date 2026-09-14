"""Frozen v1 contracts at the analysis-platform service boundary.

The contracts are intentionally small and use only the Python standard
library.  They describe immutable research inputs, terminal GPU artifacts and
the reference a business service may consume after publication.  They do not
carry player identity, score or venue operations data.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping


SCHEMA_VERSION = "analysis-boundary.v1"

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_CODE_FINGERPRINT = re.compile(
    r"^git:[a-f0-9]{40}(?:\+dirty:[a-f0-9]{64})?$"
)
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_TERMINAL_GPU_STATUSES = {"succeeded", "partial", "failed", "cancelled"}
_FORBIDDEN_BUSINESS_KEYS = {
    "person_id",
    "player_id",
    "player_name",
    "user_id",
    "team_id",
    "score",
    "venue_id",
    "tenant_id",
}

_COMMON_FIELDS = {
    "schema_version",
    "contract_type",
    "run_id",
    "case_id",
    "dataset_version",
    "code_fingerprint",
    "model_fingerprint",
    "parameters_fingerprint",
    "input_sha256",
}


def validate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one v1 boundary object and return a detached copy."""
    if not isinstance(payload, Mapping):
        raise ValueError("contract must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    contract_type = payload.get("contract_type")
    if contract_type == "analysis_task":
        return _validate_analysis_task(payload)
    if contract_type == "gpu_analysis_result":
        return _validate_gpu_analysis_result(payload)
    if contract_type == "published_result_ref":
        return _validate_published_result_ref(payload)
    raise ValueError("contract_type must be analysis_task, gpu_analysis_result, or published_result_ref")


def contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable identity used to detect edits to a frozen contract."""
    normalized = validate_contract(payload)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_contract_sha256(payload: Mapping[str, Any], expected_sha256: str) -> None:
    """Raise when a stored contract no longer matches its published identity."""
    _sha256(expected_sha256, "expected_sha256")
    actual = contract_sha256(payload)
    if not hmac.compare_digest(actual, expected_sha256):
        raise ValueError(f"contract SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def _validate_analysis_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = _COMMON_FIELDS | {"input_path", "parameters"}
    _common(payload, allowed)
    _relative_path(payload.get("input_path"), "input_path")

    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a JSON object")
    _reject_business_data(parameters, "parameters")
    expected = _canonical_value_sha256(parameters)
    if not hmac.compare_digest(expected, payload["parameters_fingerprint"]):
        raise ValueError("parameters_fingerprint does not match parameters")
    return copy.deepcopy(dict(payload))


def _validate_gpu_analysis_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = _COMMON_FIELDS | {"status", "artifacts", "error"}
    _common(payload, allowed)

    status = payload.get("status")
    if status not in _TERMINAL_GPU_STATUSES:
        raise ValueError(f"status must be one of {sorted(_TERMINAL_GPU_STATUSES)}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be an array")
    seen_paths: set[str] = set()
    for index, artifact in enumerate(artifacts):
        field = f"artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{field} must be an object")
        _reject_extra_keys(artifact, {"kind", "path", "sha256"}, field)
        _text(artifact.get("kind"), f"{field}.kind")
        path = _relative_path(artifact.get("path"), f"{field}.path")
        if path in seen_paths:
            raise ValueError(f"duplicate artifact path: {path}")
        seen_paths.add(path)
        _sha256(artifact.get("sha256"), f"{field}.sha256")

    error = payload.get("error")
    if status == "succeeded":
        if not artifacts:
            raise ValueError("a succeeded result must contain at least one artifact")
        if error is not None:
            raise ValueError("a succeeded result must have error=null")
    else:
        _validate_error(error)
        if status == "partial" and not artifacts:
            raise ValueError("a partial result must contain at least one artifact")
    return copy.deepcopy(dict(payload))


def _validate_published_result_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = _COMMON_FIELDS | {
        "status",
        "result_manifest_path",
        "result_manifest_sha256",
        "source_result_sha256",
        "published_at",
    }
    _common(payload, allowed)
    if payload.get("status") != "published":
        raise ValueError("published result status must be published")
    _relative_path(payload.get("result_manifest_path"), "result_manifest_path")
    _sha256(payload.get("result_manifest_sha256"), "result_manifest_sha256")
    _sha256(payload.get("source_result_sha256"), "source_result_sha256")
    _utc_timestamp(payload.get("published_at"), "published_at")
    return copy.deepcopy(dict(payload))


def _common(payload: Mapping[str, Any], allowed: set[str]) -> None:
    _reject_extra_keys(payload, allowed, "contract")
    missing = allowed.difference(payload)
    if missing:
        raise ValueError(f"contract is missing required keys: {sorted(missing)}")
    _text(payload.get("run_id"), "run_id")
    _text(payload.get("case_id"), "case_id")
    _text(payload.get("dataset_version"), "dataset_version")
    code_fingerprint = payload.get("code_fingerprint")
    if not isinstance(code_fingerprint, str) or not _CODE_FINGERPRINT.fullmatch(code_fingerprint):
        raise ValueError("code_fingerprint must be git:<40 hex> with optional +dirty:<64 hex>")
    _sha256(payload.get("model_fingerprint"), "model_fingerprint")
    _sha256(payload.get("parameters_fingerprint"), "parameters_fingerprint")
    _sha256(payload.get("input_sha256"), "input_sha256")


def _validate_error(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("a non-success result must contain an error object")
    _reject_extra_keys(value, {"code", "message", "retryable"}, "error")
    if set(value) != {"code", "message", "retryable"}:
        raise ValueError("error requires code, message, and retryable")
    if not isinstance(value.get("code"), str) or not _ERROR_CODE.fullmatch(value["code"]):
        raise ValueError("error.code must be a lowercase snake_case identifier")
    _text(value.get("message"), "error.message", maximum=2000)
    if not isinstance(value.get("retryable"), bool):
        raise ValueError("error.retryable must be a JSON boolean")


def _reject_business_data(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_BUSINESS_KEYS:
                raise ValueError(f"{field} must not contain business identity field {key!r}")
            _reject_business_data(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_business_data(child, f"{field}[{index}]")


def _canonical_value_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters must contain only finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(value: Any, field: str) -> str:
    text = _text(value, field, maximum=500).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ":" in path.parts[0] or ".." in path.parts:
        raise ValueError(f"{field} must be a relative path without '..'")
    if str(path) in {"", "."}:
        raise ValueError(f"{field} must name a file")
    return str(path)


def _text(value: Any, field: str, *, maximum: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty string of at most {maximum} characters")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _utc_timestamp(value: Any, field: str) -> None:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")


def _reject_extra_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    extras = set(value).difference(allowed)
    if extras:
        raise ValueError(f"{field} contains unsupported keys: {sorted(extras)}")
