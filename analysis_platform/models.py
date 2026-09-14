"""Minimal manifest validation shared by the local platform store."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping


SCHEMA_VERSION = "analysis-platform.v1"

MANIFEST_ID_FIELDS = {
    "dataset": "dataset_id",
    "dataset_version": "dataset_version_id",
    "case": "case_id",
    "annotation_version": "annotation_version_id",
    "experiment": "experiment_id",
    "metric_result": "metric_result_id",
    "artifact": "artifact_id",
}

_REQUIRED_FIELDS = {
    "dataset": {"dataset_id", "name", "purpose", "created_at"},
    "dataset_version": {
        "dataset_version_id",
        "dataset_id",
        "version",
        "purpose",
        "split",
        "created_at",
        "created_by",
        "cases",
        "license_and_consent_status",
    },
    "case": {"case_id", "video_asset", "annotation_refs", "slices", "status"},
    "annotation_version": {
        "annotation_version_id",
        "case_id",
        "version",
        "source",
        "review_status",
        "assets",
        "created_at",
    },
    "experiment": {
        "experiment_id",
        "name",
        "hypothesis",
        "baseline_run_id",
        "success_criteria",
        "created_at",
    },
    "metric_result": {
        "metric_result_id",
        "run_id",
        "metric_key",
        "metric_definition_version",
        "scope",
        "scope_id",
        "value",
        "unit",
        "sample_count",
        "eligible_sample_count",
        "status",
    },
    "artifact": {"artifact_id", "run_id", "kind", "path", "sha256", "size_bytes"},
}

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,159}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SPLITS = {"dev", "smoke", "regression", "challenge", "holdout"}
_CASE_STATES = {"draft", "ready", "qualified", "deprecated", "restricted"}
_REVIEW_STATES = {"unlabeled", "preliminary_reviewed", "human_reviewed", "approved"}
_METRIC_SCOPES = {"run", "slice", "case"}
_METRIC_STATUSES = {"valid", "preliminary", "insufficient_data", "not_applicable"}


def validate_manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the stable identity and essential fields of one platform object."""
    if kind not in MANIFEST_ID_FIELDS:
        raise ValueError(f"unsupported manifest kind: {kind}")
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != kind:
        raise ValueError(f"manifest kind must be {kind}")
    missing = _REQUIRED_FIELDS[kind].difference(payload)
    if missing:
        raise ValueError(f"{kind} manifest is missing required keys: {sorted(missing)}")

    identity = payload.get(MANIFEST_ID_FIELDS[kind])
    _safe_id(identity, MANIFEST_ID_FIELDS[kind])
    _validate_kind_fields(kind, payload)
    canonical_json_bytes(payload)
    return dict(payload)


def manifest_identity(kind: str, payload: Mapping[str, Any]) -> str:
    validate_manifest(kind, payload)
    return str(payload[MANIFEST_ID_FIELDS[kind]])


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest must contain only finite JSON values") from exc
    return rendered.encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def safe_storage_id(value: Any, field: str = "id") -> str:
    return _safe_id(value, field)


def relative_storage_path(value: Any, field: str = "path") -> str:
    text = _text(value, field, maximum=500).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ":" in path.parts[0] or ".." in path.parts:
        raise ValueError(f"{field} must be a relative path without '..'")
    if str(path) in {"", "."}:
        raise ValueError(f"{field} must name a file")
    return str(path)


def _validate_kind_fields(kind: str, payload: Mapping[str, Any]) -> None:
    if kind in {"dataset", "dataset_version", "annotation_version", "experiment"}:
        _timestamp(payload.get("created_at"), "created_at")
    if kind == "dataset_version":
        _safe_id(payload.get("dataset_id"), "dataset_id")
        _text(payload.get("version"), "version")
        if payload.get("split") not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}")
        if not isinstance(payload.get("cases"), list) or not payload["cases"]:
            raise ValueError("dataset_version cases must be a non-empty array")
        for index, case_ref in enumerate(payload["cases"]):
            if not isinstance(case_ref, Mapping):
                raise ValueError(f"cases[{index}] must be an object")
            _safe_id(case_ref.get("case_id"), f"cases[{index}].case_id")
            _safe_id(
                case_ref.get("annotation_version_id"),
                f"cases[{index}].annotation_version_id",
            )
    elif kind == "case":
        if payload.get("status") not in _CASE_STATES:
            raise ValueError(f"case status must be one of {sorted(_CASE_STATES)}")
        video = payload.get("video_asset")
        if not isinstance(video, Mapping):
            raise ValueError("video_asset must be an object")
        relative_storage_path(video.get("path"), "video_asset.path")
        _sha256(video.get("sha256"), "video_asset.sha256")
        if not isinstance(video.get("size_bytes"), int) or video["size_bytes"] < 0:
            raise ValueError("video_asset.size_bytes must be a non-negative integer")
        if not isinstance(payload.get("annotation_refs"), list):
            raise ValueError("annotation_refs must be an array")
        if not isinstance(payload.get("slices"), Mapping):
            raise ValueError("slices must be an object")
    elif kind == "annotation_version":
        _safe_id(payload.get("case_id"), "case_id")
        if payload.get("source") not in {"human", "machine_suggestion"}:
            raise ValueError("annotation source must be human or machine_suggestion")
        if payload.get("review_status") not in _REVIEW_STATES:
            raise ValueError(f"review_status must be one of {sorted(_REVIEW_STATES)}")
        if not isinstance(payload.get("assets"), list) or not payload["assets"]:
            raise ValueError("annotation assets must be a non-empty array")
    elif kind == "experiment":
        baseline = payload.get("baseline_run_id")
        if baseline is not None:
            _safe_id(baseline, "baseline_run_id")
        if not isinstance(payload.get("success_criteria"), list):
            raise ValueError("success_criteria must be an array")
    elif kind == "metric_result":
        _safe_id(payload.get("run_id"), "run_id")
        _text(payload.get("metric_key"), "metric_key")
        _text(payload.get("metric_definition_version"), "metric_definition_version")
        if payload.get("scope") not in _METRIC_SCOPES:
            raise ValueError(f"metric scope must be one of {sorted(_METRIC_SCOPES)}")
        _text(payload.get("scope_id"), "scope_id")
        _text(payload.get("unit"), "unit")
        for field in ("sample_count", "eligible_sample_count"):
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if payload["eligible_sample_count"] > payload["sample_count"]:
            raise ValueError("eligible_sample_count cannot exceed sample_count")
        if payload.get("status") not in _METRIC_STATUSES:
            raise ValueError(f"metric status must be one of {sorted(_METRIC_STATUSES)}")
    elif kind == "artifact":
        _safe_id(payload.get("run_id"), "run_id")
        relative_storage_path(payload.get("path"), "artifact.path")
        _sha256(payload.get("sha256"), "artifact.sha256")
        if not isinstance(payload.get("size_bytes"), int) or payload["size_bytes"] < 0:
            raise ValueError("artifact.size_bytes must be a non-negative integer")


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a 2-160 character safe identifier")
    return value


def _text(value: Any, field: str, maximum: int = 300) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty string of at most {maximum} characters")
    return value


def _sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _timestamp(value: Any, field: str) -> None:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
