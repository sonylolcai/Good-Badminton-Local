"""Atomic local storage for drafts, immutable manifests and analysis runs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import contract_sha256, validate_contract
from .models import (
    MANIFEST_ID_FIELDS,
    canonical_json_bytes,
    canonical_json_sha256,
    manifest_identity,
    safe_storage_id,
    validate_manifest,
)


class LocalManifestStore:
    """A local-first store with write-once published objects and Run inputs."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_draft(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_manifest(kind, payload)
        identity = manifest_identity(kind, normalized)
        path = self.root / "drafts" / kind / f"{identity}.json"
        return self._write(path, normalized, immutable=False)

    def publish_manifest(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_manifest(kind, payload)
        identity = manifest_identity(kind, normalized)
        path = self.root / "registry" / kind / f"{identity}.json"
        return self._write(path, normalized, immutable=True)

    def read_manifest(self, kind: str, identity: str, *, draft: bool = False) -> dict[str, Any]:
        safe_storage_id(identity)
        area = "drafts" if draft else "registry"
        path = self.root / area / kind / f"{identity}.json"
        payload = self._read(path)
        return validate_manifest(kind, payload)

    def list_manifest_ids(self, kind: str, *, draft: bool = False) -> list[str]:
        if kind not in MANIFEST_ID_FIELDS:
            raise ValueError(f"unsupported manifest kind: {kind}")
        area = "drafts" if draft else "registry"
        root = self.root / area / kind
        if not root.is_dir():
            return []
        return sorted((path.stem for path in root.glob("*.json") if path.is_file()), reverse=True)

    def create_run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_contract(task)
        if normalized["contract_type"] != "analysis_task":
            raise ValueError("create_run requires an analysis_task contract")
        run_id = safe_storage_id(normalized["run_id"], "run_id")
        path = self.root / "runs" / run_id / "input.json"
        return self._write(path, normalized, immutable=True, digest=contract_sha256(normalized))

    def write_run_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_contract(result)
        if normalized["contract_type"] != "gpu_analysis_result":
            raise ValueError("write_run_result requires a gpu_analysis_result contract")
        run_id = safe_storage_id(normalized["run_id"], "run_id")
        task = self._read(self.root / "runs" / run_id / "input.json")
        for field in (
            "run_id",
            "case_id",
            "dataset_version",
            "code_fingerprint",
            "model_fingerprint",
            "parameters_fingerprint",
            "input_sha256",
        ):
            if normalized[field] != task[field]:
                raise ValueError(f"GPU result {field} does not match the frozen Run input")
        path = self.root / "runs" / run_id / "result.json"
        return self._write(path, normalized, immutable=True, digest=contract_sha256(normalized))

    def read_run_input(self, run_id: str) -> dict[str, Any]:
        run_id = safe_storage_id(run_id, "run_id")
        payload = self._read(self.root / "runs" / run_id / "input.json")
        return validate_contract(payload)

    def read_run_result(self, run_id: str) -> dict[str, Any]:
        run_id = safe_storage_id(run_id, "run_id")
        payload = self._read(self.root / "runs" / run_id / "result.json")
        return validate_contract(payload)

    def list_run_ids(self) -> list[str]:
        runs = self.root / "runs"
        if not runs.is_dir():
            return []
        return sorted(
            (path.name for path in runs.iterdir() if path.is_dir() and (path / "input.json").is_file()),
            reverse=True,
        )

    def publish_run_result(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_contract(reference)
        if normalized["contract_type"] != "published_result_ref":
            raise ValueError("publish_run_result requires a published_result_ref contract")
        run_id = safe_storage_id(normalized["run_id"], "run_id")
        result = self._read(self.root / "runs" / run_id / "result.json")
        if result.get("status") != "succeeded":
            raise ValueError("only a succeeded GPU result can be published")
        if contract_sha256(result) != normalized["source_result_sha256"]:
            raise ValueError("source_result_sha256 does not match the stored GPU result")
        path = self.root / "runs" / run_id / "published.json"
        return self._write(path, normalized, immutable=True, digest=contract_sha256(normalized))

    def promote_baseline(self, alias: str, run_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        alias = safe_storage_id(alias, "alias")
        run_id = safe_storage_id(run_id, "run_id")
        actor = safe_storage_id(actor, "actor")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason is required")
        published = self._read(self.root / "runs" / run_id / "published.json")
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = {
            "schema_version": "analysis-platform.v1",
            "kind": "baseline_alias",
            "alias": alias,
            "run_id": run_id,
            "published_result_sha256": contract_sha256(published),
            "updated_at": now,
            "updated_by": actor,
            "reason": reason.strip(),
        }
        path = self.root / "baselines" / f"{alias}.json"
        stored = self._write(path, payload, immutable=False)
        self._append_audit({"event": "baseline_promoted", **payload})
        return stored

    def _write(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        immutable: bool,
        digest: str | None = None,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        digest = digest or canonical_json_sha256(normalized)
        envelope = {"content_sha256": digest, "payload": normalized}
        if path.exists():
            existing = self._read_envelope(path)
            if immutable and existing["content_sha256"] != digest:
                raise ValueError(f"immutable object already exists with different content: {path}")
            if immutable:
                return {"path": str(path), "content_sha256": digest, "created": False}
        self._atomic_write(path, json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
        return {"path": str(path), "content_sha256": digest, "created": True}

    def _read(self, path: Path) -> dict[str, Any]:
        return dict(self._read_envelope(path)["payload"])

    def _read_envelope(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid stored JSON: {path}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"content_sha256", "payload"}:
            raise ValueError(f"invalid storage envelope: {path}")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError(f"stored payload is not an object: {path}")
        expected = envelope["content_sha256"]
        actual = (
            contract_sha256(payload)
            if payload.get("schema_version") == "analysis-boundary.v1"
            else canonical_json_sha256(payload)
        )
        if expected != actual:
            raise ValueError(f"stored content SHA-256 mismatch: {path}")
        return envelope

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _append_audit(self, event: Mapping[str, Any]) -> None:
        # ponytail: single-process append is enough for the local MVP; add a
        # file lock when multiple platform processes can promote baselines.
        path = self.root / "audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
