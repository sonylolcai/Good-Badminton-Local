"""Release identity exposed by GPU health endpoints."""

import hashlib
import json
import os
from pathlib import Path


def release_identity():
    manifest_path = Path(os.environ.get("GOOD_GPU_MODEL_MANIFEST", "weights/manifest.json"))
    model_manifest = {"status": "missing", "path": str(manifest_path)}
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        model_manifest = {
            "status": "available",
            "path": str(manifest_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            model_manifest["schema"] = manifest.get("schema")
    return {
        "build_sha": os.environ.get("GOOD_GPU_BUILD_SHA", "unknown"),
        "model_manifest": model_manifest,
    }
