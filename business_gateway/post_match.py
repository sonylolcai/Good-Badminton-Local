"""Business-owned post-match derivation from immutable analysis artifacts.

The GPU service ends after producing anonymous measurements. This module is
the explicit boundary where a business worker may add user-entered body data,
derive movement/energy metrics, and optionally request one match-level LLM
report. Re-running it overwrites the same versioned files, so profile edits do
not require video inference to run again.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from business_gateway.metrics.movement import generate_movement_metrics
from business_gateway.report.performance import generate_performance_report


def generate_business_interpretation(
    output_dir,
    *,
    body_profiles_path=None,
    include_performance_report=True,
):
    """Derive business outputs using only the public analysis artifacts.

    Required inputs are ``detections.jsonl``, ``spatial_match_summary.json``
    and ``metadata.json``. No video, model, OpenCV object, or GPU runtime is
    accepted here, which keeps the service boundary mechanically testable.
    """
    output_dir = Path(output_dir)
    detections_path = output_dir / "detections.jsonl"
    spatial_summary_path = output_dir / "spatial_match_summary.json"
    metadata_path = output_dir / "metadata.json"
    missing = [
        path.name
        for path in (detections_path, spatial_summary_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Business interpretation requires analysis artifacts: " + ", ".join(missing)
        )

    started = time.monotonic()
    metrics = generate_movement_metrics(
        output_dir=output_dir,
        detections_path=detections_path,
        spatial_summary_path=spatial_summary_path,
        metadata_path=metadata_path,
        body_profiles_path=body_profiles_path,
    )
    report = None
    if include_performance_report:
        report = generate_performance_report(
            output_dir=output_dir,
            metadata_path=metadata_path,
            spatial_summary_path=spatial_summary_path,
        )

    result = {
        "schema_version": "1.0",
        "status": "succeeded",
        "owner": "business_gateway",
        "input_contract": {
            "detections": str(detections_path),
            "spatial_match_summary": str(spatial_summary_path),
            "metadata": str(metadata_path),
            "body_profiles": str(body_profiles_path) if body_profiles_path else None,
        },
        "movement_metrics": metrics,
        "performance_report": report,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    manifest_path = output_dir / "derived" / "business_interpretation_manifest_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, result)

    metadata = _read_json(metadata_path)
    metadata["business_interpretation"] = {
        "status": result["status"],
        "owner": result["owner"],
        "manifest_path": str(manifest_path),
        "movement_metrics_path": metrics.get("metrics_path"),
        "performance_report_status": (report or {}).get("status"),
        "performance_report_path": (report or {}).get("report_path"),
    }
    _write_json(metadata_path, metadata)
    return {
        **result,
        "manifest_path": str(manifest_path),
        "movement_metrics_path": metrics.get("metrics_path"),
        "performance_report_path": (report or {}).get("report_path"),
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
