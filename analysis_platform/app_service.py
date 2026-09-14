"""Application operations shared by the platform CLI and future UI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from badminton_analysis.cancellation import AnalysisCancelled

from .compare import compare_reports
from .contracts import contract_sha256, validate_contract
from .evaluators import execute_evaluator
from .gates import validate_metric_rows
from .models import canonical_json_sha256
from .store import LocalManifestStore


def publish_dataset_version(
    store_root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return LocalManifestStore(store_root).publish_manifest("dataset_version", manifest)


def save_manifest_draft(
    store_root: str | Path,
    kind: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return LocalManifestStore(store_root).save_draft(kind, manifest)


def publish_manifest(
    store_root: str | Path,
    kind: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return LocalManifestStore(store_root).publish_manifest(kind, manifest)


def list_manifests(
    store_root: str | Path,
    kind: str,
    *,
    draft: bool = False,
) -> list[dict[str, Any]]:
    store = LocalManifestStore(store_root)
    return [store.read_manifest(kind, identity, draft=draft) for identity in store.list_manifest_ids(kind, draft=draft)]


def create_run(store_root: str | Path, task: Mapping[str, Any]) -> dict[str, Any]:
    return LocalManifestStore(store_root).create_run(task)


def verify_published_manifest(
    store_root: str | Path,
    kind: str,
    identity: str,
) -> dict[str, Any]:
    payload = LocalManifestStore(store_root).read_manifest(kind, identity)
    return {
        "kind": kind,
        "identity": identity,
        "content_sha256": canonical_json_sha256(payload),
        "valid": True,
    }


def list_run_summaries(store_root: str | Path) -> list[dict[str, Any]]:
    store = LocalManifestStore(store_root)
    summaries = []
    for run_id in store.list_run_ids():
        task = store.read_run_input(run_id)
        try:
            result = store.read_run_result(run_id)
        except FileNotFoundError:
            result = None
        summaries.append(
            {
                "run_id": run_id,
                "case_id": task["case_id"],
                "dataset_version": task["dataset_version"],
                "status": (result or {}).get("status", "created"),
                "error": ((result or {}).get("error") or {}).get("message"),
                "run_report": str(store.root / "runs" / run_id / "artifacts" / "run-report.json"),
            }
        )
    return summaries


def get_run_detail(store_root: str | Path, run_id: str) -> dict[str, Any]:
    store = LocalManifestStore(store_root)
    task = store.read_run_input(run_id)
    try:
        result = store.read_run_result(run_id)
    except FileNotFoundError:
        result = None
    artifacts = list((result or {}).get("artifacts") or [])
    for artifact_id in store.list_manifest_ids("artifact"):
        manifest = store.read_manifest("artifact", artifact_id)
        if manifest["run_id"] == run_id:
            artifacts.append(manifest)
    files = []
    seen = set()
    for artifact in artifacts:
        if artifact["path"] in seen:
            continue
        seen.add(artifact["path"])
        path = (store.root / artifact["path"]).resolve()
        try:
            path.relative_to(store.root)
        except ValueError as exc:
            raise ValueError("Run artifact escaped the platform store") from exc
        if not path.is_file() or _sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Run artifact is missing or changed: {artifact['path']}")
        files.append(str(path))
    return {"task": task, "result": result, "files": files}


def clone_run_task(store_root: str | Path, source_run_id: str, new_run_id: str) -> dict[str, Any]:
    task = LocalManifestStore(store_root).read_run_input(source_run_id)
    cloned = dict(task)
    cloned["run_id"] = new_run_id
    return validate_contract(cloned)


def compare_run_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gate_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return compare_reports(baseline, candidate, gate_profile=gate_profile)


def execute_uploaded_run(
    store_root: str | Path,
    dataset_version: Mapping[str, Any],
    task: Mapping[str, Any],
    video_path: str | Path,
    template_path: str | Path,
    corners: Sequence[Sequence[float]],
    evaluator_name: str,
    evaluator_arguments: Sequence[str],
    *,
    processing_target: str = "local",
    gpu_base_url: str | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    baseline_report: Mapping[str, Any] | None = None,
    gate_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage uploaded files by content hash, then execute one reproducible Run."""
    root = Path(store_root).resolve()
    video = Path(video_path).resolve()
    template = Path(template_path).resolve()
    staged_video = _stage_input(root, video, "video")
    staged_template = _stage_input(root, template, "template")
    normalized_corners = _corners(list(corners))

    prepared_task = dict(task)
    parameters = dict(prepared_task.get("parameters") or {})
    parameters["court_corners"] = [list(point) for point in normalized_corners]
    parameters["template_sha256"] = _sha256_file(staged_template)
    prepared_task["parameters"] = parameters
    prepared_task["parameters_fingerprint"] = canonical_json_sha256(parameters)
    prepared_task["input_path"] = staged_video.relative_to(root).as_posix()
    prepared_task["input_sha256"] = _sha256_file(staged_video)
    if processing_target not in {"local", "remote_gpu"}:
        raise ValueError("processing_target must be local or remote_gpu")
    config = {
        "template_path": staged_template.relative_to(root).as_posix(),
        "corners": parameters["court_corners"],
        "template_sha256": parameters["template_sha256"],
        "hardware_fingerprint": (
            canonical_json_sha256(
                {
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "system": platform.system(),
                }
            )
            if processing_target == "local"
            else None
        ),
        "execution_mode": "batch-local" if processing_target == "local" else "batch-remote-gpu",
        "processing_target": processing_target,
        "gpu_base_url": gpu_base_url,
    }
    return execute_local_run(
        root,
        dataset_version,
        prepared_task,
        config,
        evaluator_name,
        evaluator_arguments,
        input_root=root,
        baseline_report=baseline_report,
        gate_profile=gate_profile,
        cancel_cb=cancel_cb,
    )


def execute_uploaded_stream_run(
    store_root: str | Path,
    dataset_version: Mapping[str, Any],
    task: Mapping[str, Any],
    video_path: str | Path,
    corners: Sequence[Sequence[float]],
    *,
    gpu_base_url: str | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute the existing two-second direct-GPU research path as one Run."""
    store = LocalManifestStore(store_root)
    root = store.root
    source = _stage_input(root, Path(video_path).resolve(), "video")
    prepared_task = dict(task)
    parameters = dict(prepared_task.get("parameters") or {})
    normalized_corners = _corners(list(corners))
    parameters["court_corners"] = [list(point) for point in normalized_corners]
    prepared_task["parameters"] = parameters
    prepared_task["parameters_fingerprint"] = canonical_json_sha256(parameters)
    prepared_task["input_path"] = source.relative_to(root).as_posix()
    prepared_task["input_sha256"] = _sha256_file(source)
    normalized_task = validate_contract(prepared_task)
    _validate_dataset_task(dataset_version, normalized_task)
    store.publish_manifest("dataset_version", dataset_version)
    run_record = store.create_run(normalized_task)
    if not run_record["created"]:
        raise ValueError("Run already exists; use a new run_id")

    run_id = normalized_task["run_id"]
    output_root = root / "runs" / run_id / "artifacts" / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    events = []
    session_id = None
    try:
        for event in _iter_remote_stream(
            str(source),
            normalized_corners,
            parameters,
            str(output_root),
            gpu_base_url=gpu_base_url,
            sport_id=str(parameters.get("sport_id") or "badminton"),
            session_mode=parameters.get("session_mode"),
            cancel_cb=cancel_cb,
        ):
            events.append(event)
            session_id = event.get("analysis_session_id") or session_id
            if cancel_cb is not None and cancel_cb():
                if session_id:
                    _cancel_remote_stream(session_id, gpu_base_url, str(parameters.get("sport_id") or "badminton"))
                raise AnalysisCancelled("stream analysis cancelled")
        if not events or events[-1].get("phase") != "finalized":
            raise RuntimeError(str((events[-1] if events else {}).get("error") or "stream analysis did not finalize"))
        _atomic_json(output_root / "stream-result.json", {"events": events})
        artifacts = _artifact_contracts(root, output_root)
        if not artifacts:
            raise RuntimeError("stream analysis completed without artifacts")
        result = _gpu_result(normalized_task, "succeeded", artifacts, None)
        stored = store.write_run_result(result)
    except Exception as exc:
        artifacts = _artifact_contracts(root, output_root)
        status = "cancelled" if isinstance(exc, AnalysisCancelled) else "failed"
        store.write_run_result(
            _gpu_result(
                normalized_task,
                status,
                artifacts,
                {"code": "stream_cancelled" if status == "cancelled" else "stream_failed", "message": str(exc)[:2000], "retryable": False},
            )
        )
        raise
    return {
        "schema_version": "local-run-summary.v1",
        "run_id": run_id,
        "status": "succeeded",
        "execution_mode": "two-second-direct-gpu",
        "analysis_session_id": session_id,
        "gpu_result_sha256": stored["content_sha256"],
        "output_dir": str(output_root),
        "terminal": events[-1],
    }


def recover_stream_session(
    analysis_session_id: str,
    output_dir: str | Path,
    *,
    gpu_base_url: str | None = None,
    sport_id: str = "badminton",
) -> dict[str, Any]:
    from .gpu_client import recover_remote_two_second_stream

    return recover_remote_two_second_stream(
        analysis_session_id,
        output_dir,
        gpu_base_url=gpu_base_url,
        sport_id=sport_id,
    )


def verify_gpu_service(sport_id: str, gpu_base_url: str | None = None) -> dict[str, Any]:
    return _verify_remote_gpu(sport_id, gpu_base_url)


def execute_local_run(
    store_root: str | Path,
    dataset_version: Mapping[str, Any],
    task: Mapping[str, Any],
    runner_config: Mapping[str, Any],
    evaluator_name: str,
    evaluator_arguments: Sequence[str],
    *,
    input_root: str | Path = ".",
    baseline_report: Mapping[str, Any] | None = None,
    gate_profile: Mapping[str, Any] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the existing pipeline and evaluator while freezing all evidence."""
    store = LocalManifestStore(store_root)
    normalized_task = validate_contract(task)
    if normalized_task["contract_type"] != "analysis_task":
        raise ValueError("task must be an analysis_task contract")
    _validate_dataset_task(dataset_version, normalized_task)

    source_root = Path(input_root).resolve()
    video_path = _resolve_input(source_root, normalized_task["input_path"], "input_path")
    if _sha256_file(video_path) != normalized_task["input_sha256"]:
        raise ValueError("input file SHA-256 does not match the frozen Run input")
    template_path = _resolve_input(source_root, runner_config.get("template_path"), "template_path")
    corners = _corners(runner_config.get("corners"))
    parameters = normalized_task["parameters"]
    template_sha256 = _sha256_file(template_path)
    if runner_config.get("template_sha256") != template_sha256:
        raise ValueError("template file SHA-256 does not match runner_config")
    if parameters.get("template_sha256") != template_sha256:
        raise ValueError("template file SHA-256 does not match the frozen Run parameters")
    if _corners(parameters.get("court_corners")) != corners:
        raise ValueError("runner corners do not match the frozen Run parameters")
    processing_target = runner_config.get("processing_target", "local")
    if processing_target not in {"local", "remote_gpu"}:
        raise ValueError("processing_target must be local or remote_gpu")

    dataset_record = store.publish_manifest("dataset_version", dataset_version)
    run_record = store.create_run(normalized_task)
    if not run_record["created"]:
        raise ValueError("Run already exists; use a new run_id")
    run_id = normalized_task["run_id"]
    artifacts_root = store.root / "runs" / run_id / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    try:
        if processing_target == "remote_gpu":
            _verify_remote_gpu(
                str(normalized_task["parameters"].get("sport_id") or "badminton"),
                runner_config.get("gpu_base_url"),
            )
            runner_output = _run_remote_analysis(
                str(video_path),
                str(template_path),
                corners,
                normalized_task["parameters"],
                str(artifacts_root / "analysis"),
                cancel_cb=cancel_cb,
                gpu_base_url=runner_config.get("gpu_base_url"),
            )
        else:
            runner_output = _run_analysis(
                str(video_path),
                str(template_path),
                corners,
                normalized_task["parameters"],
                output_dir=str(artifacts_root / "analysis"),
                cleanup_outputs=False,
                cancel_cb=cancel_cb,
            )
    except Exception as exc:
        store.write_run_result(
            _gpu_result(
                normalized_task,
                "failed",
                _artifact_contracts(store.root, artifacts_root / "analysis"),
                {"code": "analysis_failed", "message": str(exc)[:2000] or type(exc).__name__, "retryable": False},
            )
        )
        raise

    analysis_artifacts = _artifact_contracts(store.root, artifacts_root / "analysis")
    if not analysis_artifacts:
        error = RuntimeError("analysis runner completed without artifacts")
        store.write_run_result(
            _gpu_result(
                normalized_task,
                "failed",
                [],
                {"code": "missing_artifacts", "message": str(error), "retryable": False},
            )
        )
        raise error
    gpu_result = _gpu_result(normalized_task, "succeeded", analysis_artifacts, None)
    gpu_record = store.write_run_result(gpu_result)

    evaluation_output = artifacts_root / "evaluation" / (
        f"{evaluator_name}.json"
        if evaluator_name in {"doubles", "performance_streaming"}
        else evaluator_name
    )
    evaluation = execute_evaluator(
        evaluator_name,
        _wire_evaluator_arguments(evaluator_name, evaluator_arguments, video_path, runner_output),
        evaluation_output,
    )
    validate_metric_rows(evaluation["metrics"])
    metric_records = [
        store.publish_manifest("metric_result", _metric_manifest(run_id, row))
        for row in evaluation["metrics"]
    ]

    candidate = {
        "schema_version": "run-report.v1",
        "run_id": run_id,
        "dataset_version": normalized_task["dataset_version"],
        "dataset_manifest_sha256": dataset_record["content_sha256"],
        "hardware_fingerprint": runner_config.get("hardware_fingerprint"),
        "execution_mode": runner_config.get("execution_mode"),
        "metrics": evaluation["metrics"],
        "source_report": evaluation["source_report"],
    }
    comparison = (
        compare_reports(baseline_report, candidate, gate_profile=gate_profile)
        if baseline_report is not None
        else None
    )
    candidate_path = artifacts_root / "run-report.json"
    _atomic_json(candidate_path, candidate)
    _publish_artifact(store, run_id, candidate_path, "run_report")
    comparison_path = None
    if comparison is not None:
        comparison_path = artifacts_root / "comparison.json"
        _atomic_json(comparison_path, comparison)
        _publish_artifact(store, run_id, comparison_path, "comparison")

    return {
        "schema_version": "local-run-summary.v1",
        "run_id": run_id,
        "status": "succeeded",
        "dataset_version": normalized_task["dataset_version"],
        "run_input_sha256": contract_sha256(normalized_task),
        "gpu_result_sha256": gpu_record["content_sha256"],
        "metric_count": len(metric_records),
        "run_report": str(candidate_path),
        "comparison_report": str(comparison_path) if comparison_path else None,
        "gate_status": (comparison or {}).get("gate_result", {}).get("status"),
        "runner_output": runner_output,
    }


def _gpu_result(
    task: Mapping[str, Any],
    status: str,
    artifacts: list[dict[str, Any]],
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity_fields = {
        key: task[key]
        for key in (
            "run_id",
            "case_id",
            "dataset_version",
            "code_fingerprint",
            "model_fingerprint",
            "parameters_fingerprint",
            "input_sha256",
        )
    }
    return {
        "schema_version": "analysis-boundary.v1",
        "contract_type": "gpu_analysis_result",
        **identity_fields,
        "status": status,
        "artifacts": artifacts,
        "error": dict(error) if error is not None else None,
    }


def _run_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    # Keep dataset browsing and comparison usable without loading CV models.
    from .runner import run_analysis

    return run_analysis(*args, **kwargs)


def _run_remote_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .gpu_client import run_remote_analysis

    return run_remote_analysis(*args, **kwargs)


def _verify_remote_gpu(sport_id: str, gpu_base_url: str | None) -> dict[str, Any]:
    from .gpu_client import verify_remote_gpu_sport

    return verify_remote_gpu_sport(sport_id, gpu_base_url)


def _iter_remote_stream(*args: Any, **kwargs: Any):
    from .gpu_client import iter_remote_two_second_stream

    return iter_remote_two_second_stream(*args, **kwargs)


def _cancel_remote_stream(session_id: str, gpu_base_url: str | None, sport_id: str) -> dict[str, Any]:
    from .gpu_client import cancel_remote_two_second_stream

    return cancel_remote_two_second_stream(session_id, gpu_base_url, sport_id=sport_id)


def _validate_dataset_task(dataset_version: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    if task["dataset_version"] != dataset_version.get("dataset_version_id"):
        raise ValueError("task dataset_version does not match the dataset manifest")
    case_ids = {case.get("case_id") for case in dataset_version.get("cases") or []}
    if task["case_id"] not in case_ids:
        raise ValueError("task case_id is not present in the dataset version")


def _metric_manifest(run_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    identity = "|".join(
        str(row.get(key)) for key in ("scope", "scope_id", "metric_key")
    )
    metric_id = f"metric-{run_id}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
    return {
        "schema_version": "analysis-platform.v1",
        "kind": "metric_result",
        "metric_result_id": metric_id,
        "run_id": run_id,
        **dict(row),
    }


def _wire_evaluator_arguments(
    name: str,
    arguments: Sequence[str],
    video_path: Path,
    runner_output: Mapping[str, Any],
) -> list[str]:
    wired = list(arguments)

    def add(flag: str, value: Any, *, required: bool = True) -> None:
        if flag in wired:
            return
        if value:
            wired.extend([flag, str(value)])
        elif required:
            raise ValueError(f"{name} requires Runner output for {flag}")

    if name == "far_player":
        add("--video", video_path)
    elif name == "doubles":
        add("--detections", runner_output.get("detections"))
    elif name == "shuttle_tracknet_ab":
        add("--video", video_path)
        add("--baseline-detections", runner_output.get("detections"))
        add("--tracknet-raw-csv", runner_output.get("tracknet_raw_csv"))
    elif name == "performance_streaming":
        add("--trace", runner_output.get("performance_trace"), required=False)
    return wired


def _publish_artifact(
    store: LocalManifestStore,
    run_id: str,
    path: Path,
    kind: str,
) -> dict[str, Any]:
    relative = path.resolve().relative_to(store.root).as_posix()
    manifest = {
        "schema_version": "analysis-platform.v1",
        "kind": "artifact",
        "artifact_id": f"artifact-{run_id}-{kind}",
        "run_id": run_id,
        "path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    return store.publish_manifest("artifact", manifest)


def _artifact_contracts(store_root: Path, output_root: Path) -> list[dict[str, Any]]:
    if not output_root.is_dir():
        return []
    return [
        {
            "kind": path.suffix.lower().lstrip(".") or "file",
            "path": path.resolve().relative_to(store_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    ]


def _resolve_input(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must stay inside input_root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stage_input(root: Path, source: Path, label: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _sha256_file(source)
    suffix = source.suffix.lower()
    destination = root / "inputs" / f"{label}-{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise ValueError(f"staged {label} content hash mismatch")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if _sha256_file(temporary) != digest:
            raise ValueError(f"staged {label} content hash mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _corners(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("corners must contain four coordinate pairs")
    corners = []
    for point in value:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in point)
        ):
            raise ValueError("corners must contain four numeric coordinate pairs")
        corners.append((float(point[0]), float(point[1])))
    return corners


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
