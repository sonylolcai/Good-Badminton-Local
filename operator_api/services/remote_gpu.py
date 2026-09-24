"""Remote-first client for the GPU video-analysis API.

The upload body is written to the socket in bounded chunks.  The current GPU
API starts a job only after FastAPI has received a complete media container;
this avoids holding a multi-GB video in WebUI memory, but is not yet live
frame-by-frame inference.
"""

import http.client
import json
import mimetypes
import os
import ssl
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from badminton_analysis.cancellation import AnalysisCancelled, raise_if_cancelled


DEFAULT_GPU_API_URL = "http://xn-g.suanjiayun.com:52028"
CHUNK_BYTES = 1024 * 1024


class RemoteAnalysisError(RuntimeError):
    """The remote service cannot be used safely for this WebUI run."""


def remote_gpu_config(base_url_override=None):
    """Read server-side configuration; secrets never enter browser state.

    ``base_url_override`` is an operator-only WebUI development convenience.
    It changes the destination for this one request and is never written into
    the process environment, so concurrent submissions cannot accidentally
    redirect one another.  API credentials remain server-side configuration.
    """
    _load_local_config_file()
    configured_base_url = (
        str(base_url_override).strip()
        if base_url_override is not None and str(base_url_override).strip()
        else os.environ.get("GOOD_BADMINTON_GPU_API_URL", DEFAULT_GPU_API_URL).strip()
    )
    parsed = urlparse(configured_base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RemoteAnalysisError("GPU 服务地址必须是无账号、无查询参数的 http(s) 基础地址")
    return {
        "base_url": configured_base_url.rstrip("/"),
        "api_key": os.environ.get("GOOD_BADMINTON_GPU_API_KEY", ""),
        "timeout_seconds": float(os.environ.get("GOOD_BADMINTON_GPU_API_TIMEOUT", "30")),
        "poll_seconds": float(os.environ.get("GOOD_BADMINTON_GPU_API_POLL_SECONDS", "2")),
    }


def _load_local_config_file():
    """Load a minimal gitignored WebUI env file without an extra dependency."""
    config_path = Path(os.environ.get("GOOD_BADMINTON_WEBUI_CONFIG", ".webui-remote-gpu.env"))
    if not config_path.is_file():
        return
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip())


def run_remote_analysis(video_path, template_path, corners, options, output_dir, progress_cb=None, status_cb=None,
                        business_task_id=None, cancel_cb=None, gpu_base_url=None):
    """Submit, wait for, and retrieve one remote job into *output_dir*."""
    raise_if_cancelled(cancel_cb)
    config = remote_gpu_config(gpu_base_url)
    if not config["api_key"]:
        raise RemoteAnalysisError("GOOD_BADMINTON_GPU_API_KEY is not configured in the WebUI process")

    options = _remote_options(options)
    job = _submit_multipart(
        config,
        video_path=video_path,
        template_path=template_path,
        corners=corners,
        options=options,
        progress_cb=progress_cb,
        status_cb=status_cb,
        idempotency_key=business_task_id,
        cancel_cb=cancel_cb,
    )
    job_id = job["job_id"]
    receipt = job.get("receipt") or {}
    _emit(status_cb, {
        "mode": "remote_gpu", "phase": "accepted", "job_id": job_id,
        "accepted_at": receipt.get("accepted_at") or job.get("created_at"),
        "status_url": receipt.get("status_url"),
        "submission_reused": bool(receipt.get("reused")),
    })
    latest = _wait_for_job(
        config, job_id, progress_cb=progress_cb, status_cb=status_cb, cancel_cb=cancel_cb,
    )
    terminal_trace = _download_terminal_performance_trace(
        config,
        job_id,
        latest,
        output_dir,
        # A terminal trace is small and must survive a user cancellation too.
        cancel_cb=None,
        status_cb=status_cb,
    )
    if latest.get("status") == "cancelled":
        exc = AnalysisCancelled("远端 GPU 任务已中断。")
        exc.performance_trace_path = terminal_trace
        raise exc
    if latest.get("status") != "succeeded":
        error = latest.get("error") or {}
        exc = RemoteAnalysisError(error.get("message") or f"remote job {job_id} ended as {latest.get('status')}")
        exc.performance_trace_path = terminal_trace
        raise exc

    try:
        raise_if_cancelled(cancel_cb)
    except AnalysisCancelled as exc:
        exc.performance_trace_path = terminal_trace
        raise
    _emit(status_cb, {"mode": "remote_gpu", "phase": "downloading", "job_id": job_id})
    result = _json_request(config, f"/api/v1/jobs/{job_id}/result")
    downloaded = _download_result(
        config, job_id, result, output_dir,
        cancel_cb=cancel_cb, status_cb=status_cb,
    )
    if terminal_trace and not downloaded.get("performance_trace"):
        downloaded["performance_trace"] = terminal_trace
    _emit(status_cb, {"mode": "remote_gpu", "phase": "downloaded", "job_id": job_id})
    return downloaded


def submit_remote_job(video_path, template_path, corners, options, business_task_id):
    """Upload one manually requested business video and return the GPU receipt."""
    config = remote_gpu_config()
    if not config["api_key"]:
        raise RemoteAnalysisError("GOOD_BADMINTON_GPU_API_KEY is not configured")
    return _submit_multipart(
        config,
        video_path=video_path,
        template_path=template_path,
        corners=corners,
        options=_remote_options(options),
        idempotency_key=business_task_id,
    )


def delete_remote_job(job_id, *, full=False):
    config = remote_gpu_config()
    if not config["api_key"]:
        raise RemoteAnalysisError("GOOD_BADMINTON_GPU_API_KEY is not configured")
    suffix = "data" if full else "resources"
    return _json_request(config, f"/api/v1/jobs/{job_id}/{suffix}", method="DELETE")


def delete_remote_stream(session_id, *, full=False):
    config = remote_gpu_config()
    if not config["api_key"]:
        raise RemoteAnalysisError("GOOD_BADMINTON_GPU_API_KEY is not configured")
    suffix = "data" if full else "resources"
    return _json_request(config, f"/api/v1/stream-sessions/{session_id}/{suffix}", method="DELETE")


def _remote_options(options):
    # GPU API intentionally rejects arbitrary model paths; it owns the pinned
    # deployed model artifacts.  Everything else is part of the public contract.
    return {
        key: value
        for key, value in options.items()
        if key not in {"yolo_pose_model", "ball_model"}
    }


def _submit_multipart(config, video_path, template_path, corners, options, progress_cb=None, status_cb=None,
                      idempotency_key=None, cancel_cb=None):
    parsed = urlparse(config["base_url"])
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RemoteAnalysisError("GOOD_BADMINTON_GPU_API_URL must be an http(s) URL")
    boundary = f"----GoodBadminton{uuid.uuid4().hex}"
    fields = {
        "court_corners": json.dumps(corners),
        "options_json": json.dumps(options),
    }
    length = _multipart_length(boundary, fields, [("video", video_path), ("template", template_path)])
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=config["timeout_seconds"])
    endpoint = (parsed.path.rstrip("/") + "/api/v1/jobs") or "/api/v1/jobs"
    sent = 0
    total_upload = os.path.getsize(video_path) + os.path.getsize(template_path)
    _emit(status_cb, {"mode": "remote_gpu", "phase": "uploading", "uploaded_bytes": 0, "total_upload_bytes": total_upload})

    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(length))
        connection.putheader("X-API-Key", config["api_key"])
        if idempotency_key:
            connection.putheader("X-Idempotency-Key", idempotency_key)
        connection.endheaders()
        for name, value in fields.items():
            raise_if_cancelled(cancel_cb)
            connection.send(_field_part(boundary, name, value))
        for name, path in (("video", video_path), ("template", template_path)):
            raise_if_cancelled(cancel_cb)
            path = Path(path)
            connection.send(_file_header(boundary, name, path))
            with path.open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    raise_if_cancelled(cancel_cb)
                    connection.send(chunk)
                    sent += len(chunk)
                    if progress_cb and total_upload:
                        progress_cb(min(sent / total_upload * 0.25, 0.25), "正在流式上传到 GPU 服务器")
                    _emit(status_cb, {"mode": "remote_gpu", "phase": "uploading", "uploaded_bytes": sent, "total_upload_bytes": total_upload})
            connection.send(b"\r\n")
        connection.send(f"--{boundary}--\r\n".encode("utf-8"))
        response = connection.getresponse()
        body = response.read().decode("utf-8", "replace")
    except TimeoutError as exc:
        raise RemoteAnalysisError(
            "远端 GPU 上传在 "
            f"{config['timeout_seconds']:g} 秒内未收到接收回执；"
            "请先检查本地 127.0.0.1:8080 SSH 隧道和远端 /api/v1/health。"
        ) from exc
    except OSError as exc:
        raise RemoteAnalysisError(
            "远端 GPU 上传连接失败；请先检查本地 127.0.0.1:8080 SSH 隧道和远端服务。"
            f" 原始错误：{exc}"
        ) from exc
    finally:
        connection.close()
    if response.status not in {200, 201, 202}:
        raise RemoteAnalysisError(f"remote GPU job submission returned HTTP {response.status}: {body[:500]}")
    return _parse_json(body, "remote job submission")


def _wait_for_job(config, job_id, progress_cb=None, status_cb=None, cancel_cb=None):
    deadline = time.monotonic() + float(os.environ.get("GOOD_BADMINTON_GPU_JOB_TIMEOUT", "43200"))
    cancellation_sent = False
    while time.monotonic() < deadline:
        if cancel_cb is not None and cancel_cb() and not cancellation_sent:
            cancelled = _json_request(config, f"/api/v1/jobs/{job_id}", method="DELETE")
            _emit(status_cb, {
                "mode": "remote_gpu", "phase": cancelled.get("status") or "cancelling",
                "job_id": job_id, "cancellation_requested": True,
            })
            cancellation_sent = True
        job = _json_request(config, f"/api/v1/jobs/{job_id}")
        state = job.get("status")
        details = job.get("progress") or {}
        timing = job.get("timing") or {}
        stages = timing.get("stages") or []
        current_stage = stages[-1] if stages else {}
        _emit(status_cb, {
            "mode": "remote_gpu", "phase": state, "job_id": job_id,
            "processed_frames": details.get("processed_frames", 0),
            "total_frames": details.get("total_frames"), "ratio": details.get("ratio", 0.0),
            "tracking": job.get("tracking"),
            "timing": timing,
            "performance_trace": job.get("performance_trace"),
            "stage": timing.get("current_stage"),
            "stage_detail": current_stage.get("details"),
        })
        if progress_cb:
            ratio = float(details.get("ratio") or 0.0)
            progress_cb(0.25 + ratio * 0.75, f"GPU {state}: {details.get('processed_frames', 0)}/{details.get('total_frames') or '?'} 帧")
        if state in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(config["poll_seconds"])
    raise RemoteAnalysisError(f"remote GPU job {job_id} polling timed out")


def recover_remote_task(business_task_id, remote_job_id, output_dir, status_cb=None):
    """Run one durable recovery pass after the business process restarts.

    If a process died after upload but before reading the HTTP 202 response, the
    idempotency key lets the updated GPU API reveal the already-accepted job.
    This performs one poll only; a service scheduler can call it repeatedly.
    """
    config = remote_gpu_config()
    job_id = remote_job_id
    if not job_id:
        recovered = _json_request(config, f"/api/v1/jobs/by-idempotency/{business_task_id}")
        job_id = recovered["job_id"]
        receipt = recovered.get("receipt") or {}
        _emit(status_cb, {
            "mode": "remote_gpu", "phase": "accepted", "job_id": job_id,
            "accepted_at": receipt.get("accepted_at") or recovered.get("created_at"),
            "status_url": receipt.get("status_url"), "recovered": True,
        })
    job = _json_request(config, f"/api/v1/jobs/{job_id}")
    details = job.get("progress") or {}
    timing = job.get("timing") or {}
    stages = timing.get("stages") or []
    current_stage = stages[-1] if stages else {}
    _emit(status_cb, {
        "mode": "remote_gpu", "phase": job.get("status"), "job_id": job_id,
        "processed_frames": details.get("processed_frames", 0),
        "total_frames": details.get("total_frames"), "ratio": details.get("ratio", 0.0),
        "tracking": job.get("tracking"),
        "timing": timing,
        "performance_trace": job.get("performance_trace"),
        "stage": timing.get("current_stage"),
        "stage_detail": current_stage.get("details"),
        "recovered": True,
    })
    if job.get("status") != "succeeded":
        # A restarted WebUI used to discover a failed job but lose the only
        # server-side timing/error evidence.  Fetch the small terminal trace
        # before returning; the reconciliation caller archives it in its
        # durable business ledger.  This never fetches a normal result bundle
        # for failed or cancelled work.
        if job.get("status") in {"failed", "cancelled"}:
            job["local_performance_trace"] = _download_terminal_performance_trace(
                config,
                job_id,
                job,
                output_dir,
                status_cb=status_cb,
            )
        return job, None
    _emit(status_cb, {"mode": "remote_gpu", "phase": "downloading", "job_id": job_id, "recovered": True})
    result = _json_request(config, f"/api/v1/jobs/{job_id}/result")
    downloaded = _download_result(config, job_id, result, output_dir, status_cb=status_cb)
    _emit(status_cb, {"mode": "remote_gpu", "phase": "downloaded", "job_id": job_id, "recovered": True})
    return job, downloaded


def _download_result(config, job_id, result, output_dir, cancel_cb=None, status_cb=None):
    """Download result artifacts in a documented, deterministic serial order.

    Result transfer is intentionally reported one artifact at a time.  The
    current implementation is serial, so these lifecycle events provide the
    evidence needed to decide whether parallel egress is worth adding rather
    than making an unverified performance claim.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts = (result.get("result") or {}).get("artifacts") or {}
    downloaded = {}
    for name, artifact in artifacts.items():
        raise_if_cancelled(cancel_cb)
        relative_path = artifact.get("relative_path") or name
        destination = (target / relative_path).resolve()
        try:
            destination.relative_to(target.resolve())
        except ValueError as exc:
            raise RemoteAnalysisError("remote API returned an unsafe artifact path") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        _emit(status_cb, {
            "mode": "remote_gpu", "phase": "artifact_downloading", "job_id": job_id,
            "artifact": name, "relative_path": relative_path,
        })
        _download(
            config,
            artifact.get("url") or f"/api/v1/jobs/{job_id}/artifacts/{name}",
            destination,
            cancel_cb=cancel_cb,
        )
        downloaded[name] = str(destination)
        _emit(status_cb, {
            "mode": "remote_gpu", "phase": "artifact_downloaded", "job_id": job_id,
            "artifact": name, "relative_path": relative_path,
            "size_bytes": destination.stat().st_size,
        })

    metadata_path = downloaded.get("metadata")
    metadata = {}
    if metadata_path and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    metadata["execution"] = {
        "mode": "remote_gpu",
        "remote_job_id": job_id,
        "remote_base_url": config["base_url"],
        "fallback_used": False,
    }
    if metadata_path:
        Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        metadata_path = str(target / "metadata.remote.json")
        Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(status_cb, {
        "mode": "remote_gpu", "phase": "local_metadata_persisted", "job_id": job_id,
        "metadata_path": metadata_path,
    })

    return {
        "output_dir": str(target),
        "video": downloaded.get("annotated_video"),
        "metadata": metadata_path,
        "detections": downloaded.get("detections"),
        "tracknet_raw_csv": downloaded.get("tracknet_raw_csv"),
        "performance_report": downloaded.get("performance_report"),
        "movement_metrics": downloaded.get("movement_metrics"),
        "movement_rallies": downloaded.get("movement_rallies"),
        "movement_rally_window_sweep": downloaded.get("movement_rally_window_sweep"),
        "performance_trace": downloaded.get("performance_trace"),
        "position_evidence_summary": downloaded.get("position_evidence_summary"),
        "visualizations": [path for name, path in downloaded.items() if name.startswith("visualization_")],
        "warnings": (result.get("result") or {}).get("warnings", []),
        "execution": metadata["execution"],
    }


def _download_terminal_performance_trace(config, job_id, job, output_dir, cancel_cb=None, status_cb=None):
    """Fetch a terminal trace for success, failure, or cancellation.

    Failed jobs do not expose the normal result manifest, so this uses the
    dedicated endpoint and deliberately never masks the original task error if
    archival download itself is unavailable.
    """
    trace = (job or {}).get("performance_trace") or {}
    if not trace:
        return None
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "performance_trace.json"
    try:
        _emit(status_cb, {
            "mode": "remote_gpu", "phase": "artifact_downloading", "job_id": job_id,
            "artifact": "performance_trace", "relative_path": "performance_trace.json",
            "terminal_trace": True,
        })
        _download(
            config,
            trace.get("url") or f"/api/v1/jobs/{job_id}/performance-trace",
            destination,
            cancel_cb=cancel_cb,
        )
    except (RemoteAnalysisError, OSError):
        return None
    _emit(status_cb, {
        "mode": "remote_gpu", "phase": "artifact_downloaded", "job_id": job_id,
        "artifact": "performance_trace", "relative_path": "performance_trace.json",
        "size_bytes": destination.stat().st_size, "terminal_trace": True,
    })
    return str(destination) if destination.is_file() else None


def _json_request(config, path, method="GET"):
    request = Request(
        config["base_url"] + path,
        headers={"X-API-Key": config["api_key"]},
        method=method,
    )
    try:
        with urlopen(request, timeout=config["timeout_seconds"], context=_ssl_context(config["base_url"])) as response:
            body = response.read().decode("utf-8", "replace")
    except OSError as exc:
        raise RemoteAnalysisError(f"remote GPU request failed: {exc}") from exc
    return _parse_json(body, "remote GPU response")


def _download(config, path, destination, cancel_cb=None):
    request = Request(config["base_url"] + path, headers={"X-API-Key": config["api_key"]}, method="GET")
    try:
        with urlopen(request, timeout=config["timeout_seconds"], context=_ssl_context(config["base_url"])) as response, destination.open("wb") as output:
            while chunk := response.read(CHUNK_BYTES):
                raise_if_cancelled(cancel_cb)
                output.write(chunk)
    except OSError as exc:
        raise RemoteAnalysisError(f"artifact download failed: {exc}") from exc


def _ssl_context(base_url):
    # Retain normal certificate verification.  The helper only keeps urlopen
    # call sites uniform for HTTP and HTTPS deployments.
    return ssl.create_default_context() if base_url.startswith("https://") else None


def _parse_json(value, label):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RemoteAnalysisError(f"{label} was not valid JSON") from exc


def _emit(callback, value):
    if callback:
        callback(value)


def _field_part(boundary, name, value):
    return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")


def _file_header(boundary, name, path):
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode("utf-8")


def _multipart_length(boundary, fields, files):
    length = sum(len(_field_part(boundary, name, value)) for name, value in fields.items())
    for name, path in files:
        path = Path(path)
        length += len(_file_header(boundary, name, path)) + path.stat().st_size + 2
    return length + len(f"--{boundary}--\r\n".encode("utf-8"))
