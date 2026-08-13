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


DEFAULT_GPU_API_URL = "http://xn-g.suanjiayun.com:52028"
CHUNK_BYTES = 1024 * 1024


class RemoteAnalysisError(RuntimeError):
    """The remote service cannot be used safely for this WebUI run."""


def remote_gpu_config():
    """Read server-side configuration; secrets never enter browser state."""
    _load_local_config_file()
    return {
        "base_url": os.environ.get("GOOD_BADMINTON_GPU_API_URL", DEFAULT_GPU_API_URL).rstrip("/"),
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


def run_remote_analysis(video_path, template_path, corners, options, output_dir, progress_cb=None):
    """Submit, wait for, and retrieve one remote job into *output_dir*."""
    config = remote_gpu_config()
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
    )
    job_id = job["job_id"]
    latest = _wait_for_job(config, job_id, progress_cb=progress_cb)
    if latest.get("status") != "succeeded":
        error = latest.get("error") or {}
        raise RemoteAnalysisError(error.get("message") or f"remote job {job_id} ended as {latest.get('status')}")

    result = _json_request(config, f"/api/v1/jobs/{job_id}/result")
    return _download_result(config, job_id, result, output_dir)


def _remote_options(options):
    # GPU API intentionally rejects arbitrary model paths; it owns the pinned
    # deployed model artifacts.  Everything else is part of the public contract.
    return {
        key: value
        for key, value in options.items()
        if key not in {"yolo_pose_model", "ball_model"}
    }


def _submit_multipart(config, video_path, template_path, corners, options, progress_cb=None):
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

    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(length))
        connection.putheader("X-API-Key", config["api_key"])
        connection.endheaders()
        for name, value in fields.items():
            connection.send(_field_part(boundary, name, value))
        for name, path in (("video", video_path), ("template", template_path)):
            path = Path(path)
            connection.send(_file_header(boundary, name, path))
            with path.open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    connection.send(chunk)
                    sent += len(chunk)
                    if progress_cb and total_upload:
                        progress_cb(min(sent / total_upload * 0.25, 0.25), "正在流式上传到 GPU 服务器")
            connection.send(b"\r\n")
        connection.send(f"--{boundary}--\r\n".encode("utf-8"))
        response = connection.getresponse()
        body = response.read().decode("utf-8", "replace")
    except OSError as exc:
        raise RemoteAnalysisError(f"remote GPU upload failed: {exc}") from exc
    finally:
        connection.close()
    if response.status not in {200, 201, 202}:
        raise RemoteAnalysisError(f"remote GPU job submission returned HTTP {response.status}: {body[:500]}")
    return _parse_json(body, "remote job submission")


def _wait_for_job(config, job_id, progress_cb=None):
    deadline = time.monotonic() + float(os.environ.get("GOOD_BADMINTON_GPU_JOB_TIMEOUT", "43200"))
    while time.monotonic() < deadline:
        job = _json_request(config, f"/api/v1/jobs/{job_id}")
        state = job.get("status")
        details = job.get("progress") or {}
        if progress_cb:
            ratio = float(details.get("ratio") or 0.0)
            progress_cb(0.25 + ratio * 0.75, f"GPU {state}: {details.get('processed_frames', 0)}/{details.get('total_frames') or '?'} 帧")
        if state in {"succeeded", "failed"}:
            return job
        time.sleep(config["poll_seconds"])
    raise RemoteAnalysisError(f"remote GPU job {job_id} polling timed out")


def _download_result(config, job_id, result, output_dir):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts = (result.get("result") or {}).get("artifacts") or {}
    downloaded = {}
    for name, artifact in artifacts.items():
        relative_path = artifact.get("relative_path") or name
        destination = (target / relative_path).resolve()
        try:
            destination.relative_to(target.resolve())
        except ValueError as exc:
            raise RemoteAnalysisError("remote API returned an unsafe artifact path") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download(config, artifact.get("url") or f"/api/v1/jobs/{job_id}/artifacts/{name}", destination)
        downloaded[name] = str(destination)

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

    return {
        "output_dir": str(target),
        "video": downloaded.get("annotated_video"),
        "metadata": metadata_path,
        "detections": downloaded.get("detections"),
        "visualizations": [path for name, path in downloaded.items() if name.startswith("visualization_")],
        "warnings": (result.get("result") or {}).get("warnings", []),
        "execution": metadata["execution"],
    }


def _json_request(config, path):
    request = Request(
        config["base_url"] + path,
        headers={"X-API-Key": config["api_key"]},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config["timeout_seconds"], context=_ssl_context(config["base_url"])) as response:
            body = response.read().decode("utf-8", "replace")
    except OSError as exc:
        raise RemoteAnalysisError(f"remote GPU request failed: {exc}") from exc
    return _parse_json(body, "remote GPU response")


def _download(config, path, destination):
    request = Request(config["base_url"] + path, headers={"X-API-Key": config["api_key"]}, method="GET")
    try:
        with urlopen(request, timeout=config["timeout_seconds"], context=_ssl_context(config["base_url"])) as response, destination.open("wb") as output:
            while chunk := response.read(CHUNK_BYTES):
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
