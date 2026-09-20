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
import re
import ssl
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from badminton_analysis.cancellation import AnalysisCancelled, raise_if_cancelled
from business_gateway.streaming.client import StreamAPIError, StreamSessionClient
from business_gateway.streaming.models import DeliveryLedger, SegmentMetadata, StreamClientConfig
from business_gateway.streaming.segmenter import GrowingVideoSegmenter


DEFAULT_GPU_API_URL = "http://xn-g.suanjiayun.com:52028"
DEFAULT_LOCAL_CPU_GPU_API_URL = "http://127.0.0.1:18080"
CHUNK_BYTES = 1024 * 1024
SUPPORTED_SPORT_IDS = {"badminton", "tennis"}


class RemoteAnalysisError(RuntimeError):
    """The remote service cannot be used safely for this WebUI run."""


def _normalize_sport_id(sport_id):
    normalized = str(sport_id or "badminton").strip().lower()
    if normalized not in SUPPORTED_SPORT_IDS:
        allowed = ", ".join(sorted(SUPPORTED_SPORT_IDS))
        raise RemoteAnalysisError(f"运动模式必须是 [{allowed}] 之一")
    return normalized


def configured_gpu_base_url(sport_id="badminton"):
    """Return the configured fixed-sport URL without performing network I/O."""
    _load_local_config_file()
    sport_id = _normalize_sport_id(sport_id)
    configured = os.environ.get(f"GOOD_{sport_id.upper()}_GPU_API_URL", "").strip()
    if configured:
        return configured
    return DEFAULT_GPU_API_URL if sport_id == "badminton" else ""


def configured_local_cpu_gpu_base_url():
    """Return the loopback-only API address used for local CPU experiments."""
    _load_local_config_file()
    return os.environ.get(
        "GOOD_LOCAL_CPU_GPU_API_URL", DEFAULT_LOCAL_CPU_GPU_API_URL
    ).strip().rstrip("/")


def remote_gpu_config(base_url_override=None, *, sport_id="badminton", local_cpu=False):
    """Read server-side configuration; secrets never enter browser state.

    ``base_url_override`` is an operator-only WebUI development convenience.
    It changes the destination for this one request and is never written into
    the process environment, so concurrent submissions cannot accidentally
    redirect one another.  API credentials remain server-side configuration.
    """
    sport_id = _normalize_sport_id(sport_id)
    _load_local_config_file()
    configured_base_url = (
        str(base_url_override).strip()
        if base_url_override is not None and str(base_url_override).strip()
        else (
            configured_local_cpu_gpu_base_url()
            if local_cpu else configured_gpu_base_url(sport_id)
        )
    )
    if not configured_base_url:
        env_name = f"GOOD_{sport_id.upper()}_GPU_API_URL"
        raise RemoteAnalysisError(f"请在 WebUI 服务器配置 {env_name}，或在页面填写 {sport_id} GPU 服务地址")
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
        # Separate service credentials are preferred. The old badminton key is
        # a compatibility fallback only, useful while both test services share
        # one reverse-proxy credential.
        "api_key": (
            (
                os.environ.get("GOOD_LOCAL_CPU_GPU_API_KEY", "")
                or os.environ.get("GOOD_BADMINTON_GPU_API_KEY", "")
            )
            if local_cpu else (
                os.environ.get(f"GOOD_{sport_id.upper()}_GPU_API_KEY", "")
                or os.environ.get("GOOD_BADMINTON_GPU_API_KEY", "")
            )
        ),
        "timeout_seconds": float(
            os.environ.get(
                f"GOOD_{sport_id.upper()}_GPU_API_TIMEOUT",
                os.environ.get("GOOD_BADMINTON_GPU_API_TIMEOUT", "30"),
            )
        ),
        "poll_seconds": float(
            os.environ.get(
                f"GOOD_{sport_id.upper()}_GPU_API_POLL_SECONDS",
                os.environ.get("GOOD_BADMINTON_GPU_API_POLL_SECONDS", "2"),
            )
        ),
        "sport_id": sport_id,
    }


def verify_remote_gpu_sport(sport_id, gpu_base_url=None, *, local_cpu=False):
    """Fail closed when a WebUI mode points at the wrong fixed-sport GPU."""
    sport_id = _normalize_sport_id(sport_id)
    config = remote_gpu_config(gpu_base_url, sport_id=sport_id, local_cpu=local_cpu)
    health = _json_request(config, "/api/v1/health")
    actual = str(health.get("sport_id") or "").strip().lower()
    if actual != sport_id:
        raise RemoteAnalysisError(
            f"当前选择的是{sport_id}，但 GPU 服务返回 sport_id={actual or 'missing'}；已阻止上传。"
        )
    if health.get("service_kind") not in {None, "pure_gpu_visual_observation"}:
        raise RemoteAnalysisError("GPU 服务不是纯视觉流式入口；已阻止将视频发送到整文件业务接口。")
    return health


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
    sport_id = _normalize_sport_id(options.get("sport_id"))
    config = remote_gpu_config(gpu_base_url, sport_id=sport_id)
    if not config["api_key"]:
        raise RemoteAnalysisError(
            f"GOOD_{sport_id.upper()}_GPU_API_KEY is not configured in the WebUI process"
        )
    verify_remote_gpu_sport(sport_id, gpu_base_url)

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


def stream_roster_configuration(expected_player_count, *, sport_id="badminton", session_mode=None):
    """Return the fixed anonymous roster policy for one continuous match.

    Track IDs are runtime implementation details, not the number of people in
    a court video.  A badminton match contains either two or four players, so
    direct stream sessions must lock that roster once enough stable on-court
    evidence has been collected.  The running session retains tracker state
    across its contiguous two-second segments; separate matches get separate
    sessions and never inherit an anonymous ID by accident.
    """
    sport_id = _normalize_sport_id(sport_id)
    if sport_id == "tennis":
        if session_mode != "singles_match" or expected_player_count != 2:
            raise RemoteAnalysisError("网球当前只支持单打对打：固定 2 名运动员")
    elif expected_player_count not in {2, 4}:
        raise RemoteAnalysisError("羽毛球场上人数只能选择 2 人或 4 人")
    return {
        "lock_match_roster": True,
        "expected_player_count": expected_player_count,
        "roster_stable_frames": 3,
        "max_roster_count": expected_player_count,
        "roster_discovery_seconds": 8.0,
    }


def iter_remote_two_second_stream(
    video_path,
    corners,
    options,
    output_dir,
    gpu_base_url=None,
    *,
    sport_id="badminton",
    session_mode=None,
    local_cpu=False,
):
    """Send independently decodable two-second MP4 fragments directly to GPU.

    This is intentionally separate from :mod:`webui.stream_replay`: that
    legacy development workflow posts the source file to a local business
    gateway first.  Here the WebUI reads only its server-side GPU credentials,
    creates a remote ``stream-session.v1`` session at *gpu_base_url*, then
    uploads the fragments to that same remote origin.
    """

    source = Path(video_path)
    if not source.is_file():
        raise RemoteAnalysisError("请先上传可读取的视频文件")
    if not corners or len(corners) != 4:
        raise RemoteAnalysisError("请先确认四个球场角点")
    sport_id = _normalize_sport_id(sport_id)
    if sport_id == "tennis" and session_mode != "singles_match":
        raise RemoteAnalysisError("网球上传当前只支持 singles_match（单打对打）")
    config = remote_gpu_config(gpu_base_url, sport_id=sport_id, local_cpu=local_cpu)
    if not config["api_key"]:
        raise RemoteAnalysisError(f"GOOD_{sport_id.upper()}_GPU_API_KEY is not configured in the WebUI process")
    health = verify_remote_gpu_sport(sport_id, gpu_base_url, local_cpu=local_cpu)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    client = StreamSessionClient(
        StreamClientConfig(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout_seconds=config["timeout_seconds"],
        ),
        DeliveryLedger(target / "delivery-ledger.json"),
    )
    request_id = f"webui-stream-{uuid.uuid4().hex}"
    normalized_corners = [[float(x), float(y)] for x, y in corners]
    far_roi = options.get("far_pose_roi")
    stream_configuration = {
        "analysis_sample_hz": int(options["analysis_sample_hz"]),
        "pose_imgsz": int(options["pose_imgsz"]),
        "shuttle_detector": str(options["shuttle_detector"]),
        # The deployed stream contract produces structured events, not an
        # annotated MP4.  Never ask the server for an unsupported export.
        "generate_annotated_video": False,
        "preserve_audio": False,
        "court_health_check_hz": 2,
        "tracker_backend": str(options["tracker_backend"]),
        "far_player_enhancement": bool(options.get("far_player_enhancement", False)),
    }
    if sport_id != "badminton":
        stream_configuration.update(
            {
                "sport_id": sport_id,
                "session_mode": session_mode,
            }
        )
    stream_configuration.update(
        stream_roster_configuration(
            int(options["expected_player_count"]),
            sport_id=sport_id,
            session_mode=session_mode,
        )
    )
    if far_roi is not None:
        stream_configuration["far_pose_roi"] = [float(value) for value in far_roi]
    create_request = {
        "schema_version": "stream-session.v1",
        "camera_id": "webui-development-camera",
        "calibration_id": f"webui-calibration-{request_id[-12:]}",
        "court_corners": normalized_corners,
        "analysis_mode": "person_only",
        "client_reference": request_id,
        "configuration": stream_configuration,
    }
    try:
        created = client.create_session(create_request, idempotency_key=request_id)
        session_id = str(created["analysis_session_id"])
        yield {
            "mode": "remote_gpu_two_second_stream",
            "phase": "session_accepted",
            "analysis_session_id": session_id,
            "remote_base_url": config["base_url"],
            "sport_id": sport_id,
            "gpu_health": health,
            "segment_seconds": 2.0,
        }
        segmenter = GrowingVideoSegmenter(
            target / "segments",
            segment_duration_sec=2.0,
            # Re-encoding forces an aligned keyframe at each boundary, so a
            # complete-file upload cannot silently become one long fragment.
            encoding_mode="h264",
            preserve_audio=False,
        )
        last_index = -1
        for artifact in segmenter.iter_input(str(source), overwrite=True):
            metadata = SegmentMetadata.from_file(
                artifact.path,
                segment_index=artifact.segment_index,
                source_start_time_sec=artifact.source_start_time_sec,
                duration_sec=artifact.duration_sec,
                idempotency_prefix=request_id,
                content_type=artifact.content_type,
                court_corners=normalized_corners,
            )
            receipt = client.submit_segment(artifact.path, metadata)
            last_index = artifact.segment_index
            yield {
                "mode": "remote_gpu_two_second_stream",
                "phase": "segment_accepted",
                "analysis_session_id": session_id,
                "remote_base_url": config["base_url"],
                "segment_index": artifact.segment_index,
                "source_start_time_sec": artifact.source_start_time_sec,
                "duration_sec": artifact.duration_sec,
                "receipt": receipt.get("receipt"),
            }
        if last_index < 0:
            raise RemoteAnalysisError("未生成可上传的 2 秒视频片段")
        completion = client.complete(last_index, allow_partial=False)
        yield {
            "mode": "remote_gpu_two_second_stream",
            "phase": "session_sealed",
            "analysis_session_id": session_id,
            "remote_base_url": config["base_url"],
            "expected_last_segment_index": last_index,
            "completion": completion,
        }
        terminal = client.wait_for_terminal(
            poll_interval_seconds=config["poll_seconds"],
            timeout_seconds=float(
                os.environ.get(
                    f"GOOD_{sport_id.upper()}_STREAM_JOB_TIMEOUT",
                    os.environ.get("GOOD_BADMINTON_STREAM_JOB_TIMEOUT", "43200"),
                )
            ),
        )
        # A stream session returns lightweight terminal status only.  Materialise
        # its immutable events on the WebUI host so the same per-track movement
        # evidence shown for a full-video upload is available immediately here.
        # Candidate crops are fetched server-side and saved locally: the browser
        # never receives the GPU API key or a credential-bearing URL.
        terminal["webui_result"] = _materialize_stream_webui_result(
            client, config, target, terminal,
        )
        trace_path = target / "stream_trace.json"
        try:
            trace_path.write_text(
                json.dumps(client.get_trace(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except StreamAPIError:
            trace_path = None
        yield {
            "mode": "remote_gpu_two_second_stream",
            "phase": terminal.get("status") or "unknown",
            "analysis_session_id": session_id,
            "remote_base_url": config["base_url"],
            "expected_last_segment_index": last_index,
            "stream_status": terminal,
            "trace_path": str(trace_path) if trace_path else None,
        }
    except (StreamAPIError, OSError, RuntimeError, ValueError) as exc:
        raise RemoteAnalysisError(f"远端 GPU 的 2 秒分片推送失败：{exc}") from exc


def recover_remote_two_second_stream(
    analysis_session_id,
    output_dir,
    gpu_base_url=None,
    *,
    sport_id="badminton",
    local_cpu=False,
):
    """Recover a completed direct-GPU session without uploading video again.

    A browser refresh loses Gradio's in-memory result values, but the local
    delivery ledger and the GPU's immutable stream events remain available.
    This read-only recovery regenerates the WebUI evidence files (metrics and
    downloaded anonymous candidate crops) in the original session directory.
    """

    session_id = str(analysis_session_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise RemoteAnalysisError("流会话 ID 格式无效")
    target = Path(output_dir)
    ledger_path = target / "delivery-ledger.json"
    if not ledger_path.is_file():
        raise RemoteAnalysisError("本机未找到该流会话的上传台账，无法安全恢复结果")
    sport_id = _normalize_sport_id(sport_id)
    config = remote_gpu_config(gpu_base_url, sport_id=sport_id, local_cpu=local_cpu)
    if not config["api_key"]:
        raise RemoteAnalysisError(f"GOOD_{sport_id.upper()}_GPU_API_KEY is not configured in the WebUI process")
    verify_remote_gpu_sport(sport_id, gpu_base_url, local_cpu=local_cpu)
    ledger = DeliveryLedger(ledger_path)
    if ledger.analysis_session_id != session_id:
        raise RemoteAnalysisError("会话 ID 与本机上传台账不匹配")
    client = StreamSessionClient(
        StreamClientConfig(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout_seconds=config["timeout_seconds"],
        ),
        ledger,
    )
    terminal = client.get_status()
    if str(terminal.get("analysis_session_id") or "") != session_id:
        raise RemoteAnalysisError("GPU 返回的会话 ID 与请求不匹配")
    terminal["webui_result"] = _materialize_stream_webui_result(
        client, config, target, terminal,
    )
    terminal["display_source"] = "loaded_existing_remote_gpu_stream"
    return terminal


def _materialize_stream_webui_result(client, config, target, terminal):
    """Create browser-ready evidence files from an immutable stream session."""

    create_request = (client.ledger.snapshot().get("create") or {}).get("request") or {}
    sport_id = str(((create_request.get("configuration") or {}).get("sport_id") or "badminton")).lower()
    try:
        if sport_id == "tennis":
            # Tennis first phase exposes only visual person evidence.  Do not
            # call the badminton business metrics / energy / ability module.
            from webui.stream_speed_summary import summarize_stream_player_speeds

            derivation = summarize_stream_player_speeds(
                target,
                client=client,
                terminal_status=terminal,
                create_request=create_request,
            )
        else:
            from business_gateway.streaming.derivation import derive_stream_movement_metrics

            derivation = derive_stream_movement_metrics(
                target,
                client=client,
                terminal_status=terminal,
                create_request=create_request,
            )
        movement_metrics_path = derivation.get("movement_metrics_path")
    except (OSError, RuntimeError, StreamAPIError, ValueError) as exc:
        derivation = {"status": "failed", "reason": str(exc)}
        movement_metrics_path = None
    photos = _download_stream_candidate_photos(
        config,
        terminal,
        target,
        candidate_records=_candidate_photo_records_from_events(
            terminal,
            derivation.get("raw_events_path") if isinstance(derivation, dict) else None,
        ),
    )
    return {
        "analysis_output_dir": str(target),
        "movement_metrics_path": movement_metrics_path,
        "derivation": derivation,
        "candidate_photos": photos,
    }


def _download_stream_candidate_photos(config, terminal_status, output_dir, *, candidate_records=None):
    """Fetch known anonymous candidate crops into the local result directory.

    Only IDs present in the completed status are requested.  The restrictive
    identifier pattern and fixed API path prevent a remote status payload from
    becoming an arbitrary URL fetch.  A missing crop is a normal evidence
    limitation, never a reason to hide the remaining player data.
    """

    target = Path(output_dir) / "candidate_photos"
    output = []
    session_id = str((terminal_status or {}).get("analysis_session_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        return output
    candidates = candidate_records or (terminal_status or {}).get("track_candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        track_id = str(candidate.get("track_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", track_id):
            continue
        photo = candidate.get("candidate_photo") or {}
        record = {
            "track_id": track_id,
            "source_time_sec": photo.get("source_time_sec"),
            "capture_quality": photo.get("capture_quality"),
            "frontal_score": photo.get("frontal_score"),
            "view_label": photo.get("view_label"),
            "selection_policy": photo.get("selection_policy"),
            "status": "not_available",
        }
        if not photo:
            output.append(record)
            continue
        request = Request(
            f"{config['base_url']}/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}",
            headers={"X-API-Key": config["api_key"]},
            method="GET",
        )
        try:
            with urlopen(request, timeout=config["timeout_seconds"]) as response:
                content_type = response.headers.get_content_type() or ""
                declared_size = int(response.headers.get("Content-Length") or 0)
                if content_type != "image/jpeg" or declared_size > 10 * 1024 * 1024:
                    raise ValueError("candidate photo response is not an acceptable JPEG")
                payload = response.read(10 * 1024 * 1024 + 1)
            if not payload or len(payload) > 10 * 1024 * 1024:
                raise ValueError("candidate photo exceeds the local size limit")
            target.mkdir(parents=True, exist_ok=True)
            destination = target / f"{track_id}.jpg"
            temporary = destination.with_suffix(".jpg.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            record.update({"status": "downloaded", "path": str(destination)})
        except (OSError, ValueError, TimeoutError):
            record["status"] = "unavailable"
        output.append(record)
    return output


def _candidate_photo_records_from_events(terminal_status, raw_events_path):
    """Restore photo metadata omitted from compact terminal status responses.

    The session's final status intentionally stays small.  Earlier immutable
    ``person_observation`` events retain the candidate-photo selection data,
    which has already been downloaded locally by the movement derivation.
    """

    candidates = {
        str(item.get("track_id")): dict(item)
        for item in (terminal_status or {}).get("track_candidates") or []
        if isinstance(item, dict) and item.get("track_id")
    }
    path = Path(raw_events_path) if raw_events_path else None
    if not path or not path.is_file():
        return list(candidates.values())
    try:
        with path.open("r", encoding="utf-8") as source:
            for raw in source:
                event = json.loads(raw)
                data = event.get("data") or {}
                track = data.get("track") or {}
                track_id = str(track.get("track_id") or "")
                photo = data.get("candidate_photo")
                if not track_id or not isinstance(photo, dict):
                    continue
                candidate = candidates.setdefault(track_id, {"track_id": track_id})
                current = candidate.get("candidate_photo") or {}
                current_score = float(current.get("selection_score") or -1.0)
                incoming_score = float(photo.get("selection_score") or photo.get("capture_quality") or 0.0)
                if incoming_score >= current_score:
                    candidate["candidate_photo"] = dict(photo)
    except (OSError, ValueError, TypeError):
        pass
    return [candidates[track_id] for track_id in sorted(candidates)]


def _remote_options(options):
    # GPU API intentionally rejects arbitrary model paths; it owns the pinned
    # deployed model artifacts.  Everything else is part of the public contract.
    return {
        key: value
        for key, value in options.items()
        if key not in {"yolo_pose_model", "ball_model", "generate_promotion_video"}
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
            "请检查当前 GPU 服务地址和远端 /api/v1/health。"
        ) from exc
    except OSError as exc:
        raise RemoteAnalysisError(
            "远端 GPU 上传连接失败；请检查当前 GPU 服务地址和远端服务。"
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
