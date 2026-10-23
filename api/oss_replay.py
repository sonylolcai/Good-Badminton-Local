"""Replay committed OSS media through the same continuous engine as the GPU.

Run ``python -m api.oss_replay --help``. Download/replay uses bounded frame
iterators; it never creates one model/tracker per uploaded fragment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from badminton_analysis.streaming import AnalysisEngine, OpenCVSegmentDecoder
from badminton_analysis.streaming.models import FramePacket, FrameSegment

from .media_archive import MediaValidationError, revision_for, write_json
from .media_identity import media_prefix, validate_media_identity
from .media_storage import DirectoryObjectStore, oss_from_environment, safe_key, sha256_file
from .stream_models import SESSION_ID_PATTERN, SHA256_PATTERN
from .stream_runtime import StreamProcessorFactory
from .stream_sessions import StreamSessionManager
from .vision_profiles import get_vision_profile


def fetch_verified(storage, key, destination, digest, size=None):
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise MediaValidationError("archive object has no valid SHA-256")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == digest and (size is None or destination.stat().st_size == size):
        return destination
    temporary = destination.with_name(destination.name + ".downloading")
    storage.download(safe_key(key), temporary)
    if sha256_file(temporary) != digest or (size is not None and temporary.stat().st_size != size):
        temporary.unlink(missing_ok=True)
        raise MediaValidationError("downloaded object differs from committed manifest")
    os.replace(temporary, destination)
    return destination


def validate_manifest(manifest, manifest_key):
    if manifest.get("schema_version") != "media-archive.v1" or manifest.get("status") != "ready":
        raise MediaValidationError("a committed media-archive.v1 manifest is required")
    get_vision_profile(manifest.get("sport_id"))
    session_id = manifest.get("analysis_session_id")
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise MediaValidationError("invalid archive session identity")
    configuration = manifest.get("configuration", {})
    if configuration.get("sport_id") != manifest["sport_id"]:
        raise MediaValidationError("archive sport and analysis configuration disagree")
    # Validate wire fields and discard no unknown derived fields silently: the
    # profile synchronizer validates derived geometry when constructing runtime.
    from .stream_models import validate_segment_metadata
    entries = manifest.get("segments", [])
    if not entries or len(entries) != manifest.get("expected_last_segment_index", -1) + 1:
        raise MediaValidationError("archive does not cover its final index")
    base = safe_key(manifest_key).rsplit("/", 1)[0] + "/"
    previous = None
    total = 0
    for index, entry in enumerate(entries):
        if entry.get("segment_index") != index:
            raise MediaValidationError("archive segment indexes must be ordered and continuous")
        body = {key: value for key, value in entry.items() if key not in {"object_key", "decoded_frame_count"}}
        validate_segment_metadata({"schema_version": "stream-session.v1", **body})
        safe_key(entry["object_key"])
        if not entry["object_key"].startswith(base):
            raise MediaValidationError("segment object is outside the committed revision")
        count, start = entry.get("source_frame_count"), entry.get("source_frame_start_index")
        offsets = entry.get("source_frame_time_offsets_sec")
        if type(count) is not int or type(start) is not int or not isinstance(offsets, list) or len(offsets) != count:
            raise MediaValidationError("source frame identities and timestamps are required")
        if previous:
            if start != previous["source_frame_start_index"] + previous["source_frame_count"]:
                raise MediaValidationError("archive source frame gap or overlap")
            if abs(entry["source_start_time_sec"] - previous["source_start_time_sec"] - previous["duration_sec"]) > 0.002:
                raise MediaValidationError("archive source timeline gap or overlap")
        previous = entry
        total += count
    if total != manifest.get("total_source_frames"):
        raise MediaValidationError("archive total frame count disagrees with segments")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not SHA256_PATTERN.fullmatch(revision):
        raise MediaValidationError("invalid archive revision")
    snapshot_fields = ("schema_version", "analysis_session_id", "sport_id", "camera_id", "calibration_id",
                       "court_corners", "configuration", "expected_last_segment_index")
    snapshot = {field: manifest[field] for field in snapshot_fields}
    if "media_identity" in manifest:
        snapshot["media_identity"] = validate_media_identity(manifest["media_identity"],
            camera_id=manifest["camera_id"], calibration_id=manifest["calibration_id"])
    snapshot["segments"] = [{key: value for key, value in entry.items()
                             if key not in {"object_key", "decoded_frame_count"}} for entry in entries]
    expected_base = (media_prefix(snapshot["media_identity"]) + "/" if "media_identity" in snapshot else None)
    if revision_for(snapshot) != revision or (base != expected_base if expected_base else not base.endswith(f"/{session_id}/{revision}/")):
        raise MediaValidationError("archive input revision does not match manifest content")
    master = manifest["master"]
    if not safe_key(master["object_key"]).startswith(base):
        raise MediaValidationError("master object is outside the committed revision")
    return manifest


def _master_segments(path, entries):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaValidationError("cannot decode master")
    try:
        for entry in entries:
            descriptor = StreamSessionManager._descriptor_from_entry(entry)
            def frames(entry=entry):
                for local, offset in enumerate(entry["source_frame_time_offsets_sec"]):
                    ok, frame = capture.read()
                    if not ok:
                        raise MediaValidationError("master ended before declared source frames")
                    yield FramePacket(
                        frame=frame, source_frame_index=entry["source_frame_start_index"] + local,
                        source_time_sec=entry["source_start_time_sec"] + offset,
                        segment_index=entry["segment_index"],
                    )
            yield FrameSegment(descriptor, frames())
        if capture.read()[0]:
            raise MediaValidationError("master contains undeclared extra frames")
    finally:
        capture.release()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (root / "api", root / "badminton_analysis"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def replay_archive(storage, manifest_key, *, manifest_sha256, output_dir, source="fragments",
                   processing_mode="analysis", processor_factory=None, cache_dir=None):
    if source not in {"fragments", "master"} or processing_mode not in {"analysis", "decode_smoke"}:
        raise ValueError("unsupported replay source or processing mode")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior experimental run.
    started = time.perf_counter()
    run = {"status": "downloading", "source": source, "processing_mode": processing_mode,
           "model_inference_tested": False, "analysis_quality_verified": False,
           "python": platform.python_version(), "source_code_sha256": source_identity(),
           "manifest_object_key": safe_key(manifest_key), "manifest_sha256": manifest_sha256}
    from importlib.metadata import PackageNotFoundError, version
    run["dependencies"] = {}
    for package in ("torch", "ultralytics", "numpy", "opencv-python", "oss2"):
        try:
            run["dependencies"][package] = version(package)
        except PackageNotFoundError:
            run["dependencies"][package] = None
    import torch
    from badminton_analysis.detection.yolo_pose import resolve_ultralytics_device
    run["devices"] = {
        "pose": resolve_ultralytics_device(os.environ.get("GOOD_BADMINTON_STREAM_DEVICE", "auto")),
        "tracknet": os.environ.get("GOOD_BADMINTON_STREAM_TRACKNET_DEVICE", "auto"),
        "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
    }
    write_json(output / "run.json", run)
    try:
        manifest_path = fetch_verified(storage, manifest_key, output / "manifest.json", manifest_sha256)
        manifest = validate_manifest(json.loads(manifest_path.read_text()), manifest_key)
        cache = Path(cache_dir).resolve() if cache_dir else output / "input"
        cache = cache / manifest["revision"]
        session = {**manifest, "processing_mode": processing_mode}
        factory = processor_factory or StreamProcessorFactory(
            output, vision_profile=get_vision_profile(manifest["sport_id"]), processing_mode=processing_mode,
        )
        measurement, temporal = factory(session)
        engine = AnalysisEngine(session["analysis_session_id"], session["configuration"]["analysis_sample_hz"],
                                measurement, temporal_processor=temporal)
        if processor_factory is None and processing_mode == "analysis":
            weights = Path(__file__).resolve().parents[1] / "weights"
            pose = Path(factory._profile_environment("POSE_MODEL", str(weights / factory.vision_profile.default_pose_checkpoint)))
            run["model_sha256"] = {"pose": sha256_file(pose)}
            if session["configuration"]["shuttle_detector"] == "tracknet_v3":
                run["model_sha256"]["tracknet"] = sha256_file(factory._tracknet_model_path())
        if source == "master":
            master = manifest["master"]
            path = fetch_verified(storage, master["object_key"], cache / "master.mp4", master["sha256"], master["content_length_bytes"])
            segments = _master_segments(path, manifest["segments"])
        else:
            def fragments():
                decoder = OpenCVSegmentDecoder()
                for entry in manifest["segments"]:
                    path = fetch_verified(storage, entry["object_key"], cache / f"{entry['segment_index']:08d}.mp4",
                                          entry["sha256"], entry["content_length_bytes"])
                    yield decoder.decode(path, StreamSessionManager._descriptor_from_entry(entry))
            segments = fragments()
        run.update({"status": "running", "sport_id": manifest["sport_id"], "configuration": session["configuration"]})
        write_json(output / "run.json", run)
        events_count = 0
        with (output / "events.jsonl").open("w", encoding="utf-8") as events:
            def save_events(rows):
                nonlocal events_count
                for row in rows:
                    events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    events_count += 1
                events.flush()
            try:
                for segment in segments:
                    result = engine.process_segment(segment)
                    save_events(result.events)
                    write_json(output / "checkpoint.json", result.checkpoint)
                save_events(engine.finalize().events)
            finally:
                close = getattr(segments, "close", None)
                if callable(close):
                    close()
        run.update({"status": "complete", "source_frames": engine.source_frames, "events": events_count,
                    "model_inference_tested": processing_mode == "analysis" and isinstance(factory, StreamProcessorFactory),
                    "elapsed_seconds": round(time.perf_counter() - started, 3)})
        write_json(output / "run.json", run)
        return run
    except Exception as error:
        run.update({"status": "failed", "error_type": type(error).__name__})
        write_json(output / "run.json", run)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True, help="From the authenticated GPU media status endpoint")
    parser.add_argument("--output", required=True, help="New run directory; must not already exist")
    parser.add_argument("--cache-dir")
    parser.add_argument("--source", choices=["fragments", "master"], default="fragments")
    parser.add_argument("--processing-mode", choices=["analysis", "decode_smoke"], default="analysis")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--local-object-root", help="Explicit local verification backend; does not contact OSS")
    args = parser.parse_args()
    for name in ("GOOD_BADMINTON_STREAM_DEVICE", "GOOD_TENNIS_STREAM_DEVICE", "GOOD_BADMINTON_STREAM_TRACKNET_DEVICE"):
        os.environ[name] = args.device
    checkpoint_dir = Path(args.output).resolve().parent / ".oss-download-checkpoints"
    storage = DirectoryObjectStore(args.local_object_root) if args.local_object_root else oss_from_environment(checkpoint_dir)
    result = replay_archive(storage, args.manifest_key, manifest_sha256=args.manifest_sha256,
                            output_dir=args.output, source=args.source, processing_mode=args.processing_mode,
                            cache_dir=args.cache_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
