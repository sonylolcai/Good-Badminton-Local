#!/usr/bin/env python
"""Run official TrackNetV3 weights with bounded, observable preprocessing.

The upstream ``predict.py`` computes a pixel-wise median over every decoded
frame before it begins inference.  For a 2-5 minute fixed-camera match this is
both needlessly expensive and opaque: it can hold the GPU idle for minutes.

This adapter retains the upstream architecture, checkpoint, image size and
temporal ``weight`` ensemble.  Its only intentional preprocessing difference
is a uniformly sampled background median, recorded beside the CSV.  It emits
raw TrackNet measurements only; InpaintNet remains a separate, inferred B*
stream and is not used in the first A/B pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterator


EVENT_PREFIX = "GOOD_BADMINTON_TRACKNET_EVENT="


def emit_event(stage: str, **details) -> None:
    """Write a machine-readable heartbeat for the API job manifest.

    The outer process forwards these lines directly into the durable job
    timing trace.  Human-readable logs remain below for SSH diagnosis.
    """
    payload = {"stage": stage, **details}
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-file", required=True, type=Path)
    parser.add_argument("--tracknet-file", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-mode", choices=("average", "weight"), default="weight")
    parser.add_argument(
        "--background-sample-count",
        type=int,
        default=120,
        help="Uniformly sampled frames for the background median; recorded in execution metadata.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=96,
        help=(
            "Maximum decoded/preprocessed frames retained at once.  This keeps host "
            "memory bounded regardless of match duration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.background_sample_count <= 0 or args.chunk_frames <= 0:
        raise SystemExit("--batch-size, --background-sample-count and --chunk-frames must be positive")
    if not args.video_file.is_file() or not args.tracknet_file.is_file():
        raise SystemExit("Video or TrackNet checkpoint does not exist")

    # These imports intentionally resolve from the official checkout supplied
    # as the process cwd by run_tracknet_v3.py.
    import numpy as np
    import torch
    from PIL import Image
    from predict import predict
    from test import get_ensemble_weight
    import cv2
    from utils.general import HEIGHT, WIDTH, get_model, write_pred_csv

    args.save_dir.mkdir(parents=True, exist_ok=True)
    emit_event("checkpoint_load")
    checkpoint = torch.load(args.tracknet_file, map_location="cpu", weights_only=False)
    param_dict = checkpoint["param_dict"]
    seq_len = int(param_dict["seq_len"])
    bg_mode = str(param_dict["bg_mode"])
    print(f"TrackNet runtime: seq_len={seq_len}, bg_mode={bg_mode}, batch={args.batch_size}", flush=True)

    emit_event("video_probe")
    video_info = _probe_video(cv2, args.video_file)
    reported_frame_count = video_info["frame_count"]
    width = video_info["width"]
    height = video_info["height"]
    print(
        f"Video probe: {reported_frame_count} reported frames, {width}x{height}, "
        f"{video_info['fps']:.3f} FPS",
        flush=True,
    )
    emit_event("background_sampling", frame_count=reported_frame_count, sample_target=args.background_sample_count)
    background, sampled_frame_count, frame_count = _sample_resized_background(
        cv2=cv2,
        np=np,
        Image=Image,
        video_file=args.video_file,
        frame_count=reported_frame_count,
        sample_count=args.background_sample_count,
        width=WIDTH,
        height=HEIGHT,
    )
    if frame_count != reported_frame_count:
        print(
            f"Video metadata reported {reported_frame_count} frames; sequential decode confirmed {frame_count}.",
            flush=True,
        )
    emit_event(
        "background_sampling",
        frame_count=frame_count,
        sampled_frame_count=sampled_frame_count,
        complete=True,
    )

    emit_event("model_initialize", frame_count=frame_count)
    model = get_model("TrackNet", seq_len, bg_mode).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint
    emit_event("inference", frame_count=frame_count)

    median_channels = _prepare_median_channels(np, Image, background, bg_mode, WIDTH, HEIGHT)
    processed_chunks = _iter_preprocessed_chunks(
        cv2=cv2,
        np=np,
        Image=Image,
        video_file=args.video_file,
        background=background,
        bg_mode=bg_mode,
        width=WIDTH,
        height=HEIGHT,
        chunk_frames=args.chunk_frames,
        expected_frame_count=frame_count,
        event_cb=emit_event,
    )
    predictions = _infer_weighted(
        np=np,
        torch=torch,
        model=model,
        processed=None,
        batch_iterator=_iter_stream_batches(
            np,
            processed_chunks,
            median_channels,
            sequence_length=seq_len,
            batch_size=args.batch_size,
            frame_count=frame_count,
        ),
        frame_count=frame_count,
        median_channels=median_channels,
        sequence_length=seq_len,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode,
        image_size=(width, height),
        predict=predict,
        get_ensemble_weight=get_ensemble_weight,
        height=HEIGHT,
        width=WIDTH,
        event_cb=emit_event,
    )

    output_csv = args.save_dir / f"{args.video_file.stem}_ball.csv"
    emit_event("csv_export", frame_count=frame_count)
    write_pred_csv(predictions, save_file=str(output_csv))
    metadata = {
        "schema_version": "1.0",
        "implementation": "good_badminton_tracknetv3_bounded_stream_v2",
        "measurement_kind": "tracknet_v3_raw_temporal_heatmap",
        "inpaint_enabled": False,
        "input_video": str(args.video_file),
        "frame_count": frame_count,
        "container_reported_frame_count": reported_frame_count,
        "image_size": {"width": width, "height": height},
        "sequence_length": seq_len,
        "temporal_ensemble": args.eval_mode,
        "batch_size": args.batch_size,
        "chunk_frames": args.chunk_frames,
        "memory_policy": {
            "kind": "bounded_decode_and_preprocess_chunks",
            "maximum_chunk_frames": args.chunk_frames,
            "sequence_overlap_frames": max(0, seq_len - 1),
            "note": "Peak host memory does not grow with full video duration.",
        },
        "background": {
            "method": "uniform_sample_median_resized",
            "sample_count": sampled_frame_count,
            "resolution": {"width": WIDTH, "height": HEIGHT},
            "upstream_full_video_median_replaced": True,
        },
    }
    (args.save_dir / "tracknet_execution.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    emit_event("complete", frame_count=frame_count, raw_csv=str(output_csv))
    print(f"Done. raw_csv={output_csv}", flush=True)
    return 0


def _sample_background(np, frames: list, sample_count: int):
    indexes = np.linspace(0, len(frames) - 1, num=min(len(frames), sample_count), dtype=int)
    samples = np.stack([frames[index] for index in np.unique(indexes)], axis=0)
    return np.median(samples, axis=0).astype("uint8")


def _probe_video(cv2, video_file: Path) -> dict:
    """Read lightweight container metadata without retaining any video frames."""
    capture = cv2.VideoCapture(str(video_file))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_file}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is incomplete; bounded TrackNet processing requires frame count and size")
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps}


def _sample_resized_background(
    *, cv2, np, Image, video_file: Path, frame_count: int, sample_count: int, width: int, height: int
):
    """Build the median background from bounded, model-sized sample frames.

    The historical adapter retained every original-resolution frame before
    reducing it.  TrackNet's deployed ``concat`` checkpoint only consumes the
    model-sized median channels, so sampling directly at this size preserves
    the intended background input while preventing a duration-sized RAM peak.
    """
    sample_indexes = np.unique(
        np.linspace(0, frame_count - 1, num=min(frame_count, sample_count), dtype=int)
    )
    samples = []
    capture = cv2.VideoCapture(str(video_file))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for background samples: {video_file}")
        # Sequential grabs are predictable for compressed video.  Seeking to
        # 120 sample positions can repeatedly decode entire GOPs and is both
        # slower and less portable across ffmpeg/OpenCV builds.
        sample_position = 0
        frame_index = 0
        while True:
            if not capture.grab():
                break
            if sample_position < len(sample_indexes) and frame_index == int(sample_indexes[sample_position]):
                ok, bgr = capture.retrieve()
                if ok and bgr is not None:
                    samples.append(_resize_rgb(np, Image, bgr[..., ::-1], width, height))
                sample_position += 1
            frame_index += 1
    finally:
        capture.release()
    if not samples:
        raise RuntimeError("Unable to decode any background sample frames")
    return np.median(np.stack(samples, axis=0), axis=0).astype("uint8"), len(samples), frame_index


def _resize_rgb(np, Image, image, width: int, height: int):
    return np.asarray(Image.fromarray(image).resize((width, height)))


def _resize_to_chw(np, Image, image, width: int, height: int):
    resized = _resize_rgb(np, Image, image, width, height)
    return np.moveaxis(resized, -1, 0)


def _preprocess_frame(np, Image, frame, background, bg_mode: str, width: int, height: int):
    """Preprocess one RGB frame; callers discard the original immediately."""
    if bg_mode not in {"", "concat", "subtract", "subtract_concat"}:
        raise ValueError(f"Unsupported TrackNetV3 bg_mode: {bg_mode!r}")
    resized_rgb = _resize_rgb(np, Image, frame, width, height)
    rgb = np.moveaxis(resized_rgb, -1, 0)
    if bg_mode == "subtract":
        difference = np.sum(np.abs(resized_rgb.astype("int16") - background.astype("int16")), axis=2)
        return np.clip(difference, 0, 255).astype("uint8")[None, ...]
    if bg_mode == "subtract_concat":
        difference = np.sum(np.abs(resized_rgb.astype("int16") - background.astype("int16")), axis=2)
        return np.concatenate((rgb, np.clip(difference, 0, 255).astype("uint8")[None, ...]), axis=0)
    return rgb


def _preprocess_frames(np, Image, frames: list, background, bg_mode: str, width: int, height: int):
    if bg_mode not in {"", "concat", "subtract", "subtract_concat"}:
        raise ValueError(f"Unsupported TrackNetV3 bg_mode: {bg_mode!r}")
    return np.stack(
        [_preprocess_frame(np, Image, frame, background, bg_mode, width, height) for frame in frames],
        axis=0,
    )


def _prepare_median_channels(np, Image, background, bg_mode: str, width: int, height: int):
    if bg_mode != "concat":
        return None
    return np.moveaxis(background, -1, 0)


def _iter_preprocessed_chunks(
    *, cv2, np, Image, video_file: Path, background, bg_mode: str, width: int, height: int,
    chunk_frames: int, expected_frame_count: int, event_cb=None,
) -> Iterator[tuple[int, object]]:
    """Sequentially decode and release chunks instead of retaining a match in RAM."""
    capture = cv2.VideoCapture(str(video_file))
    started = time.perf_counter()
    decoded = 0
    chunk_start = 0
    chunk = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for sequential decode: {video_file}")
        while True:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            chunk.append(_preprocess_frame(np, Image, bgr[..., ::-1], background, bg_mode, width, height))
            decoded += 1
            if len(chunk) < chunk_frames:
                continue
            elapsed = time.perf_counter() - started
            _emit_chunk_event(
                event_cb,
                "frame_preprocess",
                frames_completed=decoded,
                frame_count=expected_frame_count,
                chunk_frames=len(chunk),
                elapsed_seconds=round(elapsed, 3),
            )
            yield chunk_start, np.stack(chunk, axis=0)
            chunk_start = decoded
            chunk = []
        if chunk:
            elapsed = time.perf_counter() - started
            _emit_chunk_event(
                event_cb,
                "frame_preprocess",
                frames_completed=decoded,
                frame_count=expected_frame_count,
                chunk_frames=len(chunk),
                elapsed_seconds=round(elapsed, 3),
            )
            yield chunk_start, np.stack(chunk, axis=0)
    finally:
        capture.release()
    if decoded != expected_frame_count:
        raise RuntimeError(
            f"Sequential decode returned {decoded} frames but video metadata reported {expected_frame_count}; "
            "refusing to silently misalign TrackNet frame indexes"
        )


def _emit_chunk_event(event_cb, stage: str, **details) -> None:
    """Use the injected sink in tests and the normal JSON heartbeat in production."""
    (event_cb or emit_event)(stage, **details)


def _iter_stream_batches(
    np,
    processed_chunks,
    median_channels,
    *,
    sequence_length: int,
    batch_size: int,
    frame_count: int,
) -> Iterator[tuple]:
    """Emit ordered sliding-window batches while retaining only one short window.

    It intentionally yields the same valid windows as :func:`_iter_batches`.
    The prediction ensemble still owns the sequence tail, so no padded windows
    are submitted for normal videos and output rows stay frame-aligned.
    """
    if frame_count <= 0:
        return
    window = deque(maxlen=sequence_length)
    index_batch = []
    input_batch = []
    emitted_windows = 0
    for chunk_start, processed_chunk in processed_chunks:
        for offset, frame in enumerate(processed_chunk):
            frame_index = chunk_start + offset
            window.append((frame_index, frame))
            if len(window) != sequence_length:
                continue
            frame_indexes = np.asarray([item[0] for item in window], dtype=int)
            input_batch.append(np.stack([item[1] for item in window], axis=0))
            index_batch.append(frame_indexes)
            emitted_windows += 1
            if len(input_batch) < batch_size:
                continue
            yield _make_stream_batch(np, index_batch, input_batch, median_channels)
            index_batch = []
            input_batch = []
    if emitted_windows == 0 and window:
        # Preserve legacy short-clip behaviour: one last-frame-padded window
        # is emitted and the ensemble writes the remaining sequence tail.
        indexes = [item[0] for item in window]
        frames = [item[1] for item in window]
        while len(frames) < sequence_length:
            indexes.append(indexes[-1])
            frames.append(frames[-1])
        index_batch.append(np.asarray(indexes, dtype=int))
        input_batch.append(np.stack(frames, axis=0))
    if input_batch:
        yield _make_stream_batch(np, index_batch, input_batch, median_channels)


def _make_stream_batch(np, index_batch: list, input_batch: list, median_channels):
    sequence_indexes = np.stack(index_batch, axis=0)
    frame_tensor = np.stack(input_batch, axis=0)
    count, sequence_length, channels, height, width = frame_tensor.shape
    frame_tensor = frame_tensor.reshape(count, sequence_length * channels, height, width)
    if median_channels is not None:
        median_batch = np.broadcast_to(median_channels, (count,) + median_channels.shape)
        frame_tensor = np.concatenate((median_batch, frame_tensor), axis=1)
    sample_indices = np.stack((np.zeros_like(sequence_indexes), sequence_indexes), axis=2)
    return sample_indices, frame_tensor


def _iter_batches(np, processed, median_channels, sequence_length: int, batch_size: int) -> Iterator[tuple]:
    frame_count = len(processed)
    window_count = max(1, frame_count - sequence_length + 1)
    for start in range(0, window_count, batch_size):
        count = min(batch_size, window_count - start)
        sequence_starts = np.arange(start, start + count, dtype=int)
        frame_indexes = sequence_starts[:, None] + np.arange(sequence_length, dtype=int)[None, :]
        frame_indexes = np.clip(frame_indexes, 0, frame_count - 1)
        frame_tensor = processed[frame_indexes].reshape(count, -1, processed.shape[2], processed.shape[3])
        if median_channels is not None:
            median_batch = np.broadcast_to(median_channels, (count,) + median_channels.shape)
            frame_tensor = np.concatenate((median_batch, frame_tensor), axis=1)
        sample_indices = np.stack(
            (np.zeros_like(frame_indexes), frame_indexes),
            axis=2,
        )
        yield sample_indices, frame_tensor


def _infer_weighted(
    *,
    np,
    torch,
    model,
    processed=None,
    batch_iterator=None,
    frame_count=None,
    median_channels,
    sequence_length: int,
    batch_size: int,
    eval_mode: str,
    image_size: tuple[int, int],
    predict,
    get_ensemble_weight,
    height: int,
    width: int,
    event_cb=None,
) -> dict:
    if processed is None and batch_iterator is None:
        raise ValueError("Either processed frames or a bounded batch iterator is required")
    if frame_count is None:
        if processed is None:
            raise ValueError("frame_count is required with a bounded batch iterator")
        frame_count = len(processed)
    image_scaler = (image_size[0] / width, image_size[1] / height)
    results = {"Frame": [], "X": [], "Y": [], "Visibility": [], "Inpaint_Mask": []}
    buffer_size = sequence_length - 1
    batch_i = torch.arange(sequence_length)
    frame_i = torch.arange(sequence_length - 1, -1, -1)
    prediction_buffer = torch.zeros((buffer_size, sequence_length, height, width), dtype=torch.float32)
    weight = get_ensemble_weight(sequence_length, eval_mode)
    window_count = max(1, frame_count - sequence_length + 1)
    sample_count = 0
    total_batches = math.ceil(window_count / batch_size)
    started = time.perf_counter()
    if event_cb is not None:
        event_cb(
            "inference",
            batch_completed=0,
            batch_total=total_batches,
            windows_completed=0,
            windows_total=window_count,
        )

    if batch_iterator is None:
        batch_iterator = _iter_batches(np, processed, median_channels, sequence_length, batch_size)
    for batch_number, (indices_np, inputs_np) in enumerate(batch_iterator, start=1):
        indices = torch.from_numpy(indices_np)
        inputs = torch.from_numpy(inputs_np.astype("float32", copy=False)).div_(255.0).cuda()
        with torch.no_grad():
            batch_prediction = model(inputs).detach().cpu()
        prediction_buffer = torch.cat((prediction_buffer, batch_prediction), dim=0)
        ensemble_indices = []
        ensemble_predictions = []

        for row in range(indices.shape[0]):
            if sample_count < buffer_size:
                ensemble_prediction = prediction_buffer[batch_i + row, frame_i].sum(0) / (sample_count + 1)
            else:
                ensemble_prediction = (prediction_buffer[batch_i + row, frame_i] * weight[:, None, None]).sum(0)
            ensemble_indices.append(indices[row][0].reshape(1, 1, 2))
            ensemble_predictions.append(ensemble_prediction.reshape(1, 1, height, width))
            sample_count += 1
            if sample_count == window_count:
                padded = torch.zeros((buffer_size, sequence_length, height, width), dtype=torch.float32)
                prediction_buffer = torch.cat((prediction_buffer, padded), dim=0)
                for tail in range(1, sequence_length):
                    tail_prediction = prediction_buffer[batch_i + row + tail, frame_i].sum(0) / (sequence_length - tail)
                    ensemble_indices.append(indices[-1][tail].reshape(1, 1, 2))
                    ensemble_predictions.append(tail_prediction.reshape(1, 1, height, width))

        temporary = predict(
            torch.cat(ensemble_indices, dim=0),
            y_pred=torch.cat(ensemble_predictions, dim=0),
            img_scaler=image_scaler,
        )
        for key, values in temporary.items():
            results[key].extend(values)
        prediction_buffer = prediction_buffer[-buffer_size:]

        if batch_number == 1 or batch_number == total_batches or batch_number % max(1, total_batches // 20) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"Inference progress: batch {batch_number}/{total_batches}, "
                f"windows {sample_count}/{window_count}, elapsed {elapsed:.1f}s",
                flush=True,
            )
            if event_cb is not None:
                event_cb(
                    "inference",
                    batch_completed=batch_number,
                    batch_total=total_batches,
                    windows_completed=sample_count,
                    windows_total=window_count,
                    elapsed_seconds=round(elapsed, 3),
                )
    return results


if __name__ == "__main__":
    raise SystemExit(main())


