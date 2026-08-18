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
from pathlib import Path
from typing import Iterator


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.background_sample_count <= 0:
        raise SystemExit("--batch-size and --background-sample-count must be positive")
    if not args.video_file.is_file() or not args.tracknet_file.is_file():
        raise SystemExit("Video or TrackNet checkpoint does not exist")

    # These imports intentionally resolve from the official checkout supplied
    # as the process cwd by run_tracknet_v3.py.
    import numpy as np
    import torch
    from PIL import Image
    from predict import predict
    from test import get_ensemble_weight
    from utils.general import HEIGHT, WIDTH, generate_frames, get_model, write_pred_csv

    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.tracknet_file, map_location="cpu", weights_only=False)
    param_dict = checkpoint["param_dict"]
    seq_len = int(param_dict["seq_len"])
    bg_mode = str(param_dict["bg_mode"])
    print(f"TrackNet runtime: seq_len={seq_len}, bg_mode={bg_mode}, batch={args.batch_size}", flush=True)

    started = time.perf_counter()
    bgr_frames = generate_frames(str(args.video_file))
    if not bgr_frames:
        raise SystemExit("Video contains no decodable frames")
    print(f"Decoded {len(bgr_frames)} frames in {time.perf_counter() - started:.1f}s", flush=True)
    height, width = bgr_frames[0].shape[:2]
    rgb_frames = [frame[..., ::-1] for frame in bgr_frames]

    background = _sample_background(np, rgb_frames, args.background_sample_count)
    processed = _preprocess_frames(np, Image, rgb_frames, background, bg_mode, WIDTH, HEIGHT)
    # Original-resolution frames are no longer needed.  Releasing them before
    # the sliding-window loop keeps peak host RAM bounded on 60 GiB instances.
    del bgr_frames, rgb_frames
    print(f"Prepared {len(processed)} resized frames for GPU batches", flush=True)

    model = get_model("TrackNet", seq_len, bg_mode).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint

    median_channels = _prepare_median_channels(np, Image, background, bg_mode, WIDTH, HEIGHT)
    predictions = _infer_weighted(
        np=np,
        torch=torch,
        model=model,
        processed=processed,
        median_channels=median_channels,
        sequence_length=seq_len,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode,
        image_size=(width, height),
        predict=predict,
        get_ensemble_weight=get_ensemble_weight,
        height=HEIGHT,
        width=WIDTH,
    )

    output_csv = args.save_dir / f"{args.video_file.stem}_ball.csv"
    write_pred_csv(predictions, save_file=str(output_csv))
    metadata = {
        "schema_version": "1.0",
        "implementation": "good_badminton_tracknetv3_sampled_background_v1",
        "measurement_kind": "tracknet_v3_raw_temporal_heatmap",
        "inpaint_enabled": False,
        "input_video": str(args.video_file),
        "frame_count": len(processed),
        "image_size": {"width": width, "height": height},
        "sequence_length": seq_len,
        "temporal_ensemble": args.eval_mode,
        "batch_size": args.batch_size,
        "background": {
            "method": "uniform_sample_median",
            "sample_count": min(len(processed), args.background_sample_count),
            "upstream_full_video_median_replaced": True,
        },
    }
    (args.save_dir / "tracknet_execution.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. raw_csv={output_csv}", flush=True)
    return 0


def _sample_background(np, frames: list, sample_count: int):
    indexes = np.linspace(0, len(frames) - 1, num=min(len(frames), sample_count), dtype=int)
    samples = np.stack([frames[index] for index in np.unique(indexes)], axis=0)
    return np.median(samples, axis=0).astype("uint8")


def _resize_to_chw(np, Image, image, width: int, height: int):
    resized = np.asarray(Image.fromarray(image).resize((width, height)))
    return np.moveaxis(resized, -1, 0)


def _preprocess_frames(np, Image, frames: list, background, bg_mode: str, width: int, height: int):
    if bg_mode not in {"", "concat", "subtract", "subtract_concat"}:
        raise ValueError(f"Unsupported TrackNetV3 bg_mode: {bg_mode!r}")
    channels = 1 if bg_mode == "subtract" else 4 if bg_mode == "subtract_concat" else 3
    output = np.empty((len(frames), channels, height, width), dtype="uint8")
    for index, frame in enumerate(frames):
        rgb = _resize_to_chw(np, Image, frame, width, height)
        if bg_mode == "subtract":
            # Keep the upstream operation order: per-pixel RGB difference,
            # channel sum, then image resize.
            difference = np.sum(np.abs(frame.astype("int16") - background.astype("int16")), axis=2)
            output[index, 0] = np.asarray(
                Image.fromarray(np.clip(difference, 0, 255).astype("uint8")).resize((width, height))
            )
        elif bg_mode == "subtract_concat":
            difference = np.sum(np.abs(frame.astype("int16") - background.astype("int16")), axis=2)
            output[index, :3] = rgb
            output[index, 3] = np.asarray(
                Image.fromarray(np.clip(difference, 0, 255).astype("uint8")).resize((width, height))
            )
        else:
            output[index] = rgb
    return output


def _prepare_median_channels(np, Image, background, bg_mode: str, width: int, height: int):
    if bg_mode != "concat":
        return None
    return _resize_to_chw(np, Image, background, width, height)


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
    processed,
    median_channels,
    sequence_length: int,
    batch_size: int,
    eval_mode: str,
    image_size: tuple[int, int],
    predict,
    get_ensemble_weight,
    height: int,
    width: int,
) -> dict:
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

    for batch_number, (indices_np, inputs_np) in enumerate(
        _iter_batches(np, processed, median_channels, sequence_length, batch_size),
        start=1,
    ):
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
    return results


if __name__ == "__main__":
    raise SystemExit(main())
