"""Standalone Ultralytics runner for the three fixed-camera benchmark methods."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


METHODS = ("full_640", "full_1280", "full_640+far_roi_640")


@dataclass(frozen=True)
class MethodOutput:
    detections: list[dict]
    elapsed_ms: float


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        roi = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("ROI must be four comma-separated normalized values") from exc
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return roi


def roi_to_pixels(roi_norm: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi_norm
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )


def nms_merge(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Confidence NMS that retains provenance from suppressed duplicates."""
    if not detections:
        return []
    boxes = [detection["bbox_xyxy"] for detection in detections]
    scores = [float(detection["confidence"]) for detection in detections]
    xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    kept_indices = [int(index) for index in np.asarray(indices).reshape(-1)] if len(indices) else []
    merged = []
    for kept_index in kept_indices:
        best = dict(detections[kept_index])
        sources = {str(best["source"])}
        best_box = best["bbox_xyxy"]
        for index, candidate in enumerate(detections):
            if index == kept_index:
                continue
            if _iou(best_box, candidate["bbox_xyxy"]) >= iou_threshold:
                sources.add(str(candidate["source"]))
        best["sources"] = sorted(sources)
        merged.append(best)
    return merged


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


class UltralyticsBenchmarkRunner:
    def __init__(self, model_path: str, device: str, confidence: float, inference_iou: float, merge_iou: float):
        from ultralytics import YOLO

        self.model_path = model_path
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.confidence = confidence
        self.inference_iou = inference_iou
        self.merge_iou = merge_iou
        self.model = YOLO(model_path)

    def warmup(self, frame: np.ndarray, roi_norm: Sequence[float]) -> None:
        """Warm both input sizes and the ROI path outside measured timings."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = roi_to_pixels(roi_norm, width, height)
        self._predict(frame, imgsz=640, source="warmup_full")
        self._predict(frame, imgsz=1280, source="warmup_full")
        self._predict(frame[y1:y2, x1:x2], imgsz=640, source="warmup_roi", offset=(x1, y1), roi_px=(x1, y1, x2, y2))
        self._synchronize()

    def run(self, frame: np.ndarray, method: str, roi_norm: Sequence[float]) -> MethodOutput:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self._synchronize()
        start = time.perf_counter()
        if method == "full_640":
            detections = self._predict(frame, imgsz=640, source="full_frame")
        elif method == "full_1280":
            detections = self._predict(frame, imgsz=1280, source="full_frame")
        else:
            height, width = frame.shape[:2]
            roi_px = roi_to_pixels(roi_norm, width, height)
            x1, y1, x2, y2 = roi_px
            full = self._predict(frame, imgsz=640, source="full_frame")
            roi = self._predict(frame[y1:y2, x1:x2], imgsz=640, source="far_roi", offset=(x1, y1), roi_px=roi_px)
            detections = nms_merge(full + roi, self.merge_iou)
        self._synchronize()
        return MethodOutput(detections=detections, elapsed_ms=(time.perf_counter() - start) * 1000.0)

    def _predict(
        self,
        image: np.ndarray,
        imgsz: int,
        source: str,
        offset: tuple[int, int] = (0, 0),
        roi_px: tuple[int, int, int, int] | None = None,
    ) -> list[dict]:
        result = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=self.confidence,
            iou=self.inference_iou,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        keypoints = None
        keypoint_confidences = None
        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints = result.keypoints.xy.detach().cpu().numpy()
            if result.keypoints.conf is not None:
                keypoint_confidences = result.keypoints.conf.detach().cpu().numpy()
        detections = []
        ox, oy = offset
        for index, (box, confidence) in enumerate(zip(boxes, confidences)):
            mapped_box = [float(box[0] + ox), float(box[1] + oy), float(box[2] + ox), float(box[3] + oy)]
            mapped_keypoints = None
            if keypoints is not None and index < len(keypoints):
                mapped_keypoints = [[float(x + ox), float(y + oy)] for x, y in keypoints[index]]
            detections.append(
                {
                    "bbox_xyxy": mapped_box,
                    "confidence": float(confidence),
                    "source": source,
                    "sources": [source],
                    "keypoints_xy": mapped_keypoints,
                    "keypoint_confidence": (
                        [float(value) for value in keypoint_confidences[index]]
                        if keypoint_confidences is not None and index < len(keypoint_confidences)
                        else None
                    ),
                    "inference": {
                        "imgsz": imgsz,
                        "device": str(self.device),
                        "confidence_threshold": self.confidence,
                        "iou_threshold": self.inference_iou,
                        "source_image_width": int(image.shape[1]),
                        "source_image_height": int(image.shape[0]),
                        "roi_xyxy": list(roi_px) if roi_px else None,
                    },
                }
            )
        return detections

    @staticmethod
    def _synchronize() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    @staticmethod
    def _resolve_device(device: str):
        if device not in ("auto", ""):
            return device
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
