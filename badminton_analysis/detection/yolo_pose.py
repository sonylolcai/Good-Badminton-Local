from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np


def resolve_ultralytics_device(device="auto"):
    """Choose CUDA, Apple MPS, or CPU for Ultralytics inference.

    MPS is intentionally checked separately from CUDA: Apple Silicon exposes
    neither a CUDA device nor an integer GPU index.
    """
    requested = "auto" if device is None else str(device).strip().lower()
    if requested not in {"", "auto"}:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YOLOPoseProcessor:
    """Ultralytics YOLO pose processor with inspectable fixed-camera inference.

    ``process_frame`` keeps the original ``(keypoints, scores)`` return contract.
    Consumers that need bounding boxes and inference provenance can use
    ``process_frame_detailed`` / ``process_fixed_camera`` or inspect
    ``get_last_detections`` after the legacy call.
    """

    SUPPORTED_IMAGE_SIZES = (640, 960, 1280)

    def __init__(
        self,
        model_path="yolo11n-pose.pt",
        device="auto",
        conf=0.25,
        imgsz=640,
        asymmetric=False,
        far_roi=None,
        merge_iou=0.45,
        model=None,
    ):
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = self._validate_imgsz(imgsz)
        self.asymmetric = bool(asymmetric)
        self.far_roi = far_roi
        self.merge_iou = float(merge_iou)
        if not 0.0 <= self.merge_iou <= 1.0:
            raise ValueError("merge_iou must be between 0 and 1")
        self.inference_name = "YOLO-Pose"
        self._last_detections = []

        self.device = resolve_ultralytics_device(device)

        if model is None:
            from ultralytics import YOLO

            print(
                "Initializing YOLO pose model "
                f"(model: {self.model_path}, device: {self.device}, imgsz: {self.imgsz})"
            )
            model = YOLO(self.model_path)
        self.model = model

    @classmethod
    def _validate_imgsz(cls, imgsz):
        try:
            value = int(imgsz)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"imgsz must be one of {cls.SUPPORTED_IMAGE_SIZES}") from exc
        if value not in cls.SUPPORTED_IMAGE_SIZES:
            raise ValueError(f"imgsz must be one of {cls.SUPPORTED_IMAGE_SIZES}, got {imgsz}")
        return value

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _bbox_from_keypoints(keypoints):
        points = np.asarray(keypoints, dtype=float)
        valid = np.isfinite(points[:, :2]).all(axis=1)
        valid &= (points[:, 0] > 1) & (points[:, 1] > 1)
        if not np.any(valid):
            return np.zeros(4, dtype=float)
        visible = points[valid, :2]
        return np.asarray(
            [visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()],
            dtype=float,
        )

    def _infer(self, frame, *, imgsz, source, roi):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be an image array")
        imgsz = self._validate_imgsz(imgsz)
        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            imgsz=imgsz,
            verbose=False,
        )
        if results is None or len(results) == 0:
            return []
        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return []

        keypoints = self._to_numpy(result.keypoints.xy)
        if keypoints is None or keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return []
        keypoint_scores = self._to_numpy(getattr(result.keypoints, "conf", None))

        boxes_obj = getattr(result, "boxes", None)
        boxes = self._to_numpy(getattr(boxes_obj, "xyxy", None))
        box_scores = self._to_numpy(getattr(boxes_obj, "conf", None))
        if box_scores is not None:
            box_scores = box_scores.reshape(-1)

        offset_x, offset_y = int(roi[0]), int(roi[1])
        detections = []
        for index, person_keypoints in enumerate(keypoints):
            person_keypoints = np.asarray(person_keypoints, dtype=float).copy()
            person_keypoints[:, 0] += offset_x
            person_keypoints[:, 1] += offset_y

            if boxes is not None and index < len(boxes):
                bbox = np.asarray(boxes[index], dtype=float).reshape(-1)[:4].copy()
                bbox[[0, 2]] += offset_x
                bbox[[1, 3]] += offset_y
            else:
                bbox = self._bbox_from_keypoints(person_keypoints)

            scores = None
            if keypoint_scores is not None and index < len(keypoint_scores):
                scores = np.asarray(keypoint_scores[index], dtype=float).copy()

            if box_scores is not None and index < len(box_scores):
                confidence = float(box_scores[index])
            elif scores is not None and scores.size:
                confidence = float(np.nanmean(scores))
            else:
                confidence = 0.0

            detections.append(
                {
                    "bbox": bbox,
                    "keypoints": person_keypoints,
                    "keypoint_scores": scores,
                    "confidence": confidence,
                    "source": source,
                    "merged_sources": [source],
                    "inference": {
                        "model": str(self.model_path),
                        "imgsz": imgsz,
                        "conf": self.conf,
                        "device": self.device,
                        "roi": [int(value) for value in roi],
                        "input_shape": [int(frame.shape[0]), int(frame.shape[1])],
                    },
                }
            )
        return detections

    @staticmethod
    def _resolve_roi(frame_shape, roi):
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if roi is None:
            return (0, 0, width, max(1, height // 2))
        if isinstance(roi, dict):
            roi = (roi["x1"], roi["y1"], roi["x2"], roi["y2"])
        if not isinstance(roi, Sequence) or len(roi) != 4:
            raise ValueError("far_roi must contain (x1, y1, x2, y2)")

        values = [float(value) for value in roi]
        if all(0.0 <= value <= 1.0 for value in values):
            x1, y1, x2, y2 = (
                round(values[0] * width),
                round(values[1] * height),
                round(values[2] * width),
                round(values[3] * height),
            )
        else:
            x1, y1, x2, y2 = (round(value) for value in values)

        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"far_roi resolves to an empty crop: {(x1, y1, x2, y2)}")
        return (x1, y1, x2, y2)

    @staticmethod
    def _bbox_iou(first, second):
        a = np.asarray(first, dtype=float)
        b = np.asarray(second, dtype=float)
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _pose_quality(detection):
        scores = detection.get("keypoint_scores")
        if scores is None or np.asarray(scores).size == 0:
            keypoint_quality = 0.0
            visible_count = 0
        else:
            scores = np.asarray(scores, dtype=float)
            visible = np.isfinite(scores) & (scores > 0.05)
            visible_count = int(np.count_nonzero(visible))
            keypoint_quality = float(np.mean(scores[visible])) if visible_count else 0.0
        return (visible_count, keypoint_quality, float(detection.get("confidence", 0.0)))

    def merge_detections(self, detections: Iterable[dict[str, Any]]):
        """Class-agnostic pose NMS that retains provenance for suppressed boxes."""
        candidates = [deepcopy(detection) for detection in detections]
        candidates.sort(key=self._pose_quality, reverse=True)
        kept = []
        for candidate in candidates:
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if self._bbox_iou(candidate["bbox"], existing["bbox"]) >= self.merge_iou
                ),
                None,
            )
            if duplicate is None:
                candidate.setdefault("supporting_detections", [])
                kept.append(candidate)
                continue

            duplicate.setdefault("merged_sources", [duplicate["source"]])
            if candidate["source"] not in duplicate["merged_sources"]:
                duplicate["merged_sources"].append(candidate["source"])
            duplicate.setdefault("supporting_detections", []).append(
                {
                    "source": candidate["source"],
                    "confidence": candidate["confidence"],
                    "bbox": np.asarray(candidate["bbox"], dtype=float).copy(),
                    "inference": deepcopy(candidate["inference"]),
                }
            )
        return kept

    def process_frame_detailed(self, frame, *, imgsz=None):
        """Run one full-frame inference and return rich detection records."""
        selected_imgsz = self.imgsz if imgsz is None else self._validate_imgsz(imgsz)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        detections = self._infer(
            frame,
            imgsz=selected_imgsz,
            source="full_frame",
            roi=(0, 0, width, height),
        )
        self._last_detections = detections
        return detections

    def process_fixed_camera(self, frame, far_roi=None, *, imgsz=None):
        """Run full-frame plus far-half ROI at one explicit input size.

        ``imgsz`` intentionally applies to *both* passes.  A caller that asks
        for 960 or 1280 must not silently fall back to 640 merely because the
        fixed-camera far-ROI enhancement is enabled.
        """
        height, width = int(frame.shape[0]), int(frame.shape[1])
        selected_imgsz = self.imgsz if imgsz is None else self._validate_imgsz(imgsz)
        resolved_roi = self._resolve_roi(frame.shape, self.far_roi if far_roi is None else far_roi)
        full_detections = self._infer(
            frame,
            imgsz=selected_imgsz,
            source="full_frame",
            roi=(0, 0, width, height),
        )
        x1, y1, x2, y2 = resolved_roi
        far_detections = self._infer(
            frame[y1:y2, x1:x2],
            imgsz=selected_imgsz,
            source="far_roi",
            roi=resolved_roi,
        )
        merged = self.merge_detections([*full_detections, *far_detections])
        self._last_detections = merged
        return merged

    def process_frame(self, frame, *, imgsz=None, asymmetric=None, far_roi=None):
        """Process one frame while preserving the original tuple return value."""
        use_asymmetric = self.asymmetric if asymmetric is None else bool(asymmetric)
        if use_asymmetric:
            detections = self.process_fixed_camera(frame, far_roi=far_roi, imgsz=imgsz)
        else:
            detections = self.process_frame_detailed(frame, imgsz=imgsz)
        if not detections:
            return None, None

        keypoints = np.stack([detection["keypoints"] for detection in detections])
        if all(detection.get("keypoint_scores") is None for detection in detections):
            scores = None
        else:
            scores = np.stack(
                [
                    detection["keypoint_scores"]
                    if detection.get("keypoint_scores") is not None
                    else np.zeros(keypoints.shape[1], dtype=float)
                    for detection in detections
                ]
            )
        return keypoints, scores

    def get_last_detections(self):
        """Return the latest rich detections without exposing internal mutable state."""
        return deepcopy(self._last_detections)
