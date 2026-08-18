from collections import deque
import time

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


class ShuttlecockTracker:
    """Detect, filter, track, and draw shuttlecock positions."""

    def __init__(
        self,
        yolo_ball_model,
        trajectory_length=30,
        show_trajectory=True,
        show_performance_stats=False,
        max_jump_pixels=220,
        prediction_gate_pixels=260,
        max_missing_frames=5,
        roi_padding_ratio=0.08,
        max_box_area_ratio=0.004,
        max_aspect_ratio=4.0,
        max_prediction_frames=3,
        prediction_confidence_decay=0.6,
    ):
        self.yolo_ball_model = yolo_ball_model
        self.trajectory_length = trajectory_length
        self.show_trajectory = show_trajectory
        self.show_performance_stats = show_performance_stats
        self.max_jump_pixels = max_jump_pixels
        self.prediction_gate_pixels = prediction_gate_pixels
        self.max_missing_frames = max_missing_frames
        self.roi_padding_ratio = roi_padding_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_prediction_frames = max(0, int(max_prediction_frames))
        self.prediction_confidence_decay = float(prediction_confidence_decay)
        if not 0.0 <= self.prediction_confidence_decay <= 1.0:
            raise ValueError("prediction_confidence_decay must be between 0 and 1")

        self.shuttlecock_trajectory = deque(maxlen=trajectory_length)
        self.actual_history = deque(maxlen=trajectory_length)
        self.last_valid_position = None
        self.last_valid_confidence = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0
        self.frame_index = 0

        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            self.ultra_device = 0
        else:
            self.ultra_device = "cpu"

    def detect_ball(self, frame, conf=0.18, roi_corners=None):
        t0 = time.time()
        try:
            ball_results = self.yolo_ball_model(frame, conf=conf, device=self.ultra_device, verbose=False)[0]
        except TypeError:
            ball_results = self.yolo_ball_model(frame, conf=conf, verbose=False)[0]

        if self.show_performance_stats:
            print(f"YOLO shuttlecock inference took {time.time() - t0:.2f} sec")

        candidates, candidate_diagnostics = self._extract_candidates(
            ball_results, frame.shape, roi_corners
        )
        selected = self._select_candidate(candidates)
        self.last_candidate = selected
        self.last_detection = {
            "status": "candidate" if selected is not None else "missing",
            "visible": selected is not None,
            "accepted": False,
            "image": list(selected["point"]) if selected else None,
            "confidence": selected["confidence"] if selected else None,
            "candidate_count": len(candidates),
            "raw_candidate_count": candidate_diagnostics["raw_candidate_count"],
            "filtered_rejections": candidate_diagnostics["filtered_rejections"],
            "source": "ball_model" if selected is not None else None,
            "gap_frames": 0,
            "rejection_reason": None if selected is not None else "no_candidate",
        }
        return list(selected["point"]) if selected else [0, 0]

    def update_trajectory(self, ball_position, roi_corners=None):
        self.frame_index += 1
        if ball_position == [0, 0] or ball_position is None:
            return self._handle_missing("no_candidate", roi_corners)

        point = tuple(ball_position)
        if not self._point_in_roi(point, roi_corners):
            return self._handle_missing("outside_court_roi", roi_corners)

        if self._is_outlier(point):
            return self._handle_missing("motion_gate", roi_corners)

        self._append_valid_point(point)
        self.last_detection["accepted"] = True
        self.last_detection["status"] = "detected"
        self.last_detection["image"] = list(point)
        self.last_detection["source"] = "ball_model"
        self.last_detection["gap_frames"] = 0
        self.last_detection["rejection_reason"] = None
        return list(point)

    def update_external_measurement(self, measurement, roi_corners=None):
        """Use one externally computed raw measurement as the ball evidence.

        TrackNetV3 has already applied temporal heatmap reasoning.  Reapplying
        the YOLO distance gate here would incorrectly discard fast shots, so
        this method preserves each raw visible point.  Missing frames still use
        the existing, explicitly-labelled short prediction policy for display;
        those predicted points remain ``accepted=False``.
        """
        self.frame_index += 1
        measurement = measurement or {}
        source = measurement.get("source") or "external_measurement"
        point = measurement.get("image")
        visible = bool(measurement.get("visible")) and self._valid_external_point(point)
        if not visible:
            self.last_detection = {
                **self._empty_detection_state(),
                "source": source,
                "measurement_kind": measurement.get("measurement_kind"),
                "confidence_status": measurement.get("confidence_status", "not_visible"),
                "rejection_reason": "external_measurement_not_visible",
            }
            return self._handle_missing("external_measurement_not_visible", roi_corners)

        normalized = (float(point[0]), float(point[1]))
        self.last_candidate = {"point": normalized, "confidence": measurement.get("confidence")}
        self.last_detection = {
            "status": "detected",
            "visible": True,
            "accepted": True,
            "image": [normalized[0], normalized[1]],
            # The value records TrackNet's binary visibility threshold.  It is
            # deliberately accompanied by confidence_status, not presented as
            # a calibrated detector score.
            "confidence": measurement.get("confidence"),
            "confidence_status": measurement.get("confidence_status"),
            "measurement_kind": measurement.get("measurement_kind"),
            "candidate_count": 1,
            "raw_candidate_count": 1,
            "filtered_rejections": {},
            "source": source,
            "gap_frames": 0,
            "rejection_reason": None,
        }
        self._append_valid_point(normalized)
        return [normalized[0], normalized[1]]

    @staticmethod
    def _valid_external_point(point):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        try:
            return np.isfinite(float(point[0])) and np.isfinite(float(point[1]))
        except (TypeError, ValueError):
            return False

    def _handle_missing(self, reason, roi_corners):
        self._record_missing_detection()
        predicted = self._predict_missing_position()
        if predicted is not None and self._point_in_roi(predicted, roi_corners):
            confidence = (self.last_valid_confidence or 0.0) * (
                self.prediction_confidence_decay ** self.missing_frames
            )
            self.last_detection.update(
                {
                    "status": "predicted",
                    "visible": False,
                    "accepted": False,
                    "image": [float(predicted[0]), float(predicted[1])],
                    "confidence": float(confidence),
                    "source": "constant_velocity",
                    "gap_frames": self.missing_frames,
                    "rejection_reason": reason,
                }
            )
            return [float(predicted[0]), float(predicted[1])]

        self._mark_detection_rejected(reason)
        return [0, 0]

    def _extract_candidates(self, ball_results, frame_shape, roi_corners):
        boxes = ball_results.boxes
        if boxes is None or boxes.xywh.shape[0] < 1:
            return [], {
                "raw_candidate_count": 0,
                "filtered_rejections": {},
            }

        xywh = boxes.xywh.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
        frame_area = max(1, frame_shape[0] * frame_shape[1])

        candidates = []
        rejected = {
            "invalid_geometry": 0,
            "box_too_large": 0,
            "aspect_ratio": 0,
            "outside_court_roi": 0,
        }
        for box, confidence in zip(xywh, confidences):
            center_x, center_y, width, height = [float(value) for value in box]
            if width <= 0 or height <= 0:
                rejected["invalid_geometry"] += 1
                continue

            point = (int(center_x), int(center_y))
            area_ratio = (width * height) / frame_area
            aspect_ratio = max(width / height, height / width)
            if area_ratio > self.max_box_area_ratio:
                rejected["box_too_large"] += 1
                continue
            if aspect_ratio > self.max_aspect_ratio:
                rejected["aspect_ratio"] += 1
                continue
            if not self._point_in_roi(point, roi_corners):
                rejected["outside_court_roi"] += 1
                continue

            candidates.append(
                {
                    "point": point,
                    "confidence": float(confidence),
                    "area_ratio": float(area_ratio),
                    "aspect_ratio": float(aspect_ratio),
                }
            )

        return candidates, {
            "raw_candidate_count": int(len(xywh)),
            "filtered_rejections": {
                reason: count for reason, count in rejected.items() if count
            },
        }

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        if not self.shuttlecock_trajectory:
            return max(candidates, key=lambda item: item["confidence"])

        predicted = self._predict_next_position()

        def score(candidate):
            distance = self._distance(candidate["point"], predicted)
            size_penalty = candidate["area_ratio"] * 4000
            return candidate["confidence"] * 1000 - distance * 1.4 - size_penalty

        return max(candidates, key=score)

    def _point_in_roi(self, point, roi_corners):
        if roi_corners is None:
            return True

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        padding = int(max(x2 - x1, y2 - y1) * self.roi_padding_ratio)
        return (x1 - padding) <= point[0] <= (x2 + padding) and (y1 - padding) <= point[1] <= (y2 + padding)

    def _is_outlier(self, point):
        if not self.shuttlecock_trajectory:
            return False

        last_point = self.shuttlecock_trajectory[-1]
        jump_distance = self._distance(point, last_point)
        strict_gate = self.missing_frames <= self.max_missing_frames
        if jump_distance > self.max_jump_pixels and strict_gate:
            return True

        predicted = self._predict_next_position()
        predicted_distance = self._distance(point, predicted)
        if predicted_distance > self.prediction_gate_pixels and strict_gate:
            return True

        return False

    def _predict_next_position(self):
        if len(self.actual_history) < 2:
            return self.shuttlecock_trajectory[-1]

        (prev_frame, (prev_x, prev_y)), (last_frame, (last_x, last_y)) = list(self.actual_history)[-2:]
        elapsed = max(1, last_frame - prev_frame)
        return (
            last_x + (last_x - prev_x) / elapsed,
            last_y + (last_y - prev_y) / elapsed,
        )

    def _predict_missing_position(self):
        if self.missing_frames > self.max_prediction_frames or len(self.actual_history) < 2:
            return None
        (prev_frame, (prev_x, prev_y)), (last_frame, (last_x, last_y)) = list(self.actual_history)[-2:]
        elapsed = max(1, last_frame - prev_frame)
        target_elapsed = self.frame_index - last_frame
        return (
            last_x + ((last_x - prev_x) / elapsed) * target_elapsed,
            last_y + ((last_y - prev_y) / elapsed) * target_elapsed,
        )

    def _append_valid_point(self, point):
        self.shuttlecock_trajectory.append(point)
        self.actual_history.append((self.frame_index, point))
        self.last_valid_position = point
        self.last_valid_confidence = self.last_detection.get("confidence")
        self.missing_frames = 0

    def _record_missing_detection(self):
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.last_valid_position = None

    def _mark_detection_rejected(self, reason=None):
        self.last_detection["status"] = "missing"
        self.last_detection["accepted"] = False
        self.last_detection["image"] = None
        self.last_detection["confidence"] = None
        self.last_detection["source"] = None
        self.last_detection["gap_frames"] = self.missing_frames
        self.last_detection["rejection_reason"] = reason

    def _empty_detection_state(self):
        return {
            "status": "missing",
            "visible": False,
            "accepted": False,
            "image": None,
            "confidence": None,
            "candidate_count": 0,
            "raw_candidate_count": 0,
            "filtered_rejections": {},
            "source": None,
            "gap_frames": 0,
            "rejection_reason": None,
        }

    def _distance(self, point_a, point_b):
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    def draw_trajectory(self, frame):
        if not self.shuttlecock_trajectory:
            return

        t0 = time.time()
        color = (87, 108, 255)
        points = list(self.shuttlecock_trajectory)

        for i, point in enumerate(points):
            radius = int(3 + (i / len(points)) * 4)
            cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)

        latest_point = points[-1]
        cv2.circle(frame, latest_point, 6, (0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing shuttlecock trajectory took {time.time() - t0:.2f} sec")

    def handle_visualization(self, frame):
        if not self.show_trajectory:
            return
        if self.shuttlecock_trajectory:
            self.draw_trajectory(frame)
        if self.last_detection.get("status") == "predicted" and self.last_detection.get("image"):
            point = tuple(int(round(value)) for value in self.last_detection["image"])
            cv2.circle(frame, point, 7, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    def clear_trajectory(self):
        self.shuttlecock_trajectory.clear()
        self.actual_history.clear()
        self.last_valid_position = None
        self.last_valid_confidence = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0
        self.frame_index = 0

    def get_trajectory(self):
        return list(self.shuttlecock_trajectory)

    def get_last_detection(self):
        return dict(self.last_detection)
