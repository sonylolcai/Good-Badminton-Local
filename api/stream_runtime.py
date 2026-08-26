"""Production composition root for anonymous fixed-camera stream analysis.

The business service owns camera calibration.  Every stream session and its
segments carry the same four image-space court corners; the GPU uses only that
per-session input, never estimates a court or keeps a camera registry.

TrackNetV3 is a temporal plug-in seam here.  The repository's current official
TrackNet runner is complete-file/batch oriented, so production streaming must
provide a bounded-state temporal processor factory explicitly; the runtime
fails clearly instead of silently substituting YOLO or fabricated ball data.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from badminton_analysis.detection.shuttlecock import ShuttlecockTracker
from badminton_analysis.detection.yolo_pose import YOLOPoseProcessor
from badminton_analysis.streaming.models import FinalizationContext, FrameContext, ProcessorEvent
from badminton_analysis.tracking.person_only import PersonOnlyFrameProcessor, PersonOnlyTracker

from .candidate_photos import CandidatePhotoCollector


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def _load_ultralytics_model(model_path: str, cache_kind: str):
    resolved = str(Path(model_path).resolve())
    key = (cache_kind, resolved)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            if not Path(resolved).is_file():
                raise FileNotFoundError(f"model checkpoint not found: {resolved}")
            from ultralytics import YOLO

            model = YOLO(resolved)
            _MODEL_CACHE[key] = model
        return model


def _derived_far_court_roi(corners, frame_shape):
    """Return a tight rectangular far-half ROI from ordered court corners.

    The session contract uses the existing calibration order
    ``far-left, far-right, near-right, near-left``.  A business-owned camera
    profile may provide ``far_pose_roi`` instead; this deterministic fallback
    avoids the old full-width upper-half heuristic that often included judges.
    """
    points = np.asarray(corners, dtype=float)
    if points.shape != (4, 2):
        raise ValueError("court corners must contain four [x, y] points for far ROI")
    far_left, far_right, near_right, near_left = points
    far_half = np.asarray(
        [
            far_left,
            far_right,
            (far_right + near_right) / 2.0,
            (far_left + near_left) / 2.0,
        ],
        dtype=float,
    )
    height, width = int(frame_shape[0]), int(frame_shape[1])
    pad_x, pad_y = max(4, round(width * 0.025)), max(4, round(height * 0.025))
    x1 = max(0, int(np.floor(far_half[:, 0].min())) - pad_x)
    y1 = max(0, int(np.floor(far_half[:, 1].min())) - pad_y)
    x2 = min(width, int(np.ceil(far_half[:, 0].max())) + pad_x)
    y2 = min(height, int(np.ceil(far_half[:, 1].max())) + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("derived far court ROI is empty")
    return (x1, y1, x2, y2)


class PoseObservationProvider:
    """Convert rich YOLO Pose detections into court-space tracker evidence."""

    def __init__(
        self,
        pose: YOLOPoseProcessor,
        tracker: PersonOnlyTracker,
        *,
        pose_imgsz: int,
        keypoint_confidence: float = 0.25,
        asymmetric: bool = False,
        far_roi=None,
    ):
        self.pose = pose
        self.tracker = tracker
        self.pose_imgsz = int(pose_imgsz)
        self.keypoint_confidence = float(keypoint_confidence)
        self.asymmetric = bool(asymmetric)
        self.far_roi = far_roi

    def __call__(self, frame, _context: FrameContext):
        if self.asymmetric:
            detections = self.pose.process_fixed_camera(
                frame,
                far_roi=(
                    self.far_roi
                    if self.far_roi is not None
                    else _derived_far_court_roi(self.tracker.image_corners, frame.shape)
                ),
                imgsz=self.pose_imgsz,
            )
        else:
            detections = self.pose.process_frame_detailed(frame, imgsz=self.pose_imgsz)
        observations = []
        for detection in detections:
            observation = self._observation(detection)
            if observation is not None:
                observations.append(observation)
        return {"observations": observations, "has_fresh_observations": True}

    def _observation(self, detection):
        keypoints = np.asarray(detection.get("keypoints"), dtype=float)
        if keypoints.ndim != 2 or keypoints.shape[0] < 17 or keypoints.shape[1] < 2:
            return None
        scores_value = detection.get("keypoint_scores")
        scores = None if scores_value is None else np.asarray(scores_value, dtype=float).reshape(-1)
        bbox = np.asarray(detection.get("bbox"), dtype=float).reshape(-1)
        confidence = float(detection.get("confidence") or 0.0)

        def visible(index):
            point = keypoints[index, :2]
            if not np.isfinite(point).all() or point[0] <= 1 or point[1] <= 1:
                return False
            return scores is None or (
                index < len(scores)
                and np.isfinite(scores[index])
                and float(scores[index]) >= self.keypoint_confidence
            )

        left_visible, right_visible = visible(15), visible(16)
        if left_visible and right_visible:
            image_xy = (keypoints[15, :2] + keypoints[16, :2]) / 2.0
            ankle_confidence = 1.0 if scores is None else min(float(scores[15]), float(scores[16]))
            method, degraded = "ankles_midpoint", False
            location_confidence = confidence * ankle_confidence
        elif left_visible or right_visible:
            index = 15 if left_visible else 16
            image_xy = keypoints[index, :2]
            ankle_confidence = 1.0 if scores is None else float(scores[index])
            method, degraded = "single_ankle", True
            location_confidence = confidence * ankle_confidence * 0.75
        elif bbox.size >= 4 and np.isfinite(bbox[:4]).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            image_xy = np.asarray([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=float)
            method, degraded = "bbox_bottom_center", True
            location_confidence = confidence * 0.35
        else:
            return None

        court_xy = self.tracker.court_space.image_to_court(image_xy)
        if court_xy is None or not self.tracker.court_space.contains(court_xy, margin_m=0.35):
            return None
        hands = {}
        if visible(9):
            hands["left"] = [float(value) for value in keypoints[9, :2]]
        if visible(10):
            hands["right"] = [float(value) for value in keypoints[10, :2]]
        return {
            "court_xy": [float(court_xy[0]), float(court_xy[1])],
            "image_xy": [float(image_xy[0]), float(image_xy[1])],
            "bbox_xyxy": [float(value) for value in bbox[:4]],
            "confidence": max(0.0, min(1.0, confidence)),
            "location_method": method,
            "location_confidence": max(0.0, min(1.0, float(location_confidence))),
            "location_degraded": degraded,
            "source": str(detection.get("source") or "full_frame"),
            "hands_image": hands or None,
            "keypoints_image": keypoints[:, :2].tolist(),
            "keypoint_scores": scores.tolist() if scores is not None else None,
            "inference": deepcopy(detection.get("inference") or {}),
        }


class YoloShuttleFrameProcessor:
    """Low-latency measurement processor with durable, explicit evidence."""

    def __init__(self, tracker: ShuttlecockTracker, court_corners):
        self.tracker = tracker
        self.court_corners = [tuple(float(value) for value in point) for point in court_corners]

    def process_frame(self, frame, context: FrameContext):
        detected = self.tracker.detect_ball(frame, roi_corners=self.court_corners)
        self.tracker.update_trajectory(detected, roi_corners=self.court_corners)
        state = self.tracker.get_last_detection()
        status = str(state.get("status") or "missing")
        evidence = "detected" if status == "detected" else "predicted" if status == "predicted" else "missing"
        confidence = float(state.get("confidence") or 0.0)
        return [
            ProcessorEvent(
                event_type="shuttle_observation",
                evidence_state=evidence,
                confidence=max(0.0, min(1.0, confidence)),
                data={
                    "detector": "yolo",
                    "measurement": deepcopy(state),
                    "source_frame_index": int(context.source_frame_index),
                    "measurement_bucket": int(context.measurement_bucket),
                },
            )
        ]

    def finalize(self, _context: FinalizationContext):
        return []

    def snapshot_state(self):
        return {
            "state_version": "yolo-shuttle-stream.v1",
            "trajectory": [list(point) for point in self.tracker.shuttlecock_trajectory],
            "actual_history": [
                [int(frame), list(point)] for frame, point in self.tracker.actual_history
            ],
            "last_valid_position": list(self.tracker.last_valid_position)
            if self.tracker.last_valid_position is not None
            else None,
            "last_valid_confidence": self.tracker.last_valid_confidence,
            "last_candidate": deepcopy(self.tracker.last_candidate),
            "last_detection": deepcopy(self.tracker.last_detection),
            "missing_frames": int(self.tracker.missing_frames),
            "frame_index": int(self.tracker.frame_index),
        }

    def restore_state(self, state: Mapping[str, Any]):
        if state.get("state_version") != "yolo-shuttle-stream.v1":
            raise ValueError("unsupported YOLO shuttle checkpoint")
        self.tracker.shuttlecock_trajectory = deque(
            (tuple(point) for point in state.get("trajectory", [])),
            maxlen=self.tracker.trajectory_length,
        )
        self.tracker.actual_history = deque(
            (
                (int(item[0]), tuple(item[1]))
                for item in state.get("actual_history", [])
            ),
            maxlen=self.tracker.trajectory_length,
        )
        point = state.get("last_valid_position")
        self.tracker.last_valid_position = tuple(point) if point is not None else None
        self.tracker.last_valid_confidence = state.get("last_valid_confidence")
        self.tracker.last_candidate = deepcopy(state.get("last_candidate"))
        self.tracker.last_detection = deepcopy(state.get("last_detection") or {})
        self.tracker.missing_frames = int(state.get("missing_frames", 0))
        self.tracker.frame_index = int(state.get("frame_index", 0))


class CompositeMeasurementProcessor:
    """Run person and optional YOLO-ball measurement on the same sampled frame."""

    def __init__(self, person, shuttle=None, candidate_photos=None):
        self.person = person
        self.shuttle = shuttle
        self.candidate_photos = candidate_photos

    def process_frame(self, frame, context):
        events = list(self.person.process_frame(frame, context))
        if self.candidate_photos is not None:
            events = self.candidate_photos.attach(frame, context, events)
        if self.shuttle is not None:
            events.extend(self.shuttle.process_frame(frame, context))
        return events

    def finalize(self, context):
        events = list(self.person.finalize(context))
        if self.shuttle is not None:
            events.extend(self.shuttle.finalize(context))
        return events

    def snapshot_state(self):
        return {
            "state_version": "stream-composite.v1",
            "person": dict(self.person.snapshot_state()),
            "shuttle": dict(self.shuttle.snapshot_state()) if self.shuttle is not None else None,
            "candidate_photos": (
                dict(self.candidate_photos.snapshot_state())
                if self.candidate_photos is not None
                else None
            ),
        }

    def restore_state(self, state):
        if state.get("state_version") != "stream-composite.v1":
            raise ValueError("unsupported composite processor checkpoint")
        self.person.restore_state(state.get("person") or {})
        shuttle_state = state.get("shuttle")
        if shuttle_state is not None:
            if self.shuttle is None:
                raise ValueError("checkpoint requires the YOLO shuttle processor")
            self.shuttle.restore_state(shuttle_state)
        photo_state = state.get("candidate_photos")
        if photo_state is not None:
            if self.candidate_photos is None:
                raise ValueError("checkpoint requires the candidate photo collector")
            self.candidate_photos.restore_state(photo_state)


class StreamProcessorFactory:
    """Session-aware factory used by StreamSessionManager and task F routes."""

    def __init__(
        self,
        data_dir: Path,
        *,
        tracknet_processor_factory: Optional[Callable[[dict, dict], Any]] = None,
        pose_model_factory: Optional[Callable[[str], Any]] = None,
        ball_model_factory: Optional[Callable[[str], Any]] = None,
        byte_tracker_factory=None,
    ):
        self.data_dir = Path(data_dir).resolve()
        # Ultralytics creates a settings directory while importing ``YOLO``.
        # A service process may run under a restricted system account, where the
        # normal roaming-profile location is unavailable.  Keep this harmless
        # runtime state beside the already writable API data instead, before the
        # first lazy model import takes place.
        os.environ.setdefault(
            "YOLO_CONFIG_DIR",
            str(self.data_dir / "runtime" / "ultralytics"),
        )
        self.tracknet_processor_factory = tracknet_processor_factory
        self.pose_model_factory = pose_model_factory
        self.ball_model_factory = ball_model_factory
        self.byte_tracker_factory = byte_tracker_factory

    def validate_session_request(self, request):
        if not isinstance(request.get("court_corners"), list) or len(request["court_corners"]) != 4:
            raise ValueError("stream session requires exactly four business-supplied court_corners")
        configuration = request["configuration"]
        if configuration.get("generate_annotated_video"):
            raise ValueError(
                "streaming annotated-video export is not implemented; keep generate_annotated_video=false"
            )
        if configuration.get("shuttle_detector") == "tracknet_v3" and self.tracknet_processor_factory is None:
            raise ValueError(
                "TrackNetV3 streaming requires a configured bounded-state temporal processor factory"
            )

    def __call__(self, session):
        self.validate_session_request(session)
        configuration = session["configuration"]
        calibration = {
            "calibration_id": session["calibration_id"],
            "image_corners": session["court_corners"],
            "far_player_enhancement": bool(
                configuration.get("far_player_enhancement", False)
            ),
            "far_roi": configuration.get("far_pose_roi"),
        }
        sample_hz = int(configuration["analysis_sample_hz"])
        # This is a session contract choice, not a GPU-process default.  The
        # previous environment-only setting made a WebUI ByteTrack selection
        # disappear between the business gateway and the streaming worker.
        tracker_backend = str(
            configuration.get("tracker_backend", "court_association")
        ).strip()
        enable_bytetrack = tracker_backend == "bytetrack"
        tracker = PersonOnlyTracker(
            calibration["image_corners"],
            fps=sample_hz,
            tracker_backend=tracker_backend,
            enable_bytetrack=enable_bytetrack,
            byte_tracker_factory=self.byte_tracker_factory,
            max_missed_frames=max(2, int(round(sample_hz * 0.4))),
            max_retained_missing_frames=max(10, int(round(sample_hz * 3.0))),
            # Direct composition tests and third-party factories that bypass
            # request validation retain historical open-set behaviour.  All
            # accepted stream-session.v1 requests are normalised with the
            # explicit default (currently true) by ``stream_models``.
            lock_match_roster=bool(configuration.get("lock_match_roster", False)),
            expected_roster_count=configuration.get("expected_player_count"),
            roster_stable_frames=int(configuration.get("roster_stable_frames", 3)),
            max_roster_count=int(configuration.get("max_roster_count", 4)),
            roster_discovery_seconds=float(
                configuration.get("roster_discovery_seconds", 8.0)
            ),
        )

        project_root = Path(__file__).resolve().parents[1]
        pose_path = os.environ.get(
            "GOOD_BADMINTON_STREAM_POSE_MODEL",
            str(project_root / "weights" / "yolo11n-pose.pt"),
        )
        pose_model = (
            self.pose_model_factory(pose_path)
            if self.pose_model_factory is not None
            else _load_ultralytics_model(pose_path, "pose")
        )
        pose = YOLOPoseProcessor(
            model_path=pose_path,
            model=pose_model,
            conf=float(os.environ.get("GOOD_BADMINTON_STREAM_POSE_CONF", "0.15")),
            imgsz=int(configuration["pose_imgsz"]),
            device=os.environ.get("GOOD_BADMINTON_STREAM_DEVICE", "auto"),
        )
        far_enabled = bool(calibration.get("far_player_enhancement", False))
        provider = PoseObservationProvider(
            pose,
            tracker,
            pose_imgsz=int(configuration["pose_imgsz"]),
            asymmetric=far_enabled,
            far_roi=calibration.get("far_roi"),
        )
        person = PersonOnlyFrameProcessor(tracker, provider)
        # Direct composition tests may omit the manager-owned ID. Real accepted
        # stream sessions always have one; a private factory fallback keeps
        # those tests isolated without changing the production path.
        session_id = str(session.get("analysis_session_id") or f"factory-{id(session)}")
        candidate_photos = CandidatePhotoCollector(
            self.data_dir / "stream_sessions" / session_id / "candidate_photos"
        )

        shuttle_detector = configuration["shuttle_detector"]
        if shuttle_detector == "none":
            return CompositeMeasurementProcessor(person, candidate_photos=candidate_photos), None
        if shuttle_detector == "yolo":
            ball_path = os.environ.get(
                "GOOD_BADMINTON_STREAM_BALL_MODEL",
                str(project_root / "weights" / "yolo11s-ball.pt"),
            )
            ball_model = (
                self.ball_model_factory(ball_path)
                if self.ball_model_factory is not None
                else _load_ultralytics_model(ball_path, "ball")
            )
            shuttle = YoloShuttleFrameProcessor(
                ShuttlecockTracker(ball_model, show_trajectory=False),
                calibration["image_corners"],
            )
            return CompositeMeasurementProcessor(person, shuttle, candidate_photos), None

        temporal = self.tracknet_processor_factory(session, calibration)
        if temporal is None:
            raise RuntimeError("TrackNetV3 temporal processor factory returned no processor")
        return CompositeMeasurementProcessor(person, candidate_photos=candidate_photos), temporal
