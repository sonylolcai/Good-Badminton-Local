"""Court-coordinate match primitives for a fixed-camera badminton match.

This module deliberately has no YOLO, OpenCV display, or WebUI dependency.
It receives already-detected positions and preserves the distinction between
measurements, short-lived predictions, and conclusions that are not supported
by enough evidence.  That makes it safe to test with several camera views and
keeps future doubles rules outside the visual tracking layer.
"""

from dataclasses import dataclass, field
from math import hypot
from typing import Optional, Tuple

from ..court.mapper import CourtMapper
from ..court.reference import BADMINTON_COURT_LENGTH, BADMINTON_COURT_WIDTH


SCHEMA_VERSION = "2.0"


class CourtSpace:
    """A standard-court coordinate system independent of image orientation."""

    def __init__(self, image_corners, court_dimensions=(BADMINTON_COURT_WIDTH, BADMINTON_COURT_LENGTH)):
        self.width_m, self.length_m = (float(value) for value in court_dimensions)
        self.mapper = CourtMapper(image_corners, court_dimensions=court_dimensions)
        self.net_y_m = self.length_m / 2.0

    def image_to_court(self, image_xy):
        point = self.mapper.image_to_court(image_xy)
        return (float(point[0]), float(point[1])) if len(point) == 2 else None

    def court_to_image(self, court_xy):
        point = self.mapper.court_to_image(court_xy)
        return (float(point[0]), float(point[1])) if len(point) == 2 else None

    def zone_for(self, court_xy):
        """Return a transient 3x3 court zone, never an identity or team label."""
        if court_xy is None:
            return None
        x, y = (float(value) for value in court_xy)
        column = "left" if x < self.width_m / 3 else "right" if x > self.width_m * 2 / 3 else "center"
        row = "rear" if y < self.length_m / 3 else "front" if y > self.length_m * 2 / 3 else "mid"
        return f"{row}_{column}"

    def contains(self, court_xy, margin_m=0.0):
        if court_xy is None:
            return False
        x, y = (float(value) for value in court_xy)
        return -margin_m <= x <= self.width_m + margin_m and -margin_m <= y <= self.length_m + margin_m


@dataclass
class _Track:
    track_id: str
    court_xy: Tuple[float, float]
    image_xy: Optional[Tuple[float, float]]
    confidence: float
    last_frame: int
    velocity_mps: Tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    observations: int = 1
    last_evidence: dict = field(default_factory=dict)


class CourtMultiObjectTracker:
    """Small deterministic multi-target tracker using court metres as distance.

    It intentionally does not infer player identity, side, or team.  A caller
    can claim ``track_id -> person_id`` after a match, and doubles can use the
    same collection without changing the tracker data model.
    """

    def __init__(self, court_space, fps, max_missed_frames=12, max_speed_mps=10.0):
        self.court_space = court_space
        self.fps = float(fps)
        self.max_missed_frames = int(max_missed_frames)
        self.max_speed_mps = float(max_speed_mps)
        self.tracks = {}
        self._next_id = 1
        self.identity_claims = {}
        self.track_metrics = {}

    def update(self, frame_index, observations):
        observations = [item for item in observations if item.get("court_xy") is not None]
        unmatched_track_ids = set(self.tracks)
        unmatched_observations = set(range(len(observations)))
        assignments = []

        # Greedy assignment is deterministic, and sufficient for the two-player
        # first version.  Its gate is in metres, so a side/oblique image view
        # has exactly the same identity behavior as a rear view.
        candidates = []
        for track_id, track in self.tracks.items():
            predicted = self._predict_position(track, frame_index)
            for index, observation in enumerate(observations):
                distance = self._distance(predicted, observation["court_xy"])
                if distance <= self._association_gate(track, frame_index):
                    candidates.append((distance, track_id, index))
        for _distance, track_id, index in sorted(candidates):
            if track_id not in unmatched_track_ids or index not in unmatched_observations:
                continue
            assignments.append((track_id, index))
            unmatched_track_ids.remove(track_id)
            unmatched_observations.remove(index)

        for track_id, index in assignments:
            self._apply_observation(self.tracks[track_id], frame_index, observations[index])
        for index in sorted(unmatched_observations):
            self._create_track(frame_index, observations[index])
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_frames += max(1, frame_index - track.last_frame)
            track.last_frame = frame_index
            if track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        return self.snapshot(frame_index)

    def claim_identities(self, claims):
        """Apply optional post-match names without changing any track IDs."""
        unknown = set(claims).difference(self.track_metrics)
        if unknown:
            raise ValueError(f"Unknown track IDs: {sorted(unknown)}")
        values = list(claims.values())
        if len(values) != len(set(values)):
            raise ValueError("A person_id can be claimed by only one track_id")
        self.identity_claims.update({str(key): str(value) for key, value in claims.items()})
        return dict(self.identity_claims)

    def summaries(self):
        """Return safe movement inputs for a later player-style report.

        These are movement/coverage observations only.  No win rate, error
        rate, or ability score is inferred here because those require verified
        hit and score evidence.
        """
        return [
            {
                "track_id": track_id,
                "person_id": self.identity_claims.get(track_id),
                "detected_frames": metrics["detected_frames"],
                "predicted_frames": metrics["predicted_frames"],
                "distance_m": round(metrics["distance_m"], 3),
                "zone_frames": dict(sorted(metrics["zone_frames"].items())),
                "analytics_scope": "movement_and_space_only",
            }
            for track_id, metrics in sorted(self.track_metrics.items())
        ]

    def snapshot(self, frame_index):
        records = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            predicted = track.missed_frames > 0
            court_xy = self._predict_position(track, frame_index) if predicted else track.court_xy
            if predicted:
                self.track_metrics[track_id]["predicted_frames"] += 1
            records.append(
                {
                    "track_id": track.track_id,
                    "person_id": self.identity_claims.get(track.track_id),
                    "image_xy": self._as_list(track.image_xy),
                    "court_xy_m": self._as_list(court_xy),
                    "zone_id": self.court_space.zone_for(court_xy),
                    "status": "predicted" if predicted else "detected",
                    "confidence": round(track.confidence * (0.72 ** track.missed_frames), 4),
                    "missed_frames": track.missed_frames,
                    "location_evidence": dict(track.last_evidence),
                }
            )
        return records

    def _create_track(self, frame_index, observation):
        track_id = f"track_{self._next_id:03d}"
        self._next_id += 1
        self.tracks[track_id] = _Track(
            track_id=track_id,
            court_xy=tuple(float(value) for value in observation["court_xy"]),
            image_xy=self._tuple_or_none(observation.get("image_xy")),
            confidence=float(observation.get("confidence", 0.0)),
            last_frame=int(frame_index),
            last_evidence=self._evidence(observation),
        )
        self.track_metrics[track_id] = {
            "detected_frames": 1,
            "predicted_frames": 0,
            "distance_m": 0.0,
            "zone_frames": {self.court_space.zone_for(observation["court_xy"]): 1},
        }

    def _apply_observation(self, track, frame_index, observation):
        new_xy = tuple(float(value) for value in observation["court_xy"])
        distance = self._distance(track.court_xy, new_xy)
        elapsed = max(1, int(frame_index) - track.last_frame) / self.fps
        velocity = ((new_xy[0] - track.court_xy[0]) / elapsed, (new_xy[1] - track.court_xy[1]) / elapsed)
        speed = hypot(*velocity)
        if speed <= self.max_speed_mps:
            track.velocity_mps = velocity
        track.court_xy = new_xy
        track.image_xy = self._tuple_or_none(observation.get("image_xy"))
        track.confidence = float(observation.get("confidence", 0.0))
        track.last_frame = int(frame_index)
        track.missed_frames = 0
        track.observations += 1
        track.last_evidence = self._evidence(observation)
        metrics = self.track_metrics[track.track_id]
        metrics["detected_frames"] += 1
        metrics["distance_m"] += distance
        zone = self.court_space.zone_for(new_xy)
        metrics["zone_frames"][zone] = metrics["zone_frames"].get(zone, 0) + 1

    def _predict_position(self, track, frame_index):
        elapsed = max(0, int(frame_index) - track.last_frame) / self.fps
        return (track.court_xy[0] + track.velocity_mps[0] * elapsed, track.court_xy[1] + track.velocity_mps[1] * elapsed)

    def _association_gate(self, track, frame_index):
        elapsed = max(1, int(frame_index) - track.last_frame) / self.fps
        return max(0.8, self.max_speed_mps * elapsed + 0.35)

    @staticmethod
    def _distance(left, right):
        return hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))

    @staticmethod
    def _tuple_or_none(value):
        return tuple(float(item) for item in value[:2]) if value is not None else None

    @staticmethod
    def _as_list(value):
        return [round(float(item), 4) for item in value] if value is not None else None

    @staticmethod
    def _evidence(observation):
        return {
            "method": observation.get("location_method"),
            "confidence": observation.get("location_confidence"),
            "source": observation.get("source"),
        }


class MonocularShuttleReconstructor:
    """Report an honest, low-confidence 3D proxy for a single fixed view.

    The homography supplies a ground-plane XY estimate.  The Z coordinate is a
    physics-constrained proxy (not a calibrated measurement), so it is never
    eligible as score or error ground truth.  A future calibrated camera model
    may replace this estimator without changing the JSON contract.
    """

    ASSUMPTIONS = [
        "xy is projected to the court plane via a fixed homography",
        "z is a single-view projectile proxy, not metric camera reconstruction",
        "predicted or missing shuttle detections are excluded from hit and score evidence",
    ]

    def __init__(self):
        self.history = []

    def update(self, frame_index, time_sec, court_xy, detection_confidence, detected):
        if not detected or court_xy is None:
            return {
                "status": "missing",
                "xyz_m": None,
                "confidence": 0.0,
                "source": "single_view_physics_fit",
                "assumptions": list(self.ASSUMPTIONS),
            }
        self.history.append((int(frame_index), float(time_sec), tuple(float(value) for value in court_xy)))
        self.history = self.history[-12:]
        # A gentle parabola is intentionally capped below an authoritative
        # confidence.  It is visual/analytic context only until calibrated.
        age = min(1.0, max(0.0, (len(self.history) - 1) / 8.0))
        z_m = round(1.15 + 1.35 * age * (1.0 - age * 0.45), 3)
        confidence = round(min(0.45, max(0.05, float(detection_confidence) * (0.18 + 0.22 * age))), 4)
        return {
            "status": "approximate",
            "xyz_m": [round(court_xy[0], 4), round(court_xy[1], 4), z_m],
            "confidence": confidence,
            "source": "single_view_physics_fit",
            "assumptions": list(self.ASSUMPTIONS),
        }


class RallyStateMachine:
    """Conservative rally segmentation with rejectable score conclusions."""

    def __init__(self, fps, min_active_frames=4, end_gap_frames=None):
        self.fps = float(fps)
        self.min_active_frames = int(min_active_frames)
        self.end_gap_frames = int(end_gap_frames or max(10, round(self.fps * 1.2)))
        self.state = "idle"
        self._active_frames = 0
        self._missing_frames = 0
        self._rally_id = 0
        self._current = None
        self.completed = []

    def update(self, frame_index, tracks, shuttle, hit_events):
        has_players = len([track for track in tracks if track["status"] == "detected"]) >= 2
        has_shuttle = shuttle is not None and shuttle.get("status") == "approximate"
        if self.state == "idle" and has_players and has_shuttle:
            self.state = "candidate"
            self._active_frames = 1
        elif self.state == "candidate":
            self._active_frames = self._active_frames + 1 if has_shuttle else 0
            if self._active_frames >= self.min_active_frames:
                self._start(frame_index - self._active_frames + 1)
        elif self.state == "active":
            self._missing_frames = 0 if has_shuttle else self._missing_frames + 1
            if hit_events:
                self._current["hit_events"].extend(hit_events)
            if self._missing_frames >= self.end_gap_frames:
                self._finish(frame_index, reason="shuttle_evidence_gap")
        return self.snapshot()

    def finalize(self, frame_index=None):
        if self.state == "candidate":
            self._start(0)
        if self.state == "active":
            self._finish(frame_index, reason="video_end")
        return list(self.completed)

    def snapshot(self):
        return {
            "state": self.state,
            "rally_id": self._current["rally_id"] if self._current else None,
            "score_status": self._current["score"]["status"] if self._current else "unknown",
        }

    def _start(self, start_frame):
        self._rally_id += 1
        self.state = "active"
        self._current = {
            "rally_id": self._rally_id,
            "start_frame": int(start_frame),
            "end_frame": None,
            "hit_events": [],
            "confidence": 0.0,
            "score": {
                "status": "unknown",
                "winner_track_id": None,
                "reason": "no_high_confidence_terminal_evidence",
                "included_in_player_statistics": False,
            },
        }

    def _finish(self, frame_index, reason):
        self._current["end_frame"] = int(frame_index) if frame_index is not None else None
        self._current["end_reason"] = reason
        self._current["confidence"] = round(min(0.7, 0.3 + 0.05 * len(self._current["hit_events"])), 3)
        self.completed.append(self._current)
        self._current = None
        self.state = "idle"
        self._active_frames = 0
        self._missing_frames = 0


class FixedCameraMatchPipeline:
    """Compose spatial tracking, low-confidence shuttle proxy, and rallies."""

    def __init__(self, image_corners, fps, net_image_line=None):
        self.court_space = CourtSpace(image_corners)
        self.net_image_line = net_image_line or [
            self.court_space.court_to_image((0.0, self.court_space.net_y_m)),
            self.court_space.court_to_image((self.court_space.width_m, self.court_space.net_y_m)),
        ]
        self.tracker = CourtMultiObjectTracker(self.court_space, fps=fps)
        self.shuttle = MonocularShuttleReconstructor()
        self.rallies = RallyStateMachine(fps=fps)
        self._last_frame = 0

    def update(self, frame_index, observations, shuttlecock):
        self._last_frame = int(frame_index)
        tracks = self.tracker.update(frame_index, observations)
        shuttle = self._shuttle_record(frame_index, shuttlecock)
        hit_events = self._detect_hit_events(tracks, shuttle)
        rally = self.rallies.update(frame_index, tracks, shuttle, hit_events)
        return {
            "schema_version": SCHEMA_VERSION,
            "coordinate_system": "standard_badminton_court_m",
            "net": {"court_line": [[0.0, self.court_space.net_y_m], [self.court_space.width_m, self.court_space.net_y_m]], "image_line": self.net_image_line},
            "tracks": tracks,
            "shuttlecock_3d": shuttle,
            "hit_events": hit_events,
            "rally": rally,
        }

    def finalize(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "identity_claims": dict(self.tracker.identity_claims),
            "player_style_inputs": self.tracker.summaries(),
            "rallies": self.rallies.finalize(self._last_frame),
            "score_policy": "unknown scores are excluded from ability, win/loss, challenge, leaderboard, and key-point statistics",
        }

    def claim_identities(self, claims):
        """Record a post-match human identity claim without rewriting tracks."""
        return self.tracker.claim_identities(claims)

    def _shuttle_record(self, frame_index, shuttlecock):
        if not shuttlecock:
            return self.shuttle.update(frame_index, frame_index / self.tracker.fps, None, 0.0, False)
        court_xy = shuttlecock.get("court_xy")
        if court_xy is None and shuttlecock.get("image_xy") is not None:
            court_xy = self.court_space.image_to_court(shuttlecock["image_xy"])
        return self.shuttle.update(
            frame_index,
            frame_index / self.tracker.fps,
            court_xy,
            shuttlecock.get("confidence", 0.0),
            bool(shuttlecock.get("detected", False)),
        )

    @staticmethod
    def _detect_hit_events(tracks, shuttle):
        if shuttle.get("status") != "approximate" or not tracks:
            return []
        shuttle_xy = shuttle["xyz_m"][:2]
        nearest = min(tracks, key=lambda track: hypot(track["court_xy_m"][0] - shuttle_xy[0], track["court_xy_m"][1] - shuttle_xy[1]))
        distance = hypot(nearest["court_xy_m"][0] - shuttle_xy[0], nearest["court_xy_m"][1] - shuttle_xy[1])
        if nearest["status"] != "detected" or distance > 1.35:
            return []
        return [{
            "status": "candidate",
            "hitter_track_id": nearest["track_id"],
            "confidence": round(min(0.45, shuttle["confidence"] * nearest["confidence"]), 4),
            "reason": "spatial_proximity_only; not score evidence",
        }]
