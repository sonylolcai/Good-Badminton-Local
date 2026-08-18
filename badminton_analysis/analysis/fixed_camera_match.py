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
from ..tracking.bytetrack_adapter import ByteTrackAdapter


SCHEMA_VERSION = "2.1"


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
    last_observation_frame: int
    velocity_mps: Tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    observations: int = 1
    last_evidence: dict = field(default_factory=dict)
    association_key: Optional[str] = None
    association_source: str = "court_association"
    image_history: list = field(default_factory=list)


class CourtMultiObjectTracker:
    """Court-constrained persistent tracker shared by singles and doubles.

    The tracker never uses image up/down/left/right as an identity rule.
    ``track_id`` is the only durable individual key, while ``zone_id`` and
    ``court_end`` are momentary spatial facts.  ByteTrack, when enabled by the
    evaluation gate, supplies an association hint; court metres remain a
    transparent constraint and fallback.
    """

    def __init__(
        self,
        court_space,
        fps,
        max_missed_frames=12,
        max_speed_mps=10.0,
        match_mode="singles",
        max_retained_missing_frames=None,
        lock_match_roster=False,
        expected_roster_count=None,
        roster_stable_frames=2,
        roster_reacquire_seconds=1.0,
    ):
        if match_mode not in {"singles", "doubles"}:
            raise ValueError("match_mode must be 'singles' or 'doubles'")
        self.court_space = court_space
        self.fps = float(fps)
        self.max_missed_frames = int(max_missed_frames)
        self.max_retained_missing_frames = int(
            max_retained_missing_frames or max(self.max_missed_frames * 10, self.max_missed_frames + 1)
        )
        self.max_speed_mps = float(max_speed_mps)
        self.match_mode = match_mode
        self.max_players_per_team = 1 if match_mode == "singles" else 2
        self.lock_match_roster = bool(lock_match_roster)
        self.expected_roster_count = int(
            expected_roster_count or (2 if match_mode == "singles" else 4)
        )
        if self.expected_roster_count <= 0:
            raise ValueError("expected_roster_count must be positive")
        self.roster_stable_frames = max(1, int(roster_stable_frames))
        # Keep the normal short prediction window unchanged, then allow one
        # additional bounded window for a real detection to reclaim a locked
        # roster ID. Beyond this, a location-only guess is not trustworthy.
        self.roster_reacquire_frames = self.max_missed_frames + max(
            1,
            int(round(float(roster_reacquire_seconds) * self.fps)),
        )
        self.tracks = {}
        self._next_id = 1
        self.identity_claims = {}
        self.team_claims = {}
        self._association_keys = {}
        self.track_metrics = {}
        self.roster_status = "bootstrapping" if self.lock_match_roster else "disabled"
        self.roster_locked_frame = None
        self.roster_track_ids = []
        self._roster_stable_observation_frames = 0
        self._last_roster_candidate_count = 0
        self._last_roster_reason = "awaiting_stable_on_court_detections" if self.lock_match_roster else "disabled"
        self._last_unassigned_observation_count = 0

    def update(self, frame_index, observations):
        observations = [
            item for item in observations
            if item.get("court_xy") is not None
            and self.court_space.contains(item["court_xy"], margin_m=0.35)
        ]
        if self.lock_match_roster and self.roster_status != "locked":
            return self._bootstrap_roster(frame_index, observations)

        return self._update_locked_or_open_tracks(frame_index, observations)

    def _bootstrap_roster(self, frame_index, observations):
        """Lock the match roster only after a stable, complete on-court sample.

        A camera may initially see only one singles player or briefly include a
        referee.  Locking that frame would permanently encode a bad roster, so
        the tracker waits for the configured count on consecutive frames.
        """
        self._last_roster_candidate_count = len(observations)
        self._last_unassigned_observation_count = 0
        if len(observations) != self.expected_roster_count:
            self._roster_stable_observation_frames = 0
            self._last_roster_reason = (
                "waiting_for_expected_on_court_count"
                f" (observed={len(observations)}, expected={self.expected_roster_count})"
            )
            return self.snapshot(frame_index)

        self._roster_stable_observation_frames += 1
        if self._roster_stable_observation_frames < self.roster_stable_frames:
            self._last_roster_reason = "awaiting_second_stable_roster_observation"
            return self.snapshot(frame_index)

        # The first IDs are deterministic labels only; they do not claim a
        # real name, team, or image-side identity.  Court coordinates make the
        # ordering independent of whether the camera is rear, side, or oblique.
        ordered = sorted(
            observations,
            key=lambda item: (float(item["court_xy"][1]), float(item["court_xy"][0])),
        )
        for observation in ordered:
            self._create_track(frame_index, observation, association_source="roster_bootstrap")
        self.roster_track_ids = sorted(self.tracks)
        self.roster_status = "locked"
        self.roster_locked_frame = int(frame_index)
        self._last_roster_reason = "stable_expected_on_court_count"
        return self.snapshot(frame_index)

    def _update_locked_or_open_tracks(self, frame_index, observations):
        unmatched_track_ids = set(self.tracks)
        unmatched_observations = set(range(len(observations)))
        assignments = []
        assignment_sources = {}
        self._last_roster_candidate_count = len(observations)
        self._last_unassigned_observation_count = 0

        # A confirmed ByteTrack key takes priority over metric association.
        # It may revive a temporarily missing court track after an occlusion.
        for index, observation in enumerate(observations):
            association_key = observation.get("association_key")
            track_id = self._association_keys.get(association_key)
            if track_id not in unmatched_track_ids or index not in unmatched_observations:
                continue
            assignments.append((track_id, index))
            assignment_sources[(track_id, index)] = "bytetrack"
            unmatched_track_ids.remove(track_id)
            unmatched_observations.remove(index)

        # Greedy assignment is deterministic, and sufficient for the two-player
        # first version.  Its gate is in metres, so a side/oblique image view
        # has exactly the same identity behavior as a rear view.
        candidates = []
        for track_id, track in self.tracks.items():
            if track_id not in unmatched_track_ids:
                continue
            if track.missed_frames > self.max_missed_frames:
                if not (
                    self.lock_match_roster
                    and track.missed_frames <= self.roster_reacquire_frames
                ):
                    continue
                source = "roster_reassociation"
                gate = self._roster_reassociation_gate(track)
            else:
                source = "court_association"
                gate = self._association_gate(track, frame_index)
            predicted = self._predict_position(track, frame_index)
            for index, observation in enumerate(observations):
                distance = self._distance(predicted, observation["court_xy"])
                if distance <= gate:
                    candidates.append((distance, track_id, index, source))
        for _distance, track_id, index, source in sorted(candidates):
            if track_id not in unmatched_track_ids or index not in unmatched_observations:
                continue
            assignments.append((track_id, index))
            assignment_sources[(track_id, index)] = source
            unmatched_track_ids.remove(track_id)
            unmatched_observations.remove(index)

        for track_id, index in assignments:
            self._apply_observation(
                self.tracks[track_id],
                frame_index,
                observations[index],
                association_source=assignment_sources[(track_id, index)],
            )
        if self.lock_match_roster:
            # The roster is a match fact: detections outside it are retained as
            # a count in the evidence, but may not silently become a new player.
            self._last_unassigned_observation_count = len(unmatched_observations)
        else:
            for index in sorted(unmatched_observations):
                self._create_track(frame_index, observations[index])
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_frames += max(1, frame_index - track.last_frame)
            track.last_frame = frame_index
            self.track_metrics[track_id]["missing_frames"] += 1
            if not self.lock_match_roster and track.missed_frames > self.max_retained_missing_frames:
                if track.association_key:
                    self._association_keys.pop(track.association_key, None)
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

    def claim_teams(self, claims):
        """Attach a human-confirmed match team without encoding a court side.

        Teams change court ends during a match; therefore this method accepts
        a stable match relation (for example ``team_a``/``team_b``) rather than
        deriving it from the current location.  It deliberately does not force
        a team when evidence is absent.
        """
        unknown = set(claims).difference(self.track_metrics)
        if unknown:
            raise ValueError(f"Unknown track IDs: {sorted(unknown)}")
        normalized = {str(key): str(value) for key, value in claims.items() if value is not None and str(value).strip()}
        self.team_claims.update(normalized)
        return dict(self.team_claims)

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
                "team_id": self.team_claims.get(track_id),
                "detected_frames": metrics["detected_frames"],
                "predicted_frames": metrics["predicted_frames"],
                "missing_frames": metrics["missing_frames"],
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
            predicted = 0 < track.missed_frames <= self.max_missed_frames
            missing = track.missed_frames > self.max_missed_frames
            court_xy = self._predict_position(track, frame_index) if predicted else track.court_xy
            if predicted:
                self.track_metrics[track_id]["predicted_frames"] += 1
            location_evidence = dict(track.last_evidence)
            location_evidence["measurement_frame"] = int(track.last_observation_frame)
            location_evidence["is_current_measurement"] = not predicted and not missing
            records.append(
                {
                    "track_id": track.track_id,
                    "person_id": self.identity_claims.get(track.track_id),
                    "team_id": self.team_claims.get(track.track_id),
                    "image_xy": self._as_list(track.image_xy),
                    "court_xy_m": self._as_list(court_xy),
                    "zone_id": self.court_space.zone_for(court_xy),
                    "court_end": self._court_end(court_xy),
                    "status": "missing" if missing else "predicted" if predicted else "detected",
                    "confidence": 0.0 if missing else round(track.confidence * (0.72 ** track.missed_frames), 4),
                    "missed_frames": track.missed_frames,
                    "association": {
                        "key": track.association_key,
                        "source": track.association_source,
                    },
                    "location_evidence": location_evidence,
                    "trajectory_image": [self._as_list(point) for point in track.image_history],
                }
            )
        return records

    def roster_summary(self):
        """Expose roster state without turning a transient zone into identity."""
        return {
            "enabled": self.lock_match_roster,
            "status": self.roster_status,
            "expected_player_count": self.expected_roster_count if self.lock_match_roster else None,
            "observed_on_court_candidate_count": self._last_roster_candidate_count,
            "stable_observation_frames": self._roster_stable_observation_frames,
            "stable_frames_required": self.roster_stable_frames if self.lock_match_roster else None,
            "locked_frame": self.roster_locked_frame,
            "track_ids": list(self.roster_track_ids),
            "unassigned_observation_count": self._last_unassigned_observation_count,
            "reason": self._last_roster_reason,
            "policy": (
                "Roster members are fixed after bootstrap. Extra detections are not new players; "
                "detected, predicted, and missing remain explicit evidence states."
                if self.lock_match_roster else "Roster locking disabled for backward-compatible callers."
            ),
        }

    def _create_track(self, frame_index, observation, association_source="court_association"):
        track_id = f"track_{self._next_id:03d}"
        self._next_id += 1
        image_xy = self._tuple_or_none(observation.get("image_xy"))
        if (
            association_source == "court_association"
            and str(observation.get("association_key") or "").startswith("bytetrack_")
        ):
            association_source = "bytetrack"
        self.tracks[track_id] = _Track(
            track_id=track_id,
            court_xy=tuple(float(value) for value in observation["court_xy"]),
            image_xy=image_xy,
            confidence=float(observation.get("confidence", 0.0)),
            last_frame=int(frame_index),
            last_observation_frame=int(frame_index),
            last_evidence=self._evidence(observation),
            association_key=observation.get("association_key"),
            association_source=association_source,
            image_history=[image_xy] if image_xy is not None else [],
        )
        if observation.get("association_key"):
            self._association_keys[observation["association_key"]] = track_id
        self.track_metrics[track_id] = {
            "detected_frames": 1,
            "predicted_frames": 0,
            "missing_frames": 0,
            "distance_m": 0.0,
            "zone_frames": {self.court_space.zone_for(observation["court_xy"]): 1},
        }

    def _apply_observation(self, track, frame_index, observation, association_source="court_association"):
        new_xy = tuple(float(value) for value in observation["court_xy"])
        distance = self._distance(track.court_xy, new_xy)
        elapsed = max(1, int(frame_index) - track.last_frame) / self.fps
        velocity = ((new_xy[0] - track.court_xy[0]) / elapsed, (new_xy[1] - track.court_xy[1]) / elapsed)
        speed = hypot(*velocity)
        if speed <= self.max_speed_mps:
            track.velocity_mps = velocity
        track.court_xy = new_xy
        track.image_xy = self._tuple_or_none(observation.get("image_xy"))
        if track.image_xy is not None:
            track.image_history.append(track.image_xy)
            track.image_history = track.image_history[-30:]
        track.confidence = float(observation.get("confidence", 0.0))
        track.last_frame = int(frame_index)
        track.last_observation_frame = int(frame_index)
        track.missed_frames = 0
        track.observations += 1
        track.last_evidence = self._evidence(observation)
        track.association_source = association_source
        association_key = observation.get("association_key")
        if association_key:
            if track.association_key and track.association_key != association_key:
                self._association_keys.pop(track.association_key, None)
            track.association_key = association_key
            self._association_keys[association_key] = track.track_id
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

    def _roster_reassociation_gate(self, track):
        """A conservative, short-gap recovery gate for a locked roster.

        The rule allows a player who reappears after a brief detector gap to
        reclaim the existing ID.  It intentionally stops before a long gap;
        such a recovery requires ByteTrack evidence or remains ``missing``.
        """
        return min(3.5, max(1.2, 0.35 + self.max_speed_mps * self.roster_reacquire_frames / self.fps))

    @staticmethod
    def _distance(left, right):
        return hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))

    @staticmethod
    def _tuple_or_none(value):
        return tuple(float(item) for item in value[:2]) if value is not None else None

    @staticmethod
    def _as_list(value):
        return [round(float(item), 4) for item in value] if value is not None else None

    def _court_end(self, court_xy):
        if court_xy is None:
            return None
        return "end_a" if float(court_xy[1]) < self.court_space.net_y_m else "end_b"

    @staticmethod
    def _evidence(observation):
        bbox = observation.get("bbox_xyxy")
        if bbox is None:
            bbox = observation.get("bbox")
        return {
            "method": observation.get("location_method"),
            "confidence": observation.get("location_confidence"),
            "source": observation.get("source"),
            "bbox_xyxy": CourtMultiObjectTracker._as_list(bbox),
            "hands_image": observation.get("hands_image"),
            "keypoints_image": CourtMultiObjectTracker._points_or_none(
                observation.get("keypoints_image")
            ),
            "keypoint_scores": CourtMultiObjectTracker._numbers_or_none(
                observation.get("keypoint_scores")
            ),
            "degraded": bool(observation.get("location_degraded", False)),
        }

    @staticmethod
    def _points_or_none(value):
        if value is None:
            return None
        try:
            points = []
            for point in value:
                if len(point) < 2:
                    return None
                points.append([round(float(point[0]), 3), round(float(point[1]), 3)])
            return points
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _numbers_or_none(value):
        if value is None:
            return None
        try:
            return [round(float(item), 4) for item in value]
        except (TypeError, ValueError):
            return None


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
    """Compose multi-target tracking, shuttle proxy, and conservative rallies."""

    def __init__(
        self,
        image_corners,
        fps,
        net_image_line=None,
        match_mode="singles",
        tracker_backend="court_association",
        enable_bytetrack=False,
        byte_tracker_factory=None,
        lock_match_roster=False,
        roster_stable_frames=2,
    ):
        if tracker_backend not in {"court_association", "bytetrack"}:
            raise ValueError("tracker_backend must be 'court_association' or 'bytetrack'")
        if tracker_backend == "bytetrack" and not enable_bytetrack:
            raise ValueError(
                "ByteTrack is evaluation-gated. Set enable_bytetrack=True only for a recorded tracker evaluation."
            )
        self.court_space = CourtSpace(image_corners)
        self.net_image_line = net_image_line or [
            self.court_space.court_to_image((0.0, self.court_space.net_y_m)),
            self.court_space.court_to_image((self.court_space.width_m, self.court_space.net_y_m)),
        ]
        self.match_mode = match_mode
        self.tracker_backend = tracker_backend
        self.byte_tracker = (
            ByteTrackAdapter(fps=fps, tracker_factory=byte_tracker_factory)
            if tracker_backend == "bytetrack"
            else None
        )
        self.tracker = CourtMultiObjectTracker(
            self.court_space,
            fps=fps,
            match_mode=match_mode,
            lock_match_roster=lock_match_roster,
            expected_roster_count=2 if match_mode == "singles" else 4,
            roster_stable_frames=roster_stable_frames,
        )
        self.shuttle = MonocularShuttleReconstructor()
        self.rallies = RallyStateMachine(fps=fps)
        self._last_frame = 0

    def update(self, frame_index, observations, shuttlecock):
        self._last_frame = int(frame_index)
        observations = [dict(item) for item in observations]
        if self.byte_tracker is not None:
            association_keys = self.byte_tracker.update(observations)
            for index, association_key in association_keys.items():
                observations[index]["association_key"] = association_key
        tracks = self.tracker.update(frame_index, observations)
        shuttle = self._shuttle_record(frame_index, shuttlecock)
        hit_events = self._detect_hit_events(tracks, shuttle)
        rally = self.rallies.update(frame_index, tracks, shuttle, hit_events)
        return {
            "schema_version": SCHEMA_VERSION,
            "coordinate_system": "standard_badminton_court_m",
            "match": {
                "mode": self.match_mode,
                "max_players_per_team": self.tracker.max_players_per_team,
                "identity_policy": "track_id is persistent; court_end and zone_id are transient; team_id requires confirmation",
            },
            "match_roster": self.tracker.roster_summary(),
            "tracking": {
                "backend": self.tracker_backend,
                "measurement_statuses": ["detected", "predicted", "missing"],
                "prediction_policy": "predicted and missing locations are not detector measurements or score evidence",
            },
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
            "team_claims": dict(self.tracker.team_claims),
            "match": {
                "mode": self.match_mode,
                "max_players_per_team": self.tracker.max_players_per_team,
            },
            "tracking": {
                "backend": self.tracker_backend,
                "bytetrack_evaluation_gate": self.tracker_backend == "bytetrack",
            },
            "match_roster": self.tracker.roster_summary(),
            "player_style_inputs": self.tracker.summaries(),
            "rallies": self.rallies.finalize(self._last_frame),
            "score_policy": "unknown scores are excluded from ability, win/loss, challenge, leaderboard, and key-point statistics",
        }

    def claim_identities(self, claims):
        """Record a post-match human identity claim without rewriting tracks."""
        return self.tracker.claim_identities(claims)

    def claim_teams(self, claims):
        """Record post-match team membership without inferring it from court side."""
        return self.tracker.claim_teams(claims)

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
