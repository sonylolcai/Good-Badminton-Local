"""Court-coordinate match primitives for a fixed-camera badminton match.

This module deliberately has no YOLO, OpenCV display, or WebUI dependency.
It receives already-detected positions and preserves the distinction between
measurements, short-lived predictions, and conclusions that are not supported
by enough evidence.  That makes it safe to test with several camera views and
keeps future doubles rules outside the visual tracking layer.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from math import hypot
from typing import Optional, Tuple

from ..court.mapper import CourtMapper
from ..court.reference import BADMINTON_COURT_LENGTH, BADMINTON_COURT_WIDTH
from ..tracking.bytetrack_adapter import ByteTrackAdapter


SCHEMA_VERSION = "2.3"
TRACKER_STATE_VERSION = "court-multi-object-tracker.v1"

# The current-speed label is an operator-facing display value, not a raw
# detector output.  It must be derived from the same current ``track_id``
# evidence drawn on screen, and it must not turn small foot-point jitter into
# apparent movement.  These limits deliberately affect display only; raw
# per-frame positions remain available in detections.jsonl for audit.
DISPLAY_SPEED_WINDOW_SECONDS = 0.5
DISPLAY_SPEED_MAX_OBSERVATION_GAP_SECONDS = 0.15
DISPLAY_SPEED_STEP_DEAD_ZONE_M = 0.05
DISPLAY_SPEED_DEAD_ZONE_MPS = 0.35

# Ultralytics YOLO Pose emits the standard 17-point COCO skeleton.  The raw
# order is declared once in metadata rather than repeated beside every frame,
# while each ``spatial.tracks[].pose`` record keeps the aligned point/value
# arrays needed for later biomechanical analysis.
POSE_KEYPOINT_FORMAT = "coco17_image_v1"
COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class CourtSpace:
    """A standard-court coordinate system independent of image orientation."""

    def __init__(
        self,
        image_corners,
        court_dimensions=(BADMINTON_COURT_WIDTH, BADMINTON_COURT_LENGTH),
        *,
        world_points_m=None,
        athlete_observation_region="full_court_athletes",
    ):
        self.width_m, self.length_m = (float(value) for value in court_dimensions)
        self.mapper = CourtMapper(
            image_corners,
            court_dimensions=court_dimensions,
            world_points_m=world_points_m,
        )
        self.net_y_m = self.length_m / 2.0
        self.athlete_observation_region = str(athlete_observation_region)

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

    def contains_athlete(
        self,
        court_xy,
        margin_m=0.0,
        *,
        lateral_margin_m=None,
        baseline_margin_m=None,
    ):
        """Return whether a pose may enter the anonymous roster.

        The map itself stays a full sport coordinate system.  A training mode
        can therefore map a near half-court to its true global coordinates
        while excluding people on the far side from tracker bootstrap.
        """
        if court_xy is None:
            return False
        lateral_margin = float(
            margin_m if lateral_margin_m is None else lateral_margin_m
        )
        baseline_margin = float(
            margin_m if baseline_margin_m is None else baseline_margin_m
        )
        x, y = (float(value) for value in court_xy)
        if not (
            -lateral_margin <= x <= self.width_m + lateral_margin
            and -baseline_margin <= y <= self.length_m + baseline_margin
        ):
            return False
        if self.athlete_observation_region == "near_court_athlete":
            # The baseline allowance is for the player's own baseline.  It
            # must not turn a near-half training session into a far-side
            # observer zone around the net.
            return y >= self.net_y_m - float(margin_m)
        return True


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
    # This describes certainty that the current detector measurement belongs
    # to this durable ID.  It is intentionally separate from pose confidence:
    # a crisp pose can still be an uncertain identity after a long occlusion.
    association_identity_confidence: float = 0.85
    image_history: list = field(default_factory=list)
    # A short window of *measured* court positions used only for stable
    # annotation rendering. Predicted positions deliberately never enter this
    # history, so the mini-court does not invent a path through an occlusion.
    court_history: list = field(default_factory=list)
    court_observation_history: list = field(default_factory=list)


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
        max_roster_count=4,
        roster_discovery_seconds=8.0,
        roster_reacquire_seconds=1.0,
        require_association_keys=False,
    ):
        if match_mode not in {"singles", "doubles", "person_only"}:
            raise ValueError("match_mode must be 'singles', 'doubles', or 'person_only'")
        self.court_space = court_space
        self.fps = float(fps)
        self.max_missed_frames = int(max_missed_frames)
        self.max_retained_missing_frames = int(
            max_retained_missing_frames or max(self.max_missed_frames * 10, self.max_missed_frames + 1)
        )
        self.max_speed_mps = float(max_speed_mps)
        self.match_mode = match_mode
        self.max_players_per_team = (
            1 if match_mode == "singles" else 2 if match_mode == "doubles" else None
        )
        self.lock_match_roster = bool(lock_match_roster)
        default_roster_count = 2 if match_mode == "singles" else 4
        # Anonymous streaming cannot ask the user whether a game is singles,
        # doubles, or an informal uneven game.  When roster locking is enabled
        # in ``person_only`` mode, a stable on-court count becomes the roster
        # size during bootstrap.  It is still anonymous: this is a continuity
        # constraint, never a player/team/side identity claim.
        self.expected_roster_count = (
            None
            if match_mode == "person_only" and expected_roster_count is None
            else int(expected_roster_count or default_roster_count)
        )
        if self.expected_roster_count is not None and self.expected_roster_count <= 0:
            raise ValueError("expected_roster_count must be positive")
        self.roster_stable_frames = max(1, int(roster_stable_frames))
        self.max_roster_count = max(1, int(max_roster_count))
        if (
            self.expected_roster_count is not None
            and self.expected_roster_count > self.max_roster_count
        ):
            raise ValueError("expected_roster_count cannot exceed max_roster_count")
        self.roster_discovery_seconds = max(0.0, float(roster_discovery_seconds))
        self.roster_discovery_frames = max(
            1,
            int(round(self.roster_discovery_seconds * self.fps)),
        )
        # A production ByteTrack roster must be built from confirmed tracker
        # keys.  Otherwise the roster can lock during ByteTrack's tentative
        # warm-up period and lose the durable association it was meant to
        # preserve.  The court-only fallback intentionally remains usable for
        # deterministic tests and backwards-compatible callers.
        self.require_association_keys = bool(require_association_keys)
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
        # In a fixed ByteTrack roster, a count of four boxes is not evidence
        # of four people.  Retain the exact confirmed-key set that began the
        # discovery window so a warm-up/key-fragment frame cannot become four
        # permanent player slots.
        self._roster_bootstrap_association_keys = None
        self._last_roster_candidate_count = 0
        self._roster_discovery_started_frame = None
        self._roster_observed_max_candidate_count = 0
        self._last_roster_reason = "awaiting_stable_on_court_detections" if self.lock_match_roster else "disabled"
        self._last_unassigned_observation_count = 0
        self._last_unassigned_observations = []

    def update(self, frame_index, observations, has_fresh_observations=True):
        """Update tracks from a pose measurement when one is available.

        ``has_fresh_observations`` distinguishes a real empty detector result
        from a source frame intentionally skipped by the configured sampling
        policy.  Treating both as ``[]`` made a 10 Hz detector reset the
        roster on every intervening 30 fps source frame.
        """
        observations = [
            item for item in observations
            if item.get("court_xy") is not None
            and self.court_space.contains(item["court_xy"], margin_m=0.35)
        ]
        if self.lock_match_roster and self.roster_status != "locked":
            if not has_fresh_observations:
                self._last_roster_reason = "awaiting_next_fresh_pose_measurement"
                return self.snapshot(frame_index)
            return self._bootstrap_roster(frame_index, observations)

        return self._update_locked_or_open_tracks(frame_index, observations)

    def _bootstrap_roster(self, frame_index, observations):
        """Start at two real people, then expand to at most four before lock.

        The visual service cannot know whether a game is singles, doubles, or
        informal. It starts only after at least two on-court people have stable
        evidence. Until the discovery deadline, newly confirmed detections may
        expand that anonymous roster to three or four; missing people are never
        created merely to fill capacity.
        """
        previous_candidate_count = self._last_roster_candidate_count
        self._last_roster_candidate_count = len(observations)
        self._last_unassigned_observation_count = 0
        self._last_unassigned_observations = []
        candidate_count = len(observations)
        if self.expected_roster_count is None:
            if self._roster_discovery_started_frame is None:
                self._roster_discovery_started_frame = int(frame_index)
            if candidate_count > self.max_roster_count:
                self._roster_stable_observation_frames = 0
                self._last_roster_reason = (
                    "waiting_for_on_court_count_within_max"
                    f" (observed={candidate_count}, max={self.max_roster_count})"
                )
                return self.snapshot(frame_index)
            if self.require_association_keys and any(
                not observation.get("association_key") for observation in observations
            ):
                self._roster_stable_observation_frames = 0
                self._last_roster_reason = "awaiting_bytetrack_confirmation"
                return self.snapshot(frame_index)
            self._roster_observed_max_candidate_count = max(
                self._roster_observed_max_candidate_count,
                candidate_count,
            )
            if not self.tracks:
                if candidate_count < 2:
                    self._roster_stable_observation_frames = 0
                    self._last_roster_reason = "waiting_for_at_least_two_on_court_people"
                    return self.snapshot(frame_index)
                if previous_candidate_count != candidate_count:
                    self._roster_stable_observation_frames = 1
                else:
                    self._roster_stable_observation_frames += 1
                if self._roster_stable_observation_frames < self.roster_stable_frames:
                    self._last_roster_reason = "awaiting_second_stable_roster_observation"
                    return self.snapshot(frame_index)

                ordered = sorted(
                    observations,
                    key=lambda item: (float(item["court_xy"][1]), float(item["court_xy"][0])),
                )
                for observation in ordered:
                    self._create_track(
                        frame_index,
                        observation,
                        association_source="roster_discovery_bootstrap",
                    )
                self.roster_track_ids = sorted(self.tracks)
                self.roster_status = "discovering"
                self._last_roster_reason = "at_least_two_people_confirmed_discovery_open"
                if len(self.roster_track_ids) >= self.max_roster_count:
                    return self._lock_discovered_roster(frame_index, "discovery_observed_max_roster_count")
                return self.snapshot(frame_index)

            # The initial two-or-more people have valid IDs. While discovery is
            # open, use normal association but permit newly confirmed on-court
            # observations to add the third/fourth anonymous ID.
            tracks = self._update_discovering_roster(frame_index, observations)
            self.roster_track_ids = sorted(self.tracks)
            discovery_elapsed = int(frame_index) - int(self._roster_discovery_started_frame)
            if len(self.roster_track_ids) >= self.max_roster_count:
                return self._lock_discovered_roster(frame_index, "discovery_observed_max_roster_count")
            if discovery_elapsed >= self.roster_discovery_frames:
                return self._lock_discovered_roster(
                    frame_index,
                    "discovery_deadline_observed_roster_count",
                )
            self.roster_status = "discovering"
            self._last_roster_reason = (
                "roster_discovery_in_progress"
                f" (elapsed_frames={discovery_elapsed}, "
                f"required_frames={self.roster_discovery_frames}, "
                f"observed_max={self._roster_observed_max_candidate_count}, "
                f"current_tracks={len(self.roster_track_ids)}, max={self.max_roster_count})"
            )
            return tracks

        if candidate_count != self.expected_roster_count:
            self._roster_stable_observation_frames = 0
            self._roster_bootstrap_association_keys = None
            self._roster_discovery_started_frame = None
            self._last_roster_reason = (
                "waiting_for_expected_on_court_count"
                f" (observed={candidate_count}, expected={self.expected_roster_count})"
            )
            return self.snapshot(frame_index)

        if self.require_association_keys and any(
            not observation.get("association_key") for observation in observations
        ):
            self._roster_stable_observation_frames = 0
            self._roster_bootstrap_association_keys = None
            self._roster_discovery_started_frame = None
            self._last_roster_reason = "awaiting_bytetrack_confirmation"
            return self.snapshot(frame_index)

        # The direct GPU stream runs person_only and carries a user-selected
        # count.  It needs a real discovery window because pose/ByteTrack are
        # still warming up.  The match pipeline keeps its established
        # fixed-camera semantics: an explicit singles/doubles mode may lock as
        # soon as its stable count is available.
        if self.match_mode != "person_only":
            self._roster_stable_observation_frames += 1
            if self._roster_stable_observation_frames < self.roster_stable_frames:
                self._last_roster_reason = "awaiting_second_stable_roster_observation"
                return self.snapshot(frame_index)
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

        association_keys = None
        if self.require_association_keys:
            association_keys = tuple(
                sorted(str(observation["association_key"]) for observation in observations)
            )
            if len(set(association_keys)) != self.expected_roster_count:
                self._roster_stable_observation_frames = 0
                self._roster_bootstrap_association_keys = None
                self._roster_discovery_started_frame = None
                self._last_roster_reason = "awaiting_distinct_bytetrack_confirmation"
                return self.snapshot(frame_index)

        if (
            self._roster_discovery_started_frame is None
            or association_keys != self._roster_bootstrap_association_keys
        ):
            self._roster_bootstrap_association_keys = association_keys
            self._roster_stable_observation_frames = 1
            self._roster_discovery_started_frame = int(frame_index)
        else:
            self._roster_stable_observation_frames += 1
        if self._roster_stable_observation_frames < self.roster_stable_frames:
            self._last_roster_reason = "awaiting_second_stable_roster_observation"
            return self.snapshot(frame_index)

        discovery_elapsed = int(frame_index) - int(self._roster_discovery_started_frame)
        if discovery_elapsed < self.roster_discovery_frames:
            self._last_roster_reason = (
                "awaiting_expected_roster_discovery_window"
                f" (elapsed_frames={discovery_elapsed}, "
                f"required_frames={self.roster_discovery_frames}, "
                f"expected={self.expected_roster_count})"
            )
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
        self._last_roster_reason = "stable_expected_bytetrack_roster_after_discovery_window"
        return self.snapshot(frame_index)

    def _update_discovering_roster(self, frame_index, observations):
        """Associate provisional tracks while allowing evidence-based expansion."""
        original_lock = self.lock_match_roster
        self.lock_match_roster = False
        try:
            return self._update_locked_or_open_tracks(frame_index, observations)
        finally:
            self.lock_match_roster = original_lock

    def _lock_discovered_roster(self, frame_index, reason):
        """Seal only the real IDs accumulated during the discovery window."""
        self.expected_roster_count = len(self.roster_track_ids)
        self.roster_status = "locked"
        self.roster_locked_frame = int(frame_index)
        self._last_roster_reason = reason
        return self.snapshot(frame_index)

    def _update_locked_or_open_tracks(self, frame_index, observations):
        unmatched_track_ids = set(self.tracks)
        unmatched_observations = set(range(len(observations)))
        assignments = []
        assignment_sources = {}
        self._last_roster_candidate_count = len(observations)
        self._last_unassigned_observation_count = 0
        self._last_unassigned_observations = []

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

        # Greedy metric assignment covers continuous observations when no
        # durable identity backend is required.  In a locked ByteTrack roster,
        # a newly seen key must stay unassigned: assigning it solely because it
        # happens to be near an old court coordinate is how one player gets
        # copied into another player's permanent slot.
        candidates = []
        if not self.require_association_keys:
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

        # After a long physical overlap, the normal motion gate deliberately
        # expires.  In a locked doubles roster we may still recover a *real*
        # returning pose when there is exactly one missing ID and exactly one
        # unassigned detector observation on the same current court end.
        #
        # This is not a team/upper/lower identity rule: court end is only a
        # short-lived spatial constraint at the recovery timestamp.  Two
        # teammates returning on the same end remain unassigned rather than
        # being guessed, and the recovered association is explicitly low
        # confidence for downstream analytics.
        if not self.require_association_keys:
            for track_id, index in self._unambiguous_long_gap_recoveries(
                unmatched_track_ids,
                unmatched_observations,
                observations,
            ):
                assignments.append((track_id, index))
                assignment_sources[(track_id, index)] = "roster_end_recovery"
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
            self._last_unassigned_observations = [
                self._unassigned_observation_evidence(observations[index])
                for index in sorted(unmatched_observations)
            ]
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
                "recovered_detected_frames": metrics["recovered_detected_frames"],
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
            # ``pose`` below is the sole exported raw-skeleton field.  Keeping
            # historical points here would duplicate large arrays and make a
            # predicted row look as though it contained a new pose reading.
            location_evidence.pop("keypoints_image", None)
            location_evidence.pop("keypoint_scores", None)
            location_evidence["measurement_frame"] = int(track.last_observation_frame)
            location_evidence["is_current_measurement"] = not predicted and not missing
            is_current_measurement = not predicted and not missing
            pose = self._pose_record(track, is_current_measurement=is_current_measurement)
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
                    "motion": self._motion_evidence(
                        track,
                        frame_index,
                        is_current_measurement=is_current_measurement,
                    ),
                    "association": {
                        "key": track.association_key,
                        "source": track.association_source,
                        "identity_confidence": round(track.association_identity_confidence, 4),
                    },
                    "location_evidence": location_evidence,
                    # Keep pose separate from foot-point evidence.  Consumers can
                    # read one stable, per-track raw-pose contract without
                    # mistaking a retained location record for a fresh skeleton.
                    "pose": pose,
                    "trajectory_image": [self._as_list(point) for point in track.image_history],
                    "trajectory_court_m": [self._as_list(point) for point in track.court_history],
                }
            )
        return records

    def roster_summary(self):
        """Expose roster state without turning a transient zone into identity."""
        return {
            "enabled": self.lock_match_roster,
            "status": self.roster_status,
            "expected_player_count": self.expected_roster_count if self.lock_match_roster else None,
            "minimum_player_count": 2 if self.match_mode == "person_only" and self.lock_match_roster else None,
            "maximum_player_count": self.max_roster_count if self.lock_match_roster else None,
            "discovery_seconds": self.roster_discovery_seconds if self.lock_match_roster else None,
            "discovery_started_frame": self._roster_discovery_started_frame,
            "observed_max_candidate_count": self._roster_observed_max_candidate_count,
            "observed_on_court_candidate_count": self._last_roster_candidate_count,
            "stable_observation_frames": self._roster_stable_observation_frames,
            "stable_frames_required": self.roster_stable_frames if self.lock_match_roster else None,
            "locked_frame": self.roster_locked_frame,
            "track_ids": list(self.roster_track_ids),
            "unassigned_observation_count": self._last_unassigned_observation_count,
            "unassigned_observations": list(self._last_unassigned_observations),
            "reason": self._last_roster_reason,
            "policy": (
                "Roster members are fixed after bootstrap. Extra detections are not new players; "
                "detected, predicted, and missing remain explicit evidence states."
                if self.lock_match_roster else "Roster locking disabled for backward-compatible callers."
            ),
        }

    def snapshot_state(self):
        """Return JSON-safe association state for segment checkpointing.

        This captures only court-space tracking state.  An external ByteTrack
        runtime owns additional Kalman/filter state and must either provide its
        own checkpoint or fail closed during restore; callers may not silently
        claim that a newly created ByteTrack instance preserves old IDs.
        """
        return {
            "state_version": TRACKER_STATE_VERSION,
            "match_mode": self.match_mode,
            "fps": self.fps,
            "max_missed_frames": self.max_missed_frames,
            "max_retained_missing_frames": self.max_retained_missing_frames,
            "max_speed_mps": self.max_speed_mps,
            "lock_match_roster": self.lock_match_roster,
            "expected_roster_count": self.expected_roster_count,
            "max_roster_count": self.max_roster_count,
            "roster_discovery_seconds": self.roster_discovery_seconds,
            "next_id": self._next_id,
            "tracks": {
                track_id: {
                    "track_id": track.track_id,
                    "court_xy": list(track.court_xy),
                    "image_xy": list(track.image_xy) if track.image_xy is not None else None,
                    "confidence": track.confidence,
                    "last_frame": track.last_frame,
                    "last_observation_frame": track.last_observation_frame,
                    "velocity_mps": list(track.velocity_mps),
                    "missed_frames": track.missed_frames,
                    "observations": track.observations,
                    "last_evidence": deepcopy(track.last_evidence),
                    "association_key": track.association_key,
                    "association_source": track.association_source,
                    "association_identity_confidence": track.association_identity_confidence,
                    "image_history": [list(point) for point in track.image_history],
                    "court_history": [list(point) for point in track.court_history],
                    "court_observation_history": [
                        [int(frame), list(point)]
                        for frame, point in track.court_observation_history
                    ],
                }
                for track_id, track in sorted(self.tracks.items())
            },
            "association_keys": dict(self._association_keys),
            "track_metrics": deepcopy(self.track_metrics),
            "identity_claims": dict(self.identity_claims),
            "team_claims": dict(self.team_claims),
            "roster": {
                "status": self.roster_status,
                "locked_frame": self.roster_locked_frame,
                "track_ids": list(self.roster_track_ids),
                "stable_observation_frames": self._roster_stable_observation_frames,
                "bootstrap_association_keys": list(self._roster_bootstrap_association_keys)
                if self._roster_bootstrap_association_keys is not None
                else None,
                "last_candidate_count": self._last_roster_candidate_count,
                "discovery_started_frame": self._roster_discovery_started_frame,
                "observed_max_candidate_count": self._roster_observed_max_candidate_count,
                "last_reason": self._last_roster_reason,
                "last_unassigned_observation_count": self._last_unassigned_observation_count,
                "last_unassigned_observations": deepcopy(self._last_unassigned_observations),
            },
        }

    def restore_state(self, state):
        """Restore a state created by :meth:`snapshot_state`, failing closed."""
        if not isinstance(state, dict) or state.get("state_version") != TRACKER_STATE_VERSION:
            raise ValueError("unsupported court tracker checkpoint")
        if state.get("match_mode") != self.match_mode:
            raise ValueError("court tracker checkpoint match_mode does not match")
        if abs(float(state.get("fps", 0.0)) - self.fps) > 1e-6:
            raise ValueError("court tracker checkpoint fps does not match")
        # A person-only session may infer its anonymous roster size only after
        # a stable opening sample.  A fresh processor starts with ``None``;
        # carry the persisted inferred size forward before verifying the
        # remaining checkpoint configuration.
        persisted_roster_count = state.get("expected_roster_count")
        if (
            self.match_mode == "person_only"
            and self.lock_match_roster
            and self.expected_roster_count is None
            and persisted_roster_count is not None
        ):
            self.expected_roster_count = int(persisted_roster_count)
        expected_configuration = {
            "max_missed_frames": self.max_missed_frames,
            "max_retained_missing_frames": self.max_retained_missing_frames,
            "max_speed_mps": self.max_speed_mps,
            "lock_match_roster": self.lock_match_roster,
            "expected_roster_count": self.expected_roster_count,
            "max_roster_count": self.max_roster_count,
            "roster_discovery_seconds": self.roster_discovery_seconds,
        }
        restored_configuration = {
            "max_missed_frames": int(state.get("max_missed_frames", -1)),
            "max_retained_missing_frames": int(
                state.get("max_retained_missing_frames", -1)
            ),
            "max_speed_mps": float(state.get("max_speed_mps", -1.0)),
            "lock_match_roster": bool(state.get("lock_match_roster", False)),
            "expected_roster_count": state.get("expected_roster_count"),
            "max_roster_count": int(state.get("max_roster_count", self.max_roster_count)),
            "roster_discovery_seconds": float(
                state.get("roster_discovery_seconds", self.roster_discovery_seconds)
            ),
        }
        if restored_configuration != expected_configuration:
            raise ValueError("court tracker checkpoint configuration does not match")

        restored_tracks = {}
        for track_id, record in dict(state.get("tracks") or {}).items():
            restored_tracks[str(track_id)] = _Track(
                track_id=str(record["track_id"]),
                court_xy=tuple(float(value) for value in record["court_xy"][:2]),
                image_xy=self._tuple_or_none(record.get("image_xy")),
                confidence=float(record.get("confidence", 0.0)),
                last_frame=int(record["last_frame"]),
                last_observation_frame=int(record["last_observation_frame"]),
                velocity_mps=tuple(float(value) for value in record.get("velocity_mps", [0.0, 0.0])[:2]),
                missed_frames=int(record.get("missed_frames", 0)),
                observations=int(record.get("observations", 0)),
                last_evidence=deepcopy(record.get("last_evidence") or {}),
                association_key=record.get("association_key"),
                association_source=str(record.get("association_source") or "court_association"),
                association_identity_confidence=float(
                    record.get("association_identity_confidence", 0.0)
                ),
                image_history=[
                    tuple(float(value) for value in point[:2])
                    for point in record.get("image_history", [])
                ],
                court_history=[
                    tuple(float(value) for value in point[:2])
                    for point in record.get("court_history", [])
                ],
                court_observation_history=[
                    (int(item[0]), tuple(float(value) for value in item[1][:2]))
                    for item in record.get("court_observation_history", [])
                ],
            )
        self.tracks = restored_tracks
        self._next_id = int(state.get("next_id", 1))
        self._association_keys = {
            str(key): str(value)
            for key, value in dict(state.get("association_keys") or {}).items()
        }
        self.track_metrics = deepcopy(state.get("track_metrics") or {})
        self.identity_claims = {
            str(key): str(value)
            for key, value in dict(state.get("identity_claims") or {}).items()
        }
        self.team_claims = {
            str(key): str(value)
            for key, value in dict(state.get("team_claims") or {}).items()
        }
        roster = dict(state.get("roster") or {})
        self.roster_status = str(roster.get("status") or self.roster_status)
        self.roster_locked_frame = roster.get("locked_frame")
        self.roster_track_ids = [str(value) for value in roster.get("track_ids", [])]
        self._roster_stable_observation_frames = int(
            roster.get("stable_observation_frames", 0)
        )
        bootstrap_keys = roster.get("bootstrap_association_keys")
        self._roster_bootstrap_association_keys = (
            tuple(str(value) for value in bootstrap_keys)
            if isinstance(bootstrap_keys, list)
            else None
        )
        self._last_roster_candidate_count = int(roster.get("last_candidate_count", 0))
        self._roster_discovery_started_frame = roster.get("discovery_started_frame")
        self._roster_observed_max_candidate_count = int(
            roster.get("observed_max_candidate_count", 0)
        )
        self._last_roster_reason = str(roster.get("last_reason") or self._last_roster_reason)
        self._last_unassigned_observation_count = int(
            roster.get("last_unassigned_observation_count", 0)
        )
        self._last_unassigned_observations = deepcopy(
            roster.get("last_unassigned_observations") or []
        )

    def _create_track(self, frame_index, observation, association_source="court_association"):
        track_id = f"track_{self._next_id:03d}"
        self._next_id += 1
        image_xy = self._tuple_or_none(observation.get("image_xy"))
        court_xy = tuple(float(value) for value in observation["court_xy"])
        if (
            association_source == "court_association"
            and str(observation.get("association_key") or "").startswith("bytetrack_")
        ):
            association_source = "bytetrack"
        self.tracks[track_id] = _Track(
            track_id=track_id,
            court_xy=court_xy,
            image_xy=image_xy,
            confidence=float(observation.get("confidence", 0.0)),
            last_frame=int(frame_index),
            last_observation_frame=int(frame_index),
            last_evidence=self._evidence(observation),
            association_key=observation.get("association_key"),
            association_source=association_source,
            association_identity_confidence=self._identity_confidence_for_source(association_source),
            image_history=[image_xy] if image_xy is not None else [],
            court_history=[court_xy],
            court_observation_history=[
                (int(frame_index), court_xy)
            ],
        )
        if observation.get("association_key"):
            self._association_keys[observation["association_key"]] = track_id
        self.track_metrics[track_id] = {
            "detected_frames": 1,
            "predicted_frames": 0,
            "missing_frames": 0,
            "recovered_detected_frames": 0,
            "distance_m": 0.0,
            "zone_frames": {self.court_space.zone_for(observation["court_xy"]): 1},
        }

    def _apply_observation(self, track, frame_index, observation, association_source="court_association"):
        new_xy = tuple(float(value) for value in observation["court_xy"])
        observation_gap_seconds = max(
            0.0,
            (int(frame_index) - int(track.last_observation_frame)) / self.fps,
        )
        # Never calculate a current speed across a detector gap.  A returning
        # pose is real evidence, but its movement during the unseen interval
        # is unknown and must not be reconstructed for a screen statistic.
        if observation_gap_seconds > DISPLAY_SPEED_MAX_OBSERVATION_GAP_SECONDS:
            track.court_observation_history = []
        track.court_observation_history.append((int(frame_index), new_xy))
        earliest_frame = int(frame_index - self.fps * DISPLAY_SPEED_WINDOW_SECONDS)
        track.court_observation_history = [
            item for item in track.court_observation_history if item[0] >= earliest_frame
        ]
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
        track.court_history.append(new_xy)
        track.court_history = track.court_history[-30:]
        track.confidence = float(observation.get("confidence", 0.0))
        track.last_frame = int(frame_index)
        track.last_observation_frame = int(frame_index)
        track.missed_frames = 0
        track.observations += 1
        track.last_evidence = self._evidence(observation)
        track.association_source = association_source
        track.association_identity_confidence = self._identity_confidence_for_source(association_source)
        association_key = observation.get("association_key")
        if association_key:
            if track.association_key and track.association_key != association_key:
                self._association_keys.pop(track.association_key, None)
            track.association_key = association_key
            self._association_keys[association_key] = track.track_id
        metrics = self.track_metrics[track.track_id]
        metrics["detected_frames"] += 1
        if association_source == "roster_end_recovery":
            # This measurement is kept for review, but the long missing gap
            # must not become a fabricated movement segment in player stats.
            metrics["recovered_detected_frames"] += 1
        else:
            metrics["distance_m"] += distance
        zone = self.court_space.zone_for(new_xy)
        metrics["zone_frames"][zone] = metrics["zone_frames"].get(zone, 0) + 1

    def _motion_evidence(self, track, frame_index, *, is_current_measurement):
        """Return a non-fabricated speed label for the current spatial track.

        ``predicted`` and ``missing`` track records deliberately return no
        speed.  For a current detection, only a short uninterrupted sequence
        of real court observations is used.  Individual steps within five
        centimetres are treated as localisation noise; a remaining speed below
        0.35 m/s is shown as stationary rather than as a false slow walk.
        """
        base = {
            "current_speed_mps": None,
            "status": "not_currently_measured",
            "window_seconds": DISPLAY_SPEED_WINDOW_SECONDS,
            "dead_zone_mps": DISPLAY_SPEED_DEAD_ZONE_MPS,
            "measurement_count": 0,
            "source": "fresh_spatial_track_measurements_only",
        }
        if not is_current_measurement:
            return base
        history = list(track.court_observation_history)
        base["measurement_count"] = len(history)
        if len(history) < 2:
            base["status"] = "not_enough_fresh_measurements"
            return base
        start_frame, _start_xy = history[0]
        end_frame, _end_xy = history[-1]
        elapsed_seconds = (int(end_frame) - int(start_frame)) / self.fps
        if elapsed_seconds <= 0:
            base["status"] = "not_enough_fresh_measurements"
            return base
        distance_m = 0.0
        for (left_frame, left_xy), (right_frame, right_xy) in zip(history, history[1:]):
            step_seconds = (int(right_frame) - int(left_frame)) / self.fps
            if step_seconds <= 0 or step_seconds > DISPLAY_SPEED_MAX_OBSERVATION_GAP_SECONDS:
                base["status"] = "not_enough_fresh_measurements"
                return base
            step_distance = self._distance(left_xy, right_xy)
            if step_distance >= DISPLAY_SPEED_STEP_DEAD_ZONE_M:
                distance_m += step_distance
        raw_speed_mps = distance_m / elapsed_seconds
        speed_mps = 0.0 if raw_speed_mps < DISPLAY_SPEED_DEAD_ZONE_MPS else raw_speed_mps
        base.update({
            "current_speed_mps": round(speed_mps, 3),
            "status": "stationary" if speed_mps == 0.0 else "moving",
            "measured_distance_m": round(distance_m, 4),
            "measured_elapsed_seconds": round(elapsed_seconds, 4),
        })
        return base

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

    def _unambiguous_long_gap_recoveries(self, unmatched_track_ids, unmatched_observations, observations):
        """Return conservative one-to-one locked-roster recovery pairs.

        A player seen again after a long detector gap cannot safely be joined
        by a wider distance gate alone.  The only allowed fallback is one
        missing track and one raw observation on a current court end.  This
        preserves the evidence when an overlap separates while avoiding a
        forced ID switch between two same-side doubles partners.
        """
        if not self.lock_match_roster or self.match_mode != "doubles":
            return []

        tracks_by_end = {}
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            if track.missed_frames <= self.roster_reacquire_frames:
                continue
            court_end = self._court_end(track.court_xy)
            if court_end is not None:
                tracks_by_end.setdefault(court_end, []).append(track_id)

        observations_by_end = {}
        for index in unmatched_observations:
            court_end = self._court_end(observations[index].get("court_xy"))
            if court_end is not None:
                observations_by_end.setdefault(court_end, []).append(index)

        recoveries = []
        for court_end in sorted(set(tracks_by_end).intersection(observations_by_end)):
            candidates = tracks_by_end[court_end]
            detected = observations_by_end[court_end]
            if len(candidates) == 1 and len(detected) == 1:
                recoveries.append((candidates[0], detected[0]))
        return recoveries

    @staticmethod
    def _identity_confidence_for_source(association_source):
        """Keep association uncertainty available to consumers and reviewers."""
        return {
            "bytetrack": 0.95,
            "roster_bootstrap": 0.95,
            "court_association": 0.85,
            "roster_reassociation": 0.72,
            "roster_end_recovery": 0.55,
        }.get(str(association_source), 0.60)

    @staticmethod
    def _unassigned_observation_evidence(observation):
        """Export a reviewable real pose candidate without inventing an ID."""
        return {
            "image_xy": CourtMultiObjectTracker._as_list(observation.get("image_xy")),
            "bbox_xyxy": CourtMultiObjectTracker._as_list(observation.get("bbox_xyxy")),
            "court_xy_m": CourtMultiObjectTracker._as_list(observation.get("court_xy")),
            "confidence": round(float(observation.get("confidence", 0.0)), 4),
            "location_confidence": round(float(observation.get("location_confidence", 0.0)), 4),
            "source": observation.get("source"),
            "reason": "unassigned_after_locked_roster_association",
        }

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
    def pose_keypoint_contract():
        """Return the immutable interpretation contract for raw pose arrays."""
        return {
            "field": "spatial.tracks[].pose",
            "format": POSE_KEYPOINT_FORMAT,
            "coordinate_system": "full_source_image_pixels",
            "keypoint_order": list(COCO17_KEYPOINT_NAMES),
            "point_encoding": "[x, y] or null; array order is always COCO17",
            "score_encoding": "confidence in [0, 1] aligned with keypoint_order",
            "evidence_policy": (
                "Only current detected measurements carry pose points. "
                "Predicted and missing track states use null points and null scores."
            ),
        }

    @classmethod
    def _pose_record(cls, track, is_current_measurement):
        """Serialize raw joints without copying a prior frame into a new one.

        Court tracking can bridge a short foot-point gap, but it must never
        imply that a skeleton was detected in the bridged source frame.  The
        last measurement frame remains traceable for diagnostics while the raw
        values themselves are deliberately null outside a current detection.
        """
        keypoints = None
        scores = None
        if is_current_measurement:
            keypoints = cls._keypoints_or_none(track.last_evidence.get("keypoints_image"))
            scores = cls._keypoint_scores_or_none(track.last_evidence.get("keypoint_scores"))
        return {
            "format": POSE_KEYPOINT_FORMAT,
            "coordinate_system": "full_source_image_pixels",
            "is_current_measurement": bool(is_current_measurement),
            "measurement_frame": int(track.last_observation_frame) if is_current_measurement else None,
            "last_measurement_frame": int(track.last_observation_frame),
            "keypoints_image": keypoints,
            "keypoint_scores": scores,
        }

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
            # These raw arrays are retained inside tracker state so the
            # top-level ``pose`` record can be emitted for the exact source
            # measurement frame.  They are intentionally not duplicated in
            # ``location_evidence`` on every JSONL row.
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
                if point is None:
                    points.append(None)
                    continue
                if len(point) < 2:
                    return None
                points.append([round(float(point[0]), 3), round(float(point[1]), 3)])
            return points
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _numbers_or_none(value):
        if value is None:
            return None
        try:
            return [round(float(item), 4) if item is not None else None for item in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _keypoints_or_none(value):
        """Return exactly 17 aligned image points, preserving invisible slots."""
        points = CourtMultiObjectTracker._points_or_none(value)
        return points[:17] if points is not None and len(points) >= 17 else None

    @staticmethod
    def _keypoint_scores_or_none(value):
        """Return exactly 17 confidence values aligned to the COCO17 points."""
        scores = CourtMultiObjectTracker._numbers_or_none(value)
        return scores[:17] if scores is not None and len(scores) >= 17 else None


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
    """Conservative rally segmentation with rejectable score conclusions.

    When shuttle measurement is explicitly disabled, a missing shuttle must
    never be treated as a terminal event.  The fallback mode below is a
    deliberately weaker *movement-only* boundary signal: a rally may finish
    only after every expected player has a fresh, low-speed court measurement
    throughout a configured window.  It is useful for person-only workload
    analysis, but is not ball, score, or hit evidence.
    """

    MOVEMENT_SETTLE_WINDOWS_SECONDS = (0.5, 0.7, 1.0)

    # A rally may end on a *detected* shuttle that is nearly stationary for a
    # short window (landed, or only small residual movement).  The landed signal
    # never fires until the shuttle has first travelled far enough that a stop
    # can be distinguished from a shuttle held in one place before a serve.
    SHUTTLE_LANDED_WINDOW_SECONDS = 0.4
    SHUTTLE_LANDED_MAX_MOVE_M = 0.5
    SHUTTLE_LANDED_MIN_SPAN_SECONDS = 0.2
    SHUTTLE_LANDED_MIN_PRIOR_MOVE_M = 1.0

    def __init__(
        self,
        fps,
        min_active_frames=4,
        end_gap_frames=None,
        *,
        shuttle_enabled=True,
        expected_player_count=2,
        settle_window_seconds=0.7,
        stable_speed_mps=0.35,
    ):
        self.fps = float(fps)
        self.min_active_frames = int(min_active_frames)
        self.end_gap_frames = int(end_gap_frames or max(10, round(self.fps * 1.2)))
        self.shuttle_enabled = bool(shuttle_enabled)
        self.expected_player_count = max(1, int(expected_player_count))
        self.settle_window_seconds = float(settle_window_seconds)
        if self.settle_window_seconds not in self.MOVEMENT_SETTLE_WINDOWS_SECONDS:
            raise ValueError(
                "settle_window_seconds must be one of "
                f"{self.MOVEMENT_SETTLE_WINDOWS_SECONDS}"
            )
        self.stable_speed_mps = float(stable_speed_mps)
        if self.stable_speed_mps <= 0:
            raise ValueError("stable_speed_mps must be greater than 0")
        self.state = "idle"
        self._active_frames = 0
        self._missing_frames = 0
        self._settled_frames = 0
        self._settled_start_frame = None
        self._last_player_positions = {}
        self._shuttle_positions = []
        self._shuttle_max_move_m = 0.0
        self._rally_id = 0
        self._current = None
        self.completed = []
        self._last_update_frame = None

    def update(self, frame_index, tracks, shuttle, hit_events):
        frame_index = int(frame_index)
        elapsed_source_frames = (
            1 if self._last_update_frame is None
            else max(1, frame_index - self._last_update_frame)
        )
        self._last_update_frame = frame_index
        if not self.shuttle_enabled:
            return self._update_from_player_settle(
                frame_index,
                tracks,
                elapsed_source_frames,
            )
        has_players = len([track for track in tracks if track["status"] == "detected"]) >= 2
        has_shuttle = shuttle is not None and shuttle.get("status") == "approximate"
        if has_shuttle:
            self._record_shuttle_position(frame_index, shuttle)
            if self.state == "idle" and has_players:
                self.state = "candidate"
                self._active_frames = elapsed_source_frames
            elif self.state == "candidate":
                self._active_frames = self._active_frames + elapsed_source_frames if has_players else 0
                if self._active_frames >= self.min_active_frames:
                    self._start(frame_index - self._active_frames + 1)
            elif self.state == "active":
                if hit_events:
                    self._current["hit_events"].extend(hit_events)
                if self._shuttle_landed():
                    self._finish(frame_index, reason="shuttle_landed")
            return self.snapshot()

        # The shuttle is not detected on this frame.  A missing-ball gap is not
        # itself terminal: person stillness marks the rest between two rallies.
        # If an old rally is still open it is closed there, and player motion
        # after that rest opens the next candidate rally.
        if self.state == "active" and hit_events:
            self._current["hit_events"].extend(hit_events)
        return self._update_from_player_settle(frame_index, tracks, elapsed_source_frames)

    def _record_shuttle_position(self, frame_index, shuttle):
        """Keep a bounded window of detected court positions for a landed check."""
        xyz = shuttle.get("xyz_m")
        if not xyz or len(xyz) < 2:
            return
        point = (int(frame_index), float(xyz[0]), float(xyz[1]))
        if self._shuttle_positions:
            previous = self._shuttle_positions[-1]
            self._shuttle_max_move_m = max(
                self._shuttle_max_move_m,
                hypot(point[1] - previous[1], point[2] - previous[2]),
            )
        self._shuttle_positions.append(point)
        cutoff = int(frame_index) - int(round(self.fps * self.SHUTTLE_LANDED_WINDOW_SECONDS))
        self._shuttle_positions = [
            item for item in self._shuttle_positions if item[0] >= cutoff
        ]

    def _shuttle_landed(self):
        """Return True only when a travelled shuttle has just stopped moving."""
        if self._shuttle_max_move_m < self.SHUTTLE_LANDED_MIN_PRIOR_MOVE_M:
            return False
        positions = self._shuttle_positions
        if len(positions) < 2:
            return False
        span_seconds = (positions[-1][0] - positions[0][0]) / self.fps
        if span_seconds < self.SHUTTLE_LANDED_MIN_SPAN_SECONDS:
            return False
        max_move = max(
            hypot(positions[i][1] - positions[j][1], positions[i][2] - positions[j][2])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        return max_move <= self.SHUTTLE_LANDED_MAX_MOVE_M

    def _update_from_player_settle(self, frame_index, tracks, elapsed_source_frames):
        """Use fresh multi-player court measurements to locate rest windows.

        A stable window is invalidated by any missing/predicted player, a
        roster-count mismatch, a malformed court coordinate, or one measured
        player exceeding ``stable_speed_mps``.  This keeps detector gaps from
        being silently mistaken for a rally ending.
        """
        stable, reason = self._all_players_stable(tracks, elapsed_source_frames)
        if stable:
            if self._settled_start_frame is None:
                self._settled_start_frame = int(frame_index - elapsed_source_frames + 1)
            self._settled_frames += elapsed_source_frames
            if (
                self.state == "active"
                and self._settled_frames / self.fps >= self.settle_window_seconds
            ):
                # The terminal time is the start of the verified stable window,
                # not the later frame at which it merely became long enough.
                self._finish(
                    self._settled_start_frame,
                    reason=f"all_players_stable_for_{self.settle_window_seconds:.1f}s",
                )
            elif self.state in {"idle", "candidate"}:
                self.state = "awaiting_serve"
                self._active_frames = 0
        else:
            self._settled_frames = 0
            self._settled_start_frame = None
            if self.state in {"idle", "awaiting_serve", "candidate"}:
                # A detector gap, a new/recovered ID, or partial roster must
                # not manufacture a new rally.  Only measured motion after a
                # complete roster can leave the stable/awaiting state.
                self._active_frames = (
                    self._active_frames + elapsed_source_frames
                    if reason == "player_motion_above_threshold"
                    else 0
                )
                if self._active_frames >= self.min_active_frames:
                    self._start(
                        frame_index - self._active_frames + 1,
                        start_reason="player_motion_after_settle"
                        if self.state == "awaiting_serve"
                        else "player_motion_video_start",
                    )
            elif self.state == "active":
                self._active_frames = 0
        snapshot = self.snapshot()
        snapshot.update(
            {
                "movement_terminal_signal": {
                    "all_players_stable": bool(stable),
                    "reason": reason,
                    "window_seconds": self.settle_window_seconds,
                    "stable_speed_mps": self.stable_speed_mps,
                }
            }
        )
        return snapshot

    def _all_players_stable(self, tracks, elapsed_source_frames):
        detected = [track for track in tracks if track.get("status") == "detected"]
        if len(detected) != self.expected_player_count:
            self._last_player_positions = {}
            return False, "expected_players_not_all_fresh_detected"
        elapsed_seconds = max(1, int(elapsed_source_frames)) / self.fps
        current = {}
        for track in detected:
            track_id = str(track.get("track_id") or "")
            point = track.get("court_xy_m")
            if not track_id or not isinstance(point, (list, tuple)) or len(point) < 2:
                self._last_player_positions = {}
                return False, "missing_track_or_court_position"
            try:
                current[track_id] = (float(point[0]), float(point[1]))
            except (TypeError, ValueError):
                self._last_player_positions = {}
                return False, "invalid_court_position"
        if set(current) != set(self._last_player_positions):
            self._last_player_positions = current
            return False, "insufficient_contiguous_measurements"
        speeds = [
            hypot(current[track_id][0] - previous[0], current[track_id][1] - previous[1]) / elapsed_seconds
            for track_id, previous in self._last_player_positions.items()
        ]
        self._last_player_positions = current
        if any(speed > self.stable_speed_mps for speed in speeds):
            return False, "player_motion_above_threshold"
        return True, "all_expected_players_stable"

    def finalize(self, frame_index=None):
        if self.state == "active":
            self._finish(
                frame_index,
                reason="video_end_without_movement_terminal"
                if not self.shuttle_enabled
                else "video_end",
            )
        return list(self.completed)

    def snapshot(self):
        return {
            "state": self.state,
            "rally_id": self._current["rally_id"] if self._current else None,
            "score_status": self._current["score"]["status"] if self._current else "unknown",
            "evidence_mode": "shuttle_measurement" if self.shuttle_enabled else "player_stability_only",
        }

    def _start(self, start_frame, start_reason=None):
        self._rally_id += 1
        self.state = "active"
        self._current = {
            "rally_id": self._rally_id,
            "start_frame": int(start_frame),
            "end_frame": None,
            "hit_events": [],
            "start_reason": start_reason or "shuttle_measurement",
            "evidence_mode": "shuttle_measurement" if self.shuttle_enabled else "player_stability_only",
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
        self._current["confidence"] = (
            round(min(0.7, 0.3 + 0.05 * len(self._current["hit_events"])), 3)
            if self.shuttle_enabled
            else 0.35
        )
        self.completed.append(self._current)
        self._current = None
        self.state = "idle" if self.shuttle_enabled else "awaiting_serve"
        self._active_frames = 0
        self._missing_frames = 0
        self._settled_frames = 0
        self._settled_start_frame = None
        self._shuttle_positions = []
        self._shuttle_max_move_m = 0.0


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
        shuttle_enabled=True,
        movement_rally_settle_seconds=0.7,
        court_dimensions=(BADMINTON_COURT_WIDTH, BADMINTON_COURT_LENGTH),
        world_points_m=None,
        coordinate_system_id="standard_badminton_court_m",
    ):
        if match_mode not in {"singles", "doubles"}:
            raise ValueError(
                "FixedCameraMatchPipeline handles singles/doubles rules; "
                "use PersonOnlyTracker/PersonOnlyFrameProcessor for analysis_mode=person_only"
            )
        if tracker_backend not in {"court_association", "bytetrack"}:
            raise ValueError("tracker_backend must be 'court_association' or 'bytetrack'")
        if tracker_backend == "bytetrack" and not enable_bytetrack:
            raise ValueError(
                "ByteTrack is evaluation-gated. Set enable_bytetrack=True only for a recorded tracker evaluation."
            )
        self.court_space = CourtSpace(
            image_corners,
            court_dimensions=tuple(float(value) for value in court_dimensions),
            world_points_m=world_points_m,
        )
        self.coordinate_system_id = str(coordinate_system_id)
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
            require_association_keys=tracker_backend == "bytetrack",
        )
        self.shuttle = MonocularShuttleReconstructor()
        self.rallies = RallyStateMachine(
            fps=fps,
            shuttle_enabled=shuttle_enabled,
            expected_player_count=2 if match_mode == "singles" else 4,
            settle_window_seconds=movement_rally_settle_seconds,
        )
        self._last_frame = 0
        # Contact hysteresis: a single swing keeps the shuttle within the hit
        # radius for many frames, so one approach must emit at most one hit
        # candidate until the shuttle leaves the larger release radius.
        self._hit_contact_frames = {}

    def update(self, frame_index, observations, shuttlecock, has_fresh_observations=True):
        self._last_frame = int(frame_index)
        observations = [dict(item) for item in observations]
        if self.byte_tracker is not None:
            association_keys = self.byte_tracker.update(observations)
            for index, association_key in association_keys.items():
                observations[index]["association_key"] = association_key
        tracks = self.tracker.update(
            frame_index,
            observations,
            has_fresh_observations=has_fresh_observations,
        )
        shuttle = self._shuttle_record(frame_index, shuttlecock)
        hit_events = self._detect_hit_events(tracks, shuttle, frame_index)
        rally = self.rallies.update(frame_index, tracks, shuttle, hit_events)
        return {
            "schema_version": SCHEMA_VERSION,
            "coordinate_system": self.coordinate_system_id,
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
            "rally_policy": {
                "evidence_mode": "shuttle_measurement" if self.rallies.shuttle_enabled else "player_stability_only",
                "movement_terminal_rule": (
                    None if self.rallies.shuttle_enabled else {
                        "all_expected_players_must_be_fresh_detected": True,
                        "settle_window_seconds": self.rallies.settle_window_seconds,
                        "stable_speed_mps": self.rallies.stable_speed_mps,
                        "score_policy": "movement-only boundaries never produce score or hit evidence",
                    }
                ),
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

    HIT_CONTACT_DISTANCE_M = 1.35
    HIT_RELEASE_DISTANCE_M = 1.75
    HIT_CONTACT_STALE_FRAMES = 10 * 60

    def _detect_hit_events(self, tracks, shuttle, frame_index):
        """Emit one candidate per shuttle-approach, not one per sampled frame.

        A fixed camera sees the shuttle inside the hit radius for several
        consecutive samples during one swing.  Contact hysteresis keeps a single
        approach as a single candidate: once a track enters the contact radius it
        is not re-reported until the shuttle leaves the wider release radius.
        """
        frame_index = int(frame_index)
        self._hit_contact_frames = {
            track_id: last_frame
            for track_id, last_frame in self._hit_contact_frames.items()
            if frame_index - int(last_frame) <= self.HIT_CONTACT_STALE_FRAMES
        }
        if shuttle.get("status") != "approximate" or not tracks:
            return []
        shuttle_xy = shuttle["xyz_m"][:2]
        nearest = min(tracks, key=lambda track: hypot(track["court_xy_m"][0] - shuttle_xy[0], track["court_xy_m"][1] - shuttle_xy[1]))
        distance = hypot(nearest["court_xy_m"][0] - shuttle_xy[0], nearest["court_xy_m"][1] - shuttle_xy[1])
        if nearest["status"] != "detected":
            return []
        track_id = nearest["track_id"]
        if distance > self.HIT_RELEASE_DISTANCE_M:
            self._hit_contact_frames.pop(track_id, None)
            return []
        if track_id in self._hit_contact_frames or distance > self.HIT_CONTACT_DISTANCE_M:
            return []
        self._hit_contact_frames[track_id] = frame_index
        return [{
            "status": "candidate",
            "hitter_track_id": track_id,
            "confidence": round(min(0.45, shuttle["confidence"] * nearest["confidence"]), 4),
            "reason": "spatial_proximity_only; not score evidence",
        }]
