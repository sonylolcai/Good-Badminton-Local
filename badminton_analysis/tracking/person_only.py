"""Anonymous, open-set person tracking policy for production video analysis.

The visual service does not know whether a session is singles, doubles, 1v2 or
1v3.  This module therefore treats every sustainable on-court trajectory as an
anonymous candidate.  It never emits user, team, score or winner fields; those
relations belong to the business service after the match.

Detection and identity evidence remain separate.  A short court-space recovery
may retain a track ID for continuity, but is explicitly labelled
``reassociation_uncertain`` unless a confirmed ByteTrack association key proves
the recovery.  Downstream analytics can then reject uncertain fragments instead
of silently merging them into a person's ability report.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping, Optional

from ..analysis.fixed_camera_match import CourtMultiObjectTracker, CourtSpace
from ..streaming.models import FinalizationContext, FrameContext, ProcessorEvent
from .bytetrack_adapter import ByteTrackAdapter


PERSON_ONLY_SCHEMA_VERSION = "person-only.v1"
PERSON_ONLY_STATES = {
    "candidate",
    "active",
    "predicted",
    "missing",
    "closed",
    "reassociation_uncertain",
}
_FORBIDDEN_IDENTITY_FIELDS = {"person_id", "team_id"}


class PersonOnlyTracker:
    """Maintain an anonymous, dynamically sized set of court-space tracks."""

    def __init__(
        self,
        image_corners,
        *,
        fps: float,
        tracker_backend: str = "court_association",
        enable_bytetrack: bool = False,
        byte_tracker_factory=None,
        min_confirmed_detections: int = 3,
        reliable_coverage_threshold: float = 0.60,
        max_missed_frames: int = 12,
        max_retained_missing_frames: Optional[int] = None,
        max_speed_mps: float = 10.0,
        lock_match_roster: bool = False,
        expected_roster_count: Optional[int] = None,
        roster_stable_frames: int = 3,
        max_roster_count: int = 4,
        roster_discovery_seconds: float = 8.0,
        roster_reacquire_seconds: float = 1.0,
    ) -> None:
        if tracker_backend not in {"court_association", "bytetrack"}:
            raise ValueError("tracker_backend must be court_association or bytetrack")
        if tracker_backend == "bytetrack" and not enable_bytetrack:
            raise ValueError("ByteTrack must be explicitly enabled after its evaluation gate")
        if int(min_confirmed_detections) < 1:
            raise ValueError("min_confirmed_detections must be positive")
        if not 0 < float(reliable_coverage_threshold) <= 1:
            raise ValueError("reliable_coverage_threshold must be in (0, 1]")

        self.court_space = CourtSpace(image_corners)
        self.image_corners = [
            [float(point[0]), float(point[1])] for point in image_corners
        ]
        self.fps = float(fps)
        self.tracker_backend = tracker_backend
        self.lock_match_roster = bool(lock_match_roster)
        self.expected_roster_count = (
            None if expected_roster_count is None else int(expected_roster_count)
        )
        self.roster_stable_frames = max(1, int(roster_stable_frames))
        self.max_roster_count = max(2, int(max_roster_count))
        self.roster_discovery_seconds = max(0.0, float(roster_discovery_seconds))
        self.roster_reacquire_seconds = float(roster_reacquire_seconds)
        self.min_confirmed_detections = int(min_confirmed_detections)
        self.reliable_coverage_threshold = float(reliable_coverage_threshold)
        self._byte_tracker = (
            ByteTrackAdapter(fps=fps, tracker_factory=byte_tracker_factory)
            if tracker_backend == "bytetrack"
            else None
        )
        self._tracker = CourtMultiObjectTracker(
            self.court_space,
            fps=fps,
            max_missed_frames=max_missed_frames,
            max_retained_missing_frames=max_retained_missing_frames,
            max_speed_mps=max_speed_mps,
            match_mode="person_only",
            lock_match_roster=self.lock_match_roster,
            expected_roster_count=self.expected_roster_count,
            roster_stable_frames=self.roster_stable_frames,
            max_roster_count=self.max_roster_count,
            roster_discovery_seconds=self.roster_discovery_seconds,
            roster_reacquire_seconds=self.roster_reacquire_seconds,
            require_association_keys=(
                tracker_backend == "bytetrack" and self.lock_match_roster
            ),
        )
        self._registry: dict[str, dict] = {}
        self._last_frame_index = -1
        self._last_source_time_sec = 0.0

    def update(
        self,
        frame_index: int,
        observations: Iterable[Mapping[str, Any]],
        *,
        source_time_sec: Optional[float] = None,
        has_fresh_observations: bool = True,
    ) -> dict:
        """Update from one actual pose measurement.

        A caller that skipped this source frame should set
        ``has_fresh_observations=False``.  Such a call returns the last public
        state without counting an artificial miss or denominator sample.
        """
        frame_index = int(frame_index)
        source_time_sec = (
            float(source_time_sec)
            if source_time_sec is not None
            else max(0.0, frame_index / self.fps)
        )
        if frame_index < self._last_frame_index:
            raise ValueError("person_only frame index cannot move backwards")
        if source_time_sec + 1e-9 < self._last_source_time_sec:
            raise ValueError("person_only source time cannot move backwards")

        if not has_fresh_observations:
            return self._public_snapshot(
                self._tracker.snapshot(self._last_frame_index),
                update_quality=False,
                source_time_sec=self._last_source_time_sec,
            )

        normalized = [dict(item) for item in (observations if observations is not None else ())]
        if self._byte_tracker is not None:
            association_keys = self._byte_tracker.update(normalized)
            for index, association_key in association_keys.items():
                normalized[index]["association_key"] = association_key

        tracks = self._tracker.update(
            frame_index,
            normalized,
            has_fresh_observations=True,
        )
        self._last_frame_index = frame_index
        self._last_source_time_sec = source_time_sec
        snapshot = self._public_snapshot(
            tracks,
            update_quality=True,
            source_time_sec=source_time_sec,
        )
        snapshot["roster_review_candidates"] = self._roster_review_candidates(
            normalized,
            snapshot,
        )
        return snapshot

    def _roster_review_candidates(self, observations, snapshot):
        """Expose real pre-lock poses without treating them as players.

        A direct four-person stream may correctly refuse an unstable roster.
        Returning an empty result hid the evidence needed to diagnose that
        decision.  These records deliberately use a separate event type and
        are excluded from movement analytics and person identity claims.
        """
        roster = (snapshot.get("tracking") or {}).get("roster") or {}
        if (
            not self.lock_match_roster
            or roster.get("status") == "locked"
            or not roster.get("expected_player_count")
        ):
            return []
        output = []
        for index, observation in enumerate(observations):
            court_xy = observation.get("court_xy")
            if court_xy is None or not self.court_space.contains(court_xy, margin_m=0.35):
                continue
            association_key = str(observation.get("association_key") or "")
            safe_key = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in association_key
            )
            candidate_id = (
                f"candidate_{safe_key}"
                if safe_key
                else f"candidate_frame_{self._last_frame_index}_{index + 1}"
            )
            keypoints = deepcopy(observation.get("keypoints_image"))
            keypoint_scores = deepcopy(observation.get("keypoint_scores"))
            output.append(
                {
                    "track_id": candidate_id,
                    "status": "detected",
                    "confidence": max(0.0, min(1.0, float(observation.get("confidence") or 0.0))),
                    "court_xy_m": [float(court_xy[0]), float(court_xy[1])],
                    "image_xy": deepcopy(observation.get("image_xy")),
                    "lifecycle_state": "unconfirmed_roster",
                    "identity_status": "unconfirmed_roster",
                    "analytics_eligible": False,
                    "association": {
                        "key": association_key or None,
                        "source": "roster_review_candidate",
                        "identity_confidence": 0.0,
                    },
                    "location_evidence": {
                        "method": observation.get("location_method"),
                        "confidence": observation.get("location_confidence"),
                        "source": observation.get("source"),
                        "bbox_xyxy": deepcopy(observation.get("bbox_xyxy")),
                        "hands_image": deepcopy(observation.get("hands_image")),
                    },
                    "pose": {
                        "format": "coco17_image_v1",
                        "coordinate_system": "full_source_image_pixels",
                        "is_current_measurement": True,
                        "keypoints_image": keypoints,
                        "keypoint_scores": keypoint_scores,
                    },
                    "review_reason": "expected_roster_not_yet_stable",
                }
            )
        return output

    def finalize(self) -> dict:
        """Close all remaining anonymous candidates and return audit metrics."""
        for record in self._registry.values():
            if record["lifecycle_state"] != "closed":
                record["lifecycle_state"] = "closed"
                record["closed_frame"] = self._last_frame_index
                record["closed_source_time_sec"] = self._last_source_time_sec
        return {
            "schema_version": PERSON_ONLY_SCHEMA_VERSION,
            "analysis_mode": "person_only",
            "track_candidates": self.track_candidates(),
            "track_profiles": self.track_profiles(),
            "summary": self.summary(),
        }

    def track_candidates(self) -> list[dict]:
        """Return the exact frozen stream-session.v1 ``trackCandidate`` shape."""
        candidates = []
        for track_id, record in sorted(self._registry.items()):
            quality = self._quality(record)
            representative_confidence = max(
                0.0,
                float(record.get("representative_confidence", 0.0)),
            )
            identity_confidence = min(
                representative_confidence,
                quality["coverage_rate"],
            ) * (0.7 ** quality["reassociation_uncertain_count"])
            candidates.append(
                {
                    "track_id": track_id,
                    "state": record["lifecycle_state"],
                    "first_source_time_sec": record["first_seen_source_time_sec"],
                    "last_source_time_sec": record.get(
                        "last_seen_source_time_sec",
                        record["first_seen_source_time_sec"],
                    ),
                    "detected_coverage": quality["coverage_rate"],
                    "confidence": round(max(0.0, min(1.0, identity_confidence)), 4),
                }
            )
        return candidates

    def track_profiles(self) -> list[dict]:
        """Return rich visual analytics kept outside the frozen status shape."""
        candidates = []
        movement = {
            item["track_id"]: item for item in self._tracker.summaries()
        }
        for track_id, record in sorted(self._registry.items()):
            movement_record = movement.get(track_id, {})
            candidates.append(
                {
                    "track_id": track_id,
                    "lifecycle_state": record["lifecycle_state"],
                    "first_seen_frame": record["first_seen_frame"],
                    "last_seen_frame": record.get("last_seen_frame"),
                    "first_seen_source_time_sec": record["first_seen_source_time_sec"],
                    "last_seen_source_time_sec": record.get("last_seen_source_time_sec"),
                    "representative_observation": deepcopy(
                        record.get("representative_observation")
                    ),
                    "quality": self._quality(record),
                    "movement": {
                        "distance_m": movement_record.get("distance_m", 0.0),
                        "zone_frames": deepcopy(movement_record.get("zone_frames", {})),
                        "analytics_scope": "movement_and_space_only",
                    },
                    "trajectory_summary": {
                        "first_court_xy_m": deepcopy(record.get("first_court_xy_m")),
                        "last_court_xy_m": deepcopy(record.get("last_court_xy_m")),
                        "detected_sample_count": int(record.get("detected_frames", 0)),
                    },
                }
            )
        return candidates

    def summary(self) -> dict:
        states = {state: 0 for state in sorted(PERSON_ONLY_STATES)}
        reliable_count = 0
        for record in self._registry.values():
            states[record["lifecycle_state"]] += 1
            reliable_count += int(self._quality(record)["reliable_for_person_analytics"])
        interruption_count = sum(
            int(record.get("interruption_count", 0))
            for record in self._registry.values()
        )
        uncertain_count = sum(
            int(record.get("reassociation_uncertain_count", 0))
            for record in self._registry.values()
        )
        return {
            "anonymous_track_count": len(self._registry),
            "reliable_track_count": reliable_count,
            "lifecycle_counts": states,
            "identity_policy": (
                "track_id is anonymous; uncertain recovery remains explicit; "
                "zone_id is never identity"
            ),
            "analytics_scope": "movement_and_space_only",
            "business_fields_present": False,
            "identity_quality": {
                "track_interruption_count": interruption_count,
                "reassociation_uncertain_count": uncertain_count,
                "id_switch_count": None,
                "id_switch_status": "requires_ground_truth_or_manual_review",
            },
        }

    def snapshot_state(self) -> dict:
        """Return checkpoint state for task A's AnalysisEngine.

        Ultralytics ByteTrack does not expose a stable public checkpoint API.
        A running process can still persist the court-space/public-track state
        after every segment. The checkpoint explicitly records that ByteTrack
        itself is not restorable; a process restart then fails closed in
        ``restore_state`` instead of making the first segment fail.
        """
        return {
            "state_version": PERSON_ONLY_SCHEMA_VERSION,
            "tracker_backend": self.tracker_backend,
            "lock_match_roster": self.lock_match_roster,
            "expected_roster_count": self._tracker.expected_roster_count,
            "roster_stable_frames": self.roster_stable_frames,
            "max_roster_count": self.max_roster_count,
            "roster_discovery_seconds": self.roster_discovery_seconds,
            "roster_reacquire_seconds": self.roster_reacquire_seconds,
            "association_runtime_checkpointable": self._byte_tracker is None,
            "min_confirmed_detections": self.min_confirmed_detections,
            "reliable_coverage_threshold": self.reliable_coverage_threshold,
            "last_frame_index": self._last_frame_index,
            "last_source_time_sec": self._last_source_time_sec,
            "court_tracker": self._tracker.snapshot_state(),
            "registry": deepcopy(self._registry),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("state_version") != PERSON_ONLY_SCHEMA_VERSION:
            raise ValueError("unsupported person_only checkpoint")
        if state.get("tracker_backend") != self.tracker_backend:
            raise ValueError("person_only checkpoint tracker backend does not match")
        if bool(state.get("lock_match_roster", False)) != self.lock_match_roster:
            raise ValueError("person_only roster lock setting does not match")
        if (
            self.expected_roster_count is not None
            and state.get("expected_roster_count") != self.expected_roster_count
        ):
            raise ValueError("person_only expected roster count does not match")
        if int(state.get("roster_stable_frames", -1)) != self.roster_stable_frames:
            raise ValueError("person_only roster stability setting does not match")
        if int(state.get("max_roster_count", self.max_roster_count)) != self.max_roster_count:
            raise ValueError("person_only maximum roster setting does not match")
        if abs(
            float(state.get("roster_discovery_seconds", self.roster_discovery_seconds))
            - self.roster_discovery_seconds
        ) > 1e-9:
            raise ValueError("person_only roster discovery setting does not match")
        if abs(
            float(state.get("roster_reacquire_seconds", -1.0))
            - self.roster_reacquire_seconds
        ) > 1e-9:
            raise ValueError("person_only roster recovery setting does not match")
        if int(state.get("min_confirmed_detections", -1)) != self.min_confirmed_detections:
            raise ValueError("person_only confirmation threshold does not match")
        if abs(
            float(state.get("reliable_coverage_threshold", -1.0))
            - self.reliable_coverage_threshold
        ) > 1e-9:
            raise ValueError("person_only coverage threshold does not match")
        if self._byte_tracker is not None:
            raise RuntimeError(
                "ByteTrack runtime state cannot be restored without rebuilding identity evidence"
            )
        self._tracker.restore_state(dict(state.get("court_tracker") or {}))
        self.expected_roster_count = self._tracker.expected_roster_count
        self._registry = deepcopy(dict(state.get("registry") or {}))
        self._last_frame_index = int(state.get("last_frame_index", -1))
        self._last_source_time_sec = float(state.get("last_source_time_sec", 0.0))

    def _public_snapshot(self, tracks, *, update_quality: bool, source_time_sec: float) -> dict:
        current_ids = {str(track["track_id"]) for track in tracks}
        if update_quality:
            for track_id, record in self._registry.items():
                if track_id not in current_ids and record["lifecycle_state"] != "closed":
                    record["lifecycle_state"] = "closed"
                    record["closed_frame"] = self._last_frame_index
                    record["closed_source_time_sec"] = source_time_sec

        public_tracks = []
        for track in tracks:
            track = {
                key: deepcopy(value)
                for key, value in track.items()
                if key not in _FORBIDDEN_IDENTITY_FIELDS
            }
            track_id = str(track["track_id"])
            record = self._registry.get(track_id)
            if record is None:
                record = self._new_registry_record(track, source_time_sec)
                self._registry[track_id] = record
            if update_quality:
                lifecycle = self._advance_registry(record, track, source_time_sec)
            else:
                lifecycle = record["lifecycle_state"]
            track["lifecycle_state"] = lifecycle
            track["quality"] = self._quality(record)
            public_tracks.append(track)

        return {
            "schema_version": PERSON_ONLY_SCHEMA_VERSION,
            "analysis_mode": "person_only",
            "coordinate_system": "standard_badminton_court_m",
            "tracking": {
                "backend": self.tracker_backend,
                "open_set": not self.lock_match_roster,
                "expected_player_count": self._tracker.roster_summary().get(
                    "expected_player_count"
                ),
                "roster": self._tracker.roster_summary(),
                "measurement_statuses": ["detected", "predicted", "missing"],
                "lifecycle_states": sorted(PERSON_ONLY_STATES),
                "identity_policy": "no forced user, team, side, singles, or doubles identity",
            },
            "tracks": public_tracks,
            "track_candidates": self.track_candidates(),
            "track_profiles": self.track_profiles(),
        }

    def _new_registry_record(self, track, source_time_sec):
        return {
            "track_id": str(track["track_id"]),
            "lifecycle_state": "candidate",
            "first_seen_frame": self._last_frame_index,
            "last_seen_frame": self._last_frame_index,
            "first_seen_source_time_sec": source_time_sec,
            "last_seen_source_time_sec": source_time_sec,
            "opportunity_frames": 0,
            "detected_frames": 0,
            "predicted_frames": 0,
            "missing_frames": 0,
            "reassociation_uncertain_count": 0,
            "interruption_count": 0,
            "last_observation_status": None,
            "representative_observation": None,
            "representative_confidence": -1.0,
            "first_court_xy_m": deepcopy(track.get("court_xy_m")),
            "last_court_xy_m": deepcopy(track.get("court_xy_m")),
        }

    def _advance_registry(self, record, track, source_time_sec):
        status = str(track.get("status") or "missing")
        previous_status = record.get("last_observation_status")
        record["opportunity_frames"] += 1

        if status == "detected":
            record["detected_frames"] += 1
            record["last_seen_frame"] = self._last_frame_index
            record["last_seen_source_time_sec"] = source_time_sec
            record["last_court_xy_m"] = deepcopy(track.get("court_xy_m"))
            association_source = str((track.get("association") or {}).get("source") or "")
            if previous_status in {"predicted", "missing"}:
                record["interruption_count"] += 1
                if association_source != "bytetrack":
                    record["reassociation_uncertain_count"] += 1
                    lifecycle = "reassociation_uncertain"
                else:
                    lifecycle = "active"
            else:
                lifecycle = (
                    "active"
                    if record["detected_frames"] >= self.min_confirmed_detections
                    else "candidate"
                )
            confidence = float(track.get("confidence") or 0.0)
            if confidence > record["representative_confidence"]:
                record["representative_confidence"] = confidence
                record["representative_observation"] = {
                    "frame_index": self._last_frame_index,
                    "source_time_sec": source_time_sec,
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": deepcopy(
                        (track.get("location_evidence") or {}).get("bbox_xyxy")
                    ),
                    "court_xy_m": deepcopy(track.get("court_xy_m")),
                }
        elif status == "predicted":
            record["predicted_frames"] += 1
            lifecycle = "predicted"
        else:
            record["missing_frames"] += 1
            lifecycle = "missing"

        record["last_observation_status"] = status
        record["lifecycle_state"] = lifecycle
        return lifecycle

    def _quality(self, record):
        opportunities = max(1, int(record.get("opportunity_frames", 0)))
        detected = int(record.get("detected_frames", 0))
        predicted = int(record.get("predicted_frames", 0))
        missing = int(record.get("missing_frames", 0))
        uncertain = int(record.get("reassociation_uncertain_count", 0))
        coverage = detected / opportunities
        reliable = (
            detected >= self.min_confirmed_detections
            and coverage >= self.reliable_coverage_threshold
            and uncertain == 0
            and record.get("lifecycle_state") not in {"candidate", "reassociation_uncertain"}
        )
        return {
            "opportunity_frames": opportunities,
            "detected_frames": detected,
            "predicted_frames": predicted,
            "missing_frames": missing,
            "coverage_rate": round(coverage, 4),
            "detected_ratio": round(detected / opportunities, 4),
            "predicted_ratio": round(predicted / opportunities, 4),
            "missing_ratio": round(missing / opportunities, 4),
            "interruption_count": int(record.get("interruption_count", 0)),
            "reassociation_uncertain_count": uncertain,
            "fragment_count": 1 + uncertain,
            "reliable_for_person_analytics": reliable,
        }


class PersonOnlyFrameProcessor:
    """Task A adapter: pose observations in, anonymous ProcessorEvents out."""

    def __init__(
        self,
        tracker: PersonOnlyTracker,
        observation_provider: Callable[[Any, FrameContext], Any],
    ) -> None:
        if not callable(observation_provider):
            raise ValueError("observation_provider must be callable")
        self.tracker = tracker
        self.observation_provider = observation_provider

    def process_frame(self, frame: Any, context: FrameContext) -> Iterable[ProcessorEvent]:
        provided = self.observation_provider(frame, context)
        has_fresh_observations = True
        if isinstance(provided, Mapping):
            observations = provided.get("observations") or []
            has_fresh_observations = bool(provided.get("has_fresh_observations", True))
        else:
            observations = provided or []
        snapshot = self.tracker.update(
            # The tracker cadence is the configured 10/15/30 Hz measurement
            # clock, not the source video's 30/60 fps clock. Using source frame
            # numbers here inflated missed-frame gaps and speed calculations.
            context.measurement_bucket,
            observations,
            source_time_sec=context.source_time_sec,
            has_fresh_observations=has_fresh_observations,
        )
        events = []
        for raw_candidate in snapshot.get("roster_review_candidates") or []:
            candidate = deepcopy(raw_candidate)
            candidate["source_frame_index"] = int(context.source_frame_index)
            candidate["measurement_bucket"] = int(context.measurement_bucket)
            events.append(
                ProcessorEvent(
                    event_type="roster_candidate_observation",
                    evidence_state="detected",
                    confidence=float(candidate.get("confidence") or 0.0),
                    data={
                        "analysis_mode": "person_only",
                        "track": candidate,
                        "tracking": {
                            "backend": snapshot["tracking"]["backend"],
                            "roster": deepcopy(snapshot["tracking"]["roster"]),
                        },
                    },
                )
            )
        for raw_track in snapshot["tracks"]:
            track = deepcopy(raw_track)
            track["source_frame_index"] = int(context.source_frame_index)
            track["measurement_bucket"] = int(context.measurement_bucket)
            if isinstance(track.get("location_evidence"), dict):
                track["location_evidence"]["source_frame_index"] = int(
                    context.source_frame_index
                )
                track["location_evidence"]["measurement_bucket"] = int(
                    context.measurement_bucket
                )
            # Evidence describes the current observation; lifecycle describes
            # identity maturity. A real first detection is still detected
            # evidence while its anonymous identity remains a candidate.
            observation_status = str(track.get("status") or "missing")
            evidence_state = {
                "predicted": "predicted",
                "missing": "missing",
            }.get(observation_status, "detected")
            events.append(
                ProcessorEvent(
                    event_type="person_observation",
                    evidence_state=evidence_state,
                    confidence=float(track.get("confidence") or 0.0),
                    data={
                        "analysis_mode": "person_only",
                        "track": deepcopy(track),
                        "tracking": {
                            "backend": snapshot["tracking"]["backend"],
                            "roster": deepcopy(snapshot["tracking"]["roster"]),
                        },
                    },
                )
            )
        return events

    def finalize(self, context: FinalizationContext) -> Iterable[ProcessorEvent]:
        finalized = self.tracker.finalize()
        return [
            ProcessorEvent(
                event_type="session_status",
                evidence_state="derived",
                confidence=1.0,
                source_time_sec=context.last_source_time_sec,
                segment_index=max(0, context.last_segment_index),
                data=finalized,
            )
        ]

    def snapshot_state(self) -> Mapping[str, Any]:
        return self.tracker.snapshot_state()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.tracker.restore_state(state)
