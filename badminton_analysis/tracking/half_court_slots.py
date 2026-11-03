"""Opt-in match slots, deliberately NOT persistent physical-person identity.

Players may leave, be replaced, or change equipment without phone interaction.
After a balanced opening, each physical half owns a fixed number of slots.
ByteTrack continuity is preferred; remaining observations fill that half's free
slots by minimum total displacement. A replacement starts a new measured piece.
"""
from itertools import permutations
from collections import deque

from ..analysis.fixed_camera_match import CourtMultiObjectTracker


HALF_COURT_SLOTS = "half_court_slots.v1"
INDIVIDUAL_TRACKS = "individual_tracks.v1"


class HalfCourtSlotTracker(CourtMultiObjectTracker):
    @staticmethod
    def _identity_confidence_for_source(source):
        # Confidence here concerns assignment to an occupied SLOT, never a
        # claim that the physical person before/after a gap is the same.
        if str(source).startswith("half_court_slot_"):
            return .95
        return CourtMultiObjectTracker._identity_confidence_for_source(source)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (not self.lock_match_roster or self.expected_roster_count not in {2, 4}
                or self.person_association_policy.get("version") != "person_association.v2"):
            raise ValueError("Half-court slots require a locked badminton roster of two or four")
        self.slot_ends = {}
        self.occupancy_segments = {}
        self._ready_samples = deque()
        self._last_balanced_frame = None

    def _end(self, observation):
        return "end_a" if observation["court_xy"][1] < self.court_space.length_m / 2 else "end_b"

    def _qualified(self, observations):
        # Side spectators and low-confidence duplicate boxes cannot fill a slot.
        qualified = []
        seen_keys = set()
        for row in sorted(observations, key=lambda r: -float(r.get("confidence", 0))):
            xy = row["court_xy"]
            key = row.get("association_key")
            if (not self.court_space.contains(xy)
                    or float(row.get("confidence", 0)) < self.person_association_policy["recovery_body_min"]
                    or float(row.get("location_confidence", 0)) < self.person_association_policy["recovery_location_min"]
                    or (key and key in seen_keys)
                    or any(self._distance(xy, other["court_xy"]) < .25 for other in qualified)):
                continue
            qualified.append(row)
            if key:
                seen_keys.add(key)
        return qualified

    def _bootstrap_roster(self, frame_index, observations):
        rows = self._qualified(observations)
        self._last_roster_candidate_count = len(rows)
        per_half = self.expected_roster_count // 2
        counts = {end: sum(self._end(r) == end for r in rows) for end in ("end_a", "end_b")}
        balanced = all(count == per_half for count in counts.values())
        grace_frames = max(1, round(min(.75, self.roster_discovery_seconds * .25) * self.fps))
        if (self._ready_samples and frame_index - self._ready_samples[-1][0] > grace_frames):
            self._ready_samples.clear()
        if balanced:
            self._last_balanced_frame = frame_index
        elif self._last_balanced_frame is None or frame_index - self._last_balanced_frame > grace_frames:
            self._ready_samples.clear()
        if balanced or self._ready_samples:
            self._ready_samples.append((int(frame_index), balanced))
        while self._ready_samples and frame_index - self._ready_samples[0][0] > self.roster_discovery_frames:
            self._ready_samples.popleft()
        self._roster_discovery_started_frame = self._ready_samples[0][0] if self._ready_samples else None
        if not balanced:
            self._roster_stable_observation_frames = 0
            self._last_roster_reason = "waiting_for_balanced_half_court_slots"
            return self.snapshot(frame_index)
        self._roster_stable_observation_frames += 1
        if (self._roster_stable_observation_frames < self.roster_stable_frames
                or frame_index - self._roster_discovery_started_frame < self.roster_discovery_frames
                or sum(valid for _, valid in self._ready_samples) / len(self._ready_samples) < .8):
            self._last_roster_reason = "awaiting_balanced_half_court_readiness_window"
            return self.snapshot(frame_index)
        for row in sorted(rows, key=lambda r: (self._end(r), r["court_xy"][0])):
            self._create_track(frame_index, row, association_source="half_court_slot_opening")
            track_id = f"track_{self._next_id - 1:03d}"
            self.slot_ends[track_id] = self._end(row)
            self.occupancy_segments[track_id] = 1
        self.roster_track_ids = sorted(self.tracks)
        self.roster_status = "locked"
        self.roster_locked_frame = int(frame_index)
        self._last_roster_reason = "balanced_half_court_slots_ready"
        return self.snapshot(frame_index)

    def _update_locked_or_open_tracks(self, frame_index, observations):
        rows = self._qualified(observations)
        self._last_roster_candidate_count = len(rows)
        assignments = {}
        used = set()
        # Known visual keys only keep a slot while still on its physical half.
        for index, row in enumerate(rows):
            owner = self._association_keys.get(row.get("association_key"))
            if owner in self.tracks and owner not in assignments and self.slot_ends[owner] == self._end(row):
                assignments[owner] = index
                used.add(index)
        for end in ("end_a", "end_b"):
            slots = sorted(t for t in self.tracks if t not in assignments and self.slot_ends[t] == end)
            candidates = [i for i, row in enumerate(rows) if i not in used and self._end(row) == end]
            # At most two slots per half. Enumerate the small bipartite match,
            # allowing extra detections without creating new permanent IDs.
            count = min(len(slots), len(candidates))
            if not count:
                continue
            choices = (
                (sum(self._distance(self.tracks[t].court_xy, rows[i]["court_xy"]) for t, i in zip(ts, indices)), ts, indices)
                for ts in permutations(slots, count)
                for indices in permutations(candidates, count)
            )
            _, selected_slots, selected_indices = min(choices)
            assignments.update(zip(selected_slots, selected_indices))
            used.update(selected_indices)
        for track_id, track in self.tracks.items():
            if track_id not in assignments:
                track.missed_frames += max(1, frame_index - track.last_frame)
                track.last_frame = int(frame_index)
                self.track_metrics[track_id]["missing_frames"] += 1
                continue
            row = rows[assignments[track_id]]
            elapsed = max(1, frame_index - track.last_observation_frame) / self.fps
            replacement = (track.missed_frames > 0 or (row.get("association_key") and row["association_key"] != track.association_key)
                           or self._distance(track.court_xy, row["court_xy"]) / elapsed > self.max_speed_mps)
            source = "half_court_slot_replacement" if replacement else "half_court_slot_continuity"
            if replacement:
                self.occupancy_segments[track_id] += 1
                track.court_observation_history = []
                track.velocity_mps = (0., 0.)
                track.image_history = []
                track.court_history = []
                if not row.get("association_key"):
                    if self._association_keys.get(track.association_key) == track_id:
                        self._association_keys.pop(track.association_key, None)
                    track.association_key = None
            previous_distance = self.track_metrics[track_id]["distance_m"]
            self._apply_observation(track, frame_index, row, association_source=source)
            if replacement:
                self.track_metrics[track_id]["distance_m"] = previous_distance
                self.track_metrics[track_id]["recovered_detected_frames"] += 1
        self._last_unassigned_observation_count = len(rows) - len(used)
        self._last_unassigned_observations = [self._unassigned_observation_evidence(row) for i, row in enumerate(rows) if i not in used]
        self._last_roster_reason = "half_court_slots_recording" if len(assignments) == len(self.tracks) else "half_court_slots_waiting_for_return"
        return self.snapshot(frame_index)

    def snapshot(self, frame_index):
        rows = super().snapshot(frame_index)
        for row in rows:
            row["identity_scope"] = HALF_COURT_SLOTS
            row["slot_court_end"] = self.slot_ends.get(row["track_id"])
            row["occupancy_segment"] = self.occupancy_segments.get(row["track_id"], 1)
        return rows

    def roster_summary(self):
        result = super().roster_summary()
        result.update(identity_scope=HALF_COURT_SLOTS, policy="Physical half-court slots; later occupants inherit earlier slots automatically, without physical-person verification.")
        return result

    def snapshot_state(self):
        result = super().snapshot_state()
        result["half_court_slots"] = {"version": HALF_COURT_SLOTS, "slot_ends": dict(self.slot_ends), "occupancy_segments": dict(self.occupancy_segments),
                                     "ready_samples": list(self._ready_samples), "last_balanced_frame": self._last_balanced_frame}
        return result

    def restore_state(self, state):
        slots = state.get("half_court_slots") or {}
        if slots.get("version") != HALF_COURT_SLOTS:
            raise ValueError("Half-court slot checkpoint policy mismatch")
        if (set(slots.get("slot_ends", {})) != set(state.get("tracks", {}))
                or set(slots.get("occupancy_segments", {})) != set(state.get("tracks", {}))):
            raise ValueError("Incomplete half-court slot checkpoint")
        super().restore_state(state)
        self.slot_ends = dict(slots["slot_ends"])
        self.occupancy_segments = dict(slots["occupancy_segments"])
        self._ready_samples = deque((int(frame), bool(valid)) for frame, valid in slots.get("ready_samples", []))
        self._last_balanced_frame = slots.get("last_balanced_frame")
