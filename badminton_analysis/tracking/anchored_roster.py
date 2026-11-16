"""A same-frame anonymous roster with conservative, fixed clothing references.

These small colour histograms are visual evidence, not person recognition.
Similar clothes, overlap, a long absence, or an impossible motion remain
unassigned. In particular there is no nearest-empty-slot recovery rule.
"""
from copy import deepcopy
from math import isfinite, sqrt

import cv2
import numpy as np

from .half_court_slots import HalfCourtSlotTracker, INDIVIDUAL_TRACKS


ANCHORED_ROSTER = "anchored_roster.v1"
# Conservative initial operating points, not measured identity-accuracy claims.
MAX_APPEARANCE_DISTANCE = .35
APPEARANCE_MARGIN = .12
CONTINUITY_APPEARANCE_VETO = .65
MAX_GALLERY_SAMPLES = 8


def overlapping_boxes(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller <= 0 or intersection / smaller > .35


def full_body_supported(row, box):
    """Leg-only ROI detections cannot supply a person's clothing anchor."""
    points, scores = row.get("keypoints_image"), row.get("keypoint_scores")
    if points is None or scores is None or len(points) < 17 or len(scores) < 17:
        return False
    x1, y1, x2, y2 = box
    def visible(i):
        point = points[i]
        return (point is not None and len(point) >= 2 and all(isfinite(v) for v in point[:2])
                and scores[i] is not None and isfinite(scores[i]) and scores[i] >= .25
                and x1 <= point[0] <= x2 and y1 <= point[1] <= y2)
    if not all(visible(i) for i in (5, 6, 11, 12)) or not any(visible(i) for i in (15, 16)):
        return False
    shoulders = (points[5][1] + points[6][1]) / 2.
    hips = (points[11][1] + points[12][1]) / 2.
    ankles = max(points[i][1] for i in (15, 16) if visible(i))
    return hips - shoulders >= .10 * (y2-y1) and ankles - hips >= .15 * (y2-y1)


def prepare_anchor_observations(frame, observations, context):
    """Derive bounded clothing evidence from each current source-frame box."""
    rows = [dict(row) for row in observations]
    for row in rows:
        row.pop("anchor_appearance", None)
        row.pop("anchor_frame", None)
        row.pop("anchor_clear", None)
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        return rows
    height, width = image.shape[:2]
    boxes = {}
    for index, row in enumerate(rows):
        try:
            x1, y1, x2, y2 = map(float, row.get("bbox_xyxy", []))
        except (ValueError, TypeError):
            continue
        if not all(isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        # A clipped body is not a clear anchor, even if the padded crop fits.
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 - x1 < 24 or y2 - y1 < 48:
            continue
        if not full_body_supported(row, [x1, y1, x2, y2]):
            continue
        boxes[index] = [x1, y1, x2, y2]
    blocked = {i for i, a in boxes.items() for j, b in boxes.items() if i != j and overlapping_boxes(a, b)}
    for index, box in boxes.items():
        if index in blocked:
            continue
        x1, y1, x2, y2 = box
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        descriptor = []
        for top, bottom in ((.20, .50), (.52, .78)):
            patch = image[int(y1 + (y2-y1)*top):int(y1 + (y2-y1)*bottom),
                          int(x1 + (x2-x1)*.20):int(x1 + (x2-x1)*.80)]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 4], [0, 180, 0, 256, 0, 256]).flatten()
            descriptor.extend((hist / hist.sum() / 2.).tolist())
        rows[index]["anchor_appearance"] = descriptor
        rows[index]["anchor_clear"] = bool(sharpness >= 9.)
        rows[index]["anchor_frame"] = {
            "source_frame_index": int(context.source_frame_index),
            "source_time_sec": float(context.source_time_sec),
            "measurement_bucket": int(context.measurement_bucket),
        }
    return rows


def appearance_distance(a, b):
    if len(a) != 192 or len(b) != 192:
        return 1.
    return sqrt(max(0., 1. - min(1., sum(sqrt(max(0., x*y)) for x, y in zip(a, b)))))


class AnchoredRosterTracker(HalfCourtSlotTracker):
    """Reuse the balanced opening window, never its slot-inheritance policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appearance_anchors = {}
        self.anchor_frame = None
        self._current_identity = {}
        self._pending_reference_recovery = {}
        self.appearance_gallery = {}
        self.gallery_evidence = {}
        self._pending_gallery = {}

    def _reference_distance(self, tid, feature):
        references = self.appearance_gallery.get(tid) or [self.appearance_anchors[tid]]
        return min(appearance_distance(reference, feature) for reference in references)

    def _unobserved_net_crossing(self, track, row, frame_index):
        # Court half is not identity: continuous measured movement can cross
        # it. A gap followed by a pose well across the net, however, is NOT
        # evidence that this person traversed the entire court unseen.
        return (int(frame_index)-track.last_observation_frame > .5*self.fps
                and self._end(row) != self._court_end(track.court_xy)
                and abs(float(row['court_xy'][1])-self.court_space.net_y_m) > .8
                and abs(float(track.court_xy[1])-self.court_space.net_y_m) > .8)

    def _record_continuous_view(self, tid, row, frame_index, source, *, recovered):
        """Admit new views only from already resolved continuous observations.

        Never learn from an unknown pose or a recovery itself. A view must
        pass the SAME positive appearance/uniqueness thresholds for three
        consecutive frames and be continuous in both image and court space.
        The original reference and the displayed photos remain immutable.
        """
        gallery = self.appearance_gallery[tid]
        if (source != "anchored_roster_continuity" or recovered
                or not row.get("anchor_clear") or len(gallery) >= MAX_GALLERY_SAMPLES
                or float(row.get("confidence", 0)) < self.person_association_policy["anchor_body_min"]
                or float(row.get("location_confidence", 0)) < self.person_association_policy["anchor_location_min"]):
            self._pending_gallery.pop(tid, None)
            return
        feature = row["anchor_appearance"]
        own = self._reference_distance(tid, feature)
        other = min(self._reference_distance(other, feature) for other in self.tracks if other != tid)
        if (own > MAX_APPEARANCE_DISTANCE or other - own < APPEARANCE_MARGIN
                or appearance_distance(self.appearance_anchors[tid], feature) > CONTINUITY_APPEARANCE_VETO
                or own < APPEARANCE_MARGIN):
            self._pending_gallery.pop(tid, None)
            return
        previous = self._pending_gallery.get(tid) or {}
        consecutive = (previous.get("frame") == frame_index - 1
            and appearance_distance(previous["feature"], feature) <= MAX_APPEARANCE_DISTANCE)
        count = previous.get("count", 0) + 1 if consecutive else 1
        self._pending_gallery[tid] = {"frame": int(frame_index), "count": count, "feature": list(feature)}
        if count >= max(3, self.roster_stable_frames):
            evidence = {**row["anchor_frame"], "source": source,
                "consecutive_confirmed_frames": count, "own_distance": own, "competitor_margin": other-own}
            gallery.append(list(feature))
            self.gallery_evidence[tid].append(evidence)
            self._pending_gallery.pop(tid, None)

    @staticmethod
    def _identity_confidence_for_source(source):
        return .85 if str(source).startswith("anchored_roster_") else 0.

    def _eligible_observations(self, observations):
        # Match the PersonOnlyTracker/provider observation contract exactly.
        # The calibrated polygon is the floor, not an identity boundary. The
        # slot tracker's contains() check accidentally discarded valid poses
        # a few centimetres outside a baseline and broke their continuity.
        rows = []
        keys = set()
        for row in sorted(observations, key=lambda r: -float(r.get("confidence", 0))):
            xy, key = row.get("court_xy"), row.get("association_key")
            if (xy is None or not self.court_space.contains_athlete(xy,
                    margin_m=self.athlete_observation_margin_m,
                    lateral_margin_m=self.athlete_observation_lateral_margin_m,
                    baseline_margin_m=self.athlete_observation_baseline_margin_m)
                    or float(row.get("confidence", 0)) < self.person_association_policy["recovery_body_min"]
                    or float(row.get("location_confidence", 0)) < self.person_association_policy["recovery_location_min"]
                    or (key and key in keys)
                    or any(self._distance(xy, other["court_xy"]) < .25 for other in rows)):
                continue
            rows.append(row)
            if key:
                keys.add(key)
        return rows

    def _qualified(self, observations):
        return [row for row in self._eligible_observations(observations)
                if len(row.get("anchor_appearance", [])) == 192 and row.get("anchor_frame")]

    def _bootstrap_roster(self, frame_index, observations):
        present = HalfCourtSlotTracker._qualified(self, observations)
        self._last_roster_candidate_count = len(present)
        per_half = self.expected_roster_count // 2
        balanced = all(sum(self._end(r) == end for r in present) == per_half for end in ("end_a", "end_b"))
        if present and self._roster_discovery_started_frame is None:
            self._roster_discovery_started_frame = int(frame_index)
        self._roster_stable_observation_frames = self._roster_stable_observation_frames + 1 if balanced else 0
        rows = [row for row in self._qualified(observations) if row.get("anchor_clear")]
        # Discovery observes stable presence; only the FINAL chosen frame
        # needs clear crops. Requiring a whole clear discovery window would
        # discard exactly the brief all-visible frame the user requested.
        clear = (balanced and len(rows) == self.expected_roster_count
                 and all(row["anchor_frame"] == rows[0]["anchor_frame"] for row in rows))
        ready = (self._roster_discovery_started_frame is not None
                 and frame_index - self._roster_discovery_started_frame >= self.roster_discovery_frames
                 and self._roster_stable_observation_frames >= self.roster_stable_frames)
        if not clear or not ready:
            self._last_roster_reason = "awaiting_clear_same_frame_roster"
            return self.snapshot(frame_index)
        ordered = sorted(rows, key=lambda r: (self._end(r), r["court_xy"][0]))
        self.anchor_frame = deepcopy(rows[0]["anchor_frame"])
        for row in ordered:
            self._create_track(frame_index, row, association_source="anchored_roster_opening")
            tid = f"track_{self._next_id - 1:03d}"
            self.appearance_anchors[tid] = list(row["anchor_appearance"])
            self.appearance_gallery[tid] = [list(row["anchor_appearance"])]
            self.gallery_evidence[tid] = [{**self.anchor_frame, "source": "same_frame_roster"}]
            self.slot_ends[tid] = self._end(row)
            self.occupancy_segments[tid] = 1
            self._current_identity[tid] = "anchored"
        self.roster_track_ids = sorted(self.tracks)
        self.roster_status = "locked"
        self.roster_locked_frame = int(frame_index)
        self._last_roster_reason = "clear_same_frame_roster_ready"
        return self.snapshot(frame_index)

    def _update_locked_or_open_tracks(self, frame_index, observations):
        rows = self._qualified(observations)
        distances = {(tid, i): self._reference_distance(tid, row["anchor_appearance"])
                     for tid in self.appearance_anchors for i, row in enumerate(rows)}
        assignments = {}
        sources = {}
        continuity = {}
        for tid, track in self.tracks.items():
            gap = int(frame_index) - track.last_observation_frame
            if gap / self.fps > .5:
                continue
            old_box = track.last_evidence.get("bbox_xyxy")
            if not old_box:
                continue
            for i, row in enumerate(rows):
                if distances[tid, i] > CONTINUITY_APPEARANCE_VETO:
                    continue
                if any(distances[other, i]+APPEARANCE_MARGIN < distances[tid, i] for other in self.tracks if other != tid):
                    # Position continuity must never override a clear visual
                    # contradiction, even if the visually matching ID is lost.
                    continue
                box = row.get("bbox_xyxy")
                if not box:
                    continue
                intersection = max(0., min(old_box[2], box[2])-max(old_box[0], box[0])) * max(0., min(old_box[3], box[3])-max(old_box[1], box[1]))
                union = (old_box[2]-old_box[0])*(old_box[3]-old_box[1]) + (box[2]-box[0])*(box[3]-box[1]) - intersection
                if union <= 0 or intersection/union < .2:
                    continue
                gate = max(.8, self.max_speed_mps * max(1, gap)/self.fps + .35)
                if self._distance(track.court_xy, row["court_xy"]) <= gate:
                    continuity[tid, i] = self._distance(self._predict_position(track, frame_index), row["court_xy"])
        # Adjacent-frame box overlap + plausible motion + no appearance
        # contradiction preserve continuity. Position alone cannot fill a gap.
        margin = self.person_association_policy["weak_recovery_slot_margin_m"]
        for tid in self.tracks:
            ranking = sorted((cost, i) for (owner, i), cost in continuity.items() if owner == tid)
            if not ranking:
                continue
            score, index = ranking[0]
            alternatives = [cost for (owner, i), cost in continuity.items() if owner != tid and i == index]
            if ((len(ranking) > 1 and ranking[1][0] - score < margin)
                    or (alternatives and min(alternatives) - score < margin)):
                continue
            assignments[tid] = index
            sources[tid] = "anchored_roster_continuity"
        # Require a clear winner on BOTH sides of the bipartite relation.
        # Motion may veto a clothing match; motion never supplies identity.
        pending_reference_recovery = {}
        for tid, track in self.tracks.items():
            if not rows or tid in assignments:
                continue
            ranking = sorted((distances[tid, i], i) for i in range(len(rows)))
            score, index = ranking[0]
            if index in assignments.values():
                continue
            competitors = sorted(distances[other, index] for other in self.tracks if other != tid)
            if (score > MAX_APPEARANCE_DISTANCE
                    or (len(ranking) > 1 and ranking[1][0] - score < APPEARANCE_MARGIN)
                    or (competitors and competitors[0] - score < APPEARANCE_MARGIN)):
                continue
            gap = int(frame_index) - track.last_observation_frame
            if self._unobserved_net_crossing(track, rows[index], frame_index):
                continue
            if gap > self.roster_reacquire_frames:
                # A longer interruption needs repeated clear matches to the
                # ORIGINAL reference, not an inherited court-side slot. Keep
                # all earlier samples unknown until this evidence is present.
                row = rows[index]
                if (not row.get("anchor_clear") or score > .25
                        or (len(ranking) > 1 and ranking[1][0] - score < .20)
                        or (competitors and competitors[0] - score < .20)
                        or float(row.get("confidence", 0)) < self.person_association_policy["anchor_body_min"]
                        or float(row.get("location_confidence", 0)) < self.person_association_policy["anchor_location_min"]):
                    continue
                previous = self._pending_reference_recovery.get(tid) or {}
                consecutive = (previous.get("frame") == frame_index - 1
                    and self._distance(previous["court_xy"], row["court_xy"]) <= self.max_speed_mps / self.fps + .35)
                count = previous.get("count", 0) + 1 if consecutive else 1
                pending_reference_recovery[tid] = {"frame": int(frame_index), "court_xy": list(row["court_xy"]), "count": count}
                if count < max(3, self.roster_stable_frames):
                    continue
            gate = max(.8, self.max_speed_mps * max(1, gap) / self.fps + .35)
            if self._distance(track.court_xy, rows[index]["court_xy"]) > gate:
                continue
            assignments[tid] = index
            sources[tid] = ("anchored_roster_reference_recovery" if gap > self.roster_reacquire_frames else "anchored_roster_appearance")
        self._pending_reference_recovery = {tid: value for tid, value in pending_reference_recovery.items() if tid not in assignments}
        used = set(assignments.values())
        for tid, track in self.tracks.items():
            if tid not in assignments:
                track.missed_frames += max(1, int(frame_index) - track.last_frame)
                track.last_frame = int(frame_index)
                self.track_metrics[tid]["missing_frames"] += 1
                self._current_identity[tid] = "unknown"
                self._pending_gallery.pop(tid, None)
                continue
            row = rows[assignments[tid]]
            recovered = track.missed_frames > 0
            old_distance = self.track_metrics[tid]["distance_m"]
            if recovered:
                track.court_observation_history = []
                track.velocity_mps = (0., 0.)
                track.image_history = []
                track.court_history = []
            self._apply_observation(track, frame_index, row, association_source=sources[tid])
            if recovered:
                self.track_metrics[tid]["distance_m"] = old_distance
                self.track_metrics[tid]["recovered_detected_frames"] += 1
            self._current_identity[tid] = "anchored"
            self._record_continuous_view(tid, row, frame_index, sources[tid], recovered=recovered)
        self._last_roster_candidate_count = len(rows)
        self._last_unassigned_observation_count = len(rows) - len(used)
        self._last_unassigned_observations = [self._unassigned_observation_evidence(r) for i, r in enumerate(rows) if i not in used]
        self._last_roster_reason = "anchored_roster_recording" if len(assignments) == len(self.tracks) else "anchored_roster_identity_unknown"
        return self.snapshot(frame_index)

    def snapshot(self, frame_index):
        rows = super().snapshot(frame_index)
        for row in rows:
            # A slot occupancy counter must not leak into person analytics.
            row.pop("occupancy_segment", None)
            row.pop("slot_court_end", None)
            row["identity_scope"] = INDIVIDUAL_TRACKS
            row["roster_identity_policy"] = ANCHORED_ROSTER
            row["identity_status"] = self._current_identity.get(row["track_id"], "unknown")
            row["identity_anchor"] = ({**self.anchor_frame, "track_ids": sorted(self.tracks), "policy": ANCHORED_ROSTER}
                                      if self.anchor_frame else None)
            if row["identity_status"] == "unknown":
                row["association"]["identity_confidence"] = 0.
        return rows

    def roster_summary(self):
        result = super().roster_summary()
        result.update(identity_scope=INDIVIDUAL_TRACKS, identity_policy=ANCHORED_ROSTER,
                      anchor_frame=deepcopy(self.anchor_frame),
                      policy="Fixed same-frame anonymous clothing references; ambiguity/long absence stays unknown; no slot inheritance.")
        return result

    def snapshot_state(self):
        state = super().snapshot_state()
        state["anchored_roster"] = {"version": ANCHORED_ROSTER, "appearance_anchors": deepcopy(self.appearance_anchors),
                                    "gallery_version": "continuous-views.v1",
                                    "anchor_frame": deepcopy(self.anchor_frame), "current_identity": dict(self._current_identity),
                                    "pending_reference_recovery": deepcopy(self._pending_reference_recovery),
                                    "appearance_gallery": deepcopy(self.appearance_gallery),
                                    "gallery_evidence": deepcopy(self.gallery_evidence),
                                    "pending_gallery": deepcopy(self._pending_gallery)}
        return state

    def restore_state(self, state):
        anchor = state.get("anchored_roster") or {}
        if anchor.get("version") != ANCHORED_ROSTER:
            raise ValueError("Anchored roster checkpoint policy mismatch")
        anchors = anchor.get("appearance_anchors") or {}
        if (set(anchors) != set(state.get("tracks", {}))
                or any(len(v) != 192 or not all(isfinite(x) and x >= 0 for x in v)
                       or abs(sum(v) - 1.) > .001 for v in anchors.values())):
            raise ValueError("Incomplete anchored roster checkpoint")
        if anchors and not anchor.get("anchor_frame"):
            raise ValueError("Missing anchored roster reference frame")
        galleries = anchor.get("appearance_gallery")
        evidence = anchor.get("gallery_evidence")
        if galleries is not None:
            if (anchor.get("gallery_version") != "continuous-views.v1"
                    or set(galleries) != set(anchors) or not isinstance(evidence, dict) or set(evidence) != set(anchors)
                    or any(not 1 <= len(views) <= MAX_GALLERY_SAMPLES or views[0] != anchors[tid]
                        or len(evidence[tid]) != len(views)
                        or any(len(v) != 192 or not all(isfinite(x) and x >= 0 for x in v)
                            or abs(sum(v)-1.) > .001 for v in views)
                        for tid, views in galleries.items())):
                raise ValueError("Invalid anchored appearance gallery checkpoint")
        super().restore_state(state)
        self.appearance_anchors = deepcopy(anchors)
        self.anchor_frame = deepcopy(anchor.get("anchor_frame"))
        self._current_identity = dict(anchor.get("current_identity") or {})
        self._pending_reference_recovery = deepcopy(anchor.get("pending_reference_recovery") or {})
        # An old checkpoint safely starts with only its original reference;
        # no new view is invented during migration.
        self.appearance_gallery = deepcopy(galleries if galleries is not None else {tid:[v] for tid,v in anchors.items()})
        self.gallery_evidence = deepcopy(evidence if galleries is not None else
            {tid:[{**self.anchor_frame, "source":"same_frame_roster"}] for tid in anchors})
        self._pending_gallery = deepcopy(anchor.get("pending_gallery") or {})
