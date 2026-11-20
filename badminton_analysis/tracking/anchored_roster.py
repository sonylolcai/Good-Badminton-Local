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
# Empirically calibrated operating points for fixed-camera clothing references.
MAX_APPEARANCE_DISTANCE = .52
APPEARANCE_MARGIN = .08
CONTINUITY_APPEARANCE_VETO = .65
MAX_GALLERY_SAMPLES = 8


def overlapping_boxes(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller <= 0 or intersection / smaller > .35


def _visible_joint_checker(row, box):
    points, scores = row.get("keypoints_image"), row.get("keypoint_scores")
    if points is None or scores is None or len(points) < 17 or len(scores) < 17:
        return None, None
    x1, y1, x2, y2 = box

    def visible(i):
        point = points[i]
        return (
            point is not None
            and len(point) >= 2
            and all(isfinite(v) for v in point[:2])
            and scores[i] is not None
            and isfinite(scores[i])
            and scores[i] >= .25
            and x1 <= point[0] <= x2
            and y1 <= point[1] <= y2
        )

    return points, visible


def full_body_supported(row, box):
    """Leg-only ROI detections cannot supply a person's clothing anchor."""
    points, visible = _visible_joint_checker(row, box)
    if points is None:
        return False
    _, y1, _, y2 = box
    shoulder_ys = [points[i][1] for i in (5, 6) if visible(i)]
    hip_ys = [points[i][1] for i in (11, 12) if visible(i)]
    ankle_ys = [points[i][1] for i in (15, 16) if visible(i)]
    if not shoulder_ys or not hip_ys or not ankle_ys:
        return False
    shoulders = sum(shoulder_ys) / len(shoulder_ys)
    hips = sum(hip_ys) / len(hip_ys)
    ankles = max(ankle_ys)
    return hips - shoulders >= .10 * (y2 - y1) and ankles - hips >= .15 * (y2 - y1)


def torso_supported(row, box):
    """Allow crouching/lunging players with visible torso to keep clothing features."""
    points, visible = _visible_joint_checker(row, box)
    if points is None:
        return False
    _, y1, _, y2 = box
    shoulder_ys = [points[i][1] for i in (5, 6) if visible(i)]
    hip_ys = [points[i][1] for i in (11, 12) if visible(i)]
    if not shoulder_ys or not hip_ys:
        return False
    shoulders = sum(shoulder_ys) / len(shoulder_ys)
    hips = sum(hip_ys) / len(hip_ys)
    return hips - shoulders >= .06 * (y2 - y1)


def prepare_anchor_observations(frame, observations, context):
    """Derive occlusion-masked clothing evidence from each current source-frame box."""
    rows = [dict(row) for row in observations]
    for row in rows:
        row.pop("anchor_appearance", None)
        row.pop("anchor_frame", None)
        row.pop("anchor_clear", None)
        row.pop("anchor_occluded", None)
        row.pop("overlap_ratio", None)
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        return rows
    height, width = image.shape[:2]
    boxes = {}
    raw_clipped = {}
    full_body = {}
    for index, row in enumerate(rows):
        try:
            x1, y1, x2, y2 = map(float, row.get("bbox_xyxy", []))
        except (ValueError, TypeError):
            continue
        if not all(isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        is_clipped = x1 < 0 or y1 < 0 or x2 > width or y2 > height
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(width, int(x2)), min(height, int(y2))
        if cx2 - cx1 < 24 or cy2 - cy1 < 48:
            continue
        is_full_body = full_body_supported(row, [x1, y1, x2, y2])
        if not (is_full_body or torso_supported(row, [x1, y1, x2, y2])):
            continue
        boxes[index] = [cx1, cy1, cx2, cy2]
        raw_clipped[index] = is_clipped
        full_body[index] = is_full_body
    anchor_frame = {
        "source_frame_index": int(context.source_frame_index),
        "source_time_sec": float(context.source_time_sec),
        "measurement_bucket": int(context.measurement_bucket),
    }
    for index, box in boxes.items():
        cx1, cy1, cx2, cy2 = box
        area_self = (cx2 - cx1) * (cy2 - cy1)
        overlap_ratio = 0.0
        is_blocked = False
        mask = np.full((cy2 - cy1, cx2 - cx1), 255, dtype=np.uint8)
        for j, other in boxes.items():
            if j == index:
                continue
            ox1, oy1, ox2, oy2 = other
            ix1, iy1 = max(cx1, ox1), max(cy1, oy1)
            ix2, iy2 = min(cx2, ox2), min(cy2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(area_self, (ox2 - ox1) * (oy2 - oy1))
                if smaller > 0:
                    ratio = inter / smaller
                    if ratio > overlap_ratio:
                        overlap_ratio = ratio
                    if ratio > .35:
                        is_blocked = True
                # Depth ordering from fixed-camera floor projection: larger y2 is foreground.
                if oy2 >= cy2 - 8 or (smaller > 0 and inter / smaller > .35):
                    mask[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = 0
        crop = image[cy1:cy2, cx1:cx2]
        sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        descriptor = []
        occluded = False
        for top, bottom in ((.20, .50), (.52, .78)):
            py1, py2 = int(cy1 + (cy2 - cy1) * top), int(cy1 + (cy2 - cy1) * bottom)
            px1, px2 = int(cx1 + (cx2 - cx1) * .20), int(cx1 + (cx2 - cx1) * .80)
            if py2 <= py1 or px2 <= px1:
                occluded = True
                break
            pmask = mask[py1 - cy1:py2 - cy1, px1 - cx1:px2 - cx1]
            if float(np.count_nonzero(pmask)) / float(pmask.size) < .25:
                occluded = True
                break
            patch = image[py1:py2, px1:px2]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], pmask, [8, 3, 4], [0, 180, 0, 256, 0, 256]).flatten()
            total = float(hist.sum())
            if total <= 0.:
                occluded = True
                break
            descriptor.extend((hist / total / 2.).tolist())
        if not occluded and len(descriptor) == 192:
            rows[index]["anchor_appearance"] = descriptor
        elif occluded:
            rows[index]["anchor_occluded"] = True
        rows[index]["overlap_ratio"] = round(float(overlap_ratio), 4)
        rows[index]["anchor_clear"] = bool(
            full_body[index] and not raw_clipped[index] and not is_blocked and sharpness >= 9.
            and len(rows[index].get("anchor_appearance", [])) == 192
        )
        rows[index]["anchor_frame"] = dict(anchor_frame)
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
        self._discovery_slot_anchors = {}
        self._last_overlap_ratio = {}
        self._last_anchor_clear = {}

    def _reference_distance(self, tid, feature):
        references = self.appearance_gallery.get(tid) or [self.appearance_anchors[tid]]
        return min(appearance_distance(reference, feature) for reference in references)

    def _reachable_competitors(self, tid, row, frame_index, *, excluded=()):
        row_end = self._end(row)
        row_xy = row["court_xy"]
        competitors = []
        for other, track in self.tracks.items():
            if other == tid or other in excluded:
                continue
            if self._unobserved_net_crossing(track, row, frame_index):
                continue
            gap_sec = max(1, int(frame_index) - track.last_observation_frame) / self.fps
            reach = max(2.2, self.max_speed_mps * gap_sec + 0.5)
            if (self._court_end(track.court_xy) == row_end
                    or self._distance(track.court_xy, row_xy) <= reach):
                competitors.append(other)
        return competitors

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
        feature = row.get("anchor_appearance")
        if (source != "anchored_roster_continuity" or recovered
                or not row.get("anchor_clear") or float(row.get("overlap_ratio") or 0.) > .10
                or not isinstance(feature, list) or len(feature) != 192
                or len(gallery) >= MAX_GALLERY_SAMPLES
                or float(row.get("confidence", 0)) < self.person_association_policy["anchor_body_min"]
                or float(row.get("location_confidence", 0)) < self.person_association_policy["anchor_location_min"]):
            self._pending_gallery.pop(tid, None)
            return
        own = self._reference_distance(tid, feature)
        other = min(self._reference_distance(other, feature) for other in self.tracks if other != tid)
        if (own > MAX_APPEARANCE_DISTANCE or other - own < APPEARANCE_MARGIN
                or appearance_distance(self.appearance_anchors[tid], feature) > CONTINUITY_APPEARANCE_VETO
                or own < APPEARANCE_MARGIN):
            self._pending_gallery.pop(tid, None)
            return
        previous = self._pending_gallery.get(tid) or {}
        consecutive = (1 <= int(frame_index) - int(previous.get("frame", -99)) <= 2
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
        present = self._eligible_observations(observations)
        self._last_roster_candidate_count = len(present)
        per_half = self.expected_roster_count // 2
        balanced = all(sum(self._end(r) == end for r in present) == per_half for end in ("end_a", "end_b"))
        grace_frames = max(1, round(min(.75, self.roster_discovery_seconds * .25) * self.fps))
        if self._ready_samples and frame_index - self._ready_samples[-1][0] > grace_frames:
            self._ready_samples.clear()
            self._discovery_slot_anchors.clear()
            self._roster_discovery_started_frame = None
        if balanced:
            self._last_balanced_frame = frame_index
        elif self._last_balanced_frame is None or frame_index - self._last_balanced_frame > grace_frames:
            self._ready_samples.clear()
            self._discovery_slot_anchors.clear()
            self._roster_discovery_started_frame = None
        if balanced or self._ready_samples:
            self._ready_samples.append((int(frame_index), balanced))
        if self._ready_samples and self._roster_discovery_started_frame is None:
            self._roster_discovery_started_frame = self._ready_samples[0][0]
        while self._ready_samples and frame_index - self._ready_samples[0][0] > self.roster_discovery_frames:
            self._ready_samples.popleft()
        self._roster_stable_observation_frames = self._roster_stable_observation_frames + 1 if balanced else 0
        if balanced:
            ordered_present = sorted(present, key=lambda r: (self._end(r), r["court_xy"][0]))
            for idx, r in enumerate(ordered_present):
                if r.get("anchor_clear") and len(r.get("anchor_appearance", [])) == 192 and r.get("anchor_frame"):
                    self._discovery_slot_anchors[idx] = {
                        "anchor_appearance": list(r["anchor_appearance"]),
                        "anchor_frame": deepcopy(r["anchor_frame"]),
                        "court_xy": list(r["court_xy"]),
                        "frame_index": int(frame_index),
                    }
        rows = [row for row in self._qualified(observations) if row.get("anchor_clear")]
        clear_balanced = all(sum(self._end(r) == end for r in rows) == per_half for end in ("end_a", "end_b"))
        clear = (clear_balanced and len(rows) == self.expected_roster_count
                 and all(row["anchor_frame"] == rows[0]["anchor_frame"] for row in rows))
        ready = (self._roster_discovery_started_frame is not None
                 and frame_index - self._roster_discovery_started_frame >= self.roster_discovery_frames
                 and self._roster_stable_observation_frames >= self.roster_stable_frames
                 and len(self._ready_samples) > 0
                 and sum(valid for _, valid in self._ready_samples) / len(self._ready_samples) >= .8)
        selected_rows = None
        if clear and ready:
            selected_rows = sorted(rows, key=lambda r: (self._end(r), r["court_xy"][0]))
            self.anchor_frame = deepcopy(rows[0]["anchor_frame"])
        elif (balanced and ready
              and frame_index - self._roster_discovery_started_frame >= self.roster_discovery_frames + round(self.fps)):
            ordered_present = sorted(present, key=lambda r: (self._end(r), r["court_xy"][0]))
            max_age = max(1, round(2.0 * self.fps))
            candidate_rows = []
            for idx, r in enumerate(ordered_present):
                if r.get("anchor_clear") and len(r.get("anchor_appearance", [])) == 192 and r.get("anchor_frame"):
                    candidate_rows.append(dict(r))
                else:
                    cached = self._discovery_slot_anchors.get(idx)
                    if (cached and int(frame_index) - int(cached["frame_index"]) <= max_age
                            and self._distance(r["court_xy"], cached["court_xy"]) <= 1.0):
                        merged = dict(r)
                        merged["anchor_appearance"] = list(cached["anchor_appearance"])
                        merged.setdefault("anchor_frame", deepcopy(cached["anchor_frame"]))
                        candidate_rows.append(merged)
            if len(candidate_rows) == self.expected_roster_count:
                ref_frame = next((r["anchor_frame"] for r in candidate_rows if r.get("anchor_frame")), None)
                if ref_frame is not None:
                    selected_rows = candidate_rows
                    self.anchor_frame = deepcopy(ref_frame)
        if selected_rows is None:
            self._last_roster_reason = "awaiting_clear_same_frame_roster"
            return self.snapshot(frame_index)
        for row in selected_rows:
            self._create_track(frame_index, row, association_source="anchored_roster_opening")
            tid = f"track_{self._next_id - 1:03d}"
            self.appearance_anchors[tid] = list(row["anchor_appearance"])
            self.appearance_gallery[tid] = [list(row["anchor_appearance"])]
            self.gallery_evidence[tid] = [{**self.anchor_frame, "source": "same_frame_roster"}]
            self.slot_ends[tid] = self._end(row)
            self.occupancy_segments[tid] = 1
            self._current_identity[tid] = "anchored"
            self._last_overlap_ratio[tid] = float(row.get("overlap_ratio") or 0.0)
            self._last_anchor_clear[tid] = bool(row.get("anchor_clear", False))
        self.roster_track_ids = sorted(self.tracks)
        self.roster_status = "locked"
        self.roster_locked_frame = int(frame_index)
        self._last_roster_reason = "clear_same_frame_roster_ready"
        return self.snapshot(frame_index)

    def _update_locked_or_open_tracks(self, frame_index, observations):
        rows = self._eligible_observations(observations)
        distances = {}
        for tid in self.appearance_anchors:
            for i, row in enumerate(rows):
                feat = row.get("anchor_appearance")
                if isinstance(feat, list) and len(feat) == 192:
                    distances[tid, i] = self._reference_distance(tid, feat)
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
                box = row.get("bbox_xyxy")
                if not box:
                    continue
                intersection = max(0., min(old_box[2], box[2])-max(old_box[0], box[0])) * max(0., min(old_box[3], box[3])-max(old_box[1], box[1]))
                union = (old_box[2]-old_box[0])*(old_box[3]-old_box[1]) + (box[2]-box[0])*(box[3]-box[1]) - intersection
                dist = distances.get((tid, i))
                if dist is not None:
                    if dist > CONTINUITY_APPEARANCE_VETO:
                        continue
                    reachable = self._reachable_competitors(tid, row, frame_index)
                    if any(distances.get((other, i)) is not None
                           and distances[other, i] + APPEARANCE_MARGIN < dist
                           for other in reachable):
                        # Position continuity must never override a clear visual
                        # contradiction, even if the visually matching ID is lost.
                        continue
                    if union <= 0 or intersection/union < .2:
                        continue
                else:
                    if union <= 0 or intersection/union < .25:
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
            ranking = sorted((distances[tid, i], i) for i in range(len(rows)) if (tid, i) in distances)
            if not ranking:
                continue
            score, index = ranking[0]
            if index in assignments.values():
                continue
            reachable = self._reachable_competitors(tid, rows[index], frame_index, excluded=assignments)
            competitors = sorted(distances[other, index] for other in reachable if (other, index) in distances)
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
                if (row.get("anchor_occluded")
                        or float(row.get("overlap_ratio") or 0.) > .35
                        or score > .50
                        or (len(ranking) > 1 and ranking[1][0] - score < .08)
                        or (competitors and competitors[0] - score < .08)
                        or float(row.get("confidence", 0)) < self.person_association_policy["anchor_body_min"]
                        or float(row.get("location_confidence", 0)) < self.person_association_policy["anchor_location_min"]):
                    continue
                previous = self._pending_reference_recovery.get(tid) or {}
                prev_gap = int(frame_index) - int(previous.get("frame", -99))
                consecutive = (1 <= prev_gap <= 3
                    and self._distance(previous["court_xy"], row["court_xy"]) <= self.max_speed_mps * max(1, prev_gap) / self.fps + .35)
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
            self._last_overlap_ratio[tid] = float(row.get("overlap_ratio") or 0.0)
            self._last_anchor_clear[tid] = bool(row.get("anchor_clear", False))
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
            if isinstance(row.get("location_evidence"), dict) and row["track_id"] in self._last_overlap_ratio:
                row["location_evidence"]["overlap_ratio"] = self._last_overlap_ratio[row["track_id"]]
                row["location_evidence"]["anchor_clear"] = self._last_anchor_clear.get(row["track_id"], False)
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
                                    "pending_gallery": deepcopy(self._pending_gallery),
                                    "discovery_slot_anchors": {str(k): deepcopy(v) for k, v in self._discovery_slot_anchors.items()}}
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
        self._discovery_slot_anchors = {
            int(k): deepcopy(v)
            for k, v in dict(anchor.get("discovery_slot_anchors") or {}).items()
        }
