"""Anonymous full-body candidate photos for post-match role claiming.

The GPU never identifies a person.  It keeps one high-quality full-body crop
per anonymous ``track_id`` and prefers a pose-supported front view, so the
business service can later let a player recognise themselves without face
recognition or biometric identity storage.
"""

from __future__ import annotations

import os
import re
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from badminton_analysis.streaming.models import FrameContext, ProcessorEvent


_TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_POSE_KEYPOINT_CONFIDENCE = 0.25
_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR = range(5)
_UPGRADE_MIN_FRONTAL = 0.42
_UPGRADE_FRONTAL_DELTA = 0.15
_UPGRADE_RANK_DELTA = 0.05
_MAX_PHOTO_OVERLAP_RATIO = 0.05


def _valid_box(box: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        coords = tuple(float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in coords):
        return None
    return coords  # type: ignore[return-value]


def _box_overlap_ratio(
    box: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
) -> float:
    inter_w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    inter_h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    self_area = (box[2] - box[0]) * (box[3] - box[1])
    if self_area <= 0.0:
        return 1.0
    return (inter_w * inter_h) / self_area


class CandidatePhotoCollector:
    """Persist the best anonymous player crop seen for each track.

    A crop is selected from the same pose observation that produced the public
    track event.  The collector stores no user identity, face embedding, or
    score; ``track_id`` remains anonymous until the business service accepts a
    later, user-confirmed claim.
    """

    def __init__(self, directory: Path | str, *, anchored_roster_count: int | None = None):
        self.directory = Path(directory)
        if anchored_roster_count is not None and anchored_roster_count not in {2, 4}:
            raise ValueError("A photo roster requires two or four people")
        self.anchored_roster_count = anchored_roster_count
        self._frozen_set = None
        self._best_quality: dict[str, float] = {}
        self._best_rank: dict[str, float] = {}

    def attach(
        self,
        frame: Any,
        context: FrameContext,
        events: Iterable[ProcessorEvent],
    ) -> list[ProcessorEvent]:
        event_list = list(events)
        if self.anchored_roster_count is not None:
            return self._attach_roster(frame, context, event_list)
        frame_boxes = self._frame_boxes(event_list)
        output: list[ProcessorEvent] = []
        for index, event in enumerate(event_list):
            other_boxes = [box for other_index, box in frame_boxes if other_index != index]
            candidate = self._capture(
                frame,
                context,
                event,
                other_boxes=other_boxes or None,
            )
            if candidate is None:
                output.append(event)
                continue
            data = deepcopy(dict(event.data))
            data["candidate_photo"] = candidate
            output.append(
                ProcessorEvent(
                    event_type=event.event_type,
                    evidence_state=event.evidence_state,
                    confidence=event.confidence,
                    data=data,
                    source_time_sec=event.source_time_sec,
                    segment_index=event.segment_index,
                )
            )
        return output

    @staticmethod
    def _frame_boxes(
        events: Iterable[ProcessorEvent],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for index, event in enumerate(events):
            if (
                event.event_type not in {"person_observation", "roster_candidate_observation"}
                or event.evidence_state not in {"detected", "predicted"}
            ):
                continue
            track = (event.data or {}).get("track")
            if not isinstance(track, dict):
                continue
            coords = _valid_box((track.get("location_evidence") or {}).get("bbox_xyxy"))
            if coords is not None:
                boxes.append((index, coords))
        return boxes

    def _attach_roster(self, frame, context, events):
        if self._frozen_set is not None:
            return self._upgrade_roster(frame, context, events)
        selected = [e for e in events if e.event_type == "person_observation"]
        if len(selected) != self.anchored_roster_count or any(e.evidence_state != "detected" for e in selected):
            return events
        tracks = [e.data.get("track") or {} for e in selected]
        ids = sorted(str(t.get("track_id") or "") for t in tracks)
        anchors = [t.get("identity_anchor") for t in tracks]
        reference = anchors[0]
        ref_frame = reference.get("source_frame_index") if isinstance(reference, dict) else None
        ref_time = reference.get("source_time_sec") if isinstance(reference, dict) else None
        if (len(set(ids)) != self.anchored_roster_count or not reference
                or reference.get("policy") != "anchored_roster.v1"
                or reference.get("track_ids") != ids
                or any(anchor != reference for anchor in anchors)
                or not isinstance(ref_frame, int) or isinstance(ref_frame, bool)
                or not isinstance(ref_time, (int, float)) or isinstance(ref_time, bool)
                or not np.isfinite(ref_time)
                or int(context.source_frame_index) < int(ref_frame)
                or float(context.source_time_sec) + 1e-6 < float(ref_time)
                or any(t.get("identity_status") != "anchored" for t in tracks)):
            return events
        from badminton_analysis.tracking.anchored_roster import overlapping_boxes
        boxes = [_valid_box((t.get("location_evidence") or {}).get("bbox_xyxy")) for t in tracks]
        if any(box is None for box in boxes) or any(
            overlapping_boxes(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]
        ):
            return events
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / "anchor_set"
        set_id = f"anchor_{context.source_frame_index}_{context.measurement_bucket}"
        # One directory rename publishes all images + manifest together. A
        # crash before rename exposes none; a retry after rename reuses the
        # exact committed set rather than mixing in a later frame.
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest.get("set_id") != set_id or sorted(manifest.get("photos", {})) != ids:
                raise ValueError("Photo roster frame differs from committed set")
            self._verify_photo_set(manifest)
            photos = manifest["photos"]
        else:
            staging = Path(tempfile.mkdtemp(prefix=".anchor-set-", dir=self.directory))
            try:
                collector = CandidatePhotoCollector(staging)
                photos = {}
                for event in selected:
                    photo = collector._capture(frame, context, event)
                    if photo is None:
                        raise ValueError("Incomplete same-frame photo roster")
                    tid = photo["track_id"]
                    photo.update(photo_set_id=set_id, source_frame_index=int(context.source_frame_index),
                                 selection_policy="same_frame_roster_v1", frozen=True)
                    photo["sha256"] = hashlib.sha256((staging / f"{tid}.jpg").read_bytes()).hexdigest()
                    photos[tid] = photo
                manifest = {"version": "same-frame-roster.v1", "set_id": set_id, "photos": photos}
                (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        self._frozen_set = manifest
        for tid, photo in photos.items():
            self._best_quality[tid] = float(photo.get("capture_quality") or 0.0)
            self._best_rank[tid] = float(photo.get("selection_score") or 0.0)
        output = []
        for event in events:
            tid = (event.data.get("track") or {}).get("track_id")
            if event.event_type == "person_observation" and tid in photos:
                data = deepcopy(dict(event.data)); data["candidate_photo"] = deepcopy(photos[tid])
                event = ProcessorEvent(event_type=event.event_type, evidence_state=event.evidence_state,
                                       confidence=event.confidence, data=data,
                                       source_time_sec=event.source_time_sec, segment_index=event.segment_index)
            output.append(event)
        return output

    def _upgrade_roster(self, frame: Any, context: FrameContext, events: list[ProcessorEvent]) -> list[ProcessorEvent]:
        from badminton_analysis.tracking.anchored_roster import full_body_supported, overlapping_boxes

        manifest = deepcopy(self._frozen_set or {})
        photos = dict(manifest.get("photos") or {})
        set_id = manifest.get("set_id")
        if not set_id or not photos:
            return events
        for tid, photo in photos.items():
            self._best_quality.setdefault(tid, float(photo.get("capture_quality") or 0.0))
            self._best_rank.setdefault(tid, float(photo.get("selection_score") or 0.0))

        frame_boxes = self._frame_boxes(events)
        box_by_index = dict(frame_boxes)
        destination = self.directory / "anchor_set"
        upgraded: dict[str, dict[str, Any]] = {}
        manifest_dirty = False
        roster_ids = sorted(photos)

        for index, event in enumerate(events):
            if event.event_type != "person_observation" or event.evidence_state != "detected":
                continue
            track = (event.data or {}).get("track")
            if not isinstance(track, dict):
                continue
            tid = str(track.get("track_id") or "")
            if tid not in photos or track.get("identity_status") != "anchored":
                continue
            anchor = track.get("identity_anchor")
            if (
                not isinstance(anchor, dict)
                or anchor.get("policy") != "anchored_roster.v1"
                or anchor.get("track_ids") != roster_ids
            ):
                continue
            old_photo = photos[tid]
            if int(context.source_frame_index) <= int(old_photo.get("source_frame_index", -1)):
                continue
            box = box_by_index.get(index)
            if box is None:
                continue
            loc = track.get("location_evidence") or {}
            if (
                loc.get("anchor_clear") is False
                or float(loc.get("overlap_ratio") or 0.0) > _MAX_PHOTO_OVERLAP_RATIO
            ):
                continue
            other_boxes = [other for other_idx, other in frame_boxes if other_idx != index]
            if any(
                overlapping_boxes(box, other) or _box_overlap_ratio(box, other) > _MAX_PHOTO_OVERLAP_RATIO
                for other in other_boxes
            ):
                continue
            pose = track.get("pose") if isinstance(track.get("pose"), dict) else {}
            pose_row = {
                "keypoints_image": pose.get("keypoints_image") or track.get("keypoints_image"),
                "keypoint_scores": pose.get("keypoint_scores") or track.get("keypoint_scores"),
            }
            if not full_body_supported(pose_row, box):
                continue
            old_frontal = float(old_photo.get("frontal_score") or 0.0)
            old_rank = float(old_photo.get("selection_score") or self._best_rank.get(tid, 0.0))
            min_frontal = max(_UPGRADE_MIN_FRONTAL, old_frontal + _UPGRADE_FRONTAL_DELTA)
            scored = self._score_and_encode(
                frame,
                context,
                event,
                min_rank=old_rank + _UPGRADE_RANK_DELTA,
                min_frontal=min_frontal,
                other_boxes=other_boxes or None,
            )
            if scored is None:
                continue
            photo, payload, quality, rank = scored
            photo.update(
                photo_set_id=set_id,
                source_frame_index=int(context.source_frame_index),
                selection_policy="anchored_front_upgrade_v1",
                frozen=True,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            destination.mkdir(parents=True, exist_ok=True)
            img_path = destination / f"{tid}.jpg"
            img_tmp = destination / f".{tid}.jpg.tmp"
            img_tmp.write_bytes(payload)
            os.replace(img_tmp, img_path)
            photos[tid] = photo
            self._best_quality[tid] = quality
            self._best_rank[tid] = rank
            upgraded[tid] = photo
            manifest_dirty = True

        if not manifest_dirty:
            return events
        manifest["photos"] = photos
        manifest_tmp = destination / ".manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, sort_keys=True))
        os.replace(manifest_tmp, destination / "manifest.json")
        self._frozen_set = manifest

        output: list[ProcessorEvent] = []
        for event in events:
            tid = (event.data.get("track") or {}).get("track_id")
            if event.event_type == "person_observation" and tid in upgraded:
                data = deepcopy(dict(event.data))
                data["candidate_photo"] = deepcopy(upgraded[tid])
                event = ProcessorEvent(
                    event_type=event.event_type,
                    evidence_state=event.evidence_state,
                    confidence=event.confidence,
                    data=data,
                    source_time_sec=event.source_time_sec,
                    segment_index=event.segment_index,
                )
            output.append(event)
        return output

    def _verify_photo_set(self, manifest):
        photos = manifest.get("photos") or {}
        if (manifest.get("version") != "same-frame-roster.v1" or not manifest.get("set_id")
                or len(photos) != self.anchored_roster_count):
            raise ValueError("Incomplete photo roster checkpoint")
        same_frame_coords = set()
        upgrade_frames: list[int] = []
        for tid, photo in photos.items():
            if not _TRACK_ID_PATTERN.fullmatch(tid):
                raise ValueError("Invalid photo roster ID")
            policy = photo.get("selection_policy")
            if (photo.get("track_id") != tid or photo.get("photo_set_id") != manifest["set_id"]
                    or photo.get("frozen") is not True
                    or policy not in {"same_frame_roster_v1", "anchored_front_upgrade_v1"}):
                raise ValueError("Photo roster checkpoint metadata mismatch")
            frame_idx = photo.get("source_frame_index")
            src_time = photo.get("source_time_sec")
            if (not isinstance(frame_idx, int) or isinstance(frame_idx, bool) or frame_idx < 0
                    or not isinstance(src_time, (int, float)) or isinstance(src_time, bool)
                    or not np.isfinite(src_time)):
                raise ValueError("Photo roster checkpoint source frame mismatch")
            if policy == "same_frame_roster_v1":
                same_frame_coords.add((frame_idx, round(float(src_time), 6)))
            else:
                upgrade_frames.append(frame_idx)
            path = self.directory / "anchor_set" / f"{tid}.jpg"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != photo.get("sha256"):
                raise ValueError("Photo roster checkpoint file mismatch")
        if same_frame_coords:
            if len(same_frame_coords) != 1:
                raise ValueError("Photo roster checkpoint source frame mismatch")
            base_frame = next(iter(same_frame_coords))[0]
            if any(frame_idx <= base_frame for frame_idx in upgrade_frames):
                raise ValueError("Photo roster checkpoint source frame mismatch")

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "state_version": "candidate-photo.v3",
            "anchored_roster_count": self.anchored_roster_count,
            "frozen_set": deepcopy(self._frozen_set),
            "best_quality": dict(self._best_quality),
            "best_rank": dict(self._best_rank),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        version = state.get("state_version")
        if version not in {"candidate-photo.v1", "candidate-photo.v2", "candidate-photo.v3"}:
            raise ValueError("unsupported candidate photo checkpoint")
        if state.get("anchored_roster_count") != self.anchored_roster_count:
            raise ValueError("Candidate photo checkpoint roster policy mismatch")
        frozen = state.get("frozen_set")
        if frozen is not None:
            self._verify_photo_set(frozen)
            manifest_path = self.directory / "anchor_set" / "manifest.json"
            if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != frozen:
                raise ValueError("Photo roster checkpoint manifest mismatch")
        self._frozen_set = deepcopy(frozen)
        self._best_quality = {
            str(track_id): float(quality)
            for track_id, quality in dict(state.get("best_quality") or {}).items()
            if _TRACK_ID_PATTERN.fullmatch(str(track_id)) and float(quality) >= 0.0
        }
        if version in {"candidate-photo.v2", "candidate-photo.v3"}:
            rank_source = dict(state.get("best_rank") or {})
        else:
            # A v1 checkpoint has no pose-orientation evidence. Preserve its
            # old quality ordering until a newly scored candidate is observed.
            rank_source = self._best_quality
        self._best_rank = {
            str(track_id): float(rank)
            for track_id, rank in rank_source.items()
            if _TRACK_ID_PATTERN.fullmatch(str(track_id)) and float(rank) >= 0.0
        }

    def _capture(
        self,
        frame: Any,
        context: FrameContext,
        event: ProcessorEvent,
        *,
        other_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> dict[str, Any] | None:
        scored = self._score_and_encode(
            frame,
            context,
            event,
            other_boxes=other_boxes,
        )
        if scored is None:
            return None
        photo, payload, quality, rank = scored
        track_id = photo["track_id"]
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{track_id}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        self._best_quality[track_id] = quality
        self._best_rank[track_id] = rank
        return photo

    def _score_and_encode(
        self,
        frame: Any,
        context: FrameContext,
        event: ProcessorEvent,
        *,
        min_rank: float | None = None,
        min_frontal: float = 0.0,
        other_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[dict[str, Any], bytes, float, float] | None:
        if event.event_type not in {
            "person_observation",
            "roster_candidate_observation",
        } or event.evidence_state != "detected":
            return None
        track = (event.data or {}).get("track")
        if not isinstance(track, dict):
            return None
        track_id = str(track.get("track_id") or "")
        if not _TRACK_ID_PATTERN.fullmatch(track_id):
            return None
        box = _valid_box((track.get("location_evidence") or {}).get("bbox_xyxy"))
        if box is None:
            return None
        if other_boxes and any(
            _box_overlap_ratio(box, other) > _MAX_PHOTO_OVERLAP_RATIO
            for other in other_boxes
        ):
            return None
        image = np.asarray(frame)
        if image.ndim < 2:
            return None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = max(4, int(round((x2 - x1) * 0.12)))
        pad_y = max(4, int(round((y2 - y1) * 0.08)))
        left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
        right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
        if right - left < 24 or bottom - top < 48:
            return None
        confidence = max(0.0, min(1.0, float(track.get("confidence") or 0.0)))
        size_factor = min(1.0, (bottom - top) / max(1.0, height * 0.28))
        frontal_score, view_label = self._front_view_score(track, bbox_width=right - left)
        if frontal_score < min_frontal:
            return None
        threshold = self._best_rank.get(track_id, -1.0) if min_rank is None else min_rank
        # Sharpness is bounded by one. Skip its image work only when even the
        # best possible sharpness cannot beat the existing selection.
        upper_quality = round(confidence * (0.55 + 0.30 * size_factor + 0.15), 4)
        upper_rank = round(0.55 * frontal_score + 0.45 * upper_quality, 4)
        if upper_rank <= threshold:
            return None
        crop = image[top:bottom, left:right]
        sharpness = self._sharpness(crop)
        quality = round(confidence * (0.55 + 0.30 * size_factor + 0.15 * sharpness), 4)
        # Front-facing evidence is intentionally weighted above marginal
        # detection-quality differences. A very poor crop still cannot win,
        # because quality remains nearly half of the final rank.
        rank = round(0.55 * frontal_score + 0.45 * quality, 4)
        if rank <= threshold:
            return None
        encoded, payload = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            return None
        photo = {
            "track_id": track_id,
            "source_time_sec": round(float(context.source_time_sec), 6),
            "capture_quality": quality,
            "frontal_score": frontal_score,
            "view_label": view_label,
            "selection_score": rank,
            "selection_policy": "pose_front_priority_v1",
            "media_type": "image/jpeg",
        }
        return photo, payload.tobytes(), quality, rank

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        """Return a bounded non-biometric sharpness signal for crop ranking."""

        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(max(0.0, min(1.0, variance / 180.0)), 4)

    @classmethod
    def _front_view_score(cls, track: dict[str, Any], *, bbox_width: int) -> tuple[float, str]:
        """Estimate face orientation from anonymous COCO17 pose points.

        This is not facial recognition: it only asks whether both eyes and a
        centered nose were visible in the current measurement. When the camera
        is too distant or the model lacks head points, callers get an explicit
        ``not_assessed`` fallback instead of a fabricated front-view claim.
        """

        pose = track.get("pose") if isinstance(track.get("pose"), dict) else {}
        points = pose.get("keypoints_image") or track.get("keypoints_image")
        scores = pose.get("keypoint_scores") or track.get("keypoint_scores")

        def point(index: int) -> tuple[float, float] | None:
            if not isinstance(points, (list, tuple)) or index >= len(points):
                return None
            value = points[index]
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return None
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
            if not np.isfinite([x, y]).all() or x <= 1 or y <= 1:
                return None
            if isinstance(scores, (list, tuple)) and index < len(scores):
                try:
                    if float(scores[index]) < _POSE_KEYPOINT_CONFIDENCE:
                        return None
                except (TypeError, ValueError):
                    return None
            return x, y

        nose = point(_NOSE)
        left_eye, right_eye = point(_LEFT_EYE), point(_RIGHT_EYE)
        left_ear, right_ear = point(_LEFT_EAR), point(_RIGHT_EAR)
        visible_head_points = sum(item is not None for item in (nose, left_eye, right_eye, left_ear, right_ear))
        if not (left_eye and right_eye and nose):
            return (0.0, "side_or_back" if visible_head_points else "not_assessed")

        eye_span = abs(right_eye[0] - left_eye[0])
        minimum_span = max(3.0, float(bbox_width) * 0.06)
        if eye_span < minimum_span:
            return 0.0, "not_assessed"
        eye_midpoint = (left_eye[0] + right_eye[0]) / 2.0
        nose_offset = abs(nose[0] - eye_midpoint) / eye_span
        symmetry = max(0.0, 1.0 - nose_offset / 0.65)
        ears_visible = sum(item is not None for item in (left_ear, right_ear)) / 2.0
        score = round(max(0.0, min(1.0, 0.80 * symmetry + 0.20 * ears_visible)), 4)
        if score >= 0.72:
            return score, "front"
        if score >= 0.42:
            return score, "three_quarter"
        return score, "side_or_back"
