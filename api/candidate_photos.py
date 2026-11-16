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
        if self.anchored_roster_count is not None:
            return self._attach_roster(frame, context, list(events))
        output: list[ProcessorEvent] = []
        for event in events:
            candidate = self._capture(frame, context, event)
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

    def _attach_roster(self, frame, context, events):
        if self._frozen_set is not None:
            return events
        selected = [e for e in events if e.event_type == "person_observation"]
        if len(selected) != self.anchored_roster_count or any(e.evidence_state != "detected" for e in selected):
            return events
        tracks = [e.data.get("track") or {} for e in selected]
        ids = sorted(str(t.get("track_id") or "") for t in tracks)
        anchors = [t.get("identity_anchor") for t in tracks]
        reference = anchors[0]
        if (len(set(ids)) != self.anchored_roster_count or not reference
                or reference.get("policy") != "anchored_roster.v1"
                or reference.get("track_ids") != ids
                or any(anchor != reference for anchor in anchors)
                or reference.get("source_frame_index") != context.source_frame_index
                or abs(reference.get("source_time_sec", -1) - context.source_time_sec) > 1e-6
                or any(t.get("identity_status") != "anchored" for t in tracks)):
            return events
        from badminton_analysis.tracking.anchored_roster import overlapping_boxes
        boxes = [(t.get("location_evidence") or {}).get("bbox_xyxy") for t in tracks]
        if (any(not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box)
                for box in boxes)
                or any(overlapping_boxes(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])):
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

    def _verify_photo_set(self, manifest):
        photos = manifest.get("photos") or {}
        if (manifest.get("version") != "same-frame-roster.v1" or not manifest.get("set_id")
                or len(photos) != self.anchored_roster_count):
            raise ValueError("Incomplete photo roster checkpoint")
        frames = set()
        for tid, photo in photos.items():
            if not _TRACK_ID_PATTERN.fullmatch(tid):
                raise ValueError("Invalid photo roster ID")
            if (photo.get("track_id") != tid or photo.get("photo_set_id") != manifest["set_id"]
                    or photo.get("frozen") is not True):
                raise ValueError("Photo roster checkpoint metadata mismatch")
            frames.add((photo.get("source_frame_index"), photo.get("source_time_sec")))
            path = self.directory / "anchor_set" / f"{tid}.jpg"
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != photo.get("sha256"):
                raise ValueError("Photo roster checkpoint file mismatch")
        if len(frames) != 1 or None in next(iter(frames)):
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

    def _capture(self, frame: Any, context: FrameContext, event: ProcessorEvent) -> dict[str, Any] | None:
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
        bbox = ((track.get("location_evidence") or {}).get("bbox_xyxy"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        image = np.asarray(frame)
        if image.ndim < 2:
            return None
        height, width = image.shape[:2]
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        pad_x = max(4, int(round((x2 - x1) * 0.12)))
        pad_y = max(4, int(round((y2 - y1) * 0.08)))
        left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
        right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
        if right - left < 24 or bottom - top < 48:
            return None
        confidence = max(0.0, min(1.0, float(track.get("confidence") or 0.0)))
        size_factor = min(1.0, (bottom - top) / max(1.0, height * 0.28))
        frontal_score, view_label = self._front_view_score(track, bbox_width=right - left)
        # Sharpness is bounded by one. Skip its image work only when even the
        # best possible sharpness cannot beat the existing selection.
        upper_quality = round(confidence * (0.55 + 0.30 * size_factor + 0.15), 4)
        upper_rank = round(0.55 * frontal_score + 0.45 * upper_quality, 4)
        if upper_rank <= self._best_rank.get(track_id, -1.0):
            return None
        crop = image[top:bottom, left:right]
        sharpness = self._sharpness(crop)
        quality = round(confidence * (0.55 + 0.30 * size_factor + 0.15 * sharpness), 4)
        # Front-facing evidence is intentionally weighted above marginal
        # detection-quality differences. A very poor crop still cannot win,
        # because quality remains nearly half of the final rank.
        rank = round(0.55 * frontal_score + 0.45 * quality, 4)
        if rank <= self._best_rank.get(track_id, -1.0):
            return None
        encoded, payload = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{track_id}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload.tobytes())
        os.replace(temporary, destination)
        self._best_quality[track_id] = quality
        self._best_rank[track_id] = rank
        return {
            "track_id": track_id,
            "source_time_sec": round(float(context.source_time_sec), 6),
            "capture_quality": quality,
            "frontal_score": frontal_score,
            "view_label": view_label,
            "selection_score": rank,
            "selection_policy": "pose_front_priority_v1",
            "media_type": "image/jpeg",
        }

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
