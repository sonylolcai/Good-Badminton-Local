from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from api.candidate_photos import CandidatePhotoCollector
from badminton_analysis.streaming.models import FrameContext, ProcessorEvent


def _context(source_time_sec: float) -> FrameContext:
    return FrameContext(
        analysis_session_id="candidate-photo-test",
        segment_index=0,
        source_frame_index=int(source_time_sec * 10),
        source_time_sec=source_time_sec,
        is_measurement_frame=True,
        measurement_bucket=int(source_time_sec * 10),
    )


def _pose(*, frontal: bool | None) -> dict:
    points: list[list[float] | None] = [None] * 17
    scores: list[float | None] = [None] * 17
    if frontal is True:
        values = {
            0: (50, 29),  # nose
            1: (43, 25),  # left eye
            2: (57, 25),  # right eye
            3: (37, 28),  # left ear
            4: (63, 28),  # right ear
            5: (39, 47),
            6: (61, 47),
        }
    elif frontal is False:
        values = {
            0: (57, 29),  # nose moves towards the visible side
            2: (58, 25),
            4: (65, 28),
            5: (40, 47),
            6: (60, 47),
        }
    else:
        values = {}
    for index, value in values.items():
        points[index] = [float(value[0]), float(value[1])]
        scores[index] = 0.95
    return {"keypoints_image": points, "keypoint_scores": scores}


def _event(*, frontal: bool | None, confidence: float = 0.9, event_type: str = "person_observation") -> ProcessorEvent:
    return ProcessorEvent(
        event_type=event_type,
        evidence_state="detected",
        confidence=confidence,
        data={
            "track": {
                "track_id": "track_001",
                "confidence": confidence,
                "location_evidence": {"bbox_xyxy": [25, 12, 75, 112]},
                "pose": _pose(frontal=frontal),
            }
        },
    )


class CandidatePhotoCollectorTests(unittest.TestCase):
    def test_unbeatable_selection_skips_sharpness_work_without_changing_photo(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            collector = CandidatePhotoCollector(Path(temporary))
            frame = np.zeros((128, 100, 3), dtype=np.uint8)
            collector.attach(frame, _context(1), [_event(frontal=True)])
            before = (Path(temporary) / 'track_001.jpg').read_bytes()
            state = collector.snapshot_state()
            with patch.object(collector, '_sharpness', side_effect=AssertionError('unnecessary image work')):
                result = collector.attach(frame, _context(2), [_event(frontal=False, confidence=1)])[0]
            self.assertNotIn('candidate_photo', result.data)
            self.assertEqual(collector.snapshot_state(), state)
            self.assertEqual((Path(temporary) / 'track_001.jpg').read_bytes(), before)

    def test_frontal_pose_wins_over_a_higher_quality_side_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = CandidatePhotoCollector(Path(temporary))
            frontal_frame = np.full((128, 100, 3), 30, dtype=np.uint8)
            side_frame = np.full((128, 100, 3), 220, dtype=np.uint8)

            first = collector.attach(frontal_frame, _context(1.0), [_event(frontal=True)])[0]
            second = collector.attach(side_frame, _context(2.0), [_event(frontal=False, confidence=1.0)])[0]

            self.assertEqual(first.data["candidate_photo"]["view_label"], "front")
            self.assertNotIn("candidate_photo", second.data)
            stored = cv2.imread(str(Path(temporary) / "track_001.jpg"))
            self.assertIsNotNone(stored)
            self.assertLess(float(stored.mean()), 100.0)

    def test_missing_head_keypoints_falls_back_without_claiming_a_front_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = CandidatePhotoCollector(Path(temporary))
            output = collector.attach(
                np.full((128, 100, 3), 90, dtype=np.uint8),
                _context(1.0),
                [_event(frontal=None)],
            )[0]

            photo = output.data["candidate_photo"]
            self.assertEqual(photo["view_label"], "not_assessed")
            self.assertEqual(photo["frontal_score"], 0.0)

    def test_unconfirmed_roster_candidate_receives_a_review_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = CandidatePhotoCollector(Path(temporary))
            output = collector.attach(
                np.full((128, 100, 3), 90, dtype=np.uint8),
                _context(1.0),
                [_event(frontal=None, event_type="roster_candidate_observation")],
            )[0]

            self.assertIn("candidate_photo", output.data)
            self.assertTrue((Path(temporary) / "track_001.jpg").is_file())

    def test_checkpoint_preserves_the_front_priority_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector = CandidatePhotoCollector(root)
            collector.attach(np.full((128, 100, 3), 30, dtype=np.uint8), _context(1.0), [_event(frontal=True)])
            restored = CandidatePhotoCollector(root)
            restored.restore_state(collector.snapshot_state())

            later_side = restored.attach(
                np.full((128, 100, 3), 220, dtype=np.uint8),
                _context(2.0),
                [_event(frontal=False, confidence=1.0)],
            )[0]
            self.assertNotIn("candidate_photo", later_side.data)

    def test_anchored_roster_upgrades_only_unoccluded_front_facing_track(self):
        import json
        from badminton_analysis.streaming.models import FrameContext, ProcessorEvent

        def full_body_pose(x1: float, y1: float, x2: float, y2: float, *, frontal: bool):
            cx = (x1 + x2) / 2.0
            pts = [None] * 17
            scores = [None] * 17
            if frontal:
                pts[0] = [cx, y1 + 14.0]
                pts[1] = [cx - 7.0, y1 + 10.0]
                pts[2] = [cx + 7.0, y1 + 10.0]
                pts[3] = [cx - 12.0, y1 + 13.0]
                pts[4] = [cx + 12.0, y1 + 13.0]
                for idx in range(5):
                    scores[idx] = 0.95
            pts[5], pts[6] = [cx - 14.0, y1 + 30.0], [cx + 14.0, y1 + 30.0]
            pts[11], pts[12] = [cx - 12.0, y1 + 70.0], [cx + 12.0, y1 + 70.0]
            pts[15], pts[16] = [cx - 12.0, y1 + 120.0], [cx + 12.0, y1 + 120.0]
            for idx in (5, 6, 11, 12, 15, 16):
                scores[idx] = 0.95
            return pts, scores

        anchor = {
            "policy": "anchored_roster.v1",
            "track_ids": ["track_001", "track_002", "track_003", "track_004"],
            "source_frame_index": 10,
            "source_time_sec": 1.0,
            "measurement_bucket": 10,
        }

        def make_events(boxes, frontal_flags, time_sec: float):
            evs = []
            for tid, box, front in zip(anchor["track_ids"], boxes, frontal_flags):
                pts, scores = full_body_pose(*box, frontal=front)
                evs.append(
                    ProcessorEvent(
                        event_type="person_observation",
                        evidence_state="detected",
                        confidence=0.9,
                        data={
                            "track": {
                                "track_id": tid,
                                "confidence": 0.9,
                                "identity_status": "anchored",
                                "identity_anchor": anchor,
                                "location_evidence": {"bbox_xyxy": list(box)},
                                "pose": {"keypoints_image": pts, "keypoint_scores": scores},
                            }
                        },
                        source_time_sec=time_sec,
                        segment_index=0,
                    )
                )
            return evs

        sep_boxes = [[20, 20, 100, 160], [140, 20, 220, 160], [280, 20, 360, 160], [420, 20, 500, 160]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector = CandidatePhotoCollector(root, anchored_roster_count=4)
            frame0 = np.full((240, 560, 3), 80, dtype=np.uint8)
            ctx10 = FrameContext(analysis_session_id="s1", segment_index=0, source_frame_index=10, source_time_sec=1.0, is_measurement_frame=True, measurement_bucket=10)
            init_out = collector.attach(frame0, ctx10, make_events(sep_boxes, [False, False, False, False], 1.0))
            self.assertEqual([e.data["candidate_photo"]["selection_policy"] for e in init_out], ["same_frame_roster_v1"] * 4)

            # Frame 20: track_001 is front-facing BUT overlaps track_002 (>5%) -> MUST NOT upgrade.
            ov_boxes = [[20, 20, 100, 160], [75, 20, 155, 160], [280, 20, 360, 160], [420, 20, 500, 160]]
            frame1 = np.full((240, 560, 3), 210, dtype=np.uint8)
            ctx20 = FrameContext(analysis_session_id="s1", segment_index=0, source_frame_index=20, source_time_sec=2.0, is_measurement_frame=True, measurement_bucket=20)
            ov_out = collector.attach(frame1, ctx20, make_events(ov_boxes, [True, False, False, False], 2.0))
            self.assertFalse(any("candidate_photo" in e.data for e in ov_out))

            # Frame 30: separated front-facing track_001 -> upgrades ONLY track_001.
            frame2 = np.full((240, 560, 3), 230, dtype=np.uint8)
            ctx30 = FrameContext(analysis_session_id="s1", segment_index=0, source_frame_index=30, source_time_sec=3.0, is_measurement_frame=True, measurement_bucket=30)
            up_out = collector.attach(frame2, ctx30, make_events(sep_boxes, [True, False, False, False], 3.0))
            self.assertEqual(up_out[0].data["candidate_photo"]["selection_policy"], "anchored_front_upgrade_v1")
            self.assertFalse(any("candidate_photo" in e.data for e in up_out[1:]))
            manifest = json.loads((root / "anchor_set" / "manifest.json").read_text())
            self.assertEqual(manifest["photos"]["track_001"]["selection_policy"], "anchored_front_upgrade_v1")
            self.assertEqual(manifest["photos"]["track_002"]["selection_policy"], "same_frame_roster_v1")
            restored = CandidatePhotoCollector(root, anchored_roster_count=4)
            restored.restore_state(collector.snapshot_state())

    def test_non_anchored_collector_skips_overlapping_other_track(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as temporary:
            collector = CandidatePhotoCollector(Path(temporary))
            ev1 = _event(frontal=True)
            ev2 = deepcopy(ev1)
            ev2.data["track"]["track_id"] = "track_002"
            # Overlap ev2 heavily onto ev1 ([25, 12, 75, 112]).
            ev2.data["track"]["location_evidence"]["bbox_xyxy"] = [35, 12, 85, 112]
            frame = np.full((128, 128, 3), 120, dtype=np.uint8)
            out = collector.attach(frame, _context(1.0), [ev1, ev2])
            self.assertFalse(any("candidate_photo" in e.data for e in out))


if __name__ == "__main__":
    unittest.main()
