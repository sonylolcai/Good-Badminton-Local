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


def _event(*, frontal: bool | None, confidence: float = 0.9) -> ProcessorEvent:
    return ProcessorEvent(
        event_type="person_observation",
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


if __name__ == "__main__":
    unittest.main()
