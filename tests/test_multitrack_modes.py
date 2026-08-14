import unittest

import numpy as np

from badminton_analysis.analysis.fixed_camera_match import (
    CourtMultiObjectTracker,
    CourtSpace,
    FixedCameraMatchPipeline,
)
from badminton_analysis.tracking.bytetrack_adapter import ByteTrackAdapter


class _FakeByteTracker:
    def __init__(self, _args):
        self.received = []

    def update(self, batch):
        self.received.append(batch)
        if not len(batch):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray([[10, 20, 30, 60, 7, 0.88, 0, 0]], dtype=np.float32)


class MultiTrackModeTests(unittest.TestCase):
    CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]

    def test_doubles_keeps_four_independent_tracks_and_team_is_not_court_side(self):
        pipeline = FixedCameraMatchPipeline(self.CORNERS, fps=10, match_mode="doubles")
        first = pipeline.update(
            1,
            [
                self._observation((0.8, 1.0), "pose_a"),
                self._observation((4.9, 1.3), "pose_b"),
                self._observation((1.2, 12.0), "pose_c"),
                self._observation((4.7, 11.8), "pose_d"),
            ],
            None,
        )
        self.assertEqual(first["match"]["mode"], "doubles")
        self.assertEqual(first["match"]["max_players_per_team"], 2)
        self.assertEqual(len(first["tracks"]), 4)
        track_ids = [item["track_id"] for item in first["tracks"]]
        self.assertTrue(all(item["team_id"] is None for item in first["tracks"]))

        # Pose A crossed the net in court coordinates. Its ByteTrack-like
        # association key remains durable; it must not become another person.
        second = pipeline.update(
            2,
            [
                self._observation((1.0, 11.4), "pose_a"),
                self._observation((4.8, 1.6), "pose_b"),
                self._observation((1.1, 11.7), "pose_c"),
                self._observation((4.6, 11.5), "pose_d"),
            ],
            None,
        )
        by_key = {item["association"]["key"]: item for item in second["tracks"]}
        self.assertEqual(by_key["pose_a"]["track_id"], track_ids[0])
        self.assertEqual(by_key["pose_a"]["court_end"], "end_b")
        self.assertEqual(len(second["tracks"]), 4)

        pipeline.claim_teams({track_ids[0]: "team_a", track_ids[1]: "team_a", track_ids[2]: "team_b", track_ids[3]: "team_b"})
        finished = pipeline.finalize()
        self.assertEqual(finished["team_claims"][track_ids[0]], "team_a")
        self.assertEqual(finished["match"]["max_players_per_team"], 2)

    def test_status_distinguishes_detected_predicted_and_missing(self):
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS), fps=10, max_missed_frames=2, max_retained_missing_frames=10
        )
        tracker.update(1, [self._observation((2.0, 2.0), "pose_a")])
        self.assertEqual(tracker.update(2, [])[0]["status"], "predicted")
        missing = tracker.update(4, [])[0]
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["confidence"], 0.0)

    def test_bytetrack_is_not_enabled_without_recorded_gate(self):
        with self.assertRaisesRegex(ValueError, "evaluation-gated"):
            FixedCameraMatchPipeline(self.CORNERS, fps=10, tracker_backend="bytetrack")

    def test_bytetrack_adapter_uses_official_output_index_contract(self):
        adapter = ByteTrackAdapter(fps=30, tracker_factory=_FakeByteTracker)
        keys = adapter.update([self._observation((2.0, 2.0), "pose_a")])
        self.assertEqual(keys, {0: "bytetrack_7"})

    def test_official_bytetrack_can_supply_a_persistent_association_key(self):
        available, reason = ByteTrackAdapter.availability()
        self.assertTrue(available, reason)
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=30,
            tracker_backend="bytetrack",
            enable_bytetrack=True,
        )
        first = pipeline.update(1, [self._observation((2.0, 2.0), None)], None)
        second = pipeline.update(2, [self._observation((2.1, 2.0), None)], None)
        self.assertEqual(first["tracks"][0]["association"]["source"], "bytetrack")
        self.assertEqual(first["tracks"][0]["track_id"], second["tracks"][0]["track_id"])

    @staticmethod
    def _observation(court_xy, association_key):
        return {
            "court_xy": court_xy,
            "image_xy": (court_xy[0] * 100, court_xy[1] * 100),
            "bbox_xyxy": [10, 20, 30, 60],
            "confidence": 0.9,
            "location_method": "ankles_midpoint",
            "location_confidence": 0.9,
            "source": "full_frame",
            "association_key": association_key,
        }


if __name__ == "__main__":
    unittest.main()
