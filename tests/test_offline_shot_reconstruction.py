import copy
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.analysis.offline_shot_reconstruction import (
    build_shot_events,
    generate_offline_artifacts,
    reconstruct_shuttle_track,
    write_jsonl,
)


class OfflineShotReconstructionTests(unittest.TestCase):
    def test_reconstructs_only_a_short_bounded_gap_and_marks_long_gap_unknown(self):
        rows = [
            self._row(1, [10, 10], accepted=True, confidence=0.50),
            self._row(2, [10, 10], accepted=True, confidence=0.50),
            self._row(3, [10, 10], accepted=True, confidence=0.50),
            self._row(4, [100, 100], accepted=True, confidence=0.90),
            self._row(5, None),
            self._row(6, [140, 120], accepted=True, confidence=0.90),
            self._row(7, None),
            self._row(8, None),
            self._row(9, None),
            self._row(10, None),
            self._row(11, [500, 300], accepted=True, confidence=0.90),
        ]
        original = copy.deepcopy(rows)
        tracks = reconstruct_shuttle_track(rows, fps=10, width=640, height=480, max_gap_sec=0.20)

        self.assertEqual([item["status"] for item in tracks[:3]], ["rejected_artifact"] * 3)
        self.assertEqual(tracks[4]["status"], "reconstructed")
        self.assertEqual(tracks[4]["image_xy"], [120.0, 110.0])
        self.assertEqual(tracks[7]["status"], "unknown_gap")
        self.assertEqual(rows, original)

    def test_event_artifacts_keep_machine_candidates_out_of_statistics(self):
        rows = [
            self._row(1, [100, 100], accepted=True, confidence=0.90, hit="track_001", zone="front_center"),
            self._row(2, [120, 120], accepted=True, confidence=0.90),
            self._row(3, [180, 180], accepted=True, confidence=0.90),
            self._row(5, [250, 240], accepted=True, confidence=0.90, hit="track_002", zone="rear_center"),
        ]
        tracks = reconstruct_shuttle_track(rows, fps=10)
        events = build_shot_events(rows, tracks, fps=10)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["hitter"]["track_id"], "track_001")
        self.assertEqual(events[0]["receiver"]["track_id"], "track_002")
        self.assertEqual(events[0]["proposal"]["label"], "lift")
        self.assertFalse(events[0]["decision"]["eligible_for_statistics"])

    def test_generator_writes_separate_derived_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "detections.jsonl"
            write_jsonl(source, [self._row(1, [10, 20], accepted=True, confidence=0.8)])
            result = generate_offline_artifacts(source, fps=10)

            self.assertTrue(Path(result["tracks_path"]).is_file())
            self.assertTrue(Path(result["events_path"]).is_file())
            self.assertEqual(result["frame_count"], 1)

    @staticmethod
    def _row(frame, point, accepted=False, confidence=0.0, hit=None, zone=None):
        tracks = []
        if hit:
            tracks.append(
                {
                    "track_id": hit,
                    "status": "detected",
                    "confidence": 0.9,
                    "zone_id": zone,
                    "court_xy_m": [3.0, 3.0],
                    "location_evidence": {"hands_image": {"left": [point[0] if point else 0, point[1] if point else 0], "right": None}},
                }
            )
        return {
            "frame": frame,
            "time_sec": frame / 10,
            "shuttlecock": {
                "image": point,
                "status": "detected" if accepted else "missing",
                "accepted": accepted,
                "confidence": confidence,
            },
            "spatial": {
                "tracks": tracks,
                "hit_events": ([{"status": "candidate", "hitter_track_id": hit, "confidence": 0.4, "reason": "synthetic"}] if hit else []),
            },
        }


if __name__ == "__main__":
    unittest.main()
