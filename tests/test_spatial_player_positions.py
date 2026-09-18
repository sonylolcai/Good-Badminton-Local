import json
import tempfile
import unittest
from pathlib import Path

from business_gateway.metrics.detections_reader import (
    collect_track_position_evidence,
)
from badminton_analysis.visualization.spatial_player_positions import _track_summary


class SpatialPlayerPositionsTests(unittest.TestCase):
    def test_speed_summary_uses_only_contiguous_high_confidence_measurements(self):
        summary = _track_summary(
            {
                "usable_points": [
                    {"frame": 0, "court_xy_m": [0.0, 0.0], "detection_confidence": 0.9, "location_confidence": 0.9, "identity_confidence": 0.9},
                    {"frame": 10, "court_xy_m": [1.0, 0.0], "detection_confidence": 0.9, "location_confidence": 0.9, "identity_confidence": 0.9},
                    {"frame": 20, "court_xy_m": [3.0, 0.0], "detection_confidence": 0.9, "location_confidence": 0.9, "identity_confidence": 0.9},
                ],
                "track_rows": 3,
                "state_counts": {"detected": 3},
                "excluded": {},
            },
            source_frames=3,
            fps=30,
        )

        self.assertEqual(summary["movement_distance_m"], 3.0)
        self.assertEqual(summary["movement_time_sec"], 0.667)
        self.assertEqual(summary["movement_mean_speed_mps"], 4.5)
        self.assertEqual(summary["movement_peak_speed_mps"], 6.0)

    def test_four_tracks_are_separated_and_only_high_confidence_measurements_are_usable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.jsonl"
            detections.write_text(
                "\n".join([
                    json.dumps(self._row(1, [
                        self._track("track_001", (1.0, 1.0)),
                        self._track("track_002", (4.5, 1.5)),
                        self._track("track_003", (1.2, 12.0)),
                        self._track("track_004", (4.8, 11.5)),
                    ])),
                    json.dumps(self._row(2, [
                        self._track("track_001", (1.1, 1.1), status="predicted"),
                        self._track("track_002", (4.4, 1.6), location_confidence=0.2),
                        self._track("track_003", (1.4, 11.8), association_confidence=0.55),
                        self._track("track_004", (4.7, 11.3)),
                    ])),
                    "",
                ]),
                encoding="utf-8",
            )

            evidence = collect_track_position_evidence(detections)

        self.assertTrue(evidence["has_spatial_tracks"])
        self.assertEqual(sorted(evidence["tracks"]), ["track_001", "track_002", "track_003", "track_004"])
        self.assertEqual(len(evidence["tracks"]["track_001"]["usable_points"]), 1)
        self.assertEqual(len(evidence["tracks"]["track_002"]["usable_points"]), 1)
        self.assertEqual(len(evidence["tracks"]["track_003"]["usable_points"]), 1)
        self.assertEqual(len(evidence["tracks"]["track_004"]["usable_points"]), 2)
        self.assertEqual(evidence["tracks"]["track_001"]["excluded"]["predicted"], 1)
        self.assertEqual(evidence["tracks"]["track_002"]["excluded"]["low_location_confidence"], 1)
        self.assertEqual(evidence["tracks"]["track_003"]["excluded"]["low_identity_confidence"], 1)

    @staticmethod
    def _row(frame, tracks):
        return {"frame": frame, "spatial": {"tracks": tracks}}

    @staticmethod
    def _track(
        track_id,
        point,
        status="detected",
        location_confidence=0.9,
        association_confidence=0.9,
    ):
        return {
            "track_id": track_id,
            "court_xy_m": list(point),
            "status": status,
            "confidence": 0.9,
            "association": {"source": "court_association", "identity_confidence": association_confidence},
            "location_evidence": {"confidence": location_confidence},
        }


if __name__ == "__main__":
    unittest.main()
