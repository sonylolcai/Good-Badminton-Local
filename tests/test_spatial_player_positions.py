import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from business_gateway.metrics.detections_reader import (
    collect_track_position_evidence,
)
from badminton_analysis.visualization.spatial_player_positions import (
    _distance_from_measurements,
    extract_high_confidence_player_portraits,
)


class SpatialPlayerPositionsTests(unittest.TestCase):
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

    def test_high_confidence_detected_track_exports_an_anonymous_person_crop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60))
            writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            frame[10:50, 20:45] = (0, 255, 0)
            writer.write(frame)
            writer.release()
            detections = root / "detections.jsonl"
            track = self._track("track_001", (2.0, 4.0))
            track["location_evidence"]["bbox_xyxy"] = [20, 10, 45, 50]
            detections.write_text(json.dumps(self._row(2, [track])) + "\n", encoding="utf-8")

            portraits = extract_high_confidence_player_portraits(video, detections, root / "outputs")
            self.assertIn("track_001", portraits)
            crop = cv2.imread(portraits["track_001"])
            self.assertIsNotNone(crop)
            self.assertGreater(crop.shape[0], 24)
            self.assertGreater(crop.shape[1], 16)

    def test_speed_evidence_excludes_roster_rebind_without_dropping_visual_track(self):
        # A same-half doubles rebind remains useful to keep a visual identity
        # alive, but it is not evidence that a player crossed four metres in
        # half a second.  The following direct observations should still form
        # the only reported speed segment.
        points = [
            {"frame": 0, "court_xy_m": (1.0, 1.0), "association_source": "bytetrack"},
            {"frame": 5, "court_xy_m": (5.0, 1.0), "association_source": "doubles_team_side_reacquisition"},
            {"frame": 10, "court_xy_m": (5.2, 1.0), "association_source": "bytetrack"},
            {"frame": 15, "court_xy_m": (5.5, 1.0), "association_source": "bytetrack"},
        ]

        distance, accepted, excluded, reasons, seconds, peak = _distance_from_measurements(points, fps=10)

        self.assertEqual(0.3, round(distance, 3))
        self.assertEqual(1, accepted)
        self.assertEqual(2, excluded)
        self.assertEqual(2, reasons["non_direct_tracker_association"])
        self.assertEqual(0.5, seconds)
        self.assertEqual(0.6, round(peak, 3))


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
