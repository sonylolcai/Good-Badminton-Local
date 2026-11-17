import unittest

from evaluation.doubles.metrics import evaluate_doubles_tracking


class DoublesMetricTests(unittest.TestCase):
    def test_tracks_four_people_across_two_camera_view_labels(self):
        annotations = [
            self._annotation(1, "rear", []),
            self._annotation(2, "rear", ["p2"]),
            self._annotation(3, "oblique", []),
        ]
        predictions = [
            self._prediction(1, {"p1": "t1", "p2": "t2", "p3": "t3", "p4": "t4"}),
            self._prediction(2, {"p1": "t1", "p3": "t3", "p4": "t4"}),
            self._prediction(3, {"p1": "t1", "p2": "t2", "p3": "t3", "p4": "t4"}, hitter="t3"),
        ]
        result = evaluate_doubles_tracking(annotations, predictions)
        self.assertEqual(result["player_recall"], 1.0)
        self.assertEqual(result["id_switches"], 0)
        self.assertEqual(result["occlusion_recovery_rate"], 1.0)
        self.assertEqual(result["hit_attribution_accuracy"], 1.0)

    @staticmethod
    def _annotation(frame, view_id, occluded):
        points = {"p1": [1.0, 2.0], "p2": [5.0, 2.0], "p3": [1.0, 11.0], "p4": [5.0, 11.0]}
        return {
            "frame": frame,
            "view_id": view_id,
            "players": [{"person_id": key, "court_xy_m": value} for key, value in points.items()],
            "occluded_person_ids": occluded,
            "hits": ([{"person_id": "p3"}] if frame == 3 else []),
        }

    @staticmethod
    def _prediction(frame, mapping, hitter=None):
        positions = {"p1": [1.0, 2.0], "p2": [5.0, 2.0], "p3": [1.0, 11.0], "p4": [5.0, 11.0]}
        inverse = {track_id: person_id for person_id, track_id in mapping.items()}
        tracks = [
            {"track_id": track_id, "status": "detected", "court_xy_m": positions[person_id]}
            for track_id, person_id in inverse.items()
        ]
        events = ([{"status": "candidate", "hitter_track_id": hitter, "confidence": 0.4}] if hitter else [])
        return {"frame": frame, "spatial": {"tracks": tracks, "hit_events": events}}


if __name__ == "__main__":
    unittest.main()
