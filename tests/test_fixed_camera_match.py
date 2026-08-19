import unittest

from badminton_analysis.analysis.fixed_camera_match import (
    CourtSpace,
    FixedCameraMatchPipeline,
    MonocularShuttleReconstructor,
)


class FixedCameraMatchTests(unittest.TestCase):
    def test_zone_is_defined_in_court_coordinates_for_two_camera_views(self):
        rear = CourtSpace([(100, 80), (540, 80), (620, 620), (20, 620)])
        oblique = CourtSpace([(280, 45), (660, 160), (530, 640), (80, 470)])

        for court in (rear, oblique):
            for point, expected_zone in [
                ((0.8, 1.0), "rear_left"),
                ((3.05, 6.7), "mid_center"),
                ((5.3, 12.4), "front_right"),
            ]:
                recovered = court.image_to_court(court.court_to_image(point))
                self.assertEqual(court.zone_for(recovered), expected_zone)

    def test_track_id_persists_while_player_crosses_any_zone(self):
        pipeline = FixedCameraMatchPipeline(
            [(0, 0), (610, 0), (610, 1340), (0, 1340)], fps=10
        )
        first = pipeline.update(
            frame_index=1,
            observations=[self._observation((0.8, 1.0))],
            shuttlecock=None,
        )
        track_id = first["tracks"][0]["track_id"]
        second = pipeline.update(
            frame_index=10,
            observations=[self._observation((3.05, 6.7))],
            shuttlecock=None,
        )
        third = pipeline.update(
            frame_index=20,
            observations=[self._observation((5.3, 12.4))],
            shuttlecock=None,
        )

        self.assertEqual(second["tracks"][0]["track_id"], track_id)
        self.assertEqual(third["tracks"][0]["track_id"], track_id)
        self.assertEqual(
            [first["tracks"][0]["zone_id"], second["tracks"][0]["zone_id"], third["tracks"][0]["zone_id"]],
            ["rear_left", "mid_center", "front_right"],
        )
        pipeline.claim_identities({track_id: "player_a"})
        self.assertEqual(pipeline.finalize()["identity_claims"][track_id], "player_a")

    def test_monocular_reconstruction_is_explicitly_low_confidence(self):
        reconstructor = MonocularShuttleReconstructor()
        result = reconstructor.update(
            frame_index=10,
            time_sec=1.0,
            court_xy=(3.0, 6.2),
            detection_confidence=0.9,
            detected=True,
        )
        self.assertEqual(result["status"], "approximate")
        self.assertEqual(result["source"], "single_view_physics_fit")
        self.assertLess(result["confidence"], 0.5)
        self.assertIn("assumptions", result)

    def test_rally_score_remains_unknown_without_terminal_evidence(self):
        pipeline = FixedCameraMatchPipeline(
            [(0, 0), (610, 0), (610, 1340), (0, 1340)], fps=10
        )
        for frame in range(1, 8):
            pipeline.update(
                frame_index=frame,
                observations=[self._observation((2.0, 2.0)), self._observation((4.0, 11.0))],
                shuttlecock={"court_xy": (3.0, 6.7), "confidence": 0.9, "detected": True},
            )
        result = pipeline.finalize()
        self.assertEqual(result["rallies"][0]["score"]["status"], "unknown")
        self.assertFalse(result["rallies"][0]["score"]["included_in_player_statistics"])
        self.assertEqual(result["player_style_inputs"][0]["analytics_scope"], "movement_and_space_only")

    @staticmethod
    def _observation(court_xy):
        return {
            "court_xy": court_xy,
            "image_xy": (court_xy[0] * 100, court_xy[1] * 100),
            "confidence": 0.9,
            "location_method": "ankles_midpoint",
            "location_confidence": 0.9,
            "source": "full_frame",
        }


if __name__ == "__main__":
    unittest.main()
