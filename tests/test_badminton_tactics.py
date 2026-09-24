"""Unit tests for Badminton 3D Spatial Zoning, Shot Classification, and Biomechanical Pose Analysis."""

import unittest
from badminton_analysis.sports.tactics import (
    BadmintonSpatialZoning,
    BadmintonShotClassifier,
    BiomechanicalPoseAnalyzer,
    TacticalNarrativeEngine,
    TacticalShot,
    CourtZone,
    DepthZone,
    LateralZone,
    HeightZone,
    HandSide,
)


class BadmintonTacticsTests(unittest.TestCase):
    def setUp(self):
        # Default zoning: net_y=1200.0, court_x_bounds=(1400.0, 3150.0), near=(1200.0, 2100.0), far=(400.0, 1200.0)
        self.zoning = BadmintonSpatialZoning(
            net_y=1200.0,
            court_x_bounds=(1400.0, 3150.0),
            near_bounds_y=(1200.0, 2100.0),
            far_bounds_y=(400.0, 1200.0),
        )
        self.classifier = BadmintonShotClassifier(self.zoning)

    def test_spatial_zoning_near_and_far(self):
        # Near court: y >= 1200
        near_rear = self.zoning.get_zone(2275.0, 1950.0, HeightZone.OVERHEAD)
        self.assertEqual(near_rear.court_side, "near")
        self.assertEqual(near_rear.depth, DepthZone.REARCOURT)
        self.assertEqual(near_rear.lateral, LateralZone.CENTER)
        self.assertEqual(near_rear.height, HeightZone.OVERHEAD)

        # Far court: y < 1200
        far_front = self.zoning.get_zone(1500.0, 1000.0, HeightZone.UNDERHAND)
        self.assertEqual(far_front.court_side, "far")
        self.assertEqual(far_front.depth, DepthZone.NET)
        self.assertEqual(far_front.lateral, LateralZone.LEFT)
        self.assertEqual(far_front.height, HeightZone.UNDERHAND)

    def test_shot_classifier_smash(self):
        # A downward smash from near rear court to far front/mid court at high speed
        shot_code, shot_cn = self.classifier.classify_shot(
            shot_index=2,
            start_xy=(2275.0, 1900.0),
            end_xy=(2275.0, 800.0),
            duration_s=0.22,
            peak_speed_kmh=165.0,
            tactical_line="straight",
            hand_side=HandSide.FOREHAND,
            height_zone=HeightZone.OVERHEAD,
            dominant_hand="right",
            is_rally_ender=True,
            terminal_type="out_of_bounds",
        )
        self.assertEqual(shot_code, "smash")
        self.assertIn("正手重杀", shot_cn)

    def test_shot_classifier_low_serve(self):
        shot_code, shot_cn = self.classifier.classify_shot(
            shot_index=1,
            start_xy=(2000.0, 1400.0),
            end_xy=(2000.0, 1050.0),
            duration_s=0.9,
            peak_speed_kmh=45.0,
            tactical_line="straight",
            hand_side=HandSide.BACKHAND,
            height_zone=HeightZone.UNDERHAND,
            dominant_hand="right",
            is_rally_starter=True,
            serve_observed=True,
        )
        self.assertEqual(shot_code, "low_serve")
        self.assertIn("反手低发球", shot_cn)

    def test_shot_classifier_lift(self):
        # Underhand lift from near front to far rearcourt
        shot_code, shot_cn = self.classifier.classify_shot(
            shot_index=3,
            start_xy=(2000.0, 1300.0),
            end_xy=(2000.0, 500.0),
            duration_s=1.2,
            peak_speed_kmh=80.0,
            tactical_line="straight",
            hand_side=HandSide.FOREHAND,
            height_zone=HeightZone.UNDERHAND,
            dominant_hand="right",
        )
        self.assertEqual(shot_code, "lift")
        self.assertIn("正手挑球", shot_cn)

    def test_narrative_engine_generation(self):
        near_rear = self.zoning.get_zone(2275.0, 1950.0)
        far_front = self.zoning.get_zone(2275.0, 1000.0)
        shots = [
            TacticalShot(
                shot_index=1,
                hitter="near_team",
                dominant_hand="right",
                hand_side=HandSide.BACKHAND,
                shot_type="low_serve",
                shot_type_cn="反手低发球",
                start_time_s=2.0,
                end_time_s=2.9,
                duration_s=0.9,
                peak_speed_kmh=45.0,
                start_xy=(2000.0, 1400.0),
                end_xy=(2000.0, 1050.0),
                start_zone=near_rear,
                target_zone=far_front,
                flight_direction="near_to_far",
                tactical_line="straight",
            ),
            TacticalShot(
                shot_index=2,
                hitter="far_team",
                dominant_hand="right",
                hand_side=HandSide.FOREHAND,
                shot_type="lift",
                shot_type_cn="正手挑球",
                start_time_s=3.0,
                end_time_s=4.2,
                duration_s=1.2,
                peak_speed_kmh=85.0,
                start_xy=(2000.0, 1050.0),
                end_xy=(2275.0, 1950.0),
                start_zone=far_front,
                target_zone=near_rear,
                flight_direction="far_to_near",
                tactical_line="straight",
            ),
        ]
        story = TacticalNarrativeEngine.generate_rally_narrative(
            rally_id=1,
            shots=shots,
            terminal={"terminal_type": "out_of_bounds", "scoring_side": "far_team"},
        )
        self.assertIn("运动片段1 (2段轨迹)", story)
        self.assertIn("近端选手反手低发球", story)
        self.assertIn("远端选手正手挑球", story)
        self.assertIn("界外落地", story)
        self.assertNotIn("得分", story)

    def test_first_segment_is_not_automatically_a_serve(self):
        code, _ = self.classifier.classify_shot(
            shot_index=1, start_xy=(2000, 1300), end_xy=(2000, 500),
            duration_s=1.2, peak_speed_kmh=80, tactical_line="unknown",
            hand_side=HandSide.UNKNOWN, height_zone=HeightZone.UNKNOWN,
            is_rally_starter=True,
        )
        self.assertNotIn("serve", code)

    def test_missing_calibration_has_unknown_zone(self):
        zone = BadmintonSpatialZoning().get_zone(2000, 1500)
        self.assertEqual(zone.name, "unknown")


if __name__ == "__main__":
    unittest.main()
