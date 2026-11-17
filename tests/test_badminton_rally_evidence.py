"""Regressions for sample-independent, score-free rally observations."""
import dataclasses
import math
import unittest

from badminton_analysis.sports.badminton import BadmintonRuleEngine
from badminton_analysis.sports.physics import PhysicalTrajectoryAnalyzer, TrajectoryPoint, FlightArc
from badminton_analysis.sports.observation import ObservationConfig
from badminton_analysis.sports.tactics import HandSide, HeightZone, BiomechanicalPoseAnalyzer


class RallyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = ObservationConfig.load()
        self.engine = BadmintonRuleEngine(image_size=(1000, 1000), config=self.config)
        self.analyzer = PhysicalTrajectoryAnalyzer(fps=10, image_size=(1000, 1000), config=self.config)

    @staticmethod
    def arc(start=0, end=1, start_frame=0, end_frame=10):
        return FlightArc(start_frame, end_frame, start, end, (100, 100), (200, 100),
                         None, None, 100, 0, "unknown", "unknown")

    @staticmethod
    def point(frame, x=200, y=100, **kwargs):
        return TrajectoryPoint(frame, frame / 10, x, y, **kwargs)

    def test_no_terminal_from_arc_endpoint_or_missing_ball(self):
        self.assertIsNone(self.engine.evaluate_dead_ball([], self.arc()))
        points = [self.point(i, visible=False) for i in range(10, 30)]
        self.assertIsNone(self.engine.evaluate_dead_ball(points, self.arc()))

    def test_single_slow_sample_does_not_end_rally(self):
        self.assertIsNone(self.engine.evaluate_dead_ball([self.point(10, speed_kmh=0)], self.arc()))

    def test_low_speed_and_displacement_infer_landing_without_score(self):
        points = [self.point(i) for i in range(10, 18)]
        event = self.engine.evaluate_dead_ball(points, self.arc())
        self.assertEqual(event.terminal_type, "dead_ball")
        self.assertEqual(event.evidence, "low_speed_low_displacement")
        self.assertEqual(event.landing_xy, (200, 100))
        self.assertNotIn("scoring_side", dataclasses.asdict(event))

    def test_missing_observations_reset_stationary_window(self):
        points = [self.point(10), self.point(11), self.point(12, visible=False), self.point(18), self.point(19)]
        self.assertIsNone(self.engine.evaluate_dead_ball(points, self.arc()))
        self.assertIsNone(self.engine.evaluate_dead_ball([self.point(10), self.point(18)], self.arc()))

    def test_landing_requires_calibration_for_out_call(self):
        point = self.point(10, x=200, ground_contact=True)
        self.assertEqual(self.engine.evaluate_dead_ball([point], self.arc()).terminal_type, "dead_ball")
        engine = BadmintonRuleEngine(court_polygon=[(0, 0), (150, 0), (150, 150), (0, 150)])
        self.assertEqual(engine.evaluate_dead_ball([point], self.arc()).terminal_type, "out_of_bounds")
        self.assertEqual(engine.evaluate_dead_ball([self.point(10, x=150, ground_contact=True)], self.arc()).terminal_type, "in_court_landing")
        self.assertIsNone(engine.evaluate_dead_ball([self.point(10, x=200)], self.arc()))

    def test_inferred_landing_uses_calibrated_boundary(self):
        engine = BadmintonRuleEngine(image_size=(1000, 1000), court_polygon=[(0, 0), (150, 0), (150, 150), (0, 150)])
        event = engine.evaluate_dead_ball([self.point(i) for i in range(10, 18)], self.arc())
        self.assertEqual(event.terminal_type, "out_of_bounds")
        self.assertEqual(event.evidence, "low_speed_low_displacement")

    def test_no_video_time_or_fixed_shot_count_overrides(self):
        for time in (2.7, 18.88, 29.0, 129.0):
            arc = self.arc(time, time + 6, int(time * 60), int((time + 6) * 60))
            rally = self.engine.segment_rallies([], [arc])[0]
            self.assertEqual(rally.shots, [arc])
            self.assertIsNone(rally.shot_count)
            self.assertEqual(len(rally.shots), 1)
            self.assertIsNone(self.engine.evaluate_serve_condition(arc, True))
            self.assertIsNone(rally.terminal)
            self.assertFalse(rally.start_confirmed)
            self.assertEqual(rally.interruption_reason, "end_unconfirmed")

    def test_gap_alone_does_not_split_rally(self):
        arcs = [self.arc(), self.arc(50, 51, 500, 510)]
        result = self.engine.segment_rallies([], arcs)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].terminal)

    def test_end_signal_splits_before_next_motion(self):
        points = [self.point(i) for i in range(10, 18)]
        arcs = [self.arc(), self.arc(2, 3, 20, 30)]
        result = self.engine.segment_rallies(points, arcs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].terminal.evidence, "low_speed_low_displacement")
        self.assertIsNone(result[1].terminal)
        self.assertEqual(result[1].start_frame, 20)

    def test_no_hitter_or_hand_from_rally_number(self):
        shots = [self.arc(i, i + 1, i * 10, (i + 1) * 10) for i in range(7)]
        for rally_id in (1, 2, 42):
            tactical = self.engine._build_tactical_shots(rally_id, shots, None)
            self.assertTrue(all(s.hitter == "unknown" and s.hand_side == HandSide.UNKNOWN for s in tactical))
            self.assertTrue(all(s.shot_type == "unknown" and not s.is_terminal for s in tactical))

    def test_pose_without_calibration_does_not_assume_hand(self):
        hand, height, _ = BiomechanicalPoseAnalyzer().analyze_stroke_hand(None, (0, 0))
        self.assertEqual(hand, HandSide.UNKNOWN)
        self.assertEqual(height, HeightZone.UNKNOWN)

    def test_motion_is_time_and_resolution_invariant(self):
        outputs = []
        for scale, offset in ((1, 0), (2, 29), (1, 19)):
            analyzer = PhysicalTrajectoryAnalyzer(fps=10, image_size=(1000 * scale, 1000 * scale), config=self.config)
            rows = [dict(frame_index=i, timestamp_s=offset + i / 10, visible=True,
                         x=(100 + i * 10) * scale, y=100 * scale, status="detected") for i in range(10)]
            points = analyzer.extract_trajectory_points(rows)
            arcs = analyzer.segment_flight_arcs(points)
            outputs.append([(a.start_frame, a.end_frame) for a in arcs])
            self.assertTrue(all(p.speed_kmh is None for p in points))
        self.assertEqual(outputs, [outputs[0]] * len(outputs))
        self.assertTrue(outputs[0])

    def test_predicted_and_interpolated_points_cannot_end_rally(self):
        for status in ("predicted", "interpolated"):
            rows = [dict(frame_index=i, visible=True, x=200, y=100, status=status, ground_contact=True) for i in range(20)]
            points = self.analyzer.extract_trajectory_points(rows)
            self.assertIsNone(self.engine.evaluate_dead_ball(points, None))
            self.assertEqual(self.analyzer.segment_flight_arcs(points), [])

    def test_ground_contact_closes_motion_arc(self):
        rows = [dict(frame_index=i, visible=True, x=100 + i * 10, y=100,
                     ground_contact=(i == 5)) for i in range(10)]
        points = self.analyzer.extract_trajectory_points(rows)
        arcs = self.analyzer.segment_flight_arcs(points)
        rallies = self.engine.segment_rallies(points, arcs)
        self.assertEqual(rallies[0].terminal.frame_index, 5)
        self.assertEqual(len(rallies), 2)

    def test_small_displacement_but_high_speed_is_not_landing(self):
        # Sub-pixel oscillation stays within the displacement limit but is fast.
        points = [TrajectoryPoint(i, i / 1000, 200 + i % 2, 100) for i in range(700)]
        self.assertIsNone(self.engine.evaluate_dead_ball(points, None))

    def test_low_speed_but_large_window_displacement_is_not_landing(self):
        points = [self.point(i, x=200 + i) for i in range(20)]
        # 10 px/s is below the speed limit; a half-second drift exceeds 2.83 px.
        self.assertIsNone(self.engine.evaluate_dead_ball(points, None))

    def test_low_speed_window_must_last_long_enough(self):
        points = [self.point(i) for i in range(4)]
        self.assertIsNone(self.engine.evaluate_dead_ball(points, None))

    def test_low_speed_and_displacement_thresholds_are_configurable(self):
        points = [self.point(i, x=200 + i) for i in range(10)]
        config = dataclasses.replace(self.config, stationary_radius_diagonals=0.01)
        engine = BadmintonRuleEngine(image_size=(1000, 1000), config=config)
        self.assertIsNotNone(engine.evaluate_dead_ball(points, None))
        config = dataclasses.replace(config, stationary_max_speed_diagonals_s=0.001)
        engine = BadmintonRuleEngine(image_size=(1000, 1000), config=config)
        self.assertIsNone(engine.evaluate_dead_ball(points, None))

    def test_landing_inference_is_fps_and_resolution_independent(self):
        for fps in (10, 30, 60):
            for scale in (1, 2):
                engine = BadmintonRuleEngine(image_size=(1000 * scale, 1000 * scale))
                points = [TrajectoryPoint(i, i / fps, (200 + i / fps) * scale, 100 * scale)
                          for i in range(fps + 1)]
                event = engine.evaluate_dead_ball(points, None)
                self.assertIsNotNone(event)
                self.assertEqual(event.evidence, "low_speed_low_displacement")
                self.assertGreaterEqual(event.timestamp_s, self.config.stationary_duration_s)
                self.assertLessEqual(event.timestamp_s, self.config.stationary_duration_s + 1 / fps)

    def test_hd_2k_qhd_and_4k_landing_decisions(self):
        resolutions = ((1920, 1080), (2048, 1080), (2560, 1440),
                       (3840, 2160), (4096, 2160))
        for width, height in resolutions:
            diagonal = math.hypot(width, height)
            polygon = [(0, 0), (width * 0.6, 0),
                       (width * 0.6, height), (0, height)]
            engine = BadmintonRuleEngine(image_size=(width, height), court_polygon=polygon)
            for fps in (24, 30, 60):
                with self.subTest(size=(width, height), fps=fps):
                    for x_ratio, expected in ((0.3, "in_court_landing"), (0.8, "out_of_bounds")):
                        points = [TrajectoryPoint(i, i / fps,
                                  width * x_ratio + diagonal * 0.001 * i / fps,
                                  height * 0.5) for i in range(fps + 1)]
                        event = engine.evaluate_dead_ball(points, None)
                        self.assertIsNotNone(event)
                        self.assertEqual(event.terminal_type, expected)
                        self.assertEqual(event.evidence, "low_speed_low_displacement")
                        self.assertAlmostEqual(event.timestamp_s, 0.5)

                    # Small, fast oscillation must fail the speed criterion.
                    fast = [TrajectoryPoint(i, i / fps,
                            width * 0.3 + diagonal * 0.0009 * (i % 2), height * 0.5)
                            for i in range(fps + 1)]
                    self.assertIsNone(engine.evaluate_dead_ball(fast, None))
                    # Slow drift must still fail the window displacement criterion.
                    drift = [TrajectoryPoint(i, i / fps,
                             width * 0.3 + diagonal * 0.006 * i / fps, height * 0.5)
                             for i in range(fps + 1)]
                    self.assertIsNone(engine.evaluate_dead_ball(drift, None))

    def test_bad_config_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.config, stationary_duration_s=-1)
        with self.assertRaises(ValueError):
            BadmintonRuleEngine(court_polygon=[(0, 0), (0, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
