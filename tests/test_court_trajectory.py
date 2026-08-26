import unittest

import numpy as np

from badminton_analysis.visualization.court_trajectory import CourtTrajectoryVisualizer


class CourtTrajectoryVisualizerTests(unittest.TestCase):
    def test_spatial_tracks_render_deterministically_between_detector_samples(self):
        tracks = [{
            "track_id": "track_001",
            "status": "detected",
            "court_xy_m": [3.2, 9.1],
            "trajectory_court_m": [[3.0, 9.7], [3.1, 9.4], [3.2, 9.1]],
        }]
        visualizer = CourtTrajectoryVisualizer(width=80, height=150)

        first = visualizer.draw_overlay(np.zeros((480, 854, 3), dtype=np.uint8), spatial_tracks=tracks)
        second = visualizer.draw_overlay(np.zeros((480, 854, 3), dtype=np.uint8), spatial_tracks=tracks)

        # Reusing the exact last spatial snapshot must render the same mini
        # court on each full-rate video frame, rather than blinking with the
        # lower detector sampling cadence.
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.count_nonzero(first), 0)

    def test_missing_track_does_not_draw_a_stale_current_marker(self):
        visualizer = CourtTrajectoryVisualizer(width=80, height=150)
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        rendered = visualizer.draw_overlay(
            frame,
            spatial_tracks=[{
                "track_id": "track_001",
                "status": "missing",
                "court_xy_m": [3.2, 9.1],
                "trajectory_court_m": [[3.0, 9.7], [3.1, 9.4]],
            }],
        )

        # Court lines remain; a missing player contributes no colour marker.
        yellow = np.asarray([0, 255, 255], dtype=np.uint8)
        self.assertEqual(np.count_nonzero(np.all(rendered == yellow, axis=2)), 0)


if __name__ == "__main__":
    unittest.main()
