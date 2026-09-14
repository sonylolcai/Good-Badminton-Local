"""2/3/4-person streaming scenarios using task E PersonOnlyFrameProcessor.

This exercises the open-set anonymous person tracker end-to-end: a synthetic
observation provider emits N persons per measurement frame, the engine tracks
each as a distinct anonymous track_id, and the replay harness records a valid
benchmark.  It does not require the real YOLO Pose provider (task F).
"""

import math
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.tracking.person_only import PersonOnlyFrameProcessor, PersonOnlyTracker

from badminton_analysis.streaming_validation import run_stream_replay, validate_trace
from tests.stream_test_utils import segment_metadata, write_video_segment_bytes


CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]


class SyntheticMultiPersonProvider:
    """Return N persons at distinct, slowly moving court positions."""

    def __init__(self, person_count):
        self.person_count = person_count

    def __call__(self, frame, context):
        t = context.source_time_sec
        observations = []
        for person in range(self.person_count):
            x = 0.8 + person * 1.4
            y = 3.0 + 2.5 * math.sin(t * 1.5 + person)
            x = min(max(x + 0.4 * math.cos(t + person), 0.4), 5.7)
            y = min(max(y, 0.5), 12.9)
            observations.append({
                "court_xy": [x, y],
                "image_xy": [x * 100.0, y * 100.0],
                "confidence": 0.9,
            })
        return observations


def person_only_factory(person_count):
    """Return a processor_factory producing a fresh tracker + processor."""

    def factory():
        tracker = PersonOnlyTracker(CORNERS, fps=30)
        provider = SyntheticMultiPersonProvider(person_count)
        return PersonOnlyFrameProcessor(tracker, provider), None

    return factory


def _config(person_count):
    return {
        "video": "synthetic-multi-person",
        "resolution": "64x64",
        "sample_hz": 10,
        "pose_imgsz": 960,
        "shuttle_detector": "none",
        "generate_annotated_video": False,
        "gpu": "cpu-test-harness",
        "driver": "n/a",
        "match_mode": "person_only",
        "person_count": person_count,
        "switches": {"tracknet": False, "annotated_video": False},
    }


class PersonOnlyStreamingPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data_dir = Path(self.temp_dir.name)
        self.segment_bytes = write_video_segment_bytes()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _segments(self, count):
        return [
            (index, self.segment_bytes, segment_metadata(index, self.segment_bytes, source_start=index * 1.0))
            for index in range(count)
        ]

    def test_two_three_four_person_replay_emits_distinct_tracks(self):
        for person_count in (2, 3, 4):
            with self.subTest(person_count=person_count):
                benchmark, manager = run_stream_replay(
                    self.data_dir / f"persons_{person_count}",
                    self._segments(3),
                    person_only_factory(person_count),
                    _config(person_count),
                )
                self.assertEqual(validate_trace(benchmark), [])
                self.assertEqual(benchmark["segments"]["count"], 3)
                status, payload = manager.read_events(benchmark["session_id"], limit=1000)
                events = payload["events"]
                track_ids = {
                    event["data"]["track"]["track_id"]
                    for event in events
                    if event["event_type"] == "person_observation"
                }
                self.assertEqual(
                    len(track_ids), person_count,
                    f"expected {person_count} distinct tracks, got {sorted(track_ids)}",
                )


if __name__ == "__main__":
    unittest.main()
