import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.streaming import (
    AnalysisEngine,
    FileReplayAdapter,
    OpenCVSegmentDecoder,
    ProcessorEvent,
    ReplayUsageError,
    SegmentDescriptor,
)


class ReplayProcessor:
    def __init__(self):
        self.measurement_frames = []

    def process_frame(self, frame, context):
        self.measurement_frames.append(context.source_frame_index)
        return [
            ProcessorEvent(
                event_type="person_observation",
                evidence_state="detected",
                confidence=0.8,
                data={"track_id": "track_001"},
            )
        ]

    def finalize(self, context):
        return []

    def snapshot_state(self):
        return {"measurement_frames": list(self.measurement_frames)}

    def restore_state(self, state):
        self.measurement_frames = [int(item) for item in state.get("measurement_frames", [])]


def write_test_video(path: Path, *, fps=30.0, frame_count=45):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer is unavailable")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class StreamingFileReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temp_dir.name) / "replay.avi"
        write_test_video(self.video_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_file_replays_in_bounded_lazy_segments(self):
        processor = ReplayProcessor()
        engine = AnalysisEngine("ssn_replay_adapter_001", 10, processor)
        results = []

        with FileReplayAdapter(self.video_path, segment_duration_sec=1.0) as replay:
            self.assertEqual(replay.total_frames, 45)
            self.assertAlmostEqual(replay.fps, 30.0, places=1)
            for segment in replay:
                results.append(engine.process_segment(segment))

        self.assertEqual([item.source_frames for item in results], [30, 15])
        self.assertEqual(sum(item.measurement_frames for item in results), 15)
        self.assertEqual(len(processor.measurement_frames), 15)
        self.assertEqual(engine.next_expected_segment_index, 2)

    def test_replay_requires_each_lazy_segment_to_be_consumed(self):
        with FileReplayAdapter(self.video_path, segment_duration_sec=1.0) as replay:
            iterator = iter(replay)
            first = next(iterator)
            self.assertEqual(first.descriptor.segment_index, 0)
            with self.assertRaises(ReplayUsageError):
                next(iterator)

    def test_independently_playable_segment_decoder_uses_absolute_source_time(self):
        digest = hashlib.sha256(self.video_path.read_bytes()).hexdigest()
        descriptor = SegmentDescriptor(
            segment_index=3,
            source_start_time_sec=6.0,
            duration_sec=1.5,
            sha256=digest,
            idempotency_key="uploaded-segment-3",
            content_type="video/mp4",
            content_length_bytes=self.video_path.stat().st_size,
        )

        segment = OpenCVSegmentDecoder().decode(self.video_path, descriptor)
        packets = list(segment.frames)

        self.assertEqual(len(packets), 45)
        self.assertEqual(packets[0].segment_index, 3)
        self.assertAlmostEqual(packets[0].source_time_sec, 6.0)
        self.assertAlmostEqual(packets[-1].source_time_sec, 6.0 + 44 / 30.0, places=3)
        self.assertEqual(packets[0].source_frame_index, 180)


if __name__ == "__main__":
    unittest.main()
