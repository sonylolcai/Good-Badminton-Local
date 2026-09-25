"""Source-frame TrackNet continuity and checkpoint behavior."""

import json
import hashlib
import unittest

import numpy as np
import torch

from api.tracknet_stream import TrackNetStreamProcessor
from badminton_analysis.streaming import AnalysisEngine, FrameContext, FramePacket, FrameSegment, SegmentDescriptor
from tests.stream_test_utils import CountingProcessor


class _Model:
    def __call__(self, tensor):
        assert tensor.shape == (1, 27, 288, 512)
        return torch.zeros((1, 8, 288, 512), dtype=torch.float32)


class _Detector:
    seq_len = 8
    device = torch.device("cpu")

    def __init__(self):
        self.model = _Model()

    def _extract_coordinate_from_heatmap(self, _heatmap, scaler):
        return True, 100.0 * scaler[0], 50.0 * scaler[1], 0.9


def _context(index, segment_index, fps=30):
    return FrameContext(
        analysis_session_id="ssn_tracknet_stream",
        segment_index=segment_index,
        source_frame_index=index,
        source_time_sec=index / fps,
        is_measurement_frame=True,
        measurement_bucket=index,
        source_frame_identity_declared=True,
    )


class TrackNetStreamTests(unittest.TestCase):
    def test_engine_persists_cross_segment_ball_evidence_in_source_order(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        def segment(index):
            start = index * 4
            descriptor = SegmentDescriptor(
                segment_index=index,
                source_start_time_sec=start / 30.0,
                duration_sec=4 / 30.0,
                sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                idempotency_key=f"segment-{index}",
                source_frame_start_index=start,
                source_frame_count=4,
            )
            return FrameSegment(descriptor, (
                FramePacket(frame, source_frame_index=number,
                            source_time_sec=number / 30.0, segment_index=index)
                for number in range(start, start + 4)
            ))

        first = AnalysisEngine(
            "ssn_tracknet_stream", 30, CountingProcessor(),
            temporal_processor=TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64),
        )
        checkpoint = first.process_segment(segment(0)).checkpoint
        restored = AnalysisEngine.restore(
            checkpoint, CountingProcessor(),
            temporal_processor=TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64),
        )
        result = restored.process_segment(segment(1))
        ball_events = [event for event in result.events if event["event_type"] == "shuttle_observation"]
        self.assertEqual(len(ball_events), 1)
        self.assertEqual(ball_events[0]["source_time_sec"], 7 / 30.0)
        self.assertEqual(ball_events[0]["segment_index"], 1)
        self.assertEqual(result.checkpoint["temporal_processor_state"]["last_frame_index"], 7)
        self.assertTrue(all(
            left["source_time_sec"] <= right["source_time_sec"]
            for left, right in zip(result.events, result.events[1:])
        ))

    def test_window_crosses_segment_and_checkpoint_without_duplicate_evidence(self):
        frame = np.zeros((1080, 2048, 3), dtype=np.uint8)
        first = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        for index in range(4):
            self.assertEqual(first.process_frame(frame, _context(index, 0)), [])
        state = first.snapshot_state()
        json.dumps(state, allow_nan=False)

        restored = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        restored.restore_state(state)
        events = []
        for index in range(4, 8):
            events.extend(restored.process_frame(frame, _context(index, 1)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "shuttle_observation")
        self.assertEqual(events[0].data["source_frame_index"], 7)
        self.assertEqual(events[0].data["measurement"]["image"], [400.0, 187.5])

    def test_4k_scaling_uses_source_dimensions(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="b" * 64)
        events = []
        for index in range(8):
            events.extend(processor.process_frame(frame, _context(index, 0)))
        self.assertEqual(events[0].data["measurement"]["image"], [750.0, 375.0])

    def test_60fps_source_frames_remain_continuous_across_segment_boundary(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="d" * 64)
        events = []
        for index in range(9):
            events.extend(processor.process_frame(
                frame, _context(index, 0 if index < 3 else 1, fps=60)
            ))
        self.assertEqual([event.event_type for event in events],
                         ["shuttle_observation", "shuttle_observation"])
        self.assertEqual([event.data["source_frame_index"] for event in events], [7, 8])

    def test_gap_resets_window_and_marks_continuity_unknown(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="c" * 64)
        for index in range(8):
            processor.process_frame(frame, _context(index, 0))
        gap = processor.process_frame(frame, _context(12, 1))
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0].data["ball_continuity"], "unknown")
        self.assertEqual(gap[0].data["reason"], "source_frame_index_gap")
        self.assertEqual(len(processor.frames), 1)

    def test_unverified_boundary_resets_window_without_claiming_continuity(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="e" * 64)
        for index in range(8):
            processor.process_frame(frame, _context(index, 0))
        unverified = FrameContext(
            analysis_session_id="ssn_tracknet_stream",
            segment_index=1,
            source_frame_index=8,
            source_time_sec=8 / 30,
            is_measurement_frame=True,
            measurement_bucket=8,
        )
        events = processor.process_frame(frame, unverified)
        self.assertEqual([event.event_type for event in events], ["session_status"])
        self.assertEqual(events[0].data["reason"], "source_frame_identity_unverified")
        self.assertEqual(len(processor.frames), 1)

    def test_restore_rejects_different_model(self):
        original = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        changed = TrackNetStreamProcessor(_Detector(), model_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "model changed"):
            changed.restore_state(original.snapshot_state())


if __name__ == "__main__":
    unittest.main()
