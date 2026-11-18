"""Source-frame TrackNet batched chunking (sliding_step=8) and checkpoint behavior."""

import hashlib
import json
import unittest

import numpy as np
import torch

from api.tracknet_stream import TrackNetStreamProcessor
from badminton_analysis.streaming import (
    AnalysisEngine,
    FrameContext,
    FramePacket,
    FrameSegment,
    SegmentDescriptor,
)
from tests.stream_test_utils import CountingProcessor


class _Model:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, tensor):
        assert tensor.ndim == 4 and tensor.shape[1:] == (27, 288, 512)
        self.batch_sizes.append(int(tensor.shape[0]))
        return torch.zeros((tensor.shape[0], 8, 288, 512), dtype=torch.float32)


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
    def test_60_frame_segment_runs_in_one_batched_forward_and_emits_all_frames(self):
        detector = _Detector()
        processor = TrackNetStreamProcessor(detector, model_sha256="a" * 64)
        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(60)]
        contexts = [_context(index, 0) for index in range(60)]

        batches = processor.process_segment_frames(frames, contexts)
        events = [event for _, batch in batches for event in batch]

        # 60 = 7 * 8 + 4 -> 8 chunks in a single batched GPU call (87% fewer forwards than 60)
        self.assertEqual(detector.model.batch_sizes, [8])
        self.assertEqual(len(events), 60)
        self.assertEqual(
            [event.data["source_frame_index"] for event in events],
            list(range(60)),
        )
        self.assertEqual(processor.pending_entries, [])
        self.assertEqual(len(processor.context_frames), 7)

        # Next 2-second segment also runs in a single batched call of 8 chunks.
        frames_next = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(60)]
        contexts_next = [_context(60 + index, 1) for index in range(60)]
        events_next = [
            event
            for _, batch in processor.process_segment_frames(frames_next, contexts_next)
            for event in batch
        ]
        self.assertEqual(detector.model.batch_sizes, [8, 8])
        self.assertEqual(
            [event.data["source_frame_index"] for event in events_next],
            list(range(60, 120)),
        )

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
            return FrameSegment(
                descriptor,
                (
                    FramePacket(
                        frame,
                        source_frame_index=number,
                        source_time_sec=number / 30.0,
                        segment_index=index,
                    )
                    for number in range(start, start + 4)
                ),
            )

        first = AnalysisEngine(
            "ssn_tracknet_stream",
            30,
            CountingProcessor(),
            temporal_processor=TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64),
        )
        first_result = first.process_segment(segment(0))
        first_ball = [event for event in first_result.events if event["event_type"] == "shuttle_observation"]
        self.assertEqual([event["data"]["source_frame_index"] for event in first_ball], [0, 1, 2, 3])
        restored = AnalysisEngine.restore(
            first_result.checkpoint,
            CountingProcessor(),
            temporal_processor=TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64),
        )
        result = restored.process_segment(segment(1))
        ball_events = [event for event in result.events if event["event_type"] == "shuttle_observation"]
        self.assertEqual([event["data"]["source_frame_index"] for event in ball_events], [4, 5, 6, 7])
        self.assertEqual(result.checkpoint["temporal_processor_state"]["last_frame_index"], 7)
        self.assertTrue(
            all(
                left["source_time_sec"] <= right["source_time_sec"]
                for left, right in zip(result.events, result.events[1:])
            )
        )

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
        self.assertEqual(len(events), 8)
        self.assertTrue(all(event.event_type == "shuttle_observation" for event in events))
        self.assertEqual([event.data["source_frame_index"] for event in events], list(range(8)))
        self.assertEqual(events[0].data["measurement"]["image"], [400.0, 187.5])

    def test_4k_scaling_uses_source_dimensions(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="b" * 64)
        events = []
        for index in range(8):
            events.extend(processor.process_frame(frame, _context(index, 0)))
        self.assertEqual(len(events), 8)
        self.assertEqual(events[0].data["measurement"]["image"], [750.0, 375.0])

    def test_short_session_flushes_on_finalize_with_tail_padding(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        detector = _Detector()
        processor = TrackNetStreamProcessor(detector, model_sha256="a" * 64)
        batches = processor.process_segment_frames([frame] * 5, [_context(i, 0) for i in range(5)])
        events = [e for _, b in batches for e in b]
        self.assertEqual([e.data["source_frame_index"] for e in events], [0, 1, 2, 3, 4])
        self.assertEqual(detector.model.batch_sizes, [1])
        self.assertEqual(processor.pending_entries, [])
        self.assertEqual(processor.finalize(None), [])

    def test_short_segment_settles_before_cursor_and_survives_next_gap(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        def make_seg(index, start_frame, count):
            descriptor = SegmentDescriptor(
                segment_index=index,
                source_start_time_sec=start_frame / 30.0,
                duration_sec=count / 30.0,
                sha256=hashlib.sha256(f"{index}:{start_frame}".encode()).hexdigest(),
                idempotency_key=f"segment-{index}",
                source_frame_start_index=start_frame,
                source_frame_count=count,
            )
            return FrameSegment(
                descriptor,
                (
                    FramePacket(
                        frame,
                        source_frame_index=start_frame + offset,
                        source_time_sec=(start_frame + offset) / 30.0,
                        segment_index=index,
                    )
                    for offset in range(count)
                ),
            )

        engine = AnalysisEngine(
            "ssn_tracknet_short_seg",
            30,
            CountingProcessor(),
            temporal_processor=TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64),
        )
        res0 = engine.process_segment(make_seg(0, 0, 5))
        ball0 = [e for e in res0.events if e["event_type"] == "shuttle_observation"]
        self.assertEqual([e["data"]["source_frame_index"] for e in ball0], [0, 1, 2, 3, 4])

        # Subsequent segment starts with a source-frame gap; pre-gap ball evidence is already committed.
        res1 = engine.process_segment(make_seg(1, 20, 8))
        ball1 = [e for e in res1.events if e["event_type"] == "shuttle_observation"]
        status1 = [e for e in res1.events if e["event_type"] == "session_status"]
        self.assertEqual(status1[0]["data"]["reason"], "source_frame_index_gap")
        self.assertEqual([e["data"]["source_frame_index"] for e in ball1], list(range(20, 28)))
        self.assertTrue(all(e["source_time_sec"] >= 20 / 30.0 for e in res1.events))
        self.assertEqual(
            [e for e in engine.finalize().events if e["event_type"] == "shuttle_observation"],
            [],
        )

    def test_batch_chunks_respects_max_batch_chunks_bound(self):
        detector = _Detector()
        processor = TrackNetStreamProcessor(
            detector, model_sha256="a" * 64, max_batch_chunks=3
        )
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(60)]
        contexts = [_context(index, 0) for index in range(60)]
        batches = processor.process_segment_frames(frames, contexts)
        events = [e for _, b in batches for e in b]
        self.assertEqual(len(events), 60)
        self.assertEqual(detector.model.batch_sizes, [3, 3, 2])

    def test_restore_rejects_legacy_v1_checkpoint(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        helper = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        encoded = helper._encode_image(helper._resized_rgb(frame))
        v1_state = {
            "state_version": "tracknet-stream.v1",
            "model_sha256": "a" * 64,
            "model_checkpoint": "TrackNet_best.pt",
            "max_frame_interval_ratio": 1.5,
            "frames_png": [encoded] * 8,
            "background_png": encoded,
            "last_frame_index": 59,
            "last_source_time_sec": 59 / 30.0,
            "frame_interval_sec": 1 / 30.0,
            "frame_shape": [64, 64],
            "last_segment_index": 0,
            "last_source_frame_identity_declared": True,
        }
        restored = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "unsupported TrackNet stream checkpoint"):
            restored.restore_state(v1_state)

    def test_60fps_source_frames_remain_continuous_across_segment_boundary(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="d" * 64)
        seg0 = processor.process_segment_frames([frame] * 3, [_context(i, 0, fps=60) for i in range(3)])
        seg1 = processor.process_segment_frames([frame] * 6, [_context(i, 1, fps=60) for i in range(3, 9)])
        events = [e for _, b in (*seg0, *seg1) for e in b]
        self.assertEqual([e.event_type for e in events], ["shuttle_observation"] * 9)
        self.assertEqual([e.data["source_frame_index"] for e in events], list(range(9)))

    def test_gap_resets_window_and_marks_continuity_unknown(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="c" * 64)
        for index in range(8):
            processor.process_frame(frame, _context(index, 0))
        gap = processor.process_frame(frame, _context(12, 1))
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0].data["ball_continuity"], "unknown")
        self.assertEqual(gap[0].data["reason"], "source_frame_index_gap")
        self.assertEqual(len(processor.context_frames), 0)
        self.assertEqual(len(processor.pending_entries), 1)

    def test_timestamp_gap_resets_window_even_with_consecutive_frame_indexes(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        processor = TrackNetStreamProcessor(_Detector(), model_sha256="f" * 64)
        for index in range(8):
            processor.process_frame(frame, _context(index, 0))
        discontinuous = FrameContext(
            analysis_session_id="ssn_tracknet_stream",
            segment_index=0,
            source_frame_index=8,
            source_time_sec=8 / 30 + 0.1,
            is_measurement_frame=True,
            measurement_bucket=8,
            source_frame_identity_declared=True,
        )
        events = processor.process_frame(frame, discontinuous)
        self.assertEqual(events[0].data["ball_continuity"], "unknown")
        self.assertEqual(events[0].data["reason"], "source_timestamp_gap")

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
        self.assertEqual(len(processor.pending_entries), 1)

    def test_restore_rejects_different_model(self):
        original = TrackNetStreamProcessor(_Detector(), model_sha256="a" * 64)
        changed = TrackNetStreamProcessor(_Detector(), model_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "model changed"):
            changed.restore_state(original.snapshot_state())


if __name__ == "__main__":
    unittest.main()
