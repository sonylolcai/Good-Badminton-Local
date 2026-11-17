import json
import unittest

import numpy as np

from badminton_analysis.streaming import (
    AnalysisEngine,
    EngineNotRunnableError,
    FramePacket,
    FrameSegment,
    ProcessorEvent,
    SegmentConflictError,
    SegmentDescriptor,
    SegmentOrderError,
)


class CountingProcessor:
    def __init__(self, *, track_id="track_001", fail_restore=False):
        self.track_id = track_id
        self.count = 0
        self.finalized = False
        self.fail_restore = fail_restore

    def process_frame(self, frame, context):
        self.count += 1
        return [
            ProcessorEvent(
                event_type="person_observation",
                evidence_state="detected",
                confidence=0.9,
                data={
                    "track_id": self.track_id,
                    "processor_count": self.count,
                    "measurement_bucket": context.measurement_bucket,
                },
            )
        ]

    def finalize(self, context):
        self.finalized = True
        return [
            ProcessorEvent(
                event_type="session_status",
                evidence_state="derived",
                confidence=1.0,
                data={"processor_finalized": True, "count": self.count},
            )
        ]

    def snapshot_state(self):
        return {
            "track_id": self.track_id,
            "count": self.count,
            "finalized": self.finalized,
        }

    def restore_state(self, state):
        if self.fail_restore:
            raise RuntimeError("deliberate restore failure")
        self.track_id = str(state["track_id"])
        self.count = int(state["count"])
        self.finalized = bool(state["finalized"])


def make_segment(segment_index, start_time, frame_count=30, fps=30.0, digest_char="a"):
    descriptor = SegmentDescriptor(
        segment_index=segment_index,
        source_start_time_sec=start_time,
        duration_sec=frame_count / fps,
        sha256=digest_char * 64,
        idempotency_key=f"segment-{segment_index}",
        content_length_bytes=frame_count,
    )

    def frames():
        start_frame = int(round(start_time * fps))
        for local_index in range(frame_count):
            yield FramePacket(
                frame=np.full((4, 4, 3), local_index, dtype=np.uint8),
                source_frame_index=start_frame + local_index,
                source_time_sec=start_time + local_index / fps,
                segment_index=segment_index,
            )

    return FrameSegment(descriptor=descriptor, frames=frames())


class StreamingAnalysisEngineTests(unittest.TestCase):
    def test_measurement_processor_uses_one_shared_10hz_schedule_across_segments(self):
        measurement = CountingProcessor()
        temporal = CountingProcessor(track_id="temporal_stream")
        engine = AnalysisEngine(
            "ssn_engine_schedule_001",
            10,
            measurement,
            temporal_processor=temporal,
        )

        first = engine.process_segment(make_segment(0, 0.0, digest_char="a"))
        second = engine.process_segment(make_segment(1, 1.0, digest_char="b"))

        self.assertEqual(first.source_frames + second.source_frames, 60)
        self.assertEqual(first.measurement_frames + second.measurement_frames, 20)
        self.assertEqual(measurement.count, 20)
        self.assertEqual(temporal.count, 60)
        self.assertEqual(engine.progress()["next_expected_segment_index"], 2)
        measurement_events = [
            event for event in first.events if event["data"]["track_id"] == "track_001"
        ]
        self.assertEqual(len(measurement_events), 10)

    def test_checkpoint_restore_keeps_processor_and_event_identity_continuity(self):
        original_processor = CountingProcessor()
        original = AnalysisEngine("ssn_engine_restore_001", 10, original_processor)
        first = original.process_segment(make_segment(0, 0.0, digest_char="a"))
        json.dumps(first.checkpoint, allow_nan=False)

        restored_processor = CountingProcessor()
        restored = AnalysisEngine.restore(first.checkpoint, restored_processor)
        second = restored.process_segment(make_segment(1, 1.0, digest_char="b"))

        self.assertEqual(restored.status, "running")
        self.assertEqual(restored_processor.track_id, "track_001")
        self.assertEqual(restored_processor.count, 20)
        self.assertEqual(second.events[0]["data"]["processor_count"], 11)
        self.assertNotEqual(first.events[-1]["event_id"], second.events[0]["event_id"])
        self.assertGreaterEqual(second.events[0]["source_time_sec"], first.events[-1]["source_time_sec"])

    def test_duplicate_is_idempotent_but_digest_conflict_is_rejected(self):
        engine = AnalysisEngine("ssn_engine_idempotency_001", 10, CountingProcessor())
        engine.process_segment(make_segment(0, 0.0, digest_char="a"))

        duplicate = engine.process_segment(make_segment(0, 0.0, digest_char="a"))
        self.assertTrue(duplicate.reused)
        self.assertEqual(duplicate.events, ())

        with self.assertRaises(SegmentConflictError):
            engine.process_segment(make_segment(0, 0.0, digest_char="f"))

    def test_gap_is_rejected_before_any_processor_state_changes(self):
        processor = CountingProcessor()
        engine = AnalysisEngine("ssn_engine_gap_0001", 10, processor)

        with self.assertRaises(SegmentOrderError):
            engine.process_segment(make_segment(1, 1.0, digest_char="b"))

        self.assertEqual(processor.count, 0)
        self.assertEqual(engine.status, "running")
        self.assertEqual(engine.next_expected_segment_index, 0)

    def test_restore_failure_is_explicit_and_never_silently_restarts_tracks(self):
        engine = AnalysisEngine("ssn_engine_restore_failure", 10, CountingProcessor())
        checkpoint = engine.process_segment(make_segment(0, 0.0, digest_char="a")).checkpoint

        restored = AnalysisEngine.restore(
            checkpoint,
            CountingProcessor(fail_restore=True),
        )

        self.assertEqual(restored.status, "interrupted_needs_rebuild")
        self.assertIn("deliberate restore failure", restored.restore_error)
        with self.assertRaises(EngineNotRunnableError):
            restored.process_segment(make_segment(1, 1.0, digest_char="b"))

    def test_invalid_checkpoint_configuration_fails_closed(self):
        engine = AnalysisEngine("ssn_engine_bad_checkpoint", 10, CountingProcessor())
        checkpoint = engine.checkpoint()
        checkpoint["analysis_sample_hz"] = 12

        restored = AnalysisEngine.restore(checkpoint, CountingProcessor())

        self.assertEqual(restored.status, "interrupted_needs_rebuild")
        self.assertIn("analysis_sample_hz", restored.restore_error)

    def test_finalize_is_idempotent_and_closes_the_engine(self):
        processor = CountingProcessor()
        engine = AnalysisEngine("ssn_engine_finalize_001", 10, processor)
        engine.process_segment(make_segment(0, 0.0, digest_char="a"))

        finalized = engine.finalize()
        repeated = engine.finalize()

        self.assertEqual(finalized.status, "finalized")
        self.assertTrue(processor.finalized)
        self.assertEqual(finalized.events[-1]["event_type"], "session_finalized")
        self.assertEqual(repeated.events, ())
        with self.assertRaises(EngineNotRunnableError):
            engine.process_segment(make_segment(1, 1.0, digest_char="b"))


if __name__ == "__main__":
    unittest.main()
