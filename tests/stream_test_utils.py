"""Shared lightweight fixtures for stream-session integration tests.

These intentionally mirror task A's CountingProcessor so the storage layer can
be exercised end-to-end with the real AnalysisEngine and OpenCVSegmentDecoder
without loading YOLO or ByteTrack.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import cv2
import numpy as np

from badminton_analysis.streaming import ProcessorEvent


class CountingProcessor:
    """A minimal StatefulFrameProcessor whose state survives checkpoint/restore."""

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


def counting_processor_factory(track_id="track_001"):
    """Return a processor_factory producing fresh CountingProcessor instances."""

    def factory():
        return CountingProcessor(track_id=track_id), None

    return factory


def write_video_segment_bytes(frame_count=30, fps=30.0, size=(64, 64)):
    """Encode a tiny mp4 segment and return its raw bytes plus (fps, frames)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        try:
            for index in range(frame_count):
                frame = np.full((size[1], size[0], 3), index % 255, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        with open(path, "rb") as handle:
            raw = handle.read()
        return raw
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def segment_metadata(index, raw, source_start=0.0, duration=1.0, content_type="video/mp4"):
    return {
        "schema_version": "stream-session.v1",
        "segment_index": index,
        "source_start_time_sec": source_start,
        "duration_sec": duration,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "idempotency_key": f"segment-key-{index:03d}",
        "content_type": content_type,
        "content_length_bytes": len(raw),
        "court_corners": [[0, 0], [64, 0], [64, 64], [0, 64]],
    }


def create_request():
    return {
        "schema_version": "stream-session.v1",
        "camera_id": "court-01-camera-a",
        "calibration_id": "court-01-calibration-20260823",
        "court_corners": [[0, 0], [64, 0], [64, 64], [0, 64]],
        "analysis_mode": "person_only",
        "client_reference": "opaque-business-session",
        "configuration": {
            "analysis_sample_hz": 10,
            "pose_imgsz": 960,
            "shuttle_detector": "none",
            "generate_annotated_video": False,
            "preserve_audio": False,
            "court_health_check_hz": 2,
        },
    }
