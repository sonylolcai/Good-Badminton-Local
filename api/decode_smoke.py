"""Exercise decoding and the continuous engine without claiming model inference."""

from badminton_analysis.streaming.models import ProcessorEvent


class DecodeSmokeProcessor:
    def __init__(self):
        self.count = 0

    def process_frame(self, frame, context):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("invalid decoded video frame")
        self.count += 1
        return []

    def finalize(self, context):
        return [ProcessorEvent(
            event_type="session_status", evidence_state="derived", confidence=1.0,
            data={"decode_smoke_passed": True, "measurement_frames": self.count,
                  "model_inference_tested": False, "analysis_quality_verified": False},
        )]

    def snapshot_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = int(state["count"])
