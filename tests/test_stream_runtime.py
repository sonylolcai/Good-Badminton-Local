"""Task F production processor composition without loading real checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from api.stream_runtime import StreamProcessorFactory
from api.vision_profiles import BADMINTON_PROFILE, TENNIS_PROFILE
from badminton_analysis.streaming import FinalizationContext, FrameContext
from tests.stream_test_utils import create_request


class _PoseModel:
    def __call__(self, _frame, **_kwargs):
        points = np.zeros((2, 17, 2), dtype=float)
        points[:, :, 0] = np.asarray([[32.0], [44.0]])
        points[:, :, 1] = np.linspace(12.0, 50.0, 17)
        points[0, 15] = (28.0, 50.0)
        points[0, 16] = (36.0, 50.0)
        points[1, 15] = (40.0, 50.0)
        points[1, 16] = (48.0, 50.0)
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(
                    xy=points,
                    conf=np.full((2, 17), 0.9, dtype=float),
                ),
                boxes=SimpleNamespace(
                    xyxy=np.asarray([[22.0, 10.0, 42.0, 52.0], [34.0, 10.0, 54.0, 52.0]], dtype=float),
                    conf=np.asarray([0.95, 0.95], dtype=float),
                ),
            )
        ]


class _ArrayView:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)
        self.shape = self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _BallModel:
    def __init__(self, *, names=None, class_ids=None):
        self.names = names or {0: "badminton"}
        self.class_ids = class_ids if class_ids is not None else [0]

    def __call__(self, _frame, **_kwargs):
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xywh=_ArrayView([[32.0, 30.0, 3.0, 3.0]]),
                    conf=_ArrayView([0.8]),
                    cls=_ArrayView(self.class_ids),
                )
            )
        ]


class _TemporalProcessor:
    def process_frame(self, _frame, _context):
        return []

    def finalize(self, _context):
        return []

    def snapshot_state(self):
        return {"state_version": "test-tracknet.v1"}

    def restore_state(self, state):
        if state.get("state_version") != "test-tracknet.v1":
            raise ValueError("bad test state")


class _StableByteTracker:
    """Minimal confirmed ByteTrack stand-in used to test contract plumbing."""

    def __init__(self, _args):
        pass

    def update(self, batch):
        if not len(batch):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(
            [
                [*batch.xyxy[index].tolist(), index + 1, float(batch.conf[index]), 0, index]
                for index in range(len(batch))
            ],
            dtype=np.float32,
        )


class StreamRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(
            "os.environ",
            {
                "GOOD_BADMINTON_STREAM_TRACKER_BACKEND": "court_association",
            },
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    @staticmethod
    def context():
        return FrameContext(
            analysis_session_id="ssn_task_f_runtime",
            segment_index=0,
            source_frame_index=3,
            source_time_sec=0.1,
            is_measurement_frame=True,
            measurement_bucket=1,
        )

    def factory(self, **kwargs):
        kwargs.setdefault("ball_model_factory", lambda _path: _BallModel())
        return StreamProcessorFactory(
            self.root,
            pose_model_factory=lambda _path: _PoseModel(),
            **kwargs,
        )

    def test_none_mode_emits_real_person_pose_and_checkpoint_state(self):
        session = create_request()
        session["configuration"].update(
            {"lock_match_roster": True, "roster_stable_frames": 1}
        )
        measurement, temporal = self.factory()(session)
        events = list(
            measurement.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), self.context())
        )
        self.assertIsNone(temporal)
        self.assertTrue(any(event.event_type == "person_observation" for event in events))
        person = next(event for event in events if event.event_type == "person_observation")
        self.assertEqual(person.evidence_state, "detected")
        self.assertEqual(person.data["track"]["source_frame_index"], 3)
        self.assertEqual(person.data["tracking"]["roster"]["status"], "discovering")
        self.assertEqual(person.data["tracking"]["roster"]["minimum_player_count"], 2)
        self.assertEqual(person.data["tracking"]["roster"]["maximum_player_count"], 4)
        json.dumps(measurement.snapshot_state())

    def test_pose_weight_configuration_is_isolated_by_sport(self):
        badminton_pose = self.root / "badminton-pose.pt"
        tennis_pose = self.root / "tennis-pose.pt"
        badminton_pose.write_bytes(b"badminton pose")
        tennis_pose.write_bytes(b"tennis pose")
        paths = []
        session = create_request()
        session["configuration"].update(
            {"sport_id": "tennis", "session_mode": "singles_match"}
        )
        factory = StreamProcessorFactory(
            self.root,
            vision_profile=TENNIS_PROFILE,
            pose_model_factory=lambda path: paths.append(path) or _PoseModel(),
        )
        with patch.dict(
            "os.environ",
            {
                "GOOD_BADMINTON_STREAM_POSE_MODEL": str(badminton_pose),
                "GOOD_TENNIS_STREAM_POSE_MODEL": str(tennis_pose),
                "GOOD_TENNIS_STREAM_POSE_CONF": "0.31",
                "GOOD_TENNIS_STREAM_DEVICE": "cpu",
            },
            clear=False,
        ):
            measurement, _ = factory(session)
            events = list(
                measurement.process_frame(
                    np.zeros((64, 64, 3), dtype=np.uint8), self.context()
                )
            )

        self.assertEqual(paths, [str(tennis_pose)])
        self.assertEqual(measurement.person.observation_provider.pose.conf, 0.31)
        self.assertEqual(measurement.person.observation_provider.pose.device, "cpu")
        person = next(event for event in events if event.event_type == "person_observation")
        identity = person.data["model_identity"]
        self.assertEqual(identity["sport_id"], "tennis")
        self.assertEqual(identity["model_checkpoint"], "tennis-pose.pt")
        self.assertEqual(len(identity["model_sha256"]), 64)

        self.assertNotEqual(
            BADMINTON_PROFILE.default_pose_checkpoint,
            TENNIS_PROFILE.default_pose_checkpoint,
        )

    def test_yolo_mode_adds_shuttle_evidence_on_the_same_sample_clock(self):
        session = create_request()
        session["configuration"]["shuttle_detector"] = "yolo"
        session["configuration"]["roster_stable_frames"] = 1
        measurement, temporal = self.factory()(session)
        events = list(
            measurement.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), self.context())
        )
        self.assertIsNone(temporal)
        shuttle = next(event for event in events if event.event_type == "shuttle_observation")
        self.assertEqual(shuttle.evidence_state, "detected")
        self.assertEqual(shuttle.data["measurement_bucket"], 1)
        state = measurement.snapshot_state()
        restored, _ = self.factory()(session)
        restored.restore_state(state)

    def test_tracknet_mode_requires_and_uses_explicit_temporal_adapter(self):
        session = create_request()
        session["configuration"]["shuttle_detector"] = "tracknet_v3"
        session["configuration"]["tracknet_overlap_frames"] = 7
        missing = self.factory()
        with self.assertRaisesRegex(ValueError, "bounded-state temporal"):
            missing.validate_session_request(session)

        configured = self.factory(
            tracknet_processor_factory=lambda _session, _calibration: _TemporalProcessor()
        )
        measurement, temporal = configured(session)
        self.assertIsNotNone(measurement)
        self.assertIsInstance(temporal, _TemporalProcessor)

    def test_tennis_yolo_requires_tennis_checkpoint_and_emits_ball_observation(self):
        checkpoint = self.root / "tennis-ball-yolo.pt"
        checkpoint.write_bytes(b"test checkpoint")
        session = create_request()
        session["configuration"].update(
            {
                "sport_id": "tennis",
                "session_mode": "singles_match",
                "shuttle_detector": "yolo",
                "roster_stable_frames": 1,
            }
        )
        factory = self.factory(
            vision_profile=TENNIS_PROFILE,
            ball_model_factory=lambda _path: _BallModel(names={0: "tennis_ball"}),
        )
        with patch.dict(
            "os.environ", {"GOOD_TENNIS_STREAM_BALL_MODEL": str(checkpoint)}, clear=False
        ):
            measurement, temporal = factory(session)
            events = list(
                measurement.process_frame(
                    np.zeros((64, 64, 3), dtype=np.uint8), self.context()
                )
            )
            state = measurement.snapshot_state()
            restored, _ = factory(session)
            restored.restore_state(state)

        self.assertIsNone(temporal)
        ball = next(event for event in events if event.event_type == "ball_observation")
        self.assertEqual(ball.evidence_state, "detected")
        self.assertEqual(ball.data["sport_id"], "tennis")
        self.assertEqual(ball.data["ball_kind"], "tennis_ball")
        self.assertEqual(ball.data["model_required_class"], "tennis_ball")
        self.assertEqual(ball.data["model_identity"]["sport_id"], "tennis")
        self.assertEqual(ball.data["model_identity"]["model_kind"], "ball")
        self.assertEqual(len(ball.data["model_identity"]["model_sha256"]), 64)
        self.assertNotIn("shuttle_observation", [event.event_type for event in events])

    def test_tennis_yolo_uses_badminton_label_only_as_explicit_experiment(self):
        checkpoint = self.root / "tennis-ball-yolo.pt"
        checkpoint.write_bytes(b"test checkpoint")
        session = create_request()
        session["configuration"].update(
            {"sport_id": "tennis", "session_mode": "singles_match", "shuttle_detector": "yolo"}
        )
        factory = self.factory(vision_profile=TENNIS_PROFILE)
        with patch.dict(
            "os.environ",
            {
                "GOOD_TENNIS_STREAM_BALL_MODEL": "",
                "GOOD_TENNIS_EXPERIMENTAL_BALL_MODEL": str(checkpoint),
            },
            clear=False,
        ):
            measurement, _ = factory(session)
            events = list(measurement.process_frame(
                np.zeros((64, 64, 3), dtype=np.uint8), self.context()
            ))

        ball = next(event for event in events if event.event_type == "ball_observation")
        self.assertEqual(ball.evidence_state, "detected")
        self.assertEqual(ball.data["ball_kind"], "experimental_badminton_ball_candidate")
        self.assertEqual(ball.data["model_required_class"], "badminton")
        self.assertTrue(ball.data["experimental"])

    def test_tennis_yolo_rejects_missing_checkpoint_before_session_runs(self):
        session = create_request()
        session["configuration"].update(
            {"sport_id": "tennis", "session_mode": "singles_match", "shuttle_detector": "yolo"}
        )
        factory = self.factory(vision_profile=TENNIS_PROFILE)
        with patch.dict(
            "os.environ",
            {
                "GOOD_TENNIS_STREAM_BALL_MODEL": "",
                "GOOD_TENNIS_EXPERIMENTAL_BALL_MODEL": str(self.root / "missing.pt"),
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "experimental YOLO ball detection requires"):
                factory.validate_session_request(session)

    def test_tennis_yolo_discards_non_tennis_detection_classes(self):
        checkpoint = self.root / "tennis-ball-yolo.pt"
        checkpoint.write_bytes(b"test checkpoint")
        session = create_request()
        session["configuration"].update(
            {"sport_id": "tennis", "session_mode": "singles_match", "shuttle_detector": "yolo"}
        )
        factory = self.factory(
            vision_profile=TENNIS_PROFILE,
            ball_model_factory=lambda _path: _BallModel(
                names={0: "tennis_ball", 1: "person"}, class_ids=[1]
            ),
        )
        with patch.dict(
            "os.environ", {"GOOD_TENNIS_STREAM_BALL_MODEL": str(checkpoint)}, clear=False
        ):
            measurement, _ = factory(session)
            events = list(
                measurement.process_frame(
                    np.zeros((64, 64, 3), dtype=np.uint8), self.context()
                )
            )

        ball = next(event for event in events if event.event_type == "ball_observation")
        self.assertEqual(ball.evidence_state, "missing")
        self.assertEqual(ball.data["measurement"]["filtered_rejections"], {"unexpected_class": 1})

    def test_session_selected_bytetrack_overrides_process_default_and_locks_roster(self):
        session = create_request()
        session["configuration"].update(
            {
                "tracker_backend": "bytetrack",
                "lock_match_roster": True,
                "roster_stable_frames": 1,
            }
        )
        measurement, _ = self.factory(byte_tracker_factory=_StableByteTracker)(session)
        events = list(
            measurement.process_frame(
                np.zeros((64, 64, 3), dtype=np.uint8), self.context()
            )
        )
        person = next(event for event in events if event.event_type == "person_observation")
        self.assertEqual(person.data["track"]["association"]["key"], "bytetrack_1")
        self.assertEqual(measurement.person.tracker.tracker_backend, "bytetrack")
        self.assertTrue(measurement.person.tracker.lock_match_roster)


if __name__ == "__main__":
    unittest.main()
