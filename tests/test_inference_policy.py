import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.app import create_app, _parse_options
from badminton_analysis.inference_policy import inference_contract, tracknet_checkpoint, tracknet_runtime
from fastapi.testclient import TestClient
from webui.pipeline import resolve_full_video_models


class InferencePolicyTests(unittest.TestCase):
    def test_badmiton_defaults_agree_across_api_pipeline_and_health(self):
        self.assertEqual(_parse_options('{}')['shuttle_detector'], 'tracknet_v3')
        self.assertEqual(resolve_full_video_models('badminton', {})['ball_observation_identity']['detector_mode'], 'badminton_tracknet_v3')
        self.assertEqual(_parse_options('{}', sport_id='tennis')['shuttle_detector'], 'yolo')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tracknet_runtime(), 'lightweight')
        profile = inference_contract('badminton')
        self.assertEqual(profile['default_ball_detector'], 'tracknet_v3')
        batch = profile['batch']['tracknet_v3']
        stream = profile['stream']['tracknet_v3']
        self.assertEqual(batch['person_association_policy'], stream['person_association_policy'])
        self.assertEqual(batch['person_association_policy']['bytetrack_match_cost_limit'], .70)
        self.assertEqual(batch['person_association_implementation_sha256'],
                         stream['person_association_implementation_sha256'])
        self.assertNotEqual(profile['batch']['tracknet_v3']['sha256'], profile['stream']['tracknet_v3']['sha256'])

    def test_conflicting_batch_stream_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / 'a.pt', Path(root) / 'b.pt'
            a.write_bytes(b'a'); b.write_bytes(b'b')
            with patch.dict(os.environ, {'GOOD_BADMINTON_TRACKNET_CHECKPOINT': str(a),
                                        'GOOD_BADMINTON_STREAM_TRACKNET_MODEL': str(b)}):
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    tracknet_checkpoint()
                b.write_bytes(b'a')
                self.assertEqual(tracknet_checkpoint(), a.resolve())

    def test_profile_mismatch_is_rejected_before_a_job_is_created(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'GOOD_BADMINTON_API_KEY': 'test-key'}):
            app = create_app(data_dir=root, start_worker=False)
            with TestClient(app) as client:
                response = client.post('/api/v1/jobs', headers={'X-API-Key': 'test-key', 'X-Expected-Inference-Profile': 'wrong'},
                    files={'video': ('v.mp4', b'video'), 'template': ('t.jpg', b'image')},
                    data={'court_corners': '[[1,1],[2,1],[2,2],[1,2]]'})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(list((Path(root) / 'jobs').glob('*/job.json')), [])
