import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.app import create_app
from api.inference_lock import GPU_EXECUTION_LOCK, serialized_inference
from operator_api.main import app as legacy
from tests.stream_test_utils import counting_processor_factory

class GpuBoundaryTests(unittest.TestCase):
    def test_legacy_operator_is_retired_without_a_key_and_rejects_wrong_key(self):
        client=TestClient(legacy)
        with patch.dict(os.environ, {'GOOD_BADMINTON_LEGACY_OPERATOR_API_KEY':''}):
            self.assertEqual(client.post('/api/v1/gpu/operate', json={'operation':'stop'}).status_code,410)
        with patch.dict(os.environ, {'GOOD_BADMINTON_LEGACY_OPERATOR_API_KEY':'test-legacy-key-that-is-at-least-32chars'}):
            self.assertEqual(client.post('/api/v1/gpu/config',json={'base_url':'http://169.254.169.254'}).status_code,401)

    def test_shared_gpu_slot_serializes_workers(self):
        active=[]; overlapped=[]
        @serialized_inference
        def inference():
            if active: overlapped.append(True)
            active.append(True); time.sleep(0.01); active.pop()
        workers=[threading.Thread(target=inference) for _ in range(8)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(2)
        self.assertFalse(overlapped)

    def test_oversized_batch_request_is_rejected_before_multipart_parsing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'GOOD_BADMINTON_API_KEY':'unit-key','GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED':'0','GPU_RESULT_CALLBACK_URL':'','GOOD_BADMINTON_STREAM_MIN_FREE_BYTES':'1'}):
            app=create_app(data_dir=Path(temporary),start_worker=False,stream_processor_factory=counting_processor_factory())
            response=TestClient(app).post('/api/v1/jobs',content=b'',headers={'X-API-Key':'unit-key','Content-Length':str(2*1024**3),'Content-Type':'multipart/form-data; boundary=test'})
            self.assertEqual(response.status_code,413)
