import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from badminton_analysis.detection.lightweight_tracknet import LightweightTrackNetDetector
from tools.run_lightweight_tracknet_raw import export_raw
from webui.pipeline import _prepare_tracknet_v3_raw, resolve_full_video_models


class LightweightExportTests(unittest.TestCase):
    def test_partial_tail_has_no_padded_or_duplicate_frames_even_with_wrong_container_count(self):
        class Capture:
            def __init__(self, declared): self.declared=declared; self.index=0
            def isOpened(self): return True
            def get(self, key):
                return {cv2.CAP_PROP_FRAME_COUNT:self.declared,cv2.CAP_PROP_FRAME_WIDTH:32,cv2.CAP_PROP_FRAME_HEIGHT:32}.get(key, 0)
            def read(self):
                if self.index==10:return False,None
                self.index+=1
                return True,np.zeros((32,32,3),dtype=np.uint8)
            def release(self): pass
        for declared in (7,15):
            with self.subTest(declared=declared):
                engine=object.__new__(LightweightTrackNetDetector)
                engine.bg_tensor=torch.zeros((3,288,512));engine.device=torch.device('cpu')
                engine.seq_len=8;engine.heatmap_threshold=.5;engine.subpixel_centroid=True
                engine.model=lambda batch:torch.zeros((len(batch),8,288,512))
                with patch('cv2.VideoCapture',return_value=Capture(declared)):
                    rows=engine.predict_video('whole.mp4',batch_chunks=2,interpolate_isolated_missing=False)
                self.assertEqual([r['frame_index'] for r in rows],list(range(10)))
                self.assertEqual(engine.last_decoded_frame_count,10)
                with tempfile.TemporaryDirectory() as root:
                    export_raw(rows,10,Path(root)/'raw.csv')

    def test_complete_short_batch_is_not_lost(self):
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / 'eight_frames.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (32, 32))
            for _ in range(8):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            engine = object.__new__(LightweightTrackNetDetector)
            engine.bg_tensor = torch.zeros((3, 288, 512))
            engine.device = torch.device('cpu')
            engine.seq_len = 8
            engine.heatmap_threshold = .5
            engine.subpixel_centroid = True
            engine.model = lambda batch: torch.zeros((len(batch), 8, 288, 512))
            rows = engine.predict_video(video, batch_chunks=2, interpolate_isolated_missing=False)
            self.assertEqual([r['frame_index'] for r in rows], list(range(8)))

    def test_raw_csv_refuses_interpolation_or_missing_source_frames(self):
        rows = [{'frame_index': 0, 'visible': True, 'x': 100.25, 'y': 200.5,
                 'confidence': .9, 'status': 'detected'}]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'raw.csv'
            export_raw(rows, 1, path)
            with path.open() as f:
                self.assertEqual(list(csv.DictReader(f))[0]['X'], '100.25')
            with self.assertRaises(ValueError):
                export_raw(rows, 2, path)
            with self.assertRaises(ValueError):
                export_raw([{**rows[0], 'status': 'interpolated'}], 1, path)

    def test_explicit_lightweight_runtime_uses_existing_engine_without_yolo_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / 'TrackNet.pt'
            checkpoint.touch()
            env = {'GOOD_BADMINTON_TRACKNET_RUNTIME': 'lightweight',
                   'GOOD_BADMINTON_TRACKNET_CHECKPOINT': str(checkpoint)}
            with patch.dict(os.environ, env), patch('webui.pipeline._run_tracknet_command', return_value='raw.csv') as run:
                self.assertEqual(_prepare_tracknet_v3_raw('video.mp4', root,court_corners=[[0,0],[30,0],[30,30],[0,30]]), 'raw.csv')
                self.assertIn('run_lightweight_tracknet_raw.py', run.call_args.args[0][1])
            with patch.dict(os.environ, {**env, 'GOOD_BADMINTON_TRACKNET_CHECKPOINT': str(Path(root) / 'missing.pt')}):
                with self.assertRaises(RuntimeError):
                    _prepare_tracknet_v3_raw('video.mp4', root,court_corners=[[0,0],[30,0],[30,30],[0,30]])
        config = resolve_full_video_models('badminton', {'shuttle_detector': 'tracknet_v3'})
        self.assertEqual(config['ball_observation_identity']['detector_mode'], 'badminton_tracknet_v3')
