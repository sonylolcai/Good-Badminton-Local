import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from badminton_analysis.tracknet.fast_predict_tracknet_v3 import _iter_batches, _iter_stream_batches
from badminton_analysis.tracknet.run_tracknet_v3 import _run_predict


class TrackNetRuntimeTests(unittest.TestCase):
    def test_fast_predictor_receives_hyphenated_cli_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "match.mp4"
            checkpoint = root / "TrackNet_best.pt"
            predictor = root / "fast_predict_tracknet_v3.py"
            for path in (video, checkpoint, predictor):
                path.write_bytes(b"fixture")
            args = Namespace(
                tracknet_python="python3",
                fast_predictor=predictor,
                tracknet_root=root,
                video=video,
                tracknet_checkpoint=checkpoint,
                batch_size=16,
                eval_mode="weight",
                background_sample_count=120,
            )
            with patch("badminton_analysis.tracknet.run_tracknet_v3.subprocess.run") as process_run:
                _run_predict(args, root / "output", inpaint_checkpoint=None)

        command = process_run.call_args.args[0]
        self.assertIn("--batch-size", command)
        self.assertIn("--eval-mode", command)
        self.assertNotIn("--batch_size", command)
        self.assertNotIn("--eval_mode", command)
        python_path = process_run.call_args.kwargs["env"]["PYTHONPATH"]
        self.assertEqual(root.name, Path(python_path.split(os.pathsep)[0]).name)

    def test_fast_predictor_uses_only_valid_sliding_windows(self):
        processed = np.zeros((10, 3, 2, 2), dtype=np.uint8)
        batches = list(_iter_batches(np, processed, None, sequence_length=8, batch_size=2))
        indexes = np.concatenate([item for item, _ in batches], axis=0)
        self.assertEqual((3, 8, 2), indexes.shape)
        self.assertEqual(list(range(2, 10)), indexes[-1, :, 1].tolist())

    def test_bounded_batches_match_full_buffer(self):
        processed = np.zeros((10, 3, 2, 2), dtype=np.uint8)
        chunks = iter(((0, processed[:3]), (3, processed[3:7]), (7, processed[7:])))
        streamed = list(_iter_stream_batches(np, chunks, None, sequence_length=8, batch_size=2, frame_count=10))
        full = list(_iter_batches(np, processed, None, sequence_length=8, batch_size=2))
        self.assertTrue(np.array_equal(
            np.concatenate([inputs for _, inputs in streamed], axis=0),
            np.concatenate([inputs for _, inputs in full], axis=0),
        ))


if __name__ == "__main__":
    unittest.main()
