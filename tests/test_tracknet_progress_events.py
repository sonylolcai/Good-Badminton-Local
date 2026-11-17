import io
import json
import unittest
from contextlib import redirect_stdout

from webui.pipeline import _TRACKNET_EVENT_PREFIX, _forward_tracknet_output


class TrackNetProgressEventTests(unittest.TestCase):
    def test_structured_tracknet_event_becomes_a_stage_update(self):
        updates = []
        line = _TRACKNET_EVENT_PREFIX + json.dumps({
            "stage": "inference",
            "batch_completed": 16,
            "batch_total": 170,
            "elapsed_seconds": 5.8,
        })

        with redirect_stdout(io.StringIO()):
            _forward_tracknet_output(line + "\n", updates.append)

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["phase"], "tracknet_preprocessing")
        self.assertEqual(updates[0]["stage"], "tracknet.inference")
        self.assertEqual(updates[0]["stage_detail"]["batch_completed"], 16)

    def test_non_structured_tracknet_log_does_not_invent_a_stage(self):
        updates = []
        with redirect_stdout(io.StringIO()):
            _forward_tracknet_output("Decoded 2723 frames in 8.4s\n", updates.append)

        self.assertEqual(updates, [])


if __name__ == "__main__":
    unittest.main()
