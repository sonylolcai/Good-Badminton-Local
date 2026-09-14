import unittest

from analysis_platform.gpu_client import _wait_for_stream_terminal
from badminton_analysis.cancellation import AnalysisCancelled


class AnalysisPlatformGpuClientTests(unittest.TestCase):
    def test_stream_wait_sends_remote_cancel_before_stopping(self):
        class Client:
            cancelled = False

            def cancel(self):
                self.cancelled = True

            def get_status(self):
                return {"status": "running"}

        client = Client()
        with self.assertRaises(AnalysisCancelled):
            _wait_for_stream_terminal(
                client,
                poll_interval_seconds=0.01,
                timeout_seconds=1,
                cancel_cb=lambda: True,
            )
        self.assertTrue(client.cancelled)


if __name__ == "__main__":
    unittest.main()
