import signal
import unittest
from unittest.mock import Mock, patch

from tools.run_local_gpu import supervise


class LocalGpuLauncherTests(unittest.TestCase):
    def test_documented_timeout_restart_is_supervised(self):
        first, second = Mock(), Mock()
        first.wait.return_value = 75
        second.wait.return_value = 0
        with patch('tools.run_local_gpu.signal.signal'), patch('tools.run_local_gpu.subprocess.Popen', side_effect=[first, second]) as start:
            self.assertEqual(supervise(['worker']), 0)
            self.assertEqual(start.call_count, 2)

    def test_other_failures_are_not_hidden_by_an_infinite_restart(self):
        process = Mock(); process.wait.return_value = 2
        with patch('tools.run_local_gpu.signal.signal'), patch('tools.run_local_gpu.subprocess.Popen', return_value=process) as start:
            self.assertEqual(supervise(['worker']), 2)
            self.assertEqual(start.call_count, 1)

    def test_stopping_the_supervisor_stops_its_worker(self):
        handlers = {}
        process = Mock(); process.poll.return_value = None
        def wait(**kwargs):
            if kwargs: return 0
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        process.wait.side_effect = wait
        with patch('tools.run_local_gpu.signal.signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), patch('tools.run_local_gpu.subprocess.Popen', return_value=process):
            with self.assertRaises(SystemExit): supervise(['worker'])
        process.terminate.assert_called_once()
