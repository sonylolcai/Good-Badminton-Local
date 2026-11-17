import unittest
from unittest.mock import patch

from webui.app import _review_follow_playback


def _candidate(shot_id, time_sec, rally_id, shot_index):
    return {
        "shot_id": shot_id,
        "hit_time_sec": time_sec,
        "active": True,
        "candidate_source": "trajectory_turn",
        "proposal": {"label": "clear", "confidence": 0.2},
        "review": {"decision": "pending"},
        "evidence": {
            "rally_id": rally_id,
            "shot_index_in_rally": shot_index,
        },
    }


class ReviewPlaybackSyncTests(unittest.TestCase):
    def test_playback_switches_editor_only_when_touch_changes(self):
        session = {
            "candidates": [
                _candidate("shot_0001", 1.0, "rally_0001", 1),
                _candidate("shot_0002", 3.0, "rally_0001", 2),
            ]
        }
        with patch("webui.app.create_or_load_review_session", return_value=session):
            first = _review_follow_playback("analysis", 1.1, None)
            same = _review_follow_playback("analysis", 1.3, "shot_0001")
            second = _review_follow_playback("analysis", 3.1, "shot_0001")

        self.assertEqual("shot_0001", first[3]["value"])
        self.assertEqual("shot_0002", second[3]["value"])
        self.assertIn("候选回合 1 / 1", second[2])
        self.assertEqual("update", same[3]["__type__"])

