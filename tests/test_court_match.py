import unittest
from unittest.mock import patch

from business_gateway.court_match import CourtMatchManager


class ReplayStub:
    def __init__(self) -> None:
        self.fixed_video_ids: list[str] = []

    def submit_fixed_video(self, fixed_video_id: str):
        self.fixed_video_ids.append(fixed_video_id)
        return {"business_task_id": f"bstr_{len(self.fixed_video_ids)}"}


class CourtMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.replay = ReplayStub()
        self.manager = CourtMatchManager(self.replay)
        self.fixed_video = patch(
            "business_gateway.court_match.get_fixed_video",
            return_value={"expected_player_count": 2},
        )
        self.fixed_video.start()

    def tearDown(self) -> None:
        self.fixed_video.stop()

    def test_one_court_has_one_waiting_match_and_two_valid_slots_start_singles(self):
        first = self.manager.join("court-1", "user-a", "left-1")
        second = self.manager.join("court-1", "user-b", "right-1")
        self.assertEqual(first["match_id"], second["match_id"])
        self.assertEqual(second["startable_count"], 2)
        self.assertEqual(second["start_label"], "开始单打比赛")

    def test_three_people_cannot_start_and_four_people_start_doubles(self):
        waiting = self.manager.join("court-1", "user-a", "left-1")
        match_id = waiting["match_id"]
        self.manager.join("court-1", "user-b", "right-1")
        self.manager.join("court-1", "user-c", "left-2")
        with self.assertRaisesRegex(ValueError, "合法的 2 人单打站位或 4 人双打站位"):
            self.manager.start(match_id, "user-a", "customer-inter-test-video-v1")

    def test_only_self_can_leave_waiting_roster_and_empty_roster_releases_court(self):
        match = self.manager.join("court-1", "user-a", "left-1")
        self.manager.join("court-1", "user-b", "right-1")
        after_leave = self.manager.leave(match["match_id"], "user-a")
        self.assertEqual(after_leave["participant_count"], 1)
        self.assertFalse(after_leave["slots"][0]["occupied"])
        cancelled = self.manager.leave(match["match_id"], "user-b")
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(self.manager.active("court-1", "user-c"))

    def test_end_releases_court_before_delivery_window(self):
        match = self.manager.join("court-1", "user-a", "left-1")
        self.manager.join("court-1", "user-b", "right-1")
        started = self.manager.start(match["match_id"], "user-a", "customer-inter-test-video-v1")
        self.assertEqual(started["status"], "playing")
        ended = self.manager.end(match["match_id"], "user-b")
        self.assertEqual(ended["status"], "score_pending")
        self.assertIsNone(self.manager.active("court-1", "user-c"))
        next_match = self.manager.join("court-1", "user-c", "left-1")
        self.assertNotEqual(next_match["match_id"], match["match_id"])


if __name__ == "__main__":
    unittest.main()
