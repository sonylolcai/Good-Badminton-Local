import json
import tempfile
import unittest
from pathlib import Path

from webui.shot_review import (
    add_manual_rally_terminal,
    add_manual_candidate,
    analysis_run_label,
    candidate_at_table_row,
    candidate_table,
    create_or_load_review_session,
    find_analysis_runs,
    merge_review_candidates,
    rally_playback_state,
    reviewed_rally_table,
    save_human_review,
    split_review_candidate,
    timeline_state,
)


def _row(frame, time_sec, ball, hit_events=None):
    return {
        "frame": frame,
        "time_sec": time_sec,
        "players": {"lower": {"hands": {"right": [100, 100]}}},
        "shuttlecock": {
            "accepted": True,
            "image": ball,
            "confidence": 0.9,
        },
        "spatial": {"hit_events": hit_events or []},
    }


class ShotReviewTests(unittest.TestCase):
    def test_rally_playback_state_exposes_current_rally_and_shot_count(self):
        def candidate(shot_id, time_sec, rally_id, shot_index):
            return {
                "shot_id": shot_id,
                "hit_time_sec": time_sec,
                "active": True,
                "evidence": {
                    "rally_id": rally_id,
                    "shot_index_in_rally": shot_index,
                },
            }

        session = {
            "candidates": [
                candidate("shot_0001", 1.0, "rally_0001", 1),
                candidate("shot_0002", 2.0, "rally_0001", 2),
                candidate("shot_0003", 7.0, "rally_0002", 1),
            ]
        }

        before = rally_playback_state(session, 0.5)
        current = rally_playback_state(session, 2.1)

        self.assertEqual("before_first_touch", before["status"])
        self.assertEqual("assigned", current["status"])
        self.assertEqual("rally_0001", current["rally_id"])
        self.assertEqual(1, current["rally_number"])
        self.assertEqual(2, current["rally_count"])
        self.assertEqual(2, current["shot_index"])
        self.assertEqual(2, current["shot_count"])

    def test_review_results_sort_by_analysis_timestamp_not_review_file_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_run = root / "webui_legacy_match_20260814_231420"
            new_run = root / "20260815_000100_webui_new_match"
            old_run.mkdir()
            new_run.mkdir()
            (old_run / "detections.jsonl").write_text("{}\n", encoding="utf-8")
            (new_run / "detections.jsonl").write_text("{}\n", encoding="utf-8")
            # A later human review must not make an older analysis appear new.
            (old_run / "shot_review").mkdir()
            (old_run / "shot_review" / "annotations.jsonl").write_text("{}\n", encoding="utf-8")

            runs = find_analysis_runs(root)

            self.assertEqual([Path(item).name for item in runs], [new_run.name, old_run.name])
            self.assertTrue(analysis_run_label(new_run).startswith("20260815_000100 · 本地 ·"))

    def _write_detections(self, root):
        rows = [
            _row(100, 4.0, [100, 100], [{"status": "candidate", "hitter_track_id": "track_a", "confidence": 0.3}]),
            _row(101, 4.04, [140, 95], [{"status": "candidate", "hitter_track_id": "track_a", "confidence": 0.4}]),
            _row(110, 4.4, [360, 70]),
            _row(130, 5.2, [580, 40]),
        ]
        target = Path(root) / "detections.jsonl"
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return target

    def test_generates_deduplicated_low_confidence_review_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_detections(directory)
            session = create_or_load_review_session(directory)

            self.assertGreaterEqual(len(session["candidates"]), 1)
            candidate = next(item for item in session["candidates"] if item["candidate_source"] == "spatial_hit_candidate")
            self.assertEqual("needs_human_review", candidate["proposal"]["status"])
            self.assertLessEqual(candidate["proposal"]["confidence"], 0.55)
            self.assertTrue(candidate["evidence"]["image_plane_only"])

    def test_manual_review_is_append_only_and_raw_evidence_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            detections = self._write_detections(directory)
            before = detections.read_bytes()
            session = create_or_load_review_session(directory)
            shot_id = session["candidates"][0]["shot_id"]

            saved = save_human_review(session, shot_id, "smash", "corrected", "coach", "visible steep return")

            self.assertEqual(before, detections.read_bytes())
            candidate = saved["candidates"][0]
            self.assertEqual("smash", candidate["review"]["label"])
            annotations = Path(directory) / "shot_review" / "annotations.jsonl"
            self.assertEqual(1, len(annotations.read_text(encoding="utf-8").splitlines()))
            table_row = next(row for row in candidate_table(saved) if row[0] == shot_id)
            self.assertEqual("人工修正：杀球", table_row[-1])

    def test_lift_is_a_valid_human_shot_label(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_detections(directory)
            session = create_or_load_review_session(directory)
            shot_id = session["candidates"][0]["shot_id"]

            saved = save_human_review(session, shot_id, "lift", "corrected", "coach")

            table_row = next(row for row in candidate_table(saved) if row[0] == shot_id)
            self.assertEqual("lift", saved["candidates"][0]["review"]["label"])
            self.assertEqual("人工修正：挑球", table_row[-1])

    def test_manual_add_merge_and_split_preserve_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            detections = self._write_detections(directory)
            before = detections.read_bytes()
            session = create_or_load_review_session(directory)
            automatic = session["candidates"][0]

            session, manual = add_manual_candidate(session, 6.0, "coach")
            self.assertEqual("manual_timeline", manual["candidate_source"])
            session, merged = merge_review_candidates(session, [automatic["shot_id"], manual["shot_id"]], "coach")
            self.assertEqual("manual_merge", merged["candidate_source"])
            self.assertFalse(automatic["active"])
            self.assertFalse(manual["active"])

            session, first, second = split_review_candidate(session, merged["shot_id"], 0.2, "coach")
            self.assertFalse(merged["active"])
            self.assertTrue(first["active"])
            self.assertTrue(second["active"])
            self.assertEqual(before, detections.read_bytes())

    def test_timeline_state_selects_only_the_latest_prior_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_detections(directory)
            session = create_or_load_review_session(directory)
            candidates = sorted(session["candidates"], key=lambda item: item["hit_time_sec"])
            first_time = candidates[0]["hit_time_sec"]

            before = timeline_state(session, first_time - 0.01)
            at = timeline_state(session, first_time)

            self.assertIsNone(before["current"])
            self.assertEqual(candidates[0]["shot_id"], at["current"]["shot_id"])
            if len(candidates) > 1:
                self.assertEqual(candidates[1]["shot_id"], at["next"]["shot_id"])

    def test_table_row_resolves_to_the_matching_active_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_detections(directory)
            session = create_or_load_review_session(directory)
            expected = candidate_table(session)[0]
            candidate = candidate_at_table_row(session, 0)

            self.assertEqual(expected[0], candidate["shot_id"])
            with self.assertRaises(ValueError):
                candidate_at_table_row(session, 999)

    def test_human_terminal_facts_override_candidate_rally_display_without_changing_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            detections = self._write_detections(directory)
            before = detections.read_bytes()
            add_manual_rally_terminal(directory, 4.5, "out_of_bounds", "coach", "visible out")
            add_manual_rally_terminal(directory, 6.0, "landed_in_bounds", "coach", "visible landing")

            session = create_or_load_review_session(directory)
            state = rally_playback_state(session, 4.6)

            self.assertEqual(before, detections.read_bytes())
            self.assertEqual("human_reviewed", state["status"])
            self.assertEqual(2, state["rally_number"])
            self.assertEqual(2, state["rally_count"])
            self.assertEqual(2, len(reviewed_rally_table(session)))
