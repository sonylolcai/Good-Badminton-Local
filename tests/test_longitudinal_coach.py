"""Longitudinal AI-coach contracts: baseline first, follow-up thereafter."""

import json
import tempfile
import unittest
from pathlib import Path

from business_gateway.coach.longitudinal import (
    build_confirmed_session_observation,
    build_training_plan_candidate,
    create_coach_profile,
    load_or_create_athlete_profile,
    load_coach_profile,
    record_coach_followup_from_analysis,
    save_athlete_profile,
    update_coach_profile,
)


class LongitudinalCoachTests(unittest.TestCase):
    def test_baseline_freezes_after_three_sessions_then_requires_repeated_change(self):
        profile = create_coach_profile("player_001")
        for index, mean_speed in enumerate((1.00, 0.98, 1.02), start=1):
            profile, follow_up = update_coach_profile(
                profile,
                self._observation(f"session_{index}", mean_speed),
            )

        self.assertEqual(follow_up["mode"], "baseline_established")
        self.assertEqual(profile["baseline"]["metrics"]["mean_speed_mps"]["median"], 1.0)
        self.assertEqual(len(profile["baseline"]["eligible_session_ids"]), 3)

        profile, fourth = update_coach_profile(profile, self._observation("session_4", 1.20))
        mean_comparison = self._comparison(fourth, "mean_speed_mps")
        self.assertEqual(fourth["mode"], "follow_up")
        self.assertEqual(mean_comparison["trend_status"], "single_match_variation")
        self.assertIn("下一场", fourth["focus_items"][0]["message"])

        profile, fifth = update_coach_profile(profile, self._observation("session_5", 1.22))
        mean_comparison = self._comparison(fifth, "mean_speed_mps")
        self.assertEqual(mean_comparison["trend_status"], "sustained_improved")
        self.assertEqual(profile["baseline"]["metrics"]["mean_speed_mps"]["median"], 1.0)

    def test_low_coverage_never_enters_the_baseline_or_generates_a_change_claim(self):
        profile = create_coach_profile("player_001")
        profile, follow_up = update_coach_profile(
            profile,
            self._observation("blurred_match", 1.4, coverage=0.42),
        )

        self.assertEqual(follow_up["mode"], "insufficient_evidence")
        self.assertEqual(profile["baseline"]["eligible_session_ids"], [])
        self.assertIn("measurement_coverage_below_threshold", follow_up["ineligible_reasons"])

    def test_only_a_human_confirmed_non_ambiguous_binding_can_create_a_persistent_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_a"
            run_dir.mkdir()
            (run_dir / "derived").mkdir()
            (run_dir / "match_identity_claims.json").write_text(
                json.dumps(
                    {
                        "bindings": [
                            {
                                "track_id": "track_001",
                                "person_id": "player_001",
                                "binding_source": "post_match_human_review",
                                "identity_alias": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metadata.json").write_text(
                json.dumps({"analysis_session_id": "ssn_001", "video": {"duration_sec": 600}}),
                encoding="utf-8",
            )
            (run_dir / "derived" / "player_movement_metrics_v1.json").write_text(
                json.dumps(self._metrics(1.1)), encoding="utf-8"
            )

            result = record_coach_followup_from_analysis(
                run_dir,
                person_id="player_001",
                profile_root=Path(temporary) / "coach_profiles",
                manual_context={
                    "match_context": {"coach_match_notes": "对手节奏较快，第三局需要复核回中。"}
                },
                llm_request=lambda prompt: '{"mode":"baseline_collection"}',
            )

            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["follow_up"]["mode"], "baseline_collection")
            self.assertTrue(Path(result["report_path"]).is_file())
            self.assertTrue(Path(result["training_plan_path"]).is_file())
            self.assertEqual(
                result["training_plan_candidate"]["style_update"]["status"],
                "not_changed_automatically",
            )
            profile = load_coach_profile(result["profile_path"])
            self.assertEqual(profile["session_history"][0]["track_id"], "track_001")
            self.assertEqual(
                profile["session_history"][0]["coach_match_context"]["source"],
                "coach_manual_input",
            )
            self.assertTrue(Path(result["prompt_path"]).is_file())

    def test_unconfirmed_identity_is_rejected_before_it_can_become_coaching_history(self):
        with self.assertRaisesRegex(ValueError, "confirmed"):
            build_confirmed_session_observation(
                analysis_session_id="session_1",
                person_id="player_001",
                track_id="track_001",
                movement_metrics=self._metrics(1.0),
                claim_status="pending",
            )

    def test_manual_athlete_profile_is_persisted_separately_from_visual_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "coach_profiles"
            profile_path, saved = save_athlete_profile(
                root,
                "player_001",
                {
                    "display_name": "王同学",
                    "dominant_hand": "right",
                    "primary_event": "singles",
                    "training_stage": "targeted_strengthening",
                    "technical_ratings": {"net_control": 4, "rear_court_attack": 3},
                    "tactical_ratings": {"court_awareness": 4},
                    "strengths": ["网前控制"],
                    "development_priorities": "后场突击\n第三拍衔接",
                    "current_training_goal": "提升单打后场连续进攻质量",
                    "training_progress_notes": "本周完成两次多球训练。",
                },
            )

            self.assertTrue(Path(profile_path).is_file())
            self.assertEqual(saved["source"], "coach_manual_input")
            self.assertEqual(saved["technical_ratings"]["net_control"], 4.0)
            self.assertIsNone(saved["technical_ratings"]["serve_receive"])
            self.assertEqual(saved["development_priorities"], ["后场突击", "第三拍衔接"])

            _, loaded = load_or_create_athlete_profile(root, "player_001")
            self.assertEqual(loaded["display_name"], "王同学")
            self.assertEqual(loaded["tactical_ratings"]["court_awareness"], 4.0)

    def test_training_candidates_keep_manual_ratings_and_visual_changes_separate(self):
        plan = build_training_plan_candidate(
            {
                "athlete_profile": {
                    "technical_ratings": {"net_control": 2},
                    "development_priorities": ["后场连续进攻"],
                    "play_style_notes": "以控网后场突击为主",
                },
                "follow_up": {
                    "mode": "follow_up",
                    "focus_items": [
                        {
                            "type": "sustained_watchout",
                            "metric": "mean_return_time_sec",
                            "message": "回中用时连续偏离个人基线。",
                        }
                    ],
                },
            }
        )

        self.assertEqual(plan["status"], "coach_review_required")
        self.assertEqual(plan["candidates"][0]["source"], "coach_manual_input")
        self.assertEqual(plan["style_update"]["status"], "not_changed_automatically")
        self.assertIn("逐拍球路", plan["style_update"]["message"])

    @staticmethod
    def _comparison(follow_up, key):
        return next(item for item in follow_up["comparisons"] if item["metric"] == key)

    def _observation(self, session_id, mean_speed, *, coverage=0.95):
        return build_confirmed_session_observation(
            analysis_session_id=session_id,
            person_id="player_001",
            track_id="track_001",
            movement_metrics=self._metrics(mean_speed, coverage=coverage),
            claim_status="confirmed",
            context={"camera_profile_id": "court_a", "pose_sample_hz": 10},
        )

    @staticmethod
    def _metrics(mean_speed, *, coverage=0.95):
        return {
            "schema_version": "1.0",
            "kind": "visual_track_movement_metrics",
            "match": {"mode": "singles", "video_duration_sec": 600},
            "players": [
                {
                    "track_id": "track_001",
                    "measurement_coverage": {"usable_measurement_ratio": coverage},
                    "movement": {
                        "mean_speed_mps": mean_speed,
                        "peak_speed_mps": mean_speed + 0.8,
                    },
                    "ability_scores": {
                        "returning": {"mean_return_time_sec": 2.0},
                        "endurance": {"speed_decline_ratio": 0.05},
                    },
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
