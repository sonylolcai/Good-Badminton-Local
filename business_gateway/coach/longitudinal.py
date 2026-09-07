"""Longitudinal, evidence-bound coaching follow-ups.

An initial sample of full, comparable matches establishes a *frozen* movement
baseline for a human-confirmed player.  Later matches are deliberately not
sent through a generic whole-career diagnosis: the follow-up contains only
measured changes relative to that baseline, whether a change repeats across
matches, and one or two next-match attention points.

This is a business-side concern.  Inputs are completed analysis artifacts and
post-match identity bindings; raw video, models, and GPU clients do not enter
this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib import error, request


SCHEMA_VERSION = "coach-longitudinal.v1"
REPORT_SCHEMA_VERSION = "coach-followup-report.v1"
MIN_BASELINE_SESSIONS = 3
MIN_MEASUREMENT_COVERAGE = 0.75
MIN_MATCH_DURATION_SEC = 180.0
RECENT_CONFIRMATION_SESSIONS = 2
MAX_FOCUS_ITEMS = 2

_RATING_KEYS = {
    "technical": (
        "serve_receive",
        "net_control",
        "rear_court_attack",
        "defensive_stability",
        "shot_consistency",
    ),
    "tactical": (
        "shot_selection",
        "court_awareness",
        "pressure_building",
        "adaptation",
    ),
}
_TRAINING_STAGES = {
    "assessment",
    "technique_foundation",
    "targeted_strengthening",
    "pre_competition",
    "maintenance",
}
_PRIMARY_EVENTS = {"singles", "doubles", "mixed", "unknown"}
_DOMINANT_HANDS = {"right", "left", "unknown"}
_MANUAL_RATING_LABELS = {
    "serve_receive": "发接发",
    "net_control": "网前控制",
    "rear_court_attack": "后场进攻",
    "defensive_stability": "防守稳定性",
    "shot_consistency": "击球一致性",
    "shot_selection": "球路选择",
    "court_awareness": "场地意识",
    "pressure_building": "施压组织",
    "adaptation": "临场调整",
}

# These are movement-observation deltas, not universal ability grades.  A
# single match can differ because of opponent, score, fatigue or tactics; a
# signal becomes a trend only after it repeats in comparable matches.
_METRIC_SPECS = (
    {
        "key": "mean_speed_mps",
        "label": "平均移动速度",
        "unit": "m/s",
        "direction": "higher",
        "relative_threshold": 0.08,
    },
    {
        "key": "peak_speed_mps",
        "label": "峰值移动速度",
        "unit": "m/s",
        "direction": "higher",
        "relative_threshold": 0.10,
    },
    {
        "key": "mean_return_time_sec",
        "label": "回到本方中场等候区的平均用时",
        "unit": "s",
        "direction": "lower",
        "relative_threshold": 0.10,
    },
    {
        "key": "late_speed_decline_ratio",
        "label": "后段移动速度下降比例",
        "unit": "ratio",
        "direction": "lower",
        "relative_threshold": 0.15,
    },
)


def create_coach_profile(
    person_id,
    *,
    sport_id="badminton",
    subjective_ratings=None,
    tactical_preferences=None,
):
    """Create a profile whose human inputs remain separate from CV evidence."""

    person_id = _required_identifier(person_id, "person_id")
    sport_id = _required_identifier(sport_id, "sport_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "longitudinal_coach_profile",
        "person_id": person_id,
        "sport_id": sport_id,
        "manual_coach_context": {
            "subjective_ratings": _mapping_or_empty(subjective_ratings),
            "tactical_preferences": _mapping_or_empty(tactical_preferences),
            "policy": (
                "Manual coach context is not visual measurement. It provides "
                "coaching context only and must retain its human source."
            ),
        },
        "athlete_profile": _default_athlete_profile(),
        "baseline": {
            "status": "collecting",
            "required_eligible_sessions": MIN_BASELINE_SESSIONS,
            "minimum_measurement_coverage": MIN_MEASUREMENT_COVERAGE,
            "minimum_match_duration_sec": MIN_MATCH_DURATION_SEC,
            "eligible_session_ids": [],
            "context": None,
            "metrics": {},
        },
        "session_history": [],
        "last_follow_up": None,
        "policy": {
            "identity": "Only a human-confirmed person_id to track_id binding may enter the profile.",
            "baseline": "The first eligible comparable sessions establish a frozen baseline.",
            "follow_up": (
                "Later reports focus on material relative changes. A one-match "
                "difference is labelled as a variation until it repeats."
            ),
            "limits": (
                "Movement evidence alone cannot prove stroke technique, tactical "
                "intent, score, error attribution, injury or a universal skill grade."
            ),
        },
        "updated_at": _utc_timestamp(),
    }


def build_confirmed_session_observation(
    *,
    analysis_session_id,
    person_id,
    track_id,
    movement_metrics,
    metadata=None,
    claim_status="confirmed",
    context=None,
):
    """Extract one player observation only after a post-match confirmation."""

    if str(claim_status or "").strip().lower() != "confirmed":
        raise ValueError("A confirmed post-match identity binding is required for coaching history.")
    analysis_session_id = _required_identifier(analysis_session_id, "analysis_session_id")
    person_id = _required_identifier(person_id, "person_id")
    track_id = _required_identifier(track_id, "track_id")
    metrics = _mapping_or_empty(movement_metrics)
    player = next(
        (
            item
            for item in metrics.get("players") or []
            if isinstance(item, dict) and str(item.get("track_id")) == track_id
        ),
        None,
    )
    if player is None:
        raise ValueError(f"Movement metrics do not contain confirmed track_id {track_id!r}.")

    movement = _mapping_or_empty(player.get("movement"))
    returning = _mapping_or_empty(_mapping_or_empty(player.get("ability_scores")).get("returning"))
    endurance = _mapping_or_empty(_mapping_or_empty(player.get("ability_scores")).get("endurance"))
    coverage = _finite_or_none(
        _mapping_or_empty(player.get("measurement_coverage")).get("usable_measurement_ratio")
    )
    match = _mapping_or_empty(metrics.get("match"))
    metadata = _mapping_or_empty(metadata)
    video = _mapping_or_empty(metadata.get("video"))
    duration = _finite_or_none(match.get("video_duration_sec"))
    if duration is None:
        duration = _finite_or_none(video.get("duration_sec"))

    metric_values = {
        "mean_speed_mps": _finite_or_none(movement.get("mean_speed_mps")),
        "peak_speed_mps": _finite_or_none(movement.get("peak_speed_mps")),
        "mean_return_time_sec": _finite_or_none(returning.get("mean_return_time_sec")),
        "late_speed_decline_ratio": _finite_or_none(endurance.get("speed_decline_ratio")),
    }
    normalized_context = _measurement_context(
        sport_id=(context or {}).get("sport_id") or metadata.get("sport_id") or "badminton",
        match_mode=(context or {}).get("match_mode") or match.get("mode"),
        camera_profile_id=(context or {}).get("camera_profile_id"),
        coordinate_system=(context or {}).get("coordinate_system") or "court_xy_m",
        pose_sample_hz=(context or {}).get("pose_sample_hz")
        or _mapping_or_empty(_mapping_or_empty(metadata.get("models")).get("pose")).get(
            "effective_sample_hz"
        ),
    )
    eligible_reasons = []
    if coverage is None or coverage < MIN_MEASUREMENT_COVERAGE:
        eligible_reasons.append("measurement_coverage_below_threshold")
    if duration is None or duration < MIN_MATCH_DURATION_SEC:
        eligible_reasons.append("match_duration_below_threshold")
    if not any(value is not None for value in metric_values.values()):
        eligible_reasons.append("no_supported_movement_metric")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "confirmed_player_match_observation",
        "analysis_session_id": analysis_session_id,
        "person_id": person_id,
        "track_id": track_id,
        "identity": {
            "claim_status": "confirmed",
            "source": "post_match_human_review",
            "policy": "The visual track_id remains immutable and is not replaced by person_id.",
        },
        "measurement_context": normalized_context,
        "quality": {
            "usable_measurement_ratio": coverage,
            "video_duration_sec": duration,
            "eligible_for_baseline": not eligible_reasons,
            "ineligible_reasons": eligible_reasons,
        },
        "movement_observations": metric_values,
        "source": {
            "movement_metrics_kind": metrics.get("kind"),
            "movement_metrics_schema_version": metrics.get("schema_version"),
        },
        "recorded_at": _utc_timestamp(),
    }


def update_coach_profile(profile, observation, *, manual_context=None):
    """Append or replace one observation and return a bounded follow-up.

    The baseline freezes when it reaches its first sufficient, compatible
    sample.  This avoids silently moving the goalposts after a player changes
    training or form.
    """

    profile = copy.deepcopy(_mapping_or_empty(profile))
    observation = copy.deepcopy(_mapping_or_empty(observation))
    person_id = _required_identifier(profile.get("person_id"), "profile.person_id")
    if person_id != _required_identifier(observation.get("person_id"), "observation.person_id"):
        raise ValueError("Observation person_id does not match the coach profile.")
    if str(observation.get("identity", {}).get("claim_status")) != "confirmed":
        raise ValueError("Only confirmed observations can update a coach profile.")

    _merge_manual_context(profile, manual_context)
    history = [
        item
        for item in profile.get("session_history") or []
        if isinstance(item, dict)
        and str(item.get("analysis_session_id")) != str(observation.get("analysis_session_id"))
    ]
    history.append(observation)
    profile["session_history"] = history
    baseline = _ensure_baseline_shape(profile)
    baseline_just_established = False
    if baseline.get("status") != "established":
        eligible = [
            item
            for item in history
            if _is_baseline_eligible(item)
            and _contexts_match(item.get("measurement_context"), observation.get("measurement_context"))
        ]
        baseline["eligible_session_ids"] = [item["analysis_session_id"] for item in eligible]
        baseline["context"] = copy.deepcopy(observation.get("measurement_context"))
        if len(eligible) >= MIN_BASELINE_SESSIONS:
            selected = eligible[:MIN_BASELINE_SESSIONS]
            baseline.update(
                {
                    "status": "established",
                    "established_at": _utc_timestamp(),
                    "eligible_session_ids": [item["analysis_session_id"] for item in selected],
                    "metrics": _baseline_metrics(selected),
                }
            )
            baseline_just_established = True
        else:
            baseline["status"] = "collecting"
            baseline["metrics"] = {}

    follow_up = _build_follow_up(
        profile,
        observation,
        baseline_just_established=baseline_just_established,
    )
    profile["last_follow_up"] = follow_up
    profile["updated_at"] = _utc_timestamp()
    return profile, follow_up


def profile_path_for_person(profile_root, person_id):
    """Return a non-enumerable filename for a confirmed business identity."""

    person_id = _required_identifier(person_id, "person_id")
    digest = hashlib.sha256(person_id.encode("utf-8")).hexdigest()[:24]
    return Path(profile_root) / f"coach_profile_{digest}.json"


def load_or_create_athlete_profile(profile_root, person_id, *, sport_id="badminton"):
    """Return coach-entered athlete data without inventing values from video."""

    profile_path = profile_path_for_person(profile_root, person_id)
    if profile_path.is_file():
        profile = load_coach_profile(profile_path)
        return profile_path, _normalized_athlete_profile(profile.get("athlete_profile"))
    profile = create_coach_profile(person_id, sport_id=sport_id)
    return profile_path, _normalized_athlete_profile(profile.get("athlete_profile"))


def save_athlete_profile(profile_root, person_id, athlete_profile, *, sport_id="badminton"):
    """Persist coach-entered player knowledge in the business layer only."""

    person_id = _required_identifier(person_id, "person_id")
    profile_path = profile_path_for_person(profile_root, person_id)
    profile = (
        load_coach_profile(profile_path)
        if profile_path.is_file()
        else create_coach_profile(person_id, sport_id=sport_id)
    )
    if profile.get("person_id") != person_id:
        raise ValueError("Coach profile identity does not match person_id.")
    profile["athlete_profile"] = _normalized_athlete_profile(athlete_profile)
    profile["updated_at"] = _utc_timestamp()
    return save_coach_profile(profile_path, profile), profile["athlete_profile"]


def load_coach_profile(path):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read coach profile: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "longitudinal_coach_profile":
        raise ValueError("Coach profile has an unsupported schema.")
    return payload


def save_coach_profile(path, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, profile)
    os.replace(temporary, path)
    return str(path)


def record_coach_followup_from_analysis(
    output_dir,
    *,
    person_id,
    profile_root,
    manual_context=None,
    llm_request=None,
):
    """Update one confirmed player's profile from immutable analysis outputs.

    ``match_identity_claims.json`` is the explicit business boundary.  A
    missing, ambiguous or unreviewed binding refuses profile creation rather
    than guessing a player from a visual trajectory.
    """

    output_dir = Path(output_dir)
    person_id = _required_identifier(person_id, "person_id")
    binding = _confirmed_binding_for_person(output_dir, person_id)
    metrics_path = output_dir / "derived" / "player_movement_metrics_v1.json"
    if not metrics_path.is_file():
        raise FileNotFoundError("Movement metrics are required before generating a coach follow-up.")
    metrics = _read_json(metrics_path)
    metadata = _read_json(output_dir / "metadata.json")
    session_id = (
        metadata.get("analysis_session_id")
        or metadata.get("session_id")
        or metadata.get("analysis", {}).get("session_id")
        or f"local:{output_dir.resolve().name}"
    )
    context = _mapping_or_empty(manual_context).get("measurement_context")
    observation = build_confirmed_session_observation(
        analysis_session_id=session_id,
        person_id=person_id,
        track_id=binding["track_id"],
        movement_metrics=metrics,
        metadata=metadata,
        claim_status="confirmed",
        context=context,
    )
    coach_match_context = _normalized_match_context(
        _mapping_or_empty(manual_context).get("match_context")
    )
    if coach_match_context:
        observation["coach_match_context"] = coach_match_context
    profile_path = profile_path_for_person(profile_root, person_id)
    profile = (
        load_coach_profile(profile_path)
        if profile_path.is_file()
        else create_coach_profile(
            person_id,
            sport_id=(observation.get("measurement_context") or {}).get("sport_id") or "badminton",
            subjective_ratings=_mapping_or_empty(manual_context).get("subjective_ratings"),
            tactical_preferences=_mapping_or_empty(manual_context).get("tactical_preferences"),
        )
    )
    profile, follow_up = update_coach_profile(profile, observation, manual_context=manual_context)
    profile_path_text = save_coach_profile(profile_path, profile)

    derived_dir = output_dir / "derived"
    evidence = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "longitudinal_coach_followup_evidence",
        "profile_path": profile_path_text,
        "person_id": person_id,
        "observation": observation,
        "baseline": profile.get("baseline"),
        "manual_coach_context": profile.get("manual_coach_context"),
        "athlete_profile": profile.get("athlete_profile"),
        "follow_up": follow_up,
        "limits": [
            "This report compares visual movement measurements only.",
            "No unconfirmed track, predicted position, score, stroke technique or tactical intent is inferred.",
        ],
    }
    training_plan = build_training_plan_candidate(evidence)
    evidence["training_plan_candidate"] = training_plan
    derived_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = derived_dir / "coach_followup_input_v1.json"
    _write_json(evidence_path, evidence)
    prompt_path = derived_dir / "coach_followup_prompt_zh.md"
    prompt = build_chinese_followup_prompt(evidence)
    prompt_path.write_text(prompt, encoding="utf-8")
    report = _generate_optional_llm_report(evidence, prompt, llm_request=llm_request)
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "longitudinal_coach_followup_report",
            "profile_path": profile_path_text,
            "evidence_path": str(evidence_path),
            "prompt_path": str(prompt_path),
            "follow_up": follow_up,
            "training_plan_candidate": training_plan,
        }
    )
    report_path = derived_dir / "coach_followup_report_v1.json"
    _write_json(report_path, report)
    training_plan_path = derived_dir / "coach_training_plan_v1.json"
    _write_json(
        training_plan_path,
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "longitudinal_coach_training_plan_candidate",
            "person_id": person_id,
            "follow_up_mode": follow_up.get("mode"),
            "plan": training_plan,
            "policy": "Candidates require coach confirmation and do not change a player's style automatically.",
        },
    )
    return {
        **report,
        "report_path": str(report_path),
        "training_plan_path": str(training_plan_path),
    }


def build_chinese_followup_prompt(evidence):
    """Ask the LLM for a concise, longitudinal coaching response only."""

    follow_up = _mapping_or_empty(evidence.get("follow_up"))
    mode = follow_up.get("mode")
    if mode == "baseline_collection":
        scope = (
            "这是建档阶段。说明还差多少场合格样本才形成基线；只概括当前可观测的移动事实，"
            "不要为每场重复生成完整优缺点清单，也不要给稳定能力等级。"
        )
    elif mode == "follow_up":
        scope = (
            "这是基线后的单场跟进。只讨论相对个人基线的变化、保持项和最多两项下一场关注点；"
            "不得重述完整能力画像。单场差异必须标为待下一场确认，只有持续信号才可称为趋势。"
        )
    else:
        scope = "本场证据不足或条件不可比。解释为什么不能作纵向结论，并给出下一次采集条件。"
    serialized = json.dumps(evidence, ensure_ascii=False, indent=2)
    return (
        "你是羽毛球纵向 AI 教练。仅依据下列结构化证据回答，不得补充未测量的动作、球路、"
        "比分、失误归因、伤病或战术意图。track_id 不能当作姓名。\n\n"
        f"任务范围：{scope}\n\n"
        "教练手工档案可以作为训练上下文，但必须明确其来源；不得把它改写为视觉测量结论。"
        "若没有经复核的逐拍球路/回合证据，不得提出自动打法改造。\n\n"
        "请用中文输出严格 JSON：{\"mode\":string,\"change_summary\":string,"
        "\"maintained_or_improved\":[string],\"watchouts\":[string],"
        "\"next_match_focus\":[string],\"training_plan_candidates\":[string],"
        "\"style_update\":string,\"limits\":[string]}。"
        "每个数组最多两项；没有可靠变化时明确写“待下一场确认”。\n\n"
        f"证据：\n{serialized}\n"
    )


def build_training_plan_candidate(evidence):
    """Create a bounded candidate plan while preserving every data source.

    Movement comparisons can flag a repeatable physical observation.  They do
    not select a technical correction or alter play style.  Technical and
    tactical focus therefore comes only from an explicit coach rating or
    priority until reviewed stroke/rally evidence exists.
    """

    athlete = _normalized_athlete_profile(evidence.get("athlete_profile"))
    follow_up = _mapping_or_empty(evidence.get("follow_up"))
    observations = []
    for category in ("technical_ratings", "tactical_ratings"):
        for key, value in _mapping_or_empty(athlete.get(category)).items():
            rating = _rating_or_none(value)
            if rating is not None and rating <= 2:
                observations.append((rating, key))
    observations.sort(key=lambda item: item[0])

    candidates = []
    for priority in athlete.get("development_priorities") or []:
        candidates.append(
            {
                "source": "coach_manual_input",
                "type": "coach_confirmed_priority",
                "focus": priority,
                "action": "按教练已确认的培养重点安排专项训练，并在下一次比赛复核迁移效果。",
            }
        )
    for rating, key in observations:
        if len(candidates) >= MAX_FOCUS_ITEMS:
            break
        candidates.append(
            {
                "source": "coach_manual_input",
                "type": "manual_rating_followup",
                "focus": _MANUAL_RATING_LABELS.get(key, key),
                "rating": rating,
                "action": "将该教练评分较低项纳入训练计划；视觉移动数据不自动决定具体技术纠正动作。",
            }
        )
    for item in follow_up.get("focus_items") or []:
        if len(candidates) >= MAX_FOCUS_ITEMS:
            break
        if item.get("type") in {"sustained_watchout", "confirm_next_match"}:
            candidates.append(
                {
                    "source": "visual_movement_comparison",
                    "type": item.get("type"),
                    "focus": item.get("metric"),
                    "action": item.get("message"),
                }
            )

    if follow_up.get("mode") in {"baseline_collection", "baseline_established"}:
        status = "baseline_not_ready_for_match_change"
        summary = "先完成个人移动基线；保留教练既定训练计划，不依据初始样本重写打法。"
    elif follow_up.get("mode") == "follow_up":
        status = "coach_review_required"
        summary = "本场训练候选只引用教练档案和相对个人基线的移动变化，需教练确认后执行。"
    else:
        status = "insufficient_evidence"
        summary = "本场不满足纵向比较条件，不生成基于比赛变化的训练调整。"
    return {
        "status": status,
        "summary": summary,
        "candidates": candidates[:MAX_FOCUS_ITEMS],
        "training_goal": athlete.get("current_training_goal"),
        "training_progress_notes": athlete.get("training_progress_notes"),
        "coach_match_context": _mapping_or_empty(evidence.get("observation")).get(
            "coach_match_context"
        ),
        "style_update": {
            "status": "not_changed_automatically",
            "current_coach_context": athlete.get("play_style_notes"),
            "message": (
                "当前没有经复核的逐拍球路、回合和得失分证据；系统不会仅凭移动指标自动修改打法。"
                "结合下一场复核和教练确认后，才能记录打法调整。"
            ),
            "required_evidence": ["reviewed_shot_and_rally_data", "coach_confirmation"],
        },
        "limits": [
            "Coach-entered ratings and priorities are not CV measurements.",
            "Visual movement changes alone cannot prescribe a technical correction or tactical style update.",
        ],
    }


def _build_follow_up(profile, observation, *, baseline_just_established=False):
    baseline = _ensure_baseline_shape(profile)
    quality = _mapping_or_empty(observation.get("quality"))
    if not quality.get("eligible_for_baseline"):
        return {
            "mode": "insufficient_evidence",
            "baseline_status": baseline.get("status"),
            "summary": "本场视觉测量不满足纵向比较门槛，仅保留原始证据。",
            "ineligible_reasons": quality.get("ineligible_reasons") or [],
            "comparisons": [],
            "focus_items": [],
            "limits": ["Do not infer a coaching change from an ineligible observation."],
        }
    if baseline.get("status") != "established":
        collected = len(baseline.get("eligible_session_ids") or [])
        return {
            "mode": "baseline_collection",
            "baseline_status": "collecting",
            "eligible_session_count": collected,
            "required_eligible_session_count": MIN_BASELINE_SESSIONS,
            "summary": f"已收集 {collected}/{MIN_BASELINE_SESSIONS} 场可比合格样本，暂不输出稳定优缺点。",
            "comparisons": [],
            "focus_items": [
                {
                    "type": "baseline_collection",
                    "message": "继续采集完整、同条件比赛；基线形成后才开始强调单场变化。",
                }
            ],
            "limits": ["Initial sessions build a baseline; they do not establish a permanent ability label."],
        }
    if baseline_just_established:
        return {
            "mode": "baseline_established",
            "baseline_status": "established",
            "baseline_session_ids": list(baseline.get("eligible_session_ids") or []),
            "summary": "已形成个人移动基线；从下一场可比比赛开始只跟踪相对变化。",
            "comparisons": [],
            "focus_items": [
                {
                    "type": "baseline_ready",
                    "message": "下一场起只提示相对基线的变化、保持项和一到两项关注点。",
                }
            ],
            "limits": ["The baseline is a personal reference, not a universal ability grade."],
        }
    if not _contexts_match(baseline.get("context"), observation.get("measurement_context")):
        return {
            "mode": "new_context_baseline_required",
            "baseline_status": "established",
            "summary": "本场机位、赛制或测量条件与既有基线不同，不能直接比较。",
            "comparisons": [],
            "focus_items": [
                {
                    "type": "context_change",
                    "message": "在该新条件下重新积累合格样本，避免把采集差异误判为球员变化。",
                }
            ],
            "limits": ["No cross-context movement comparison is made."],
        }

    comparisons = []
    for spec in _METRIC_SPECS:
        baseline_metric = _mapping_or_empty(baseline.get("metrics")).get(spec["key"])
        baseline_value = _finite_or_none(_mapping_or_empty(baseline_metric).get("median"))
        current_value = _finite_or_none(_mapping_or_empty(observation.get("movement_observations")).get(spec["key"]))
        comparisons.append(
            _compare_metric(profile, observation, spec, baseline_value, current_value)
        )
    focus_items = _focus_items(comparisons)
    return {
        "mode": "follow_up",
        "baseline_status": "established",
        "baseline_session_ids": list(baseline.get("eligible_session_ids") or []),
        "summary": "本场只呈现相对个人基线的移动变化，不重复完整能力画像。",
        "comparisons": comparisons,
        "focus_items": focus_items,
        "limits": [
            "A sustained signal needs two comparable follow-up observations in the same direction.",
            "Movement changes are match-context observations, not proof of technique or tactical intent.",
        ],
    }


def _compare_metric(profile, observation, spec, baseline_value, current_value):
    result = {
        "metric": spec["key"],
        "label": spec["label"],
        "unit": spec["unit"],
        "baseline_median": baseline_value,
        "current_value": current_value,
        "status": "not_measured",
        "direction": None,
        "relative_change": None,
        "trend_status": "not_measured",
    }
    if baseline_value is None or current_value is None or abs(baseline_value) < 1e-9:
        return result
    raw_change = (current_value - baseline_value) / abs(baseline_value)
    result["relative_change"] = round(raw_change, 4)
    if abs(raw_change) < spec["relative_threshold"]:
        result.update({"status": "within_baseline_range", "trend_status": "within_baseline_range"})
        return result
    improved = raw_change > 0 if spec["direction"] == "higher" else raw_change < 0
    direction = "improved" if improved else "declined"
    result.update({"status": "material_variation", "direction": direction})
    prior_directions = _prior_material_directions(profile, observation, spec, baseline_value)
    if len(prior_directions) >= RECENT_CONFIRMATION_SESSIONS - 1 and all(
        item == direction for item in prior_directions[-(RECENT_CONFIRMATION_SESSIONS - 1):]
    ):
        result["trend_status"] = f"sustained_{direction}"
    else:
        result["trend_status"] = "single_match_variation"
    return result


def _prior_material_directions(profile, observation, spec, baseline_value):
    prior = []
    baseline_session_ids = set(_mapping_or_empty(profile.get("baseline")).get("eligible_session_ids") or [])
    for item in profile.get("session_history") or []:
        if (
            not isinstance(item, dict)
            or item.get("analysis_session_id") == observation.get("analysis_session_id")
            or item.get("analysis_session_id") in baseline_session_ids
        ):
            continue
        if not _is_baseline_eligible(item) or not _contexts_match(
            item.get("measurement_context"), observation.get("measurement_context")
        ):
            continue
        value = _finite_or_none(_mapping_or_empty(item.get("movement_observations")).get(spec["key"]))
        if value is None or abs(baseline_value) < 1e-9:
            continue
        change = (value - baseline_value) / abs(baseline_value)
        if abs(change) < spec["relative_threshold"]:
            continue
        improved = change > 0 if spec["direction"] == "higher" else change < 0
        prior.append("improved" if improved else "declined")
    return prior[-RECENT_CONFIRMATION_SESSIONS:]


def _focus_items(comparisons):
    sustained_declines = [item for item in comparisons if item.get("trend_status") == "sustained_declined"]
    sustained_improvements = [item for item in comparisons if item.get("trend_status") == "sustained_improved"]
    single_variations = [item for item in comparisons if item.get("trend_status") == "single_match_variation"]
    items = []
    for comparison in sustained_declines:
        items.append(
            {
                "type": "sustained_watchout",
                "metric": comparison["metric"],
                "message": f"{comparison['label']}连续两场偏离个人基线，下一场优先复核该移动现象。",
            }
        )
    for comparison in sustained_improvements:
        items.append(
            {
                "type": "sustained_positive_change",
                "metric": comparison["metric"],
                "message": f"{comparison['label']}连续两场优于个人基线，记录为可继续观察的积极变化。",
            }
        )
    if not items:
        for comparison in single_variations[:1]:
            items.append(
                {
                    "type": "confirm_next_match",
                    "metric": comparison["metric"],
                    "message": f"{comparison['label']}本场与基线存在差异，先在下一场相同条件下确认，不下能力结论。",
                }
            )
    return items[:MAX_FOCUS_ITEMS]


def _baseline_metrics(observations):
    result = {}
    for spec in _METRIC_SPECS:
        values = [
            _finite_or_none(_mapping_or_empty(item.get("movement_observations")).get(spec["key"]))
            for item in observations
        ]
        values = [value for value in values if value is not None]
        if not values:
            continue
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        result[spec["key"]] = {
            "median": round(median, 4),
            "median_absolute_deviation": round(statistics.median(deviations), 4),
            "sample_count": len(values),
            "unit": spec["unit"],
        }
    return result


def _ensure_baseline_shape(profile):
    baseline = _mapping_or_empty(profile.get("baseline"))
    baseline.setdefault("status", "collecting")
    baseline.setdefault("required_eligible_sessions", MIN_BASELINE_SESSIONS)
    baseline.setdefault("minimum_measurement_coverage", MIN_MEASUREMENT_COVERAGE)
    baseline.setdefault("minimum_match_duration_sec", MIN_MATCH_DURATION_SEC)
    baseline.setdefault("eligible_session_ids", [])
    baseline.setdefault("context", None)
    baseline.setdefault("metrics", {})
    profile["baseline"] = baseline
    return baseline


def _is_baseline_eligible(observation):
    return bool(_mapping_or_empty(observation.get("quality")).get("eligible_for_baseline"))


def _measurement_context(*, sport_id, match_mode, camera_profile_id, coordinate_system, pose_sample_hz):
    return {
        "sport_id": _required_identifier(sport_id, "sport_id"),
        "match_mode": str(match_mode or "unknown").strip() or "unknown",
        "camera_profile_id": str(camera_profile_id or "unspecified").strip() or "unspecified",
        "coordinate_system": str(coordinate_system or "court_xy_m").strip() or "court_xy_m",
        "pose_sample_hz": _finite_or_none(pose_sample_hz),
    }


def _contexts_match(left, right):
    left = _mapping_or_empty(left)
    right = _mapping_or_empty(right)
    return all(
        left.get(key) == right.get(key)
        for key in ("sport_id", "match_mode", "camera_profile_id", "coordinate_system", "pose_sample_hz")
    )


def _merge_manual_context(profile, manual_context):
    context = _mapping_or_empty(manual_context)
    if not context:
        return
    stored = _mapping_or_empty(profile.get("manual_coach_context"))
    for key in ("subjective_ratings", "tactical_preferences"):
        value = context.get(key)
        if isinstance(value, dict):
            stored[key] = copy.deepcopy(value)
    stored.setdefault(
        "policy",
        "Manual coach context is not visual measurement. It provides coaching context only and must retain its human source.",
    )
    profile["manual_coach_context"] = stored
    if isinstance(context.get("athlete_profile"), dict):
        profile["athlete_profile"] = _normalized_athlete_profile(context["athlete_profile"])


def _default_athlete_profile():
    return {
        "source": "coach_manual_input",
        "display_name": None,
        "dominant_hand": "unknown",
        "primary_event": "unknown",
        "training_stage": "assessment",
        "technical_ratings": {key: None for key in _RATING_KEYS["technical"]},
        "tactical_ratings": {key: None for key in _RATING_KEYS["tactical"]},
        "strengths": [],
        "development_priorities": [],
        "current_training_goal": None,
        "training_progress_notes": None,
        "play_style_notes": None,
        "updated_at": None,
        "policy": (
            "All fields are coach-entered context, not values inferred from CV or an LLM. "
            "A rating may be omitted when the coach has not assessed it."
        ),
    }


def _normalized_athlete_profile(value):
    value = _mapping_or_empty(value)
    profile = _default_athlete_profile()
    display_name = str(value.get("display_name") or "").strip()
    profile["display_name"] = display_name[:80] or None
    dominant_hand = str(value.get("dominant_hand") or "unknown").strip().lower()
    profile["dominant_hand"] = dominant_hand if dominant_hand in _DOMINANT_HANDS else "unknown"
    primary_event = str(value.get("primary_event") or "unknown").strip().lower()
    profile["primary_event"] = primary_event if primary_event in _PRIMARY_EVENTS else "unknown"
    training_stage = str(value.get("training_stage") or "assessment").strip().lower()
    profile["training_stage"] = training_stage if training_stage in _TRAINING_STAGES else "assessment"
    for category, field in (("technical", "technical_ratings"), ("tactical", "tactical_ratings")):
        ratings = _mapping_or_empty(value.get(field))
        profile[field] = {key: _rating_or_none(ratings.get(key)) for key in _RATING_KEYS[category]}
    profile["strengths"] = _string_list(value.get("strengths"), maximum=6)
    profile["development_priorities"] = _string_list(value.get("development_priorities"), maximum=6)
    profile["current_training_goal"] = _short_text(value.get("current_training_goal"), maximum=300)
    profile["training_progress_notes"] = _short_text(value.get("training_progress_notes"), maximum=800)
    profile["play_style_notes"] = _short_text(value.get("play_style_notes"), maximum=500)
    profile["updated_at"] = _utc_timestamp()
    return profile


def _normalized_match_context(value):
    """Preserve optional coach match notes as manual session context only."""

    value = _mapping_or_empty(value)
    note = _short_text(value.get("coach_match_notes"), maximum=1000)
    if not note:
        return None
    return {
        "source": "coach_manual_input",
        "coach_match_notes": note,
        "policy": "This match context is a coach record, not visual inference.",
    }


def _rating_or_none(value):
    """Normalize a deliberate 1-5 coach rating; zero/blank means unassessed."""

    if value is None or value == "":
        return None
    number = _finite_or_none(value)
    if number is None or number == 0:
        return None
    if number < 1 or number > 5:
        raise ValueError("Coach ratings must be between 1 and 5, or be left unassessed.")
    return round(number, 1)


def _string_list(value, *, maximum):
    """Keep small, coach-entered lists readable and bounded in persisted evidence."""

    if isinstance(value, str):
        candidates = value.replace("\r", "\n").replace(",", "\n").split("\n")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = []
    items = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text or text in items:
            continue
        if len(text) > 120:
            raise ValueError("Each coach-entered strength or priority must be at most 120 characters.")
        items.append(text)
        if len(items) >= maximum:
            break
    return items


def _short_text(value, *, maximum):
    text = str(value or "").strip()
    if len(text) > maximum:
        raise ValueError(f"Coach-entered text must be at most {maximum} characters.")
    return text or None


def _confirmed_binding_for_person(output_dir, person_id):
    path = output_dir / "match_identity_claims.json"
    claims = _read_json(path)
    candidates = [
        item
        for item in claims.get("bindings") or []
        if isinstance(item, dict)
        and str(item.get("person_id") or "").strip() == person_id
        and item.get("binding_source") == "post_match_human_review"
        and not item.get("identity_alias")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "AI 教练档案要求此场恰有一条人工确认且非重复的 person_id 绑定；"
            "请先在赛后身份与队伍确认中保存 Track ID 绑定。"
        )
    track_id = _required_identifier(candidates[0].get("track_id"), "binding.track_id")
    return {"track_id": track_id}


def _generate_optional_llm_report(evidence, prompt, *, llm_request=None):
    result = {
        "status": "not_configured",
        "generated_at": _utc_timestamp(),
        "policy": "LLM language may explain this bounded longitudinal evidence only.",
    }
    config = _llm_config_from_environment()
    if llm_request is None and config is None:
        result["reason"] = "LLM service is not configured on this worker"
        return result
    started = time.monotonic()
    try:
        if llm_request is not None:
            content = llm_request(prompt)
            model = "injected_test_client"
        else:
            content = _request_openai_compatible_report(config, prompt)
            model = config["model"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response content is empty")
        result.update(
            {
                "status": "succeeded",
                "model": model,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "content": content,
            }
        )
    except (OSError, ValueError, error.URLError, TimeoutError) as exc:
        result.update(
            {
                "status": "failed",
                "model": (config or {}).get("model"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "reason": str(exc),
            }
        )
    return result


def _llm_config_from_environment():
    base_url = os.environ.get("GOOD_BADMINTON_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("GOOD_BADMINTON_LLM_API_KEY", "").strip()
    model = os.environ.get("GOOD_BADMINTON_LLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        return None
    timeout = int(os.environ.get("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS", "75"))
    if timeout <= 0 or timeout > 180:
        raise ValueError("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS must be between 1 and 180")
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model, "timeout": timeout}


def _request_openai_compatible_report(config, prompt):
    endpoint = config["base_url"]
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    payload = json.dumps(
        {
            "model": config["model"],
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Never invent unavailable sports evidence.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"},
    )
    with request.urlopen(http_request, timeout=config["timeout"]) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response does not contain choices[0].message.content") from exc


def _mapping_or_empty(value):
    return value if isinstance(value, dict) else {}


def _required_identifier(value, name):
    value = str(value or "").strip()
    if not value or len(value) > 256:
        raise ValueError(f"{name} is required and must be at most 256 characters")
    return value


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
