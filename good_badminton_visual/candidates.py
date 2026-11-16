"""Pure candidate projection from the deployed GPU event contract."""
def project_candidates(events, session_id):
    """Build the frozen status projection from durable public events.

    The manager does not reach into processor private state. This keeps
    status reconstruction valid after restart and makes events.jsonl the
    auditable source for both live candidates and final artifacts.
    """
    candidates = {}
    artifacts = []
    events = sorted(
        events,
        key=lambda event: (
            float(event.get("source_time_sec", 0.0)),
            str(event.get("event_id") or ""),
        ),
    )
    for event in events:
        data = event.get("data") or {}
        for candidate in data.get("track_candidates") or []:
            if candidate.get("track_id"):
                candidates[str(candidate["track_id"])] = dict(candidate)

        if event.get("event_type") == "roster_candidate_observation":
            track = data.get("track") or {}
            track_id = track.get("track_id")
            if track_id:
                current = candidates.get(str(track_id), {})
                source_time = max(0.0, float(event.get("source_time_sec") or 0.0))
                candidate = {
                    "track_id": str(track_id),
                    "state": "unconfirmed_roster",
                    "first_source_time_sec": float(
                        current.get("first_source_time_sec", source_time)
                    ),
                    "last_source_time_sec": source_time,
                    "detected_coverage": None,
                    "confidence": max(
                        float(current.get("confidence") or 0.0),
                        max(0.0, min(1.0, float(track.get("confidence") or 0.0))),
                    ),
                    "analytics_eligible": False,
                    "review_reason": "expected_roster_not_yet_stable",
                }
                photo = data.get("candidate_photo")
                if isinstance(photo, dict) and str(photo.get("track_id") or "") == str(track_id):
                    candidate["candidate_photo"] = {
                        "source_time_sec": float(photo.get("source_time_sec") or source_time),
                        "capture_quality": float(photo.get("capture_quality") or 0.0),
                        "frontal_score": float(photo.get("frontal_score") or 0.0),
                        "view_label": str(photo.get("view_label") or "not_assessed"),
                        "selection_policy": str(photo.get("selection_policy") or "quality_only_v1"),
                        "media_type": "image/jpeg",
                        "fetch_path": (
                            f"/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}"
                        ),
                    }
                elif current.get("candidate_photo"):
                    candidate["candidate_photo"] = dict(current["candidate_photo"])
                candidates[str(track_id)] = candidate

        if event.get("event_type") == "person_observation":
            track = data.get("track") or {}
            track_id = track.get("track_id")
            if track_id:
                current = candidates.get(str(track_id), {})
                quality = track.get("quality") or {}
                confidence = max(0.0, min(1.0, float(track.get("confidence") or 0.0)))
                coverage = max(
                    0.0,
                    min(1.0, float(quality.get("coverage_rate") or 0.0)),
                )
                source_time = max(0.0, float(event.get("source_time_sec") or 0.0))
                candidate = {
                    "track_id": str(track_id),
                    "state": str(track.get("lifecycle_state") or "candidate"),
                    "first_source_time_sec": float(
                        current.get("first_source_time_sec", source_time)
                    ),
                    "last_source_time_sec": source_time,
                    "detected_coverage": coverage,
                    "confidence": min(confidence, coverage),
                }
                photo = data.get("candidate_photo")
                if isinstance(photo, dict) and str(photo.get("track_id") or "") == str(track_id):
                    candidate["candidate_photo"] = {
                        "source_time_sec": float(photo.get("source_time_sec") or source_time),
                        "capture_quality": float(photo.get("capture_quality") or 0.0),
                        "frontal_score": float(photo.get("frontal_score") or 0.0),
                        "view_label": str(photo.get("view_label") or "not_assessed"),
                        "selection_policy": str(photo.get("selection_policy") or "quality_only_v1"),
                        "media_type": "image/jpeg",
                        "fetch_path": (
                            f"/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}"
                        ),
                    }
                elif current.get("candidate_photo"):
                    candidate["candidate_photo"] = dict(current["candidate_photo"])
                candidates[str(track_id)] = candidate

        if event.get("event_type") == "session_finalized":
            received_artifacts = data.get("artifacts")
            if isinstance(received_artifacts, list):
                artifacts = [dict(item) for item in received_artifacts]
    return [candidates[key] for key in sorted(candidates)], artifacts
