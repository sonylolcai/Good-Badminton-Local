"""Development court lobby with one active match per court.

This adapter keeps the same concurrency rules planned for PostgreSQL: the
server, not the Mini Program, owns slot allocation, expiry and the decision
that a roster can start.  It is intentionally anonymous: ``actor_id`` is a
local development client token, not a WeChat identity, and nothing here is
sent to GPU.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .fixed_video_catalog import get_fixed_video


SLOT_IDS = ("left-1", "left-2", "right-1", "right-2")


class FixedReplaySubmitter(Protocol):
    def submit_fixed_video(self, fixed_video_id: str) -> dict[str, Any]: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


class CourtMatchManager:
    """In-memory development implementation of the court resource lock.

    Production must use the equivalent PostgreSQL row lock plus the partial
    unique index from ``0003_court_active_match.sql``.  The public payload
    deliberately exposes occupancy only; it does not reveal other users'
    personal identities.
    """

    def __init__(self, replay_manager: FixedReplaySubmitter, *, waiting_timeout_seconds: int = 300) -> None:
        self._replay_manager = replay_manager
        self._waiting_timeout = timedelta(seconds=waiting_timeout_seconds)
        self._matches: dict[str, dict[str, Any]] = {}
        self._active_by_court: dict[str, str] = {}
        self._expiry_timers: dict[str, threading.Timer] = {}
        self._lock = threading.RLock()

    def active(self, court_id: str, viewer_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._expire_waiting_locked()
            match_id = self._active_by_court.get(court_id)
            return self._public(self._matches[match_id], viewer_id) if match_id else None

    def status(self, match_id: str, viewer_id: str) -> dict[str, Any]:
        """Read a known match after its court has already been released."""
        with self._lock:
            self._expire_waiting_locked()
            return self._public(self._get_match_locked(match_id), viewer_id)

    def join(self, court_id: str, actor_id: str, slot_id: str) -> dict[str, Any]:
        if not actor_id.strip():
            raise ValueError("actor_id is required")
        if slot_id not in SLOT_IDS:
            raise ValueError("unknown court slot")
        with self._lock:
            self._expire_waiting_locked()
            match_id = self._active_by_court.get(court_id)
            if match_id is None:
                now = _now()
                match_id = f"match_{uuid.uuid4().hex}"
                match = {
                    "match_id": match_id,
                    "court_id": court_id,
                    "status": "waiting",
                    "created_at": now,
                    "expires_at": now + self._waiting_timeout,
                    "participants": {},
                    "analysis_job_id": None,
                    "ended_at": None,
                    "delivery_deadline_at": None,
                }
                self._matches[match_id] = match
                self._active_by_court[court_id] = match_id
                timer = threading.Timer(self._waiting_timeout.total_seconds(), self._expire_match, args=(match_id,))
                timer.daemon = True
                self._expiry_timers[match_id] = timer
                timer.start()
            else:
                match = self._matches[match_id]
                if match["status"] != "waiting":
                    raise ValueError("该场地正在比赛，暂时不能加入新的对局")

            occupant = match["participants"].get(slot_id)
            if occupant and occupant != actor_id:
                raise ValueError("该位置已被其他球友占用")
            current_slot = next((slot for slot, participant in match["participants"].items() if participant == actor_id), None)
            if current_slot and current_slot != slot_id:
                del match["participants"][current_slot]
            match["participants"][slot_id] = actor_id
            return self._public(match, actor_id)

    def start(self, match_id: str, actor_id: str, fixed_video_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire_waiting_locked()
            match = self._get_match_locked(match_id)
            if match["status"] != "waiting":
                raise ValueError("该对局当前不能开始")
            if actor_id not in match["participants"].values():
                raise ValueError("只有已加入对局的球友可以开始比赛")
            expected_count = self._startable_count(match)
            if expected_count is None:
                raise ValueError("需要合法的 2 人单打站位或 4 人双打站位才可开始")
            video = get_fixed_video(fixed_video_id)
            if int(video["expected_player_count"]) != expected_count:
                raise ValueError("固定视频人数与当前对局人数不一致")
            try:
                replay = self._replay_manager.submit_fixed_video(fixed_video_id)
            except Exception:
                # Keep the waiting roster intact when recording/analysis could
                # not be started; the court has not entered playing state.
                raise
            match["status"] = "playing"
            timer = self._expiry_timers.pop(match_id, None)
            if timer:
                timer.cancel()
            match["analysis_job_id"] = str(replay["business_task_id"])
            match["match_format"] = "singles" if expected_count == 2 else "doubles"
            match["started_at"] = _now()
            return self._public(match, actor_id)

    def leave(self, match_id: str, actor_id: str) -> dict[str, Any]:
        """Remove the caller from a waiting roster without affecting others.

        The queue remains owned by the court until its original five-minute
        deadline.  When the last participant leaves, however, the court is
        immediately released and the expiry timer is cancelled.
        """
        with self._lock:
            self._expire_waiting_locked()
            match = self._get_match_locked(match_id)
            if match["status"] != "waiting":
                raise ValueError("比赛开始后不能退出候场")
            current_slot = next((slot for slot, participant in match["participants"].items() if participant == actor_id), None)
            if current_slot is None:
                raise ValueError("你尚未加入本场对局")
            del match["participants"][current_slot]
            if not match["participants"]:
                match["status"] = "cancelled"
                if self._active_by_court.get(match["court_id"]) == match_id:
                    del self._active_by_court[match["court_id"]]
                timer = self._expiry_timers.pop(match_id, None)
                if timer:
                    timer.cancel()
            return self._public(match, actor_id)

    def end(self, match_id: str, actor_id: str) -> dict[str, Any]:
        with self._lock:
            match = self._get_match_locked(match_id)
            if actor_id not in match["participants"].values():
                raise ValueError("只有本场参与者可以结束比赛")
            if match["status"] == "score_pending":
                return self._public(match, actor_id)
            if match["status"] != "playing":
                raise ValueError("该对局当前不能结束")
            match["status"] = "score_pending"
            match["ended_at"] = _now()
            # Score/claim/delivery no longer consume the court resource.
            if self._active_by_court.get(match["court_id"]) == match_id:
                del self._active_by_court[match["court_id"]]
            return self._public(match, actor_id)

    def begin_delivery(self, match_id: str, actor_id: str) -> dict[str, Any]:
        with self._lock:
            match = self._get_match_locked(match_id)
            if actor_id not in match["participants"].values():
                raise ValueError("只有本场参与者可以开始个人资料生成")
            if match["status"] != "score_pending":
                raise ValueError("赛果与认领阶段尚未开始")
            match["delivery_deadline_at"] = _now() + timedelta(minutes=5)
            return self._public(match, actor_id)

    def _expire_waiting_locked(self) -> None:
        now = _now()
        for match_id, match in self._matches.items():
            if match["status"] == "waiting" and match["expires_at"] <= now:
                match["status"] = "cancelled_timeout"
                if self._active_by_court.get(match["court_id"]) == match_id:
                    del self._active_by_court[match["court_id"]]

    def _expire_match(self, match_id: str) -> None:
        """Release the physical court at the deadline even with no clients polling."""
        with self._lock:
            match = self._matches.get(match_id)
            if match is None or match["status"] != "waiting" or match["expires_at"] > _now():
                return
            match["status"] = "cancelled_timeout"
            if self._active_by_court.get(match["court_id"]) == match_id:
                del self._active_by_court[match["court_id"]]
            self._expiry_timers.pop(match_id, None)

    def _get_match_locked(self, match_id: str) -> dict[str, Any]:
        match = self._matches.get(match_id)
        if match is None:
            raise KeyError(match_id)
        return match

    @staticmethod
    def _startable_count(match: dict[str, Any]) -> int | None:
        slots = set(match["participants"])
        if slots == {"left-1", "right-1"}:
            return 2
        if slots == set(SLOT_IDS):
            return 4
        return None

    def _public(self, match: dict[str, Any], viewer_id: str) -> dict[str, Any]:
        slots = [
            {
                "slot_id": slot_id,
                "occupied": slot_id in match["participants"],
                "is_self": match["participants"].get(slot_id) == viewer_id,
            }
            for slot_id in SLOT_IDS
        ]
        startable_count = self._startable_count(match) if match["status"] == "waiting" else None
        return {
            "match_id": match["match_id"],
            "court_id": match["court_id"],
            "status": match["status"],
            "slots": slots,
            "participant_count": len(match["participants"]),
            "startable_count": startable_count,
            "start_label": "开始单打比赛" if startable_count == 2 else "开始双打比赛" if startable_count == 4 else None,
            "expires_at": _iso(match.get("expires_at")),
            "started_at": _iso(match.get("started_at")),
            "ended_at": _iso(match.get("ended_at")),
            "analysis_job_id": match.get("analysis_job_id"),
            "match_format": match.get("match_format"),
            "delivery_deadline_at": _iso(match.get("delivery_deadline_at")),
        }
