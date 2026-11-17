"""In-memory cancellation controls for the single-concurrency WebUI worker."""

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class AnalysisTaskHandle:
    task_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def is_cancelled(self):
        return self.cancel_event.is_set()


class AnalysisTaskController:
    """Keep one WebUI analysis interruptible without exposing cross-user state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = None

    def start(self):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("已有分析任务正在运行。")
            self._active = AnalysisTaskHandle(task_id=uuid.uuid4().hex)
            return self._active

    def request_cancel(self):
        with self._lock:
            if self._active is None:
                return None
            self._active.cancel_event.set()
            return {
                "webui_task_id": self._active.task_id,
                "cancel_requested": True,
            }

    def finish(self, handle):
        with self._lock:
            if self._active is handle:
                self._active = None

    def snapshot(self):
        with self._lock:
            if self._active is None:
                return None
            return {
                "webui_task_id": self._active.task_id,
                "cancel_requested": self._active.is_cancelled(),
            }
