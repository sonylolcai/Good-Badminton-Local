"""Small, dependency-free primitives for cooperative analysis cancellation."""


class AnalysisCancelled(RuntimeError):
    """Raised when an operator interrupts an in-progress video analysis."""


def raise_if_cancelled(cancel_cb, message="分析任务已中断。"):
    """Raise at a safe checkpoint when *cancel_cb* reports cancellation."""
    if cancel_cb is not None and cancel_cb():
        raise AnalysisCancelled(message)
