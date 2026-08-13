"""Resume polling remote GPU jobs recorded before a WebUI/business restart.

Run from the project root after setting the same WebUI GPU API configuration:
```
python -m webui.reconcile_remote_tasks
```
It never blindly resubmits a video. A missing receipt is reconciled through the
idempotency key; each invocation performs one poll and can safely be scheduled.
"""

import json

from webui.remote_gpu import RemoteAnalysisError, recover_remote_task
from webui.task_ledger import BusinessTaskLedger


def reconcile_once(ledger=None):
    ledger = ledger or BusinessTaskLedger()
    summary = []
    for task in ledger.pending_tasks():
        task_id = task["task_id"]

        def record(event):
            ledger.record_remote_event(task_id, event)

        try:
            job, result = recover_remote_task(
                task_id,
                (task.get("remote") or {}).get("job_id"),
                task["output_dir"],
                status_cb=record,
            )
            state = job.get("status")
            if state == "failed":
                ledger.record_terminal(task_id, status="failed", error=job.get("error"))
            elif result is not None:
                ledger.record_terminal(task_id, status="succeeded")
            summary.append({"task_id": task_id, "status": state, "remote_job_id": job.get("job_id")})
        except RemoteAnalysisError as exc:
            # Do not mark a task failed just because the GPU service is briefly
            # unavailable; the next scheduled pass can continue from this file.
            ledger.record_event(task_id, "recovery_poll_failed", details={"message": str(exc)})
            summary.append({"task_id": task_id, "status": "retry_pending", "error": str(exc)})
    return summary


if __name__ == "__main__":
    print(json.dumps(reconcile_once(), ensure_ascii=False, indent=2))
