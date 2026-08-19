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
    """Poll confirmed remote work once and recover completed artifacts.

    A missing receipt is handled separately from a confirmed remote job.  It
    is never silently resubmitted: a 404 idempotency lookup becomes a durable
    ``submission_unconfirmed`` record so restarts do not generate endless
    failed polling noise or duplicate a match upload.
    """
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
            trace_record = ledger.archive_performance_trace(
                task_id,
                job.get("local_performance_trace") or (result or {}).get("performance_trace"),
            )
            if state == "failed":
                ledger.record_terminal(
                    task_id,
                    status="failed",
                    error=job.get("error"),
                    details={"performance_trace": trace_record} if trace_record else None,
                )
                ledger.finalize_end_to_end_trace(task_id)
            elif state == "cancelled":
                ledger.record_terminal(
                    task_id,
                    status="cancelled",
                    error=job.get("error"),
                    details={"performance_trace": trace_record} if trace_record else None,
                )
                ledger.finalize_end_to_end_trace(task_id)
            elif result is not None:
                ledger.record_terminal(
                    task_id,
                    status="succeeded",
                    details={"performance_trace": trace_record} if trace_record else None,
                )
                ledger.finalize_end_to_end_trace(task_id)
            summary.append({"task_id": task_id, "status": state, "remote_job_id": job.get("job_id")})
        except RemoteAnalysisError as exc:
            message = str(exc)
            remote = task.get("remote") or {}
            if not remote.get("job_id") and "HTTP Error 404" in message:
                # The idempotency endpoint has positively stated that it has
                # no job for this no-receipt submission.  Keep it visible in
                # history but stop automatic retries; a user can re-submit.
                ledger.record_terminal(
                    task_id,
                    status="submission_unconfirmed",
                    error={"type": "RemoteReceiptNotRecovered", "message": message},
                )
                summary.append({"task_id": task_id, "status": "submission_unconfirmed", "error": message})
            else:
                # Do not mark a confirmed job failed just because its GPU API
                # is temporarily unreachable; the next pass can resume it.
                ledger.record_event(task_id, "recovery_poll_failed", details={"message": message})
                summary.append({"task_id": task_id, "status": "retry_pending", "error": message})
    return summary


if __name__ == "__main__":
    print(json.dumps(reconcile_once(), ensure_ascii=False, indent=2))
