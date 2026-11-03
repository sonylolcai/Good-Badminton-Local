"""Authenticated telemetry and drain barrier for safe instance shutdown."""
import asyncio
import json
import os
import subprocess
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from .stream_models import TERMINAL_SESSION_STATES

def register_runtime_control(app, manager, archive, require_api_key):
    app.state.accepting_sessions = True
    admission = asyncio.Lock()
    app.state.inflight_admissions = 0
    notifications = None
    callback_url = os.environ.get('GPU_RESULT_CALLBACK_URL', '')
    if callback_url:
        if archive is None:
            raise ValueError('result callbacks require media archiving')
        from .result_notifications import ResultNotifications
        notifications = ResultNotifications(manager, archive, url=callback_url,
            secret=os.environ.get('GPU_RESULT_CALLBACK_SECRET', ''),
            environment_urls={name: os.environ['GPU_RESULT_CALLBACK_' + name.upper() + '_URL']
                              for name in ('test', 'production')
                              if os.environ.get('GPU_RESULT_CALLBACK_' + name.upper() + '_URL')})
        if manager.worker_running:
            notifications.start()
        app.router.add_event_handler('shutdown', notifications.stop)
    app.state.result_notifications = notifications

    @app.middleware('http')
    async def guard_new_sessions(request, call_next):
        import re
        mutating = request.method == 'POST' and (
            request.url.path in {'/api/v1/jobs', '/api/v1/stream-sessions'}
            or re.fullmatch(r'/api/v1/stream-sessions/[^/]+/(?:repairs|repair-window|segments/[0-9]+/repair)', request.url.path))
        if not mutating:
            return await call_next(request)
        async with admission:
            if not app.state.accepting_sessions:
                return JSONResponse({'detail': 'GPU is draining; retry after startup'}, status_code=503,
                                    headers={'Retry-After': '10'})
            app.state.inflight_admissions += 1
        try:
            return await call_next(request)
        finally:
            async with admission:
                app.state.inflight_admissions -= 1

    def snapshot(*, telemetry=True):
        active, archival, unknown = 0, 0, 0
        sessions = manager._list_sessions()
        unknown += max(0, len(list(manager.sessions_dir.glob('*/manifest.json'))) - len(sessions))
        for session in sessions:
            if session.get('status') not in TERMINAL_SESSION_STATES:
                active += 1
            if archive is not None and session.get('sealed'):
                try:
                    if archive.status(session['analysis_session_id']).get('status') not in {'ready', 'invalid', 'failed'}:
                        archival += 1
                except Exception:
                    unknown += 1
        jobs = getattr(app.state, 'job_manager', None)
        pending, running = 0, 0
        if jobs:
            for directory in jobs.jobs_dir.iterdir():
                if not directory.is_dir():
                    continue
                try:
                    job = jobs.get_job(directory.name)
                    if job and job.get('status') not in {'succeeded', 'failed', 'cancelled'}:
                        if job.get('status') == 'queued': pending += 1
                        else: running += 1
                except Exception:
                    unknown += 1
        callbacks = notifications.pending_count() if notifications else 0
        output = {'accepting_sessions': app.state.accepting_sessions, 'active_stream_sessions': active,
                  'active_jobs': running, 'pending_jobs': pending, 'pending_archives': archival,
                  'pending_callbacks': callbacks, 'blocked_callbacks': notifications.blocked_count() if notifications else 0, 'unknown_workloads': unknown,
                  'inflight_admissions': app.state.inflight_admissions,
                  'busy': bool(active + archival + pending + running + callbacks + unknown + app.state.inflight_admissions),
                  'upload_token_ttl_seconds': int(os.environ.get('GOOD_BADMINTON_UPLOAD_TOKEN_TTL_SECONDS', '0')),
                  'callback_configured': notifications is not None,
                  'analysis_archive_enabled': bool(notifications and notifications.archive_results), 'gpu': None}
        if telemetry:
            try:
                value = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=2, check=True)
                name, used, total, utilization = value.stdout.strip().splitlines()[0].split(', ')
                output['gpu'] = {'name': name, 'memory_used_mib': int(used), 'memory_total_mib': int(total),
                                 'utilization_percent': int(utilization)}
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass  # Unavailable metrics are unknown, never a fabricated 0%.
        return output

    @app.get('/api/v1/runtime/status', dependencies=[Depends(require_api_key)])
    def runtime_status():
        return snapshot()

    @app.post('/api/v1/runtime/retry-notifications', dependencies=[Depends(require_api_key)])
    def retry_notifications():
        if notifications: notifications.retry_blocked()
        return snapshot(telemetry=False)

    @app.post('/api/v1/runtime/drain', dependencies=[Depends(require_api_key)])
    async def drain(body: dict):
        if set(body) != {'enabled'} or type(body['enabled']) is not bool:
            raise HTTPException(422, 'enabled must be a boolean')
        async with admission:
            app.state.accepting_sessions = not body['enabled']
            return snapshot(telemetry=False)
