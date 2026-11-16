"""Capture end outbox delivery, independent of media/result archival."""
import json
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener
from .result_notifications import ResultNotifications, NoRedirect
from good_badminton_contracts.gpu_notifications import encode, signature


def lifecycle_url(value):
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, '/api/v1/edge/gpu-lifecycle-events', '', ''))


class CaptureNotifications(ResultNotifications):
    def __init__(self, manager, *, url, secret, **kwargs):
        super().__init__(manager, None, url=url, secret=secret, archive_results=False, not_before='', **kwargs)

    def payload(self, session):
        return session.get('capture_lifecycle_event')

    def _path(self, session_id):
        return self.manager._session_dir(session_id) / 'capture-notification.json'

    def _send(self, payload):
        raw = encode(payload)
        timestamp = str(int(self.clock()))
        target = self.url
        if self.environment_urls:
            session = self.manager.get_session(payload['analysis_session_id']) or {}
            environment = (session.get('media_identity') or {}).get('environment')
            if environment not in self.environment_urls:
                raise ValueError('capture callback environment is not enabled')
            target = self.environment_urls[environment]
        request = Request(target, data=raw, method='POST', headers={
            'Content-Type': 'application/json', 'X-Gpu-Timestamp': timestamp,
            'X-Gpu-Signature': signature(self.secret, timestamp, raw)})
        with build_opener(NoRedirect).open(request, timeout=10) as reply:
            result = json.loads(reply.read(32768))
            if (reply.status not in {200, 202} or result.get('status') != 'accepted'
                    or result.get('event_id') != payload['event_id']):
                raise IOError('business server did not acknowledge this capture event')
