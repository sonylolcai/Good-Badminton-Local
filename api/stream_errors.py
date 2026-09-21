"""Structured stream-session error type mapped onto the frozen v1 errorResponse.

Route registration belongs to task F, so handlers here raise this exception and
the future FastAPI layer converts it into the contract errorResponse body plus
the matching HTTP status.  Clients branch on error.code, never on message text.
"""

from __future__ import annotations

from .stream_models import error_response, new_request_id


class StreamSessionError(Exception):
    """A contract error carrying its HTTP status, code, retryability and details."""

    def __init__(
        self,
        code,
        message,
        status_code,
        retryable=False,
        details=None,
        analysis_session_id=None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        self.analysis_session_id = analysis_session_id
        self.request_id = new_request_id()

    def to_error_response(self):
        return error_response(
            self.code,
            self.message,
            self.retryable,
            details=self.details,
            request_id=self.request_id,
            analysis_session_id=self.analysis_session_id,
        )


def validation_error(message, analysis_session_id=None):
    return StreamSessionError(
        "validation_error", message, 422, analysis_session_id=analysis_session_id
    )


def session_not_found(session_id):
    return StreamSessionError(
        "session_not_found", f"session {session_id} was not found", 404
    )


def invalid_state(message, analysis_session_id=None, details=None):
    return StreamSessionError(
        "invalid_state",
        message,
        409,
        details=details,
        analysis_session_id=analysis_session_id,
    )
