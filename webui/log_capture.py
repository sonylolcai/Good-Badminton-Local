"""Thread-safe in-process capture for the WebUI developer console."""

import sys
import threading
from collections import deque


class _BackendLogBuffer:
    def __init__(self, max_lines=3000):
        self._lines = deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def append(self, stream_name, text):
        if not text:
            return
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self._lines.append(f"[{stream_name}] {line}")

    def read(self):
        with self._lock:
            if not self._lines:
                return "等待后台输出…"
            return "\n".join(self._lines)


class _TeeStream:
    def __init__(self, original, stream_name, log_buffer):
        self._original = original
        self._stream_name = stream_name
        self._buffer = log_buffer
        self._pending = ""

    def write(self, text):
        written = self._original.write(text)
        self._original.flush()
        self._pending += text
        if "\n" in self._pending:
            complete, self._pending = self._pending.rsplit("\n", 1)
            self._buffer.append(self._stream_name, complete)
        return written

    def flush(self):
        self._original.flush()
        if self._pending:
            self._buffer.append(self._stream_name, self._pending)
            self._pending = ""

    def __getattr__(self, name):
        return getattr(self._original, name)


_LOG_BUFFER = _BackendLogBuffer()
_INSTALLED = False


def install_backend_log_capture():
    global _INSTALLED
    if _INSTALLED:
        return
    sys.stdout = _TeeStream(sys.stdout, "OUT", _LOG_BUFFER)
    sys.stderr = _TeeStream(sys.stderr, "ERR", _LOG_BUFFER)
    _INSTALLED = True


def get_backend_logs():
    return _LOG_BUFFER.read()
