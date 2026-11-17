"""Page a JSONL evidence log without reparsing its entire payload on each GET."""
import json
import os
import threading


class EventPageReader:
    def __init__(self):
        # One compact offset index, not a cache of all decoded pose objects.
        self._key = None
        self._index = []
        self._positions = {}
        self._lock = threading.Lock()

    @staticmethod
    def _stamp(stream):
        stat = os.fstat(stream.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def page(self, path, *, revision, cursor=None, limit=100):
        limit = max(1, min(int(limit), 1000))
        try:
            stream = path.open('rb')
        except FileNotFoundError:
            return [], None
        with stream, self._lock:
            stamp = self._stamp(stream)
            key = (str(path), revision, *stamp)
            if self._key != key:
                index = []
                offset = 0
                # Bound this pass to the opened file's snapshot, even while a
                # writer appends more events. A replacement uses another inode.
                while offset < stamp[2]:
                    line = stream.readline(stamp[2] - offset)
                    if not line:
                        break
                    try:
                        event = json.loads(line) if line.strip() else None
                    except json.JSONDecodeError:
                        event = None
                    if event is not None:
                        index.append((float(event.get('source_time_sec', 0.0)),
                                      str(event.get('event_id') or ''), offset, len(line)))
                    offset += len(line)
                index.sort(key=lambda item: (item[0], item[1]))
                positions = {}
                for position, item in enumerate(index):
                    positions.setdefault(item[1], position)
                self._key = key if self._stamp(stream) == stamp else None
                self._index, self._positions = index, positions
            start = self._positions.get(cursor, len(self._index)) + 1 if cursor else 0
            events = []
            for _, _, offset, size in self._index[start:start + limit]:
                stream.seek(offset)
                events.append(json.loads(stream.read(size)))
            return events, events[-1]['event_id'] if events else None
