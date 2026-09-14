"""Compatibility alias for stream summaries moved to the analysis platform."""

import sys

from analysis_platform import stream_speed_summary as _implementation

sys.modules[__name__] = _implementation
