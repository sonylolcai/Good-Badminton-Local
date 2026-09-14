"""Compatibility alias for task controls moved to the analysis platform."""

import sys

from analysis_platform import task_control as _implementation

sys.modules[__name__] = _implementation
