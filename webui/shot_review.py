"""Compatibility alias for review logic moved to the analysis platform."""

import sys

from analysis_platform import review as _implementation

sys.modules[__name__] = _implementation
