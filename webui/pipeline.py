"""Compatibility alias for the analysis runner moved out of the WebUI."""

import sys

from analysis_platform import runner as _implementation

sys.modules[__name__] = _implementation
