"""Compatibility alias for the GPU client moved to the analysis platform."""

import sys

from analysis_platform import gpu_client as _implementation

sys.modules[__name__] = _implementation
