"""Compatibility alias for the GPU-owned analysis pipeline."""

import sys

from badminton_analysis import pipeline as _implementation

sys.modules[__name__] = _implementation
