"""Start a GPU process permanently configured for badminton vision."""

import os

# The legacy ``api.app`` composition root resolves its profile during import.
# Set it here rather than accepting a request-time sport switch.
os.environ["GOOD_SPORT_VISION_PROFILE"] = "badminton"

from api.app import app

__all__ = ["app"]
