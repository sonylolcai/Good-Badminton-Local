"""Start a GPU process permanently configured for tennis vision."""

import os

# The entrypoint itself selects tennis before the composition root is loaded.
# A stream request can choose only ``singles_match`` or
# ``single_player_training`` after this point.
os.environ["GOOD_SPORT_VISION_PROFILE"] = "tennis"

from api.app import app

__all__ = ["app"]
