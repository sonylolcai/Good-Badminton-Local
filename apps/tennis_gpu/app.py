"""Fixed tennis entry point for full-video and streaming GPU analysis."""

from api.app import create_app
from api.vision_profiles import TENNIS_PROFILE


app = create_app(vision_profile=TENNIS_PROFILE)

__all__ = ["app"]
