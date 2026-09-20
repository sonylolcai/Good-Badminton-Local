"""Fixed badminton entry point for full-video and streaming GPU analysis."""

from api.app import create_app
from api.vision_profiles import BADMINTON_PROFILE


app = create_app(vision_profile=BADMINTON_PROFILE)

__all__ = ["app"]
