"""Fixed tennis entry point for the pure GPU visual-observation process."""

from api.gpu_stream_app import create_gpu_stream_app
from api.vision_profiles import TENNIS_PROFILE


app = create_gpu_stream_app(vision_profile=TENNIS_PROFILE)

__all__ = ["app"]
