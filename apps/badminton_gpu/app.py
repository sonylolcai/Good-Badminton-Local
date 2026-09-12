"""Fixed badminton entry point for the pure GPU visual-observation process."""

from api.gpu_stream_app import create_gpu_stream_app
from api.vision_profiles import BADMINTON_PROFILE


app = create_gpu_stream_app(vision_profile=BADMINTON_PROFILE)

__all__ = ["app"]
