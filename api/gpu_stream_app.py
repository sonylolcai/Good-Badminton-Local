"""Composition root for a sport-fixed, pure GPU visual-observation service.

Unlike :mod:`api.app`, this app has no whole-video job API and does not import
``api.jobs``, ``webui.pipeline`` or any business package.  It exposes only the
authenticated stream-session contract: anonymous pose/person observations,
optional ball observations, quality state, trace and candidate-photo evidence.
Business interpretation must run in a separately deployed service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException

from .stream_routes import register_stream_routes
from .stream_runtime import StreamProcessorFactory
from .stream_sessions import StreamSessionManager
from .vision_profiles import BADMINTON_PROFILE, SportVisionProfile
from .release_identity import release_identity


def create_gpu_stream_app(
    data_dir=None,
    start_worker=True,
    *,
    stream_processor_factory=None,
    stream_manager=None,
    vision_profile: SportVisionProfile = BADMINTON_PROFILE,
):
    """Create a process whose sport identity is fixed at construction time."""
    data_path = Path(data_dir or os.environ.get("GOOD_BADMINTON_API_DATA_DIR", "api_data")).resolve()
    if stream_manager is None:
        stream_processor_factory = stream_processor_factory or StreamProcessorFactory(
            data_path,
            vision_profile=vision_profile,
        )
        stream_manager = StreamSessionManager(
            data_path,
            processor_factory=stream_processor_factory,
            start_worker=start_worker,
        )

    app = FastAPI(title=f"{vision_profile.service_name} GPU stream API", version="1.0.0")
    app.state.stream_manager = stream_manager
    app.state.vision_profile = vision_profile

    def require_api_key(x_api_key: Optional[str] = Header(default=None)):
        expected = os.environ.get("GOOD_BADMINTON_API_KEY")
        if not expected:
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def stream_health_payload():
        return {
            "status": "ok",
            "service": vision_profile.service_name,
            "service_kind": "pure_gpu_visual_observation",
            "sport_id": vision_profile.sport_id,
            "coordinate_system_id": vision_profile.coordinate_system_id,
            "supported_session_modes": list(vision_profile.supported_session_modes),
            "contract_versions": ["stream-session.v1"],
            "stream_worker_running": stream_manager.worker_running,
            "api_auth_configured": bool(os.environ.get("GOOD_BADMINTON_API_KEY")),
            **release_identity(),
        }

    register_stream_routes(
        app,
        stream_manager=stream_manager,
        require_api_key=require_api_key,
        health_payload=stream_health_payload,
    )
    return app
