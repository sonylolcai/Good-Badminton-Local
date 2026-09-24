"""One retention pass shared by the standalone scheduler and tests."""

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from operator_api.services.operator_backoffice import BusinessDatabase
from operator_api.services.resource_lifecycle import BusinessResourceService


def cleanup_evaluation_videos(retention_days: int) -> dict:
    base_url = os.environ.get("GOOD_BADMINTON_EVALUATION_API_URL", "").rstrip("/")
    key = os.environ.get("GOOD_BADMINTON_EVALUATION_MAINTENANCE_API_KEY", "")
    if not base_url or not key:
        raise RuntimeError("evaluation retention endpoint and maintenance key are required")
    query = urlencode({"retention_days": retention_days})
    request = Request(
        f"{base_url}/api/v1/maintenance/video-retention?{query}",
        method="POST",
        headers={"X-Maintenance-Key": key, "Accept": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def run_retention_once(database=None, resource_service=None, evaluation_cleanup=cleanup_evaluation_videos) -> dict:
    database = database or BusinessDatabase()
    policy = database.claim_video_retention_run()
    if policy is None:
        return {"status": "skipped"}
    resource_service = resource_service or BusinessResourceService(database)
    retention_days = int(policy["retention_days"])
    business = resource_service.cleanup_expired_videos(retention_days)
    evaluation = evaluation_cleanup(retention_days)
    database.complete_video_retention_run()
    return {"status": "completed", "business": business, "evaluation": evaluation}
