"""Emit the one-time venue relay environment after an operator-controlled bind."""

from __future__ import annotations

from seed_good_yushi_pilot import CAMERA_ID, COURT_ID, DEVICE_ID, VENUE_ID
from operator_api.services.operator_backoffice import BusinessDatabase


def main() -> None:
    binding = BusinessDatabase().provision_edge_camera(
        DEVICE_ID, CAMERA_ID, VENUE_ID, COURT_ID,
        "haoyushijie-gateway-01", "haoyushijie-cam-01", "v1",
    )
    print("EDGE_GATEWAY_URL=https://for-one-dream.cloud/badminton-edge")
    print(f"EDGE_DEVICE_ID={binding['device_id']}")
    print(f"EDGE_CAMERA_ID={binding['camera_id']}")
    print(f"EDGE_CREDENTIAL_VERSION={binding['credential_version']}")
    print(f"EDGE_DEVICE_SECRET={binding['device_secret']}")
    print("# Fill CAMERA_RTSP_URL and replace COURT_CORNERS_JSON after calibration.")
    print("CAMERA_RTSP_URL=")
    print("COURT_CORNERS_JSON=[[0,0],[1,0],[1,1],[0,1]]")


if __name__ == "__main__":
    main()
