"""Standalone daily video-retention scheduler."""

import os
import time

from operator_api.services.retention import run_retention_once


def main() -> None:
    poll_seconds = max(60, int(os.environ.get("GOOD_BADMINTON_RETENTION_POLL_SECONDS", "300")))
    while True:
        try:
            run_retention_once()
        except Exception as exc:
            print(f"retention run failed: {exc}", flush=True)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
