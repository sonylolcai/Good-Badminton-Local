"""Static contract for the portable local GPU API launcher."""

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LocalDockerTests(unittest.TestCase):
    def test_local_compose_keeps_state_outside_the_image_and_binds_loopback_only(self):
        dockerfile = (REPOSITORY_ROOT / "Dockerfile.local").read_text(encoding="utf-8")
        compose = (REPOSITORY_ROOT / "docker-compose.local.yml").read_text(encoding="utf-8")
        example = (REPOSITORY_ROOT / ".gpu-api.local.env.example").read_text(encoding="utf-8")
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.11-slim-bookworm", dockerfile)
        self.assertIn("for attempt in 1 2 3", dockerfile)
        self.assertIn("api.app:app", dockerfile)
        self.assertIn("dockerfile: Dockerfile.local", compose)
        self.assertIn("127.0.0.1:${GOOD_BADMINTON_LOCAL_PORT:-8080}:8080", compose)
        self.assertIn("good_badminton_local_api_data:/var/lib/good-badminton", compose)
        self.assertIn("./weights:/app/weights:ro", compose)
        self.assertNotIn("/models", compose)
        self.assertIn("GOOD_BADMINTON_STREAM_DEVICE: cpu", compose)
        self.assertNotIn("/models", example)
        self.assertIn("GOOD_BADMINTON_API_KEY=replace-with-a-long-random-secret", example)
        self.assertIn("supported_sport_ids", compose)
        self.assertIn(".gpu-api.local.env", gitignore)


if __name__ == "__main__":
    unittest.main()
