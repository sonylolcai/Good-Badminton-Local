from __future__ import annotations

import unittest
import zipfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def archive_member(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)


class PackageParityTests(unittest.TestCase):
    """Guard against publishing platform packages with divergent relay logic."""

    def test_all_packages_embed_the_canonical_agent(self):
        canonical = (PACKAGE_ROOT / "agent.py").read_bytes()
        self.assertEqual(
            canonical,
            (REPOSITORY_ROOT / "deploy" / "venue-gateway-windows" / "agent.py").read_bytes(),
        )
        self.assertEqual(
            canonical,
            (PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway" / "agent.py").read_bytes(),
        )
        self.assertEqual(
            canonical,
            archive_member(
                PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway.zip",
                "good-badminton-venue-gateway/agent.py",
            ),
        )
        self.assertEqual(
            canonical,
            archive_member(
                REPOSITORY_ROOT / "deploy" / "venue-gateway-macos" / "dist" / "good-badminton-venue-gateway-macos.zip",
                "good-badminton-venue-gateway-macos/agent.py",
            ),
        )

    def test_all_packages_embed_the_canonical_signing_contract(self):
        canonical = (REPOSITORY_ROOT / "business_gateway" / "edge_contract.py").read_bytes()
        self.assertEqual(
            canonical,
            (REPOSITORY_ROOT / "deploy" / "venue-gateway-windows" / "business_gateway" / "edge_contract.py").read_bytes(),
        )
        self.assertEqual(
            canonical,
            (PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway" / "business_gateway" / "edge_contract.py").read_bytes(),
        )
        self.assertEqual(
            canonical,
            archive_member(
                PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway.zip",
                "good-badminton-venue-gateway/business_gateway/edge_contract.py",
            ),
        )
        self.assertEqual(
            canonical,
            archive_member(
                REPOSITORY_ROOT / "deploy" / "venue-gateway-macos" / "dist" / "good-badminton-venue-gateway-macos.zip",
                "good-badminton-venue-gateway-macos/business_gateway/edge_contract.py",
            ),
        )


if __name__ == "__main__":
    unittest.main()
