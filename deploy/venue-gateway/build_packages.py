#!/usr/bin/env python3
"""Build every venue-gateway distribution from one canonical implementation."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
SHARED_AGENT = PACKAGE_ROOT / "agent.py"
SHARED_CONTRACT = REPOSITORY_ROOT / "business_gateway" / "edge_contract.py"
SHARED_CONTRACT_INIT = REPOSITORY_ROOT / "business_gateway" / "__init__.py"


def copy_shared_files(target: Path) -> None:
    """Place the canonical runtime files in a self-contained package folder."""

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHARED_AGENT, target / "agent.py")
    contract_directory = target / "business_gateway"
    contract_directory.mkdir(exist_ok=True)
    shutil.copy2(SHARED_CONTRACT_INIT, contract_directory / "__init__.py")
    shutil.copy2(SHARED_CONTRACT, contract_directory / "edge_contract.py")


def archive_directory(source: Path, destination: Path) -> None:
    """Atomically replace an archive, preventing stale files from surviving."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(source.parent))
    temporary.replace(destination)


def build_linux_package() -> None:
    distribution = PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway"
    copy_shared_files(distribution)
    archive_directory(distribution, PACKAGE_ROOT / "dist" / "good-badminton-venue-gateway.zip")


def build_windows_package() -> None:
    # Windows has platform-specific PowerShell launchers, but uses the same
    # Python relay and signing contract as every other platform.
    copy_shared_files(REPOSITORY_ROOT / "deploy" / "venue-gateway-windows")


def build_macos_package() -> None:
    macos_source = REPOSITORY_ROOT / "deploy" / "venue-gateway-macos"
    archive_path = macos_source / "dist" / "good-badminton-venue-gateway-macos.zip"
    with tempfile.TemporaryDirectory(prefix="good-badminton-macos-") as temporary_directory:
        staging_root = Path(temporary_directory) / "good-badminton-venue-gateway-macos"
        shutil.copytree(
            macos_source,
            staging_root,
            ignore=shutil.ignore_patterns("dist", "__pycache__", "*.pyc"),
        )
        copy_shared_files(staging_root)
        archive_directory(staging_root, archive_path)


def main() -> None:
    build_linux_package()
    build_windows_package()
    build_macos_package()
    print("Built Linux, Windows, and macOS venue-gateway packages from deploy/venue-gateway.")


if __name__ == "__main__":
    main()
