#!/usr/bin/env python3
"""Maintain ignored local configuration; never print credential values.

The same dependency-free utility is kept in each independently deployable repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SECRET_NAME = re.compile(r"SECRET|PASSWORD|TOKEN|(?:^|_)KEY(?:$|_)", re.I)
TOKEN = re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")
PRIVATE_KEY = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----")


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL)


def private_path(name: str) -> bool:
    path = Path(name)
    if ".secrets" in path.parts:
        return True
    if path.name.endswith(".example"):
        return False
    return (path.name.startswith(".env")
            or ".env." in path.name or path.name.endswith(".env")
            or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
            or path.name == "project.private.config.json")


def known_secrets() -> set[bytes]:
    values: set[bytes] = set()
    for path in (ROOT / ".secrets").rglob("*.env"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if (SECRET_NAME.search(key) and len(value) >= 12
                    and not re.search(r"example|replace|change.?me|set_by|your[_ -]|placeholder", value, re.I)):
                values.add(value.encode())
    return values


def check(*, staged: bool) -> int:
    if staged:
        # Validate the entire prospective commit, including unchanged blobs.
        paths = git("ls-files", "-z").decode().split("\0")
    else:
        paths = git("ls-files", "-c", "-o", "--exclude-standard", "-z").decode().split("\0")
    secrets = known_secrets()
    findings: set[str] = set()
    checked = 0
    for name in sorted(set(paths) - {""}):
        if private_path(name):
            findings.add(f"{name}: private configuration must not be committed")
            continue
        path = ROOT / name
        if staged:
            body = git("show", f":{name}")
        elif path.is_file() and not path.is_symlink():
            body = path.read_bytes()
        else:
            continue
        checked += 1
        if b"\0" in body:
            continue
        for number, line in enumerate(body.splitlines(), 1):
            if PRIVATE_KEY.search(line) or TOKEN.search(line) or any(value in line for value in secrets):
                findings.add(f"{name}:{number}: possible credential content (value hidden)")
    # Gitignore alone cannot protect files already added to the index.
    for name in git("ls-files", "-z").decode().split("\0"):
        if name and private_path(name):
            findings.add(f"{name}: private configuration is already in the Git index")
    if findings:
        print("Local secret check FAILED:\n" + "\n".join(sorted(findings)))
        return 1
    print(f"Local secret check passed: {checked} {'staged' if staged else 'working-tree'} files; no credential values printed.")
    return 0


def initialize() -> int:
    config = json.loads((ROOT / "config" / "local-secrets.json").read_text())
    private = ROOT.resolve() / ".secrets"
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    for entry in config["files"]:
        target = private / entry["name"]
        legacy = ROOT / entry["legacy"] if entry.get("legacy") else None
        source_exists = legacy is not None and legacy.is_file()
        if legacy is not None and legacy.is_symlink() and legacy.resolve() != target:
            raise RuntimeError(f"Refusing to replace another symlink: {entry['legacy']}")
        if target.exists():
            if source_exists and legacy.resolve() != target and legacy.read_bytes() != target.read_bytes():
                raise RuntimeError(f"Different local configurations exist; reconcile privately: {entry['name']}")
        else:
            content = legacy.read_bytes() if source_exists else (ROOT / entry["template"]).read_bytes()
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
            if source_exists and hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(legacy.read_bytes()).digest():
                raise RuntimeError("Local configuration copy verification failed")
        target.chmod(0o600)
        if legacy is not None and (source_exists or entry.get("link_if_missing", False)):
            legacy.parent.mkdir(parents=True, exist_ok=True)
            if not legacy.is_symlink():
                temporary = legacy.with_name(legacy.name + ".local-secrets-link")
                temporary.symlink_to(os.path.relpath(target, legacy.parent.resolve()))
                os.replace(temporary, legacy)
        print(f"Ready: .secrets/{entry['name']} (values hidden)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    parser.add_argument("--staged", action="store_true", help="Check staged blobs, including content no longer in the working tree")
    arguments = parser.parse_args()
    try:
        return initialize() if arguments.command == "init" else check(staged=arguments.staged)
    except Exception:
        # Third-party error text may contain URLs or credentials; do not echo it.
        print("Local configuration operation failed; inspect paths/permissions privately. No values were printed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
