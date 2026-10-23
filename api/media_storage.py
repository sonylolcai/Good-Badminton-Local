"""Private, immutable media objects; credentials never enter archive manifests."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_key(key):
    if not isinstance(key, str) or not key or "\\" in key:
        raise ValueError("invalid media object key")
    parts = PurePosixPath(key)
    if parts.is_absolute() or any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError("invalid media object key")
    return key


class DirectoryObjectStore:
    """Explicit local test backend, never selected implicitly for OSS failures."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        path = (self.root / safe_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("object escapes storage root")
        return path

    def upload(self, key, source):
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".uploading")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise IOError("uploaded object checksum mismatch")
        os.replace(temporary, target)

    def download(self, key, destination):
        shutil.copyfile(self._path(key), destination)

    def head(self, key):
        path = self._path(key)
        return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


class OSSObjectStore:
    def __init__(self, *, endpoint, region, bucket, checkpoint_dir):
        from urllib.parse import urlsplit

        url = urlsplit(endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
            raise ValueError("OSS endpoint must be an HTTPS base address")
        if not region or not bucket:
            raise ValueError("OSS region and bucket are required")
        import oss2
        from oss2.credentials import EnvironmentVariableCredentialsProvider

        self.sdk = oss2
        self.bucket = oss2.Bucket(
            oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider()),
            endpoint, bucket, region=region,
        )
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, key, source):
        key = safe_key(key)
        digest = sha256_file(source)
        # SDK checks CRC; metadata plus HEAD verifies the published object identity.
        self.sdk.resumable_upload(
            self.bucket, key, str(source),
            store=self.sdk.ResumableStore(root=str(self.checkpoint_dir)),
            headers={"x-oss-meta-sha256": digest, "x-oss-object-acl": "private"},
            num_threads=2,
        )
        head = self.bucket.head_object(key)
        if (head.content_length != Path(source).stat().st_size
                or head.headers.get("x-oss-meta-sha256") != digest):
            raise IOError("uploaded object identity mismatch")

    def download(self, key, destination):
        self.sdk.resumable_download(
            self.bucket, safe_key(key), str(destination),
            store=self.sdk.ResumableDownloadStore(root=str(self.checkpoint_dir)),
            num_threads=2,
        )


    def head(self, key):
        head = self.bucket.head_object(safe_key(key))
        return {"size_bytes": head.content_length, "sha256": head.headers.get("x-oss-meta-sha256")}


def oss_from_environment(checkpoint_dir):
    from .dogecloud_storage import DogeCloudObjectStore, load_dogecloud_environment
    load_dogecloud_environment()
    provider = os.environ.get("GOOD_BADMINTON_MEDIA_STORAGE_PROVIDER", "aliyun")
    if provider == "dogecloud":
        return DogeCloudObjectStore.from_environment()
    if provider != "aliyun":
        raise ValueError("unknown media storage provider")
    return OSSObjectStore(
        endpoint=os.environ.get("GOOD_BADMINTON_OSS_ENDPOINT", ""),
        region=os.environ.get("GOOD_BADMINTON_OSS_REGION", ""),
        bucket=os.environ.get("GOOD_BADMINTON_OSS_BUCKET", ""),
        checkpoint_dir=checkpoint_dir,
    )
