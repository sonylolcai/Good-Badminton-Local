"""DogeCloud server-side S3 adapter. No permanent credentials in URLs or manifests."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit


def load_dogecloud_environment():
    # Each project owns its private settings. Explicit process values win.
    path = Path(os.environ.get("DOGECLOUD_ENV_FILE", "")).expanduser() if os.environ.get("DOGECLOUD_ENV_FILE") else Path(__file__).resolve().parents[1] / ".secrets/dogecloud.env"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            if key.startswith("DOGECLOUD_") or key in {"GOOD_BADMINTON_MEDIA_REGISTRY_ENABLED", "GOOD_BADMINTON_MEDIA_STORAGE_PROVIDER"}:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '\"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)


def _safe_key(key):
    import re
    if not isinstance(key, str) or len(key) > 1024 or not key or any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("invalid media object key")
    if "\\" in key or any(part in {"", ".", ".."} for part in key.split("/")) or re.search(r"%[0-9a-fA-F]{2}", key):
        raise ValueError("invalid media object key")
    return key


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DogeCloudObjectStore:
    def __init__(self, *, access_key, secret_key, bucket, allowed_root):
        if not all((access_key, secret_key, bucket)):
            raise ValueError("DogeCloud credentials and bucket are required")
        if allowed_root not in {"testVideo", "productionVideo"}:
            raise ValueError("DogeCloud storage root must isolate test and production")
        self._access_key, self._secret_key, self._bucket = access_key, secret_key, bucket
        self.allowed_root = allowed_root
        self._cached = None
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, *, environment=None):
        load_dogecloud_environment()
        app_env = os.environ.get("GOOD_BADMINTON_APP_ENV", "development")
        if app_env not in {"development", "test", "production"}:
            raise ValueError("invalid application environment")
        environment = environment or os.environ.get("DOGECLOUD_MEDIA_ENVIRONMENT") or ("production" if app_env == "production" else "test")
        if environment not in {"test", "production"}:
            raise ValueError("invalid media environment")
        return cls(access_key=os.environ.get("DOGECLOUD_ACCESS_KEY", ""),
                   secret_key=os.environ.get("DOGECLOUD_SECRET_KEY", ""), bucket=os.environ.get("DOGECLOUD_BUCKET", ""),
                   allowed_root="productionVideo" if environment == "production" else "testVideo")

    def _key(self, key):
        key = _safe_key(key)
        if not key.startswith(self.allowed_root + "/"):
            raise ValueError("object key is outside this environment's media root")
        return key

    @staticmethod
    def _private_acl(acl):
        for grant in acl.get("Grants", []):
            # Only canonical account grants are acceptable. Group grants can expose recordings.
            if grant.get("Grantee", {}).get("Type") != "CanonicalUser":
                raise PermissionError("private media storage rejects group or public ACL grants")

    def _client(self):
        with self._lock:
            if self._cached and time.monotonic() < self._cached[0]:
                return self._cached[1:]
            import requests
            import boto3
            from botocore.config import Config
            path = "/auth/tmp_token.json"
            body = json.dumps({"channel": "OSS_FULL", "scopes": [self._bucket]}, separators=(",", ":"))
            signature = hmac.new(self._secret_key.encode(), (path + "\n" + body).encode(), hashlib.sha1).hexdigest()
            response = requests.post("https://api.dogecloud.com" + path, data=body.encode(),
                headers={"Authorization": "TOKEN " + self._access_key + ":" + signature,
                         "Content-Type": "application/json"}, timeout=(10, 30), allow_redirects=False)
            if response.status_code != 200:
                raise PermissionError("DogeCloud authentication failed")
            value = response.json()
            if value.get("code") != 200:
                raise PermissionError("DogeCloud authentication or bucket scope failed")
            data = value["data"]
            if len(data["Buckets"]) != 1:
                raise ValueError("expected exactly one scoped storage bucket")
            bucket = data["Buckets"][0]
            endpoint = urlsplit(bucket["s3Endpoint"])
            if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in {"", "/"}:
                raise ValueError("invalid DogeCloud S3 endpoint")
            credentials = data["Credentials"]
            client = boto3.client("s3", endpoint_url=bucket["s3Endpoint"], region_name="us-east-1",
                aws_access_key_id=credentials["accessKeyId"], aws_secret_access_key=credentials["secretAccessKey"],
                aws_session_token=credentials["sessionToken"],
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"},
                    request_checksum_calculation="when_required", response_checksum_validation="when_required",
                    connect_timeout=10, read_timeout=60, retries={"max_attempts": 2}))
            self._private_acl(client.get_bucket_acl(Bucket=bucket["s3Bucket"]))
            self._cached = (time.monotonic() + 45, client, bucket["s3Bucket"])
            return self._cached[1:]

    def upload(self, key, source):
        from botocore.exceptions import ClientError
        import mimetypes
        key = self._key(key)
        digest, size = _sha256(source), Path(source).stat().st_size
        client, bucket = self._client()
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
                raise
        else:
            if head.get("ContentLength") != size or head.get("Metadata", {}).get("sha256") != digest:
                raise FileExistsError("immutable media object has different content")
            self._private_acl(client.get_object_acl(Bucket=bucket, Key=key))
            return
        client.upload_file(str(source), bucket, key, ExtraArgs={"ACL": "private", "Metadata": {"sha256": digest},
            "ContentType": mimetypes.guess_type(str(source))[0] or "application/octet-stream"})
        head = client.head_object(Bucket=bucket, Key=key)
        if head.get("ContentLength") != size or head.get("Metadata", {}).get("sha256") != digest:
            raise IOError("uploaded object identity mismatch")
        self._private_acl(client.get_object_acl(Bucket=bucket, Key=key))

    def download(self, key, destination):
        key = self._key(key)
        client, bucket = self._client()
        client.download_file(bucket, key, str(destination))

    def head(self, key):
        key = self._key(key)
        client, bucket = self._client()
        head = client.head_object(Bucket=bucket, Key=key)
        return {"size_bytes": head["ContentLength"], "sha256": head.get("Metadata", {}).get("sha256")}

    def download_url(self, key, *, expires=60):
        if type(expires) is not int or not 1 <= expires <= 300:
            raise ValueError("media download link lifetime must be 1-300 seconds")
        key = self._key(key)
        client, bucket = self._client()
        return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key,
            "ResponseContentDisposition": "attachment"}, ExpiresIn=expires)
