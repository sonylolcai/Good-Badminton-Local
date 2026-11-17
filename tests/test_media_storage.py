"""OSS SDK adapter contract checks; all HTTP operations are replaced locally."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api.media_storage import DirectoryObjectStore, OSSObjectStore, safe_key, sha256_file


class MediaStorageTests(unittest.TestCase):
    def test_object_keys_cannot_escape_local_cache_or_become_absolute_paths(self):
        with tempfile.TemporaryDirectory() as root:
            storage = DirectoryObjectStore(root)
            for key in ("../secret", "/secret", "a/../../secret", "a\\secret", "a//b", "a/./b", ""):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    storage._path(key)

    @unittest.skipUnless(importlib.util.find_spec("oss2"), "install requirements-media.txt for OSS SDK tests")
    def test_real_sdk_uses_v4_private_resumable_upload_and_checks_head(self):
        import oss2
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "OSS_ACCESS_KEY_ID": "test-only-identity", "OSS_ACCESS_KEY_SECRET": "test-only-secret",
        }):
            store = OSSObjectStore(endpoint="https://oss-cn-hangzhou.aliyuncs.com", region="cn-hangzhou",
                                   bucket="test-only-bucket", checkpoint_dir=Path(root) / "checkpoints")
            self.assertIsInstance(store.bucket.auth, oss2.ProviderAuthV4)
            source = Path(root) / "input.mp4"
            source.write_bytes(b"local adapter contract test")
            digest = sha256_file(source)
            head = SimpleNamespace(content_length=source.stat().st_size, headers={"x-oss-meta-sha256": digest})
            with patch.object(oss2, "resumable_upload") as upload, patch.object(store.bucket, "head_object", return_value=head):
                store.upload("prefix/video.mp4", source)
                call = upload.call_args
                self.assertEqual(call.args[1], "prefix/video.mp4")
                self.assertIsInstance(call.kwargs["store"], oss2.ResumableStore)
                self.assertEqual(call.kwargs["headers"]["x-oss-object-acl"], "private")
                self.assertEqual(call.kwargs["headers"]["x-oss-meta-sha256"], digest)
                head.content_length += 1
                with self.assertRaisesRegex(IOError, "identity mismatch"):
                    store.upload("prefix/video.mp4", source)
            destination = Path(root) / "download.mp4"
            with patch.object(oss2, "resumable_download") as download:
                store.download("prefix/video.mp4", destination)
                self.assertIsInstance(download.call_args.kwargs["store"], oss2.ResumableDownloadStore)

    def test_oss_endpoint_rejects_embedded_credentials_and_plain_http(self):
        for endpoint in ("http://example.com", "https://user:secret@example.com", "https://example.com?token=x"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                OSSObjectStore(endpoint=endpoint, region="test", bucket="test", checkpoint_dir="unused")


if __name__ == "__main__":
    unittest.main()
