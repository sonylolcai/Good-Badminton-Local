import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from api.dogecloud_storage import DogeCloudObjectStore
from api.media_storage import sha256_file


class DogeCloudStorageTests(unittest.TestCase):
    def setUp(self):
        self.store=DogeCloudObjectStore(access_key='test-only',secret_key='test-only',bucket='test-only',allowed_root='testVideo')
        self.client=Mock();self.store._cached=(time.monotonic()+100,self.client,'private-test-bucket')
        self.client.get_object_acl.return_value={'Grants':[{'Grantee':{'Type':'CanonicalUser'}}]}

    def test_rejects_traversal_encoded_paths_controls_and_other_environment(self):
        for key in ('productionVideo/haoyu/a','testVideo/../a','testVideo/%2e%2e/a','testVideo/a\\b','testVideo/a\nsecret','testVideo//a'):
            with self.subTest(key=key),self.assertRaises(ValueError):self.store._key(key)
        self.client.upload_file.assert_not_called()

    def test_media_test_environment_is_explicit_and_keeps_production_application_auth(self):
        with patch('api.dogecloud_storage.load_dogecloud_environment'), patch.dict('os.environ',{
            'GOOD_BADMINTON_APP_ENV':'production','DOGECLOUD_MEDIA_ENVIRONMENT':'test',
            'DOGECLOUD_ACCESS_KEY':'test-only','DOGECLOUD_SECRET_KEY':'test-only','DOGECLOUD_BUCKET':'test-only',
        },clear=True):
            store=DogeCloudObjectStore.from_environment()
            self.assertEqual(store.allowed_root,'testVideo')
            with self.assertRaises(ValueError):store._key('productionVideo/other/file.mp4')
        self.client.generate_presigned_url.assert_not_called()

    def test_private_upload_checks_head_and_is_idempotent_without_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'video.mp4';source.write_bytes(b'video')
            good={'ContentLength':5,'Metadata':{'sha256':sha256_file(source)}}
            missing=ClientError({'Error':{'Code':'NoSuchKey'},'ResponseMetadata':{'HTTPStatusCode':404}},'HeadObject')
            self.client.head_object.side_effect=[missing,good]
            self.store.upload('testVideo/haoyu/master.mp4',source)
            self.assertEqual(self.client.upload_file.call_args.kwargs['ExtraArgs']['ACL'],'private')
            self.client.head_object.side_effect=None;self.client.head_object.return_value=good
            self.store.upload('testVideo/haoyu/master.mp4',source)
            self.client.upload_file.assert_called_once()
            source.write_bytes(b'changed')
            with self.assertRaises(FileExistsError):self.store.upload('testVideo/haoyu/master.mp4',source)
            self.client.upload_file.assert_called_once()

    def test_public_grants_and_long_lived_links_are_rejected(self):
        for kind in ('Group','AmazonCustomerByEmail'):
            with self.assertRaises(PermissionError):self.store._private_acl({'Grants':[{'Grantee':{'Type':kind}}]})
        for ttl in (0,301,True):
            with self.assertRaises(ValueError):self.store.download_url('testVideo/haoyu/master.mp4',expires=ttl)
        self.store.download_url('testVideo/haoyu/master.mp4',expires=60)
        self.assertEqual(self.client.generate_presigned_url.call_args.kwargs['ExpiresIn'],60)


if __name__=='__main__':unittest.main()
