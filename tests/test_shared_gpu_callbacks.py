import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from api.result_notifications import ResultNotifications
from good_badminton_contracts.gpu_notifications import signature


class SharedCallbackTests(unittest.TestCase):
    def box(self, **kwargs):
        return ResultNotifications(Mock(), Mock(), url='https://legacy.example.test/callback',
                                   secret='test-callback-secret-at-least-32-bytes',
                                   clock=lambda: 1000, **kwargs)

    def test_routes_by_signed_business_identity_and_preserves_signature(self):
        urls={'test':'https://dev.example.test/callback',
              'production':'https://production.example.test/callback'}
        box=self.box(environment_urls=urls)
        opener=MagicMock()
        response=opener.open.return_value.__enter__.return_value
        response.status=202
        response.read.return_value=b'{"status":"accepted","event_id":"event-1"}'
        with patch('api.result_notifications.build_opener',return_value=opener):
            for environment,target in urls.items():
                payload={'event_id':'event-1','media_identity':{'environment':environment}}
                box._send(payload)
                request=opener.open.call_args.args[0]
                self.assertEqual(request.full_url,target)
                self.assertEqual(json.loads(request.data),payload)
                self.assertEqual(request.get_header('X-gpu-signature'),
                                 signature(box.secret,'1000',request.data))
        self.assertEqual(opener.open.call_count,2)

    def test_unknown_environment_never_falls_back_to_another_database(self):
        box=self.box(environment_urls={'production':'https://production.example.test/callback'})
        with patch('api.result_notifications.build_opener') as opener:
            for identity in ({}, {'environment':'test'}, {'environment':'other'}):
                with self.subTest(identity=identity),self.assertRaises(ValueError):
                    box._send({'event_id':'event-1','media_identity':identity})
            opener.assert_not_called()

    def test_routed_urls_keep_the_existing_https_and_no_credentials_rules(self):
        for target in ('http://dev.example.test/callback', 'https://user:secret@dev.example.test/',
                       'https://dev.example.test/callback?token=secret'):
            with self.subTest(target=target),self.assertRaises(ValueError):
                self.box(environment_urls={'test':target})
        with self.assertRaises(ValueError):
            self.box(environment_urls={'other':'https://other.example.test/callback'})
