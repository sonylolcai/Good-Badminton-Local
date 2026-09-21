import os
import unittest
from unittest.mock import patch

from runtime_config import RuntimeConfigurationError, business_api_base_url, business_service_listener, webui_listener


class RuntimeConfigurationTests(unittest.TestCase):
    def test_development_defaults_to_local_business_port_8080(self):
        with patch.dict(os.environ, {"GOOD_BADMINTON_APP_ENV": "development"}, clear=True):
            self.assertEqual(business_api_base_url(), "http://127.0.0.1:8080")
            self.assertEqual(business_service_listener(), ("127.0.0.1", 8080))
            self.assertEqual(webui_listener(), ("127.0.0.1", 7861))

    def test_production_requires_a_non_local_https_business_endpoint(self):
        with patch.dict(os.environ, {"GOOD_BADMINTON_APP_ENV": "production"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "required in production"):
                business_api_base_url()
        with patch.dict(
            os.environ,
            {
                "GOOD_BADMINTON_APP_ENV": "production",
                "GOOD_BADMINTON_BUSINESS_API_URL": "http://127.0.0.1:8080",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "non-local HTTPS"):
                business_api_base_url()

    def test_production_accepts_deployment_endpoint_and_overrides_listener(self):
        with patch.dict(
            os.environ,
            {
                "GOOD_BADMINTON_APP_ENV": "production",
                "GOOD_BADMINTON_BUSINESS_API_URL": "https://api.badminton.example.com/",
                "GOOD_BADMINTON_BUSINESS_HOST": "0.0.0.0",
                "GOOD_BADMINTON_BUSINESS_PORT": "9080",
                "GOOD_BADMINTON_WEBUI_HOST": "10.0.0.5",
                "GOOD_BADMINTON_WEBUI_PORT": "7862",
            },
            clear=True,
        ):
            self.assertEqual(business_api_base_url(), "https://api.badminton.example.com")
            self.assertEqual(business_service_listener(), ("0.0.0.0", 9080))
            self.assertEqual(webui_listener(), ("10.0.0.5", 7862))


if __name__ == "__main__":
    unittest.main()
