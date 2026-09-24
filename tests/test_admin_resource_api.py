import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from operator_api.main import app
from tests.test_operator_api import FakeDatabase


class AdminResourceApiContractTests(unittest.TestCase):
    def setUp(self):
        self.database = FakeDatabase()
        self.database_patch = patch("operator_api.main.get_db", return_value=self.database)
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.auth_patch = patch("operator_api.main.get_auth_service", return_value=_RejectingAuth())
        self.auth_patch.start()
        self.addCleanup(self.auth_patch.stop)
        app.state.auth_override = None
        self.client = TestClient(app)

    def test_existing_business_data_requires_an_admin_session(self):
        response = self.client.get("/api/v1/venues")

        self.assertEqual(response.status_code, 401)

    def test_invalid_password_login_is_rejected_as_credentials_not_missing_route(self):
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "unknown", "password": "wrong"},
        )

        self.assertEqual(response.status_code, 401)

    def test_resource_inventory_requires_an_admin_session(self):
        response = self.client.get("/api/v1/resources")

        self.assertEqual(response.status_code, 401)

    def test_retention_settings_require_a_platform_admin_session(self):
        response = self.client.get("/api/v1/settings/video-retention")

        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()


class _RejectingAuth:
    def login(self, _username, _password):
        raise PermissionError("invalid")
