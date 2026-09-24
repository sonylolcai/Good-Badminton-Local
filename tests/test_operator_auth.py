import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from operator_api.main import app
from operator_api.services.auth import AuthService, hash_password, principal_has_permission, verify_password
from tests.test_operator_api import FakeDatabase


class OperatorAuthTests(unittest.TestCase):
    def test_passwords_use_scrypt_with_random_salt(self):
        first = hash_password("correct-horse-battery-staple")
        second = hash_password("correct-horse-battery-staple")

        self.assertNotEqual(first[0], second[0])
        self.assertTrue(verify_password("correct-horse-battery-staple", *first))
        self.assertFalse(verify_password("incorrect-password", *first))

    def test_platform_admin_is_global_and_venue_admin_is_scoped(self):
        platform = {"roles": [{"role": "platform_admin", "venue_id": None}]}
        venue = {"roles": [{"role": "venue_admin", "venue_id": "venue-1"}]}

        self.assertTrue(principal_has_permission(platform, "admins.manage"))
        self.assertTrue(principal_has_permission(venue, "players.manage", "venue-1"))
        self.assertFalse(principal_has_permission(venue, "players.manage", "venue-2"))
        self.assertFalse(principal_has_permission(venue, "admins.manage", "venue-1"))

    def test_short_password_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "12"):
            hash_password("too-short")

    def test_empty_database_bootstraps_one_forced_change_platform_admin(self):
        connection, cursor = MagicMock(), MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchone.return_value = {"count": 0}
        database = MagicMock()
        database._connect.return_value = connection

        with patch.dict(
            "os.environ",
            {
                "GOOD_BADMINTON_BOOTSTRAP_ADMIN_USERNAME": "system-admin",
                "GOOD_BADMINTON_BOOTSTRAP_ADMIN_PASSWORD": "bootstrap-password-01",
            },
            clear=True,
        ):
            created = AuthService(database).bootstrap_if_needed()

        self.assertTrue(created)
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("INSERT INTO business.admin_accounts" in statement for statement in statements))
        self.assertTrue(any("'platform_admin'" in statement for statement in statements))


class OperatorAuthApiTests(unittest.TestCase):
    def tearDown(self):
        app.state.auth_override = None
        app.state.auth_service_override = None

    def test_login_sets_an_http_only_strict_session_cookie(self):
        principal = {
            "id": "admin-1",
            "username": "system-admin",
            "must_change_password": True,
            "roles": [{"role": "platform_admin", "venue_id": None}],
        }
        service = MagicMock()
        service.login.return_value = ("secret-session", principal)
        app.state.auth_service_override = service

        response = TestClient(app).post(
            "/api/v1/auth/login",
            json={"username": "system-admin", "password": "bootstrap-password-01"},
        )

        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertNotIn("secret-session", response.text)

    def test_forced_password_change_blocks_business_data_but_allows_change(self):
        app.state.auth_override = {
            "id": "admin-1",
            "username": "system-admin",
            "must_change_password": True,
            "roles": [{"role": "platform_admin", "venue_id": None}],
        }
        service = MagicMock()
        app.state.auth_service_override = service

        blocked = TestClient(app).get("/api/v1/venues")
        changed = TestClient(app).post(
            "/api/v1/auth/change-password",
            json={"current_password": "bootstrap-password-01", "new_password": "replacement-password-01"},
        )

        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(changed.status_code, 200)
        service.change_password.assert_called_once()

    def test_venue_admin_is_rejected_outside_the_assigned_venue(self):
        app.state.auth_override = {
            "id": "admin-2",
            "username": "venue-admin",
            "must_change_password": False,
            "roles": [{"role": "venue_admin", "venue_id": "venue-1"}],
        }
        with patch("operator_api.main.get_db", return_value=FakeDatabase()):
            own = TestClient(app).get("/api/v1/venues/venue-1")
            other = TestClient(app).get("/api/v1/venues/venue-2")

        self.assertEqual(own.status_code, 200)
        self.assertEqual(other.status_code, 403)


if __name__ == "__main__":
    unittest.main()
