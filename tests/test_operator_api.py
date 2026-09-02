import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from operator_api.main import _edge_gateway_base_url, _edge_gateway_internal_url, app
from operator_api.services.operator_backoffice import _format_china_time


class FakeDatabase:
    def readiness(self):
        return {"status": "ready", "database": "test"}

    def list_tenants(self):
        return [["tenant-1", "试点租户", "active"]]

    def list_venues(self):
        return [["venue-1", "tenant-1", "venue-code", "测试球馆", "Asia/Shanghai", "", "active", "1"]]

    def list_courts(self):
        return [["court-1", "venue-1", "court-01", "一号场", "0", "active"]]

    def register_venue(self, **kwargs):
        return {
            "tenant": {"id": "tenant-2", "name": kwargs["tenant_name"], "created": True},
            "venue": {"id": "venue-2", "tenant_id": "tenant-2", "code": kwargs["venue_code"], "name": kwargs["venue_name"], "timezone": "Asia/Shanghai", "address": None, "status": "active"},
            "courts": [{"id": "court-2", **kwargs["courts"][0]}],
        }

    def set_court_status(self, venue_id, court_id, status):
        self.status_update = (venue_id, court_id, status)

    def save_court(self, *_):
        return "court-1"

    def venue_control_snapshot(self, _):
        return {"venue_name": "测试球馆", "venue_code": "venue-code", "total_courts": 1, "active_courts": 1, "terminal_online": 0, "camera_online": 0, "analysis_active": 0, "rows": []}

    def venue_live_operations(self, _):
        return {"camera_connected": 0, "active_cases": 0, "courts": [{
            "court": {"id": "court-1", "code": "court-01", "name": "一号场", "status": "active"},
            "camera": {"connected": False}, "case": None,
        }]}

    def set_case_gpu_forwarding(self, venue_id, court_id, enabled):
        self.gpu_forwarding_update = (venue_id, court_id, enabled)
        return {"case_id": "case-1", "gpu_forwarding_enabled": enabled, "status": "receiving"}

    def set_court_capture_mode(self, venue_id, court_id, mode):
        self.capture_update = (venue_id, court_id, mode)
        return {"court_id": court_id, "mode": mode, "revision": 1, "updated_at": "2026-09-01T00:00:00Z"}

    def case_event_log(self, case_id, limit):
        return [{"event_id": "saved-1", "source": "gpu", "message": "saved", "payload": {}}]

    def provision_edge_camera(self, *_):
        return {"device_id": "device-1", "camera_id": "camera-1", "device_secret": "one-time"}


class OperatorApiTests(unittest.TestCase):
    def setUp(self):
        self.database = FakeDatabase()
        self.patch = patch("operator_api.main.get_db", return_value=self.database)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(app)

    def test_venue_list_is_named_json_not_positional_rows(self):
        response = self.client.get("/api/v1/venues")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["venues"][0]["name"], "测试球馆")
        self.assertNotIsInstance(response.json()["venues"][0], list)

    def test_registration_contract_creates_venue_with_courts(self):
        response = self.client.post("/api/v1/venue-registrations", json={
            "tenant_name": "新租户", "venue_code": "new-venue", "venue_name": "新球馆",
            "courts": [{"code": "court-01", "name": "一号场", "sort_order": 0, "status": "active"}],
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["registration"]["venue"]["id"], "venue-2")

    def test_registration_rejects_empty_court_list_before_write(self):
        response = self.client.post("/api/v1/venue-registrations", json={
            "tenant_name": "新租户", "venue_code": "new-venue", "venue_name": "新球馆", "courts": [],
        })
        self.assertEqual(response.status_code, 422)

    def test_court_status_is_a_patch_with_named_response(self):
        response = self.client.patch("/api/v1/venues/venue-1/courts/court-1/status", json={"status": "maintenance"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["court"]["id"], "court-1")
        self.assertEqual(self.database.status_update, ("venue-1", "court-1", "maintenance"))

    def test_operations_return_named_court_records(self):
        response = self.client.get("/api/v1/venues/venue-1/operations")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["total_courts"], 1)
        self.assertIsNone(response.json()["courts"][0]["case"])

    def test_gpu_forwarding_is_controlled_by_venue_and_court(self):
        response = self.client.post("/api/v1/venues/venue-1/courts/court-1/case/gpu-forwarding", json={"enabled": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["case"]["gpu_forwarding_enabled"])
        self.assertEqual(self.database.gpu_forwarding_update, ("venue-1", "court-1", True))

    def test_capture_is_controlled_by_business_console_not_the_venue_mac(self):
        response = self.client.post("/api/v1/venues/venue-1/courts/court-1/capture", json={"mode": "preview"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["capture"]["mode"], "preview")
        self.assertEqual(self.database.capture_update, ("venue-1", "court-1", "preview"))

    def test_public_preview_origin_is_separate_from_internal_edge_origin(self):
        with patch.dict("os.environ", {
            "GOOD_BADMINTON_EDGE_GATEWAY_PUBLIC_URL": "https://api.example.com/",
            "GOOD_BADMINTON_EDGE_GATEWAY_INTERNAL_URL": "http://127.0.0.1:18080/",
        }, clear=False):
            self.assertEqual(_edge_gateway_base_url(), "https://api.example.com")
            self.assertEqual(_edge_gateway_internal_url(), "http://127.0.0.1:18080")

    def test_operator_display_time_is_china_standard_time_while_storage_stays_utc(self):
        stored_utc = datetime(2026, 9, 1, 16, 30, 45, tzinfo=timezone.utc)
        self.assertEqual(_format_china_time(stored_utc), "2026-09-02 00:30:45")
        self.assertEqual(_format_china_time("2026-09-01 16:30:45+00"), "2026-09-02 00:30:45")


if __name__ == "__main__":
    unittest.main()
