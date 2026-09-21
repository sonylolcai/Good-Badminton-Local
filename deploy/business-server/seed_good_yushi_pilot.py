"""Idempotently register the first venue, court and edge binding.

The terminal secret is deliberately discarded here. Retrieve it once through
the protected operator binding endpoint and write it directly to the venue
relay host's 0600 environment file.
"""

from __future__ import annotations

import json

from operator_api.services.operator_backoffice import BusinessDatabase


# These are opaque UUIDv4 identifiers, not sequential business numbers.  The
# human-readable identifiers used by operators are the venue/court ``code``
# fields below.  Keeping the two separate means a renamed court never changes
# the identifier that is signed by an edge relay.
TENANT_ID = "64c28b72-1bd7-4c09-b73a-5cc8c43d33a3"
VENUE_ID = "d73a28e0-f4cb-4cfc-9ece-9424d252c10d"
COURT_ID = "84b70d2e-8e5e-49bd-a6e7-a506aa68c7cc"
DEVICE_ID = "d958591c-ddcd-46ec-a47a-dd449cb9c9eb"
CAMERA_ID = "2e5b3f68-01d1-4168-9b06-2440aeb68b4e"

# Only the pre-production numeric-looking sample records may be retired.  Do
# not generalize this cleanup to arbitrary tenant data.
LEGACY_TENANT_ID = "66666666-6666-6666-6666-666666666666"
LEGACY_VENUE_ID = "77777777-7777-7777-7777-777777777777"
LEGACY_COURT_ID = "88888888-8888-8888-8888-888888888888"
LEGACY_DEVICE_ID = "99999999-9999-9999-9999-999999999999"
LEGACY_CAMERA_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def retire_legacy_pilot(database: BusinessDatabase) -> None:
    """Replace the old seeded IDs only while they have never ingested video.

    A real session is evidence that this is no longer disposable demo data;
    failing closed prevents an ID migration from severing its audit trail.
    """

    with database._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """select count(*) as session_count from business.edge_ingest_sessions
               where camera_id = %s or edge_device_id = %s""",
            (LEGACY_CAMERA_ID, LEGACY_DEVICE_ID),
        )
        if cursor.fetchone()["session_count"]:
            raise RuntimeError(
                "Legacy pilot IDs already have ingest sessions; do not reseed. "
                "Migrate the active deployment with an explicit data migration."
            )

        # Audit records created by this disposable seed have nullable scopes.
        # Removing them avoids leaving references to a pilot tenant that no
        # longer exists, without touching any other venue's audit history.
        cursor.execute(
            """delete from business.audit_events
               where tenant_id = %s or venue_id = %s
                  or resource_id in (%s, %s, %s)""",
            (LEGACY_TENANT_ID, LEGACY_VENUE_ID, LEGACY_CAMERA_ID, LEGACY_DEVICE_ID, LEGACY_COURT_ID),
        )
        cursor.execute("delete from business.cameras where id = %s", (LEGACY_CAMERA_ID,))
        cursor.execute("delete from business.edge_devices where id = %s", (LEGACY_DEVICE_ID,))
        cursor.execute("delete from business.courts where id = %s", (LEGACY_COURT_ID,))
        cursor.execute("delete from business.venues where id = %s", (LEGACY_VENUE_ID,))
        cursor.execute("delete from business.tenants where id = %s", (LEGACY_TENANT_ID,))


def main() -> None:
    database = BusinessDatabase()
    retire_legacy_pilot(database)
    with database._connect() as connection, connection.cursor() as cursor:  # explicit seed transaction
        cursor.execute(
            """insert into business.tenants (id, name, status) values (%s, %s, 'active')
               on conflict (id) do update set name=excluded.name, status='active'""",
            (TENANT_ID, "好雨时节试点"),
        )
        cursor.execute(
            """insert into business.venues (id, tenant_id, code, name, timezone, status)
               values (%s, %s, %s, %s, 'Asia/Shanghai', 'active')
               on conflict (id) do update set name=excluded.name, code=excluded.code, status='active'""",
            (VENUE_ID, TENANT_ID, "haoyushijie-venue-01", "好雨时节球馆"),
        )
        cursor.execute(
            """insert into business.courts (id, venue_id, code, name, sort_order, status)
               values (%s, %s, 'court-01', '一号场', 0, 'active')
               on conflict (id) do update set name=excluded.name, code=excluded.code, status='active'""",
            (COURT_ID, VENUE_ID),
        )
    binding = database.provision_edge_camera(
        DEVICE_ID, CAMERA_ID, VENUE_ID, COURT_ID,
        "haoyushijie-gateway-01", "haoyushijie-cam-01", "v1",
    )
    print(json.dumps({key: binding[key] for key in ("schema_version", "device_id", "camera_id", "credential_version")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
