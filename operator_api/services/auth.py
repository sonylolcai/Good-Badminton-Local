"""Password sessions and the fixed operator RBAC policy."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone


SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SESSION_COOKIE = "gb_admin_session"

ROLE_PERMISSIONS = {
    "platform_admin": {"*"},
    "venue_admin": {
        "venues.read",
        "players.manage",
        "resources.read",
        "resources.upload",
        "resources.delete",
        "analysis.create",
    },
}


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes, int, int, int]:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
        maxmem=128 * 1024 * 1024,
    )
    return salt, digest, SCRYPT_N, SCRYPT_R, SCRYPT_P


def verify_password(password: str, salt: bytes, digest: bytes, n: int, r: int, p: int) -> bool:
    candidate = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes(salt),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(digest),
        maxmem=128 * 1024 * 1024,
    )
    return secrets.compare_digest(candidate, bytes(digest))


def principal_has_permission(principal: dict, permission: str, venue_id: str | None = None) -> bool:
    assignments = principal.get("roles") or []
    if any(item.get("role") == "platform_admin" for item in assignments):
        return True
    return any(
        item.get("role") == "venue_admin"
        and permission in ROLE_PERMISSIONS["venue_admin"]
        and venue_id is not None
        and item.get("venue_id") == venue_id
        for item in assignments
    )


class AuthService:
    def __init__(self, database):
        self.database = database

    def bootstrap_if_needed(self) -> bool:
        username = os.environ.get("GOOD_BADMINTON_BOOTSTRAP_ADMIN_USERNAME", "").strip()
        password = os.environ.get("GOOD_BADMINTON_BOOTSTRAP_ADMIN_PASSWORD", "")
        with self.database._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('good-badminton-admin-bootstrap'))")
            cursor.execute("SELECT count(*) AS count FROM business.admin_accounts")
            if int(cursor.fetchone()["count"]) > 0:
                return False
            if not username or not password:
                raise RuntimeError(
                    "bootstrap admin credentials are required while business.admin_accounts is empty"
                )
            account_id = str(uuid.uuid4())
            salt, digest, n, r, p = hash_password(password)
            cursor.execute(
                "INSERT INTO business.admin_accounts "
                "(id, username, password_salt, password_digest, scrypt_n, scrypt_r, scrypt_p, must_change_password) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, true)",
                (account_id, username, salt, digest, n, r, p),
            )
            cursor.execute(
                "INSERT INTO business.admin_role_assignments (id, admin_account_id, role) "
                "VALUES (%s, %s, 'platform_admin')",
                (str(uuid.uuid4()), account_id),
            )
            self._audit(cursor, account_id, "admin.bootstrap_created", "admin_account", account_id, {})
        return True

    def login(self, username: str, password: str) -> tuple[str, dict]:
        with self.database._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT id::text, username, password_salt, password_digest, scrypt_n, scrypt_r, scrypt_p, "
                "status, must_change_password FROM business.admin_accounts WHERE lower(username) = lower(%s)",
                (username.strip(),),
            )
            row = cursor.fetchone()
            if row is None or row["status"] != "active":
                hash_password("invalid-password")
                raise PermissionError("invalid username or password")
            if not verify_password(
                password,
                row["password_salt"],
                row["password_digest"],
                row["scrypt_n"],
                row["scrypt_r"],
                row["scrypt_p"],
            ):
                raise PermissionError("invalid username or password")
            account_id = row["id"]
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=int(os.environ.get("GOOD_BADMINTON_ADMIN_SESSION_HOURS", "12"))
            )
            cursor.execute(
                "INSERT INTO business.admin_sessions "
                "(id, admin_account_id, token_digest, expires_at) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), account_id, _token_digest(token), expires_at),
            )
            cursor.execute(
                "UPDATE business.admin_accounts SET last_login_at = now(), updated_at = now() WHERE id = %s",
                (account_id,),
            )
        return token, self.authenticate(token)

    def authenticate(self, token: str) -> dict | None:
        if not token:
            return None
        with self.database._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT a.id::text, a.username, a.must_change_password "
                "FROM business.admin_sessions s JOIN business.admin_accounts a ON a.id = s.admin_account_id "
                "WHERE s.token_digest = %s AND s.revoked_at IS NULL AND s.expires_at > now() "
                "AND a.status = 'active'",
                (_token_digest(token),),
            )
            account = cursor.fetchone()
            if account is None:
                return None
            cursor.execute(
                "SELECT role, venue_id::text FROM business.admin_role_assignments "
                "WHERE admin_account_id = %s ORDER BY role, venue_id",
                (account["id"],),
            )
            roles = [dict(row) for row in cursor.fetchall()]
            cursor.execute(
                "UPDATE business.admin_sessions SET last_used_at = now() WHERE token_digest = %s",
                (_token_digest(token),),
            )
        return {
            "id": account["id"],
            "username": account["username"],
            "must_change_password": bool(account["must_change_password"]),
            "roles": roles,
        }

    def logout(self, token: str) -> None:
        with self.database._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE business.admin_sessions SET revoked_at = now() "
                "WHERE token_digest = %s AND revoked_at IS NULL",
                (_token_digest(token),),
            )

    def change_password(self, principal: dict, current_password: str, new_password: str) -> None:
        with self.database._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT password_salt, password_digest, scrypt_n, scrypt_r, scrypt_p "
                "FROM business.admin_accounts WHERE id = %s AND status = 'active'",
                (principal["id"],),
            )
            row = cursor.fetchone()
            if row is None or not verify_password(
                current_password,
                row["password_salt"],
                row["password_digest"],
                row["scrypt_n"],
                row["scrypt_r"],
                row["scrypt_p"],
            ):
                raise PermissionError("current password is invalid")
            salt, digest, n, r, p = hash_password(new_password)
            cursor.execute(
                "UPDATE business.admin_accounts SET password_salt=%s, password_digest=%s, scrypt_n=%s, "
                "scrypt_r=%s, scrypt_p=%s, must_change_password=false, updated_at=now() WHERE id=%s",
                (salt, digest, n, r, p, principal["id"]),
            )
            self._audit(cursor, principal["id"], "admin.password_changed", "admin_account", principal["id"], {})

    def list_admins(self) -> list[dict]:
        with self.database._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT a.id::text, a.username, a.status, a.must_change_password, r.role, r.venue_id::text "
                "FROM business.admin_accounts a JOIN business.admin_role_assignments r ON r.admin_account_id=a.id "
                "ORDER BY a.created_at, r.role, r.venue_id"
            )
            return [dict(row) for row in cursor.fetchall()]

    def create_admin(self, actor: dict, username: str, password: str, role: str, venue_id: str | None) -> dict:
        if not principal_has_permission(actor, "admins.manage"):
            raise PermissionError("platform administrator permission is required")
        if role not in ROLE_PERMISSIONS or (role == "platform_admin") != (venue_id is None):
            raise ValueError("role and venue scope do not match")
        account_id = str(uuid.uuid4())
        salt, digest, n, r, p = hash_password(password)
        with self.database._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO business.admin_accounts "
                "(id, username, password_salt, password_digest, scrypt_n, scrypt_r, scrypt_p, must_change_password) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, true)",
                (account_id, username.strip(), salt, digest, n, r, p),
            )
            cursor.execute(
                "INSERT INTO business.admin_role_assignments (id, admin_account_id, role, venue_id) "
                "VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), account_id, role, venue_id),
            )
            self._audit(cursor, actor["id"], "admin.created", "admin_account", account_id, {"role": role, "venue_id": venue_id})
        return {"id": account_id, "username": username.strip(), "role": role, "venue_id": venue_id}

    @staticmethod
    def _audit(cursor, actor_id: str, action: str, resource_type: str, resource_id: str, summary: dict) -> None:
        cursor.execute(
            "INSERT INTO business.audit_events "
            "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, after_summary) "
            "VALUES (%s, 'admin', %s, %s, %s, %s, %s::jsonb)",
            (str(uuid.uuid4()), actor_id, action, resource_type, resource_id, json.dumps(summary, ensure_ascii=False)),
        )


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
