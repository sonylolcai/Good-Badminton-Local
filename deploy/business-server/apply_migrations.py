"""Apply business-gateway SQL migrations exactly once on the business DB."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "business_gateway" / "migrations"


def main() -> None:
    database_url = os.environ.get("GOOD_BADMINTON_BUSINESS_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("GOOD_BADMINTON_BUSINESS_DATABASE_URL is required")

    import psycopg

    migration_files = sorted(MIGRATIONS.glob("*.sql"))
    if not migration_files:
        raise SystemExit(f"No SQL migrations found in {MIGRATIONS}")

    with psycopg.connect(database_url, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute("create schema if not exists business")
            cursor.execute(
                """create table if not exists business.schema_migrations (
                       version text primary key,
                       description text not null default '',
                       applied_at timestamptz not null default now()
                   )"""
            )
            cursor.execute(
                "alter table business.schema_migrations "
                "add column if not exists description text not null default ''"
            )
            for migration_path in migration_files:
                version = migration_path.name
                cursor.execute("select 1 from business.schema_migrations where version=%s", (version,))
                if cursor.fetchone():
                    print(f"skip {version}")
                    continue
                cursor.execute(migration_path.read_text(encoding="utf-8"))
                cursor.execute(
                    "insert into business.schema_migrations (version, description) values (%s, %s)",
                    (version, migration_path.stem),
                )
                print(f"apply {version}")
        connection.commit()


if __name__ == "__main__":
    main()
