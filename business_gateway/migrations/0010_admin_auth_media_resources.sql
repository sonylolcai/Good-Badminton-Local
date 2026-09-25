BEGIN;

CREATE TABLE IF NOT EXISTS business.admin_accounts (
  id uuid PRIMARY KEY,
  username text NOT NULL,
  password_salt bytea NOT NULL,
  password_digest bytea NOT NULL,
  scrypt_n integer NOT NULL,
  scrypt_r integer NOT NULL,
  scrypt_p integer NOT NULL,
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  must_change_password boolean NOT NULL DEFAULT true,
  wechat_openid text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  last_login_at timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS admin_accounts_username_unique
  ON business.admin_accounts (lower(username));
CREATE UNIQUE INDEX IF NOT EXISTS admin_accounts_wechat_openid_unique
  ON business.admin_accounts (wechat_openid) WHERE wechat_openid IS NOT NULL;

CREATE TABLE IF NOT EXISTS business.admin_role_assignments (
  id uuid PRIMARY KEY,
  admin_account_id uuid NOT NULL REFERENCES business.admin_accounts(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('platform_admin', 'venue_admin')),
  venue_id uuid REFERENCES business.venues(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (
    (role = 'platform_admin' AND venue_id IS NULL)
    OR (role = 'venue_admin' AND venue_id IS NOT NULL)
  )
);
CREATE UNIQUE INDEX IF NOT EXISTS admin_platform_role_unique
  ON business.admin_role_assignments (admin_account_id, role)
  WHERE venue_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS admin_venue_role_unique
  ON business.admin_role_assignments (admin_account_id, role, venue_id)
  WHERE venue_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS business.admin_sessions (
  id uuid PRIMARY KEY,
  admin_account_id uuid NOT NULL REFERENCES business.admin_accounts(id) ON DELETE CASCADE,
  token_digest char(64) NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  last_used_at timestamptz NOT NULL DEFAULT now(),
  revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS admin_sessions_account_active_idx
  ON business.admin_sessions (admin_account_id, expires_at DESC)
  WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS business.managed_media_resources (
  id uuid PRIMARY KEY,
  tenant_id uuid REFERENCES business.tenants(id),
  venue_id uuid REFERENCES business.venues(id),
  player_id uuid REFERENCES business.users(id),
  match_id uuid REFERENCES business.matches(id),
  analysis_job_id uuid REFERENCES business.analysis_jobs(id),
  asset_type text NOT NULL CHECK (asset_type IN ('video', 'json', 'image', 'other')),
  media_type text NOT NULL,
  original_filename text,
  upload_succeeded_at timestamptz NOT NULL,
  status text NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'deleting', 'resources_deleted', 'delete_failed', 'deleted')),
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS managed_media_resources_venue_uploaded_idx
  ON business.managed_media_resources (venue_id, upload_succeeded_at DESC);
CREATE INDEX IF NOT EXISTS managed_media_resources_video_retention_idx
  ON business.managed_media_resources (upload_succeeded_at)
  WHERE asset_type = 'video' AND status IN ('active', 'delete_failed');

CREATE TABLE IF NOT EXISTS business.managed_media_resource_locations (
  id uuid PRIMARY KEY,
  media_asset_id uuid NOT NULL REFERENCES business.managed_media_resources(id) ON DELETE CASCADE,
  storage_backend text NOT NULL CHECK (storage_backend IN ('local_disk', 'gpu_http', 'object_storage')),
  location_ref text NOT NULL,
  object_key text,
  sha256 char(64),
  size_bytes bigint CHECK (size_bytes IS NULL OR size_bytes >= 0),
  deletion_status text NOT NULL DEFAULT 'active'
    CHECK (deletion_status IN ('active', 'deleting', 'deleted', 'failed')),
  deletion_error text,
  deleted_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (media_asset_id, storage_backend, location_ref)
);
CREATE INDEX IF NOT EXISTS managed_media_resource_locations_retry_idx
  ON business.managed_media_resource_locations (media_asset_id, deletion_status);

CREATE TABLE IF NOT EXISTS business.video_retention_policy (
  id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  enabled boolean NOT NULL DEFAULT false,
  retention_days integer NOT NULL DEFAULT 7 CHECK (retention_days > 0),
  timezone text NOT NULL DEFAULT 'Asia/Shanghai',
  daily_run_time time NOT NULL DEFAULT '03:00:00',
  last_started_at timestamptz,
  last_completed_at timestamptz,
  updated_by_admin_id uuid REFERENCES business.admin_accounts(id),
  updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO business.video_retention_policy (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

ALTER TABLE business.analysis_jobs
  ADD COLUMN IF NOT EXISTS input_media_asset_id uuid REFERENCES business.managed_media_resources(id),
  ADD COLUMN IF NOT EXISTS requested_by_admin_id uuid REFERENCES business.admin_accounts(id),
  ADD COLUMN IF NOT EXISTS trigger_type text NOT NULL DEFAULT 'automatic'
    CHECK (trigger_type IN ('automatic', 'manual'));

ALTER TABLE business.audit_events
  ADD COLUMN IF NOT EXISTS actor_admin_account_id uuid REFERENCES business.admin_accounts(id);
ALTER TABLE business.audit_events
  DROP CONSTRAINT IF EXISTS audit_events_actor_type_check;
ALTER TABLE business.audit_events
  ADD CONSTRAINT audit_events_actor_type_check
  CHECK (actor_type IN ('user', 'system', 'edge', 'gpu', 'admin'));

COMMIT;
