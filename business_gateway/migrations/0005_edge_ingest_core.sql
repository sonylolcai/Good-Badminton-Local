-- Signed terminal bindings and direct business-to-GPU ingest state.
-- Raw video bytes are never stored in PostgreSQL.

alter table business.audit_events add column if not exists tenant_id uuid references business.tenants(id);
alter table business.audit_events add column if not exists venue_id uuid references business.venues(id);

create table if not exists business.edge_devices (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  court_id uuid not null references business.courts(id),
  device_code text not null,
  credential_version text not null default 'v1',
  status text not null default 'offline' check (status in ('offline', 'online', 'disabled')),
  last_heartbeat_at timestamptz,
  disk_free_bytes bigint,
  agent_version text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (venue_id, device_code)
);

create table if not exists business.cameras (
  id uuid primary key,
  court_id uuid not null references business.courts(id),
  edge_device_id uuid not null references business.edge_devices(id),
  camera_code text not null,
  stream_ref_ciphertext bytea not null,
  status text not null default 'offline' check (status in ('offline', 'active', 'disabled')),
  last_heartbeat_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (edge_device_id, camera_code)
);

create table if not exists business.camera_calibrations (
  id uuid primary key,
  camera_id uuid not null references business.cameras(id),
  version integer not null check (version > 0),
  court_corners jsonb not null,
  quality_status text not null default 'pending' check (quality_status in ('pending', 'validated', 'rejected')),
  created_at timestamptz not null default now(),
  unique (camera_id, version)
);

create table if not exists business.edge_request_nonces (
  edge_device_id uuid not null references business.edge_devices(id) on delete cascade,
  nonce text not null,
  expires_at timestamptz not null,
  created_at timestamptz not null default now(),
  primary key (edge_device_id, nonce)
);

create table if not exists business.edge_ingest_sessions (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  court_id uuid not null references business.courts(id),
  camera_id uuid not null references business.cameras(id),
  edge_device_id uuid not null references business.edge_devices(id),
  calibration_id uuid not null references business.camera_calibrations(id),
  status text not null default 'requested' check (status in ('requested', 'receiving', 'relaying', 'processing', 'succeeded', 'partial', 'failed', 'cancelled')),
  gpu_analysis_session_id text,
  gpu_status text,
  configuration jsonb not null default '{}'::jsonb,
  received_segment_count integer not null default 0,
  forwarded_segment_count integer not null default 0,
  last_segment_index integer,
  last_received_at timestamptz,
  last_forwarded_at timestamptz,
  error_code text,
  error_message text,
  finished_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists one_active_edge_session_per_camera
  on business.edge_ingest_sessions(camera_id)
  where status in ('requested', 'receiving', 'relaying', 'processing');

create table if not exists business.edge_ingest_segments (
  id uuid primary key,
  edge_ingest_session_id uuid not null references business.edge_ingest_sessions(id) on delete cascade,
  segment_index integer not null check (segment_index >= 0),
  source_start_time_sec numeric not null,
  duration_sec numeric not null check (duration_sec > 0 and duration_sec <= 10),
  sha256 text not null,
  content_type text not null check (content_type in ('video/mp4', 'video/iso.segment')),
  content_length_bytes bigint not null check (content_length_bytes > 0),
  idempotency_key text not null,
  status text not null default 'received' check (status in ('received', 'forwarded', 'failed')),
  gpu_receipt jsonb not null default '{}'::jsonb,
  error_code text,
  error_message text,
  forwarded_at timestamptz,
  created_at timestamptz not null default now(),
  unique (edge_ingest_session_id, segment_index)
);
