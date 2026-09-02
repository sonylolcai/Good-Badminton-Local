-- Live court control for the direct-ingest pilot.
-- Raw video remains transient on the business gateway. This migration stores
-- only operational state, case identifiers and safe GPU execution events.

alter table business.edge_ingest_sessions
  add column if not exists gpu_forwarding_enabled boolean not null default false,
  add column if not exists preview_available boolean not null default false,
  add column if not exists last_preview_at timestamptz,
  add column if not exists gpu_event_cursor text;

create table if not exists business.edge_ingest_event_logs (
  id uuid primary key,
  edge_ingest_session_id uuid not null references business.edge_ingest_sessions(id) on delete cascade,
  source text not null check (source in ('gateway', 'gpu')),
  event_id text,
  level text not null default 'info' check (level in ('debug', 'info', 'warning', 'error')),
  event_type text not null,
  message text,
  payload jsonb not null default '{}'::jsonb,
  occurred_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  unique (edge_ingest_session_id, source, event_id)
);

create index if not exists edge_ingest_event_logs_session_time_idx
  on business.edge_ingest_event_logs (edge_ingest_session_id, occurred_at desc, id desc);
create index if not exists edge_ingest_sessions_court_live_idx
  on business.edge_ingest_sessions (court_id, updated_at desc);
