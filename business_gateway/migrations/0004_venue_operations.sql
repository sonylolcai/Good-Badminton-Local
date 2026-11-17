-- Business-side venue operations.  Apply after the core identity/tenant
-- migration.  This schema deliberately contains no raw video, pose or image
-- bytes: the GPU service stays anonymous and is referenced by session/job id.

create schema if not exists business;

create table if not exists business.venues (
  id uuid primary key,
  tenant_id uuid not null,
  code text not null,
  name text not null,
  timezone text not null default 'Asia/Shanghai',
  address text,
  status text not null default 'active' check (status in ('active', 'inactive')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (tenant_id, code)
);

create table if not exists business.courts (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  code text not null,
  name text not null,
  sort_order integer not null default 0 check (sort_order >= 0),
  status text not null default 'active' check (status in ('active', 'maintenance', 'inactive')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (venue_id, code)
);

create table if not exists business.users (
  id uuid primary key,
  nickname text,
  status text not null default 'active' check (status in ('active', 'disabled')),
  profile_visibility text not null default 'private' check (profile_visibility in ('private', 'venue')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists business.venue_memberships (
  venue_id uuid not null references business.venues(id),
  user_id uuid not null references business.users(id),
  role text not null check (role in ('owner', 'operator', 'viewer')),
  status text not null default 'active' check (status in ('active', 'revoked')),
  created_at timestamptz not null default now(),
  revoked_at timestamptz,
  primary key (venue_id, user_id)
);

create table if not exists business.audit_events (
  id uuid primary key,
  actor_type text not null,
  action text not null,
  resource_type text not null,
  resource_id text,
  after_summary jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- QR payloads contain only an opaque token.  Resolve the token server-side;
-- never encode database ids, a user id or a privileged URL in a printed code.
create table if not exists business.qr_tokens (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  court_id uuid references business.courts(id),
  scope text not null check (scope in ('venue', 'court')),
  label text not null,
  token_digest text not null unique,
  status text not null default 'active' check (status in ('active', 'revoked')),
  created_at timestamptz not null default now(),
  revoked_at timestamptz,
  expires_at timestamptz,
  check ((scope = 'venue' and court_id is null) or (scope = 'court' and court_id is not null))
);
create index if not exists qr_tokens_venue_status_idx on business.qr_tokens (venue_id, status, created_at desc);

create table if not exists business.venue_activities (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  title text not null,
  description text,
  starts_at timestamptz not null,
  registration_deadline_at timestamptz,
  capacity integer check (capacity is null or capacity > 0),
  status text not null default 'draft' check (status in ('draft', 'published', 'cancelled', 'completed')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create table if not exists business.activity_registrations (
  activity_id uuid not null references business.venue_activities(id),
  user_id uuid not null references business.users(id),
  status text not null default 'registered' check (status in ('registered', 'waitlisted', 'cancelled')),
  created_at timestamptz not null default now(),
  primary key (activity_id, user_id)
);

create table if not exists business.community_posts (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  author_user_id uuid not null references business.users(id),
  activity_id uuid references business.venue_activities(id),
  post_type text not null check (post_type in ('official', 'match_card', 'member_share')),
  body text not null,
  status text not null default 'published' check (status in ('published', 'hidden')),
  visibility text not null default 'venue' check (visibility in ('venue', 'private')),
  created_at timestamptz not null default now(),
  hidden_at timestamptz
);
create index if not exists community_posts_feed_idx on business.community_posts (venue_id, status, created_at desc);
alter table business.community_posts add column if not exists activity_id uuid references business.venue_activities(id);
alter table business.community_posts add column if not exists hidden_at timestamptz;

-- Match state is business-owned.  The recorder must be stopped after one
-- participant submits a score and the recording gateway acknowledges it;
-- opponent confirmation is asynchronous and only gates official outcomes.
create table if not exists business.matches (
  id uuid primary key,
  venue_id uuid not null references business.venues(id),
  court_id uuid not null references business.courts(id),
  match_format text not null check (match_format in ('singles', 'doubles')),
  lifecycle_status text not null check (lifecycle_status in ('waiting', 'playing', 'score_submitted', 'record_stop_requested', 'recording_stopped', 'analysis_ready', 'claiming', 'result_confirmed', 'disputed', 'cancelled')),
  score_summary jsonb,
  score_submitted_by uuid references business.users(id),
  score_submitted_at timestamptz,
  recording_stop_requested_at timestamptz,
  recording_stopped_at timestamptz,
  recording_gateway_receipt text,
  external_analysis_session_id text,
  started_at timestamptz,
  ended_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create unique index if not exists one_active_match_per_court on business.matches (court_id)
  where lifecycle_status in ('waiting', 'playing', 'score_submitted', 'record_stop_requested');

create table if not exists business.match_participants (
  match_id uuid not null references business.matches(id),
  user_id uuid not null references business.users(id),
  team_id text,
  slot_id text,
  result_confirmation text not null default 'pending' check (result_confirmation in ('pending', 'confirmed', 'disputed')),
  joined_at timestamptz not null default now(),
  confirmed_at timestamptz,
  primary key (match_id, user_id),
  unique (match_id, slot_id)
);

-- Claiming binds a verified anonymous track to a user after analysis.  It is
-- intentionally a reviewable business record, never a GPU-side user field.
create table if not exists business.track_claims (
  id uuid primary key,
  match_id uuid not null references business.matches(id),
  analysis_session_id text not null,
  track_id text not null,
  user_id uuid not null references business.users(id),
  status text not null default 'pending' check (status in ('pending', 'confirmed', 'rejected', 'corrected')),
  confidence numeric(4,3),
  evidence_ref text,
  claimed_at timestamptz not null default now(),
  reviewed_at timestamptz,
  reviewed_by uuid references business.users(id),
  unique (analysis_session_id, track_id),
  unique (match_id, user_id)
);

-- The pilot database already has its own core match, participant and claim
-- tables.  Add the operations fields in place so historical rows survive.
alter table business.matches add column if not exists score_summary jsonb;
alter table business.matches add column if not exists score_submitted_by uuid references business.users(id);
alter table business.matches add column if not exists score_submitted_at timestamptz;
alter table business.matches add column if not exists recording_stop_requested_at timestamptz;
alter table business.matches add column if not exists recording_stopped_at timestamptz;
alter table business.matches add column if not exists recording_gateway_receipt text;
alter table business.matches add column if not exists external_analysis_session_id text;
alter table business.match_participants add column if not exists team_id text;
alter table business.match_participants add column if not exists slot_id text;
alter table business.match_participants add column if not exists result_confirmation text not null default 'pending'
  check (result_confirmation in ('pending', 'confirmed', 'disputed'));
alter table business.match_participants add column if not exists confirmed_at timestamptz;
alter table business.track_claims add column if not exists analysis_session_id text;
alter table business.track_claims add column if not exists status text not null default 'pending'
  check (status in ('pending', 'confirmed', 'rejected', 'corrected'));
alter table business.track_claims add column if not exists confidence numeric(4,3);
alter table business.track_claims add column if not exists evidence_ref text;
alter table business.track_claims add column if not exists reviewed_at timestamptz;
alter table business.track_claims add column if not exists reviewed_by uuid references business.users(id);
create unique index if not exists track_claims_session_track_unique
  on business.track_claims (analysis_session_id, track_id) where analysis_session_id is not null;

-- Per-user delivery eligibility is materialized only after role claiming and
-- consent checks.  Asset storage remains external and is referenced by an
-- opaque internal asset reference rather than an open URL.
create table if not exists business.match_deliveries (
  id uuid primary key,
  match_id uuid not null references business.matches(id),
  user_id uuid not null references business.users(id),
  track_claim_id uuid references business.track_claims(id),
  delivery_type text not null check (delivery_type in ('photo', 'video', 'movement_report', 'tactical_report')),
  asset_ref text,
  status text not null default 'pending' check (status in ('pending', 'processing', 'ready', 'failed', 'withheld')),
  reason text,
  available_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (match_id, user_id, delivery_type)
);

create table if not exists business.compute_instances (
  id uuid primary key,
  provider text not null,
  external_instance_id text not null,
  display_name text not null,
  status text not null,
  hourly_cost numeric(12,4),
  currency text not null default 'CNY',
  last_heartbeat_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (provider, external_instance_id)
);
create table if not exists business.analysis_jobs (
  id uuid primary key,
  match_id uuid references business.matches(id),
  status text not null,
  job_type text not null,
  external_analysis_session_id text,
  compute_instance_id uuid references business.compute_instances(id),
  created_at timestamptz not null default now(),
  finished_at timestamptz
);
