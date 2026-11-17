-- Business-owned desired capture state.  The venue Mac only polls this state;
-- it never needs a human at the venue to start or stop a camera session.

create table if not exists business.edge_capture_controls (
  court_id uuid primary key references business.courts(id) on delete cascade,
  desired_mode text not null default 'idle'
    check (desired_mode in ('idle', 'preview', 'record')),
  revision integer not null default 0 check (revision >= 0),
  updated_at timestamptz not null default now()
);

create index if not exists edge_capture_controls_updated_idx
  on business.edge_capture_controls (updated_at desc);
