-- Foundational tenant table required before venue registration.

create schema if not exists business;

create table if not exists business.tenants (
  id uuid primary key,
  name text not null,
  status text not null default 'active' check (status in ('active', 'suspended')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (name)
);
