-- Calibration is business-owned. Preserve how its four image corners were
-- obtained so a later operator can distinguish manually visible corners from
-- corners extrapolated from visible court lines.

alter table business.camera_calibrations
  add column if not exists evidence jsonb not null default '{}'::jsonb;
