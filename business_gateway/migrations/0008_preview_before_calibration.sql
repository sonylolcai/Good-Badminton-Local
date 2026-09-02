-- A fixed-camera court must be visible before an operator can mark its four
-- image corners. Permit a signed, preview-only ingest session without a
-- calibration record. GPU forwarding remains blocked by edge_api.py until a
-- later session is created with a validated calibration.

alter table business.edge_ingest_sessions
  alter column calibration_id drop not null;
