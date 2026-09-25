BEGIN;

COMMENT ON TABLE business.users IS
  'Platform-wide player profiles. A player is not owned by or assigned to a venue.';
COMMENT ON TABLE business.venue_memberships IS
  'Legacy operational memberships. This table must not represent player ownership; player venue history comes from match participation.';

CREATE INDEX IF NOT EXISTS match_participants_player_idx
  ON business.match_participants (user_id, match_id);
CREATE INDEX IF NOT EXISTS matches_venue_court_started_idx
  ON business.matches (venue_id, court_id, started_at DESC);

CREATE OR REPLACE VIEW business.player_play_records AS
SELECT
  mp.user_id AS player_id,
  m.id AS match_id,
  m.venue_id,
  v.name AS venue_name,
  m.court_id,
  c.name AS court_name,
  m.match_format,
  m.lifecycle_status,
  mp.team_id,
  mp.slot_id,
  mp.result_confirmation,
  m.started_at,
  m.ended_at,
  m.created_at
FROM business.match_participants mp
JOIN business.matches m ON m.id = mp.match_id
JOIN business.venues v ON v.id = m.venue_id
JOIN business.courts c ON c.id = m.court_id;

COMMENT ON VIEW business.player_play_records IS
  'Derived player history. Each row binds a platform player to the venue and court of one match.';

COMMIT;
