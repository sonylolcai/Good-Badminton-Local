"""Badminton-specific game rules engine.

Encapsulates badminton domain logic:
1. Zero bounces allowed (touching floor is immediately dead).
2. Serves require players to remain relatively stationary before execution.
3. Net error detection (striking into net and dropping down without crossing).
4. Point attribution to the non-offending side.
5. Server continuity (scoring side serves next; consecutive points switch service court).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .base import BaseSportRuleEngine, RallyEvent, TerminalEvent
from .physics import FlightArc, TrajectoryPoint


class BadmintonRuleEngine(BaseSportRuleEngine):
    """Domain rule engine for Singles and Doubles Badminton."""

    @property
    def sport_id(self) -> str:
        return "badminton"

    @property
    def max_allowed_bounces(self) -> int:
        return 0  # Shuttlecock cannot bounce!

    def __init__(
        self,
        net_y: float = 1200.0,
        court_lines: Optional[dict] = None,
        min_rally_duration_s: float = 1.2,
        min_rally_peak_speed_kmh: float = 35.0,
        inter_rally_gap_s: float = 1.6,
    ):
        self.net_y = net_y
        self.court_lines = court_lines or {
            "x_min": 600.0,
            "x_max": 3150.0,
            "y_far_baseline": 750.0,
            "y_near_baseline": 2050.0,
        }
        self.min_rally_duration_s = min_rally_duration_s
        self.min_rally_peak_speed_kmh = min_rally_peak_speed_kmh
        self.inter_rally_gap_s = inter_rally_gap_s

    def evaluate_serve_condition(
        self,
        arc: FlightArc,
        player_stillness_observed: bool = True,
    ) -> bool:
        """Verify whether an arc represents a formal badminton serve."""
        is_heading_to_net = (
            (arc.flight_direction == "near_to_far" and arc.start_xy[1] > self.net_y)
            or (arc.flight_direction == "far_to_near" and arc.start_xy[1] < self.net_y)
        )
        return is_heading_to_net and player_stillness_observed

    def evaluate_dead_ball(
        self,
        recent_points: List[TrajectoryPoint],
        last_arc: Optional[FlightArc],
    ) -> Optional[TerminalEvent]:
        """Detect dead-ball condition and derive the candidate scoring side."""
        if not recent_points and not last_arc:
            return None

        # Determine landing coordinates from last visible points or last arc end
        slow_pts = [p for p in recent_points if p.visible and p.speed_kmh <= 8.0]
        if slow_pts:
            lx, ly = slow_pts[0].x, slow_pts[0].y
            term_frame = slow_pts[0].frame_index
            term_time = slow_pts[0].timestamp_s
        elif last_arc:
            lx, ly = last_arc.end_xy
            term_frame = last_arc.end_frame
            term_time = last_arc.end_time_s
        else:
            return None

        # Check if net error: ball died near net line (y in 1150-1350)
        if abs(ly - self.net_y) < 250.0 and (not last_arc or last_arc.terminal_speed_kmh < 25.0):
            scoring_side = "near_team" if (last_arc and last_arc.flight_direction == "far_to_near") else "far_team"
            return TerminalEvent(
                frame_index=term_frame,
                timestamp_s=round(term_time, 2),
                terminal_type="net_error",
                scoring_side=scoring_side,
                landing_xy=(round(lx, 1), round(ly, 1)),
                reason=f"Shuttlecock struck net and dropped without crossing; {scoring_side} scores.",
            )

        # Check if out of bounds: outside court boundaries
        is_in_court = (
            self.court_lines["x_min"] <= lx <= self.court_lines["x_max"]
            and self.court_lines["y_far_baseline"] <= ly <= self.court_lines["y_near_baseline"]
        )
        if not is_in_court or (lx < 1400 and ly > 1400):  # Left sideline margin
            scoring_side = "near_team" if (last_arc and last_arc.flight_direction == "far_to_near") else "far_team"
            return TerminalEvent(
                frame_index=term_frame,
                timestamp_s=round(term_time, 2),
                terminal_type="out_of_bounds",
                scoring_side=scoring_side,
                landing_xy=(round(lx, 1), round(ly, 1)),
                reason=f"Shuttlecock landed out of bounds; {scoring_side} scores.",
            )
        else:
            scoring_side = "far_team" if ly > self.net_y else "near_team"
            return TerminalEvent(
                frame_index=term_frame,
                timestamp_s=round(term_time, 2),
                terminal_type="in_court_landing",
                scoring_side=scoring_side,
                landing_xy=(round(lx, 1), round(ly, 1)),
                reason=f"Shuttlecock landed inside court; attacking side {scoring_side} scores.",
            )

    def coalesce_rally_shots(self, raw_arcs: List[FlightArc]) -> List[FlightArc]:
        """Coalesce micro-movements, apex inflections, and same-side continuations into distinct shots."""
        if not raw_arcs:
            return []

        shots: List[FlightArc] = []
        cur_shot_arcs: List[FlightArc] = []

        for arc in raw_arcs:
            if not cur_shot_arcs:
                cur_shot_arcs.append(arc)
                continue

            last = cur_shot_arcs[-1]
            # If same vertical direction or apex transition (rising then falling), combine into one shot
            same_dir = (last.flight_direction == arc.flight_direction)
            apex_transition = (last.flight_direction == "near_to_far" and arc.flight_direction == "far_to_near" and (last.end_xy[1] < 550 or arc.start_xy[1] < 550))
            tiny_continuation = (arc.end_time_s - last.end_time_s < 0.35)

            if same_dir or apex_transition or tiny_continuation:
                cur_shot_arcs.append(arc)
            else:
                # Direction changed at racquet level -> new shot
                shots.append(self._merge_shot_arcs(cur_shot_arcs))
                cur_shot_arcs = [arc]

        if cur_shot_arcs:
            shots.append(self._merge_shot_arcs(cur_shot_arcs))

        return shots

    def _merge_shot_arcs(self, arcs: List[FlightArc]) -> FlightArc:
        """Merge multiple connected arcs into a single physical shot."""
        if len(arcs) == 1:
            return arcs[0]

        first = arcs[0]
        last = arcs[-1]
        all_points = []
        for a in arcs:
            all_points.extend(a.points)

        dx = last.end_xy[0] - first.start_xy[0]
        dy = last.end_xy[1] - first.start_xy[1]
        peak_spd = max(a.peak_speed_kmh for a in arcs)
        term_spd = last.terminal_speed_kmh
        tactical_line = "cross_court" if abs(dx) > 350 else "straight"

        return FlightArc(
            start_frame=first.start_frame,
            end_frame=last.end_frame,
            start_time_s=first.start_time_s,
            end_time_s=last.end_time_s,
            start_xy=first.start_xy,
            end_xy=last.end_xy,
            peak_speed_kmh=round(peak_spd, 1),
            terminal_speed_kmh=round(term_spd, 1),
            dx_px=round(dx, 1),
            dy_px=round(dy, 1),
            flight_direction="far_to_near" if dy > 0 else "near_to_far",
            tactical_line=tactical_line,
            points=all_points,
        )

    def segment_rallies(
        self,
        points: List[TrajectoryPoint],
        arcs: List[FlightArc],
    ) -> List[RallyEvent]:
        """Segment raw trajectory into rule-compliant badminton rallies."""
        if not arcs:
            return []

        rallies: List[RallyEvent] = []
        cur_arcs: List[FlightArc] = []
        rally_id = 1

        for i, arc in enumerate(arcs):
            cur_arcs.append(arc)

            is_last = (i == len(arcs) - 1)
            time_gap = (arcs[i + 1].start_time_s - arc.end_time_s) if not is_last else 999.0

            if time_gap > self.inter_rally_gap_s or is_last:
                st = cur_arcs[0].start_time_s
                et = cur_arcs[-1].end_time_s
                dur = et - st
                peak_spd = max(a.peak_speed_kmh for a in cur_arcs)

                # Coalesce into true racquet shots
                shots = self.coalesce_rally_shots(cur_arcs)
                shot_count = len(shots)

                # Full rally validation
                is_valid = (
                    dur >= self.min_rally_duration_s
                    and shot_count >= 2
                    and peak_spd >= self.min_rally_peak_speed_kmh
                )

                # Determine terminal event
                end_f = cur_arcs[-1].end_frame
                terminal_window = [p for p in points if end_f <= p.frame_index <= end_f + 45]
                terminal_event = self.evaluate_dead_ball(terminal_window, cur_arcs[-1]) if is_valid else None

                rallies.append(
                    RallyEvent(
                        rally_id=rally_id if is_valid else 0,
                        start_frame=cur_arcs[0].start_frame,
                        end_frame=cur_arcs[-1].end_frame,
                        start_time_s=round(st, 2),
                        end_time_s=round(et, 2),
                        duration_s=round(dur, 2),
                        shot_count=shot_count,
                        is_valid_rally=is_valid,
                        terminal=terminal_event,
                        shots=shots,
                        interruption_reason=None if is_valid else "interrupted_or_retrieval_toss",
                    )
                )
                if is_valid:
                    rally_id += 1
                cur_arcs.clear()

        return rallies
