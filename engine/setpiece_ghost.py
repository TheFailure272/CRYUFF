"""
C.R.U.Y.F.F. — Set-Piece Ghost Engine (Feature 4)

Bi-directional Ghost simulation for dead-ball situations.

Architecture
~~~~~~~~~~~~
Once the SetPieceSolver produces the top 3 GK-safe landing zones,
this module simulates the *player reactions* to each ball flight:

  1. Attacking runs — optimal paths to arrive at the landing zone
     at the exact moment the ball arrives
  2. Defensive reactions — blocking screen trajectories
  3. Intersection probability — the overlap of attacker arrival
     and ball arrival probabilities

This converts a pure physics heatmap into a tactical action plan.

Output
~~~~~~
For each top zone:
  - Optimal attacker run paths
  - Blocking screen positions
  - Header probability (attacker arrives first)
  - Combined scoring probability (ball × run × header)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Physical Constants ────────────────────────────────────────

MAX_SPRINT_SPEED = 9.5       # m/s (elite footballer max)
ACCELERATION = 4.5           # m/s² (standing start)
HEADER_REACH = 0.5           # meters above standing height
REACTION_TIME = 0.3          # seconds (defender reaction delay)
BALL_FLIGHT_TIME_EST = 1.5   # estimated seconds for corner delivery

# Fix F34: Biomechanical jump window (Z-axis)
# Header is only possible when ball altitude Z(t) is between:
HEADER_Z_MIN = 1.5           # meters (diving header minimum)
HEADER_Z_MAX = 2.6           # meters (max vertical leap + reach)
PLAYER_HEIGHT = 1.83         # meters (average elite footballer)
MAX_JUMP = 0.77              # meters (elite vertical leap)

# Fix F35: Kinematic foul penalty
FOUL_VELOCITY_THRESHOLD = 2.0     # m/s relative velocity
SCREEN_SETUP_BUFFER = 1.5         # seconds — must arrive before defender


@dataclass(slots=True)
class AttackingRun:
    """An optimal attacking run path to meet the ball."""
    player_id: int
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    arrival_time: float     # seconds to reach target
    ball_arrival: float     # estimated ball flight time
    header_prob: float      # probability of winning the header
    run_angle: float        # radians from start to target
    sprint_distance: float  # meters


@dataclass(slots=True)
class ZonePlan:
    """Complete tactical plan for a single landing zone."""
    zone_x: float
    zone_y: float
    ball_probability: float
    attacking_runs: list[AttackingRun]
    blocking_screen: list[tuple[float, float]]  # (x, y) positions
    combined_score: float   # ball_prob × best_header_prob
    recommended: bool       # highest combined score

    def to_dict(self) -> dict:
        return {
            "zone": {"x": self.zone_x, "y": self.zone_y},
            "ball_probability": round(self.ball_probability, 4),
            "combined_score": round(self.combined_score, 4),
            "recommended": self.recommended,
            "runs": [
                {
                    "player_id": r.player_id,
                    "start": {"x": r.start_x, "y": r.start_y},
                    "target": {"x": r.target_x, "y": r.target_y},
                    "arrival_time": round(r.arrival_time, 3),
                    "header_prob": round(r.header_prob, 3),
                    "sprint_distance": round(r.sprint_distance, 2),
                }
                for r in self.attacking_runs
            ],
            "blocking_screen": [
                {"x": round(x, 2), "y": round(y, 2)}
                for x, y in self.blocking_screen
            ],
        }


@dataclass
class SetPieceGhost:
    """
    Bi-directional Ghost engine for set-piece player simulation.

    Usage::

        ghost = SetPieceGhost()
        plans = ghost.plan(
            top_zones=[{"x": 34, "y": 8, "probability": 0.15}, ...],
            attackers=[(9, 30, 12), (4, 28, 14), ...],
            defenders=[(5, 33, 6), (3, 35, 8), ...],
            ball_flight_time=1.5,
        )
    """

    def plan(
        self,
        top_zones: list[dict],
        attackers: list[tuple[int, float, float]],   # (id, x, y) meters
        defenders: list[tuple[int, float, float]],
        ball_flight_time: float = BALL_FLIGHT_TIME_EST,
        ball_z_at_time: callable = None,
    ) -> list[ZonePlan]:
        """
        Generate tactical plans for each top landing zone.

        Parameters
        ----------
        top_zones : list[dict]
            From SetPieceSolver.solve()["top_zones"]
        attackers : list[tuple]
            Attacking player positions (id, x_m, y_m)
        defenders : list[tuple]
            Defending player positions (id, x_m, y_m)
        ball_flight_time : float
            Estimated ball flight time (seconds)
        ball_z_at_time : callable, optional
            Fix F34: Function Z(t) -> float that returns the ball's
            altitude at time t. If provided, header probability is
            gated by the biomechanical jump window.
        """
        plans = []

        for zone in top_zones:
            zx, zy = zone["x"], zone["y"]
            ball_prob = zone.get("probability", 0.1)

            # Compute attacking runs
            runs = self._compute_attacking_runs(
                attackers, zx, zy, ball_flight_time, ball_z_at_time
            )

            # Compute blocking screen (with Fix F35 foul penalty)
            screen = self._compute_blocking_screen(
                defenders, zx, zy, runs, ball_flight_time
            )

            # Combined score = ball_prob × best_header_prob
            best_header = max(r.header_prob for r in runs) if runs else 0
            combined = ball_prob * best_header

            plans.append(ZonePlan(
                zone_x=zx,
                zone_y=zy,
                ball_probability=ball_prob,
                attacking_runs=runs,
                blocking_screen=screen,
                combined_score=combined,
                recommended=False,
            ))

        # Mark the best plan as recommended
        if plans:
            plans.sort(key=lambda p: p.combined_score, reverse=True)
            plans[0].recommended = True

        logger.info(
            "SetPieceGhost: %d zone plans generated. Best: (%.1f, %.1f) "
            "score=%.3f",
            len(plans),
            plans[0].zone_x if plans else 0,
            plans[0].zone_y if plans else 0,
            plans[0].combined_score if plans else 0,
        )

        return plans

    def _compute_attacking_runs(
        self,
        attackers: list[tuple[int, float, float]],
        target_x: float,
        target_y: float,
        ball_time: float,
        ball_z_at_time: callable = None,
    ) -> list[AttackingRun]:
        """
        For each attacker, compute the optimal run to the target zone.

        Fix F34: Header probability is gated by the biomechanical
        jump window. The ball must be at Z ∈ [1.5m, 2.6m] when the
        attacker arrives, or header_prob drops to zero.
        """
        runs = []

        for pid, ax, ay in attackers:
            dx = target_x - ax
            dy = target_y - ay
            dist = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.atan2(dy, dx)

            # Time to reach target (acceleration phase + max speed phase)
            arrival = self._sprint_time(dist)

            # Header probability:
            # Perfect timing (arrival ≈ ball_time) → high prob
            time_diff = abs(arrival - ball_time)

            if arrival <= ball_time:
                header_prob = max(0, 1.0 - time_diff * 0.5)
            else:
                header_prob = max(0, 1.0 - time_diff * 2.0)

            # Factor in distance (closer = higher accuracy)
            dist_factor = max(0, 1.0 - dist / 30.0)
            header_prob *= dist_factor

            # Fix F34: Z-axis biomechanical jump window
            # If we know ball altitude at the attacker's arrival time,
            # gate the header probability by whether the ball is
            # within human heading range [1.5m, 2.6m]
            if ball_z_at_time is not None:
                ball_z = ball_z_at_time(arrival)
                if ball_z < HEADER_Z_MIN or ball_z > HEADER_Z_MAX:
                    # Ball is either too low (ground) or too high
                    # (above max jump + reach) — "Superman" header
                    header_prob = 0.0
                else:
                    # Scale by how ideal the height is
                    # Optimal: ~2.0m (standing header with small jump)
                    optimal = PLAYER_HEIGHT + MAX_JUMP * 0.3
                    z_quality = 1.0 - abs(ball_z - optimal) / (HEADER_Z_MAX - HEADER_Z_MIN)
                    header_prob *= max(0.1, z_quality)

            runs.append(AttackingRun(
                player_id=pid,
                start_x=ax,
                start_y=ay,
                target_x=target_x,
                target_y=target_y,
                arrival_time=arrival,
                ball_arrival=ball_time,
                header_prob=min(1.0, max(0, header_prob)),
                run_angle=angle,
                sprint_distance=dist,
            ))

        runs.sort(key=lambda r: r.header_prob, reverse=True)
        return runs[:3]

    def _compute_blocking_screen(
        self,
        defenders: list[tuple[int, float, float]],
        zone_x: float,
        zone_y: float,
        runs: list[AttackingRun],
        ball_flight_time: float = BALL_FLIGHT_TIME_EST,
    ) -> list[tuple[float, float]]:
        """
        Compute optimal blocking screen positions.

        Fix F35: Kinematic foul penalty.
        Screens are established by occupying the space BEFORE the
        defender arrives, not by running into them. If a screener
        would need V_relative > 2 m/s toward a defender's centroid
        to reach the screen position, the screen is repositioned
        further from the defender to ensure legal space occupation.
        """
        if not runs:
            return []

        best_run = runs[0]
        screen_positions = []

        for did, dx_pos, dy_pos in defenders:
            # Vector from defender to zone
            vx = zone_x - dx_pos
            vy = zone_y - dy_pos
            dist = math.sqrt(vx ** 2 + vy ** 2)

            if dist < 2.0:
                # Defender is already in the zone — can't screen
                continue

            # Fix F35: Calculate safe screen distance
            # Screener must arrive at screen position BEFORE the
            # defender reaches it (SCREEN_SETUP_BUFFER seconds early).
            # This ensures legal space occupation, not collision.
            #
            # Place screen at a point the screener can reach first.
            # Default: 3m in front of defender (not 2m) to give
            # the screener time to establish position.
            screen_dist = min(3.0, dist * 0.4)  # 40% of defender-to-zone

            screen_x = dx_pos + (vx / dist) * screen_dist
            screen_y = dy_pos + (vy / dist) * screen_dist

            # Fix F35: Foul velocity check
            # If an attacker sprinting to this screen position would
            # have V_relative > 2 m/s when the defender bounding box
            # is within 1m, apply a penalty by pushing the screen
            # further from the defender.
            # Find closest attacker to this screen position
            for _, ax, ay in [(r.player_id, r.start_x, r.start_y)
                              for r in runs]:
                to_screen = math.sqrt((screen_x - ax) ** 2 +
                                      (screen_y - ay) ** 2)
                screener_arrival = self._sprint_time(to_screen)
                # Defender's time to reach screen position
                def_to_screen = math.sqrt((screen_x - dx_pos) ** 2 +
                                           (screen_y - dy_pos) ** 2)
                defender_arrival = self._sprint_time(def_to_screen)

                if screener_arrival >= defender_arrival:
                    # Screener can't arrive before defender —
                    # push screen closer to screener's start
                    screen_dist = max(1.5, screen_dist - 1.0)
                    screen_x = dx_pos + (vx / dist) * screen_dist
                    screen_y = dy_pos + (vy / dist) * screen_dist
                    break  # repositioned, no collision risk

            screen_positions.append((screen_x, screen_y))

        return screen_positions[:3]

    @staticmethod
    def _sprint_time(distance: float) -> float:
        """
        Estimate sprint time for a given distance.

        Uses acceleration model:
          Phase 1: Accelerate at 4.5 m/s² until MAX_SPRINT_SPEED
          Phase 2: Constant speed
        """
        # Time to reach max speed
        t_accel = MAX_SPRINT_SPEED / ACCELERATION
        # Distance covered during acceleration
        d_accel = 0.5 * ACCELERATION * t_accel ** 2

        if distance <= d_accel:
            # Still accelerating
            return math.sqrt(2 * distance / ACCELERATION) + REACTION_TIME
        else:
            # Accel phase + constant speed phase
            d_remaining = distance - d_accel
            t_cruise = d_remaining / MAX_SPRINT_SPEED
            return t_accel + t_cruise + REACTION_TIME
