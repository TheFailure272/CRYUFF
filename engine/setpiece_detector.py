"""
C.R.U.Y.F.F. — Set-Piece Detector (Feature 4)

Detects dead-ball situations from live optical tracking data.

Detection Logic
~~~~~~~~~~~~~~~
A set-piece is detected when:
  1. Ball is stationary (speed < 0.3 m/s) for > 3 seconds
  2. 6+ players are clustered inside the penalty box area
  3. Ball position classifies the type:
     - Corner (left/right)
     - Free-kick (central/wide)

Output
~~~~~~
A ``SetPieceEvent`` containing:
  - Type classification
  - Defensive formation snapshot (player positions in box)
  - Ball position (delivery origin)
  - Attacking/defending player identification
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SetPieceType(str, Enum):
    CORNER_LEFT = "corner_left"
    CORNER_RIGHT = "corner_right"
    FREEKICK_CENTRAL = "freekick_central"
    FREEKICK_WIDE = "freekick_wide"


@dataclass(slots=True)
class BoxPlayer:
    """A player inside the penalty box during a set-piece."""
    player_id: int
    x: float
    y: float
    is_attacking: bool  # True = attacking team, False = defending


@dataclass(slots=True)
class SetPieceEvent:
    """A detected dead-ball event."""
    type: SetPieceType
    ball_x: float
    ball_y: float
    ball_z: float
    box_players: list[BoxPlayer]
    defender_positions: list[tuple[float, float]]
    gk_position: Optional[tuple[float, float]]
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "ball": {"x": self.ball_x, "y": self.ball_y, "z": self.ball_z},
            "box_players": [
                {"id": p.player_id, "x": p.x, "y": p.y, "atk": p.is_attacking}
                for p in self.box_players
            ],
            "gk": {"x": self.gk_position[0], "y": self.gk_position[1]}
                  if self.gk_position else None,
            "timestamp": self.timestamp,
        }


# ─── Pitch Geometry (normalised [0,1]) ─────────────────────────

# Penalty box boundaries (both ends)
BOX_Y_NEAR = 0.0
BOX_Y_NEAR_EDGE = 0.165        # ~16.5m / 100m ≈ 0.165
BOX_Y_FAR = 1.0
BOX_Y_FAR_EDGE = 1.0 - 0.165   # ~0.835
BOX_X_MIN = 0.5 - 0.20         # ~20m from center
BOX_X_MAX = 0.5 + 0.20

# Corner arc zones
CORNER_THRESHOLD = 0.03         # ball within 3% of corner


@dataclass
class SetPieceDetector:
    """
    Detects set-piece situations from live tracking data.

    Usage::

        detector = SetPieceDetector()

        # At 25Hz:
        event = detector.update(
            ball_xyz=(0.02, 0.01, 0.0),
            player_positions=[(pid, x, y), ...],
        )
        if event:
            # Dead ball detected — trigger solver
            pass
    """

    stationary_threshold: float = 0.003   # normalised speed threshold
    stationary_time: float = 3.0          # seconds ball must be still
    min_box_players: int = 6              # minimum players in box area

    _ball_stationary_since: Optional[float] = field(
        default=None, init=False, repr=False
    )
    _last_ball_pos: Optional[tuple[float, float, float]] = field(
        default=None, init=False, repr=False
    )
    _active_event: Optional[SetPieceEvent] = field(
        default=None, init=False, repr=False
    )
    _cooldown_until: float = field(default=0.0, init=False, repr=False)

    def update(
        self,
        ball_xyz: tuple[float, float, float],
        player_positions: list[tuple[int, float, float]],
        attacking_ids: set[int] | None = None,
    ) -> Optional[SetPieceEvent]:
        """
        Process a frame and return a SetPieceEvent if detected.

        Parameters
        ----------
        ball_xyz : tuple
            (x, y, z) normalised ball position
        player_positions : list
            [(player_id, x, y), ...] for all 22 players
        attacking_ids : set, optional
            Player IDs on the attacking team
        """
        now = time.monotonic()

        # Cooldown after a previous detection (prevent re-firing)
        if now < self._cooldown_until:
            return None

        bx, by, bz = ball_xyz

        # Check if ball is stationary
        if self._last_ball_pos is not None:
            dx = bx - self._last_ball_pos[0]
            dy = by - self._last_ball_pos[1]
            speed = (dx * dx + dy * dy) ** 0.5

            if speed < self.stationary_threshold:
                if self._ball_stationary_since is None:
                    self._ball_stationary_since = now
            else:
                self._ball_stationary_since = None
        else:
            self._ball_stationary_since = now

        self._last_ball_pos = (bx, by, bz)

        # Not stationary long enough
        if (self._ball_stationary_since is None or
                (now - self._ball_stationary_since) < self.stationary_time):
            return None

        # Count players in box area
        box_players = self._detect_box_players(
            player_positions, bx, by, attacking_ids or set()
        )

        if len(box_players) < self.min_box_players:
            return None

        # Classify set-piece type
        sp_type = self._classify(bx, by)

        # Find goalkeeper (closest defender to goal center)
        gk_pos = self._find_goalkeeper(box_players, by)

        # Extract defender positions
        defenders = [
            (p.x, p.y) for p in box_players if not p.is_attacking
        ]

        event = SetPieceEvent(
            type=sp_type,
            ball_x=bx,
            ball_y=by,
            ball_z=bz,
            box_players=box_players,
            defender_positions=defenders,
            gk_position=gk_pos,
            timestamp=now,
        )

        # Cooldown to prevent re-firing for 30 seconds
        self._cooldown_until = now + 30.0
        self._active_event = event

        logger.info(
            "Set-piece detected: %s at (%.3f, %.3f), %d players in box",
            sp_type.value, bx, by, len(box_players),
        )

        return event

    def _detect_box_players(
        self,
        positions: list[tuple[int, float, float]],
        ball_x: float,
        ball_y: float,
        attacking_ids: set[int],
    ) -> list[BoxPlayer]:
        """Find players inside the penalty box near the ball."""
        players = []

        # Determine which box the ball is nearest
        near_y0 = ball_y < 0.5

        if near_y0:
            y_min, y_max = BOX_Y_NEAR, BOX_Y_NEAR_EDGE
        else:
            y_min, y_max = BOX_Y_FAR_EDGE, BOX_Y_FAR

        for pid, px, py in positions:
            if BOX_X_MIN <= px <= BOX_X_MAX and y_min <= py <= y_max:
                players.append(BoxPlayer(
                    player_id=pid,
                    x=px,
                    y=py,
                    is_attacking=pid in attacking_ids,
                ))

        return players

    def _classify(self, bx: float, by: float) -> SetPieceType:
        """Classify the set-piece type based on ball position."""
        # Corner kicks
        if by < CORNER_THRESHOLD:
            return (SetPieceType.CORNER_LEFT if bx < 0.5
                    else SetPieceType.CORNER_RIGHT)
        if by > (1 - CORNER_THRESHOLD):
            return (SetPieceType.CORNER_LEFT if bx < 0.5
                    else SetPieceType.CORNER_RIGHT)

        # Free kicks
        if 0.3 <= bx <= 0.7:
            return SetPieceType.FREEKICK_CENTRAL
        return SetPieceType.FREEKICK_WIDE

    def _find_goalkeeper(
        self, box_players: list[BoxPlayer], ball_y: float
    ) -> Optional[tuple[float, float]]:
        """Find the goalkeeper (closest defender to goal center)."""
        goal_y = 0.0 if ball_y < 0.5 else 1.0
        goal_x = 0.5

        closest = None
        min_dist = float("inf")

        for p in box_players:
            if p.is_attacking:
                continue
            d = ((p.x - goal_x) ** 2 + (p.y - goal_y) ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                closest = (p.x, p.y)

        return closest

    @property
    def is_active(self) -> bool:
        return self._active_event is not None
