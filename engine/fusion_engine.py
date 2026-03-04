"""
C.R.U.Y.F.F. — Fusion Engine (Feature 3)

Cross-references Internal Load (HR, metabolic power) with
External Topology (distance-to-nearest-attacker, void proximity)
to generate compound fatigue alerts.

Alert Types
~~~~~~~~~~~
* ``structural_collapse`` — HR > 85% max AND positional drift > 1.5m
  over 3 minutes.  The player is physically failing AND structurally
  exposing the team.

* ``masking`` — HR > 90% max BUT position stable.  The player is
  hiding their fatigue — they haven't collapsed yet, but they will
  within 5 minutes.  Proactive substitution alert.

* ``overload`` — Metabolic power > threshold for > 2 minutes.
  The player is accumulating unsustainable internal load.

Dependencies
~~~~~~~~~~~~
* ``sensor_ekf.py`` — provides fused state per player
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class AlertType(str, Enum):
    STRUCTURAL_COLLAPSE = "structural_collapse"
    MASKING = "masking"
    OVERLOAD = "overload"
    OPTICAL_ONLY = "optical_only"  # legacy camera-only alert


@dataclass(slots=True)
class FusionAlert:
    """A compound fatigue alert from fused sensor data."""
    player_id: int
    alert_type: AlertType
    level: AlertLevel
    message: str
    hr_bpm: float
    hr_max_pct: float
    metabolic_power: float
    positional_drift: float  # meters over tracking window
    source: str = "optical+wearable"
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "type": self.alert_type.value,
            "status": self.level.value,
            "message": self.message,
            "hr_bpm": round(self.hr_bpm, 1),
            "hr_max_pct": round(self.hr_max_pct, 3),
            "metabolic_power": round(self.metabolic_power, 2),
            "positional_drift": round(self.positional_drift, 2),
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class FusionEngine:
    """
    Generates compound fatigue alerts from fused EKF state.

    Usage::

        engine = FusionEngine()

        # Each frame (25Hz): feed fused state + topology
        alerts = engine.evaluate(
            player_id=8,
            fused_state=ekf.state,
            dist_to_attacker=12.3,
        )
    """

    # Thresholds
    hr_masking_pct: float = 0.90
    hr_collapse_pct: float = 0.85
    drift_threshold: float = 1.5   # meters over window
    metabolic_threshold: float = 15.0  # W/kg

    # Tracking windows
    _position_history: dict[int, list[tuple[float, float, float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _metabolic_history: dict[int, list[tuple[float, float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    WINDOW_SECONDS: float = 180.0  # 3 minutes

    def evaluate(self, player_id: int, fused_state: dict,
                 dist_to_attacker: float = 0.0) -> list[FusionAlert]:
        """
        Evaluate a player's fused state and return any active alerts.

        Parameters
        ----------
        player_id : int
        fused_state : dict
            Output of SensorEKF.state
        dist_to_attacker : float
            Distance to closest opponent (from topology engine)

        Returns
        -------
        list[FusionAlert]
        """
        now = time.monotonic()
        alerts: list[FusionAlert] = []

        x = fused_state["x"]
        y = fused_state["y"]
        hr = fused_state["hr_bpm"]
        hr_pct = fused_state["hr_max_pct"]
        mp = fused_state["metabolic_power"]

        # ── Track position history ───────────────────────────
        if player_id not in self._position_history:
            self._position_history[player_id] = []
        self._position_history[player_id].append((now, x, y))

        # Prune old entries
        cutoff = now - self.WINDOW_SECONDS
        self._position_history[player_id] = [
            (t, px, py) for t, px, py in self._position_history[player_id]
            if t >= cutoff
        ]

        # Compute positional drift (total displacement over window)
        drift = self._compute_drift(player_id)

        # ── Track metabolic history ──────────────────────────
        if player_id not in self._metabolic_history:
            self._metabolic_history[player_id] = []
        self._metabolic_history[player_id].append((now, mp))
        self._metabolic_history[player_id] = [
            (t, m) for t, m in self._metabolic_history[player_id]
            if t >= cutoff
        ]

        # ── Evaluate alert conditions ────────────────────────

        # 1. STRUCTURAL COLLAPSE
        # HR elevated AND position drifting (player physically failing
        # AND leaving their tactical zone)
        if hr_pct >= self.hr_collapse_pct and drift >= self.drift_threshold:
            alerts.append(FusionAlert(
                player_id=player_id,
                alert_type=AlertType.STRUCTURAL_COLLAPSE,
                level=AlertLevel.RED,
                message=(
                    f"#{player_id} HR at {hr_pct:.0%} max, drifted "
                    f"{drift:.1f}m in {self.WINDOW_SECONDS/60:.0f}min. "
                    f"Structural collapse imminent."
                ),
                hr_bpm=hr,
                hr_max_pct=hr_pct,
                metabolic_power=mp,
                positional_drift=drift,
                timestamp=now,
            ))

        # 2. MASKING
        # HR dangerously high BUT position stable (player hiding fatigue)
        elif hr_pct >= self.hr_masking_pct and drift < self.drift_threshold:
            alerts.append(FusionAlert(
                player_id=player_id,
                alert_type=AlertType.MASKING,
                level=AlertLevel.AMBER,
                message=(
                    f"#{player_id} HR at {hr_pct:.0%} max but position "
                    f"stable (drift {drift:.1f}m). Masking fatigue. "
                    f"Proactive sub recommended."
                ),
                hr_bpm=hr,
                hr_max_pct=hr_pct,
                metabolic_power=mp,
                positional_drift=drift,
                timestamp=now,
            ))

        # 3. OVERLOAD
        # Sustained high metabolic power
        avg_mp = self._avg_metabolic(player_id, window=120.0)
        if avg_mp >= self.metabolic_threshold:
            alerts.append(FusionAlert(
                player_id=player_id,
                alert_type=AlertType.OVERLOAD,
                level=AlertLevel.AMBER,
                message=(
                    f"#{player_id} metabolic power {avg_mp:.1f} W/kg "
                    f"sustained > 2min. Internal overload."
                ),
                hr_bpm=hr,
                hr_max_pct=hr_pct,
                metabolic_power=avg_mp,
                positional_drift=drift,
                timestamp=now,
            ))

        return alerts

    def _compute_drift(self, player_id: int) -> float:
        """Compute total positional displacement over the tracking window."""
        history = self._position_history.get(player_id, [])
        if len(history) < 2:
            return 0.0

        # Distance between oldest and newest position
        _, x0, y0 = history[0]
        _, x1, y1 = history[-1]
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    def _avg_metabolic(self, player_id: int,
                       window: float = 120.0) -> float:
        """Average metabolic power over the last `window` seconds."""
        now = time.monotonic()
        history = self._metabolic_history.get(player_id, [])
        recent = [m for t, m in history if (now - t) <= window]
        return sum(recent) / max(len(recent), 1)
