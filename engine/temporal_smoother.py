"""
C.R.U.Y.F.F. — Temporal Void Smoother (Kalman Filter)

Prevents the "Strobe Light" effect on the Tactical Glass by applying
a lightweight Kalman Filter to each tracked topological void.

Unlike the previous EMA implementation, the Kalman filter maintains
a **velocity state** for each void's centroid, predicting where the
void *will be* on the next frame.  This completely eliminates the
spatial phase-lag that causes the "Blue River" to trail behind a
fast-moving counter-attack gap.

Algorithm
---------
1. On each frame, match new voids to tracked voids via the
   **Hungarian Algorithm** (scipy ``linear_sum_assignment``) using
   predicted centroids (not lagged historical centroids).
2. For matched voids: Kalman *update* step (correct prediction with
   measurement).  Increment stability.
3. For unmatched tracked voids: Kalman *predict-only* (coast).  Decay
   stability.  Evict after ``max_missing_frames``.
4. For new (unmatched) voids: initialise fresh Kalman state.
5. Only emit voids with ``stability ≥ stability_threshold``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from shared.schemas import TopologicalVoid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-void Kalman state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _KalmanVoidState:
    """
    Kalman filter state for a single tracked void.

    State vector: [x, y, vx, vy]  (position + velocity of centroid)
    Measurement:  [x, y]
    """
    # State: [x, y, vx, vy]
    x: NDArray[np.float64]       # (4,)
    P: NDArray[np.float64]       # (4, 4) — covariance

    # Topological metadata (latest measurement)
    birth: float = 0.0
    death: float = 0.0
    persistence: float = 0.0
    death_triangle_indices: tuple[int, int, int] = (0, 0, 0)
    stability: float = 0.0      # 0.0 → 1.0
    frames_missing: int = 0

    @property
    def cx(self) -> float:
        return float(self.x[0])

    @property
    def cy(self) -> float:
        return float(self.x[1])


def _init_kalman_state(
    cx: float, cy: float,
    process_noise: float,
    measurement_noise: float,
) -> _KalmanVoidState:
    """Create a fresh Kalman state at (cx, cy) with zero velocity."""
    x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
    P = np.eye(4, dtype=np.float64) * measurement_noise
    return _KalmanVoidState(x=x, P=P)


# ---------------------------------------------------------------------------
# Kalman matrices (constant — shared across all voids)
# ---------------------------------------------------------------------------

def _make_F(dt: float) -> NDArray[np.float64]:
    """State transition matrix (constant velocity model)."""
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float64)


# Measurement matrix: we observe [x, y]
_H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# TemporalSmoother (Kalman-based)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TemporalSmoother:
    """
    Kalman-filter-based void tracker with Hungarian matching.

    Eliminates both flicker AND phase-lag by predicting void positions
    before matching, then correcting with the actual measurement.

    Parameters
    ----------
    process_noise : float
        Kalman Q diagonal — how much we expect voids to accelerate
        between frames.  Lower = smoother, higher = more responsive.
    measurement_noise : float
        Kalman R diagonal — how much jitter we expect in the raw
        solver centroid.
    stability_threshold : float
        Minimum stability to emit a void to the frontend.
    match_radius : float
        Maximum distance between predicted and measured centroid for
        a valid association.
    max_missing_frames : int
        Frames a void can coast without measurement before eviction.
    dt : float
        Time between frames (1/25 Hz = 0.04s).
    stability_increment : float
        How much stability increases per matched frame.
    """
    process_noise: float = 0.001
    measurement_noise: float = 0.01
    stability_threshold: float = 0.6
    match_radius: float = 0.05
    max_missing_frames: int = 5
    dt: float = 0.04                    # 25 Hz
    stability_increment: float = 0.15

    _tracked: list[_KalmanVoidState] = field(
        default_factory=list, init=False, repr=False,
    )

    # Precomputed Kalman matrices
    _F: NDArray[np.float64] = field(init=False, repr=False)
    _Q: NDArray[np.float64] = field(init=False, repr=False)
    _R: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._F = _make_F(self.dt)
        self._Q = np.eye(4, dtype=np.float64) * self.process_noise
        self._R = np.eye(2, dtype=np.float64) * self.measurement_noise

    # ── public API ─────────────────────────────────────────────────────────

    def smooth(
        self,
        raw_voids: list[TopologicalVoid],
    ) -> list[TopologicalVoid]:
        """
        Ingest one frame's raw solver output, run Kalman predict/update,
        and return only *stable* voids with zero phase-lag centroids.
        """
        # 1. Predict all tracked voids forward
        for tv in self._tracked:
            self._predict(tv)

        # 2. Hungarian matching using *predicted* positions (not lagged)
        self._match_and_update(raw_voids)

        # 3. Emit stable voids
        return self._emit()

    def reset(self) -> None:
        """Clear all tracked state (e.g. at halftime)."""
        self._tracked.clear()

    # ── Kalman predict / update ────────────────────────────────────────────

    def _predict(self, tv: _KalmanVoidState) -> None:
        """Kalman predict step: propagate state forward by dt."""
        tv.x = self._F @ tv.x
        tv.P = self._F @ tv.P @ self._F.T + self._Q

    def _update(
        self,
        tv: _KalmanVoidState,
        z: NDArray[np.float64],
    ) -> None:
        """Kalman update step: correct prediction with measurement z = [x, y]."""
        y = z - _H @ tv.x                              # innovation
        S = _H @ tv.P @ _H.T + self._R                 # innovation covariance
        K = tv.P @ _H.T @ np.linalg.inv(S)             # Kalman gain
        tv.x = tv.x + K @ y                            # corrected state
        tv.P = (np.eye(4) - K @ _H) @ tv.P             # corrected covariance

    # ── matching ───────────────────────────────────────────────────────────

    def _match_and_update(
        self,
        raw_voids: list[TopologicalVoid],
    ) -> None:
        """Hungarian matching on predicted centroids → Kalman update."""
        used_raw: set[int] = set()
        used_tracked: set[int] = set()

        if self._tracked and raw_voids:
            # Predicted centroids (already advanced by _predict)
            t_centroids = np.array(
                [(tv.cx, tv.cy) for tv in self._tracked],
                dtype=np.float64,
            )
            r_centroids = np.array(
                [(rv.centroid_x, rv.centroid_y) for rv in raw_voids],
                dtype=np.float64,
            )

            diffs = t_centroids[:, np.newaxis, :] - r_centroids[np.newaxis, :, :]
            cost_matrix = np.linalg.norm(diffs, axis=2)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for ti, ri in zip(row_ind, col_ind):
                if cost_matrix[ti, ri] > self.match_radius:
                    continue

                tv = self._tracked[ti]
                rv = raw_voids[ri]

                # Kalman update with measurement
                z = np.array([rv.centroid_x, rv.centroid_y], dtype=np.float64)
                self._update(tv, z)

                # Update topological metadata
                tv.birth = rv.birth
                tv.death = rv.death
                tv.persistence = rv.persistence
                tv.death_triangle_indices = rv.death_triangle_indices
                tv.stability = min(1.0, tv.stability + self.stability_increment)
                tv.frames_missing = 0

                used_tracked.add(int(ti))
                used_raw.add(int(ri))

        # Coast unmatched tracked voids (already predicted, just decay)
        to_remove: list[int] = []
        for ti, tv in enumerate(self._tracked):
            if ti not in used_tracked:
                tv.frames_missing += 1
                tv.stability = max(0.0, tv.stability - self.stability_increment * 0.5)
                if tv.frames_missing > self.max_missing_frames:
                    to_remove.append(ti)

        for ti in reversed(to_remove):
            self._tracked.pop(ti)

        # Initialise new voids
        for ri, rv in enumerate(raw_voids):
            if ri not in used_raw:
                state = _init_kalman_state(
                    rv.centroid_x, rv.centroid_y,
                    self.process_noise, self.measurement_noise,
                )
                state.birth = rv.birth
                state.death = rv.death
                state.persistence = rv.persistence
                state.death_triangle_indices = rv.death_triangle_indices
                state.stability = self.stability_increment
                self._tracked.append(state)

    def _emit(self) -> list[TopologicalVoid]:
        """Return only voids that have crossed the stability threshold."""
        return [
            TopologicalVoid(
                centroid_x=tv.cx,     # Kalman-corrected, zero-lag
                centroid_y=tv.cy,
                birth=tv.birth,
                death=tv.death,
                persistence=tv.persistence,
                death_triangle_indices=tv.death_triangle_indices,
                stability=tv.stability,
            )
            for tv in self._tracked
            if tv.stability >= self.stability_threshold
        ]
