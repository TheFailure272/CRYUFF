"""
C.R.U.Y.F.F. — Spatial Bridge (Feature 3)

Dynamic Affine Transform: WGS84 → Pitch Coordinates.

Fix F42 + F45: Scapular Biomechanical Offset (Anatomical Axis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optical tracks feet/pelvis. GPS pod sits between scapulae (upper back).
During sprint (45° torso lean), GPS is 0.5-1.0m ahead of pelvis.

F42 originally shifted backward along velocity vector, but this fails
for backpedaling defenders (velocity is backward, offset goes through
the chest).

F45: Uses the Optical Orientation Tensor (facing angle from hip
orientation) to project along the anatomical posterior axis,
completely independent of velocity direction.

Problem
-------
Catapult GPS outputs absolute WGS84 (lat, lon) with 1-3m error.
Optical tracking outputs relative pitch coordinates (x, y) in meters
with centimeter accuracy.  Direct fusion is impossible without
projecting GPS coords into the optical pitch space.

Solution
--------
Maintain a Dynamic Affine Transform matrix that maps WGS84 → pitch:

    [x_pitch]     [a  b  tx] [x_gps]
    [y_pitch]  =  [c  d  ty] [y_gps]
    [1      ]     [0  0   1] [1    ]

The transform is continuously re-estimated every 5 seconds using
least-squares on matched player positions (GPS vs optical).  The
optical data is the "ground truth" anchor, correcting the wandering
GPS constellation error.

Dependencies
~~~~~~~~~~~~
* ``numpy`` for matrix operations
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fix F42: Scapular offset constants
# Distance from scapular GPS pod to pelvis center-of-mass
SCAPULAR_MAX_OFFSET = 0.85   # meters (max torso lean at full sprint)
SCAPULAR_ACCEL_SCALE = 0.18  # offset per m/s² of acceleration


@dataclass
class SpatialBridge:
    """
    Dynamic affine transform from WGS84 GPS to pitch-relative coordinates.

    Usage::

        bridge = SpatialBridge()

        # Each frame: feed matched pairs (optical + GPS) for the same player
        bridge.update_pair(player_id=8, optical_xy=(45.2, 32.1), gps_latlon=(51.555, -0.279))

        # Convert a GPS reading to pitch coords
        pitch_xy = bridge.gps_to_pitch(lat=51.555, lon=-0.279)
    """

    update_interval: float = 5.0  # Re-estimate transform every N seconds

    # Affine matrix [2x3]: [[a, b, tx], [c, d, ty]]
    _affine: np.ndarray = field(
        default=None, init=False, repr=False
    )
    # Collected matched pairs for least-squares
    _pairs_optical: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _pairs_gps: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )
    _last_update: float = field(default=0.0, init=False, repr=False)
    _is_calibrated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Initialize with identity transform (no correction)
        self._affine = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

    @staticmethod
    def _latlon_to_local(lat: float, lon: float,
                         ref_lat: float = 0.0,
                         ref_lon: float = 0.0) -> tuple[float, float]:
        """
        Convert WGS84 lat/lon to local meters relative to a reference.
        Uses equirectangular approximation (valid for < 1km distances).
        """
        R = 6_371_000  # Earth radius in meters
        x = (lon - ref_lon) * np.radians(1) * R * np.cos(np.radians(ref_lat))
        y = (lat - ref_lat) * np.radians(1) * R
        return float(x), float(y)

    def update_pair(self, player_id: int,
                    optical_xy: tuple[float, float],
                    gps_latlon: tuple[float, float],
                    velocity_xy: tuple[float, float] = (0.0, 0.0),
                    acceleration: float = 0.0,
                    facing_angle_rad: float | None = None) -> None:
        """
        Feed a matched pair of optical + GPS coordinates for a player.

        Parameters
        ----------
        player_id : int
            Player identifier (used for logging).
        optical_xy : tuple
            (x, y) from optical tracking (meters, pitch-relative).
        gps_latlon : tuple
            (lat, lon) from GPS vest (WGS84).
        velocity_xy : tuple
            (vx, vy) player velocity vector from EKF (m/s).
        acceleration : float
            Player acceleration magnitude (m/s²).
        facing_angle_rad : float, optional
            Fix F45: Player's anatomical facing direction (radians)
            from optical tracking hip orientation tensor.
            If provided, scapular offset uses this instead of velocity.
        """
        # Convert GPS to local meters
        if not self._pairs_gps:
            # First pair becomes the reference origin
            self._ref_lat = gps_latlon[0]
            self._ref_lon = gps_latlon[1]

        ref_lat = getattr(self, '_ref_lat', gps_latlon[0])
        ref_lon = getattr(self, '_ref_lon', gps_latlon[1])
        gps_local = self._latlon_to_local(
            gps_latlon[0], gps_latlon[1], ref_lat, ref_lon
        )

        # Fix F42 + F45: Apply scapular biomechanical offset
        # Uses anatomical facing angle (F45) if available,
        # falls back to velocity vector (F42)
        gps_local = self._apply_scapular_offset(
            gps_local, velocity_xy, acceleration, facing_angle_rad
        )

        self._pairs_optical.append(optical_xy)
        self._pairs_gps.append(gps_local)

        # Limit buffer to 100 most recent pairs
        if len(self._pairs_optical) > 100:
            self._pairs_optical = self._pairs_optical[-100:]
            self._pairs_gps = self._pairs_gps[-100:]

        # Re-estimate transform periodically
        now = time.monotonic()
        if (now - self._last_update) >= self.update_interval:
            self._estimate_affine()
            self._last_update = now

    def _estimate_affine(self) -> None:
        """
        Estimate the affine transform using least-squares.

        Solves: optical = A @ gps + t  (over-determined system)
        """
        n = len(self._pairs_optical)
        if n < 3:
            # Need at least 3 non-collinear points for affine
            return

        # Build matrices
        # src = GPS local coords, dst = optical coords
        src = np.array(self._pairs_gps, dtype=np.float64)    # (n, 2)
        dst = np.array(self._pairs_optical, dtype=np.float64)  # (n, 2)

        # Augment source with ones for translation
        # [x_gps, y_gps, 1] → [a*x + b*y + tx, c*x + d*y + ty]
        ones = np.ones((n, 1), dtype=np.float64)
        A = np.hstack([src, ones])  # (n, 3)

        # Solve via least-squares: A @ params = dst
        # params shape: (3, 2) — two columns for x_out and y_out
        result_x, _, _, _ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
        result_y, _, _, _ = np.linalg.lstsq(A, dst[:, 1], rcond=None)

        self._affine = np.array([result_x, result_y], dtype=np.float64)
        self._is_calibrated = True

        # Compute residual error
        predicted = A @ self._affine.T
        residuals = np.sqrt(np.mean((predicted - dst) ** 2, axis=0))
        logger.info(
            "Spatial bridge re-estimated (n=%d pairs, "
            "residual: x=%.3fm, y=%.3fm)",
            n, residuals[0], residuals[1],
        )

    def gps_to_pitch(self, lat: float, lon: float,
                     velocity_xy: tuple[float, float] = (0.0, 0.0),
                     acceleration: float = 0.0,
                     facing_angle_rad: float | None = None,
                     ) -> tuple[float, float] | None:
        """
        Convert a GPS lat/lon to pitch-relative coordinates.

        Returns None if the bridge is not yet calibrated.
        """
        ref_lat = getattr(self, '_ref_lat', 0.0)
        ref_lon = getattr(self, '_ref_lon', 0.0)
        gps_local = self._latlon_to_local(lat, lon, ref_lat, ref_lon)

        # Fix F42 + F45: Apply scapular offset before transform
        gps_local = self._apply_scapular_offset(
            gps_local, velocity_xy, acceleration, facing_angle_rad
        )

        # Apply affine transform
        vec = np.array([gps_local[0], gps_local[1], 1.0], dtype=np.float64)
        result = self._affine @ vec
        return (float(result[0]), float(result[1]))

    @staticmethod
    def _apply_scapular_offset(
        gps_xy: tuple[float, float],
        velocity_xy: tuple[float, float],
        acceleration: float,
        facing_angle_rad: float | None = None,
    ) -> tuple[float, float]:
        """
        Fix F42 + F45: Scapular biomechanical posture offset.

        F45 upgrade: Uses the Optical Orientation Tensor (facing angle)
        instead of velocity vector for offset direction.

        The GPS pod sits on the player's BACK. The offset must always
        be along the player's anatomical posterior axis — the direction
        their spine is facing — NOT their velocity.

        A backpedaling defender moves backward (velocity = -Y) while
        facing forward (facing = +Y). The GPS pod is still on their
        back, so the offset must be along -facing (posterior), not
        along -velocity (which would push through their chest).

        Hierarchy:
          1. facing_angle_rad (optical hip orientation) — F45
          2. velocity direction (fallback) — F42
        """
        speed = (velocity_xy[0] ** 2 + velocity_xy[1] ** 2) ** 0.5

        if acceleration < 0.5:
            # Standing or walking — no significant torso lean
            return gps_xy

        # Offset magnitude: scales with acceleration
        offset = min(SCAPULAR_MAX_OFFSET, acceleration * SCAPULAR_ACCEL_SCALE)

        if facing_angle_rad is not None:
            # Fix F45: Use anatomical facing direction (optical tensor)
            # Offset is along POSTERIOR axis = opposite of facing
            import math
            facing_x = math.cos(facing_angle_rad)
            facing_y = math.sin(facing_angle_rad)
            # Posterior = -facing (offset toward the player's back)
            corrected_x = gps_xy[0] - offset * facing_x
            corrected_y = gps_xy[1] - offset * facing_y
        elif speed >= 0.5:
            # Fallback F42: use velocity vector
            v_hat_x = velocity_xy[0] / speed
            v_hat_y = velocity_xy[1] / speed
            corrected_x = gps_xy[0] - offset * v_hat_x
            corrected_y = gps_xy[1] - offset * v_hat_y
        else:
            return gps_xy

        return (corrected_x, corrected_y)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def residual_error(self) -> float:
        """Mean residual error of the current transform (meters)."""
        if len(self._pairs_optical) < 3:
            return float('inf')
        src = np.array(self._pairs_gps, dtype=np.float64)
        dst = np.array(self._pairs_optical, dtype=np.float64)
        ones = np.ones((len(src), 1), dtype=np.float64)
        A = np.hstack([src, ones])
        predicted = A @ self._affine.T
        return float(np.sqrt(np.mean((predicted - dst) ** 2)))
