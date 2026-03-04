"""
C.R.U.Y.F.F. — Extended Kalman Filter for Sensor Fusion (Feature 3)

Asynchronous multi-rate fusion of:
  * Optical tracking at 25Hz (centimeter-accurate XY)
  * Wearable GPS+HR at 10Hz (1-3m GPS error, precise HR)

State Vector
~~~~~~~~~~~~
::

    x = [x, y, vx, vy, hr, metabolic_power]

The EKF runs a prediction step at 25Hz (optical rate) and an
update step whenever a wearable reading arrives (asynchronous 10Hz).

Fix F33: Mahalanobis Distance Gating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stadium steel structures cause GPS multipath interference,
creating phantom 15m teleportation spikes.  Before accepting
any 10Hz GPS position update, the Mahalanobis distance is
computed.  If D_M > χ²(2, 0.99) = 9.21, the GPS position
is rejected and the EKF coasts on optical data.  HR and
metabolic power are always accepted (no multipath effect).

Dependencies
~~~~~~~~~~~~
* ``numpy``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# State indices
IX, IY, IVX, IVY, IHR, IMP = range(6)
STATE_DIM = 6

# Fix F33: χ²(2, 0.99) = 9.21 — 99% confidence gate for 2-DOF position
MAHALANOBIS_GATE = 9.21


@dataclass
class SensorEKF:
    """
    Extended Kalman Filter for asynchronous optical + wearable fusion.

    Usage::

        ekf = SensorEKF(player_id=8, hr_max=189)

        # At 25Hz: optical update
        ekf.predict_and_update_optical(x=45.2, y=32.1, dt=0.04)

        # At 10Hz (async): wearable update
        ekf.update_wearable(x_pitch=45.0, y_pitch=31.9, hr=172, speed=7.2)

        # Read fused state
        state = ekf.state  # [x, y, vx, vy, hr, metabolic_power]
    """

    player_id: int
    hr_max: float = 200.0  # Maximum heart rate for this player

    # State vector
    _x: np.ndarray = field(default=None, init=False, repr=False)
    # Covariance matrix
    _P: np.ndarray = field(default=None, init=False, repr=False)

    # Process noise
    _Q_optical: np.ndarray = field(default=None, init=False, repr=False)
    _Q_wearable: np.ndarray = field(default=None, init=False, repr=False)

    # Clock drift estimation
    _speed_buffer_optical: list[float] = field(
        default_factory=list, init=False, repr=False
    )
    _speed_buffer_wearable: list[float] = field(
        default_factory=list, init=False, repr=False
    )
    _clock_offset: float = field(default=0.0, init=False, repr=False)
    _gps_rejections: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        # Initial state: centered, stationary, resting HR
        self._x = np.array([0.5, 0.5, 0.0, 0.0, 60.0, 0.0], dtype=np.float64)

        # Initial covariance (high uncertainty)
        self._P = np.diag([
            10.0,   # x position (m²)
            10.0,   # y position
            5.0,    # vx
            5.0,    # vy
            100.0,  # HR
            10.0,   # metabolic power
        ]).astype(np.float64)

        # Process noise — optical (tight position, loose HR/MP)
        self._Q_optical = np.diag([
            0.01,   # x
            0.01,   # y
            0.5,    # vx
            0.5,    # vy
            1.0,    # HR (drifts without measurement)
            0.5,    # MP
        ]).astype(np.float64)

    def predict(self, dt: float) -> None:
        """
        EKF prediction step (constant velocity model).
        Called at 25Hz (optical rate).
        """
        # State transition matrix (constant velocity)
        F = np.eye(STATE_DIM, dtype=np.float64)
        F[IX, IVX] = dt   # x += vx * dt
        F[IY, IVY] = dt   # y += vy * dt

        # Predict state
        self._x = F @ self._x

        # Predict covariance
        Q = self._Q_optical * dt
        self._P = F @ self._P @ F.T + Q

    def update_optical(self, x: float, y: float) -> None:
        """
        EKF update with optical measurement (25Hz, cm-accurate).
        Observation: z = [x, y]
        """
        z = np.array([x, y], dtype=np.float64)

        # Observation matrix (we observe x, y directly)
        H = np.zeros((2, STATE_DIM), dtype=np.float64)
        H[0, IX] = 1.0
        H[1, IY] = 1.0

        # Measurement noise (optical: very low)
        R = np.diag([0.001, 0.001]).astype(np.float64)  # ~3cm accuracy

        self._kalman_update(z, H, R)

        # Update velocity estimate from position delta
        # (optical doesn't directly measure velocity)
        speed = np.sqrt(self._x[IVX] ** 2 + self._x[IVY] ** 2)
        self._speed_buffer_optical.append(speed)
        if len(self._speed_buffer_optical) > 50:
            self._speed_buffer_optical = self._speed_buffer_optical[-50:]

    def predict_and_update_optical(self, x: float, y: float,
                                    dt: float) -> None:
        """Combined predict + optical update (convenience for 25Hz loop)."""
        self.predict(dt)
        self.update_optical(x, y)

    def update_wearable(self, x_pitch: float, y_pitch: float,
                        hr: float, speed: float) -> None:
        """
        EKF update with wearable measurement (10Hz, async).

        Fix F33: Mahalanobis distance gating.
        GPS position is gated; HR/metabolic power always accepted.

        x_pitch, y_pitch are already projected via SpatialBridge.
        """
        # Estimate metabolic power from HR (Banister TRIMP-like)
        hr_reserve = max(0, (hr - 60) / max(1, self.hr_max - 60))
        metabolic_power = hr_reserve * speed * 4.0  # W/kg estimate

        # ── Fix F33: Mahalanobis gate on GPS position ────────
        # Test the position component BEFORE fusing.
        # If the GPS reading is a multipath phantom, reject it.
        z_pos = np.array([x_pitch, y_pitch], dtype=np.float64)
        H_pos = np.zeros((2, STATE_DIM), dtype=np.float64)
        H_pos[0, IX] = 1.0
        H_pos[1, IY] = 1.0
        R_pos = np.diag([2.0, 2.0]).astype(np.float64)

        # Innovation and innovation covariance for position only
        y_pos = z_pos - H_pos @ self._x
        S_pos = H_pos @ self._P @ H_pos.T + R_pos

        # Mahalanobis distance: D_M² = yᵀ S⁻¹ y
        dm_sq = float(y_pos @ np.linalg.inv(S_pos) @ y_pos)

        gps_accepted = dm_sq <= MAHALANOBIS_GATE

        if not gps_accepted:
            # GPS multipath detected — reject position, keep HR/MP
            self._gps_rejections += 1
            logger.warning(
                "Player %d: GPS rejected (D_M²=%.1f > %.1f). "
                "Multipath? Rejections: %d. Coasting on optical.",
                self.player_id, dm_sq, MAHALANOBIS_GATE,
                self._gps_rejections,
            )
            # Fuse HR and metabolic power only (no position)
            z_bio = np.array([hr, metabolic_power], dtype=np.float64)
            H_bio = np.zeros((2, STATE_DIM), dtype=np.float64)
            H_bio[0, IHR] = 1.0
            H_bio[1, IMP] = 1.0
            R_bio = np.diag([1.0, 2.0]).astype(np.float64)
            self._kalman_update(z_bio, H_bio, R_bio)
        else:
            # GPS is clean — fuse full observation
            z = np.array([x_pitch, y_pitch, hr, metabolic_power],
                         dtype=np.float64)
            H = np.zeros((4, STATE_DIM), dtype=np.float64)
            H[0, IX] = 1.0
            H[1, IY] = 1.0
            H[2, IHR] = 1.0
            H[3, IMP] = 1.0
            R = np.diag([2.0, 2.0, 1.0, 2.0]).astype(np.float64)
            self._kalman_update(z, H, R)

        # Track speed for clock drift estimation
        self._speed_buffer_wearable.append(speed)
        if len(self._speed_buffer_wearable) > 50:
            self._speed_buffer_wearable = self._speed_buffer_wearable[-50:]

        # Periodically estimate clock drift
        if len(self._speed_buffer_wearable) >= 20:
            self._estimate_clock_drift()

    def _kalman_update(self, z: np.ndarray, H: np.ndarray,
                       R: np.ndarray) -> None:
        """Standard Kalman update step."""
        # Innovation
        y = z - H @ self._x

        # Innovation covariance
        S = H @ self._P @ H.T + R

        # Kalman gain
        K = self._P @ H.T @ np.linalg.inv(S)

        # Update state
        self._x = self._x + K @ y

        # Update covariance (Joseph form for numerical stability)
        I = np.eye(STATE_DIM, dtype=np.float64)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ R @ K.T

    def _estimate_clock_drift(self) -> None:
        """
        Estimate temporal offset between optical and wearable clocks
        using cross-correlation of speed signals.
        """
        if len(self._speed_buffer_optical) < 20:
            return

        opt = np.array(self._speed_buffer_optical[-20:])
        wear = np.array(self._speed_buffer_wearable[-20:])

        # Normalize
        opt = (opt - opt.mean()) / max(opt.std(), 1e-6)
        wear = (wear - wear.mean()) / max(wear.std(), 1e-6)

        # Cross-correlation
        corr = np.correlate(opt, wear, mode='full')
        lag = np.argmax(corr) - (len(wear) - 1)

        # Convert lag to seconds (optical at 25Hz)
        self._clock_offset = lag / 25.0

    @property
    def state(self) -> dict:
        """Current fused state as a dictionary."""
        return {
            "player_id": self.player_id,
            "x": float(self._x[IX]),
            "y": float(self._x[IY]),
            "vx": float(self._x[IVX]),
            "vy": float(self._x[IVY]),
            "speed": float(np.sqrt(self._x[IVX] ** 2 + self._x[IVY] ** 2)),
            "hr_bpm": float(self._x[IHR]),
            "hr_max_pct": float(self._x[IHR] / self.hr_max),
            "metabolic_power": float(self._x[IMP]),
            "clock_offset": self._clock_offset,
        }

    @property
    def covariance_trace(self) -> float:
        """Trace of covariance matrix — overall uncertainty."""
        return float(np.trace(self._P))
