"""
Integration tests for Spatial Bridge (Feature 3).

Tests:
  * F42: Scapular offset (velocity-based fallback)
  * F45: Anatomical posterior axis (facing angle)
  * Backpedaling defender scenario
  * Affine transform estimation
"""
import math
import pytest
import numpy as np

from engine.spatial_bridge import (
    SpatialBridge,
    SCAPULAR_MAX_OFFSET,
    SCAPULAR_ACCEL_SCALE,
)


class TestScapularOffset:
    """F42 + F45: Biomechanical posture offset."""

    def test_no_offset_when_walking(self):
        """No offset when acceleration is below threshold."""
        result = SpatialBridge._apply_scapular_offset(
            gps_xy=(50.0, 30.0),
            velocity_xy=(1.0, 0.0),
            acceleration=0.3,  # below 0.5 threshold
            facing_angle_rad=None,
        )
        assert result == (50.0, 30.0)

    def test_f42_velocity_fallback(self):
        """F42: Without facing angle, use velocity vector."""
        result = SpatialBridge._apply_scapular_offset(
            gps_xy=(50.0, 30.0),
            velocity_xy=(5.0, 0.0),   # running right
            acceleration=3.0,
            facing_angle_rad=None,     # no facing data
        )
        # Should shift LEFT (backward along velocity)
        offset = min(SCAPULAR_MAX_OFFSET, 3.0 * SCAPULAR_ACCEL_SCALE)
        assert result[0] == pytest.approx(50.0 - offset, abs=0.01)
        assert result[1] == pytest.approx(30.0, abs=0.01)

    def test_f45_facing_angle_sprint(self):
        """F45: With facing angle, use anatomical axis."""
        # Player facing right (angle = 0 rad = +X)
        result = SpatialBridge._apply_scapular_offset(
            gps_xy=(50.0, 30.0),
            velocity_xy=(5.0, 0.0),
            acceleration=3.0,
            facing_angle_rad=0.0,  # facing +X
        )
        offset = min(SCAPULAR_MAX_OFFSET, 3.0 * SCAPULAR_ACCEL_SCALE)
        # Posterior axis = -facing = -X
        assert result[0] == pytest.approx(50.0 - offset, abs=0.01)

    def test_f45_backpedaling_defender(self):
        """
        F45 CRITICAL TEST: Backpedaling defender.

        Defender is FACING +Y (facing the attacker) but MOVING -Y
        (backpedaling). Without F45, the velocity-based offset would
        shift the GPS +Y (through their chest). With F45, the offset
        correctly uses the facing angle to shift -facing (behind them).
        """
        # Facing +Y (0, 1) = angle π/2  
        # Velocity -Y (0, -3) = backpedaling
        facing_angle = math.pi / 2  # facing +Y

        result = SpatialBridge._apply_scapular_offset(
            gps_xy=(50.0, 30.0),
            velocity_xy=(0.0, -3.0),   # backpedaling
            acceleration=4.0,
            facing_angle_rad=facing_angle,
        )

        offset = min(SCAPULAR_MAX_OFFSET, 4.0 * SCAPULAR_ACCEL_SCALE)

        # F45: offset is along -facing (posterior = -Y direction)
        # GPS should move in -Y (toward their back), NOT +Y
        assert result[1] < 30.0, (
            "F45 FAILED: Offset went through defender's chest! "
            f"Expected y < 30.0, got y = {result[1]}"
        )
        assert result[1] == pytest.approx(30.0 - offset, abs=0.01)

    def test_offset_clamped_to_max(self):
        """Offset should never exceed SCAPULAR_MAX_OFFSET."""
        result = SpatialBridge._apply_scapular_offset(
            gps_xy=(50.0, 30.0),
            velocity_xy=(8.0, 0.0),
            acceleration=100.0,  # extreme acceleration
            facing_angle_rad=0.0,
        )
        max_shift = SCAPULAR_MAX_OFFSET
        assert abs(50.0 - result[0]) <= max_shift + 0.01


class TestAffineTransform:
    @pytest.fixture
    def bridge(self):
        return SpatialBridge(update_interval=0.0)

    def test_identity_transform_before_calibration(self, bridge):
        assert not bridge.is_calibrated

    def test_calibration_with_pairs(self, bridge):
        """3+ pairs should calibrate the bridge."""
        pairs = [
            ((10, 10), (51.5550, -0.2790)),
            ((50, 30), (51.5554, -0.2785)),
            ((90, 60), (51.5558, -0.2780)),
        ]
        for optical, gps in pairs:
            bridge.update_pair(
                player_id=1,
                optical_xy=optical,
                gps_latlon=gps,
            )
        assert bridge.is_calibrated
        assert bridge.residual_error < 5.0  # rough calibration from 3 points
