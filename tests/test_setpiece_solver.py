"""
Integration tests for Set-Piece Solver (Feature 4).

Tests:
  * F41: Drag crisis — Reynolds-dependent Cd
  * Air density calculation
  * Heatmap generation
  * Top zone extraction
"""
import math
import pytest
import numpy as np

from engine.setpiece_solver import (
    SetPieceSolver,
    compute_air_density,
    _compute_cd_reynolds,
    BALL_DIAMETER,
    AIR_VISCOSITY,
    CD_SUBCRITICAL,
    CD_TURBULENT,
    RE_CRITICAL_LOW,
    RE_CRITICAL_HIGH,
)


class TestDragCrisis:
    """F41: Reynolds-dependent drag coefficient."""

    def test_low_speed_subcritical(self):
        """At low speed, Cd should be subcritical (~0.25)."""
        # 10 m/s → Re ≈ 1.49e5 (below critical)
        cd = _compute_cd_reynolds(10.0, 1.225)
        assert cd == pytest.approx(CD_SUBCRITICAL, abs=0.01)

    def test_high_speed_turbulent(self):
        """At high speed, Cd drops to turbulent (~0.15)."""
        # 40 m/s → Re ≈ 5.95e5 (above critical)
        cd = _compute_cd_reynolds(40.0, 1.225)
        assert cd == pytest.approx(CD_TURBULENT, abs=0.01)

    def test_crisis_zone_interpolation(self):
        """In the crisis zone, Cd interpolates smoothly."""
        # Find speed at mid-crisis Re
        mid_Re = (RE_CRITICAL_LOW + RE_CRITICAL_HIGH) / 2
        speed = mid_Re * AIR_VISCOSITY / (1.225 * BALL_DIAMETER)
        cd = _compute_cd_reynolds(speed, 1.225)
        expected_cd = (CD_SUBCRITICAL + CD_TURBULENT) / 2
        assert cd == pytest.approx(expected_cd, abs=0.01)

    def test_reynolds_number_formula(self):
        """Verify Re calculation."""
        speed = 30.0
        rho = 1.225
        Re = rho * speed * BALL_DIAMETER / AIR_VISCOSITY
        # 1.225 * 30.0 * 0.22 / 1.81e-5 ≈ 446,685
        assert Re > RE_CRITICAL_HIGH  # should be turbulent


class TestAirDensity:
    def test_sea_level(self):
        rho = compute_air_density(0, 15)
        assert rho == pytest.approx(1.225, abs=0.01)

    def test_altitude_decreases_density(self):
        """Higher altitude → less dense air → less drag."""
        rho_sea = compute_air_density(0, 20)
        rho_madrid = compute_air_density(610, 20)
        rho_mexico = compute_air_density(2200, 20)
        assert rho_sea > rho_madrid > rho_mexico

    def test_temperature_decreases_density(self):
        """Higher temperature → less dense air."""
        rho_cold = compute_air_density(0, 0)
        rho_hot = compute_air_density(0, 35)
        assert rho_cold > rho_hot


class TestSolver:
    @pytest.fixture
    def solver(self):
        return SetPieceSolver(altitude_m=0, temperature_c=15, n_samples=100)

    def test_solve_returns_heatmap(self, solver):
        result = solver.solve(
            origin_m=(0.0, 0.0, 0.3),
            target_y_m=16.5,
            delivery_type="inswing",
            n_samples=100,
        )
        assert "heatmap" in result
        assert "top_zones" in result
        assert result["n_samples"] == 100
        assert result["rho"] == pytest.approx(1.225, abs=0.01)

    def test_solve_with_gk_filter(self, solver):
        result = solver.solve(
            origin_m=(0.0, 0.0, 0.3),
            target_y_m=16.5,
            delivery_type="driven",
            gk_position_m=(34.0, 5.5),
            gk_catching_radius=3.0,
            n_samples=100,
        )
        assert "top_zones" in result
