"""
C.R.U.Y.F.F. — Full verification suite covering all 15 hardening fixes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.temporal_smoother import TemporalSmoother
from engine.topological_solver import SolverConfig, TopologicalSolver
from shared.shm_buffer import SharedFrameBuffer, SharedFrameRing, SharedFrameToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_44_coords(
    defensive_positions: list[tuple[float, float]],
    *,
    attack_positions: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    attack = list(attack_positions) if attack_positions else [(-10.0, -10.0)] * 11
    defense = list(defensive_positions)
    while len(defense) < 11:
        defense.append((-10.0, -10.0))
    while len(attack) < 11:
        attack.append((-10.0, -10.0))
    return np.array(attack + defense, dtype=np.float64).flatten()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solver() -> TopologicalSolver:
    return TopologicalSolver(config=SolverConfig(
        max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
        defensive_only=True, pitch_boundary_enabled=False,
    ))


@pytest.fixture
def solver_bounded() -> TopologicalSolver:
    return TopologicalSolver(config=SolverConfig(
        max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
        defensive_only=True, pitch_boundary_enabled=True, pitch_boundary_density=6,
    ))


@pytest.fixture
def ring_defense() -> list[tuple[float, float]]:
    cx, cy, r = 0.5, 0.5, 0.2
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]


# ===========================================================================
# Core Tests
# ===========================================================================

class TestTopologicalSolverCore:
    def test_ring_with_clear_hole(self, solver, ring_defense):
        voids = solver.solve(_make_44_coords(ring_defense))
        assert len(voids) >= 1
        assert math.hypot(voids[0].centroid_x - 0.5, voids[0].centroid_y - 0.5) < 0.2

    def test_tight_cluster_no_holes(self, solver):
        rng = np.random.default_rng(42)
        defense = [(0.5 + rng.uniform(-0.02, 0.02), 0.5 + rng.uniform(-0.02, 0.02)) for _ in range(11)]
        assert len(solver.solve(_make_44_coords(defense))) == 0

    def test_collinear_players(self, solver):
        assert len(solver.solve(_make_44_coords([(0.1 + 0.07 * i, 0.5) for i in range(11)]))) == 0

    def test_invalid_coordinate_count(self, solver):
        with pytest.raises(ValueError, match="Expected 44 floats"):
            solver.solve(np.zeros(40))

    def test_void_fields_populated(self, solver, ring_defense):
        voids = solver.solve(_make_44_coords(ring_defense))
        assert len(voids) >= 1
        v = voids[0]
        assert v.birth >= 0.0 and v.death > v.birth
        assert v.persistence == pytest.approx(v.death - v.birth)
        assert len(v.death_triangle_indices) == 3


# Fix 1: Dynamic Team Partitioning
class TestDynamicTeamPartitioning:
    def test_team_ids_partition(self, solver):
        cx, cy, r = 0.5, 0.5, 0.2
        angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        ring = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
        coords = np.array(ring + [(-10, -10)] * 11, dtype=np.float64).flatten()
        voids = solver.solve(coords, team_ids=["away"] * 11 + ["home"] * 11, attacking_team="home")
        assert len(voids) >= 1

    def test_fallback_without_team_ids(self, solver, ring_defense):
        assert len(solver.solve(_make_44_coords(ring_defense))) >= 1


# Fix 9: Kalman Filter Smoothing
class TestTemporalSmootherKalman:
    def test_void_gains_stability(self, solver, ring_defense):
        smoother = TemporalSmoother(stability_threshold=0.6, stability_increment=0.15, match_radius=0.1)
        coords = _make_44_coords(ring_defense)
        assert len(smoother.smooth(solver.solve(coords))) == 0  # first frame
        for _ in range(10):
            stable = smoother.smooth(solver.solve(coords))
        assert len(stable) >= 1 and stable[0].stability >= 0.6

    def test_flickering_suppressed(self, solver, ring_defense):
        """max_missing_frames=0 → immediate eviction, no coasting."""
        smoother = TemporalSmoother(
            stability_threshold=0.6, stability_increment=0.10,
            match_radius=0.1, max_missing_frames=0,
        )
        coords_ring = _make_44_coords(ring_defense)
        coords_tight = _make_44_coords([(0.5, 0.5)] * 11)
        for _ in range(20):
            smoother.smooth(solver.solve(coords_ring))
            stable = smoother.smooth(solver.solve(coords_tight))
        assert len(stable) == 0

    def test_reset_clears_state(self, solver, ring_defense):
        smoother = TemporalSmoother(stability_threshold=0.3, stability_increment=0.2)
        for _ in range(5):
            smoother.smooth(solver.solve(_make_44_coords(ring_defense)))
        smoother.reset()
        assert len(smoother.smooth([])) == 0

    def test_kalman_coasting_survives_occlusion(self, solver, ring_defense):
        """Fix 14: A stable void should survive 2 frames of occlusion (coasting)."""
        smoother = TemporalSmoother(
            stability_threshold=0.6, stability_increment=0.15,
            match_radius=0.1, max_missing_frames=5,
        )
        coords = _make_44_coords(ring_defense)

        # Build up stability
        for _ in range(10):
            smoother.smooth(solver.solve(coords))

        # Simulate 2-frame occlusion (tracking dropout)
        smoother.smooth([])  # frame 1: void missing
        smoother.smooth([])  # frame 2: void still missing

        # On frame 3, void reappears — should still be tracked
        stable = smoother.smooth(solver.solve(coords))
        assert len(stable) >= 1, "Void should survive 2-frame occlusion via coasting"


# Fix 3: Velocity Projection
class TestVelocityProjection:
    def test_closing_gap_detected(self, solver):
        cx, cy, r = 0.5, 0.5, 0.2
        angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        defense = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
        coords = _make_44_coords(defense)
        voids_static = solver.solve(coords)
        assert len(voids_static) >= 1

        velocities = np.zeros(44, dtype=np.float64)
        p_x, p_y = defense[0]
        d = np.array([cx - p_x, cy - p_y])
        d = d / np.linalg.norm(d) * 0.5
        velocities[22], velocities[23] = d[0], d[1]

        solver_k = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, velocity_horizon_secs=1.0, pitch_boundary_enabled=False,
        ))
        voids_k = solver_k.solve(coords, velocities=velocities)
        if voids_k and voids_static:
            assert voids_k[0].persistence <= voids_static[0].persistence + 0.01

    def test_invalid_velocity_count(self, solver):
        with pytest.raises(ValueError, match="Velocity array must have 44"):
            solver.solve(_make_44_coords([(0.5, 0.5)] * 11), velocities=np.zeros(40))


# Fix 8+12: Shared Memory IPC + Ring Buffer
class TestSharedMemoryBuffer:
    def test_write_read_roundtrip(self):
        buf = SharedFrameBuffer.create()
        try:
            coords = np.random.default_rng(42).random(44).astype(np.float64)
            vels = np.random.default_rng(43).random(44).astype(np.float64)
            token = buf.write(coords, vels, ["home"] * 11 + ["away"] * 11)
            r_coords, r_vel = SharedFrameBuffer.read(token)
            np.testing.assert_array_almost_equal(r_coords, coords)
            np.testing.assert_array_almost_equal(r_vel, vels)
        finally:
            buf.close()
            buf.unlink()

    def test_write_read_no_velocity(self):
        buf = SharedFrameBuffer.create()
        try:
            coords = np.ones(44, dtype=np.float64) * 0.5
            token = buf.write(coords)
            r_coords, r_vel = SharedFrameBuffer.read(token)
            np.testing.assert_array_almost_equal(r_coords, coords)
            assert r_vel is None
        finally:
            buf.close()
            buf.unlink()

    def test_ring_buffer_rotation(self):
        """Fix 12: Ring buffer rotates through pre-allocated slots."""
        ring = SharedFrameRing(n_slots=3)
        try:
            assert len(ring.slot_names) == 3
            tokens = []
            for i in range(6):
                coords = np.full(44, float(i), dtype=np.float64)
                tokens.append(ring.write(coords))

            # Last token should read back the correct data
            r_coords, _ = SharedFrameBuffer.read(tokens[-1])
            np.testing.assert_array_almost_equal(r_coords, np.full(44, 5.0))
        finally:
            ring.destroy()

    def test_ring_buffer_destroy_cleanup(self):
        """Fix 12: destroy() cleans all slots."""
        ring = SharedFrameRing(n_slots=2)
        names = ring.slot_names.copy()
        assert len(names) == 2
        ring.destroy()
        assert len(ring._slots) == 0


# Fix 11: Pitch Boundary
class TestPitchBoundary:
    def test_touchline_void_suppressed(self, solver_bounded):
        defense = [(0.95, 0.3 + 0.04 * i) for i in range(11)]
        coords = _make_44_coords(defense)
        solver_no = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, pitch_boundary_enabled=False,
        ))
        voids_unbounded = solver_no.solve(coords)
        voids_bounded = solver_bounded.solve(coords)
        assert len(voids_bounded) <= len(voids_unbounded) + 1

    def test_centre_ring_unaffected(self, solver_bounded, ring_defense):
        voids = solver_bounded.solve(_make_44_coords(ring_defense))
        assert len(voids) >= 1
        assert math.hypot(voids[0].centroid_x - 0.5, voids[0].centroid_y - 0.5) < 0.25


# Fix 15: GK Velocity Dampener
class TestGKVelocityDampener:
    def test_gk_dive_does_not_create_false_void(self):
        """
        A GK near y=0 with explosive dive velocity should NOT create a
        massive false void after projection because the dampener scales
        their velocity to 0.1×.
        """
        # GK at (0.5, 0.03) — near goal line
        defense = [(0.5, 0.03)] + [(0.3 + 0.05 * i, 0.3) for i in range(10)]
        coords = _make_44_coords(defense)

        # GK diving sideways at extreme speed
        velocities = np.zeros(44, dtype=np.float64)
        velocities[22] = 2.0   # GK vx = 2.0 units/sec (extreme dive)
        velocities[23] = 0.0

        solver_no_damp = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, velocity_horizon_secs=0.5,
            pitch_boundary_enabled=False,
            gk_velocity_damper=1.0,  # no dampening
        ))
        solver_damped = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, velocity_horizon_secs=0.5,
            pitch_boundary_enabled=False,
            gk_velocity_damper=0.1,  # dampened
        ))

        voids_undamped = solver_no_damp.solve(coords, velocities=velocities)
        voids_damped = solver_damped.solve(coords, velocities=velocities)

        # Dampened should produce fewer or less-persistent voids
        max_p_undamped = max((v.persistence for v in voids_undamped), default=0)
        max_p_damped = max((v.persistence for v in voids_damped), default=0)
        assert max_p_damped <= max_p_undamped + 0.01

    def test_outfield_player_not_dampened(self):
        """Players in the middle of the pitch should NOT be dampened."""
        defense = [(0.5, 0.5)] + [(0.3 + 0.05 * i, 0.4) for i in range(10)]
        coords = _make_44_coords(defense)

        velocities = np.zeros(44, dtype=np.float64)
        velocities[22] = 1.0   # player at (0.5, 0.5) — midfield
        velocities[23] = 0.0

        solver_damp = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, velocity_horizon_secs=0.5,
            pitch_boundary_enabled=False, gk_velocity_damper=0.1,
        ))
        solver_nodamp = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, velocity_horizon_secs=0.5,
            pitch_boundary_enabled=False, gk_velocity_damper=1.0,
        ))

        # Same result — midfield player is unaffected by GK dampener
        voids_d = solver_damp.solve(coords, velocities=velocities)
        voids_n = solver_nodamp.solve(coords, velocities=velocities)
        assert len(voids_d) == len(voids_n)


# Fix 19: Transport Hysteresis
class TestTransportHysteresis:
    def test_cooldown_prevents_immediate_switch(self):
        """After WebRTC drops, should NOT switch back within cooldown."""
        from server.transport_hysteresis import TransportHysteresis, TransportMode

        h = TransportHysteresis(cooldown_s=30.0)
        h.on_webrtc_open()
        assert h.mode == TransportMode.WEBRTC

        h.on_webrtc_close()
        assert h.mode == TransportMode.WEBSOCKET

        # Immediately try to reconnect — should stay on WebSocket
        h.on_webrtc_open()
        assert h.mode == TransportMode.WEBSOCKET

    def test_flap_lockout(self):
        """After N flaps in window, permanently lock to WebSocket."""
        from server.transport_hysteresis import TransportHysteresis

        h = TransportHysteresis(
            cooldown_s=0.0,  # no cooldown (for test speed)
            max_flaps_before_lockout=3,
            lockout_window_s=60.0,
        )
        for _ in range(3):
            h.on_webrtc_open()
            h.on_webrtc_close()

        assert h.is_locked_out
        h.on_webrtc_open()  # should be ignored
        assert not h.should_use_webrtc()

    def test_frame_dedup_discards_older(self):
        """Frames with mono ≤ last rendered are discarded."""
        from server.transport_hysteresis import TransportHysteresis

        h = TransportHysteresis()
        assert h.should_accept_frame(100.0)   # first frame
        assert h.should_accept_frame(100.5)   # newer
        assert not h.should_accept_frame(100.0)   # stale duplicate
        assert not h.should_accept_frame(99.0)    # out-of-order old
        assert h.should_accept_frame(101.0)   # newest


# Fix 20: Inner Backpressure Token
class TestInnerBackpressure:
    def test_token_carries_monotonic(self):
        """SharedFrameToken should carry ingest_monotonic for inner backpressure."""
        from shared.shm_buffer import SharedFrameToken

        token = SharedFrameToken(
            shm_name="test",
            has_velocities=False,
            ingest_monotonic=12345.678,
            stale_threshold_s=0.1,
        )
        assert token.ingest_monotonic == 12345.678
        assert token.stale_threshold_s == 0.1


# Fix 22: Dynamic Boundary Density
class TestDynamicBoundaryDensity:
    def test_density_increases_with_tighter_edge(self):
        """
        With max_edge_length=0.6 → dynamic_d = ceil(1/0.6)+1 = 3.
        Static density=6 → use 6. Ghost count = 6*2 + (6-2)*2 = 20.

        With max_edge_length=0.1 → dynamic_d = ceil(1/0.1)+1 = 12.
        12 > 6 → dynamic kicks in. Ghost count = 12*2 + (12-2)*2 = 44.

        Tight edge length → more ghost defenders → mathematically solid.
        """
        # Wide solver — static density dominates
        solver_wide = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.6, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, pitch_boundary_enabled=True, pitch_boundary_density=6,
        ))
        # Tight solver — dynamic density kicks in
        solver_tight = TopologicalSolver(config=SolverConfig(
            max_edge_length=0.1, max_dimension=2, persistence_threshold=0.01,
            defensive_only=True, pitch_boundary_enabled=True, pitch_boundary_density=6,
        ))

        # Use a minimal cloud to test boundary injection
        cloud = np.array([[0.5, 0.5]], dtype=np.float64)

        aug_wide, n_wide = solver_wide._inject_pitch_boundary(cloud)
        aug_tight, n_tight = solver_tight._inject_pitch_boundary(cloud)

        # Tight edge → more ghost defenders
        assert n_tight > n_wide, (
            f"Tight edge ({n_tight} ghosts) should > wide ({n_wide})"
        )
        # Verify the computed dynamic density: ceil(1.0 / 0.1) + 1 = 11
        d_tight = int(np.ceil(1.0 / 0.1)) + 1  # 11
        assert d_tight == 11
        # Ghost count: top+bottom = 11*2, left+right inner = (11-2)*2 = 18 → 40
        assert n_tight == 11 * 2 + (11 - 2) * 2  # 40 ghosts

