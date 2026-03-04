"""
C.R.U.Y.F.F. — Layer 1: "The Flow"

TopologicalSolver
=================
Ingests a 44-float tracking array (+ optional team_ids and velocities),
builds a Vietoris-Rips complex on the defensive point cloud, computes
persistent homology, and extracts the (x, y) centroids of β₁ voids
(passing lanes) via the death-triangle heuristic.

Live-match hardening
--------------------
* **Dynamic team partitioning** — uses ``team_ids`` when available;
  falls back to index-based split (11–21) only as a last resort.
* **Kinematic projection** — when velocities are provided, positions
  are projected forward by ``velocity_horizon`` seconds *before*
  computing the Rips complex, so the topology reflects the *future*
  defensive structure, not the present snapshot.

Designed for ≤40ms per call on a modern laptop — fast enough for 25Hz
when offloaded to a thread pool via ``run_in_executor``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

import gudhi                          # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from shared.schemas import TopologicalVoid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_PLAYERS: Final[int] = 22
_COORDS_PER_PLAYER: Final[int] = 2
_TOTAL_FLOATS: Final[int] = _NUM_PLAYERS * _COORDS_PER_PLAYER  # 44
_DEFENSIVE_SLICE: Final[slice] = slice(11, 22)  # legacy fallback


@dataclass(slots=True, frozen=True)
class SolverConfig:
    """Immutable solver hyper-parameters."""
    max_edge_length: float = 0.35
    max_dimension: int = 2            # must be ≥2 to detect 1-cycles
    persistence_threshold: float = 0.04
    defensive_only: bool = True
    velocity_horizon_secs: float = 0.5  # capped: linear projection unreliable beyond ~0.5s
    pitch_boundary_enabled: bool = True  # inject ghost defenders along pitch edges
    pitch_boundary_density: int = 6      # ghost defenders per edge (4 edges)
    gk_velocity_damper: float = 0.1      # scalar multiplier for GK velocity (Fix 15)
    gk_goal_line_threshold: float = 0.08  # players within this y-dist of goal line = GK


@dataclass(slots=True)
class TopologicalSolver:
    """
    Stateless, re-entrant solver.  Safe to share across threads.

    Usage::

        solver = TopologicalSolver(config=SolverConfig())

        # Basic (index-based partitioning)
        voids = solver.solve(raw_44_floats)

        # Live-match (dynamic team + velocity projection)
        voids = solver.solve(
            raw_44_floats,
            team_ids=["home"]*11 + ["away"]*11,
            velocities=raw_44_velocity_floats,
            attacking_team="home",
        )
    """
    config: SolverConfig = field(default_factory=SolverConfig)

    # ── public API ─────────────────────────────────────────────────────────

    def solve(
        self,
        coordinates: NDArray[np.float64] | Sequence[float],
        *,
        team_ids: list[Literal["home", "away"]] | None = None,
        velocities: NDArray[np.float64] | Sequence[float] | None = None,
        attacking_team: Literal["home", "away"] = "home",
    ) -> list[TopologicalVoid]:
        """
        End-to-end pipeline: flat 44-float array → list of tactical voids
        sorted by descending persistence (most exploitable first).

        Parameters
        ----------
        coordinates : array-like, shape (44,)
            Flat [x0, y0, x1, y1, …, x21, y21] in normalised pitch coords.
        team_ids : list of "home"|"away", optional
            Per-player team label.  Enables dynamic partitioning and
            survives halftime switches / red cards.
        velocities : array-like, shape (44,), optional
            Flat [vx0, vy0, …, vx21, vy21] in pitch-units/sec.
            When provided, positions are projected forward before
            computing the Rips complex.
        attacking_team : "home" | "away"
            Which team we are analysing *for*.  The opposing team's
            structure is used for the Rips complex.

        Returns
        -------
        list[TopologicalVoid]
            Each entry carries the death-triangle centroid and persistence.
        """
        points = self._reshape(np.asarray(coordinates, dtype=np.float64))

        # ── Kinematic projection (Fix 3) + GK dampener (Fix 15) ─────
        if velocities is not None:
            vel = np.asarray(velocities, dtype=np.float64)
            if vel.size != _TOTAL_FLOATS:
                raise ValueError(
                    f"Velocity array must have {_TOTAL_FLOATS} floats, "
                    f"got {vel.size}"
                )
            vel_2d = vel.reshape(_NUM_PLAYERS, _COORDS_PER_PLAYER).copy()

            # Dampen goalkeeper velocity — diving GK projected 0.5s forward
            # can end up outside the goal, hallucinating a massive false void
            gk_thresh = self.config.gk_goal_line_threshold
            for p_idx in range(_NUM_PLAYERS):
                y = points[p_idx, 1]
                if y < gk_thresh or y > (1.0 - gk_thresh):
                    vel_2d[p_idx] *= self.config.gk_velocity_damper

            points = points + vel_2d * self.config.velocity_horizon_secs

        # ── Team partitioning (Fix 1) ─────────────────────────────────
        cloud, global_indices = self._partition(
            points, team_ids, attacking_team,
        )

        # ── Pitch boundary clamping (Fix 11) ─────────────────────────────
        if self.config.pitch_boundary_enabled:
            cloud, n_ghosts = self._inject_pitch_boundary(cloud)
        else:
            n_ghosts = 0

        stree = self._build_filtration(cloud)
        intervals = self._extract_h1(stree)
        # Pass only real-player cloud portion for centroid calc
        voids = self._intervals_to_voids(
            stree, intervals, cloud, global_indices, n_ghosts,
        )

        # Descending persistence — biggest gap first
        voids.sort(key=lambda v: v.persistence, reverse=True)
        return voids

    # ── private pipeline stages ────────────────────────────────────────────

    @staticmethod
    def _reshape(coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate and reshape flat 44-array → (22, 2)."""
        if coords.size != _TOTAL_FLOATS:
            raise ValueError(
                f"Expected {_TOTAL_FLOATS} floats, got {coords.size}"
            )
        return coords.reshape(_NUM_PLAYERS, _COORDS_PER_PLAYER)

    def _partition(
        self,
        points: NDArray[np.float64],
        team_ids: list[Literal["home", "away"]] | None,
        attacking_team: Literal["home", "away"],
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """
        Extract the *defending* team's point cloud.

        Returns
        -------
        cloud : ndarray, shape (N, 2)
            Positions of the defending players.
        global_indices : ndarray, shape (N,)
            Original 0–21 indices of those players (for output mapping).
        """
        if not self.config.defensive_only:
            return points, np.arange(_NUM_PLAYERS, dtype=np.intp)

        if team_ids is not None and len(team_ids) == _NUM_PLAYERS:
            # Dynamic partitioning — survives halftime, red cards, etc.
            defending_team = "away" if attacking_team == "home" else "home"
            mask = np.array(
                [tid == defending_team for tid in team_ids], dtype=bool,
            )
            global_indices = np.flatnonzero(mask).astype(np.intp)
        else:
            # Legacy fallback — hardcoded indices
            global_indices = np.arange(
                _DEFENSIVE_SLICE.start, _DEFENSIVE_SLICE.stop, dtype=np.intp,
            )

        cloud = points[global_indices]
        if cloud.shape[0] < 3:
            logger.warning(
                "Defending team has only %d players — topology requires ≥3",
                cloud.shape[0],
            )
        return cloud, global_indices

    def _build_filtration(
        self,
        cloud: NDArray[np.float64],
    ) -> gudhi.SimplexTree:
        """Construct Rips complex and return its simplex tree."""
        rips = gudhi.RipsComplex(
            points=cloud.tolist(),
            max_edge_length=self.config.max_edge_length,
        )
        stree: gudhi.SimplexTree = rips.create_simplex_tree(
            max_dimension=self.config.max_dimension,
        )
        stree.compute_persistence()
        return stree

    def _extract_h1(
        self,
        stree: gudhi.SimplexTree,
    ) -> list[tuple[float, float]]:
        """
        Pull H₁ persistence intervals that exceed the noise threshold.

        Returns list of (birth, death) pairs.  Infinite-death intervals
        (holes that never close) are discarded — they indicate the point
        cloud is too sparse to form a useful structure.
        """
        raw: list[tuple[int, tuple[float, float]]] = (
            stree.persistence_intervals_in_dimension(1)     # type: ignore[assignment]
        )
        threshold = self.config.persistence_threshold
        return [
            (b, d)
            for b, d in raw                                 # type: ignore[misc]
            if np.isfinite(d) and (d - b) >= threshold
        ]

    def _intervals_to_voids(
        self,
        stree: gudhi.SimplexTree,
        intervals: list[tuple[float, float]],
        cloud: NDArray[np.float64],
        global_indices: NDArray[np.intp],
        n_ghosts: int = 0,
    ) -> list[TopologicalVoid]:
        """
        For each surviving H₁ interval, locate the death triangle and
        compute its centroid.

        ``global_indices`` maps local cloud indices → original 0–21 player IDs.
        ``n_ghosts`` is the number of ghost boundary vertices appended
        to the cloud — death triangles composed entirely of ghosts
        are discarded.
        """
        # Pre-collect all 2-simplices (triangles) with their filtration values
        triangles: list[tuple[list[int], float]] = [
            (simplex, filt)
            for simplex, filt in stree.get_filtration()     # type: ignore[misc]
            if len(simplex) == 3
        ]

        if not triangles:
            return []

        tri_vertices = np.array(
            [s for s, _ in triangles], dtype=np.intp,
        )                                                    # (T, 3)
        tri_filts = np.array(
            [f for _, f in triangles], dtype=np.float64,
        )                                                    # (T,)

        n_real = len(cloud) - n_ghosts  # real-player count in the cloud

        voids: list[TopologicalVoid] = []

        for birth, death in intervals:
            void = self._death_triangle_centroid(
                death, tri_vertices, tri_filts, cloud,
                global_indices, n_real,
            )
            if void is not None:
                # Patch in the real birth/persistence from the interval
                void.birth = birth
                void.persistence = death - birth
                voids.append(void)

        return voids

    @staticmethod
    def _death_triangle_centroid(
        death_value: float,
        tri_vertices: NDArray[np.intp],
        tri_filts: NDArray[np.float64],
        cloud: NDArray[np.float64],
        global_indices: NDArray[np.intp],
        n_real: int,
    ) -> TopologicalVoid | None:
        """
        Locate the 2-simplex whose filtration value matches ``death_value``
        and return a ``TopologicalVoid`` with the centroid and global indices.

        Skips triangles composed entirely of ghost boundary vertices.
        """
        mask = np.isclose(tri_filts, death_value, atol=1e-12)
        candidates = np.flatnonzero(mask)

        if candidates.size == 0:
            idx = int(np.argmin(np.abs(tri_filts - death_value)))
        else:
            idx = int(candidates[0])

        local_v0, local_v1, local_v2 = tri_vertices[idx]

        # Skip if ALL vertices are ghost boundary points (not real players)
        if local_v0 >= n_real and local_v1 >= n_real and local_v2 >= n_real:
            return None

        centroid = cloud[[local_v0, local_v1, local_v2]].mean(axis=0)

        # Map local simplex-tree indices → original 0–21 player IDs
        # Ghost vertices (>= n_real) get mapped to -1 (sentinel)
        def _map(v: int) -> int:
            if v < len(global_indices):
                return int(global_indices[v])
            return -1  # ghost defender

        g0, g1, g2 = _map(local_v0), _map(local_v1), _map(local_v2)

        return TopologicalVoid(
            centroid_x=float(centroid[0]),
            centroid_y=float(centroid[1]),
            birth=0.0,        # placeholder — caller sets real value
            death=death_value,
            persistence=0.0,  # placeholder — caller sets real value
            death_triangle_indices=(g0, g1, g2),
        )

    def _inject_pitch_boundary(
        self,
        cloud: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int]:
        """
        Inject synthetic 'ghost defender' points along the four pitch
        edges to prevent gudhi from computing topological voids that
        extend into the stands.

        The pitch is normalised to [0, 1] × [0, 1].

        Parameters
        ----------
        cloud : ndarray, shape (N, 2)
            Real defensive point cloud.

        Returns
        -------
        augmented_cloud : ndarray, shape (N + n_ghosts, 2)
            Cloud with ghost defenders appended.
        n_ghosts : int
            Number of ghost points added.
        """
        # Fix 22: Dynamic density — spacing ≤ max_edge_length ensures
        # the boundary is "solid" relative to the Rips complex.
        # ceil(1.0 / max_edge_length) guarantees no gap wider than the
        # connectivity threshold, preventing voids from bleeding through.
        dynamic_d = int(np.ceil(1.0 / self.config.max_edge_length)) + 1
        d = max(self.config.pitch_boundary_density, dynamic_d)
        edge_pts = np.linspace(0.0, 1.0, d, endpoint=True)

        ghosts: list[NDArray[np.float64]] = []

        # Bottom edge (y=0)
        ghosts.append(np.column_stack([edge_pts, np.zeros(d)]))
        # Top edge (y=1)
        ghosts.append(np.column_stack([edge_pts, np.ones(d)]))
        # Left edge (x=0), exclude corners (already covered)
        inner = edge_pts[1:-1]
        if len(inner) > 0:
            ghosts.append(np.column_stack([np.zeros(len(inner)), inner]))
            # Right edge (x=1)
            ghosts.append(np.column_stack([np.ones(len(inner)), inner]))

        ghost_cloud = np.vstack(ghosts)  # (n_ghosts, 2)
        n_ghosts = ghost_cloud.shape[0]

        augmented = np.vstack([cloud, ghost_cloud])
        logger.debug(
            "Injected %d ghost defenders (dynamic_d=%d, static_d=%d, "
            "max_edge=%.3f)",
            n_ghosts, dynamic_d, self.config.pitch_boundary_density,
            self.config.max_edge_length,
        )
        return augmented, n_ghosts
