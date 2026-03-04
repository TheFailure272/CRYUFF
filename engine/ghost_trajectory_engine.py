"""
C.R.U.Y.F.F. — Layer 3: "The Ghost"

GhostTrajectoryEngine  (SKELETON)
==================================
Generates counterfactual "ghost player" trajectories using a conditional
Trajectory Diffusion Model, then scores each sampled path against an
Expected Threat (xT) surface to find the optimal run.

    τ* = argmax_τ  𝔼[xT(τ | S_t)]

This module is a structured stub.  The diffusion model checkpoint and
the pre-computed xT grid will be plugged in during Phase 3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from shared.schemas import GhostTrajectory, PlayerPosition

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GhostTrajectoryEngine:
    """
    Stateless ghost-run generator.

    Parameters
    ----------
    diffusion_model_path : str
        Path to the pre-trained trajectory diffusion checkpoint (safetensors).
    xt_grid_path : str
        Path to the pre-computed Expected Threat grid (.npy).
    num_inference_steps : int
        DDPM / DDIM denoising steps.  Lower = faster but noisier.
    num_trajectory_samples : int
        Number of candidate trajectories to sample per call.
    trajectory_horizon : int
        Number of future timesteps to predict (at 25 Hz).
    """
    diffusion_model_path: str = ""
    xt_grid_path: str = ""
    num_inference_steps: int = 20
    num_trajectory_samples: int = 8
    trajectory_horizon: int = 50        # 2 seconds @ 25 Hz

    _xt_grid: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False,
    )
    _warmed_up: bool = field(default=False, init=False, repr=False)
    # _diffusion_pipeline — lazy-loaded HuggingFace diffusers pipeline

    # ── Fix 18: CUDA Warm-up ──────────────────────────────────────────────

    async def warm_up(self) -> None:
        """
        Run a single dummy inference to pre-compile CUDA graphs.

        Must be called during system boot (before kickoff).  Without
        this, the first real Ghost request suffers 3-5s cold-start
        latency as PyTorch/CUDA compiles kernels on-the-fly.

        The warm-up uses a synthetic 22-player state and a dummy
        player_id.  The result is discarded — only the compilation
        side-effects matter.
        """
        if self._warmed_up:
            logger.debug("GhostEngine already warmed up — skipping")
            return

        logger.info("GhostEngine warm-up: running dummy inference…")
        synthetic_state = np.random.default_rng(0).random(
            (22, 2), dtype=np.float64,
        )
        try:
            _ = await self.generate_ghost(synthetic_state, player_id=0)
            self._warmed_up = True
            logger.info("GhostEngine warm-up complete — CUDA graphs compiled")
        except Exception:
            logger.exception(
                "GhostEngine warm-up failed — first real request may be slow"
            )

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    # ── public API ─────────────────────────────────────────────────────────

    async def generate_ghost(
        self,
        state: NDArray[np.float64],
        player_id: int,
    ) -> GhostTrajectory:
        """
        Generate the optimal counterfactual run for ``player_id``.

        STUB — returns a straight-line placeholder trajectory.

        Algorithm (when implemented):
            1. Encode current pitch state ``S_t`` (22×2 positions) as
               the conditioning vector.
            2. Sample ``num_trajectory_samples`` candidate trajectories
               from the diffusion model:
                   τ_i ~ p_θ(τ | S_t, player_id)
            3. Score each trajectory against the xT grid:
                   score_i = Σ_t xT(τ_i(t))
            4. Select τ* = argmax score_i.
            5. Return as GhostTrajectory with waypoints + expected_xt.

        Parameters
        ----------
        state : ndarray, shape (22, 2)
            Current positions of all 22 players (normalised pitch coords).
        player_id : int
            Index of the player to generate the ghost run for.
        """
        # TODO: Replace with actual diffusion inference
        #   pipeline = self._get_or_load_pipeline()
        #   conditioning = self._encode_state(state, player_id)
        #   samples = pipeline(
        #       conditioning,
        #       num_inference_steps=self.num_inference_steps,
        #       batch_size=self.num_trajectory_samples,
        #   )
        #   trajectories = samples.reshape(self.num_trajectory_samples, self.trajectory_horizon, 2)
        #   scores = np.array([self._score_trajectory(t) for t in trajectories])
        #   best_idx = int(np.argmax(scores))
        #   best_traj = trajectories[best_idx]
        #   best_score = float(scores[best_idx])

        # ── Placeholder: straight line from current position forward ──
        start = state[player_id]
        end = np.array([start[0] + 0.15, start[1]], dtype=np.float64)
        waypoints = [
            PlayerPosition(
                x=float(start[0] + t * (end[0] - start[0])),
                y=float(start[1] + t * (end[1] - start[1])),
            )
            for t in np.linspace(0.0, 1.0, self.trajectory_horizon)
        ]

        return GhostTrajectory(waypoints=waypoints, expected_xt=0.0)

    # ── private helpers ────────────────────────────────────────────────────

    def _load_xt_grid(self) -> NDArray[np.float64]:
        """
        Load the pre-computed Expected Threat surface.

        The xT grid is a 2D array of shape ``(pitch_length, pitch_width)``
        where each cell holds the probability of a goal being scored if
        the ball is in that zone.  Typically derived from ~500k historical
        possessions.

        Returns
        -------
        ndarray, shape (104, 68)
            xT surface in normalised pitch coordinates.
        """
        if self._xt_grid is not None:
            return self._xt_grid

        path = Path(self.xt_grid_path)
        if path.exists():
            self._xt_grid = np.load(path).astype(np.float64)
            logger.info("Loaded xT grid from %s — shape %s", path, self._xt_grid.shape)
        else:
            # Fallback: uniform grid (no xT intelligence)
            logger.warning("xT grid not found at %s — using uniform fallback", path)
            self._xt_grid = np.ones((104, 68), dtype=np.float64) * 0.01

        return self._xt_grid

    def _score_trajectory(
        self,
        trajectory: NDArray[np.float64],
    ) -> float:
        """
        Score a single trajectory against the xT surface.

        Parameters
        ----------
        trajectory : ndarray, shape (T, 2)
            Sequence of (x, y) normalised pitch positions.

        Returns
        -------
        float
            Cumulative xT gained along the path.

        Algorithm:
            1. Map normalised coords → grid indices.
            2. Look up xT value at each waypoint.
            3. Sum the *deltas* (only count xT *gained*, not absolute).
        """
        xt_grid = self._load_xt_grid()
        grid_h, grid_w = xt_grid.shape

        # Clamp and discretise
        xs = np.clip(trajectory[:, 0] * grid_h, 0, grid_h - 1).astype(int)
        ys = np.clip(trajectory[:, 1] * grid_w, 0, grid_w - 1).astype(int)

        xt_values = xt_grid[xs, ys]
        xt_deltas = np.diff(xt_values)

        # Only count positive xT movement (advancing threat)
        return float(np.sum(np.maximum(xt_deltas, 0.0)))

    def _encode_state(
        self,
        state: NDArray[np.float64],
        player_id: int,
    ) -> NDArray[np.float64]:
        """
        Encode the full pitch state into the conditioning vector for the
        diffusion model.

        STUB — exact encoding depends on model architecture.

        Typical encoding:
            - Flatten all 22 positions → 44 floats
            - One-hot encode ``player_id`` → 22 floats
            - Concatenate → 66-dim conditioning vector
        """
        flat_positions = state.flatten()                     # (44,)
        one_hot = np.zeros(22, dtype=np.float64)
        one_hot[player_id] = 1.0
        return np.concatenate([flat_positions, one_hot])     # (66,)
