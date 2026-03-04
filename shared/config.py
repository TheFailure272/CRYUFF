"""
C.R.U.Y.F.F. — Environment-driven configuration.

All tunables are loaded from env vars with sensible defaults for local dev.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Immutable, validated config loaded once at process start."""

    model_config = {"env_prefix": "CRUYFF_"}

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_channel_raw: str = "cruyff:tracking:raw"
    redis_channel_result: str = "cruyff:analysis:result"

    # ── WebSocket gateway ──────────────────────────────────────────────────
    ws_tick_hz: int = 25
    ws_host: str = "0.0.0.0"
    ws_port: int = 8000

    # ── Topological Solver (Layer 1) ───────────────────────────────────────
    topo_max_edge_length: float = 0.35
    topo_max_dimension: int = 2          # need dim=2 simplices to detect H₁
    topo_persistence_threshold: float = 0.04
    topo_use_defensive_only: bool = True  # restrict to opponent team
    topo_dynamic_team_id: bool = True    # use team_ids from payload if available
    topo_velocity_horizon_secs: float = 0.5  # cap: linear projection unreliable beyond ~0.5s

    # ── Temporal Smoothing ─────────────────────────────────────────────────
    topo_ema_alpha: float = 0.3          # EMA blend factor (0=full history, 1=instant)
    topo_stability_threshold: float = 0.6  # min stability to broadcast a void
    topo_match_radius: float = 0.05      # max centroid drift to consider same void

    # ── BioKinetic Engine (Layer 2) ────────────────────────────────────────
    bio_scan_window_secs: float = 5.0
    bio_micro_var_threshold: float = 0.015
    bio_cognitive_collapse_threshold: float = 0.6

    # ── Ghost Trajectory Engine (Layer 3) ──────────────────────────────────
    ghost_diffusion_steps: int = 20
    ghost_xt_grid_resolution: tuple[int, int] = (104, 68)

    # ── Worker ─────────────────────────────────────────────────────────────
    worker_thread_pool_size: int = 4
    worker_stale_frame_ms: float = 100.0  # drop frames older than this (ms)


settings = Settings()
