"""
C.R.U.Y.F.F. — Async Worker Node (Final Hardening)

Production-grade worker with all 15 fixes integrated.

* **Zero-copy IPC** via ``SharedFrameRing`` (Fix 8, 12)
* **Monotonic backpressure** via ``ingest_monotonic`` (Fix 13)
* **Kalman-smoothed topology** with coasting (Fix 9, 14)
* **Decoupled Ghost RPC** (Fix 10)
* **ProcessPoolExecutor** for GIL-free gudhi (Fix 7)
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from engine.biokinetic_engine import BioKineticEngine
from engine.ghost_trajectory_engine import GhostTrajectoryEngine
from engine.temporal_smoother import TemporalSmoother
from engine.topological_solver import SolverConfig, TopologicalSolver
from server.redis_bus import RedisBus
from shared.config import settings
from shared.schemas import AnalysisResult, GhostTrajectory, TrackingFrame
from shared.shm_buffer import SharedFrameBuffer, SharedFrameRing, SharedFrameToken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level picklable dispatch (ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _solve_topology_shm(
    solver: TopologicalSolver,
    token: SharedFrameToken,
) -> list:
    """
    Child-process entry point.  Reads from SharedMemory (zero-copy),
    computes topology, returns voids.

    Fix 20: Re-checks staleness AFTER leaving the ProcessPool queue.
    If the frame sat in the queue too long, returns empty immediately
    to unblock the process for fresher frames.
    """
    # Inner backpressure gate — the ProcessPool queue may have added latency
    if token.ingest_monotonic > 0:
        queue_age = time.monotonic() - token.ingest_monotonic
        if queue_age > token.stale_threshold_s:
            return []  # stale after queuing — skip expensive computation

    coords, velocities = SharedFrameBuffer.read(token)
    return solver.solve(
        coords,
        team_ids=token.team_ids,
        velocities=velocities,
    )


class AnalysisWorker:
    """
    Production worker — Flow + Pulse at 25Hz, Ghost on-demand.
    """

    __slots__ = (
        "_bus",
        "_executor",
        "_topo_solver",
        "_temporal_smoother",
        "_bio_engine",
        "_ghost_engine",
        "_shutdown_event",
        "_stale_threshold_s",
        "_frames_processed",
        "_frames_dropped",
        "_shm_ring",
        "_latest_state",
    )

    def __init__(self) -> None:
        self._bus = RedisBus()
        self._executor = ProcessPoolExecutor(
            max_workers=settings.worker_thread_pool_size,
        )

        self._topo_solver = TopologicalSolver(
            config=SolverConfig(
                max_edge_length=settings.topo_max_edge_length,
                max_dimension=settings.topo_max_dimension,
                persistence_threshold=settings.topo_persistence_threshold,
                defensive_only=settings.topo_use_defensive_only,
                velocity_horizon_secs=settings.topo_velocity_horizon_secs,
            )
        )
        self._temporal_smoother = TemporalSmoother(
            stability_threshold=settings.topo_stability_threshold,
            match_radius=settings.topo_match_radius,
        )
        self._bio_engine = BioKineticEngine(
            scan_window_secs=settings.bio_scan_window_secs,
            micro_var_threshold=settings.bio_micro_var_threshold,
            cognitive_collapse_threshold=settings.bio_cognitive_collapse_threshold,
        )
        self._ghost_engine = GhostTrajectoryEngine(
            num_inference_steps=settings.ghost_diffusion_steps,
        )

        self._shutdown_event = asyncio.Event()
        self._stale_threshold_s = settings.worker_stale_frame_ms / 1000.0
        self._frames_processed = 0
        self._frames_dropped = 0

        # Pre-allocated ring buffer — no per-frame SHM allocation (Fix 12)
        self._shm_ring = SharedFrameRing(n_slots=4)

        # Cache latest state for on-demand Ghost requests
        self._latest_state: np.ndarray | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main entry point — run until SIGINT / SIGTERM."""
        await self._bus.connect()

        # Fix 18: Warm up Ghost engine (compile CUDA graphs before kickoff)
        await self._ghost_engine.warm_up()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass  # Windows

        logger.info(
            "Worker online — Flow+Pulse @ 25Hz, Ghost on-demand  "
            "(stale: %.0fms, ring: %d slots, ghost_warm=%s)",
            settings.worker_stale_frame_ms,
            self._shm_ring.n_slots,
            self._ghost_engine.is_warmed_up,
        )

        tracking_task = asyncio.create_task(self._consume_loop())
        ghost_task = asyncio.create_task(self._ghost_rpc_loop())

        await self._shutdown_event.wait()

        tracking_task.cancel()
        ghost_task.cancel()
        self._executor.shutdown(wait=False)
        self._shm_ring.destroy()          # Master cleanup — wipe ALL SHM (Fix 12)
        await self._bus.shutdown()
        logger.info(
            "Worker shut down — processed: %d, dropped: %d",
            self._frames_processed,
            self._frames_dropped,
        )

    # ── 25Hz tracking loop ─────────────────────────────────────────────────

    async def _consume_loop(self) -> None:
        try:
            async for raw_payload in self._bus.subscribe(settings.redis_channel_raw):
                try:
                    frame = TrackingFrame.model_validate(raw_payload)

                    # ── Monotonic backpressure (Fix 13) ────────────────
                    # Uses ingest_monotonic tagged by the gateway — NTP-immune.
                    # Falls back to absolute timestamp if not tagged.
                    now_mono = time.monotonic()
                    if frame.ingest_monotonic is not None:
                        age = now_mono - frame.ingest_monotonic
                    else:
                        # Fallback: absolute time (NTP-vulnerable)
                        age = time.time() - frame.timestamp

                    if age > self._stale_threshold_s:
                        self._frames_dropped += 1
                        if self._frames_dropped % 100 == 1:
                            logger.warning(
                                "Dropped stale frame (age=%.1fms) [total: %d]",
                                age * 1000,
                                self._frames_dropped,
                            )
                        continue

                    result = await self._process_frame(frame)
                    self._frames_processed += 1
                    await self._bus.publish(
                        settings.redis_channel_result,
                        result.model_dump(),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Frame processing failed")
        except asyncio.CancelledError:
            logger.info("Tracking loop cancelled")

    async def _process_frame(self, frame: TrackingFrame) -> AnalysisResult:
        """Flow + Pulse only.  Ghost is decoupled (Fix 10)."""
        loop = asyncio.get_running_loop()
        coords = np.asarray(frame.coordinates, dtype=np.float64)
        velocities = (
            np.asarray(frame.velocities, dtype=np.float64)
            if frame.velocities is not None
            else None
        )

        self._latest_state = coords.copy()

        # Write to ring buffer — zero-copy IPC (Fix 8 + 12)
        token = self._shm_ring.write(coords, velocities, frame.team_ids)

        # Fix 20: Embed monotonic timestamp for inner backpressure
        token.ingest_monotonic = (
            frame.ingest_monotonic if frame.ingest_monotonic is not None
            else time.monotonic()
        )
        token.stale_threshold_s = self._stale_threshold_s

        # Layer 1: Topology → process pool (GIL-free)
        topo_future = loop.run_in_executor(
            self._executor,
            _solve_topology_shm,
            self._topo_solver,
            token,
        )

        # Layer 2: BioKinetics → async
        bio_future = self._bio_engine.analyze(frame)

        topo_result, bio_result = await asyncio.gather(
            topo_future,
            bio_future,
            return_exceptions=True,
        )

        # Kalman-smoothed topology with coasting (Fix 9 + 14)
        raw_voids = topo_result if isinstance(topo_result, list) else []
        stable_voids = self._temporal_smoother.smooth(raw_voids)

        return AnalysisResult(
            timestamp=frame.timestamp,
            voids=stable_voids,
            bio_alerts=bio_result if isinstance(bio_result, list) else [],
            ghost=None,
        )

    # ── On-demand Ghost RPC (Fix 10) ───────────────────────────────────────

    async def _ghost_rpc_loop(self) -> None:
        """
        Ghost runs on-demand — triggered by manager's "Optimize" button.
        100-200ms latency is acceptable for an explicit user action.
        """
        try:
            async for payload in self._bus.subscribe("cruyff:ghost:request"):
                try:
                    player_id = int(payload.get("player_id", 10))
                    if self._latest_state is None:
                        logger.warning("Ghost request but no state available")
                        continue

                    state = self._latest_state.reshape(22, 2)
                    ghost: GhostTrajectory = await self._ghost_engine.generate_ghost(
                        state, player_id=player_id,
                    )
                    await self._bus.publish(
                        "cruyff:ghost:result",
                        ghost.model_dump(),
                    )
                    logger.info("Ghost for player %d (xT=%.3f)", player_id, ghost.expected_xt)
                except Exception:  # noqa: BLE001
                    logger.exception("Ghost RPC failed")
        except asyncio.CancelledError:
            logger.info("Ghost RPC loop cancelled")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    )
    worker = AnalysisWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
