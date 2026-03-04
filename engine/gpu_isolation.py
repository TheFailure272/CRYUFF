"""
C.R.U.Y.F.F. — GPU Resource Isolation (Fix 21)

Prevents the on-demand GhostEngine (diffusion model) from starving the
25Hz BioKineticEngine (YOLOv8) of GPU compute and VRAM.

Strategy
--------
1. **CUDA Stream Pinning**: Each engine gets its own ``torch.cuda.Stream``.
   Operations on different streams execute concurrently without blocking
   each other's kernels.

2. **Memory Fraction Capping**: The GhostEngine's CUDA context is limited
   to a fraction of total VRAM (default 30%), leaving the majority for
   the real-time pipeline.

3. **Device Assignment**: In multi-GPU environments, engines can be
   pinned to different devices entirely.

Usage::

    from engine.gpu_isolation import GPUConfig, isolate_ghost_engine

    # During worker init:
    isolate_ghost_engine(GPUConfig(
        ghost_device="cuda:1",       # separate GPU if available
        ghost_memory_fraction=0.3,   # cap at 30% VRAM
        realtime_device="cuda:0",
    ))

.. note::

    Gracefully degrades on CPU-only systems.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class GPUConfig:
    """
    GPU resource allocation for C.R.U.Y.F.F. engines.

    Parameters
    ----------
    realtime_device : str
        CUDA device for the 25Hz pipeline (topology + bio-kinetics).
    ghost_device : str
        CUDA device for the GhostEngine diffusion model.
        Set to same device as ``realtime_device`` if only one GPU.
    ghost_memory_fraction : float
        Maximum fraction of VRAM the GhostEngine can use (0.0-1.0).
        Only applies when ghost_device == realtime_device (shared GPU).
    enable_cudnn_benchmark : bool
        Enable cuDNN auto-tuner for consistent kernel selection.
    """
    realtime_device: str = "cuda:0"
    ghost_device: str = "cuda:0"
    ghost_memory_fraction: float = 0.3
    enable_cudnn_benchmark: bool = True


def isolate_ghost_engine(config: GPUConfig | None = None) -> None:
    """
    Apply GPU isolation rules.

    Must be called ONCE during worker initialization, before any
    engine loads models.

    - If two GPUs are available and configured, Ghost runs on a
      separate device — complete isolation.
    - If sharing a single GPU, Ghost is capped at ``ghost_memory_fraction``
      of VRAM. CUDA streams provide kernel-level concurrency.
    """
    if not _TORCH_AVAILABLE:
        logger.info("GPU isolation: PyTorch not available — running on CPU")
        return

    if not torch.cuda.is_available():
        logger.info("GPU isolation: CUDA not available — running on CPU")
        return

    cfg = config or GPUConfig()

    # cuDNN benchmark — stabilises kernel selection for repeating input sizes
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    num_gpus = torch.cuda.device_count()
    logger.info("GPU isolation: %d device(s) detected", num_gpus)

    if cfg.realtime_device != cfg.ghost_device:
        # Multi-GPU: complete device isolation
        logger.info(
            "GPU isolation: realtime → %s, ghost → %s (full isolation)",
            cfg.realtime_device, cfg.ghost_device,
        )
    else:
        # Single GPU: memory fraction capping
        torch.cuda.set_per_process_memory_fraction(
            cfg.ghost_memory_fraction,
            device=cfg.ghost_device,
        )
        logger.info(
            "GPU isolation: shared device %s — ghost capped at %.0f%% VRAM",
            cfg.ghost_device,
            cfg.ghost_memory_fraction * 100,
        )

    # Create dedicated CUDA streams for each engine
    realtime_stream = torch.cuda.Stream(device=cfg.realtime_device)
    ghost_stream = torch.cuda.Stream(device=cfg.ghost_device)

    logger.info(
        "GPU isolation: streams created — realtime=%s, ghost=%s",
        realtime_stream, ghost_stream,
    )

    # Store streams globally for engines to use
    _streams["realtime"] = realtime_stream
    _streams["ghost"] = ghost_stream


# Global stream registry — engines look up their assigned stream
_streams: dict[str, object] = {}


def get_stream(name: str) -> object | None:
    """Retrieve the CUDA stream assigned to the named engine."""
    return _streams.get(name)


def get_device(name: str, config: GPUConfig | None = None) -> str:
    """Retrieve the CUDA device assigned to the named engine."""
    cfg = config or GPUConfig()
    if name == "ghost":
        return cfg.ghost_device
    return cfg.realtime_device
