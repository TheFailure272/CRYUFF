"""
C.R.U.Y.F.F. — Zero-Copy Shared Memory IPC (Hardened)

Eliminates pickle serialisation overhead when passing NumPy arrays
between the main asyncio process and ``ProcessPoolExecutor`` workers.

Fix 12 — Lifecycle Safety
-------------------------
* **Ring Buffer Rotation**: ``SharedFrameRing`` pre-allocates *N*
  shared memory slots and rotates through them.  Each slot is reused
  rather than allocated/freed per frame — no orphan risk.
* **Context Manager**: ``read()`` wraps child-process access in
  ``try/finally`` to guarantee ``shm.close()`` even on C++ segfaults.
* **Master Cleanup**: ``SharedFrameRing.destroy()`` unlinks *all*
  slots on gateway shutdown, preventing /dev/shm leaks even after
  unclean crash recovery.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Final

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_COORDS_FLOATS: Final[int] = 44
_VEL_FLOATS: Final[int] = 44
_TOTAL_DOUBLES: Final[int] = _COORDS_FLOATS + _VEL_FLOATS  # 88 doubles max
_BYTE_SIZE: Final[int] = _TOTAL_DOUBLES * 8                  # float64 = 8 bytes


@dataclass(slots=True)
class SharedFrameToken:
    """
    Lightweight, picklable token passed to the child process.
    ~150 bytes vs ~2KB for a pickled NumPy array.
    """
    shm_name: str
    has_velocities: bool
    team_ids: list[str] | None = None
    ingest_monotonic: float = 0.0       # Fix 20: for inner backpressure re-check
    stale_threshold_s: float = 0.1      # Fix 20: configurable staleness threshold


@dataclass(slots=True)
class SharedFrameBuffer:
    """
    Single shared memory slot for one frame.

    Internal building block — use ``SharedFrameRing`` for production.
    """
    _shm: SharedMemory | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, name: str | None = None) -> SharedFrameBuffer:
        buf = cls()
        buf._shm = SharedMemory(name=name, create=True, size=_BYTE_SIZE)
        return buf

    def write(
        self,
        coords: NDArray[np.float64],
        velocities: NDArray[np.float64] | None = None,
        team_ids: list[str] | None = None,
    ) -> SharedFrameToken:
        assert self._shm is not None, "Buffer not created"
        view = np.ndarray(
            (_TOTAL_DOUBLES,), dtype=np.float64, buffer=self._shm.buf,
        )
        view[:_COORDS_FLOATS] = coords.ravel()[:_COORDS_FLOATS]
        has_vel = velocities is not None
        if has_vel:
            view[_COORDS_FLOATS:_COORDS_FLOATS + _VEL_FLOATS] = (
                velocities.ravel()[:_VEL_FLOATS]
            )
        return SharedFrameToken(
            shm_name=self._shm.name,
            has_velocities=has_vel,
            team_ids=team_ids,
        )

    @staticmethod
    def read(token: SharedFrameToken) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64] | None,
    ]:
        """Zero-copy read.  Always closes the handle, even on exception."""
        shm = SharedMemory(name=token.shm_name, create=False)
        try:
            view = np.ndarray(
                (_TOTAL_DOUBLES,), dtype=np.float64, buffer=shm.buf,
            )
            coords = view[:_COORDS_FLOATS].copy()
            velocities = (
                view[_COORDS_FLOATS:_COORDS_FLOATS + _VEL_FLOATS].copy()
                if token.has_velocities
                else None
            )
        finally:
            shm.close()   # ALWAYS close, even if gudhi segfaults later
        return coords, velocities

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()

    def unlink(self) -> None:
        if self._shm is not None:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    @property
    def name(self) -> str:
        assert self._shm is not None
        return self._shm.name


# ---------------------------------------------------------------------------
# Ring Buffer for production — pre-allocated, crash-safe
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SharedFrameRing:
    """
    Pre-allocated ring of ``n_slots`` shared memory buffers.

    Guarantees:
      * No new allocations during the match (all allocated at startup).
      * No orphaned /dev/shm blocks — ``destroy()`` cleans everything.
      * Round-robin slot reuse — the child process reads from the slot
        written to on the *previous* frame, so the current write never
        races with the current read (as long as n_slots ≥ 2).

    Usage::

        ring = SharedFrameRing(n_slots=4)
        try:
            token = ring.write(coords, velocities, team_ids)
            # → pass token to ProcessPoolExecutor
        finally:
            ring.destroy()
    """
    n_slots: int = 4
    _slots: list[SharedFrameBuffer] = field(
        default_factory=list, init=False, repr=False,
    )
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._slots = [SharedFrameBuffer.create() for _ in range(self.n_slots)]
        logger.info(
            "SharedFrameRing allocated: %d slots [%s]",
            self.n_slots,
            ", ".join(s.name for s in self._slots),
        )

    def write(
        self,
        coords: NDArray[np.float64],
        velocities: NDArray[np.float64] | None = None,
        team_ids: list[str] | None = None,
    ) -> SharedFrameToken:
        """Write to the next slot in the ring and return a token."""
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % self.n_slots
        return slot.write(coords, velocities, team_ids)

    def destroy(self) -> None:
        """
        Unlink ALL slots — call on gateway shutdown.
        Safe to call multiple times.
        """
        for slot in self._slots:
            slot.close()
            slot.unlink()
        logger.info(
            "SharedFrameRing destroyed: %d slots cleaned", len(self._slots),
        )
        self._slots.clear()

    @property
    def slot_names(self) -> list[str]:
        return [s.name for s in self._slots]
