"""
C.R.U.Y.F.F. — Transport Hysteresis Protocol (Fix 19)

Prevents out-of-order frame delivery and UI stuttering when the WebRTC
channel flaps (rapidly connects/disconnects) due to restrictive stadium
firewalls blocking UDP.

Protocol
--------
1. **Primary**: WebRTC data channel (UDP, unreliable, unordered).
2. **Fallback**: WebSocket (TCP, reliable, ordered).
3. **Hysteresis**: Once fallen back to WebSocket, remain there for
   ``cooldown_s`` (default 30s) before attempting WebRTC again.
4. **Frame Deduplication**: All frames carry an ``ingest_monotonic``
   timestamp. The frontend MUST discard frames where
   ``ingest_monotonic <= last_rendered_monotonic``, regardless of
   which transport delivered them.

This module defines the ``TransportHysteresis`` state machine for
use by the frontend client. It is also importable by the Gateway
to embed the protocol config in the ``reconnect_policy`` message.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TransportMode(Enum):
    """Active transport layer."""
    WEBRTC = auto()
    WEBSOCKET = auto()


@dataclass(slots=True)
class TransportHysteresis:
    """
    State machine for WebRTC ↔ WebSocket fallback with hysteresis.

    The frontend should call:
      - ``on_webrtc_open()`` when the data channel opens.
      - ``on_webrtc_close()`` when the data channel closes.
      - ``should_use_webrtc()`` on each frame to decide which transport
        to listen on.
      - ``should_accept_frame(mono)`` on each incoming frame to dedup.

    Parameters
    ----------
    cooldown_s : float
        Minimum seconds to remain on WebSocket before re-attempting
        WebRTC. Prevents flap-induced switching.
    max_flaps_before_lockout : int
        If WebRTC drops more than this many times in ``lockout_window_s``,
        permanently lock to WebSocket for the rest of the match.
    lockout_window_s : float
        Window for counting flaps.
    """
    cooldown_s: float = 30.0
    max_flaps_before_lockout: int = 5
    lockout_window_s: float = 120.0

    _mode: TransportMode = field(default=TransportMode.WEBSOCKET, init=False)
    _last_fallback_time: float = field(default=0.0, init=False)
    _flap_times: list[float] = field(default_factory=list, init=False)
    _locked_out: bool = field(default=False, init=False)
    _last_rendered_monotonic: float = field(default=0.0, init=False)

    @property
    def mode(self) -> TransportMode:
        return self._mode

    @property
    def is_locked_out(self) -> bool:
        return self._locked_out

    def on_webrtc_open(self) -> None:
        """Called when the WebRTC data channel opens."""
        if self._locked_out:
            logger.info("WebRTC opened but locked out — ignoring")
            return

        now = time.monotonic()

        # Hysteresis: don't switch back if still in cooldown
        if (
            self._mode == TransportMode.WEBSOCKET
            and now - self._last_fallback_time < self.cooldown_s
        ):
            logger.info(
                "WebRTC opened but in cooldown (%.0fs remaining) — staying on WS",
                self.cooldown_s - (now - self._last_fallback_time),
            )
            return

        self._mode = TransportMode.WEBRTC
        logger.info("Transport → WebRTC")

    def on_webrtc_close(self) -> None:
        """Called when the WebRTC data channel closes or errors."""
        if self._mode == TransportMode.WEBRTC:
            now = time.monotonic()
            self._mode = TransportMode.WEBSOCKET
            self._last_fallback_time = now

            # Track flaps for lockout detection
            self._flap_times.append(now)
            # Prune old flaps outside the window
            self._flap_times = [
                t for t in self._flap_times
                if now - t <= self.lockout_window_s
            ]

            if len(self._flap_times) >= self.max_flaps_before_lockout:
                self._locked_out = True
                logger.warning(
                    "WebRTC flapped %d times in %.0fs — LOCKED OUT to WebSocket",
                    len(self._flap_times),
                    self.lockout_window_s,
                )
            else:
                logger.info(
                    "Transport → WebSocket (cooldown %.0fs, flaps: %d/%d)",
                    self.cooldown_s,
                    len(self._flap_times),
                    self.max_flaps_before_lockout,
                )

    def should_use_webrtc(self) -> bool:
        """Whether the frontend should listen on the WebRTC channel."""
        if self._locked_out:
            return False
        return self._mode == TransportMode.WEBRTC

    def should_accept_frame(self, ingest_monotonic: float) -> bool:
        """
        Frame deduplication gate.

        Discards frames with a timestamp <= the last rendered frame,
        regardless of which transport delivered them. This handles
        the inherent unordered nature of UDP and any overlap during
        fallback transitions.

        Parameters
        ----------
        ingest_monotonic : float
            The ``ingest_monotonic`` field from the analysis result.

        Returns
        -------
        bool
            True if this frame should be rendered.
        """
        if ingest_monotonic <= self._last_rendered_monotonic:
            return False  # stale or duplicate — discard
        self._last_rendered_monotonic = ingest_monotonic
        return True

    def get_protocol_config(self) -> dict:
        """
        Returns the protocol config for embedding in the gateway's
        ``reconnect_policy`` message. The frontend uses this to
        configure its own TransportHysteresis instance.
        """
        return {
            "type": "transport_hysteresis",
            "cooldown_s": self.cooldown_s,
            "max_flaps_before_lockout": self.max_flaps_before_lockout,
            "lockout_window_s": self.lockout_window_s,
            "mode": self._mode.name,
            "locked_out": self._locked_out,
        }
