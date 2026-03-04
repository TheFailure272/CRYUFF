"""
C.R.U.Y.F.F. — WebRTC Data Channel Transport (Fix 17)

Solves TCP Head-of-Line blocking for the Tactical Glass.

Problem
-------
WebSockets run over TCP.  At 25Hz, if the stadium Wi-Fi drops a single
packet, TCP halts delivery of ALL subsequent frames until the lost
packet is retransmitted.  The manager's tablet freezes for 200-500ms,
then fast-forwards through 10 stale frames.

Solution
--------
WebRTC Data Channels use SCTP-over-DTLS-over-UDP with configurable
reliability.  By opening an *unreliable, unordered* data channel,
lost frames are simply skipped — the frontend always renders the
newest frame.

Architecture
~~~~~~~~~~~~
* ``WebRTCTransport`` wraps ``aiortc`` to establish a peer connection
  with each Tactical Glass tablet.
* The gateway offers WebRTC signalling via a REST endpoint
  (``/webrtc/offer``).
* Once the data channel is open, ``broadcast()`` sends the latest
  ``AnalysisResult`` as JSON.  Lost packets vanish — no retransmit,
  no freeze.

Fallback
~~~~~~~~
If ``aiortc`` is not installed (e.g. in dev/test), the module gracefully
degrades.  The gateway can still serve over plain WebSockets.

Dependencies
~~~~~~~~~~~~
* ``aiortc`` — Python WebRTC implementation.
  Install: ``pip install aiortc``

.. note::

   This module is a structured integration layer.  The actual
   ``aiortc`` dependency is lazy-imported to keep the core
   deployable without it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports — graceful degradation if aiortc not installed
_AIORTC_AVAILABLE = False
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore[import-untyped]
    _AIORTC_AVAILABLE = True
except ImportError:
    pass


@dataclass(slots=True)
class WebRTCTransport:
    """
    Manages WebRTC peer connections for real-time data channel delivery.

    Usage (in the gateway lifespan)::

        transport = WebRTCTransport()

        # On signalling: client sends an SDP offer
        answer_sdp = await transport.handle_offer(offer_sdp)
        # → return answer_sdp to the client via REST

        # On each analysis result:
        await transport.broadcast(result_dict)

        # On shutdown:
        await transport.close_all()
    """

    _peers: list[Any] = field(default_factory=list, init=False, repr=False)
    _channels: list[Any] = field(default_factory=list, init=False, repr=False)

    @staticmethod
    def available() -> bool:
        """Whether aiortc is installed and WebRTC is usable."""
        return _AIORTC_AVAILABLE

    async def handle_offer(self, offer_sdp: str, offer_type: str = "offer") -> str:
        """
        Process a client's SDP offer and return the server's SDP answer.

        The data channel is opened with::

            ordered=False, maxRetransmits=0

        This makes it *unreliable and unordered* — perfect for 25Hz
        real-time telemetry where only the newest frame matters.

        Parameters
        ----------
        offer_sdp : str
            Client's SDP offer string.
        offer_type : str
            SDP type (default "offer").

        Returns
        -------
        str
            Server's SDP answer string (JSON with type + sdp).

        Raises
        ------
        RuntimeError
            If aiortc is not installed.
        """
        if not _AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc not installed — cannot use WebRTC transport. "
                "Install with: pip install aiortc"
            )

        pc = RTCPeerConnection(configuration={
            "iceServers": [
                # STUN: free, stateless, works on open networks
                {"urls": ["stun:stun.l.google.com:19302"]},
                # TURN (Fix F20): relay for stadium symmetric NATs / UDP firewalls.
                # Deploy Coturn alongside FastAPI. TLS on port 443 masquerades
                # as HTTPS, bypassing even the strictest corporate firewalls.
                # In production, replace with your deployed Coturn endpoint.
                {
                    "urls": [
                        "turn:turn.cruyff.local:443?transport=tcp",
                        "turns:turn.cruyff.local:443?transport=tcp",
                    ],
                    "username": "cruyff",
                    "credential": "tactical-glass",
                },
            ],
        })
        self._peers.append(pc)

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            self._channels.append(channel)
            logger.info(
                "WebRTC data channel opened: %s (ordered=%s)",
                channel.label,
                channel.ordered,
            )

        # Set remote description (client's offer)
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)

        # Create the server-side data channel (unreliable, unordered)
        channel = pc.createDataChannel(
            "cruyff-analysis",
            ordered=False,
            maxRetransmits=0,  # fire-and-forget — no TCP-style retransmission
        )
        self._channels.append(channel)

        # Create and set local description (server's answer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return json.dumps({
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp,
        })

    async def broadcast(self, data: dict[str, Any]) -> None:
        """
        Send data to all connected WebRTC data channels.

        Lost packets are NOT retransmitted — the frontend always gets
        the newest frame or nothing (no freeze, no fast-forward).
        """
        payload = json.dumps(data, separators=(",", ":"))
        stale: list[int] = []

        for i, channel in enumerate(self._channels):
            try:
                if channel.readyState == "open":
                    channel.send(payload)
            except Exception:  # noqa: BLE001
                stale.append(i)

        for i in reversed(stale):
            self._channels.pop(i)

    async def close_all(self) -> None:
        """Gracefully close all peer connections."""
        for pc in self._peers:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass
        self._peers.clear()
        self._channels.clear()
        logger.info("WebRTC transport: all peers closed")

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def channel_count(self) -> int:
        return sum(
            1 for ch in self._channels
            if hasattr(ch, 'readyState') and ch.readyState == "open"
        )
