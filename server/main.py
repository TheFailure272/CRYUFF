"""
C.R.U.Y.F.F. — FastAPI WebSocket Gateway (Final Hardening)

Fixes integrated:
* **Fix 13**: Tags ``ingest_monotonic`` on every incoming frame.
* **Fix 16**: Reconnection rate limiter with exponential backoff
  protocol — prevents thundering herd when stadium Wi-Fi drops.

Phase 4 Routes:
* ``/ws/voice``          — Voice NLP (F36/F40/F46)
* ``POST /api/push``     — Dressing Room Push (F38)
* ``/ws/dressing-room``  — Apple TV live feed (F38)
* ``/ws/omnicam``        — Volumetric Omni-Cam (F39/F43/F47)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.redis_bus import RedisBus
from server.webrtc_transport import WebRTCTransport
from shared.config import settings
from shared.schemas import TrackingFrame

# Phase 4 engines
from engine.voice_engine import VoiceEngine
from engine.clip_engine import ClipEngine
from server.push_service import PushService
from server.dressing_room_ws import DressingRoomWS
from engine.volumetric import VolumetricEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fix 16: Reconnection Rate Limiter
# ---------------------------------------------------------------------------

class ReconnectionGuard:
    """
    Prevents "thundering herd" when multiple tablets reconnect
    simultaneously after a Wi-Fi dropout.

    On each new WebSocket connection, checks the client's IP against
    a sliding window.  If the same IP connects twice within
    ``min_reconnect_interval_s``, the second attempt is delayed by an
    exponentially increasing amount (up to ``max_backoff_s``).

    The gateway also sends the client a ``reconnect_policy`` message
    with recommended backoff parameters so the frontend can implement
    jittered exponential backoff on its side.
    """

    __slots__ = ("_last_connect", "_attempt_count", "_min_interval", "_max_backoff")

    def __init__(
        self,
        min_reconnect_interval: float = 2.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._last_connect: dict[str, float] = defaultdict(float)
        self._attempt_count: dict[str, int] = defaultdict(int)
        self._min_interval = min_reconnect_interval
        self._max_backoff = max_backoff

    async def gate(self, ws: WebSocket) -> bool:
        """
        Check if this connection should be throttled.

        Returns True if the connection is allowed, False if rejected.
        Sends the client a reconnect_policy message in both cases.
        """
        client_ip = ws.client.host if ws.client else "unknown"
        now = time.monotonic()
        elapsed = now - self._last_connect[client_ip]

        if elapsed < self._min_interval:
            # Too fast — exponential backoff
            self._attempt_count[client_ip] += 1
            backoff = min(
                2 ** self._attempt_count[client_ip] + _jitter(),
                self._max_backoff,
            )
            logger.warning(
                "Reconnection throttled for %s (attempt %d, backoff %.1fs)",
                client_ip, self._attempt_count[client_ip], backoff,
            )
            await ws.accept()
            await ws.send_json({
                "type": "reconnect_policy",
                "action": "backoff",
                "retry_after_s": backoff,
                "base_s": 1.0,
                "max_s": self._max_backoff,
                "jitter": True,
            })
            await ws.close(code=1013, reason="Try again later (backoff)")
            return False

        # Allowed
        self._last_connect[client_ip] = now
        self._attempt_count[client_ip] = 0
        return True


def _jitter() -> float:
    """Uniform jitter [0, 1) to spread reconnections."""
    import random
    return random.random()


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Thread-safe set of active WebSocket clients for broadcast."""

    __slots__ = ("_active",)

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    async def accept(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("Client connected — %d active", len(self._active))

        # Send reconnect_policy for client-side jittered backoff
        await ws.send_json({
            "type": "reconnect_policy",
            "action": "connected",
            "base_s": 1.0,
            "max_s": 30.0,
            "jitter": True,
        })

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("Client disconnected — %d active", len(self._active))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send JSON payload to every connected client; drop failures."""
        payload = json.dumps(data, separators=(",", ":"))
        stale: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                stale.append(ws)
        for ws in stale:
            self._active.discard(ws)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_manager = ConnectionManager()
_reconnect_guard = ReconnectionGuard()
_bus = RedisBus()
_webrtc = WebRTCTransport()

# Phase 4 engine singletons
_voice_engine = VoiceEngine()
_clip_engine = ClipEngine()
_push_service = PushService()
_dressing_room = DressingRoomWS()
_volumetric = VolumetricEngine()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """Startup / shutdown hooks."""
    await _bus.connect()

    # Phase 4: Initialize engines
    _clip_engine.redis_client = _bus.client
    _push_service.redis_client = _bus.client
    _dressing_room.redis_client = _bus.client
    await _voice_engine.start()
    await _volumetric.start()

    relay_task = asyncio.create_task(_relay_results())
    ghost_relay = asyncio.create_task(_relay_ghost_results())
    yield
    relay_task.cancel()
    ghost_relay.cancel()
    await _webrtc.close_all()
    await _bus.shutdown()


app = FastAPI(
    title="C.R.U.Y.F.F. Gateway",
    version="1.0.0",
    description=(
        "Computational Real-time Unified Yield Field Framework\n\n"
        "A live tactical intelligence platform for elite football.\n\n"
        "**Routes:**\n"
        "- `/ws/tracking` — 25Hz optical tracking ingest\n"
        "- `/ws/ghost` — Ghost Engine RPC\n"
        "- `/ws/voice` — Voice NLP (Whisper + speaker diarization)\n"
        "- `/ws/omnicam` — 3DGS Volumetric Omni-Cam\n"
        "- `/ws/dressing-room` — Apple TV halftime feed\n"
        "- `POST /api/push` — Dressing room push\n"
        "- `POST /webrtc/offer` — WebRTC SDP exchange\n"
        "- `GET /health` — System health check"
    ),
    lifespan=_lifespan,
)


async def _relay_results() -> None:
    """Relay analysis results → WebSocket + WebRTC clients."""
    try:
        async for payload in _bus.subscribe(settings.redis_channel_result):
            await _manager.broadcast(payload)
            await _webrtc.broadcast(payload)  # Fix 17: UDP fire-and-forget
    except asyncio.CancelledError:
        logger.info("Result relay cancelled")


async def _relay_ghost_results() -> None:
    """Relay Ghost RPC results → all WebSocket clients."""
    try:
        async for payload in _bus.subscribe("cruyff:ghost:result"):
            await _manager.broadcast({"type": "ghost", **payload})
    except asyncio.CancelledError:
        logger.info("Ghost relay cancelled")


# ---------------------------------------------------------------------------
# WebSocket endpoint — 25Hz tracking ingest
# ---------------------------------------------------------------------------

@app.websocket("/ws/tracking")
async def tracking_ingest(ws: WebSocket) -> None:
    """
    Ingest endpoint for the Tactical Glass tablet.

    Fix 16: Reconnection guard prevents thundering herd.
    Fix 13: Tags each frame with ``ingest_monotonic`` (NTP-immune).
    """
    # Thundering herd guard
    if not await _reconnect_guard.gate(ws):
        return

    await _manager.accept(ws)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame = TrackingFrame.model_validate_json(raw)
            except ValidationError as exc:
                await ws.send_json({"error": str(exc)})
                continue

            # Fix 13: Tag with monotonic ingestion time (NTP-immune)
            frame.ingest_monotonic = time.monotonic()

            await _bus.publish(
                settings.redis_channel_raw,
                frame.model_dump(),
            )
    except WebSocketDisconnect:
        _manager.disconnect(ws)
    except Exception:
        logger.exception("Unexpected error in tracking_ingest")
        _manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Ghost RPC endpoint — on-demand from frontend
# ---------------------------------------------------------------------------

@app.websocket("/ws/ghost")
async def ghost_request(ws: WebSocket) -> None:
    """
    The manager's "Optimize" button sends a Ghost request via this
    endpoint.  The gateway publishes to Redis ``cruyff:ghost:request``
    and the worker's Ghost RPC loop picks it up.
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            await _bus.publish("cruyff:ghost:request", payload)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Fix 17: WebRTC signalling endpoint
# ---------------------------------------------------------------------------

@app.post("/webrtc/offer")
async def webrtc_offer(request: dict[str, str]) -> JSONResponse:
    """
    Receive a client's SDP offer and return the server's SDP answer.

    The frontend sends::

        POST /webrtc/offer
        {"sdp": "...", "type": "offer"}

    The server opens an unreliable, unordered data channel and returns
    the SDP answer.  All subsequent analysis results are broadcast via
    this channel (UDP — no head-of-line blocking).
    """
    if not _webrtc.available():
        return JSONResponse(
            status_code=501,
            content={"error": "WebRTC not available (aiortc not installed). Using WebSocket fallback."},
        )

    try:
        answer = await _webrtc.handle_offer(
            offer_sdp=request["sdp"],
            offer_type=request.get("type", "offer"),
        )
        return JSONResponse(content=json.loads(answer))
    except Exception as exc:
        logger.exception("WebRTC signalling failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "operational",
        "system": "C.R.U.Y.F.F.",
        "webrtc": "available" if _webrtc.available() else "fallback_ws",
        "peers": _webrtc.peer_count,
        "voice_engine": "ready",
        "volumetric": "ready",
    }


# ---------------------------------------------------------------------------
# Phase 4: Voice NLP WebSocket (/ws/voice)
# ---------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    """
    Voice-to-intent endpoint.

    Client sends 250ms PCM16 audio chunks (post-RNNoise).
    Server triggers Whisper on VAD pause or 5s guillotine.
    Returns transcript + parsed tactical intent.
    """
    await ws.accept()
    logger.info("Voice WebSocket connected")

    clips_buffer = []  # accumulate clip manifests for this session

    def _on_transcript(text: str):
        asyncio.create_task(
            ws.send_json({"type": "transcript", "text": text})
        )

    def _on_intent(intent: dict):
        asyncio.create_task(
            ws.send_json({"type": "intent", **intent})
        )
        # Auto-generate clip if it's a clip_request
        if intent.get("type") == "clip_request":
            asyncio.create_task(
                _handle_clip_request(ws, intent, clips_buffer)
            )

    _voice_engine.on_transcript(_on_transcript)
    _voice_engine.on_intent(_on_intent)

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                _voice_engine.feed_chunk(data["bytes"])
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("type") == "voice_flush":
                    _voice_engine.flush()
                elif msg.get("type") == "enroll":
                    # Voice enrollment (F46)
                    import numpy as np
                    sample = np.frombuffer(
                        bytes(msg["audio"]), dtype=np.float32
                    )
                    _voice_engine.enroll_manager_voice(sample)
                    await ws.send_json({"type": "enrolled", "ok": True})
    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")


async def _handle_clip_request(
    ws: WebSocket, intent: dict, clips_buffer: list
):
    """Generate a clip manifest from a voice intent."""
    manifest = await _clip_engine.create_clip(
        event_type=intent.get("event", "unknown"),
        minutes_ago=intent.get("minutes_ago"),
    )
    if manifest:
        clips_buffer.append(manifest.to_dict())
        await ws.send_json({
            "type": "clip",
            "manifest": manifest.to_dict(),
        })


# ---------------------------------------------------------------------------
# Phase 4: Push to Dressing Room (POST /api/push)
# ---------------------------------------------------------------------------

@app.post("/api/push")
async def push_to_dressing_room(request: dict):
    """
    Push tactical insights to the dressing room Apple TV.

    Body::

        {"insight_ids": ["ghost_42", "void_17"], "ttl_minutes": 15}
    """
    insight_ids = request.get("insight_ids", [])
    ttl = request.get("ttl_minutes", 15)

    if not insight_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "insight_ids required"},
        )

    payload = await _push_service.push(
        insight_ids=insight_ids,
        ttl_minutes=ttl,
    )

    if payload:
        return JSONResponse(content={
            "push_id": payload.push_id,
            "insights": len(payload.insights),
            "telemetry_frames": len(payload.telemetry_block),
            "hls_segments": len(payload.hls_segments),
            "expires_at": payload.expires_at,
        })
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "No valid insights found"},
        )


# ---------------------------------------------------------------------------
# Phase 4: Dressing Room WebSocket (/ws/dressing-room)
# ---------------------------------------------------------------------------

@app.websocket("/ws/dressing-room")
async def dressing_room_ws(ws: WebSocket):
    """
    Apple TV dressing room endpoint.

    Fix F38: Only accessible on 192.168.x.x intranet (Ethernet).
    Receives push payloads with telemetry + HLS for local rendering.
    """
    await _dressing_room.handle(ws)


# ---------------------------------------------------------------------------
# Phase 4: Volumetric Omni-Cam WebSocket (/ws/omnicam)
# ---------------------------------------------------------------------------

@app.websocket("/ws/omnicam")
async def omnicam_ws(ws: WebSocket):
    """
    Virtual camera pose update endpoint.

    Client sends predicted touch coordinates (via TouchKalman).
    Server updates the 3DGS rendering pose.
    """
    await ws.accept()
    logger.info("OmniCam WebSocket connected")

    try:
        while True:
            text = await ws.receive_text()
            msg = json.loads(text)

            if msg.get("type") == "omnicam_pose":
                _volumetric.update_pose(msg)
                # Return current lighting state (F47)
                await ws.send_json({
                    "type": "omnicam_ack",
                    "frame": _volumetric._frame_count,
                    "lighting": _volumetric.lighting,
                })
            elif msg.get("type") == "ptz_update":
                # Fix F43: PTZ extrinsic update
                import numpy as np
                cam_idx = msg.get("camera_idx", 0)
                homography = np.array(
                    msg["homography"], dtype=np.float32
                )
                _volumetric.update_ptz_extrinsics(cam_idx, homography)

    except WebSocketDisconnect:
        logger.info("OmniCam WebSocket disconnected")
