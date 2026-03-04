"""
C.R.U.Y.F.F. — Dressing Room WebSocket (Feature 5)

WebSocket endpoint for Apple TV dressing room clients.

Fix F38: Air-Gapped LAN — this endpoint is only accessible
on the 192.168.x.x intranet via hardwired Ethernet.

Architecture:
  1. Apple TV connects to ``/ws/dressing-room``
  2. Server subscribes to Redis ``cruyff:dressing_room`` channel
  3. On push event: forwards raw telemetry JSON + HLS timestamps
  4. Apple TV renders AR overlays locally on its A15 GPU
  5. Zero server-side rendering (no Puppeteer, no ffmpeg)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DressingRoomWS:
    """
    WebSocket manager for dressing room Apple TV clients.

    Usage (with FastAPI)::

        dressing_room = DressingRoomWS(redis_client=redis)

        @app.websocket("/ws/dressing-room")
        async def ws_dressing_room(websocket):
            await dressing_room.handle(websocket)
    """

    redis_client: object = None
    redis_channel: str = "cruyff:dressing_room"

    _clients: set = field(default_factory=set, init=False, repr=False)

    async def handle(self, websocket) -> None:
        """
        Handle a single Apple TV WebSocket connection.

        The client receives push payloads containing:
          - insights (ghost runs, voids, alerts)
          - telemetry JSON (for local AR rendering)
          - HLS segment URLs (GOP=1 intra-frame video)
        """
        await websocket.accept()
        self._clients.add(websocket)

        logger.info(
            "Dressing room client connected (%d total)",
            len(self._clients),
        )

        try:
            # Send any existing push payload (late joiner)
            await self._send_latest(websocket)

            # Start Redis subscriber for new pushes
            sub_task = asyncio.create_task(
                self._subscribe_and_forward(websocket)
            )

            # Keep connection alive, handle pings
            while True:
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )
                    # Client can send "ack" or "next" commands
                    if msg == "next":
                        await self._send_latest(websocket)
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

        finally:
            sub_task.cancel()
            self._clients.discard(websocket)
            logger.info(
                "Dressing room client disconnected (%d remaining)",
                len(self._clients),
            )

    async def _send_latest(self, websocket) -> None:
        """Send the most recent push payload to a client."""
        if not self.redis_client:
            return

        try:
            # Scan for latest push key
            keys = []
            async for key in self.redis_client.scan_iter("cruyff:push:*"):
                keys.append(key)

            if keys:
                # Sort by timestamp in key name, get newest
                keys.sort(reverse=True)
                raw = await self.redis_client.get(keys[0])
                if raw:
                    await websocket.send_text(raw)
                    logger.info("Sent latest push to dressing room client")
        except Exception as e:
            logger.error("Failed to send latest push: %s", e)

    async def _subscribe_and_forward(self, websocket) -> None:
        """Subscribe to Redis channel and forward pushes."""
        if not self.redis_client:
            # Mock: wait forever
            await asyncio.sleep(float('inf'))
            return

        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(self.redis_channel)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        await websocket.send_text(
                            message["data"].decode()
                            if isinstance(message["data"], bytes)
                            else message["data"]
                        )
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Redis subscription error: %s", e)

    async def broadcast(self, payload_json: str) -> None:
        """Broadcast a push to all connected dressing room clients."""
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send_text(payload_json)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected
