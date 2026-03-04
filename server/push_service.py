"""
C.R.U.Y.F.F. — Push Service (Feature 5)

Publishes tactical insights to the dressing room Apple TV.

Fix F38: Air-Gapped LAN
~~~~~~~~~~~~~~~~~~~~~~~~
The Apple TV must be hardwired via Ethernet to the edge server's
local switch. All communication is strictly over 192.168.x.x
intranet — no cloud, no stadium Wi-Fi.

Architecture:
  1. Manager hits "PUSH" on tablet
  2. POST /push with insight IDs
  3. Backend queries Redis for telemetry + insight data
  4. Publishes to Redis channel ``cruyff:dressing_room``
  5. Apple TV WebSocket client (dressing_room_ws) receives
     and renders locally
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PushPayload:
    """Payload sent to the dressing room."""
    push_id: str
    insights: list[dict]           # ghost runs, voids, alerts
    telemetry_block: list[dict]    # 25Hz frames for AR rendering
    hls_segments: list[str]        # GOP=1 video segments
    clip_timestamps: list[float]   # start times for each insight
    created_at: float
    expires_at: float              # auto-expire after halftime

    def to_json(self) -> str:
        return json.dumps({
            "push_id": self.push_id,
            "insights": self.insights,
            "telemetry_frames": len(self.telemetry_block),
            "telemetry": self.telemetry_block,
            "hls_segments": self.hls_segments,
            "clip_timestamps": self.clip_timestamps,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        })


@dataclass
class PushService:
    """
    Manages the half-time dressing room push.

    Fix F38: All communication is strictly intranet.
    No cloud URLs, no WAN. The Apple TV connects via Ethernet
    on the same switch as the edge server.

    Usage::

        service = PushService(redis_client=redis)

        payload = await service.push(
            insight_ids=["ghost_42", "void_17"],
            ttl_minutes=15,
        )
    """

    redis_client: object = None
    hls_base_url: str = "http://192.168.1.10:8080/hls"  # Fix F38: LAN-only
    redis_channel: str = "cruyff:dressing_room"

    async def push(
        self,
        insight_ids: list[str],
        ttl_minutes: float = 15.0,
    ) -> Optional[PushPayload]:
        """
        Push insights to the dressing room.

        Parameters
        ----------
        insight_ids : list[str]
            IDs of insights to push (ghost runs, voids, alerts)
        ttl_minutes : float
            Time-to-live before payload auto-expires
        """
        now = time.time()
        push_id = f"push_{int(now)}"

        # Gather insights from Redis
        insights = []
        clip_timestamps = []
        all_start = float('inf')
        all_end = 0.0

        for iid in insight_ids:
            insight = await self._get_insight(iid)
            if insight:
                insights.append(insight)
                t = insight.get("match_time", 0)
                clip_timestamps.append(t)
                all_start = min(all_start, t - 7.5)
                all_end = max(all_end, t + 7.5)

        if not insights:
            logger.warning("Push failed — no valid insights found")
            return None

        # Package telemetry for the entire clip range
        telemetry = await self._get_telemetry(all_start, all_end)

        # GOP=1 HLS segments (Fix: intra-frame only for seek sync)
        hls_segments = self._get_gop1_segments(all_start, all_end)

        payload = PushPayload(
            push_id=push_id,
            insights=insights,
            telemetry_block=telemetry,
            hls_segments=hls_segments,
            clip_timestamps=clip_timestamps,
            created_at=now,
            expires_at=now + ttl_minutes * 60,
        )

        # Publish to Redis channel for Apple TV clients
        if self.redis_client:
            await self.redis_client.publish(
                self.redis_channel,
                payload.to_json(),
            )
            # Also store for late-joining clients
            await self.redis_client.setex(
                f"cruyff:push:{push_id}",
                int(ttl_minutes * 60),
                payload.to_json(),
            )

        logger.info(
            "Pushed %d insights to dressing room (push_id=%s, "
            "%d telemetry frames, %d segments, TTL=%dmin)",
            len(insights), push_id, len(telemetry),
            len(hls_segments), ttl_minutes,
        )

        return payload

    async def _get_insight(self, insight_id: str) -> Optional[dict]:
        """Fetch an insight from Redis."""
        if not self.redis_client:
            return {"id": insight_id, "match_time": 720, "mock": True}
        try:
            raw = await self.redis_client.get(f"cruyff:insight:{insight_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    async def _get_telemetry(
        self, start_s: float, end_s: float
    ) -> list[dict]:
        """Extract telemetry frames from Redis sorted set."""
        if not self.redis_client:
            n = int((end_s - start_s) * 25)
            return [{"t": start_s + i / 25} for i in range(n)]
        try:
            raw = await self.redis_client.zrangebyscore(
                "cruyff:telemetry", min=start_s, max=end_s
            )
            return [json.loads(r) for r in raw]
        except Exception:
            return []

    def _get_gop1_segments(
        self, start_s: float, end_s: float
    ) -> list[str]:
        """
        Generate GOP=1 HLS segment URLs.
        These are served from the intranet edge server only (Fix F38).
        """
        segments = []
        # GOP=1 segments are 0.5s each (2fps keyframe)
        seg_duration = 0.5
        s = int(start_s / seg_duration)
        e = int(end_s / seg_duration) + 1
        for i in range(s, e):
            segments.append(
                f"{self.hls_base_url}/gop1_seg_{i:06d}.ts"
            )
        return segments
