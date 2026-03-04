"""
C.R.U.Y.F.F. — Clip Engine (Feature 2)

Extracts historical video clips with packaged telemetry data.

Fix F37: Packaged Telemetry
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The frontend TelemetryRingBuffer (Fix F6) only holds 30 seconds
of live data. When a manager requests "show me the breakdown from
12 minutes ago", the clip engine must:

1. Query Redis for the exact telemetry JSON block at that timestamp
2. Extract the corresponding HLS video segments
3. Package BOTH into a single clip manifest
4. The ClipDrawer renders the AR overlay from the static JSON,
   completely bypassing the live useAnalysisStream

Fix F44: VAR Match-Time Paradox
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A football match does not run on monotonic wall-clock time.
VAR checks (4+ minutes), injury stoppages, and drinks breaks
create dead time. "10 minutes ago" must resolve to the official
match clock (from OPTA/Stats Perform API), NOT time.time().
The clip engine queries the official match clock index from Redis.

Dependencies
~~~~~~~~~~~~
* ``redis`` (async)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ClipManifest:
    """A self-contained clip with video + telemetry data."""
    clip_id: str
    event_type: str
    start_time: float       # match time in seconds
    end_time: float
    hls_segments: list[str]  # HLS segment URLs
    telemetry: list[dict]    # Fix F37: packaged 25Hz telemetry frames
    thumbnail_url: Optional[str] = None
    annotation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "event_type": self.event_type,
            "start_s": self.start_time,
            "end_s": self.end_time,
            "hls_segments": self.hls_segments,
            "telemetry_frames": len(self.telemetry),
            "telemetry": self.telemetry,
            "thumbnail": self.thumbnail_url,
            "annotation": self.annotation,
        }


@dataclass
class ClipEngine:
    """
    Generates clip manifests with packaged telemetry (Fix F37).

    Usage::

        engine = ClipEngine(redis_client=redis)
        manifest = await engine.create_clip(
            event_type="topological_void",
            minutes_ago=12,
            context_seconds=15,
        )
    """

    redis_client: object = None  # async redis client
    hls_base_url: str = "/hls"
    segment_duration: float = 2.0  # HLS segment length in seconds

    async def create_clip(
        self,
        event_type: str,
        minutes_ago: Optional[int] = None,
        match_time_s: Optional[float] = None,
        context_seconds: float = 15.0,
    ) -> Optional[ClipManifest]:
        """
        Create a clip manifest with packaged telemetry.

        Parameters
        ----------
        event_type : str
            Type of tactical event to find
        minutes_ago : int, optional
            Relative time ("12 minutes ago")
        match_time_s : float, optional
            Absolute match time in seconds
        context_seconds : float
            Total clip duration (centered on event)

        Returns
        -------
        ClipManifest or None
        """
        # Resolve timestamp
        if minutes_ago is not None:
            # Fix F44: Use OFFICIAL match clock, not wall-clock.
            # Accounts for VAR stoppages, injury time, drinks breaks.
            current_match_time = await self._get_official_match_time()
            target_time = current_match_time - (minutes_ago * 60)
        elif match_time_s is not None:
            target_time = match_time_s
        else:
            return None

        start_time = max(0, target_time - context_seconds / 2)
        end_time = target_time + context_seconds / 2

        # Extract HLS segments for this time range
        hls_segments = self._get_hls_segments(start_time, end_time)

        # Fix F37: Package telemetry from Redis
        # This is the critical fix — the frontend ring buffer
        # only has 30s of data, so we must serve the historical
        # telemetry alongside the video.
        telemetry = await self._package_telemetry(start_time, end_time)

        if not telemetry:
            logger.warning(
                "No telemetry found for %.1f-%.1fs — clip will lack AR",
                start_time, end_time,
            )

        clip_id = f"clip_{event_type}_{int(start_time)}"

        manifest = ClipManifest(
            clip_id=clip_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            hls_segments=hls_segments,
            telemetry=telemetry,
            annotation=f"{event_type} at {target_time:.0f}s",
        )

        logger.info(
            "Clip created: %s (%.0f-%.0fs, %d frames, %d segments)",
            clip_id, start_time, end_time,
            len(telemetry), len(hls_segments),
        )

        return manifest

    async def _get_official_match_time(self) -> float:
        """
        Fix F44: Get the OFFICIAL running match time from the Match Data API.

        This is NOT wall-clock time. It accounts for:
          - VAR review stoppages (4+ minutes of dead time)
          - Injury stoppages
          - Drinks breaks
          - Added time calculations

        The official match clock is sourced from OPTA / Stats Perform
        via the ingestion pipeline and stored in Redis as:
          cruyff:match_clock  (official running seconds)
          cruyff:stoppages    (JSON array of stoppage events)

        "10 minutes ago" = current_official_minute - 10, NOT
        current_wall_time - 600 seconds.
        """
        if self.redis_client:
            try:
                # Primary: official match clock
                t = await self.redis_client.get("cruyff:match_clock")
                if t:
                    return float(t)

                # Fallback: wall-clock based match time (less accurate)
                t = await self.redis_client.get("cruyff:match_time")
                if t:
                    return float(t)
            except Exception:
                pass
        return 0.0

    async def _package_telemetry(
        self, start_s: float, end_s: float
    ) -> list[dict]:
        """
        Fix F37: Extract telemetry frames from Redis for the clip range.

        Redis stores each 25Hz frame as a JSON entry in a sorted set,
        keyed by match timestamp. This method extracts the exact block
        of frames needed for the clip's AR overlay.
        """
        if not self.redis_client:
            # Mock: return empty frames
            frame_count = int((end_s - start_s) * 25)
            return [{"t": start_s + i / 25, "mock": True}
                    for i in range(frame_count)]

        try:
            # Redis sorted set: cruyff:telemetry (score = match_time)
            raw_frames = await self.redis_client.zrangebyscore(
                "cruyff:telemetry",
                min=start_s,
                max=end_s,
            )

            import json
            frames = []
            for raw in raw_frames:
                try:
                    frames.append(json.loads(raw))
                except Exception:
                    continue

            return frames

        except Exception as e:
            logger.error("Redis telemetry fetch failed: %s", e)
            return []

    def _get_hls_segments(
        self, start_s: float, end_s: float
    ) -> list[str]:
        """Generate HLS segment URLs for the time range."""
        segments = []
        seg_start = int(start_s / self.segment_duration)
        seg_end = int(end_s / self.segment_duration) + 1

        for i in range(seg_start, seg_end):
            segments.append(
                f"{self.hls_base_url}/segment_{i:06d}.ts"
            )

        return segments
