"""
C.R.U.Y.F.F. — Wearable Data Ingest (Feature 3)

Async UDP listener for Catapult/STATSports GPS+HR vests.

Architecture
~~~~~~~~~~~~
Stadium GPS vests broadcast live telemetry via UDP. This module
listens on a configurable port and parses vendor-specific payloads
into a unified schema consumed by the Spatial Bridge and EKF.

Supported Vendors
~~~~~~~~~~~~~~~~~
* Catapult OpenField Live API (JSON over UDP)
* STATSports Sonra (protobuf over UDP)
* Generic fallback (JSON with standard field names)

Unified Output Schema
~~~~~~~~~~~~~~~~~~~~~
::

    {
        "player_id": 8,
        "hr_bpm": 172,
        "hr_max_pct": 0.91,
        "lat": 51.555847,
        "lon": -0.279519,
        "speed_ms": 7.2,
        "distance_m": 8456.3,
        "accel_load": 12.4,
        "timestamp": 1709542800.123
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WearableReading:
    """Unified wearable telemetry reading."""
    player_id: int
    hr_bpm: float
    hr_max_pct: float
    lat: float
    lon: float
    speed_ms: float
    distance_m: float
    accel_load: float
    timestamp: float  # monotonic-aligned


# ─── Vendor Parsers ─────────────────────────────────────────────

def _parse_catapult(data: dict) -> Optional[WearableReading]:
    """Parse Catapult OpenField Live API JSON."""
    try:
        return WearableReading(
            player_id=int(data["athlete_id"]),
            hr_bpm=float(data.get("heart_rate", 0)),
            hr_max_pct=float(data.get("hr_max_percent", 0)) / 100,
            lat=float(data["latitude"]),
            lon=float(data["longitude"]),
            speed_ms=float(data.get("velocity", 0)),
            distance_m=float(data.get("total_distance", 0)),
            accel_load=float(data.get("player_load", 0)),
            timestamp=float(data.get("timestamp", time.monotonic())),
        )
    except (KeyError, ValueError) as e:
        logger.warning("Catapult parse error: %s", e)
        return None


def _parse_statstports(data: dict) -> Optional[WearableReading]:
    """Parse STATSports Sonra JSON."""
    try:
        return WearableReading(
            player_id=int(data["playerId"]),
            hr_bpm=float(data.get("heartRate", 0)),
            hr_max_pct=float(data.get("hrMaxPct", 0)),
            lat=float(data["gps"]["lat"]),
            lon=float(data["gps"]["lon"]),
            speed_ms=float(data.get("speed", 0)),
            distance_m=float(data.get("totalDistance", 0)),
            accel_load=float(data.get("metabolicLoad", 0)),
            timestamp=float(data.get("ts", time.monotonic())),
        )
    except (KeyError, ValueError) as e:
        logger.warning("STATSports parse error: %s", e)
        return None


def _parse_generic(data: dict) -> Optional[WearableReading]:
    """Parse generic JSON with standard field names."""
    try:
        return WearableReading(
            player_id=int(data["player_id"]),
            hr_bpm=float(data.get("hr_bpm", 0)),
            hr_max_pct=float(data.get("hr_max_pct", 0)),
            lat=float(data.get("lat", 0)),
            lon=float(data.get("lon", 0)),
            speed_ms=float(data.get("speed_ms", 0)),
            distance_m=float(data.get("distance_m", 0)),
            accel_load=float(data.get("accel_load", 0)),
            timestamp=float(data.get("timestamp", time.monotonic())),
        )
    except (KeyError, ValueError) as e:
        logger.warning("Generic parse error: %s", e)
        return None


_PARSERS = {
    "catapult": _parse_catapult,
    "statstports": _parse_statstports,
    "generic": _parse_generic,
}


# ─── UDP Protocol ───────────────────────────────────────────────

class WearableProtocol(asyncio.DatagramProtocol):
    """Asyncio UDP protocol for receiving wearable data."""

    def __init__(self, callback: Callable[[WearableReading], None],
                 vendor: str = "generic"):
        self._callback = callback
        self._parser = _PARSERS.get(vendor, _parse_generic)
        self._count = 0

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        reading = self._parser(payload)
        if reading:
            self._count += 1
            self._callback(reading)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if exc:
            logger.error("Wearable UDP connection lost: %s", exc)


# ─── Ingest Service ─────────────────────────────────────────────

@dataclass
class WearableIngest:
    """
    Manages the wearable UDP listener lifecycle.

    Usage::

        ingest = WearableIngest(port=5555, vendor="catapult")

        async with ingest:
            # ingest.latest[player_id] has most recent reading
            reading = ingest.latest.get(8)
    """

    port: int = 5555
    vendor: str = "generic"
    latest: dict[int, WearableReading] = field(
        default_factory=dict, init=False, repr=False
    )
    _transport: Optional[asyncio.DatagramTransport] = field(
        default=None, init=False, repr=False
    )
    _callbacks: list[Callable] = field(
        default_factory=list, init=False, repr=False
    )

    def on_reading(self, callback: Callable[[WearableReading], None]) -> None:
        """Register a callback for each new reading."""
        self._callbacks.append(callback)

    def _handle_reading(self, reading: WearableReading) -> None:
        self.latest[reading.player_id] = reading
        for cb in self._callbacks:
            try:
                cb(reading)
            except Exception as e:
                logger.error("Wearable callback error: %s", e)

    async def start(self) -> None:
        """Start listening for wearable UDP data."""
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: WearableProtocol(self._handle_reading, self.vendor),
            local_addr=("0.0.0.0", self.port),
        )
        self._transport = transport
        logger.info(
            "Wearable ingest listening on UDP port %d (vendor=%s)",
            self.port, self.vendor,
        )

    async def stop(self) -> None:
        """Stop the UDP listener."""
        if self._transport:
            self._transport.close()
            self._transport = None
            logger.info("Wearable ingest stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
