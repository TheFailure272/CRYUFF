"""
C.R.U.Y.F.F. — Redis Pub/Sub Bus

Thin async wrapper around ``redis.asyncio`` providing typed publish /
subscribe primitives with automatic connection pooling and graceful
shutdown.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis

from shared.config import settings

logger = logging.getLogger(__name__)


class RedisBus:
    """
    Async Redis Pub/Sub dispatcher.

    Lifecycle::

        bus = RedisBus()
        await bus.connect()
        ...
        await bus.shutdown()
    """

    __slots__ = ("_pool", "_pubsub", "_url")

    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._pool: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the connection pool and pubsub handle."""
        self._pool = aioredis.from_url(
            self._url,
            decode_responses=False,     # we handle our own (de)serialisation
            max_connections=16,
        )
        self._pubsub = self._pool.pubsub()
        logger.info("Redis bus connected → %s", self._url)

    async def shutdown(self) -> None:
        """Drain subscriptions and close the pool."""
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._pool is not None:
            await self._pool.close()
        logger.info("Redis bus shut down")

    # ── publish ────────────────────────────────────────────────────────────

    async def publish(self, channel: str, payload: dict[str, Any]) -> int:
        """
        Serialise ``payload`` as JSON and publish to ``channel``.

        Returns the number of subscribers that received the message.
        """
        assert self._pool is not None, "Call connect() first"
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        count: int = await self._pool.publish(channel, data)
        return count

    # ── subscribe ──────────────────────────────────────────────────────────

    async def subscribe(
        self,
        channel: str,
        *,
        poll_interval: float = 0.001,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yield parsed JSON payloads as they arrive on ``channel``.

        This is a long-lived generator — break out of it to unsubscribe.

        Parameters
        ----------
        channel : str
            Redis channel name.
        poll_interval : float
            Seconds between polling cycles (keeps CPU idle when quiet).
        """
        assert self._pubsub is not None, "Call connect() first"
        await self._pubsub.subscribe(channel)
        logger.info("Subscribed → %s", channel)

        try:
            while True:
                msg = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=poll_interval,
                )
                if msg is not None and msg["type"] == "message":
                    raw: bytes = msg["data"]
                    yield json.loads(raw)
                else:
                    await asyncio.sleep(poll_interval)
        finally:
            await self._pubsub.unsubscribe(channel)
