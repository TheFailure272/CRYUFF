"""
Integration tests for Push Service + Dressing Room (Feature 5).

Tests:
  * F38: LAN-only HLS URL generation
  * Push payload structure
  * GOP=1 segment generation
"""
import pytest

from server.push_service import PushService, PushPayload


@pytest.fixture
def service():
    return PushService(
        redis_client=None,  # mock mode
        hls_base_url="http://192.168.1.10:8080/hls",
    )


class TestPushPayload:
    def test_payload_serialization(self):
        p = PushPayload(
            push_id="push_test",
            insights=[{"id": "ghost_1"}],
            telemetry_block=[{"t": 0.0}],
            hls_segments=["http://192.168.1.10:8080/hls/seg.ts"],
            clip_timestamps=[720.0],
            created_at=1000.0,
            expires_at=1900.0,
        )
        j = p.to_json()
        assert "push_test" in j
        assert "192.168.1.10" in j  # F38: LAN-only


class TestPushService:
    @pytest.mark.asyncio
    async def test_push_generates_payload(self, service):
        payload = await service.push(
            insight_ids=["ghost_1", "void_2"],
            ttl_minutes=15.0,
        )
        assert payload is not None
        assert len(payload.insights) == 2
        assert payload.push_id.startswith("push_")

    @pytest.mark.asyncio
    async def test_push_empty_ids_returns_none(self, service):
        payload = await service.push(insight_ids=[], ttl_minutes=10)
        assert payload is None

    def test_gop1_segments_use_lan_url(self, service):
        """F38: All segments must be on 192.168.x.x."""
        segments = service._get_gop1_segments(10.0, 15.0)
        assert len(segments) > 0
        for seg in segments:
            assert "192.168.1.10" in seg
            assert seg.endswith(".ts")

    @pytest.mark.asyncio
    async def test_push_has_ttl(self, service):
        payload = await service.push(
            insight_ids=["test"],
            ttl_minutes=5.0,
        )
        assert payload is not None
        assert payload.expires_at > payload.created_at
        # TTL = 5 minutes = 300 seconds
        assert (payload.expires_at - payload.created_at) == pytest.approx(300, abs=1)
