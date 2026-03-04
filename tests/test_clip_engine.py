"""
Integration tests for Clip Engine (Feature 2).

Tests:
  * F37: Packaged telemetry from Redis mock
  * F44: Official match clock resolution
  * HLS segment generation
  * Clip manifest structure
"""
import pytest

from engine.clip_engine import ClipEngine, ClipManifest


@pytest.fixture
def engine():
    return ClipEngine(
        redis_client=None,  # mock mode
        hls_base_url="/hls",
        segment_duration=2.0,
    )


class TestClipManifest:
    def test_manifest_to_dict(self):
        m = ClipManifest(
            clip_id="clip_test_0",
            event_type="topological_void",
            start_time=600.0,
            end_time=615.0,
            hls_segments=["/hls/seg_0.ts"],
            telemetry=[{"t": 600.0}],
            annotation="test",
        )
        d = m.to_dict()
        assert d["clip_id"] == "clip_test_0"
        assert d["telemetry_frames"] == 1
        assert d["start_s"] == 600.0
        assert d["end_s"] == 615.0


class TestClipEngine:
    @pytest.mark.asyncio
    async def test_create_clip_with_match_time(self, engine):
        manifest = await engine.create_clip(
            event_type="ghost_run",
            match_time_s=720.0,
            context_seconds=15.0,
        )
        assert manifest is not None
        assert manifest.event_type == "ghost_run"
        assert manifest.start_time == 712.5
        assert manifest.end_time == 727.5

    @pytest.mark.asyncio
    async def test_packaged_telemetry_mock(self, engine):
        """F37: Mock mode should generate 25Hz frames."""
        manifest = await engine.create_clip(
            event_type="test",
            match_time_s=300.0,
            context_seconds=10.0,
        )
        assert manifest is not None
        # 10 seconds * 25Hz = 250 frames
        assert len(manifest.telemetry) == 250

    @pytest.mark.asyncio
    async def test_create_clip_returns_none_without_params(self, engine):
        result = await engine.create_clip(event_type="test")
        assert result is None

    def test_hls_segments_generation(self, engine):
        segments = engine._get_hls_segments(10.0, 20.0)
        assert len(segments) > 0
        assert all(s.startswith("/hls/") for s in segments)
        assert all(s.endswith(".ts") for s in segments)

    @pytest.mark.asyncio
    async def test_official_match_time_returns_zero_without_redis(self, engine):
        """F44: Without Redis, fallback to 0.0."""
        t = await engine._get_official_match_time()
        assert t == 0.0
