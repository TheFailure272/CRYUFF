"""
Integration tests for Volumetric Engine (Feature 1).

Tests:
  * F39: Semantic-masked rendering pipeline
  * F43: PTZ extrinsic updates
  * F47: Dynamic SH lighting modulation
  * Touch Kalman integration
"""
import pytest
import numpy as np

from engine.volumetric import VolumetricEngine, CameraPose


@pytest.fixture
def engine():
    return VolumetricEngine(
        camera_urls=["rtsp://cam0", "rtsp://cam1"],
        resolution=(640, 360),
    )


class TestCameraPose:
    def test_from_touch(self):
        pose = CameraPose.from_touch({
            "pos": [52.5, 34, 15],
            "quat": [1, 0, 0, 0],
            "fov": 45.0,
        })
        assert pose.fov == 45.0
        assert np.allclose(pose.position, [52.5, 34, 15])

    def test_default_pose(self):
        pose = CameraPose.from_touch({})
        assert pose.fov == 60.0


class TestVolumetricEngine:
    @pytest.mark.asyncio
    async def test_engine_start(self, engine):
        await engine.start()
        assert engine._splat_model is not None
        assert engine._sam_model is not None
        assert len(engine._camera_extrinsics) == 2

    def test_render_returns_frame(self, engine):
        frame = engine.render()
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8

    def test_update_pose(self, engine):
        engine.update_pose({"pos": [10, 20, 5]})
        assert engine._current_pose is not None

    def test_stats(self, engine):
        s = engine.stats
        assert s["cameras"] == 2
        assert s["resolution"] == (640, 360)


class TestPTZSync:
    """F43: PTZ extrinsic updates."""

    @pytest.mark.asyncio
    async def test_ptz_update_3x3(self, engine):
        await engine.start()
        H = np.eye(3, dtype=np.float32) * 1.1
        engine.update_ptz_extrinsics(0, H)
        # Should be embedded in 4x4
        assert engine._camera_extrinsics[0].shape == (4, 4)
        assert engine._camera_extrinsics[0][0, 0] == pytest.approx(1.1)

    @pytest.mark.asyncio
    async def test_ptz_update_4x4(self, engine):
        await engine.start()
        H = np.eye(4, dtype=np.float32) * 0.9
        engine.update_ptz_extrinsics(1, H)
        assert engine._camera_extrinsics[1][0, 0] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_ptz_out_of_bounds_ignored(self, engine):
        await engine.start()
        # Camera index 99 doesn't exist — should not crash
        engine.update_ptz_extrinsics(99, np.eye(3))


class TestSHLighting:
    """F47: Dynamic spherical harmonic modulation."""

    def test_initial_lighting(self, engine):
        assert engine._sh_color_temp_k == 5500.0
        assert engine._sh_luminance_offset == 0.0

    def test_lighting_property(self, engine):
        L = engine.lighting
        assert "luminance" in L
        assert "color_temp_k" in L

    def test_sh_update_with_warm_frame(self, engine):
        """Warm frame (high R, low B) → color temp ~3500K."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # high red
        frame[:, :, 1] = 100  # mid green
        frame[:, :, 2] = 80   # low blue
        engine._update_sh_lighting([frame])
        assert engine._sh_color_temp_k == 3500.0  # warm

    def test_sh_update_with_cool_frame(self, engine):
        """Cool frame (low R, high B) → color temp ~6500K."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 80   # low red
        frame[:, :, 1] = 120  # mid green
        frame[:, :, 2] = 200  # high blue
        engine._update_sh_lighting([frame])
        assert engine._sh_color_temp_k == 6500.0  # cool floodlights
