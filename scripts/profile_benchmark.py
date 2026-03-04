"""
C.R.U.Y.F.F. — Performance Profiling Benchmark

Measures latency of all critical engine operations:
  * Set-piece solver (JAX/NumPy)
  * Voice intent parsing
  * EKF update cycle
  * Spatial bridge transform
  * Volumetric render
  * Clip engine manifest creation

Usage:
    python scripts/profile_benchmark.py
"""
import asyncio
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark(name, fn, n=100):
    """Run fn() n times and report statistics."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(times)
    print(f"  {name:40s}  "
          f"mean={arr.mean():8.2f}ms  "
          f"p50={np.median(arr):8.2f}ms  "
          f"p99={np.percentile(arr, 99):8.2f}ms  "
          f"(n={n})")
    return arr.mean()


async def benchmark_async(name, fn, n=100):
    """Async version of benchmark."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        await fn()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"  {name:40s}  "
          f"mean={arr.mean():8.2f}ms  "
          f"p50={np.median(arr):8.2f}ms  "
          f"p99={np.percentile(arr, 99):8.2f}ms  "
          f"(n={n})")
    return arr.mean()


def profile_setpiece_solver():
    print("\n── Set-Piece Solver ──")
    from engine.setpiece_solver import SetPieceSolver
    solver = SetPieceSolver(altitude_m=610, temperature_c=25)

    benchmark("100 trajectories (inswing)", lambda: solver.solve(
        origin_m=(0, 0, 0.3), target_y_m=16.5,
        delivery_type="inswing", n_samples=100,
    ))
    benchmark("100 trajectories (driven)", lambda: solver.solve(
        origin_m=(0, 0, 0.3), target_y_m=16.5,
        delivery_type="driven", n_samples=100,
    ), n=10)


def profile_voice_engine():
    print("\n── Voice Engine ──")
    from engine.voice_engine import VoiceEngine
    engine = VoiceEngine()

    benchmark("Intent parse (void)", lambda:
              engine._parse_intent("show me the void from 5 minutes ago"))
    benchmark("Intent parse (ghost)", lambda:
              engine._parse_intent("ghost run 12 minutes ago"))

    # Buffer feed
    chunk = np.random.randint(-1000, 1000, 4000, dtype=np.int16).tobytes()
    benchmark("Feed 250ms chunk", lambda: engine.feed_chunk(chunk), n=200)
    engine._buffer.clear()  # cleanup


def profile_spatial_bridge():
    print("\n── Spatial Bridge ──")
    from engine.spatial_bridge import SpatialBridge
    bridge = SpatialBridge(update_interval=0)

    # Pre-calibrate
    for i in range(10):
        bridge.update_pair(
            player_id=i,
            optical_xy=(10.0 * i, 5.0 * i),
            gps_latlon=(51.555 + i * 0.0001, -0.279 + i * 0.0001),
        )

    benchmark("gps_to_pitch (no facing)", lambda:
              bridge.gps_to_pitch(51.5555, -0.2785,
                                  velocity_xy=(3, 2), acceleration=2.5))
    benchmark("gps_to_pitch (F45 facing)", lambda:
              bridge.gps_to_pitch(51.5555, -0.2785,
                                  velocity_xy=(0, -3), acceleration=4.0,
                                  facing_angle_rad=1.57))


def profile_volumetric():
    print("\n── Volumetric Engine ──")
    from engine.volumetric import VolumetricEngine
    engine = VolumetricEngine(
        camera_urls=["rtsp://cam0", "rtsp://cam1"],
        resolution=(640, 360),
    )

    benchmark("Render frame (640x360)", lambda: engine.render())
    benchmark("Update pose", lambda: engine.update_pose(
        {"pos": [52, 34, 15], "quat": [1, 0, 0, 0]}))

    # SH update
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    benchmark("SH lighting update (F47)", lambda:
              engine._update_sh_lighting([frame]), n=50)


async def profile_clip_engine():
    print("\n── Clip Engine ──")
    from engine.clip_engine import ClipEngine
    engine = ClipEngine(redis_client=None)

    await benchmark_async("Create clip (mock)", lambda:
                          engine.create_clip("ghost_run", match_time_s=720))


async def main():
    print("=" * 72)
    print("C.R.U.Y.F.F. — Performance Benchmark")
    print("=" * 72)

    profile_setpiece_solver()
    profile_voice_engine()
    profile_spatial_bridge()
    profile_volumetric()
    await profile_clip_engine()

    print("\n" + "=" * 72)
    print("Benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
