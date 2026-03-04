"""
C.R.U.Y.F.F. — WebSocket Load Test

Simulates N concurrent Tactical Glass tablets streaming 25Hz tracking data
to the /ws/tracking endpoint. Measures throughput, latency, and error rate.

Usage:
    python scripts/load_test.py --clients 10 --duration 30 --url ws://localhost:8000/ws/tracking
"""
import argparse
import asyncio
import json
import random
import time
import statistics
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


def _random_frame(seq: int) -> str:
    """Generate a synthetic 25Hz tracking frame."""
    coords = [random.uniform(0, 1) for _ in range(44)]
    return json.dumps({
        "coordinates": coords,
        "timestamp": time.time(),
        "sequence": seq,
        "team_ids": list(range(22)),
    })


async def tablet_client(
    client_id: int,
    url: str,
    duration: float,
    results: dict,
):
    """Simulate a single tablet streaming at 25Hz."""
    sent = 0
    errors = 0
    latencies = []
    interval = 1.0 / 25  # 40ms per frame

    try:
        async with websockets.connect(url) as ws:
            start = time.monotonic()
            seq = 0

            while (time.monotonic() - start) < duration:
                t0 = time.perf_counter()
                frame = _random_frame(seq)
                await ws.send(frame)
                sent += 1
                seq += 1

                # Try to receive analysis result (non-blocking)
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    latency = (time.perf_counter() - t0) * 1000
                    latencies.append(latency)
                except asyncio.TimeoutError:
                    pass

                # Maintain 25Hz cadence
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)

    except Exception as e:
        errors += 1
        print(f"  Client {client_id}: ERROR — {e}")

    results[client_id] = {
        "sent": sent,
        "responses": len(latencies),
        "errors": errors,
        "latency_ms": latencies,
    }


async def run_load_test(url: str, n_clients: int, duration: float):
    """Run N concurrent tablet clients."""
    print(f"{'='*60}")
    print(f"C.R.U.Y.F.F. — WebSocket Load Test")
    print(f"{'='*60}")
    print(f"  URL:       {url}")
    print(f"  Clients:   {n_clients}")
    print(f"  Duration:  {duration}s")
    print(f"  Target:    {n_clients * 25} frames/sec")
    print(f"{'='*60}\n")

    results = {}
    tasks = [
        tablet_client(i, url, duration, results)
        for i in range(n_clients)
    ]

    t0 = time.monotonic()
    await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.monotonic() - t0

    # ── Aggregate ──
    total_sent = sum(r["sent"] for r in results.values())
    total_recv = sum(r["responses"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())
    all_latencies = []
    for r in results.values():
        all_latencies.extend(r["latency_ms"])

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Wall time:     {wall:.1f}s")
    print(f"  Frames sent:   {total_sent:,}")
    print(f"  Responses:     {total_recv:,}")
    print(f"  Errors:        {total_errors}")
    print(f"  Throughput:    {total_sent / wall:.0f} frames/sec")

    if all_latencies:
        print(f"\n  Latency:")
        print(f"    Mean:  {statistics.mean(all_latencies):.1f}ms")
        print(f"    p50:   {statistics.median(all_latencies):.1f}ms")
        print(f"    p95:   {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.1f}ms")
        print(f"    p99:   {sorted(all_latencies)[int(len(all_latencies)*0.99)]:.1f}ms")
        print(f"    Max:   {max(all_latencies):.1f}ms")

    print(f"\n  Per client:")
    for cid in sorted(results.keys()):
        r = results[cid]
        rate = r["sent"] / wall if wall > 0 else 0
        print(f"    Client {cid:2d}: {r['sent']:5d} sent, "
              f"{r['responses']:4d} recv, "
              f"{rate:.0f} Hz")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="C.R.U.Y.F.F. WebSocket Load Test")
    parser.add_argument("--url", default="ws://localhost:8000/ws/tracking",
                        help="WebSocket endpoint URL")
    parser.add_argument("--clients", type=int, default=10,
                        help="Number of concurrent tablets (default: 10)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds (default: 30)")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.clients, args.duration))


if __name__ == "__main__":
    main()
