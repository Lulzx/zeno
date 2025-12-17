"""
Latency Benchmark

Measures per-step latency and jitter for real-time applications.
Important for control applications requiring consistent timing.

Usage:
    python -m zeno.examples benchmark_latency
    python -m zeno.examples benchmark_latency --num-envs 64
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Steps to measure")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run latency benchmark."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Latency Benchmark")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Measurement steps: {args.steps}")
    print()

    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)

    # Warmup
    print("Warming up...")
    obs = env.reset()
    for _ in range(args.warmup):
        obs, _, _, _ = env.step(actions)
    print()

    # Measure latencies
    print("Measuring latencies...")
    latencies = []

    obs = env.reset()
    for _ in range(args.steps):
        start = time.perf_counter()
        obs, rewards, dones, info = env.step(actions)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    latencies = np.array(latencies) * 1000  # Convert to milliseconds

    # Statistics
    print()
    print("=" * 60)
    print("Latency Statistics (milliseconds)")
    print("=" * 60)

    print(f"Mean:      {np.mean(latencies):>10.4f} ms")
    print(f"Std:       {np.std(latencies):>10.4f} ms")
    print(f"Min:       {np.min(latencies):>10.4f} ms")
    print(f"Max:       {np.max(latencies):>10.4f} ms")
    print(f"Median:    {np.median(latencies):>10.4f} ms")
    print()

    # Percentiles
    print("Percentiles:")
    for p in [50, 90, 95, 99, 99.9]:
        value = np.percentile(latencies, p)
        print(f"  P{p:<5}: {value:>10.4f} ms")
    print()

    # Jitter analysis
    jitter = np.diff(latencies)
    print("Jitter (step-to-step variation):")
    print(f"  Mean jitter:   {np.mean(np.abs(jitter)):>10.4f} ms")
    print(f"  Max jitter:    {np.max(np.abs(jitter)):>10.4f} ms")
    print()

    # Real-time capability
    target_dt = env.timestep * 1000  # Target timestep in ms
    realtime_capable = np.mean(latencies) < target_dt
    headroom = target_dt - np.mean(latencies)

    print("Real-time Analysis:")
    print(f"  Target timestep:  {target_dt:>10.4f} ms")
    print(f"  Mean latency:     {np.mean(latencies):>10.4f} ms")
    print(f"  Headroom:         {headroom:>10.4f} ms")
    print(f"  Real-time capable: {'Yes' if realtime_capable else 'No'}")

    if realtime_capable:
        safety_margin = (headroom / target_dt) * 100
        print(f"  Safety margin:    {safety_margin:>10.1f}%")

    # Latency histogram (text-based)
    print()
    print("Latency Distribution:")
    hist, edges = np.histogram(latencies, bins=10)
    max_count = max(hist)
    for i in range(len(hist)):
        bar_len = int(40 * hist[i] / max_count)
        bar = "#" * bar_len
        print(f"  {edges[i]:>6.3f}-{edges[i+1]:>6.3f}: {bar} ({hist[i]})")

    env.close()


if __name__ == "__main__":
    main()
