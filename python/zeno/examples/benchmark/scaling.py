"""
Scaling Benchmark

Measures how throughput scales with number of parallel environments.
Helps find optimal batch sizes for different hardware.

Usage:
    python -m zeno.examples benchmark_scaling
    python -m zeno.examples benchmark_scaling --max-envs 8192
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Environment scaling benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--min-envs", type=int, default=1, help="Minimum environments")
    parser.add_argument("--max-envs", type=int, default=4096, help="Maximum environments")
    parser.add_argument("--steps", type=int, default=500, help="Steps per measurement")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def benchmark_env_count(mjcf_path: str, num_envs: int, num_steps: int, warmup: int, seed: int) -> dict:
    """Benchmark a specific environment count."""
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=num_envs,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)

    # Warmup
    obs = env.reset()
    for _ in range(warmup):
        obs, _, _, _ = env.step(actions)

    # Benchmark
    obs = env.reset()
    start = time.perf_counter()
    for _ in range(num_steps):
        obs, _, _, _ = env.step(actions)
    elapsed = time.perf_counter() - start

    total_steps = num_steps * num_envs
    sps = total_steps / elapsed

    env.close()

    return {
        "num_envs": num_envs,
        "elapsed": elapsed,
        "sps": sps,
        "sps_per_env": sps / num_envs,
    }


def main():
    """Run scaling benchmark."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Scaling Benchmark")
    print("=" * 70)
    print(f"Environment: {args.env}")
    print(f"Environment range: {args.min_envs} - {args.max_envs}")
    print(f"Steps per measurement: {args.steps}")
    print()

    mjcf_path = get_asset(f"{args.env}.xml")

    # Generate test points (powers of 2 and intermediate values)
    env_counts = []
    n = args.min_envs
    while n <= args.max_envs:
        env_counts.append(n)
        if n * 2 <= args.max_envs:
            mid = int(n * 1.5)
            if mid not in env_counts and mid <= args.max_envs:
                env_counts.append(mid)
        n *= 2
    env_counts = sorted(set(env_counts))

    print(f"Testing {len(env_counts)} configurations...")
    print("-" * 70)
    print(f"{'Envs':>8} | {'Total SPS':>12} | {'SPS/Env':>10} | {'Time':>8} | {'Efficiency':>10}")
    print("-" * 70)

    results = []
    baseline_sps_per_env = None

    for num_envs in env_counts:
        try:
            result = benchmark_env_count(mjcf_path, num_envs, args.steps, args.warmup, args.seed)
            results.append(result)

            if baseline_sps_per_env is None:
                baseline_sps_per_env = result["sps_per_env"]
                efficiency = 100.0
            else:
                efficiency = (result["sps_per_env"] / baseline_sps_per_env) * 100

            print(f"{num_envs:>8} | {result['sps']:>12,.0f} | {result['sps_per_env']:>10,.0f} | "
                  f"{result['elapsed']:>7.3f}s | {efficiency:>9.1f}%")
        except Exception as e:
            print(f"{num_envs:>8} | {'FAILED':>12} | {str(e)[:30]}")

    # Summary
    print()
    print("=" * 70)
    print("Scaling Analysis")
    print("=" * 70)

    if results:
        sps_values = [r["sps"] for r in results]
        best_idx = np.argmax(sps_values)
        best = results[best_idx]

        print(f"Peak throughput: {best['sps']:,.0f} SPS at {best['num_envs']} environments")

        # Find knee point (where efficiency drops below 80%)
        for r in results:
            eff = (r["sps_per_env"] / baseline_sps_per_env) * 100
            if eff < 80:
                print(f"Efficiency drops below 80% at {r['num_envs']} environments")
                break

        # Scaling factor
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            env_ratio = last["num_envs"] / first["num_envs"]
            sps_ratio = last["sps"] / first["sps"]
            print(f"Scaling: {env_ratio:.0f}x environments -> {sps_ratio:.1f}x throughput")


if __name__ == "__main__":
    main()
