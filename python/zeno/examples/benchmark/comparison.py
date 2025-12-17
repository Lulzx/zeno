"""
Comparison Benchmark

Compares Zeno performance across different environments and configurations.
Useful for selecting optimal settings for different use cases.

Usage:
    python -m zeno.examples benchmark_comparison
    python -m zeno.examples benchmark_comparison --num-envs 1024
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-environment comparison benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=512, help="Environments per test")
    parser.add_argument("--steps", type=int, default=500, help="Steps per benchmark")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def benchmark_environment(env_name: str, num_envs: int, num_steps: int, warmup: int, seed: int) -> dict:
    """Benchmark a single environment type."""
    mjcf_path = get_asset(f"{env_name}.xml")

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
        obs, rewards, dones, info = env.step(actions)
    elapsed = time.perf_counter() - start

    total_steps = num_steps * num_envs
    sps = total_steps / elapsed

    # State access benchmark
    start = time.perf_counter()
    for _ in range(100):
        positions = env.get_body_positions()
        quaternions = env.get_body_quaternions()
    state_time = (time.perf_counter() - start) / 100

    result = {
        "name": env_name,
        "obs_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "sps": sps,
        "elapsed": elapsed,
        "state_access_ms": state_time * 1000,
        "obs_bytes": obs.nbytes,
    }

    env.close()
    return result


def main():
    """Run comparison benchmark."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Environment Comparison Benchmark")
    print("=" * 80)
    print(f"Environments per test: {args.num_envs}")
    print(f"Steps per benchmark: {args.steps}")
    print()

    environments = ["pendulum", "cartpole", "ant", "humanoid"]

    print("Running benchmarks...")
    print("-" * 80)

    results = []
    for env_name in environments:
        print(f"Benchmarking {env_name}...", end=" ", flush=True)
        try:
            result = benchmark_environment(env_name, args.num_envs, args.steps, args.warmup, args.seed)
            results.append(result)
            print(f"Done ({result['sps']:,.0f} SPS)")
        except Exception as e:
            print(f"Failed: {e}")

    # Results table
    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Environment':>12} | {'Obs Dim':>8} | {'Act Dim':>8} | {'SPS':>12} | {'State (ms)':>10} | {'Obs (KB)':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:>12} | "
              f"{r['obs_dim']:>8} | "
              f"{r['action_dim']:>8} | "
              f"{r['sps']:>12,.0f} | "
              f"{r['state_access_ms']:>10.3f} | "
              f"{r['obs_bytes'] / 1024:>8.1f}")

    # Analysis
    print()
    print("Analysis:")
    print("-" * 80)

    if results:
        sps_values = [r["sps"] for r in results]
        fastest = results[np.argmax(sps_values)]
        slowest = results[np.argmin(sps_values)]

        print(f"Fastest: {fastest['name']} ({fastest['sps']:,.0f} SPS)")
        print(f"Slowest: {slowest['name']} ({slowest['sps']:,.0f} SPS)")
        print(f"Speed ratio: {fastest['sps'] / slowest['sps']:.1f}x")
        print()

        # Complexity vs performance
        print("Complexity analysis:")
        for r in results:
            complexity = r["obs_dim"] + r["action_dim"]
            efficiency = r["sps"] / complexity
            print(f"  {r['name']}: {complexity} dims -> {efficiency:,.0f} SPS/dim")

    # Recommendations
    print()
    print("Recommendations:")
    print("-" * 80)
    print("- For maximum throughput: Use simpler environments (pendulum, cartpole)")
    print("- For realistic robotics: Use ant or humanoid")
    print("- Scale num_envs to maximize GPU utilization")
    print(f"- Current config processes {sum(r['sps'] for r in results) / len(results):,.0f} avg SPS")


if __name__ == "__main__":
    main()
