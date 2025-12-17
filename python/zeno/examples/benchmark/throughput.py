"""
Throughput Benchmark

Measures maximum steps per second (SPS) for different configurations.
Benchmarks raw simulation throughput without rendering overhead.

Usage:
    python -m zeno.examples benchmark_throughput
    python -m zeno.examples benchmark_throughput --num-envs 4096
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulation throughput benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--benchmark-steps", type=int, default=1000, help="Benchmark steps")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def benchmark_throughput(env, actions: np.ndarray, num_steps: int) -> tuple:
    """Run benchmark and return timing statistics."""
    # Sync before timing
    obs = env.reset()

    start = time.perf_counter()
    for _ in range(num_steps):
        obs, rewards, dones, info = env.step(actions)
    elapsed = time.perf_counter() - start

    total_steps = num_steps * env.num_envs
    sps = total_steps / elapsed

    return elapsed, sps


def main():
    """Run throughput benchmark."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Throughput Benchmark")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Benchmark steps: {args.benchmark_steps}")
    print(f"Trials: {args.trials}")
    print()

    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Timestep: {env.timestep}s")
    print()

    # Pre-generate actions
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)

    # Warmup
    print("Warming up...")
    obs = env.reset()
    for _ in range(args.warmup_steps):
        obs, rewards, dones, info = env.step(actions)
    print()

    # Run benchmark trials
    print("Running benchmark...")
    print("-" * 60)

    results = []
    for trial in range(args.trials):
        elapsed, sps = benchmark_throughput(env, actions, args.benchmark_steps)
        results.append(sps)
        print(f"Trial {trial + 1}/{args.trials}: "
              f"{sps:>12,.0f} SPS | "
              f"{elapsed:.3f}s | "
              f"{args.benchmark_steps * args.num_envs:,} steps")

    # Statistics
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    mean_sps = np.mean(results)
    std_sps = np.std(results)
    max_sps = np.max(results)
    min_sps = np.min(results)

    print(f"Mean throughput:  {mean_sps:>12,.0f} SPS")
    print(f"Std throughput:   {std_sps:>12,.0f} SPS")
    print(f"Max throughput:   {max_sps:>12,.0f} SPS")
    print(f"Min throughput:   {min_sps:>12,.0f} SPS")
    print()

    # Derived metrics
    sim_time_per_sec = mean_sps * env.timestep
    print(f"Simulation speed: {sim_time_per_sec:>12.1f}x realtime")
    print(f"Steps per env/s:  {mean_sps / args.num_envs:>12,.0f}")

    env.close()


if __name__ == "__main__":
    main()
