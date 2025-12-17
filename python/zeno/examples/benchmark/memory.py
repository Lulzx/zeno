"""
Memory Benchmark

Measures memory usage for different environment configurations.
Helps optimize batch sizes for available GPU/system memory.

Usage:
    python -m zeno.examples benchmark_memory
    python -m zeno.examples benchmark_memory --max-envs 8192
"""

import argparse
import gc
import sys

import numpy as np

import zeno
from zeno.examples import get_asset


def get_process_memory() -> int:
    """Get current process memory usage in bytes."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss * 1024  # Convert to bytes on macOS
    except ImportError:
        return 0


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory usage benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--min-envs", type=int, default=1, help="Minimum environments")
    parser.add_argument("--max-envs", type=int, default=4096, help="Maximum environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def measure_env_memory(mjcf_path: str, num_envs: int, seed: int) -> dict:
    """Measure memory usage for a given environment configuration."""
    gc.collect()
    base_memory = get_process_memory()

    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=num_envs,
        seed=seed,
    )

    # Get state arrays to ensure memory is allocated
    obs = env.reset()
    positions = env.get_body_positions()
    quaternions = env.get_body_quaternions()

    after_memory = get_process_memory()
    env_memory = after_memory - base_memory

    # Calculate array sizes
    obs_size = obs.nbytes
    pos_size = positions.nbytes
    quat_size = quaternions.nbytes
    state_size = obs_size + pos_size + quat_size

    result = {
        "num_envs": num_envs,
        "total_memory": env_memory,
        "obs_size": obs_size,
        "pos_size": pos_size,
        "quat_size": quat_size,
        "state_size": state_size,
        "per_env": env_memory / num_envs if num_envs > 0 else 0,
    }

    env.close()
    gc.collect()

    return result


def main():
    """Run memory benchmark."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Memory Benchmark")
    print("=" * 70)
    print(f"Environment: {args.env}")
    print(f"Environment range: {args.min_envs} - {args.max_envs}")
    print()

    mjcf_path = get_asset(f"{args.env}.xml")

    # Test environment counts (powers of 2)
    env_counts = []
    n = args.min_envs
    while n <= args.max_envs:
        env_counts.append(n)
        n *= 2

    print("Measuring memory usage...")
    print("-" * 70)
    print(f"{'Envs':>8} | {'Total Memory':>12} | {'Per Env':>10} | {'State Arrays':>12} | {'Overhead':>10}")
    print("-" * 70)

    results = []

    for num_envs in env_counts:
        try:
            result = measure_env_memory(mjcf_path, num_envs, args.seed)
            results.append(result)

            overhead = result["total_memory"] - result["state_size"]

            print(f"{num_envs:>8} | "
                  f"{format_bytes(result['total_memory']):>12} | "
                  f"{format_bytes(result['per_env']):>10} | "
                  f"{format_bytes(result['state_size']):>12} | "
                  f"{format_bytes(max(0, overhead)):>10}")
        except Exception as e:
            print(f"{num_envs:>8} | FAILED: {str(e)[:40]}")

    # Summary
    print()
    print("=" * 70)
    print("Memory Analysis")
    print("=" * 70)

    if len(results) >= 2:
        # Calculate memory scaling
        first = results[0]
        last = results[-1]

        env_ratio = last["num_envs"] / first["num_envs"]
        mem_ratio = last["total_memory"] / max(first["total_memory"], 1)

        print(f"Scaling: {first['num_envs']} -> {last['num_envs']} envs")
        print(f"  Environment increase: {env_ratio:.0f}x")
        print(f"  Memory increase: {mem_ratio:.1f}x")
        print(f"  Scaling efficiency: {(env_ratio / mem_ratio) * 100:.1f}%")
        print()

        # Estimate maximum environments
        available_memory = 8 * 1024 * 1024 * 1024  # Assume 8GB
        per_env_memory = last["per_env"]
        max_envs = int(available_memory / per_env_memory) if per_env_memory > 0 else 0

        print(f"Estimated capacity (8GB):")
        print(f"  Per-env memory: {format_bytes(per_env_memory)}")
        print(f"  Max environments: ~{max_envs:,}")

    # Breakdown for largest config
    if results:
        last = results[-1]
        print()
        print(f"Memory breakdown ({last['num_envs']} envs):")
        print(f"  Observations: {format_bytes(last['obs_size'])}")
        print(f"  Positions: {format_bytes(last['pos_size'])}")
        print(f"  Quaternions: {format_bytes(last['quat_size'])}")
        print(f"  Total state: {format_bytes(last['state_size'])}")


if __name__ == "__main__":
    main()
