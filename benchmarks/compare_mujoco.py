#!/usr/bin/env python3
"""
Benchmark comparing Zeno vs MuJoCo performance.

Usage:
    python compare_mujoco.py [--envs 1024] [--steps 1000]
"""

import argparse
import time
from typing import Optional

import numpy as np

# Try to import both libraries
try:
    import zeno
    HAS_ZENO = True
except ImportError:
    HAS_ZENO = False
    print("Warning: Zeno not installed")

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: MuJoCo not installed")


def benchmark_zeno(
    model_path: str,
    num_envs: int,
    num_steps: int,
) -> dict:
    """Benchmark Zeno simulation."""
    if not HAS_ZENO:
        return {"error": "Zeno not installed"}

    print(f"\n=== Zeno Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Environments: {num_envs}")
    print(f"Steps: {num_steps}")

    # Create environment
    start = time.perf_counter()
    env = zeno.make(model_path, num_envs=num_envs)
    create_time = time.perf_counter() - start

    print(f"Creation time: {create_time:.3f}s")
    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")

    # Warmup
    obs = env.reset()
    actions = np.random.uniform(-1, 1, (num_envs, env.action_dim)).astype(np.float32)
    for _ in range(10):
        env.step(actions)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        obs, rewards, dones, info = env.step(actions)
    elapsed = time.perf_counter() - start

    env.close()

    # Calculate metrics
    total_steps = num_envs * num_steps
    steps_per_sec = total_steps / elapsed
    time_per_step = elapsed / num_steps * 1000  # ms

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Steps/second: {steps_per_sec:,.0f}")
    print(f"  ms/step (all envs): {time_per_step:.3f}")
    print(f"  µs/step/env: {time_per_step * 1000 / num_envs:.3f}")

    return {
        "total_time": elapsed,
        "steps_per_sec": steps_per_sec,
        "ms_per_step": time_per_step,
        "us_per_step_env": time_per_step * 1000 / num_envs,
    }


def benchmark_mujoco(
    model_path: str,
    num_envs: int,
    num_steps: int,
) -> dict:
    """Benchmark MuJoCo simulation (sequential)."""
    if not HAS_MUJOCO:
        return {"error": "MuJoCo not installed"}

    print(f"\n=== MuJoCo Benchmark (Sequential) ===")
    print(f"Model: {model_path}")
    print(f"Environments: {num_envs}")
    print(f"Steps: {num_steps}")

    # Load model
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        return {"error": str(e)}

    # Create multiple data instances
    datas = [mujoco.MjData(model) for _ in range(num_envs)]

    print(f"Creation time: {len(datas)} environments created")
    print(f"nq: {model.nq}, nv: {model.nv}, nu: {model.nu}")

    # Generate random controls
    controls = np.random.uniform(-1, 1, (num_envs, model.nu)).astype(np.float32)

    # Warmup
    for data in datas:
        mujoco.mj_step(model, data)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        for i, data in enumerate(datas):
            data.ctrl[:] = controls[i]
            mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start

    # Calculate metrics
    total_steps = num_envs * num_steps
    steps_per_sec = total_steps / elapsed
    time_per_step = elapsed / num_steps * 1000  # ms

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Steps/second: {steps_per_sec:,.0f}")
    print(f"  ms/step (all envs): {time_per_step:.3f}")
    print(f"  µs/step/env: {time_per_step * 1000 / num_envs:.3f}")

    return {
        "total_time": elapsed,
        "steps_per_sec": steps_per_sec,
        "ms_per_step": time_per_step,
        "us_per_step_env": time_per_step * 1000 / num_envs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Zeno vs MuJoCo")
    parser.add_argument("--model", default="assets/ant.xml", help="Model file")
    parser.add_argument("--envs", type=int, default=1024, help="Number of environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    args = parser.parse_args()

    print("=" * 60)
    print("Zeno vs MuJoCo Performance Comparison")
    print("=" * 60)

    # Run benchmarks
    zeno_results = benchmark_zeno(args.model, args.envs, args.steps)
    mujoco_results = benchmark_mujoco(args.model, args.envs, args.steps)

    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    if "error" not in zeno_results and "error" not in mujoco_results:
        speedup = mujoco_results["total_time"] / zeno_results["total_time"]
        print(f"\nZeno speedup: {speedup:.1f}x faster than MuJoCo")
        print(f"\nZeno:   {zeno_results['steps_per_sec']:>12,.0f} steps/sec")
        print(f"MuJoCo: {mujoco_results['steps_per_sec']:>12,.0f} steps/sec")
    else:
        if "error" in zeno_results:
            print(f"Zeno error: {zeno_results['error']}")
        if "error" in mujoco_results:
            print(f"MuJoCo error: {mujoco_results['error']}")


if __name__ == "__main__":
    main()
