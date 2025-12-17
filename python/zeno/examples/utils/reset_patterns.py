"""
Reset Patterns Example

Demonstrates different reset strategies for vectorized environments.
Shows selective reset, auto-reset, and manual reset patterns.

Usage:
    python -m zeno.examples utils_reset_patterns
    python -m zeno.examples utils_reset_patterns --num-envs 128
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Environment reset patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run reset patterns example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Reset Patterns Example")
    print("=" * 50)
    print(f"Parallel environments: {args.num_envs}")
    print()

    mjcf_path = get_asset("cartpole.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)

    # Pattern 1: Full reset
    print("Pattern 1: Full reset (all environments)")
    print("-" * 40)
    obs = env.reset()
    print(f"  All {args.num_envs} environments reset")
    print(f"  Observation shape: {obs.shape}")
    print()

    # Pattern 2: Selective reset with mask
    print("Pattern 2: Selective reset (mask-based)")
    print("-" * 40)

    # Run some steps
    for _ in range(50):
        actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

    # Reset only environments that are done
    print(f"  Environments done: {dones.sum()}/{args.num_envs}")

    if dones.any():
        # Create reset mask
        reset_mask = dones.astype(np.uint8)
        obs = env.reset(mask=reset_mask)
        print(f"  Selectively reset {reset_mask.sum()} environments")
    print()

    # Pattern 3: Periodic reset
    print("Pattern 3: Periodic reset (every N steps)")
    print("-" * 40)

    reset_interval = 100
    reset_count = 0

    for step in range(args.steps):
        actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

        # Reset based on done flags
        if dones.any():
            reset_mask = dones.astype(np.uint8)
            obs = env.reset(mask=reset_mask)
            reset_count += reset_mask.sum()

        # Periodic full reset
        if (step + 1) % reset_interval == 0:
            obs = env.reset()
            reset_count += args.num_envs
            print(f"  Step {step + 1}: Periodic full reset")

    print(f"  Total resets: {reset_count}")
    print()

    # Pattern 4: Staggered reset
    print("Pattern 4: Staggered reset (rolling window)")
    print("-" * 40)

    episode_lengths = np.zeros(args.num_envs, dtype=int)
    max_episode_length = 100

    obs = env.reset()
    staggered_resets = 0

    for step in range(args.steps):
        actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

        episode_lengths += 1

        # Reset environments that exceed max episode length
        timeout_mask = episode_lengths >= max_episode_length
        reset_mask = (dones | timeout_mask).astype(np.uint8)

        if reset_mask.any():
            obs = env.reset(mask=reset_mask)
            episode_lengths[reset_mask.astype(bool)] = 0
            staggered_resets += reset_mask.sum()

    print(f"  Total staggered resets: {staggered_resets}")
    print(f"  Average episode length: {args.steps * args.num_envs / max(staggered_resets, 1):.1f}")
    print()

    print("=" * 50)
    print("Reset patterns demonstration complete!")

    env.close()


if __name__ == "__main__":
    main()
