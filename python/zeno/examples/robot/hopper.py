"""
Hopper Example

Single-legged hopping robot using Zeno physics engine.
The hopper must maintain balance while moving forward.

Usage:
    python -m zeno.examples robot_hopper
    python -m zeno.examples robot_hopper --num-envs 256
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hopper locomotion example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "hop"],
                        default="hop", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def hopping_controller(step: int, num_envs: int, action_dim: int,
                       timestep: float, freq: float = 2.0) -> np.ndarray:
    """
    Simple hopping controller using periodic leg extension.
    """
    t = step * timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Create hopping pattern
    phase = 2 * np.pi * freq * t
    hop_signal = np.sin(phase)

    # Apply to all actuators with slight variations
    for i in range(action_dim):
        offset = i * 0.1
        actions[:, i] = 0.7 * np.sin(phase + offset)

    return actions


def main():
    """Run the hopper example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Hopper Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("hopper.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    obs = env.reset()
    rng = np.random.default_rng(args.seed)

    max_height = np.zeros(args.num_envs)
    total_reward = np.zeros(args.num_envs)

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            actions = hopping_controller(step, args.num_envs, env.action_dim, env.timestep)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Track maximum height achieved
        heights = env.get_body_positions()[:, :, 2].max(axis=1)
        max_height = np.maximum(max_height, heights)

        if (step + 1) % 200 == 0:
            current_height = heights.mean()
            print(f"Step {step + 1}/{args.steps} | "
                  f"Current height: {current_height:.2f}m | "
                  f"Max height: {max_height.mean():.2f}m")

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Maximum height achieved: {max_height.mean():.2f}m")

    env.close()


if __name__ == "__main__":
    main()
