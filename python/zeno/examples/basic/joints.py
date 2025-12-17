"""
Joints Example

Demonstrates different joint types and their behavior in Zeno.
Shows how joints constrain motion between bodies.

Usage:
    python -m zeno.examples basic_joints
    python -m zeno.examples basic_joints --num-envs 64
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint types demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run the joints example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Joints Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print()

    # Use pendulum which has hinge joints
    mjcf_path = get_asset("pendulum.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Model loaded with {env.action_dim} actuated joints")
    print(f"Observation dim: {env.observation_dim}")
    print()

    obs = env.reset()
    rng = np.random.default_rng(args.seed)

    start_time = time.perf_counter()

    # Apply sinusoidal torques to demonstrate joint motion
    for step in range(args.steps):
        t = step * env.timestep
        # Sinusoidal control signal
        actions = np.sin(2 * np.pi * 0.5 * t) * np.ones(env.action_shape, dtype=np.float32)

        obs, rewards, dones, info = env.step(actions)

        if (step + 1) % 100 == 0:
            positions = env.get_body_positions()
            mean_height = positions[:, :, 2].mean()
            print(f"Step {step + 1}/{args.steps} | Mean body height: {mean_height:.3f}m")

    elapsed = time.perf_counter() - start_time
    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Steps per second: {args.steps * args.num_envs / elapsed:.0f}")

    env.close()


if __name__ == "__main__":
    main()
