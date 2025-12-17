"""
Shapes Example

Demonstrates collision shapes and their physical properties.
Shows spheres, boxes, capsules interacting with the ground plane.

Usage:
    python -m zeno.examples basic_shapes
    python -m zeno.examples basic_shapes --num-envs 256
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collision shapes demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run the shapes example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Shapes Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print()

    # Pendulum has capsule and sphere shapes
    mjcf_path = get_asset("pendulum.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print("Shape types in model: capsule (pole), sphere (bob)")
    print(f"Observation dim: {env.observation_dim}")
    print()

    obs = env.reset()

    start_time = time.perf_counter()

    # Track body positions to observe shape interactions
    initial_positions = env.get_body_positions().copy()

    for step in range(args.steps):
        # No control - let shapes interact with gravity
        actions = np.zeros(env.action_shape, dtype=np.float32)
        obs, rewards, dones, info = env.step(actions)

        if (step + 1) % 100 == 0:
            positions = env.get_body_positions()
            displacement = np.linalg.norm(positions - initial_positions, axis=-1).mean()
            print(f"Step {step + 1}/{args.steps} | Mean displacement: {displacement:.3f}m")

    elapsed = time.perf_counter() - start_time
    print()
    print(f"Simulation complete in {elapsed:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
