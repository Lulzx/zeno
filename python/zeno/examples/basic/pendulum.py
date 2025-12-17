"""
Pendulum Example

Demonstrates a simple pendulum simulation using the Zeno physics engine.
The pendulum swings freely under gravity with optional control torque.

Usage:
    python -m zeno.examples pendulum
    python -m zeno.examples pendulum --num-envs 16 --steps 500
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for pendulum example."""
    parser = argparse.ArgumentParser(
        description="Pendulum simulation example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Apply random control torques",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def main():
    """Run the pendulum example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Pendulum Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Control: {'enabled' if args.control else 'disabled'}")
    print()

    # Create environment
    mjcf_path = get_asset("pendulum.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Timestep: {env.timestep}s")
    print()

    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Run simulation
    rng = np.random.default_rng(args.seed)
    total_reward = np.zeros(args.num_envs)

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Generate actions
        if args.control:
            # Random control torque
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            # No control (free swing)
            actions = np.zeros(env.action_shape, dtype=np.float32)

        # Step simulation
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) * args.num_envs / elapsed
            print(f"Step {step + 1}/{args.steps} | "
                  f"SPS: {sps:.0f} | "
                  f"Mean reward: {total_reward.mean() / (step + 1):.4f}")

    elapsed = time.perf_counter() - start_time

    print()
    print("Simulation complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Steps per second: {args.steps * args.num_envs / elapsed:.0f}")
    print(f"Mean total reward: {total_reward.mean():.4f}")

    env.close()


if __name__ == "__main__":
    main()
