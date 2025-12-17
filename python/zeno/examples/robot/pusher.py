"""
Pusher Example

Object manipulation task using Zeno physics engine.
A robotic arm must push an object to a target location.

Usage:
    python -m zeno.examples robot_pusher
    python -m zeno.examples robot_pusher --num-envs 256
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Object pushing manipulation example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "push"],
                        default="push", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def push_controller(step: int, num_envs: int, action_dim: int,
                    timestep: float) -> np.ndarray:
    """
    Simple periodic pushing motion.
    """
    t = step * timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Create pushing motion pattern
    phase = 2 * np.pi * 0.5 * t

    # Forward push followed by reset
    push_phase = np.sin(phase)
    for i in range(action_dim):
        if i == 0:
            actions[:, i] = 0.8 * push_phase  # Main push actuator
        else:
            actions[:, i] = 0.2 * np.sin(phase + i * 0.5)  # Support motion

    return actions


def main():
    """Run the pusher example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Pusher Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("cartpole.xml")
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

    # Track object (cart) position as proxy for pushed object
    initial_pos = env.get_body_positions()[:, 0, 0].copy()
    max_displacement = np.zeros(args.num_envs)
    total_reward = np.zeros(args.num_envs)

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            actions = push_controller(step, args.num_envs, env.action_dim, env.timestep)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Track displacement
        current_pos = env.get_body_positions()[:, 0, 0]
        displacement = np.abs(current_pos - initial_pos)
        max_displacement = np.maximum(max_displacement, displacement)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.steps} | "
                  f"Current displacement: {displacement.mean():.3f}m | "
                  f"Max displacement: {max_displacement.mean():.3f}m")

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Maximum push distance: {max_displacement.mean():.3f}m")

    env.close()


if __name__ == "__main__":
    main()
