"""
HalfCheetah Example

Fast running quadruped using Zeno physics engine.
Optimized for forward velocity with bounding gait.

Usage:
    python -m zeno.examples robot_cheetah
    python -m zeno.examples robot_cheetah --num-envs 1024
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HalfCheetah running example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "gallop"],
                        default="gallop", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def gallop_controller(step: int, num_envs: int, action_dim: int,
                      timestep: float, freq: float = 3.0) -> np.ndarray:
    """
    Galloping gait controller for fast quadruped locomotion.
    Front and back legs move in phase pairs.
    """
    t = step * timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    phase = 2 * np.pi * freq * t

    for i in range(action_dim):
        # Divide actuators into front/back leg groups
        leg_group = i // (action_dim // 2)  # 0 = front, 1 = back
        joint_idx = i % (action_dim // 2)

        # Front and back legs are 180 degrees out of phase (bounding)
        leg_phase = leg_group * np.pi
        joint_offset = joint_idx * 0.3

        actions[:, i] = 0.9 * np.sin(phase + leg_phase + joint_offset)

    return actions


def main():
    """Run the cheetah example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno HalfCheetah Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("cheetah.xml")
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

    initial_x = env.get_body_positions()[:, 0, 0].copy()
    total_reward = np.zeros(args.num_envs)
    max_speed = np.zeros(args.num_envs)
    prev_x = initial_x.copy()

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            actions = gallop_controller(step, args.num_envs, env.action_dim, env.timestep)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Calculate instantaneous velocity
        current_x = env.get_body_positions()[:, 0, 0]
        velocity = (current_x - prev_x) / env.timestep
        max_speed = np.maximum(max_speed, np.abs(velocity))
        prev_x = current_x.copy()

        if (step + 1) % 200 == 0:
            forward_progress = (current_x - initial_x).mean()
            avg_speed = forward_progress / ((step + 1) * env.timestep)
            print(f"Step {step + 1}/{args.steps} | "
                  f"Distance: {forward_progress:.2f}m | "
                  f"Avg speed: {avg_speed:.2f} m/s | "
                  f"Max speed: {max_speed.mean():.2f} m/s")

    elapsed = time.perf_counter() - start_time
    final_x = env.get_body_positions()[:, 0, 0]
    forward_progress = (final_x - initial_x).mean()

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Total distance: {forward_progress:.2f}m")
    print(f"Average speed: {forward_progress / (args.steps * env.timestep):.2f} m/s")
    print(f"Peak speed: {max_speed.mean():.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
