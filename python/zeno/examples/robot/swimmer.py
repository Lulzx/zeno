"""
Swimmer Example

Multi-segment swimmer using Zeno physics engine.
Demonstrates undulating locomotion through coordinated joint movements.

Usage:
    python -m zeno.examples robot_swimmer
    python -m zeno.examples robot_swimmer --num-envs 512
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Swimmer locomotion example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "wave"],
                        default="wave", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def wave_controller(step: int, num_envs: int, action_dim: int,
                    timestep: float, freq: float = 1.5, wavelength: float = 2.0) -> np.ndarray:
    """
    Traveling wave controller for swimmer.
    Creates a sinusoidal wave that propagates along the body.
    """
    t = step * timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    for i in range(action_dim):
        # Phase offset increases along the body to create traveling wave
        phase_offset = i * (2 * np.pi / wavelength)
        actions[:, i] = 0.8 * np.sin(2 * np.pi * freq * t - phase_offset)

    return actions


def main():
    """Run the swimmer example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Swimmer Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("swimmer.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Segments (action dim): {env.action_dim}")
    print(f"Observation dim: {env.observation_dim}")
    print()

    obs = env.reset()
    rng = np.random.default_rng(args.seed)

    initial_pos = env.get_body_positions()[:, 0, :2].copy()
    total_reward = np.zeros(args.num_envs)

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            actions = wave_controller(step, args.num_envs, env.action_dim, env.timestep)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        if (step + 1) % 200 == 0:
            current_pos = env.get_body_positions()[:, 0, :2]
            displacement = np.linalg.norm(current_pos - initial_pos, axis=1).mean()
            print(f"Step {step + 1}/{args.steps} | "
                  f"Distance traveled: {displacement:.2f}m | "
                  f"Reward: {total_reward.mean():.2f}")

    elapsed = time.perf_counter() - start_time
    final_pos = env.get_body_positions()[:, 0, :2]
    displacement = np.linalg.norm(final_pos - initial_pos, axis=1).mean()

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Total distance: {displacement:.2f}m")
    print(f"Speed: {displacement / (args.steps * env.timestep):.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
