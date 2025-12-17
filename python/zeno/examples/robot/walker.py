"""
Walker2D Example

Bipedal walker locomotion using Zeno physics engine.
The walker must coordinate leg joints to move forward without falling.

Usage:
    python -m zeno.examples robot_walker
    python -m zeno.examples robot_walker --num-envs 512
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walker2D locomotion example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "sine", "cpg"],
                        default="cpg", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def cpg_controller(step: int, num_envs: int, action_dim: int, timestep: float,
                   freq: float = 1.0, amplitude: float = 0.8) -> np.ndarray:
    """
    Central Pattern Generator for walking gait.
    Creates alternating leg movements for bipedal locomotion.
    """
    t = step * timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    for i in range(action_dim):
        # Alternate phase between legs
        phase = (i // 3) * np.pi  # Left/right leg phase offset
        joint_phase = (i % 3) * np.pi / 6  # Hip/knee/ankle offset
        actions[:, i] = amplitude * np.sin(2 * np.pi * freq * t + phase + joint_phase)

    return actions


def main():
    """Run the walker example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Walker2D Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("walker.xml")
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

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        elif args.policy == "sine":
            t = step * env.timestep
            actions = np.sin(2 * np.pi * 0.5 * t) * np.ones(env.action_shape, dtype=np.float32)
        else:  # cpg
            actions = cpg_controller(step, args.num_envs, env.action_dim, env.timestep)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        if (step + 1) % 200 == 0:
            current_x = env.get_body_positions()[:, 0, 0]
            forward_progress = (current_x - initial_x).mean()
            height = env.get_body_positions()[:, 0, 2].mean()
            print(f"Step {step + 1}/{args.steps} | "
                  f"Forward: {forward_progress:.2f}m | "
                  f"Height: {height:.2f}m | "
                  f"Reward: {total_reward.mean():.2f}")

    elapsed = time.perf_counter() - start_time
    final_x = env.get_body_positions()[:, 0, 0]
    forward_progress = (final_x - initial_x).mean()

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Forward progress: {forward_progress:.2f}m")
    print(f"Speed: {forward_progress / (args.steps * env.timestep):.2f} m/s")

    env.close()


if __name__ == "__main__":
    main()
