"""
Reacher Example

Robotic arm reaching task using Zeno physics engine.
The arm must reach target positions in 3D space.

Usage:
    python -m zeno.examples robot_reacher
    python -m zeno.examples robot_reacher --num-envs 256
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robotic arm reaching example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--policy", type=str, choices=["random", "pd"],
                        default="pd", help="Control policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def pd_reaching_controller(obs: np.ndarray, target: np.ndarray,
                           action_dim: int, kp: float = 5.0) -> np.ndarray:
    """
    Simple proportional controller for reaching.
    """
    num_envs = obs.shape[0]
    obs_dim = obs.shape[1]

    # Use observation components as proxy for end-effector position
    # Handle different observation dimensions
    target_dim = min(obs_dim, target.shape[1])
    current_pos = obs[:, :target_dim]
    target_clipped = target[:, :target_dim]

    # Compute error
    error = target_clipped - current_pos

    # Map error to joint actions (simplified)
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    for i in range(action_dim):
        error_idx = i % error.shape[1]
        actions[:, i] = np.clip(kp * error[:, error_idx], -1, 1)

    return actions


def main():
    """Run the reacher example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Reacher Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    mjcf_path = get_asset("pendulum.xml")
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

    # Generate random target positions
    target = rng.uniform(-1, 1, (args.num_envs, 3)).astype(np.float32)
    target[:, 2] = np.abs(target[:, 2]) + 0.5  # Keep targets above ground

    total_reward = np.zeros(args.num_envs)
    min_distance = np.full(args.num_envs, np.inf)

    start_time = time.perf_counter()

    for step in range(args.steps):
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:
            actions = pd_reaching_controller(obs, target, env.action_dim)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Compute distance to target
        end_effector = env.get_body_positions()[:, -1, :3]  # Last body as end-effector
        distance = np.linalg.norm(end_effector - target, axis=1)
        min_distance = np.minimum(min_distance, distance)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.steps} | "
                  f"Mean distance: {distance.mean():.3f}m | "
                  f"Min achieved: {min_distance.mean():.3f}m")

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Closest approach: {min_distance.mean():.3f}m")

    env.close()


if __name__ == "__main__":
    main()
