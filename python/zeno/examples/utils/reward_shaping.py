"""
Reward Shaping Example

Demonstrates custom reward functions and reward shaping techniques.
Shows how to design rewards for different objectives.

Usage:
    python -m zeno.examples utils_reward_shaping
    python -m zeno.examples utils_reward_shaping --num-envs 256
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Custom reward shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


class RewardShaper:
    """Custom reward shaper with multiple objectives."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.prev_positions = None
        self.prev_heights = None

    def compute_forward_reward(self, positions: np.ndarray, timestep: float) -> np.ndarray:
        """Reward for forward progress."""
        current_x = positions[:, 0, 0]
        if self.prev_positions is None:
            self.prev_positions = current_x.copy()
            return np.zeros(self.num_envs)

        velocity = (current_x - self.prev_positions) / timestep
        self.prev_positions = current_x.copy()
        return velocity  # Reward proportional to forward velocity

    def compute_height_reward(self, positions: np.ndarray, target_height: float = 0.5) -> np.ndarray:
        """Reward for maintaining target height."""
        current_height = positions[:, 0, 2]
        height_error = np.abs(current_height - target_height)
        return np.exp(-height_error)  # Gaussian-like reward

    def compute_stability_reward(self, quaternions: np.ndarray) -> np.ndarray:
        """Reward for maintaining upright orientation."""
        # Measure deviation from upright (z-axis aligned with world z)
        w = quaternions[:, 0, 3]
        z = quaternions[:, 0, 2]
        # Approximate uprightness from quaternion
        uprightness = 2 * (w * w + z * z) - 1
        return np.clip(uprightness, 0, 1)

    def compute_energy_penalty(self, actions: np.ndarray) -> np.ndarray:
        """Penalty for high energy usage."""
        energy = np.sum(actions ** 2, axis=1)
        return -0.01 * energy  # Small penalty

    def compute_alive_bonus(self, positions: np.ndarray, min_height: float = 0.2) -> np.ndarray:
        """Bonus for staying alive (not falling)."""
        heights = positions[:, 0, 2]
        alive = heights > min_height
        return alive.astype(np.float32)


def main():
    """Run reward shaping example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Reward Shaping Example")
    print("=" * 50)
    print(f"Parallel environments: {args.num_envs}")
    print()

    mjcf_path = get_asset("ant.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    shaper = RewardShaper(args.num_envs)

    # Track different reward components
    reward_components = {
        "forward": [],
        "height": [],
        "stability": [],
        "energy": [],
        "alive": [],
        "total": [],
    }

    obs = env.reset()

    print("Running simulation with shaped rewards...")
    print("-" * 50)

    for step in range(args.steps):
        # Generate actions
        t = step * env.timestep
        actions = np.zeros(env.action_shape, dtype=np.float32)
        for i in range(env.action_dim):
            phase = i * np.pi / 4
            actions[:, i] = 0.5 * np.sin(2 * np.pi * 1.0 * t + phase)

        obs, base_rewards, dones, info = env.step(actions)

        # Get state for reward computation
        positions = env.get_body_positions()
        quaternions = env.get_body_quaternions()

        # Compute reward components
        r_forward = shaper.compute_forward_reward(positions, env.timestep)
        r_height = shaper.compute_height_reward(positions)
        r_stability = shaper.compute_stability_reward(quaternions)
        r_energy = shaper.compute_energy_penalty(actions)
        r_alive = shaper.compute_alive_bonus(positions)

        # Combine rewards with weights
        total_reward = (
            1.0 * r_forward +
            0.5 * r_height +
            0.3 * r_stability +
            1.0 * r_energy +
            0.1 * r_alive
        )

        # Track statistics
        reward_components["forward"].append(r_forward.mean())
        reward_components["height"].append(r_height.mean())
        reward_components["stability"].append(r_stability.mean())
        reward_components["energy"].append(r_energy.mean())
        reward_components["alive"].append(r_alive.mean())
        reward_components["total"].append(total_reward.mean())

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: "
                  f"Forward={r_forward.mean():.3f}, "
                  f"Height={r_height.mean():.3f}, "
                  f"Stability={r_stability.mean():.3f}, "
                  f"Total={total_reward.mean():.3f}")

    # Print summary
    print()
    print("=" * 50)
    print("Reward Component Summary")
    print("=" * 50)

    for name, values in reward_components.items():
        mean = np.mean(values)
        std = np.std(values)
        total = np.sum(values)
        print(f"{name:>12}: mean={mean:>8.4f}, std={std:>8.4f}, total={total:>10.2f}")

    env.close()


if __name__ == "__main__":
    main()
