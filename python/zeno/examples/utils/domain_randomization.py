"""
Domain Randomization Example

Demonstrates domain randomization techniques for sim-to-real transfer.
Shows observation noise, action delays, and parameter variation.

Usage:
    python -m zeno.examples utils_domain_randomization
    python -m zeno.examples utils_domain_randomization --num-envs 256
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

import zeno
from zeno.examples import get_asset


@dataclass
class RandomizationConfig:
    """Domain randomization parameters."""
    obs_noise_scale: float = 0.02
    action_noise_scale: float = 0.05
    action_delay_prob: float = 0.1
    action_delay_steps: int = 2
    gravity_range: tuple = (0.9, 1.1)
    friction_range: tuple = (0.8, 1.2)
    mass_range: tuple = (0.9, 1.1)


class DomainRandomizer:
    """Apply domain randomization to simulation."""

    def __init__(self, config: RandomizationConfig, num_envs: int, action_dim: int, seed: int = 42):
        self.config = config
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

        # Per-environment randomization factors
        self.gravity_scale = self.rng.uniform(*config.gravity_range, num_envs)
        self.friction_scale = self.rng.uniform(*config.friction_range, num_envs)
        self.mass_scale = self.rng.uniform(*config.mass_range, num_envs)

        # Action delay buffer
        self.action_buffer = []
        for _ in range(config.action_delay_steps + 1):
            self.action_buffer.append(np.zeros((num_envs, action_dim), dtype=np.float32))

    def randomize_observations(self, obs: np.ndarray) -> np.ndarray:
        """Add noise to observations."""
        noise = self.rng.normal(0, self.config.obs_noise_scale, obs.shape).astype(np.float32)
        return obs + noise

    def randomize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Apply action noise and delays."""
        # Add noise
        noise = self.rng.normal(0, self.config.action_noise_scale, actions.shape).astype(np.float32)
        noisy_actions = actions + noise

        # Apply delays
        self.action_buffer.append(noisy_actions.copy())
        self.action_buffer.pop(0)

        # Randomly select delayed or current action per environment
        use_delay = self.rng.random(self.num_envs) < self.config.action_delay_prob
        delay_idx = self.rng.integers(0, self.config.action_delay_steps + 1, self.num_envs)

        output_actions = noisy_actions.copy()
        for i in range(self.num_envs):
            if use_delay[i]:
                output_actions[i] = self.action_buffer[delay_idx[i]][i]

        return np.clip(output_actions, -1, 1)

    def get_randomization_info(self) -> dict:
        """Get current randomization factors."""
        return {
            "gravity_scale": self.gravity_scale.copy(),
            "friction_scale": self.friction_scale.copy(),
            "mass_scale": self.mass_scale.copy(),
        }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Domain randomization example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
    parser.add_argument("--obs-noise", type=float, default=0.02, help="Observation noise scale")
    parser.add_argument("--action-noise", type=float, default=0.05, help="Action noise scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run domain randomization example."""
    parser = create_parser()
    args = parser.parse_args()

    config = RandomizationConfig(
        obs_noise_scale=args.obs_noise,
        action_noise_scale=args.action_noise,
    )

    print("Zeno Domain Randomization Example")
    print("=" * 50)
    print(f"Parallel environments: {args.num_envs}")
    print(f"Observation noise: {config.obs_noise_scale}")
    print(f"Action noise: {config.action_noise_scale}")
    print(f"Action delay prob: {config.action_delay_prob}")
    print()

    mjcf_path = get_asset("ant.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    randomizer = DomainRandomizer(config, args.num_envs, env.action_dim, args.seed)
    rng = np.random.default_rng(args.seed)

    # Print randomization factors
    rand_info = randomizer.get_randomization_info()
    print("Per-Environment Randomization:")
    print(f"  Gravity scale: [{rand_info['gravity_scale'].min():.3f}, {rand_info['gravity_scale'].max():.3f}]")
    print(f"  Friction scale: [{rand_info['friction_scale'].min():.3f}, {rand_info['friction_scale'].max():.3f}]")
    print(f"  Mass scale: [{rand_info['mass_scale'].min():.3f}, {rand_info['mass_scale'].max():.3f}]")
    print()

    obs = env.reset()
    total_reward = np.zeros(args.num_envs)

    # Track effect of randomization
    obs_differences = []
    action_differences = []

    print("Running with domain randomization...")
    print("-" * 50)

    for step in range(args.steps):
        # Generate base actions
        t = step * env.timestep
        base_actions = np.zeros(env.action_shape, dtype=np.float32)
        for i in range(env.action_dim):
            base_actions[:, i] = 0.5 * np.sin(2 * np.pi * 0.5 * t + i * 0.5)

        # Apply randomization
        noisy_obs = randomizer.randomize_observations(obs)
        noisy_actions = randomizer.randomize_actions(base_actions)

        # Track differences
        obs_diff = np.abs(noisy_obs - obs).mean()
        action_diff = np.abs(noisy_actions - base_actions).mean()
        obs_differences.append(obs_diff)
        action_differences.append(action_diff)

        # Step environment with randomized actions
        obs, rewards, dones, info = env.step(noisy_actions)
        total_reward += rewards

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: "
                  f"Obs noise={obs_diff:.4f}, "
                  f"Action noise={action_diff:.4f}, "
                  f"Reward={rewards.mean():.3f}")

    print()
    print("=" * 50)
    print("Domain Randomization Summary")
    print("=" * 50)
    print(f"Mean observation noise: {np.mean(obs_differences):.4f}")
    print(f"Mean action noise: {np.mean(action_differences):.4f}")
    print(f"Total reward: {total_reward.mean():.2f}")
    print(f"Reward std (across envs): {total_reward.std():.2f}")

    env.close()


if __name__ == "__main__":
    main()
