"""
Curriculum Learning Example

Demonstrates curriculum learning by progressively increasing task difficulty.
Shows adaptive difficulty adjustment based on agent performance.

Usage:
    python -m zeno.examples training_curriculum
    python -m zeno.examples training_curriculum --num-envs 512
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

import zeno
from zeno.examples import get_asset


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
    difficulty_increment: float = 0.05
    success_threshold: float = 0.7
    window_size: int = 100


class AdaptivePolicy:
    """Policy that adapts based on difficulty level."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0, 0.1, (obs_dim, action_dim)).astype(np.float32)
        self.b = np.zeros(action_dim, dtype=np.float32)
        self.difficulty_scale = 1.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Scale actions based on difficulty
        base_action = np.tanh(obs @ self.w + self.b)
        return np.clip(base_action * self.difficulty_scale, -1, 1)

    def update_difficulty(self, difficulty: float):
        """Adjust policy behavior based on current difficulty."""
        self.difficulty_scale = 0.5 + 0.5 * difficulty  # Scale from 0.5 to 1.0


def compute_success_rate(rewards: list, threshold: float = 0.0) -> float:
    """Compute success rate from recent rewards."""
    if not rewards:
        return 0.0
    successes = sum(1 for r in rewards if r > threshold)
    return successes / len(rewards)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curriculum learning example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="cartpole", choices=["ant", "humanoid", "cartpole"])
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--episode-length", type=int, default=200, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run curriculum learning example."""
    parser = create_parser()
    args = parser.parse_args()
    config = CurriculumConfig()

    print("Zeno Curriculum Learning")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Episodes: {args.episodes}")
    print(f"Initial difficulty: {config.initial_difficulty}")
    print()

    # Create environment
    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Create policy
    policy = AdaptivePolicy(env.observation_dim, env.action_dim, args.seed)

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    # Curriculum state
    current_difficulty = config.initial_difficulty
    episode_rewards = []
    difficulty_history = []

    start_time = time.perf_counter()

    for episode in range(args.episodes):
        ep_start = time.perf_counter()

        # Update policy with current difficulty
        policy.update_difficulty(current_difficulty)

        # Run episode
        obs = env.reset()
        episode_reward = np.zeros(args.num_envs)

        for step in range(args.episode_length):
            # Add difficulty-based noise to make task harder
            noise_scale = current_difficulty * 0.1
            actions = policy(obs)
            actions += np.random.normal(0, noise_scale, actions.shape).astype(np.float32)
            actions = np.clip(actions, -1, 1)

            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards

        mean_reward = episode_reward.mean()
        episode_rewards.append(mean_reward)
        difficulty_history.append(current_difficulty)

        # Update difficulty based on recent performance
        if len(episode_rewards) >= config.window_size:
            recent_rewards = episode_rewards[-config.window_size:]
            success_rate = compute_success_rate(recent_rewards, threshold=np.median(recent_rewards))

            if success_rate > config.success_threshold and current_difficulty < config.max_difficulty:
                current_difficulty = min(current_difficulty + config.difficulty_increment, config.max_difficulty)
                print(f"  -> Difficulty increased to {current_difficulty:.2f}")

        ep_time = time.perf_counter() - ep_start

        if (episode + 1) % 50 == 0:
            recent_mean = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else mean_reward
            print(f"Episode {episode + 1:>3}/{args.episodes} | "
                  f"Reward: {mean_reward:>8.2f} | "
                  f"Avg(50): {recent_mean:>8.2f} | "
                  f"Difficulty: {current_difficulty:.2f}")

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 50)
    print("Curriculum Training Complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Final difficulty: {current_difficulty:.2f}")
    print(f"Mean reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
