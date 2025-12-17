"""
Imitation Learning Example

Demonstrates behavioral cloning from expert demonstrations.
Shows data collection and supervised learning for policy training.

Usage:
    python -m zeno.examples training_imitation
    python -m zeno.examples training_imitation --num-envs 64
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

import zeno
from zeno.examples import get_asset


@dataclass
class DemonstrationBuffer:
    """Buffer for storing expert demonstrations."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def add(self, obs: np.ndarray, action: np.ndarray):
        self.observations.append(obs.copy())
        self.actions.append(action.copy())

    def get_dataset(self) -> tuple:
        """Convert to numpy arrays."""
        return (
            np.concatenate(self.observations, axis=0),
            np.concatenate(self.actions, axis=0),
        )

    def __len__(self):
        return len(self.observations)


class ExpertPolicy:
    """Simple expert policy using sinusoidal control."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.step = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.step += 1
        num_envs = obs.shape[0]
        t = self.step * 0.02
        actions = np.zeros((num_envs, self.action_dim), dtype=np.float32)
        for i in range(self.action_dim):
            phase = i * np.pi / 4
            actions[:, i] = 0.7 * np.sin(2 * np.pi * 0.5 * t + phase)
        return actions


class LearnedPolicy:
    """Policy learned from demonstrations via behavioral cloning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, 0.1, (obs_dim, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = self.rng.normal(0, 0.1, (hidden_size, action_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)
        self.learning_rate = 0.001

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(obs @ self.w1 + self.b1)
        return np.clip(np.tanh(h @ self.w2 + self.b2), -1, 1)

    def train_step(self, obs: np.ndarray, expert_actions: np.ndarray) -> float:
        """Single training step with gradient descent."""
        # Forward pass
        h = np.tanh(obs @ self.w1 + self.b1)
        pred_actions = np.tanh(h @ self.w2 + self.b2)

        # Compute loss
        loss = np.mean((pred_actions - expert_actions) ** 2)

        # Backward pass (simplified gradient computation)
        d_output = 2 * (pred_actions - expert_actions) / obs.shape[0]
        d_output = d_output * (1 - pred_actions ** 2)  # tanh derivative

        d_w2 = h.T @ d_output
        d_b2 = d_output.sum(axis=0)

        d_h = d_output @ self.w2.T
        d_h = d_h * (1 - h ** 2)  # tanh derivative

        d_w1 = obs.T @ d_h
        d_b1 = d_h.sum(axis=0)

        # Update weights
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2

        return loss


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Imitation learning example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole"])
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--demo-episodes", type=int, default=50, help="Demonstration episodes")
    parser.add_argument("--episode-length", type=int, default=200, help="Steps per episode")
    parser.add_argument("--train-epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run imitation learning example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Imitation Learning")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Demonstration episodes: {args.demo_episodes}")
    print(f"Training epochs: {args.train_epochs}")
    print()

    # Create environment
    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    # Phase 1: Collect demonstrations
    print("Phase 1: Collecting expert demonstrations...")
    expert = ExpertPolicy(env.action_dim)
    demo_buffer = DemonstrationBuffer()
    expert_rewards = []

    for ep in range(args.demo_episodes):
        obs = env.reset()
        episode_reward = np.zeros(args.num_envs)

        for step in range(args.episode_length):
            actions = expert(obs)
            demo_buffer.add(obs, actions)

            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards

        expert_rewards.append(episode_reward.mean())

        if (ep + 1) % 10 == 0:
            print(f"  Collected {ep + 1}/{args.demo_episodes} episodes | "
                  f"Reward: {episode_reward.mean():.2f}")

    demo_obs, demo_actions = demo_buffer.get_dataset()
    print(f"  Total demonstrations: {len(demo_obs):,} samples")
    print(f"  Expert mean reward: {np.mean(expert_rewards):.2f}")
    print()

    # Phase 2: Train policy
    print("Phase 2: Training policy via behavioral cloning...")
    policy = LearnedPolicy(env.observation_dim, env.action_dim, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.train_epochs):
        # Shuffle data
        indices = rng.permutation(len(demo_obs))
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            batch_obs = demo_obs[batch_idx]
            batch_actions = demo_actions[batch_idx]

            loss = policy.train_step(batch_obs, batch_actions)
            epoch_loss += loss
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{args.train_epochs} | "
                  f"Loss: {epoch_loss / num_batches:.6f}")

    print()

    # Phase 3: Evaluate learned policy
    print("Phase 3: Evaluating learned policy...")
    learned_rewards = []

    for ep in range(10):
        obs = env.reset()
        episode_reward = np.zeros(args.num_envs)

        for step in range(args.episode_length):
            actions = policy(obs)
            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards

        learned_rewards.append(episode_reward.mean())

    print()
    print("=" * 50)
    print("Imitation Learning Complete!")
    print(f"Expert mean reward: {np.mean(expert_rewards):.2f}")
    print(f"Learned policy reward: {np.mean(learned_rewards):.2f}")
    print(f"Performance ratio: {np.mean(learned_rewards) / np.mean(expert_rewards) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
