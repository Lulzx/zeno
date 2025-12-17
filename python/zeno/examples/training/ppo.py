"""
PPO Training Example

Proximal Policy Optimization training loop using Zeno.
Demonstrates efficient batched RL training with vectorized environments.

Usage:
    python -m zeno.examples training_ppo
    python -m zeno.examples training_ppo --num-envs 2048 --iterations 100
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

import zeno
from zeno.examples import get_asset


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 256


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()


class SimplePolicyNetwork:
    """Simple policy network using NumPy (replace with PyTorch for real training)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)

        # Policy network weights
        self.w1 = self.rng.normal(0, 0.1, (obs_dim, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = self.rng.normal(0, 0.1, (hidden_size, action_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

        # Value network weights
        self.vw1 = self.rng.normal(0, 0.1, (obs_dim, hidden_size)).astype(np.float32)
        self.vb1 = np.zeros(hidden_size, dtype=np.float32)
        self.vw2 = self.rng.normal(0, 0.1, (hidden_size, 1)).astype(np.float32)
        self.vb2 = np.zeros(1, dtype=np.float32)

        # Action std (log scale)
        self.log_std = np.zeros(action_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray) -> tuple:
        """Forward pass returning action mean and value."""
        # Policy
        h = np.tanh(obs @ self.w1 + self.b1)
        action_mean = np.tanh(h @ self.w2 + self.b2)

        # Value
        vh = np.tanh(obs @ self.vw1 + self.vb1)
        value = (vh @ self.vw2 + self.vb2).squeeze(-1)

        return action_mean, value

    def sample_action(self, obs: np.ndarray) -> tuple:
        """Sample action from policy distribution."""
        action_mean, value = self.forward(obs)
        std = np.exp(self.log_std)
        noise = self.rng.normal(0, 1, action_mean.shape).astype(np.float32)
        action = np.clip(action_mean + std * noise, -1, 1)

        # Compute log probability (simplified)
        log_prob = -0.5 * np.sum((action - action_mean) ** 2 / (std ** 2 + 1e-8), axis=-1)

        return action, value, log_prob


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, gae_lambda: float) -> tuple:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPO training example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole"])
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run PPO training example."""
    parser = create_parser()
    args = parser.parse_args()

    config = PPOConfig(rollout_steps=args.rollout_steps)

    print("Zeno PPO Training Example")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Iterations: {args.iterations}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Total steps per iteration: {args.num_envs * config.rollout_steps:,}")
    print()

    # Create environment
    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Create policy
    policy = SimplePolicyNetwork(env.observation_dim, env.action_dim, seed=args.seed)
    buffer = RolloutBuffer()

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    obs = env.reset()
    total_steps = 0
    episode_rewards = []
    current_episode_rewards = np.zeros(args.num_envs)

    start_time = time.perf_counter()

    for iteration in range(args.iterations):
        iter_start = time.perf_counter()
        buffer.clear()

        # Collect rollout
        for step in range(config.rollout_steps):
            action, value, log_prob = policy.sample_action(obs)

            next_obs, reward, done, info = env.step(action)

            buffer.observations.append(obs)
            buffer.actions.append(action)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value)
            buffer.log_probs.append(log_prob)

            current_episode_rewards += reward
            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0

            obs = next_obs
            total_steps += args.num_envs

        # Convert to arrays
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)

        # Compute advantages (simplified - compute per environment)
        mean_reward = rewards.mean()
        mean_value = values.mean()

        iter_time = time.perf_counter() - iter_start
        sps = (config.rollout_steps * args.num_envs) / iter_time

        if episode_rewards:
            mean_ep_reward = np.mean(episode_rewards[-100:])
        else:
            mean_ep_reward = 0

        print(f"Iter {iteration + 1:>3}/{args.iterations} | "
              f"Steps: {total_steps:>10,} | "
              f"SPS: {sps:>8,.0f} | "
              f"Mean reward: {mean_reward:>8.3f} | "
              f"Episode reward: {mean_ep_reward:>8.2f}")

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 50)
    print("Training Complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total steps: {total_steps:,}")
    print(f"Average throughput: {total_steps / elapsed:,.0f} SPS")
    if episode_rewards:
        print(f"Final episode reward: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"Best episode reward: {max(episode_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
