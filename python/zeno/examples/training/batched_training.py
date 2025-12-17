"""
Batched Training Example

Demonstrates high-throughput batched simulation for reinforcement learning.
Shows how to efficiently collect experience from thousands of parallel environments.

Usage:
    python -m zeno.examples batched_training
    python -m zeno.examples batched_training --num-envs 4096 --steps 10000
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

import zeno
from zeno.examples import get_asset


@dataclass
class TrainingStats:
    """Statistics collected during training."""
    total_steps: int = 0
    total_episodes: int = 0
    total_reward: float = 0.0
    best_episode_reward: float = float("-inf")
    episode_rewards: list = None
    episode_lengths: list = None

    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []


class SimplePolicy:
    """
    A simple random policy for demonstration.

    In practice, you would replace this with a neural network policy
    (e.g., using PyTorch or JAX).
    """

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

        # Simple linear policy weights (for demonstration)
        self.weights = self.rng.normal(0, 0.1, (obs_dim, action_dim)).astype(np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute actions for a batch of observations."""
        # Simple linear policy: a = tanh(W @ o + b)
        actions = np.tanh(obs @ self.weights + self.bias)
        # Add exploration noise
        noise = self.rng.normal(0, 0.1, actions.shape).astype(np.float32)
        return np.clip(actions + noise, -1, 1)


class ExperienceBuffer:
    """
    Simple experience buffer for storing transitions.

    In practice, you would use a more sophisticated replay buffer
    or on-policy buffer depending on your algorithm.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        """Add a batch of transitions to the buffer."""
        batch_size = obs.shape[0]

        for i in range(batch_size):
            self.obs[self.ptr] = obs[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr] = rewards[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.dones[self.ptr] = dones[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for batched training example."""
    parser = argparse.ArgumentParser(
        description="Batched RL training example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ant",
        choices=["pendulum", "cartpole", "ant", "humanoid"],
        help="Environment to use",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Experience buffer capacity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def main():
    """Run the batched training example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Batched Training Example")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Total steps: {args.steps}")
    print(f"Buffer size: {args.buffer_size}")
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
    print(f"Timestep: {env.timestep}s")
    print()

    # Create policy and buffer
    policy = SimplePolicy(env.observation_dim, env.action_dim, args.seed)
    buffer = ExperienceBuffer(args.buffer_size, env.observation_dim, env.action_dim)

    # Initialize tracking
    stats = TrainingStats()
    episode_rewards = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)

    # Reset environment
    obs = env.reset()

    print("Starting training loop...")
    print("-" * 50)

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Get actions from policy
        actions = policy.act(obs)

        # Step environment
        next_obs, rewards, dones, info = env.step(actions)

        # Store transitions
        buffer.add(obs, actions, rewards, next_obs, dones)

        # Update episode tracking
        episode_rewards += rewards
        episode_lengths += 1
        stats.total_steps += args.num_envs

        # Handle episode completions
        if dones.any():
            completed_mask = dones.astype(bool)
            completed_rewards = episode_rewards[completed_mask]
            completed_lengths = episode_lengths[completed_mask]

            for r, l in zip(completed_rewards, completed_lengths):
                stats.episode_rewards.append(r)
                stats.episode_lengths.append(l)
                stats.total_reward += r
                stats.best_episode_reward = max(stats.best_episode_reward, r)

            stats.total_episodes += completed_mask.sum()

            # Reset tracking for completed episodes
            episode_rewards[completed_mask] = 0
            episode_lengths[completed_mask] = 0

        obs = next_obs

        # Print progress
        if (step + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start_time
            sps = stats.total_steps / elapsed
            eps = stats.total_episodes / elapsed if elapsed > 0 else 0

            if stats.episode_rewards:
                recent_rewards = stats.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)
            else:
                mean_reward = 0

            print(
                f"Step {step + 1:>6}/{args.steps} | "
                f"SPS: {sps:>8.0f} | "
                f"EPS: {eps:>6.1f} | "
                f"Episodes: {stats.total_episodes:>6} | "
                f"Mean reward (100): {mean_reward:>8.2f} | "
                f"Buffer: {buffer.size:>6}"
            )

    elapsed = time.perf_counter() - start_time

    # Final statistics
    print()
    print("=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total environment steps: {stats.total_steps:,}")
    print(f"Steps per second: {stats.total_steps / elapsed:,.0f}")
    print(f"Total episodes: {stats.total_episodes:,}")
    print(f"Episodes per second: {stats.total_episodes / elapsed:.1f}")

    if stats.episode_rewards:
        print(f"Mean episode reward: {np.mean(stats.episode_rewards):.4f}")
        print(f"Best episode reward: {stats.best_episode_reward:.4f}")
        print(f"Mean episode length: {np.mean(stats.episode_lengths):.1f}")

    print(f"Experience buffer filled: {buffer.size:,} / {buffer.capacity:,}")
    print()
    print("Throughput comparison:")
    print(f"  With {args.num_envs} envs: {stats.total_steps / elapsed:,.0f} SPS")
    estimated_single = stats.total_steps / elapsed / args.num_envs
    print(f"  Single env equivalent: ~{estimated_single:,.0f} SPS")
    print(f"  Speedup: ~{args.num_envs}x")

    env.close()


if __name__ == "__main__":
    main()
