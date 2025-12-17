"""
Checkpointing Example

Demonstrates saving and loading simulation state and trained policies.
Shows how to implement training checkpoints for long runs.

Usage:
    python -m zeno.examples utils_checkpointing
    python -m zeno.examples utils_checkpointing --num-envs 64
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np

import zeno
from zeno.examples import get_asset


class TrainablePolicy:
    """Policy with save/load capabilities."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

        self.w1 = self.rng.normal(0, 0.1, (obs_dim, 64)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.w2 = self.rng.normal(0, 0.1, (64, action_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

        self.training_steps = 0
        self.best_reward = float("-inf")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(obs @ self.w1 + self.b1)
        return np.clip(np.tanh(h @ self.w2 + self.b2), -1, 1)

    def save(self, path: str):
        """Save policy to file."""
        checkpoint = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "training_steps": self.training_steps,
            "best_reward": self.best_reward,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f)

    def load(self, path: str):
        """Load policy from file."""
        with open(path, "r") as f:
            checkpoint = json.load(f)

        self.obs_dim = checkpoint["obs_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.w1 = np.array(checkpoint["w1"], dtype=np.float32)
        self.b1 = np.array(checkpoint["b1"], dtype=np.float32)
        self.w2 = np.array(checkpoint["w2"], dtype=np.float32)
        self.b2 = np.array(checkpoint["b2"], dtype=np.float32)
        self.training_steps = checkpoint["training_steps"]
        self.best_reward = checkpoint["best_reward"]


class TrainingState:
    """Training state for checkpointing."""

    def __init__(self):
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.best_reward = float("-inf")

    def save(self, path: str):
        state = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards[-100:],  # Keep last 100
            "best_reward": self.best_reward,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path: str):
        with open(path, "r") as f:
            state = json.load(f)
        self.episode = state["episode"]
        self.total_steps = state["total_steps"]
        self.episode_rewards = state["episode_rewards"]
        self.best_reward = state["best_reward"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training checkpointing example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--episode-length", type=int, default=200, help="Steps per episode")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Episodes between checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run checkpointing example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Checkpointing Example")
    print("=" * 50)
    print(f"Parallel environments: {args.num_envs}")
    print(f"Episodes: {args.episodes}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print()

    mjcf_path = get_asset("cartpole.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Create temporary checkpoint directory
    checkpoint_dir = tempfile.mkdtemp(prefix="zeno_checkpoints_")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()

    policy = TrainablePolicy(env.observation_dim, env.action_dim, args.seed)
    train_state = TrainingState()
    rng = np.random.default_rng(args.seed)

    print("Training with checkpoints...")
    print("-" * 50)

    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = np.zeros(args.num_envs)

        for step in range(args.episode_length):
            actions = policy(obs)
            # Add exploration noise
            actions += rng.normal(0, 0.1, actions.shape).astype(np.float32)
            actions = np.clip(actions, -1, 1)

            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards
            train_state.total_steps += args.num_envs

        mean_reward = episode_reward.mean()
        train_state.episode_rewards.append(mean_reward)
        train_state.episode += 1

        if mean_reward > train_state.best_reward:
            train_state.best_reward = mean_reward
            policy.best_reward = mean_reward

        # Save checkpoint
        if (episode + 1) % args.checkpoint_interval == 0:
            policy_path = os.path.join(checkpoint_dir, f"policy_ep{episode + 1}.json")
            state_path = os.path.join(checkpoint_dir, f"state_ep{episode + 1}.json")

            policy.training_steps = train_state.total_steps
            policy.save(policy_path)
            train_state.save(state_path)

            policy_size = os.path.getsize(policy_path)
            state_size = os.path.getsize(state_path)

            print(f"Episode {episode + 1:>3}: Reward={mean_reward:>7.2f} | "
                  f"Checkpoint saved ({policy_size + state_size:,} bytes)")
        else:
            print(f"Episode {episode + 1:>3}: Reward={mean_reward:>7.2f}")

    # Demonstrate loading
    print()
    print("Demonstrating checkpoint loading...")
    print("-" * 50)

    # Load middle checkpoint
    mid_ep = (args.episodes // 2) // args.checkpoint_interval * args.checkpoint_interval
    if mid_ep == 0:
        mid_ep = args.checkpoint_interval

    policy_path = os.path.join(checkpoint_dir, f"policy_ep{mid_ep}.json")
    state_path = os.path.join(checkpoint_dir, f"state_ep{mid_ep}.json")

    if os.path.exists(policy_path):
        loaded_policy = TrainablePolicy(env.observation_dim, env.action_dim)
        loaded_policy.load(policy_path)

        loaded_state = TrainingState()
        loaded_state.load(state_path)

        print(f"Loaded checkpoint from episode {mid_ep}")
        print(f"  Training steps: {loaded_policy.training_steps:,}")
        print(f"  Best reward: {loaded_policy.best_reward:.2f}")
        print(f"  Recent rewards: {len(loaded_state.episode_rewards)} episodes")

    # List all checkpoints
    print()
    print("Available checkpoints:")
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(checkpoint_dir, f))
            print(f"  {f}: {size:,} bytes")

    # Cleanup
    import shutil
    shutil.rmtree(checkpoint_dir)
    print()
    print(f"Cleaned up checkpoint directory")

    print()
    print("=" * 50)
    print("Checkpointing demonstration complete!")
    print(f"Best reward achieved: {train_state.best_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
