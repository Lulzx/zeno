"""
Random Search Training Example

Simple random search for policy optimization.
Demonstrates baseline comparison for more advanced methods.

Usage:
    python -m zeno.examples training_random_search
    python -m zeno.examples training_random_search --num-envs 128 --iterations 200
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


class LinearPolicy:
    """Simple linear policy."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.1, (obs_dim, action_dim)).astype(np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(obs @ self.weights + self.bias), -1, 1)

    def randomize(self, scale: float = 0.1):
        """Randomize policy parameters."""
        self.weights = self.rng.normal(0, scale, (self.obs_dim, self.action_dim)).astype(np.float32)
        self.bias = self.rng.normal(0, scale, self.action_dim).astype(np.float32)

    def copy_from(self, other: "LinearPolicy"):
        """Copy parameters from another policy."""
        self.weights = other.weights.copy()
        self.bias = other.bias.copy()


def evaluate_policy(env, policy: LinearPolicy, num_steps: int) -> tuple:
    """Evaluate policy and return total reward and episode count."""
    obs = env.reset()
    total_reward = np.zeros(env.num_envs)
    episodes_completed = 0

    for _ in range(num_steps):
        actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards
        episodes_completed += dones.sum()

    return total_reward.mean(), episodes_completed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random search training example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="cartpole", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--iterations", type=int, default=200, help="Search iterations")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run random search."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Random Search")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Iterations: {args.iterations}")
    print(f"Evaluation steps: {args.eval_steps}")
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

    # Create policies
    current_policy = LinearPolicy(env.observation_dim, env.action_dim, args.seed)
    best_policy = LinearPolicy(env.observation_dim, env.action_dim, args.seed)

    best_reward = float("-inf")
    reward_history = []

    start_time = time.perf_counter()

    for iteration in range(args.iterations):
        iter_start = time.perf_counter()

        # Randomize current policy
        current_policy.randomize(scale=0.5)

        # Evaluate
        reward, episodes = evaluate_policy(env, current_policy, args.eval_steps)
        reward_history.append(reward)

        # Keep if better
        if reward > best_reward:
            best_reward = reward
            best_policy.copy_from(current_policy)
            improved = " *"
        else:
            improved = ""

        iter_time = time.perf_counter() - iter_start
        sps = args.eval_steps * args.num_envs / iter_time

        if (iteration + 1) % 20 == 0 or improved:
            print(f"Iter {iteration + 1:>3}/{args.iterations} | "
                  f"Reward: {reward:>8.2f} | "
                  f"Best: {best_reward:>8.2f} | "
                  f"SPS: {sps:>8,.0f}{improved}")

    elapsed = time.perf_counter() - start_time

    # Final evaluation of best policy
    final_reward, _ = evaluate_policy(env, best_policy, args.eval_steps * 2)

    print()
    print("=" * 50)
    print("Search Complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Best reward found: {best_reward:.2f}")
    print(f"Final evaluation: {final_reward:.2f}")
    print(f"Total steps: {args.iterations * args.eval_steps * args.num_envs:,}")

    env.close()


if __name__ == "__main__":
    main()
