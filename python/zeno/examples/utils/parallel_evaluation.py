"""
Parallel Evaluation Example

Demonstrates efficient parallel policy evaluation across many environments.
Shows how to leverage vectorization for fast evaluation.

Usage:
    python -m zeno.examples utils_parallel_evaluation
    python -m zeno.examples utils_parallel_evaluation --num-envs 4096
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


class Policy:
    """Simple policy for evaluation."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, (obs_dim, action_dim)).astype(np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(obs @ self.weights + self.bias), -1, 1)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel policy evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="ant", choices=["ant", "humanoid", "cartpole"])
    parser.add_argument("--num-envs", type=int, default=1024, help="Parallel environments")
    parser.add_argument("--num-policies", type=int, default=10, help="Policies to evaluate")
    parser.add_argument("--eval-steps", type=int, default=500, help="Steps per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def evaluate_policy(env, policy: Policy, num_steps: int) -> dict:
    """Evaluate a single policy across all environments."""
    obs = env.reset()
    total_reward = np.zeros(env.num_envs)
    episode_count = np.zeros(env.num_envs)
    episode_rewards = []

    current_episode_reward = np.zeros(env.num_envs)

    for step in range(num_steps):
        actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)

        total_reward += rewards
        current_episode_reward += rewards

        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_reward[i])
                current_episode_reward[i] = 0
                episode_count[i] += 1

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward.mean(),
        "std_reward": total_reward.std(),
        "episode_count": episode_count.sum(),
        "episode_rewards": episode_rewards,
    }


def main():
    """Run parallel evaluation example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Parallel Evaluation Example")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Policies to evaluate: {args.num_policies}")
    print(f"Steps per evaluation: {args.eval_steps}")
    print(f"Total steps: {args.num_policies * args.eval_steps * args.num_envs:,}")
    print()

    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    # Generate policies with different seeds
    policies = [
        Policy(env.observation_dim, env.action_dim, seed=args.seed + i)
        for i in range(args.num_policies)
    ]

    # Evaluate all policies
    print("Evaluating policies...")
    print("-" * 50)

    results = []
    total_start = time.perf_counter()

    for i, policy in enumerate(policies):
        start = time.perf_counter()
        result = evaluate_policy(env, policy, args.eval_steps)
        elapsed = time.perf_counter() - start

        results.append(result)
        sps = args.eval_steps * args.num_envs / elapsed

        print(f"Policy {i + 1:>2}/{args.num_policies}: "
              f"Reward={result['mean_reward']:>8.2f} +/- {result['std_reward']:.2f}, "
              f"Episodes={result['episode_count']:.0f}, "
              f"SPS={sps:,.0f}")

    total_elapsed = time.perf_counter() - total_start

    # Summary statistics
    print()
    print("=" * 50)
    print("Evaluation Summary")
    print("=" * 50)

    all_rewards = [r["mean_reward"] for r in results]
    best_idx = np.argmax(all_rewards)
    worst_idx = np.argmin(all_rewards)

    print(f"Best policy: {best_idx + 1} (reward: {all_rewards[best_idx]:.2f})")
    print(f"Worst policy: {worst_idx + 1} (reward: {all_rewards[worst_idx]:.2f})")
    print(f"Mean across policies: {np.mean(all_rewards):.2f}")
    print(f"Std across policies: {np.std(all_rewards):.2f}")
    print()
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Time per policy: {total_elapsed / args.num_policies:.2f}s")

    total_steps = args.num_policies * args.eval_steps * args.num_envs
    print(f"Total throughput: {total_steps / total_elapsed:,.0f} SPS")

    env.close()


if __name__ == "__main__":
    main()
