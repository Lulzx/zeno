"""
Evolutionary Strategy Training Example

Evolution Strategies (ES) for policy optimization using Zeno.
Demonstrates gradient-free optimization with parallel evaluation.

Usage:
    python -m zeno.examples training_evolution
    python -m zeno.examples training_evolution --num-envs 256 --generations 50
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


class LinearPolicy:
    """Linear policy with flattened parameter vector."""

    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.param_count = obs_dim * action_dim + action_dim

        # Initialize parameters
        self.weights = np.zeros((obs_dim, action_dim), dtype=np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def get_params(self) -> np.ndarray:
        """Get flattened parameter vector."""
        return np.concatenate([self.weights.flatten(), self.bias])

    def set_params(self, params: np.ndarray):
        """Set parameters from flattened vector."""
        w_size = self.obs_dim * self.action_dim
        self.weights = params[:w_size].reshape(self.obs_dim, self.action_dim)
        self.bias = params[w_size:]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(obs @ self.weights + self.bias), -1, 1)


def evaluate_policy(env, policy: LinearPolicy, num_steps: int = 200) -> float:
    """Evaluate a policy and return mean reward."""
    obs = env.reset()
    total_reward = 0

    for _ in range(num_steps):
        actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards.mean()

    return total_reward


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolution strategies training example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="cartpole", choices=["ant", "humanoid", "cartpole", "pendulum"])
    parser.add_argument("--num-envs", type=int, default=64, help="Environments for evaluation")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=200, help="Steps per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main():
    """Run evolutionary strategy training."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Evolution Strategy Training")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Evaluation envs: {args.num_envs}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Sigma: {args.sigma}")
    print(f"Learning rate: {args.learning_rate}")
    print()

    # Create environment
    mjcf_path = get_asset(f"{args.env}.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Create policy
    policy = LinearPolicy(env.observation_dim, env.action_dim)
    rng = np.random.default_rng(args.seed)

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Parameters: {policy.param_count}")
    print()

    # Current best parameters
    theta = policy.get_params()
    best_reward = float("-inf")
    best_theta = theta.copy()

    reward_history = []
    start_time = time.perf_counter()

    for gen in range(args.generations):
        gen_start = time.perf_counter()

        # Generate perturbations
        epsilon = rng.normal(0, 1, (args.population, policy.param_count)).astype(np.float32)

        # Evaluate perturbed policies
        rewards = np.zeros(args.population)
        for i in range(args.population):
            perturbed_theta = theta + args.sigma * epsilon[i]
            policy.set_params(perturbed_theta)
            rewards[i] = evaluate_policy(env, policy, args.eval_steps)

        # Update parameters using reward-weighted combination
        # Normalize rewards
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute gradient estimate
        gradient = np.zeros_like(theta)
        for i in range(args.population):
            gradient += rewards_normalized[i] * epsilon[i]
        gradient /= args.population * args.sigma

        # Update parameters
        theta = theta + args.learning_rate * gradient
        policy.set_params(theta)

        # Evaluate current policy
        current_reward = evaluate_policy(env, policy, args.eval_steps)
        reward_history.append(current_reward)

        if current_reward > best_reward:
            best_reward = current_reward
            best_theta = theta.copy()

        gen_time = time.perf_counter() - gen_start
        evals_per_sec = args.population * args.eval_steps * args.num_envs / gen_time

        print(f"Gen {gen + 1:>3}/{args.generations} | "
              f"Reward: {current_reward:>8.2f} | "
              f"Best: {best_reward:>8.2f} | "
              f"Pop mean: {rewards.mean():>8.2f} | "
              f"Evals/s: {evals_per_sec:>10,.0f}")

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 50)
    print("Training Complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final reward: {current_reward:.2f}")
    print(f"Total evaluations: {args.generations * args.population:,}")

    env.close()


if __name__ == "__main__":
    main()
