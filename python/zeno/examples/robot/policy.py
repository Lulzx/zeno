"""
Policy Loading Example

Demonstrates loading and running trained neural network policies.
Shows integration with PyTorch for policy inference.

Usage:
    python -m zeno.examples robot_policy
    python -m zeno.examples robot_policy --num-envs 1024 --policy-type mlp
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neural network policy demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--policy-type", type=str, choices=["mlp", "linear", "random"],
                        default="mlp", help="Policy architecture")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


class LinearPolicy:
    """Simple linear policy: a = W @ o + b"""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, (obs_dim, action_dim)).astype(np.float32)
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(np.tanh(obs @ self.weights + self.bias), -1, 1)


class MLPPolicy:
    """Two-layer MLP policy with tanh activation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.1, (obs_dim, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = rng.normal(0, 0.1, (hidden_size, action_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(obs @ self.w1 + self.b1)
        return np.clip(np.tanh(h @ self.w2 + self.b2), -1, 1)


class RandomPolicy:
    """Random policy for baseline comparison."""

    def __init__(self, action_dim: int, seed: int = 42):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        batch_size = obs.shape[0]
        return self.rng.uniform(-1, 1, (batch_size, self.action_dim)).astype(np.float32)


def main():
    """Run the policy example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Policy Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy type: {args.policy_type}")
    if args.policy_type == "mlp":
        print(f"Hidden size: {args.hidden_size}")
    print()

    mjcf_path = get_asset("ant.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Create policy
    if args.policy_type == "linear":
        policy = LinearPolicy(env.observation_dim, env.action_dim, args.seed)
        param_count = env.observation_dim * env.action_dim + env.action_dim
    elif args.policy_type == "mlp":
        policy = MLPPolicy(env.observation_dim, env.action_dim, args.hidden_size, args.seed)
        param_count = (env.observation_dim * args.hidden_size + args.hidden_size +
                       args.hidden_size * env.action_dim + env.action_dim)
    else:
        policy = RandomPolicy(env.action_dim, args.seed)
        param_count = 0

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Policy parameters: {param_count}")
    print()

    obs = env.reset()
    total_reward = np.zeros(args.num_envs)
    episode_count = np.zeros(args.num_envs)

    # Measure inference time
    inference_times = []

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Time policy inference
        inf_start = time.perf_counter()
        actions = policy(obs)
        inference_times.append(time.perf_counter() - inf_start)

        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards
        episode_count += dones.astype(float)

        if (step + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) * args.num_envs / elapsed
            mean_inf_time = np.mean(inference_times[-100:]) * 1000

            print(f"Step {step + 1}/{args.steps} | "
                  f"SPS: {sps:.0f} | "
                  f"Inference: {mean_inf_time:.2f}ms | "
                  f"Reward: {total_reward.mean():.2f}")

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Total steps: {args.steps * args.num_envs:,}")
    print(f"Throughput: {args.steps * args.num_envs / elapsed:,.0f} SPS")
    print(f"Mean inference time: {np.mean(inference_times) * 1000:.3f}ms")
    print(f"Mean episode reward: {total_reward.mean():.2f}")

    env.close()


if __name__ == "__main__":
    main()
