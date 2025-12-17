"""
CartPole Example

Classic CartPole control problem using the Zeno physics engine.
The goal is to balance a pole on a cart by applying horizontal forces.

Usage:
    python -m zeno.examples cartpole
    python -m zeno.examples cartpole --num-envs 1024 --steps 1000
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for cartpole example."""
    parser = argparse.ArgumentParser(
        description="CartPole balancing example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "pd"],
        default="pd",
        help="Control policy: 'random' or 'pd' (proportional-derivative)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def pd_controller(obs: np.ndarray, kp: float = 10.0, kd: float = 1.0) -> np.ndarray:
    """
    Simple PD controller for CartPole.

    Assumes observation contains [cart_pos, cart_vel, pole_angle, pole_vel].
    """
    # Extract pole angle and angular velocity (typically indices 2 and 3)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    # Simple proportional control on pole angle
    # Positive angle -> push right, negative angle -> push left
    if obs.shape[1] >= 4:
        pole_angle = obs[:, 2] if obs.shape[1] > 2 else obs[:, 0]
        pole_vel = obs[:, 3] if obs.shape[1] > 3 else obs[:, 1]
    else:
        pole_angle = obs[:, 0]
        pole_vel = obs[:, 1] if obs.shape[1] > 1 else np.zeros_like(obs[:, 0])

    control = kp * pole_angle + kd * pole_vel
    return np.clip(control, -1, 1).reshape(-1, 1).astype(np.float32)


def main():
    """Run the cartpole example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno CartPole Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    # Create environment
    mjcf_path = get_asset("cartpole.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Observation dim: {env.observation_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Timestep: {env.timestep}s")
    print()

    # Reset environment
    obs = env.reset()

    # Run simulation
    rng = np.random.default_rng(args.seed)
    total_reward = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs)
    done_count = 0

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Generate actions based on policy
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:  # pd controller
            actions = pd_controller(obs)
            # Ensure correct shape
            if actions.shape != env.action_shape:
                actions = np.broadcast_to(actions, env.action_shape).copy()

        # Step simulation
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards
        episode_lengths += 1

        # Count completed episodes
        if dones.any():
            done_count += dones.sum()
            # Reset episode tracking for done environments
            episode_lengths[dones] = 0

        # Print progress every 200 steps
        if (step + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) * args.num_envs / elapsed
            print(f"Step {step + 1}/{args.steps} | "
                  f"SPS: {sps:.0f} | "
                  f"Mean reward: {total_reward.mean() / (step + 1):.4f} | "
                  f"Episodes done: {done_count}")

    elapsed = time.perf_counter() - start_time

    print()
    print("Simulation complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Steps per second: {args.steps * args.num_envs / elapsed:.0f}")
    print(f"Mean total reward: {total_reward.mean():.4f}")
    print(f"Total episodes completed: {done_count}")

    env.close()


if __name__ == "__main__":
    main()
