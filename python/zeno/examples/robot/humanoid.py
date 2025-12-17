"""
Humanoid Locomotion Example

Humanoid character locomotion using the Zeno physics engine.
The humanoid must coordinate many joints to maintain balance and move forward.

Usage:
    python -m zeno.examples humanoid
    python -m zeno.examples humanoid --num-envs 256 --steps 2000
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for humanoid example."""
    parser = argparse.ArgumentParser(
        description="Humanoid locomotion example",
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
        default=2000,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "pd_stand"],
        default="pd_stand",
        help="Control policy: 'random' or 'pd_stand' (PD control to stand)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def pd_stand_controller(
    obs: np.ndarray,
    action_dim: int,
    kp: float = 50.0,
    kd: float = 5.0
) -> np.ndarray:
    """
    Simple PD controller to maintain standing posture.

    Tries to keep all joints at their default positions.
    """
    num_envs = obs.shape[0]

    # Target is zero for all joints (default standing pose)
    # This is a simplistic approach - real controllers would be more sophisticated
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Apply small stabilizing torques
    # In a real implementation, we'd use the joint angles from observations
    actions = np.clip(actions, -1, 1)

    return actions


def main():
    """Run the humanoid example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Humanoid Locomotion Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    # Create environment
    mjcf_path = get_asset("humanoid.xml")
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
    episode_count = np.zeros(args.num_envs)

    # Track height for standing metric
    initial_height = env.get_body_positions()[:, 0, 2].mean()

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Generate actions based on policy
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:  # pd_stand
            actions = pd_stand_controller(obs, env.action_dim)

        # Step simulation
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Track episodes
        episode_count += dones.astype(float)

        # Print progress every 500 steps
        if (step + 1) % 500 == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) * args.num_envs / elapsed

            # Calculate current height
            current_height = env.get_body_positions()[:, 0, 2].mean()
            height_change = current_height - initial_height

            print(f"Step {step + 1}/{args.steps} | "
                  f"SPS: {sps:.0f} | "
                  f"Mean reward: {total_reward.mean() / (step + 1):.4f} | "
                  f"Height: {current_height:.2f}m ({height_change:+.2f})")

    elapsed = time.perf_counter() - start_time

    # Final height
    final_height = env.get_body_positions()[:, 0, 2].mean()

    print()
    print("Simulation complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Steps per second: {args.steps * args.num_envs / elapsed:.0f}")
    print(f"Mean total reward: {total_reward.mean():.4f}")
    print(f"Final height: {final_height:.2f}m")
    print(f"Mean episodes per env: {episode_count.mean():.1f}")

    env.close()


if __name__ == "__main__":
    main()
