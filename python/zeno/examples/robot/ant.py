"""
Ant Locomotion Example

Ant quadruped locomotion using the Zeno physics engine.
The ant must learn to coordinate its 8 actuated joints to move forward.

Usage:
    python -m zeno.examples ant
    python -m zeno.examples ant --num-envs 1024 --steps 2000
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ant example."""
    parser = argparse.ArgumentParser(
        description="Ant locomotion example",
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
        choices=["random", "sine"],
        default="sine",
        help="Control policy: 'random' or 'sine' (sinusoidal gait)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def sine_gait(step: int, num_envs: int, action_dim: int, freq: float = 2.0) -> np.ndarray:
    """
    Generate a simple sinusoidal gait pattern.

    Creates phase-offset sine waves for each actuator to produce
    a basic walking pattern.
    """
    t = step * 0.02  # Assume 20ms timestep
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    for i in range(action_dim):
        # Phase offset for each joint to create walking pattern
        phase = i * np.pi / 4
        actions[:, i] = 0.5 * np.sin(2 * np.pi * freq * t + phase)

    return actions


def main():
    """Run the ant example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Ant Locomotion Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Policy: {args.policy}")
    print()

    # Create environment
    mjcf_path = get_asset("ant.xml")
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

    # Track position for forward progress
    initial_positions = env.get_body_positions()[:, 0, :3].copy()

    start_time = time.perf_counter()

    for step in range(args.steps):
        # Generate actions based on policy
        if args.policy == "random":
            actions = rng.uniform(-1, 1, env.action_shape).astype(np.float32)
        else:  # sine gait
            actions = sine_gait(step, args.num_envs, env.action_dim)

        # Step simulation
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards

        # Track episodes
        episode_count += dones.astype(float)

        # Print progress every 500 steps
        if (step + 1) % 500 == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step + 1) * args.num_envs / elapsed

            # Calculate forward progress
            current_positions = env.get_body_positions()[:, 0, :3]
            forward_progress = (current_positions[:, 0] - initial_positions[:, 0]).mean()

            print(f"Step {step + 1}/{args.steps} | "
                  f"SPS: {sps:.0f} | "
                  f"Mean reward: {total_reward.mean() / (step + 1):.4f} | "
                  f"Forward: {forward_progress:.2f}m")

    elapsed = time.perf_counter() - start_time

    # Final forward progress
    final_positions = env.get_body_positions()[:, 0, :3]
    forward_progress = (final_positions[:, 0] - initial_positions[:, 0]).mean()

    print()
    print("Simulation complete!")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Steps per second: {args.steps * args.num_envs / elapsed:.0f}")
    print(f"Mean total reward: {total_reward.mean():.4f}")
    print(f"Mean forward progress: {forward_progress:.2f}m")
    print(f"Mean episodes per env: {episode_count.mean():.1f}")

    env.close()


if __name__ == "__main__":
    main()
