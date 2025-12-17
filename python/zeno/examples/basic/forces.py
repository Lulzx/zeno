"""
Forces Example

Demonstrates applying forces and torques to bodies.
Shows how control actions affect the physics simulation.

Usage:
    python -m zeno.examples basic_forces
    python -m zeno.examples basic_forces --num-envs 32
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Force application demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--force-type", type=str, choices=["constant", "impulse", "sine", "random"],
                        default="sine", help="Type of force pattern")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def generate_forces(force_type: str, step: int, shape: tuple, rng: np.random.Generator,
                    timestep: float) -> np.ndarray:
    """Generate control forces based on the specified pattern."""
    if force_type == "constant":
        return np.ones(shape, dtype=np.float32) * 0.5
    elif force_type == "impulse":
        # Apply impulse every 50 steps
        if step % 50 == 0:
            return np.ones(shape, dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)
    elif force_type == "sine":
        t = step * timestep
        return (np.sin(2 * np.pi * 1.0 * t) * np.ones(shape)).astype(np.float32)
    elif force_type == "random":
        return rng.uniform(-1, 1, shape).astype(np.float32)
    else:
        return np.zeros(shape, dtype=np.float32)


def main():
    """Run the forces example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Forces Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print(f"Force type: {args.force_type}")
    print()

    mjcf_path = get_asset("cartpole.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    print(f"Action dim (actuators): {env.action_dim}")
    print()

    obs = env.reset()
    rng = np.random.default_rng(args.seed)

    total_displacement = np.zeros(args.num_envs)
    initial_positions = env.get_body_positions()[:, 0, 0].copy()

    start_time = time.perf_counter()

    for step in range(args.steps):
        actions = generate_forces(args.force_type, step, env.action_shape, rng, env.timestep)
        obs, rewards, dones, info = env.step(actions)

        # Track cart displacement
        current_positions = env.get_body_positions()[:, 0, 0]
        displacement = current_positions - initial_positions

        if (step + 1) % 100 == 0:
            mean_disp = displacement.mean()
            max_disp = np.abs(displacement).max()
            print(f"Step {step + 1}/{args.steps} | "
                  f"Mean displacement: {mean_disp:.3f}m | "
                  f"Max displacement: {max_disp:.3f}m")

    elapsed = time.perf_counter() - start_time
    print()
    print(f"Simulation complete in {elapsed:.2f}s")
    print(f"Final mean displacement: {displacement.mean():.3f}m")

    env.close()


if __name__ == "__main__":
    main()
