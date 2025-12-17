"""
State Access Example

Demonstrates accessing and manipulating simulation state.
Shows body positions, orientations, velocities, and contacts.

Usage:
    python -m zeno.examples utils_state_access
    python -m zeno.examples utils_state_access --num-envs 16
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulation state access patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to euler angles."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def main():
    """Run state access example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno State Access Example")
    print("=" * 50)
    print(f"Parallel environments: {args.num_envs}")
    print()

    mjcf_path = get_asset("ant.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    obs = env.reset()

    print("State Access Methods")
    print("-" * 50)

    # 1. Observations
    print("\n1. Observations:")
    print(f"   Shape: {obs.shape}")
    print(f"   Dim per env: {env.observation_dim}")
    print(f"   Sample (env 0): {obs[0, :5]}...")

    # 2. Body positions
    print("\n2. Body Positions:")
    positions = env.get_body_positions()
    print(f"   Shape: {positions.shape}")
    print(f"   Format: (num_envs, num_bodies, 4) - xyz + padding")
    print(f"   Env 0, Body 0 position: {positions[0, 0, :3]}")

    # 3. Body orientations
    print("\n3. Body Orientations:")
    quaternions = env.get_body_quaternions()
    print(f"   Shape: {quaternions.shape}")
    print(f"   Format: (num_envs, num_bodies, 4) - xyzw quaternion")
    euler = quaternion_to_euler(quaternions[0, 0])
    print(f"   Env 0, Body 0 euler (deg): {np.degrees(euler)}")

    # Run simulation and track state changes
    print("\n4. State Evolution:")
    print("-" * 50)

    initial_positions = positions[:, 0, :3].copy()
    max_displacement = np.zeros(args.num_envs)
    max_rotation = np.zeros(args.num_envs)

    for step in range(args.steps):
        actions = rng.uniform(-0.5, 0.5, env.action_shape).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

        # Track position changes
        positions = env.get_body_positions()
        current_pos = positions[:, 0, :3]
        displacement = np.linalg.norm(current_pos - initial_positions, axis=1)
        max_displacement = np.maximum(max_displacement, displacement)

        # Track rotation changes
        quaternions = env.get_body_quaternions()
        euler = quaternion_to_euler(quaternions[:, 0])
        rotation_mag = np.linalg.norm(euler, axis=1)
        max_rotation = np.maximum(max_rotation, rotation_mag)

        if (step + 1) % 50 == 0:
            print(f"   Step {step + 1}: "
                  f"Displacement: {displacement.mean():.3f}m, "
                  f"Rotation: {np.degrees(rotation_mag.mean()):.1f}deg")

    # Summary statistics
    print("\n5. State Statistics Summary:")
    print("-" * 50)
    print(f"   Max displacement: {max_displacement.max():.3f}m")
    print(f"   Mean max displacement: {max_displacement.mean():.3f}m")
    print(f"   Max rotation: {np.degrees(max_rotation.max()):.1f}deg")
    print(f"   Mean max rotation: {np.degrees(max_rotation.mean()):.1f}deg")

    # Memory layout info
    print("\n6. Memory Layout:")
    print("-" * 50)
    print(f"   Observations: {obs.nbytes:,} bytes ({obs.dtype})")
    print(f"   Positions: {positions.nbytes:,} bytes ({positions.dtype})")
    print(f"   Quaternions: {quaternions.nbytes:,} bytes ({quaternions.dtype})")
    print(f"   Total state: {obs.nbytes + positions.nbytes + quaternions.nbytes:,} bytes")

    print()
    print("=" * 50)
    print("State access demonstration complete!")

    env.close()


if __name__ == "__main__":
    main()
