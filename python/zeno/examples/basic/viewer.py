"""
Viewer Example

Demonstrates how to extract visualization data from Zeno for rendering.
Shows body positions and orientations that can be used with any renderer.

Usage:
    python -m zeno.examples basic_viewer
    python -m zeno.examples basic_viewer --num-envs 16
"""

import argparse
import time

import numpy as np

import zeno
from zeno.examples import get_asset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization data extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


def main():
    """Run the viewer example."""
    parser = create_parser()
    args = parser.parse_args()

    print("Zeno Viewer Data Example")
    print("=" * 40)
    print(f"Environments: {args.num_envs}")
    print(f"Steps: {args.steps}")
    print()

    mjcf_path = get_asset("ant.xml")
    env = zeno.ZenoEnv(
        mjcf_path=mjcf_path,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    obs = env.reset()
    rng = np.random.default_rng(args.seed)

    print("Extracting visualization data...")
    print()

    # Collect frames for "rendering"
    frames = []

    start_time = time.perf_counter()

    for step in range(args.steps):
        actions = rng.uniform(-0.5, 0.5, env.action_shape).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

        # Extract visualization data
        positions = env.get_body_positions()  # (num_envs, num_bodies, 4)
        quaternions = env.get_body_quaternions()  # (num_envs, num_bodies, 4)

        frames.append({
            "step": step,
            "positions": positions.copy(),
            "quaternions": quaternions.copy(),
        })

        if (step + 1) % 50 == 0:
            # Show sample visualization data
            pos = positions[0, 0, :3]  # First env, first body, xyz
            quat = quaternions[0, 0]  # First env, first body
            euler = quaternion_to_euler(quat)

            print(f"Step {step + 1}/{args.steps}")
            print(f"  Body 0 position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"  Body 0 rotation: ({np.degrees(euler[0]):.1f}, {np.degrees(euler[1]):.1f}, {np.degrees(euler[2]):.1f}) deg")

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Collected {len(frames)} frames in {elapsed:.2f}s")
    print(f"Frame rate: {len(frames) / elapsed:.1f} FPS")
    print(f"Data per frame: {positions.nbytes + quaternions.nbytes} bytes")
    print(f"Total data: {len(frames) * (positions.nbytes + quaternions.nbytes) / 1024:.1f} KB")

    env.close()


if __name__ == "__main__":
    main()
