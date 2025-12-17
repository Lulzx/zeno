"""Quick visualization demo for Zeno physics simulation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
sys.path.insert(0, '/Users/lulzx/work/zeno/python')

import zeno
from zeno.examples import get_asset

def visualize_cartpole():
    """Visualize cartpole simulation."""
    print("Creating Zeno CartPole simulation...")

    env = zeno.ZenoEnv(
        mjcf_path=get_asset("cartpole.xml"),
        num_envs=1,
        seed=42,
    )

    # Collect trajectory data
    obs = env.reset()
    positions_history = []

    print("Running simulation...")
    rng = np.random.default_rng(42)

    for step in range(200):
        # Simple control: push towards center
        cart_pos = env.get_body_positions()[0, 1, 0]  # Cart x position
        actions = np.array([[-0.5 * cart_pos]], dtype=np.float32)
        actions = np.clip(actions + rng.normal(0, 0.1, actions.shape), -1, 1).astype(np.float32)

        obs, rewards, dones, info = env.step(actions)

        # Store positions: cart and pole
        pos = env.get_body_positions()[0]  # First env
        quat = env.get_body_quaternions()[0]
        positions_history.append({
            'cart_pos': pos[1, :3].copy(),  # Cart body
            'pole_pos': pos[2, :3].copy(),  # Pole body
            'pole_quat': quat[2].copy(),
        })

    env.close()

    # Create visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Zeno Physics Engine - CartPole Simulation', fontsize=14, fontweight='bold')

    frames_to_show = [0, 40, 80, 120, 160, 199]

    for idx, frame in enumerate(frames_to_show):
        ax = axes[idx // 3, idx % 3]
        data = positions_history[frame]

        cart_x = data['cart_pos'][0]
        cart_z = data['cart_pos'][2]
        pole_pos = data['pole_pos']
        quat = data['pole_quat']

        # Convert quaternion to angle (simplified - extract rotation around Y axis)
        # For a pole rotating in XZ plane
        w, x, y, z = quat[3], quat[0], quat[1], quat[2]
        angle = 2 * np.arctan2(y, w)

        # Draw ground
        ax.axhline(y=0, color='brown', linewidth=3, label='Ground')
        ax.fill_between([-3, 3], [-0.2, -0.2], [0, 0], color='#8B4513', alpha=0.3)

        # Draw cart
        cart_width, cart_height = 0.4, 0.2
        cart = patches.FancyBboxPatch(
            (cart_x - cart_width/2, cart_z - cart_height/2),
            cart_width, cart_height,
            boxstyle="round,pad=0.02",
            facecolor='#3498db', edgecolor='#2980b9', linewidth=2
        )
        ax.add_patch(cart)

        # Draw wheels
        wheel_radius = 0.05
        for wheel_x in [cart_x - 0.12, cart_x + 0.12]:
            wheel = plt.Circle((wheel_x, cart_z - cart_height/2), wheel_radius,
                              color='#2c3e50', zorder=5)
            ax.add_patch(wheel)

        # Draw pole
        pole_length = 0.5
        pole_end_x = cart_x + pole_length * np.sin(angle)
        pole_end_z = cart_z + cart_height/2 + pole_length * np.cos(angle)

        ax.plot([cart_x, pole_end_x], [cart_z + cart_height/2, pole_end_z],
                color='#e74c3c', linewidth=6, solid_capstyle='round', zorder=3)

        # Draw pole tip
        ax.scatter([pole_end_x], [pole_end_z], s=100, color='#c0392b', zorder=4)

        # Draw pivot point
        ax.scatter([cart_x], [cart_z + cart_height/2], s=60, color='#f39c12', zorder=5)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.3, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Frame {frame} (t={frame*0.02:.2f}s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')

    plt.tight_layout()

    # Save the figure
    output_path = '/Users/lulzx/work/zeno/cartpole_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")

    return output_path


def visualize_ant():
    """Visualize ant locomotion from top-down view."""
    print("\nCreating Zeno Ant simulation...")

    env = zeno.ZenoEnv(
        mjcf_path=get_asset("ant.xml"),
        num_envs=1,
        seed=42,
    )

    # Collect trajectory
    obs = env.reset()
    trajectory = []

    print("Running simulation...")
    for step in range(300):
        # Sinusoidal gait
        t = step * 0.02
        actions = np.zeros((1, 8), dtype=np.float32)
        for i in range(8):
            phase = i * np.pi / 4
            actions[0, i] = 0.5 * np.sin(2 * np.pi * 2 * t + phase)

        obs, rewards, dones, info = env.step(actions)

        pos = env.get_body_positions()[0, 0, :3]  # Torso position
        trajectory.append(pos.copy())

    env.close()
    trajectory = np.array(trajectory)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Zeno Physics Engine - Ant Locomotion (Top-Down View)', fontsize=14, fontweight='bold')

    # Plot trajectory with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    for i in range(len(trajectory) - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                color=colors[i], linewidth=2)

    # Mark start and end
    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], s=200, c='green',
               marker='o', label='Start', zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=200, c='red',
               marker='s', label='End', zorder=5, edgecolors='darkred', linewidth=2)

    # Draw ant body at several points
    ant_frames = [0, 75, 150, 225, 299]
    for frame in ant_frames:
        pos = trajectory[frame]
        # Draw simple ant representation
        circle = plt.Circle((pos[0], pos[1]), 0.25, fill=False,
                           color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(circle)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 300*0.02))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time (s)')

    plt.tight_layout()

    output_path = '/Users/lulzx/work/zeno/ant_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")

    return output_path


if __name__ == "__main__":
    cartpole_path = visualize_cartpole()
    ant_path = visualize_ant()
    print(f"\nVisualization complete!")
    print(f"  CartPole: {cartpole_path}")
    print(f"  Ant: {ant_path}")
