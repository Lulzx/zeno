"""Visualize ant simulation as an mp4 video with a stable trot gait."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
import zeno

# --- Coordinated trot gait ---
def sine_gait(step, num_envs, action_dim, freq=1.2):
    t = step * 0.02
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Trot pairings:
    #   Pair A: front-right + back-left
    #   Pair B: front-left + back-right
    phase_a = np.sin(2 * np.pi * freq * t)
    phase_b = np.sin(2 * np.pi * freq * t + np.pi)
    phase_a_knee = np.sin(2 * np.pi * freq * t + np.pi / 2)
    phase_b_knee = np.sin(2 * np.pi * freq * t + np.pi + np.pi / 2)

    hip_amp = 0.25
    ankle_amp = 0.20
    ankle_bias = -0.20

    # hips
    actions[:, 0] = hip_amp * phase_a  # hip_1 (front-right)
    actions[:, 6] = hip_amp * phase_a  # hip_4 (back-left)
    actions[:, 2] = hip_amp * phase_b  # hip_2 (front-left)
    actions[:, 4] = hip_amp * phase_b  # hip_3 (back-right)

    # ankles
    actions[:, 1] = ankle_bias + ankle_amp * phase_a_knee  # ankle_1
    actions[:, 7] = ankle_bias + ankle_amp * phase_a_knee  # ankle_4
    actions[:, 3] = ankle_bias + ankle_amp * phase_b_knee  # ankle_2
    actions[:, 5] = ankle_bias + ankle_amp * phase_b_knee  # ankle_3

    return np.clip(actions, -1.0, 1.0)

# --- Run simulation ---
NUM_STEPS = 1500
env = zeno.ZenoEnv(mjcf_path="assets/ant.xml", num_envs=1, seed=42)
env.reset()

frames = []
for step in range(NUM_STEPS):
    actions = sine_gait(step, 1, env.action_dim)
    env.step(actions)
    pos = env.get_body_positions()[0]  # (num_bodies, 4)
    frames.append(pos.copy())
env.close()

frames = np.array(frames)  # (steps, bodies, 4)
torso_z = frames[:, 1, 2]
torso_x = frames[:, 1, 0]

# Body indices: 0=world, 1=torso, 2-3=FR leg, 4-5=FL leg, 6-7=BR leg, 8-9=BL leg
LINKS = [
    (1, 2), (2, 3),   # front-right
    (1, 4), (4, 5),   # front-left
    (1, 6), (6, 7),   # back-right
    (1, 8), (8, 9),   # back-left
]
LEG_COLORS = ["#ff6b6b", "#ff6b6b", "#4ecdc4", "#4ecdc4",
              "#ffd93d", "#ffd93d", "#a8e6cf", "#a8e6cf"]
LEG_NAMES = ["FR", "FL", "BR", "BL"]

# --- Figure setup ---
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Zeno — Ant Locomotion (Trot Gait)", color="#e0e0e0",
             fontsize=14, fontweight="bold", y=0.97)

gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                      left=0.06, right=0.96, top=0.92, bottom=0.08)

ax_top = fig.add_subplot(gs[0, 0])
ax_side = fig.add_subplot(gs[1, 0])
ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
ax_h = fig.add_subplot(gs[1, 1])

for ax in [ax_top, ax_side, ax_h]:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#777", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")

ax_3d.set_facecolor("#16213e")
ax_3d.tick_params(colors="#777", labelsize=6)
for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor("#333")

# --- Top view (x-y) ---
ax_top.set_title("Top View", color="#ccc", fontsize=10)
ax_top.set_xlabel("x", color="#888", fontsize=8)
ax_top.set_ylabel("y", color="#888", fontsize=8)
ax_top.set_aspect("equal")
top_torso = Circle((0, 0), 0.12, fc="#ff6b6b", ec="white", lw=1.2, zorder=5)
ax_top.add_patch(top_torso)
top_links = [ax_top.plot([], [], "-o", color=LEG_COLORS[i], lw=2.5, ms=4,
             markeredgecolor="white", markeredgewidth=0.5, zorder=4)[0]
             for i in range(len(LINKS))]
top_trail, = ax_top.plot([], [], "-", color="#ff6b6b", alpha=0.25, lw=1, zorder=1)

# --- Side view (x-z) ---
ax_side.set_title("Side View", color="#ccc", fontsize=10)
ax_side.set_xlabel("x", color="#888", fontsize=8)
ax_side.set_ylabel("z", color="#888", fontsize=8)
ax_side.axhline(0, color="#3a5a3a", lw=8, solid_capstyle="butt", zorder=1)
ax_side.axhline(0, color="#2a4a2a", lw=1, zorder=2)
side_torso = Circle((0, 0.75), 0.12, fc="#ff6b6b", ec="white", lw=1.2, zorder=5)
ax_side.add_patch(side_torso)
side_links = [ax_side.plot([], [], "-o", color=LEG_COLORS[i], lw=2.5, ms=4,
              markeredgecolor="white", markeredgewidth=0.5, zorder=4)[0]
              for i in range(len(LINKS))]

# --- 3D view ---
ax_3d.set_title("3D View", color="#ccc", fontsize=10)
ax_3d.set_xlabel("x", color="#888", fontsize=7, labelpad=1)
ax_3d.set_ylabel("y", color="#888", fontsize=7, labelpad=1)
ax_3d.set_zlabel("z", color="#888", fontsize=7, labelpad=1)
ax_3d.view_init(elev=25, azim=-60)
lines_3d = [ax_3d.plot([], [], [], "-o", color=LEG_COLORS[i], lw=2, ms=3,
            markeredgecolor="white", markeredgewidth=0.3)[0]
            for i in range(len(LINKS))]
torso_3d, = ax_3d.plot([], [], [], "o", color="#ff6b6b", ms=10,
                        markeredgecolor="white", markeredgewidth=1, zorder=5)

# --- Height plot ---
ax_h.set_title("Torso Height", color="#ccc", fontsize=10)
ax_h.set_xlabel("Step", color="#888", fontsize=8)
ax_h.set_ylabel("z (m)", color="#888", fontsize=8)
ax_h.set_xlim(0, NUM_STEPS)
ax_h.set_ylim(-0.1, max(torso_z.max(), 1.0) * 1.15)
ax_h.axhline(0, color="#3a5a3a", lw=2)
height_line, = ax_h.plot([], [], "-", color="#ffd93d", lw=1.2)
height_dot, = ax_h.plot([], [], "o", color="#ff6b6b", ms=5, zorder=4)
info_text = fig.text(0.5, 0.02, "", ha="center", color="#aaa", fontsize=10,
                     fontfamily="monospace")

# Trail buffer for top view
TRAIL_LEN = 200

def update(frame_idx):
    pos = frames[frame_idx]
    tx, ty, tz = pos[1, 0], pos[1, 1], pos[1, 2]

    # --- Top view ---
    top_torso.center = (tx, ty)
    ax_top.set_xlim(tx - 1.0, tx + 1.0)
    ax_top.set_ylim(ty - 1.0, ty + 1.0)
    for ln, (a, b) in zip(top_links, LINKS):
        ln.set_data([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]])
    # Trail
    start = max(0, frame_idx - TRAIL_LEN)
    top_trail.set_data(torso_x[start:frame_idx+1], frames[start:frame_idx+1, 1, 1])

    # --- Side view ---
    side_torso.center = (tx, tz)
    ax_side.set_xlim(tx - 1.2, tx + 1.2)
    ax_side.set_ylim(-0.15, 1.5)
    for ln, (a, b) in zip(side_links, LINKS):
        ln.set_data([pos[a, 0], pos[b, 0]], [pos[a, 2], pos[b, 2]])

    # --- 3D view ---
    torso_3d.set_data_3d([tx], [ty], [tz])
    ax_3d.set_xlim(tx - 0.8, tx + 0.8)
    ax_3d.set_ylim(ty - 0.8, ty + 0.8)
    ax_3d.set_zlim(-0.1, 1.2)
    for ln, (a, b) in zip(lines_3d, LINKS):
        ln.set_data_3d([pos[a, 0], pos[b, 0]],
                       [pos[a, 1], pos[b, 1]],
                       [pos[a, 2], pos[b, 2]])

    # --- Height plot ---
    height_line.set_data(np.arange(frame_idx + 1), torso_z[:frame_idx + 1])
    height_dot.set_data([frame_idx], [torso_z[frame_idx]])

    fwd = tx - frames[0, 1, 0]
    info_text.set_text(f"step {frame_idx:4d}    z = {tz:.3f} m    fwd = {fwd:+.2f} m")

    return ()


sample = list(range(0, NUM_STEPS, 2))
ani = animation.FuncAnimation(fig, update, frames=sample, blit=False, interval=33)
ani.save("ant_sim.mp4", writer="ffmpeg", fps=30, dpi=130)
print(f"Saved ant_sim.mp4  ({len(sample)} frames, {NUM_STEPS} steps)")
plt.close()
