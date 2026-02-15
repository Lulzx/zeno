"""3D swarm visualization — animated scatter with neighbor edges, trails, and metrics."""

import sys
sys.path.insert(0, '/Users/lulzx/work/zeno/python')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from zeno._ffi import _lib, ffi
from zeno.swarm import create_swarm_world

# --- Parameters ---
NUM_AGENTS = 64
NUM_STEPS = 600
COMM_RANGE = 3.0
TRAIL_LEN = 30
FRAME_SKIP = 2
FPS = 30
DPI = 130

# --- Create swarm world ---
print(f"Creating swarm world with {NUM_AGENTS} agents...")
world, swarm = create_swarm_world(
    num_agents=NUM_AGENTS,
    layout="circle",
    spacing=0.5,
    communication_range=COMM_RANGE,
)
world.reset()

num_bodies = world.num_bodies  # num_agents + 1 (ground)
rng = np.random.default_rng(42)
null_actions = np.zeros((1, max(world.action_dim, 1)), dtype=np.float32)

# --- Pre-simulate and collect data ---
print(f"Simulating {NUM_STEPS} steps...")
all_positions = []       # (steps, num_agents, 3)
all_neighbor_counts = [] # (steps, num_agents)
all_edges = []           # (steps, list of (i,j))
all_metrics = []         # (steps, SwarmMetrics)

# Agents have freejoints but no actuators — apply Brownian motion by
# directly setting positions each step for organic drifting.
# Order: world.step → set_body_positions → swarm.step (so swarm sees updated pos).
pos_state = world.get_body_positions(zero_copy=False).copy()  # (1, num_bodies, 4)
agent_z_base = pos_state[0, 1, 2]

for step in range(NUM_STEPS):
    # Brownian drift: small random xy displacements
    drift_xy = rng.normal(0, 0.04, (NUM_AGENTS, 2)).astype(np.float32)
    pos_state[0, 1:NUM_AGENTS + 1, 0] += drift_xy[:, 0]
    pos_state[0, 1:NUM_AGENTS + 1, 1] += drift_xy[:, 1]
    pos_state[0, 1:NUM_AGENTS + 1, 2] = agent_z_base

    world.step(null_actions, substeps=1)
    world.set_body_positions(pos_state)
    swarm.step()

    agent_pos = pos_state[0, 1:NUM_AGENTS + 1, :3].copy()
    all_positions.append(agent_pos)

    # Neighbor counts
    ncounts = swarm.get_neighbor_counts().copy()
    all_neighbor_counts.append(ncounts)

    # Metrics
    m = swarm.get_metrics()
    all_metrics.append(m)

    # Extract edges from CSR
    edges = []
    raw_row = _lib.zeno_swarm_get_neighbor_row_ptr(swarm._handle)
    raw_col = _lib.zeno_swarm_get_neighbor_index_ptr(swarm._handle)
    if raw_row != ffi.NULL and raw_col != ffi.NULL:
        row_ptr = np.frombuffer(
            ffi.buffer(raw_row, (NUM_AGENTS + 1) * 4), dtype=np.uint32
        ).copy()
        num_edges = int(row_ptr[NUM_AGENTS])
        if num_edges > 0:
            col_idx = np.frombuffer(
                ffi.buffer(raw_col, num_edges * 4), dtype=np.uint32
            ).copy()
            for i in range(NUM_AGENTS):
                for j in range(int(row_ptr[i]), int(row_ptr[i + 1])):
                    if col_idx[j] > i:  # upper-triangle
                        edges.append((i, int(col_idx[j])))
    all_edges.append(edges)

    if (step + 1) % 100 == 0:
        print(f"  step {step + 1}/{NUM_STEPS}  edges={m.total_edges}  "
              f"connectivity={m.connectivity_ratio:.2f}")

all_positions = np.array(all_positions)        # (steps, agents, 3)
all_neighbor_counts = np.array(all_neighbor_counts)  # (steps, agents)

# Metrics time series
connectivity_hist = [m.connectivity_ratio for m in all_metrics]
edges_hist = [m.total_edges for m in all_metrics]

print("Setting up animation...")

# --- Figure setup (dark theme) ---
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0f0f1a")
fig.suptitle("Zeno — 3D Swarm Visualization", color="#e0e0e0",
             fontsize=14, fontweight="bold", y=0.97)

gs = GridSpec(2, 2, hspace=0.35, wspace=0.30,
              left=0.06, right=0.96, top=0.92, bottom=0.06)

# --- Panel 1: 3D view (top-left) ---
ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
ax_3d.set_facecolor("#16213e")
ax_3d.tick_params(colors="#777", labelsize=6)
for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor("#333")
ax_3d.set_title("3D View", color="#ccc", fontsize=10)
ax_3d.set_xlabel("x", color="#888", fontsize=7, labelpad=1)
ax_3d.set_ylabel("y", color="#888", fontsize=7, labelpad=1)
ax_3d.set_zlabel("z", color="#888", fontsize=7, labelpad=1)

# --- Panel 2: Top-down XY view (top-right) ---
ax_top = fig.add_subplot(gs[0, 1])
ax_top.set_facecolor("#16213e")
ax_top.tick_params(colors="#777", labelsize=8)
for spine in ax_top.spines.values():
    spine.set_color("#333")
ax_top.set_title("Top-Down (X-Y)", color="#ccc", fontsize=10)
ax_top.set_xlabel("x", color="#888", fontsize=8)
ax_top.set_ylabel("y", color="#888", fontsize=8)
ax_top.set_aspect("equal")

# --- Panel 3: Metrics dashboard (bottom-left) ---
ax_dash = fig.add_subplot(gs[1, 0])
ax_dash.set_facecolor("#16213e")
for spine in ax_dash.spines.values():
    spine.set_color("#333")
ax_dash.set_xticks([])
ax_dash.set_yticks([])
ax_dash.set_title("Metrics Dashboard", color="#ccc", fontsize=10)

# --- Panel 4: Timeline (bottom-right) ---
ax_time = fig.add_subplot(gs[1, 1])
ax_time.set_facecolor("#16213e")
ax_time.tick_params(colors="#777", labelsize=8)
for spine in ax_time.spines.values():
    spine.set_color("#333")
ax_time.set_title("Timeline", color="#ccc", fontsize=10)
ax_time.set_xlabel("Step", color="#888", fontsize=8)
ax_time.set_xlim(0, NUM_STEPS)
ax_time.set_ylim(0, 1.05)

# Timeline second y-axis for edges
ax_time2 = ax_time.twinx()
ax_time2.tick_params(colors="#777", labelsize=8)
for spine in ax_time2.spines.values():
    spine.set_color("#333")
max_edges = max(edges_hist) if max(edges_hist) > 0 else 1
ax_time2.set_ylim(0, max_edges * 1.15)
ax_time2.set_ylabel("Edges", color="#ffd93d", fontsize=8)
ax_time.set_ylabel("Connectivity", color="#4ecdc4", fontsize=8)

# --- Create artists ---

# 3D scatter
scat_3d = ax_3d.scatter([], [], [], c=[], cmap="viridis", s=40,
                         edgecolors="white", linewidths=0.3, vmin=0,
                         vmax=max(6, all_neighbor_counts.max()),
                         depthshade=True)
# 3D ground plane
gnd_x = np.array([-8, 8, 8, -8])
gnd_y = np.array([-8, -8, 8, 8])
gnd_z = np.array([0, 0, 0, 0])
ax_3d.plot_trisurf(gnd_x, gnd_y, gnd_z, color="#16213e", alpha=0.3,
                    edgecolor="none")

# 3D edge lines (pre-allocate a collection)
edge_lines_3d = []

# Top-down scatter
scat_top = ax_top.scatter([], [], c=[], cmap="viridis", s=40,
                           edgecolors="white", linewidths=0.3, vmin=0,
                           vmax=max(6, all_neighbor_counts.max()))
# Top-down trail scatter
trail_scat = ax_top.scatter([], [], c=[], cmap="Greys", s=8, alpha=0.4)

# Top-down edge lines
edge_lines_top = []

# Dashboard text
dash_text = ax_dash.text(0.5, 0.5, "", transform=ax_dash.transAxes,
                          ha="center", va="center", fontsize=11,
                          fontfamily="monospace", color="#e0e0e0",
                          linespacing=1.8)

# Timeline lines
line_conn, = ax_time.plot([], [], "-", color="#4ecdc4", lw=1.5, label="Connectivity")
line_edges, = ax_time2.plot([], [], "-", color="#ffd93d", lw=1.5, label="Edges")
ax_time.legend(loc="upper left", fontsize=7, facecolor="#16213e",
               edgecolor="#333", labelcolor="#ccc")
ax_time2.legend(loc="upper right", fontsize=7, facecolor="#16213e",
                edgecolor="#333", labelcolor="#ccc")

# Info text at bottom
info_text = fig.text(0.5, 0.01, "", ha="center", color="#aaa", fontsize=10,
                     fontfamily="monospace")

# Compute axis limits from all positions
all_x = all_positions[:, :, 0]
all_y = all_positions[:, :, 1]
all_z = all_positions[:, :, 2]
pad = 1.0
x_lim = (all_x.min() - pad, all_x.max() + pad)
y_lim = (all_y.min() - pad, all_y.max() + pad)
z_lim = (min(0, all_z.min()) - 0.5, max(all_z.max(), 1.0) + 0.5)


def update(frame_idx):
    step = frame_idx * FRAME_SKIP
    if step >= NUM_STEPS:
        step = NUM_STEPS - 1

    pos = all_positions[step]       # (agents, 3)
    ncounts = all_neighbor_counts[step]
    edges = all_edges[step]
    m = all_metrics[step]

    # --- 3D View ---
    # Remove old edge lines
    for ln in edge_lines_3d:
        ln.remove()
    edge_lines_3d.clear()

    scat_3d._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    scat_3d.set_array(ncounts.astype(float))

    # Draw edges in 3D
    for (i, j) in edges:
        ln, = ax_3d.plot([pos[i, 0], pos[j, 0]],
                         [pos[i, 1], pos[j, 1]],
                         [pos[i, 2], pos[j, 2]],
                         color="#4ecdc4", alpha=0.15, lw=0.5)
        edge_lines_3d.append(ln)

    ax_3d.set_xlim(*x_lim)
    ax_3d.set_ylim(*y_lim)
    ax_3d.set_zlim(*z_lim)
    # Slowly rotate camera
    ax_3d.view_init(elev=25, azim=-60 + frame_idx * 0.5)

    # --- Top-down XY ---
    for ln in edge_lines_top:
        ln.remove()
    edge_lines_top.clear()

    scat_top.set_offsets(pos[:, :2])
    scat_top.set_array(ncounts.astype(float))

    # Draw edges in top view
    for (i, j) in edges:
        ln, = ax_top.plot([pos[i, 0], pos[j, 0]],
                          [pos[i, 1], pos[j, 1]],
                          color="#4ecdc4", alpha=0.15, lw=0.5)
        edge_lines_top.append(ln)

    # Trails: last TRAIL_LEN positions as fading dots
    trail_start = max(0, step - TRAIL_LEN)
    trail_pos = all_positions[trail_start:step + 1]  # (T, agents, 3)
    if len(trail_pos) > 1:
        # Flatten to (T*agents, 2) with alpha fading
        t_len = len(trail_pos)
        trail_xy = trail_pos[:, :, :2].reshape(-1, 2)
        alphas = np.repeat(np.linspace(0.05, 0.3, t_len), NUM_AGENTS)
        trail_scat.set_offsets(trail_xy)
        trail_scat.set_array(alphas)
    else:
        trail_scat.set_offsets(np.empty((0, 2)))

    ax_top.set_xlim(*x_lim)
    ax_top.set_ylim(*y_lim)

    # --- Metrics Dashboard ---
    dash_text.set_text(
        f"Step:  {step:4d} / {NUM_STEPS}\n"
        f"Connectivity:   {m.connectivity_ratio:.3f}\n"
        f"Fragmentation:  {m.fragmentation_score:.3f}\n"
        f"Total Edges:    {m.total_edges:5d}\n"
        f"Messages:       {m.message_count:5d}\n"
        f"Avg Neighbors:  {m.avg_neighbors:.2f}\n"
        f"Collisions:     {m.collision_count:5d}"
    )

    # --- Timeline ---
    steps_so_far = np.arange(step + 1)
    line_conn.set_data(steps_so_far, connectivity_hist[:step + 1])
    line_edges.set_data(steps_so_far, edges_hist[:step + 1])

    # Info bar
    info_text.set_text(
        f"step {step:4d}    agents={NUM_AGENTS}    "
        f"edges={m.total_edges}    connectivity={m.connectivity_ratio:.3f}"
    )

    return ()


# --- Animate ---
sample = list(range(0, NUM_STEPS // FRAME_SKIP))
print(f"Rendering {len(sample)} frames at {FPS}fps...")
ani = animation.FuncAnimation(fig, update, frames=sample, blit=False,
                               interval=1000 // FPS)
ani.save("swarm_3d.mp4", writer="ffmpeg", fps=FPS, dpi=DPI)
print(f"Saved swarm_3d.mp4  ({len(sample)} frames, {NUM_STEPS} steps)")
plt.close()
