"""Basic swarm simulation example.

Creates a swarm world, steps it, and prints metrics.
"""

import zeno
from zeno.swarm import create_swarm_world

# Create a 64-agent swarm on a grid layout
world, swarm = create_swarm_world(
    num_agents=64,
    layout="grid",
    spacing=0.5,
    communication_range=5.0,
)

# Reset the world
world.reset()

# Step the simulation
for step in range(100):
    swarm.step()
    world.step(actions=None, substeps=1)

    if step % 10 == 0:
        metrics = swarm.get_metrics()
        print(f"Step {step:3d}: "
              f"edges={metrics.total_edges:4d}  "
              f"avg_neighbors={metrics.avg_neighbors:.1f}  "
              f"connectivity={metrics.connectivity_ratio:.3f}  "
              f"fragmentation={metrics.fragmentation_score:.0f}")

# Final metrics
metrics = swarm.get_metrics()
print(f"\nFinal metrics after 100 steps:")
print(f"  Total edges: {metrics.total_edges}")
print(f"  Average neighbors: {metrics.avg_neighbors:.1f}")
print(f"  Connectivity ratio: {metrics.connectivity_ratio:.3f}")
print(f"  Fragmentation (components): {metrics.fragmentation_score:.0f}")
print(f"  Messages delivered: {metrics.message_count}")

# Neighbor counts
counts = swarm.get_neighbor_counts()
print(f"  Min neighbors: {counts.min()}")
print(f"  Max neighbors: {counts.max()}")
