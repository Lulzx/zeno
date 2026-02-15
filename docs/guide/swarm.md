# Swarm Platform

The Zeno swarm platform enables multi-agent simulations with neighbor detection, message passing, task evaluation, and adversarial testing. It runs on top of Zeno's physics engine and supports thousands of agents.

## Quick Start

```python
import zeno
from zeno.swarm import create_swarm_world

# Create a 64-agent swarm on a grid
world, swarm = create_swarm_world(
    num_agents=64,
    layout="grid",
    spacing=0.5,
    communication_range=5.0,
)

world.reset()

for step in range(100):
    swarm.step()
    world.step(actions=None, substeps=1)

    if step % 10 == 0:
        metrics = swarm.get_metrics()
        print(f"Step {step}: edges={metrics.total_edges}, "
              f"connectivity={metrics.connectivity_ratio:.3f}")
```

## Creating a Swarm

### `create_swarm_world()`

The easiest way to get started. Generates MJCF with a ground plane and N spherical agent bodies, then creates a physics world and swarm instance.

```python
world, swarm = create_swarm_world(
    num_agents=64,        # agents per environment
    agent_radius=0.1,     # sphere radius
    num_envs=1,           # parallel environments
    layout="grid",        # "grid", "circle", or "random"
    spacing=0.5,          # distance between agents
    communication_range=5.0,
)
```

Layouts:
- **grid**: agents placed in a square grid with the given spacing
- **circle**: agents placed evenly around a circle
- **random**: same as grid (deterministic initial placement)

### Manual Setup

For custom MJCF models, create the world and swarm separately:

```python
from zeno.swarm import ZenoSwarm, SwarmConfig

world = zeno.make(model="my_model.xml", num_envs=1)

config = SwarmConfig(
    num_agents=32,
    communication_range=8.0,
    max_neighbors=32,
    max_messages_per_step=4,
    max_message_bytes=48,
    grid_cell_size=8.0,
)

swarm = ZenoSwarm(world, config, body_offset=1)
```

The `body_offset` skips non-agent bodies (e.g., the ground plane at index 0).

## Stepping

```python
swarm.step()               # swarm step: neighbor detection + communication
world.step(actions=None)   # physics step
```

`swarm.step()` performs:
1. Spatial grid rebuild from body positions
2. Adjacency graph construction (CSR format)
3. Message delivery with optional latency/dropout
4. Metrics computation (connectivity, fragmentation, etc.)

## Metrics

```python
metrics = swarm.get_metrics()

metrics.total_edges          # total neighbor pairs
metrics.avg_neighbors        # mean neighbors per agent
metrics.connectivity_ratio   # fraction of max possible edges
metrics.fragmentation_score  # number of connected components (1 = fully connected)
metrics.message_count        # messages delivered this step
metrics.bytes_sent           # total payload bytes sent
metrics.messages_dropped     # messages lost to dropout/jamming
metrics.near_miss_count      # agents within 2x collision radius
metrics.task_success         # latest task evaluation score
```

Per-agent neighbor counts:

```python
counts = swarm.get_neighbor_counts()  # np.ndarray of shape (num_agents,)
print(f"Min: {counts.min()}, Max: {counts.max()}")
```

## Message Bus Realism

Configure realistic communication constraints through `SwarmConfig`:

```python
config = SwarmConfig(
    num_agents=100,
    communication_range=5.0,
    latency_ticks=2,                  # messages delayed 2 steps
    drop_prob=0.05,                   # 5% message loss
    max_broadcast_recipients=8,       # limit broadcast fan-out
    max_inbox_per_agent=16,           # per-agent inbox cap
    strict_determinism=True,          # deterministic dropout (seeded RNG)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latency_ticks` | `0` | Steps of delay before message delivery |
| `drop_prob` | `0.0` | Probability of dropping each message |
| `max_broadcast_recipients` | `0xFFFFFFFF` | Max recipients per broadcast |
| `max_inbox_per_agent` | `0` | Per-agent inbox limit (0 = default) |
| `strict_determinism` | `True` | Use step count as RNG seed for reproducibility |

## Task Evaluators

Evaluate cooperative objectives to score swarm performance:

```python
result = swarm.evaluate_task("formation",
    center_x=0.0, center_y=0.0,
    target_radius=10.0, formation_type=0.0)

print(f"Score: {result.score:.3f}, Complete: {result.complete}")
```

### Formation

Agents form a geometric shape around a target center.

```python
result = swarm.evaluate_task("formation",
    center_x=0, center_y=0,
    target_radius=10.0,
    formation_type=0)  # 0=circle, 1=line, 2=grid
```

Score = 1.0 - (mean_position_error / target_radius), clamped to [0, 1]. Complete when score > 0.95.

### Coverage

Agents spread out to cover a rectangular area.

```python
result = swarm.evaluate_task("coverage",
    x_min=-50, y_min=-50,
    x_max=50, y_max=50,
    cell_size=5.0)
```

Score = fraction of grid cells with at least one nearby agent. Complete when score > 0.9.

### Pursuit-Evasion

Pursuers try to capture evaders.

```python
result = swarm.evaluate_task("pursuit",
    num_pursuers=32,
    capture_radius=1.5)
```

Agents [0..num_pursuers-1] are pursuers, the rest are evaders. Score = fraction of evaders captured. Complete when all evaders are within capture radius.

### Tracking-Defense

Defenders maintain connectivity while tracking a moving target.

```python
result = swarm.evaluate_task("tracking",
    target_x=10, target_y=10, target_z=0,
    track_radius=5.0)
```

Score = (fraction in range) * connectivity_ratio. Complete when score > 0.8.

## Attack Simulation

Test swarm resilience against adversarial conditions:

```python
swarm.apply_attack(
    attack_type="byzantine",
    intensity=0.1,
    target_agents=[0, 1, 2, 3, 4],
    seed=42,
)
```

### Attack Types

| Type | Effect |
|------|--------|
| `"jamming"` | Targeted agents cannot send or receive messages |
| `"dropout"` | Random agents disconnected with probability = intensity |
| `"byzantine"` | Targeted agents' message payloads are randomized |
| `"partition"` | Graph split into two groups at threshold = num_agents * intensity |
| `"none"` | Clear any active attack |

## Deterministic Replay

Record and verify swarm execution traces:

```python
# Record
swarm.start_recording()
for _ in range(50):
    swarm.step()
swarm.stop_recording()

stats = swarm.get_replay_stats()
print(f"Recorded {stats.frame_count} frames, {stats.total_bytes} bytes")
```

Each frame captures positions, velocities, message counts, metrics, and a CRC32 checksum. Run the same simulation twice and compare checksums to verify bitwise determinism.

## Scaling

The swarm platform uses a spatial hash grid for O(n) neighbor detection and CSR adjacency graphs for efficient message routing. Typical performance on Apple M-series:

| Agents | Step Time |
|--------|-----------|
| 100 | < 0.1 ms |
| 1,000 | ~ 0.5 ms |
| 10,000 | ~ 5 ms |

Run `examples/swarm_scaling.py` to benchmark on your hardware.

## Scenarios

Pre-built scenarios are available in `assets/swarm/scenarios/`:

- `formation_circle.json` — 64 agents forming a circle
- `pursuit_evasion.json` — 32 pursuers + 8 evaders
- `byzantine_resilience.json` — 100 agents with 10% Byzantine adversaries

Load a scenario:

```python
import json
from zeno.swarm import create_swarm_world, SwarmConfig

with open("assets/swarm/scenarios/formation_circle.json") as f:
    scenario = json.load(f)

world, swarm = create_swarm_world(
    num_agents=scenario["num_agents"],
    layout=scenario["layout"],
    spacing=scenario["spacing"],
    communication_range=scenario["config"]["communication_range"],
)
```
