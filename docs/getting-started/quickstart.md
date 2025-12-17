# Quick Start

This guide will get you running your first Zeno simulation in minutes.

## Basic Python Usage

### Create an Environment

```python
import zeno
import numpy as np

# Load an Ant model with 1024 parallel environments
env = zeno.make("assets/ant.xml", num_envs=1024)

print(f"Observation dim: {env.observation_dim}")
print(f"Action dim: {env.action_dim}")
```

### Run a Simulation Loop

```python
# Reset all environments
obs = env.reset()

# Simulation loop
for step in range(1000):
    # Generate random actions
    actions = np.random.uniform(-1, 1, (1024, env.action_dim)).astype(np.float32)

    # Step all environments in parallel
    obs, rewards, dones, info = env.step(actions)

    # Reset any done environments
    if np.any(dones):
        env.reset(mask=dones)

    if step % 100 == 0:
        print(f"Step {step}: mean reward = {rewards.mean():.3f}")

env.close()
```

## Gymnasium Integration

Zeno integrates seamlessly with Gymnasium for RL training.

### Single Environment

```python
import gymnasium as gym
import zeno.gym  # Register Zeno environments

env = gym.make("Zeno/Ant-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized Environments

```python
from zeno.gym import make_vec

# Create 1024 parallel environments
envs = make_vec("ant", num_envs=1024)
obs, info = envs.reset()

for _ in range(1000):
    # Actions for all environments
    actions = envs.action_space.sample()
    obs, rewards, terminateds, truncateds, infos = envs.step(actions)

envs.close()
```

## Zig Usage

For maximum performance, use Zeno directly from Zig.

```zig
const std = @import("std");
const zeno = @import("zeno");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load scene from MJCF
    var scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    defer scene.deinit();

    // Create world with 1024 environments
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 1024,
        .timestep = 0.002,
    });
    defer world.deinit();

    // Simulation loop
    var actions: [1024 * 8]f32 = undefined;
    for (&actions) |*a| a.* = 0.0;

    for (0..1000) |_| {
        try world.step(&actions, 0);
    }

    // Access observations (zero-copy)
    const obs = world.getObservations();
    std.debug.print("First observation: {d:.3}\n", .{obs[0]});
}
```

## Available Models

Zeno includes several standard robotics environments:

| Environment | File | Actions | Description |
|-------------|------|---------|-------------|
| Pendulum | `pendulum.xml` | 1 | Simple inverted pendulum |
| Cartpole | `cartpole.xml` | 1 | Classic cart-pole balancing |
| Ant | `ant.xml` | 8 | Quadruped locomotion |
| Humanoid | `humanoid.xml` | 13 | Bipedal humanoid walking |

## Next Steps

- Learn about the [Python API](../guide/python-api.md) in detail
- Understand [MJCF model format](../guide/mjcf.md)
- Explore the [Architecture](../reference/architecture.md)
- Check out [RL Training examples](../examples/rl-training.md)
