# Zeno

**High-Performance Batched Robotics Simulation Engine**

Zeno is a GPU-accelerated rigid body physics simulation engine optimized for reinforcement learning and robot policy training. It is designed from first principles to exploit Apple Silicon's unified memory architecture, achieving **10-100x throughput improvements** over existing solutions for batched parallel environments.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Blazing Fast__

    ---

    Simulate 1024 environments in parallel with native Metal compute shaders.

-   :material-memory:{ .lg .middle } __Zero-Copy Memory__

    ---

    Unified memory architecture eliminates CPU-GPU transfer overhead.

-   :material-robot:{ .lg .middle } __RL-Optimized__

    ---

    Built for reinforcement learning with Gymnasium integration.

-   :material-file-code:{ .lg .middle } __MJCF Compatible__

    ---

    Load existing MuJoCo XML models with tendons, constraints, and terrain.

</div>

## Features

- **Collision Primitives**: Sphere, capsule, box, cylinder, plane, mesh, heightfield
- **Joint Types**: Free, ball, hinge, slide, fixed, universal
- **Tendons**: Fixed and spatial tendons with spring behavior
- **Equality Constraints**: Weld, connect, joint, tendon constraints
- **Sensors**: Joint position/velocity, accelerometer, gyro, frame pose
- **Actuators**: Motor, position servo, velocity servo
- **Soft Bodies**: PBD deformable cloth and volumetric bodies
- **Fluids**: SPH fluid simulation with spatial hashing
- **Materials**: PBR materials with texture support

## Performance

Benchmarked on Apple M4 Pro with real MJCF models:

| Environment | 1024 envs × 1000 steps | vs MuJoCo |
|-------------|------------------------|-----------|
| Pendulum    | 206 ms | **9.7x faster** |
| Cartpole    | 157 ms | **19.1x faster** |
| Ant         | 174 ms | **258x faster** |
| Humanoid    | 172 ms | **697x faster** |

**Average: 246x faster than MuJoCo**

## Quick Example

```python
import zeno
import numpy as np

# Create environment with 1024 parallel instances
env = zeno.make("ant.xml", num_envs=1024)

# Reset all environments
obs = env.reset()

# Run simulation
for _ in range(1000):
    actions = np.random.uniform(-1, 1, (1024, env.action_dim))
    obs, rewards, dones, info = env.step(actions)

env.close()
```

## Why Zeno?

The name references **Zeno of Elea**, whose paradoxes on motion and infinity are foundational to physics and mathematics — fitting for a simulation engine that discretizes continuous motion into parallel computation.

### Design Philosophy

1. **Unified Memory First**: All state lives in shared memory accessible by both CPU and GPU
2. **Throughput Over Fidelity**: Optimized for RL training, not engineering simulation
3. **Minimal Abstraction**: Direct Metal API access without intermediate frameworks

## Getting Started

<div class="grid cards" markdown>

-   [:material-download: __Installation__](getting-started/installation.md)

    ---

    Build from source and install Python bindings

-   [:material-play: __Quick Start__](getting-started/quickstart.md)

    ---

    Run your first simulation in minutes

</div>

## License

Zeno is released under the [MIT License](https://github.com/lulzx/zeno/blob/main/LICENSE).
