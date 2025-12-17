# Zeno: High-Performance Batched Robotics Simulation Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macOS-blue.svg)](https://www.apple.com/macos/)

**Zeno** is a GPU-accelerated rigid body physics simulation engine optimized for reinforcement learning and robot policy training. It is designed from first principles to exploit Apple Silicon's unified memory architecture, achieving 10-100x throughput improvements over existing solutions for batched parallel environments.

The name references Zeno of Elea, whose paradoxes on motion and infinity are foundational to physics and mathematics — fitting for a simulation engine that discretizes continuous motion into parallel computation.

## Features

- **Native Metal Compute**: Hand-written Metal shaders for maximum GPU utilization
- **Unified Memory**: Zero-copy data transfer between CPU and GPU
- **Batched Simulation**: Simulate thousands of environments in parallel
- **MJCF Support**: Compatible with MuJoCo XML model format
- **Gymnasium Integration**: Standard RL environment interface
- **Minimal Dependencies**: Pure Zig + Metal, no heavy frameworks

## Performance

| Metric | Zeno | MuJoCo |
|--------|------|--------|
| 1024 Ant envs, 1000 steps | < 1 second | ~45 seconds |
| Single env step latency | < 50 μs | ~200 μs |
| Memory per env (Ant) | < 4 KB | ~16 KB |

## Quick Start

### Building from Source

Requirements:
- macOS 13+ (Ventura or later)
- Zig 0.13+ (https://ziglang.org/download/)
- Apple Silicon (M1/M2/M3/M4) or Intel Mac with Metal support

```bash
# Clone the repository
git clone https://github.com/lulzx/zeno.git
cd zeno

# Build the library
zig build -Doptimize=ReleaseFast

# Run tests
zig build test
```

### Python Installation

```bash
cd python
pip install -e .
```

### Basic Usage (Python)

```python
import zeno
import numpy as np

# Create environment with 1024 parallel instances
env = zeno.make("ant.xml", num_envs=1024)

# Reset all environments
obs = env.reset()

# Run simulation
for _ in range(1000):
    # Random actions
    actions = np.random.uniform(-1, 1, (1024, env.action_dim))

    # Step all environments in parallel
    obs, rewards, dones, info = env.step(actions)

    # Reset done environments
    if np.any(dones):
        env.reset(mask=dones)

env.close()
```

### Gymnasium Integration

```python
import gymnasium as gym
import zeno.gym  # Register environments

# Single environment
env = gym.make("Zeno/Ant-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

# Vectorized environments
from zeno.gym import make_vec

envs = make_vec("ant", num_envs=1024)
obs, info = envs.reset()
```

### Basic Usage (Zig)

```zig
const zeno = @import("zeno");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load scene from MJCF
    var scene = try zeno.mjcf.parser.parseFile(allocator, "ant.xml");
    defer scene.deinit();

    // Create world with 1024 environments
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 1024,
        .timestep = 0.002,
    });
    defer world.deinit();

    // Simulation loop
    var actions = [_]f32{0.0} ** (1024 * 8);
    for (0..1000) |_| {
        try world.step(&actions, 0);
    }

    // Zero-copy access to observations
    const obs = world.getObservations();
    std.debug.print("Observation[0]: {}\n", .{obs[0]});
}
```

## Supported Models

Zeno includes several standard robotics environments:

| Environment | Bodies | Joints | Actions | Description |
|-------------|--------|--------|---------|-------------|
| Pendulum | 3 | 1 | 1 | Simple inverted pendulum |
| Cartpole | 4 | 2 | 1 | Classic cart-pole balancing |
| Ant | 9 | 8 | 8 | Quadruped locomotion |
| Humanoid | 14 | 13 | 13 | Bipedal humanoid walking |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                             │
│                   (cffi, zero-copy numpy)                       │
├─────────────────────────────────────────────────────────────────┤
│                          C ABI                                  │
├─────────────────────────────────────────────────────────────────┤
│                       Zig Runtime                               │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│   │   World     │ │   State     │ │   Metal     │               │
│   │  (scene)    │ │  (SoA data) │ │  (compute)  │               │
│   └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                  Unified Memory Pool                            │
│               (MTLBuffer, storageModeShared)                    │
├─────────────────────────────────────────────────────────────────┤
│                    Compute Pipeline                             │
│   Apply Actions → FK → Forces → Integrate → Collision → Solve  │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Python API

```python
# Environment creation
env = zeno.make(model, num_envs=1, timestep=0.002, ...)

# Properties
env.num_envs          # Number of parallel environments
env.observation_dim   # Observation dimension
env.action_dim        # Action dimension
env.timestep          # Physics timestep

# Methods
obs = env.reset(mask=None)                    # Reset environments
obs, rewards, dones, info = env.step(actions) # Step simulation
positions = env.get_body_positions()          # Get body positions
quaternions = env.get_body_quaternions()      # Get body orientations
```

### C API

```c
// World lifecycle
ZenoWorldHandle zeno_world_create(const char* mjcf_path, const ZenoConfig* config);
void zeno_world_destroy(ZenoWorldHandle world);

// Simulation
void zeno_world_step(ZenoWorldHandle world, const float* actions, uint32_t substeps);
void zeno_world_reset(ZenoWorldHandle world, const uint8_t* env_mask);

// State access (zero-copy pointers)
float* zeno_world_get_observations(ZenoWorldHandle world);
float* zeno_world_get_rewards(ZenoWorldHandle world);
uint8_t* zeno_world_get_dones(ZenoWorldHandle world);
```

## Project Structure

```
zeno/
├── build.zig                 # Build configuration
├── src/
│   ├── main.zig              # C ABI exports
│   ├── metal/                # Metal infrastructure
│   ├── physics/              # Physics core
│   ├── collision/            # Collision detection
│   ├── world/                # World management
│   ├── mjcf/                 # MJCF parser
│   └── shaders/              # Metal compute shaders
├── python/
│   └── zeno/                 # Python bindings
├── assets/                   # MJCF model files
├── tests/                    # Zig tests
├── benchmarks/               # Performance benchmarks
└── docs/                     # Documentation
```

## MJCF Compatibility

Zeno supports a subset of the MuJoCo XML format:

### Supported Elements
- `<option>`: timestep, gravity
- `<body>`: name, pos, quat
- `<joint>`: type (hinge, slide, ball, free), axis, range, damping
- `<geom>`: type (sphere, capsule, box, plane), size, mass, friction
- `<actuator>`: motor, ctrlrange, gear
- `<sensor>`: jointpos, jointvel, accelerometer, gyro

### Not Yet Supported
- Tendons
- Equality constraints
- Mesh geometry
- Heightfield terrain
- Soft bodies

## Benchmarking

```bash
# Run Zig benchmarks
zig build bench

# Run Python comparison
cd benchmarks
python compare_mujoco.py --envs 1024 --steps 1000
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- MuJoCo: https://mujoco.org/
- Position Based Dynamics: Müller et al., 2007
- Metal Best Practices: https://developer.apple.com/metal/

## Citation

```bibtex
@software{zeno2025,
  title = {Zeno: High-Performance Batched Robotics Simulation Engine},
  author = {Lulzx},
  year = {2025},
  url = {https://github.com/lulzx/zeno}
}
```
