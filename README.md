# Zeno: High-Performance Batched Robotics Simulation Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macOS-blue.svg)](https://www.apple.com/macos/)

**Zeno** is a GPU-accelerated rigid body physics simulation engine optimized for reinforcement learning and robot policy training. It is designed from first principles to exploit Apple Silicon's unified memory architecture, achieving 10-100x throughput improvements over existing solutions for batched parallel environments.

The name references Zeno of Elea, whose paradoxes on motion and infinity are foundational to physics and mathematics — fitting for a simulation engine that discretizes continuous motion into parallel computation.

## Features

### Core Engine
- **Native Metal Compute** — Hand-written MSL shaders, 8-stage compute pipeline
- **Unified Memory** — Zero-copy CPU↔GPU via Apple Silicon shared memory
- **Batched Simulation** — 1,024 to 16,384+ parallel environments
- **SoA Memory Layout** — float4-aligned, coalesced GPU access

### Physics
- **Rigid Body Dynamics** — Semi-implicit Euler integration, quaternion rotations
- **Joint Constraints** — Fixed, revolute, prismatic, ball, free (PBD solver)
- **Collision Detection** — Spatial hashing broad phase, sphere/capsule/box/plane primitives
- **Contact Resolution** — Position-Based Dynamics with Coulomb friction

### Integration
- **MJCF Parser** — Bodies, joints, geoms, actuators, sensors, defaults
- **Python Bindings** — cffi-based, zero-copy numpy arrays
- **Gymnasium API** — `gym.make("Zeno/Ant-v0")` compatible
- **C ABI** — Full FFI for custom language bindings

### Environments
- **Pendulum** — 3 bodies, 1 joint, 1 actuator
- **Cartpole** — 3 bodies, 2 joints, 1 actuator
- **Ant** — 9 bodies, 9 joints, 8 actuators
- **Humanoid** — 14 bodies, 14 joints, 13 actuators

## Performance

Benchmarked on Apple M4 Pro (14-core CPU, 20-core GPU) with real MJCF models:

| Environment | 1024 envs × 1000 steps | vs MuJoCo | Throughput |
|-------------|------------------------|-----------|------------|
| Pendulum    | 206 ms                 | **9.7x**  | 4.97M steps/sec |
| Cartpole    | 157 ms                 | **19.1x** | 6.52M steps/sec |
| Ant         | 174 ms                 | **258x**  | 5.89M steps/sec |
| Humanoid    | 172 ms                 | **697x**  | 5.95M steps/sec |

**Average speedup: 246x faster than MuJoCo**

### Scaling Performance (GPU Benchmark)

| Environment | Envs | Time | Target | Speedup |
|-------------|------|------|--------|---------|
| Pendulum    | 1024 | 15ms | 50ms   | 3.4x ✓ |
| Cartpole    | 1024 | 50ms | 80ms   | 1.6x ✓ |
| Ant         | 1024 | 45ms | 800ms  | 17.9x ✓ |
| Humanoid    | 1024 | 69ms | 2000ms | 29.1x ✓ |
| Ant         | 4096 | 138ms | 3000ms | 21.8x ✓ |
| Ant         | 16384 | 833ms | 10000ms | 12.0x ✓ |

## Quick Start

### Building from Source

Requirements:
- macOS 13+ (Ventura or later)
- Zig 0.15+ (https://ziglang.org/download/)
- Apple Silicon (M1/M2/M3/M4) recommended

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

| Environment | Bodies | Joints | Actuators | Description |
|-------------|--------|--------|-----------|-------------|
| Pendulum | 3 | 1 | 1 | Simple inverted pendulum |
| Cartpole | 3 | 2 | 1 | Classic cart-pole balancing |
| Ant | 9 | 9 | 8 | Quadruped locomotion |
| Humanoid | 14 | 14 | 13 | Bipedal humanoid walking |

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
- `<default>`: joint/geom default classes (parsed but not applied)
- `<body>`: name, pos, quat, euler
- `<joint>`: type (hinge, slide, ball, free), axis, range, damping, stiffness, armature
- `<geom>`: type (sphere, capsule, box, cylinder, plane), size, fromto, mass, density, friction
- `<actuator>`: motor, position, velocity, ctrlrange, forcerange, gear, kp, kv
- `<sensor>`: jointpos, jointvel, framepos, framequat, framelinvel, frameangvel, accelerometer, gyro

### Not Yet Supported
- Tendons
- Equality constraints
- Mesh geometry
- Heightfield terrain
- Soft bodies
- Default class inheritance

## Benchmarking

```bash
# Run Zig benchmarks
zig build bench

# Run Python comparison
cd benchmarks
python compare_mujoco.py --envs 1024 --steps 1000
```

## Comparison with Alternatives

| Simulator | Platform | Backend | Batched | Differentiable |
|-----------|----------|---------|---------|----------------|
| **Zeno** | macOS | Metal | Yes | No |
| MuJoCo | Cross-platform | CPU | No | No |
| Newton | Linux | CUDA/Warp | Yes | Yes |
| Isaac Lab | Linux | CUDA | Yes | Yes |
| Brax | Cross-platform | JAX/XLA | Yes | Yes |

Zeno fills a unique niche: **GPU-accelerated batched simulation for Apple Silicon**. If you need to train RL policies on a Mac, Zeno provides throughput comparable to NVIDIA-based solutions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- MuJoCo: https://mujoco.org/
- Newton: https://github.com/newton-physics/newton
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
