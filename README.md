# Zeno: Batched Rigid-Body Simulation for Apple Silicon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macOS-blue.svg)](https://www.apple.com/macos/)
[![CI](https://github.com/lulzx/zeno/actions/workflows/ci.yml/badge.svg)](https://github.com/lulzx/zeno/actions/workflows/ci.yml)

**Zeno** is a Metal-native, massively batched rigid-body physics engine written in Zig, built for reinforcement-learning workloads on Apple Silicon. It is designed around unified memory: simulation state lives in shared `MTLBuffer`s that Python sees as zero-copy numpy arrays, and thousands of environments step in parallel on the GPU.

The niche it targets is specific: **Isaac Gym–style batched simulation, but for Macs.** Most robotics simulation stacks assume NVIDIA GPUs or CPU portability; Zeno is built from first principles for Apple's GPU and memory architecture.

The name references Zeno of Elea, whose paradoxes on motion and infinity are foundational to physics and mathematics — fitting for a simulation engine that discretizes continuous motion into parallel computation.

## Project Status

Zeno is a **research engine, not a validated MuJoCo replacement**. Honest framing of where things stand:

- **Solid**: the staged Metal compute pipeline, batched SoA state layout, unified-memory zero-copy access, MJCF loading, Python/Gymnasium bindings, and the XPBD solver skeleton with graph-coloring parallelism.
- **Implemented but not rigorously validated**: contact stability under stacking, friction behavior, joint drift over long horizons, energy conservation, and agreement with trusted reference simulators. "Runs and looks plausible" is a much lower bar than "production-correct physics," and Zeno has not yet earned the higher one.
- **Experimental**: soft bodies (PBD cloth/volumetric), SPH fluids, PBR materials/rendering, tendons, and the swarm platform. These exist behind the Zig API and have unit tests, but should be treated as prototypes.

If you need physics you can trust unconditionally today, use MuJoCo. If you want high-throughput batched rollouts on a Mac and can tolerate a young engine, Zeno is for you.

## Features

### Core Engine
- **Native Metal Compute** — Hand-written MSL shaders, staged compute pipeline (actions → forces → integrate → collision → constraint solve → sensors)
- **Unified Memory** — Zero-copy CPU↔GPU via Apple Silicon shared memory
- **Batched Simulation** — 1,024 to 16,384+ parallel environments
- **SoA Memory Layout** — float4-aligned, coalesced GPU access

### Physics
- **Rigid Body Dynamics** — Semi-implicit Euler integration, quaternion rotations with renormalization
- **XPBD Constraint Solver** — Extended Position-Based Dynamics with graph coloring for race-free parallel solving
- **Joint Constraints** — Fixed, revolute, prismatic, ball, free, universal joints
- **Collision Detection** — Spatial hashing broad phase, GJK+EPA for convex hulls, sphere/capsule/box/plane/mesh/heightfield primitives
- **Contact Resolution** — XPBD contact solver with warm starting and contact caching for temporal coherence
- **Adaptive Substeps** — Dynamic substep adjustment based on constraint violation

### Integration
- **MJCF Parser** — Bodies, joints, geoms, actuators, sensors, defaults, inertia (explicit and shape-based)
- **Python Bindings** — cffi-based, zero-copy numpy arrays via unified memory
- **Gymnasium API** — Vectorized environment support
- **Stable-Baselines3** — Direct integration with SB3 and other RL libraries
- **C ABI** — FFI surface for custom language bindings

### Experimental (Zig API only, prototype quality)
- **Soft Bodies** — PBD deformable cloth and volumetric bodies
- **Fluids** — SPH fluid simulation with spatial hashing
- **Materials** — PBR material definitions (texture decoding is stubbed)
- **Swarm** — multi-agent grid/graph/message-bus layer on top of the physics world

## Performance

**What these numbers are:** wall-clock throughput of Zeno's own pipeline on its supported workload subset, measured on an Apple M4 Pro (14-core CPU, 20-core GPU). They are **not** a semantics-matched comparison with MuJoCo: MuJoCo does richer physics per step (different solver, contact model, and numerical tolerances), and Zeno's benchmark harness includes simplified kernel paths. A GPU engine stepping thousands of simplified environments will always look dramatically faster than a CPU engine doing more work per step — treat cross-simulator ratios as throughput ratios, not physics-equivalence claims.

### Engine pipeline throughput (real MJCF models, full `World.step`)

| Environment | 1024 envs × 1000 steps | Throughput |
|-------------|------------------------|-----------------|
| Pendulum    | 206 ms                 | 4.97M steps/sec |
| Cartpole    | 157 ms                 | 6.52M steps/sec |
| Ant         | 174 ms                 | 5.89M steps/sec |
| Humanoid    | 172 ms                 | 5.95M steps/sec |

For reference, single-threaded MuJoCo on the same machine steps these models 10–700× slower in wall-clock terms — but see the caveat above before quoting that as a physics speedup.

### Scaling (synthetic GPU benchmark)

| Environment | Envs  | Time  |
|-------------|-------|-------|
| Pendulum    | 1024  | 15ms  |
| Cartpole    | 1024  | 50ms  |
| Ant         | 1024  | 45ms  |
| Humanoid    | 1024  | 69ms  |
| Ant         | 4096  | 138ms |
| Ant         | 16384 | 833ms |

A rigorous MuJoCo parity benchmark (matched model semantics, timestep, solver iterations, contact counts, observations, and termination logic) is planned but does not exist yet. Until it does, Zeno makes no apples-to-apples speedup claim.

## Quick Start

### Building from Source

Requirements:
- macOS 13+ (Ventura or later)
- Zig 0.16.x (CI and local verification use 0.16.0)
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
pip install -e python/
```

### Running Tests

```bash
# Zig unit tests
zig build test

# Python integration tests
pip install pytest numpy
python -m pytest tests/test_python_integration.py -v
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

# Vectorized environments (native GPU batching)
from zeno.gym import make_vec

envs = make_vec("ant", num_envs=1024)
obs, info = envs.reset()
```

### Zero-Copy State Access

```python
# Direct access to GPU memory (no copy overhead)
positions = env._world.get_body_positions(zero_copy=True)
velocities = env._world.get_body_velocities(zero_copy=True)

# Modify state directly (changes reflected on GPU)
positions[0, 0, 2] += 0.1

# Checkpointing
state = env._world.get_state()
env._world.set_state(state)  # Restore later
```

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from zeno.gym import make_sb3_env

env = make_sb3_env("ant", num_envs=8)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
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

## Included Environments

Zeno ships 10 standard robotics environments as MJCF assets:

| Environment | Bodies | Joints | Actuators | Description |
|-------------|--------|--------|-----------|-------------|
| Pendulum | 3 | 1 | 1 | Simple inverted pendulum |
| Cartpole | 3 | 2 | 1 | Classic cart-pole balancing |
| Ant | 9 | 9 | 8 | Quadruped locomotion |
| Humanoid | 14 | 14 | 13 | Bipedal humanoid walking |
| HalfCheetah | 8 | 6 | 6 | Fast running cheetah |
| Hopper | 5 | 3 | 3 | Single-leg hopping |
| Walker2d | 8 | 6 | 6 | Bipedal walking |
| Swimmer | 4 | 2 | 2 | 3-link swimmer |
| Reacher | 5 | 2 | 2 | 2-link planar arm reaching |
| Pusher | 6 | 5 | 3 | 3-DOF arm pushing object |

These load and run in batch; their reward/termination semantics are Zeno's own and have not been verified to match the Gymnasium/MuJoCo reference implementations.

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
│  Actions → Forces → Integrate → Collision → XPBD Solve → Obs   │
└─────────────────────────────────────────────────────────────────┘
```

Shaders are embedded in the binary and compiled at runtime via `newLibraryWithSource`. That keeps development simple (no separate compile step) at the cost of startup compile time and weaker offline diagnostics; precompiled `.metallib` support is on the roadmap.

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
├── .github/workflows/        # CI pipeline (GitHub Actions)
├── src/
│   ├── main.zig              # C ABI exports
│   ├── metal/                # Metal infrastructure
│   ├── physics/              # Physics core (rigid bodies, XPBD; experimental soft bodies, fluids)
│   ├── collision/            # Collision detection (GJK+EPA, spatial hashing)
│   ├── world/                # World management
│   ├── mjcf/                 # MJCF parser (with inertia computation)
│   ├── render/               # Rendering (materials; experimental)
│   ├── swarm/                # Multi-agent swarm layer (experimental)
│   └── shaders/              # Metal compute shaders (embedded at compile time)
├── python/
│   └── zeno/                 # Python bindings
├── assets/                   # MJCF model files
├── tests/                    # Zig + Python tests
├── benchmarks/               # Performance benchmarks
└── docs/                     # Documentation
```

## MJCF Compatibility

Zeno supports a subset of the MuJoCo XML format:

### Supported Elements
- `<option>`: timestep, gravity
- `<default>`: joint/geom default classes with inheritance
- `<body>`: name, pos, quat, euler
- `<joint>`: type (hinge, slide, ball, free), axis, range, damping, stiffness, armature
- `<geom>`: type (sphere, capsule, box, cylinder, plane, mesh, hfield), size, fromto, mass, density, friction
- `<inertial>`: mass, diaginertia, fullinertia, pos, quat (with automatic shape-based fallback)
- `<actuator>`: motor, position, velocity, ctrlrange, forcerange, gear, kp, kv
- `<sensor>`: jointpos, jointvel, framepos, framequat, framelinvel, frameangvel, accelerometer, gyro
- `<tendon>`: fixed and spatial tendons with spring behavior, wrapping objects
- `<equality>`: weld, connect, joint, tendon constraints
- `<asset>`: mesh loading (STL, OBJ) with convex hull approximation, heightfield terrain

Parsing an element is not the same as matching MuJoCo's runtime semantics for it; expect behavioral differences, especially around contacts, solver parameters, and actuator dynamics.

### Not Yet Supported
- MJCF composite bodies
- MJCF flexcomp
- Advanced solver options (CG, Newton)

## Benchmarking

```bash
# Run Zig benchmarks
zig build bench

# Run Python comparison
cd benchmarks
python compare_mujoco.py --envs 1024 --steps 1000
```

The benchmark suite currently mixes four different kinds of measurement; be careful which one you cite:

1. **Synthetic kernel benchmarks** — standalone Metal workloads with simplified integration/constraint kernels; useful for GPU tuning, not physics claims.
2. **Engine pipeline benchmarks** — full `World.step` over real MJCF models; this is the number that describes Zeno itself.
3. **Cross-simulator comparisons** — wall-clock vs MuJoCo without matched semantics; throughput indication only.
4. **Semantic-equivalence benchmarks** — matched model/solver/tolerance comparisons; **not yet implemented**.

## Comparison with Alternatives

| Simulator | Platform | Backend | Batched | Differentiable |
|-----------|----------|---------|---------|----------------|
| **Zeno** | macOS | Metal | Yes | No |
| MuJoCo | Cross-platform | CPU | No | No |
| Newton | Linux | CUDA/Warp | Yes | Yes |
| Isaac Lab | Linux | CUDA | Yes | Yes |
| Brax | Cross-platform | JAX/XLA | Yes | Yes |

Zeno fills a niche none of these target: GPU-accelerated batched simulation on Apple Silicon. It is younger and less validated than all of them; what it offers is throughput on hardware the others ignore.

## Roadmap

- Semantic-equivalence benchmark harness against MuJoCo (matched models, solver settings, tolerances)
- Physics validation suite: stacking stability, friction cones, joint drift, energy behavior, long-horizon determinism
- Precompiled `.metallib` shader artifacts alongside runtime source compilation
- A backend dispatch boundary so the compute stages can be implemented by more than one GPU backend
- Zig 0.16 stdlib migration

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
  title = {Zeno: Batched Rigid-Body Simulation Engine for Apple Silicon},
  author = {Lulzx},
  year = {2025},
  url = {https://github.com/lulzx/zeno}
}
```
