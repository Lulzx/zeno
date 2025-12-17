# Zeno Architecture

## Overview

Zeno is designed as a layered architecture optimized for GPU-accelerated physics simulation on Apple Silicon. The key design principles are:

1. **Unified Memory First**: All state lives in shared memory accessible by both CPU and GPU
2. **Throughput Over Fidelity**: Optimized for RL training, not engineering simulation
3. **Minimal Abstraction**: Direct Metal API access without intermediate frameworks

## System Layers

### Python Layer

The top-level interface for users. Provides:
- NumPy-compatible arrays with zero-copy access to GPU memory
- Gymnasium-compatible environment interface
- High-level `make()` and `ZenoEnv` API

```
zeno/
├── __init__.py      # Public API
├── _ffi.py          # CFFI bindings
├── env.py           # Main environment class
└── gym/             # Gymnasium integration
```

### C ABI Layer

Stable C interface for language bindings:
- Opaque handle-based API
- Zero-copy pointer returns
- Error codes for error handling

### Zig Runtime

Core implementation in Zig:

```
src/
├── main.zig         # C ABI exports
├── metal/           # GPU infrastructure
├── physics/         # Physics types and math
├── collision/       # Collision detection
├── world/           # Simulation orchestration
└── mjcf/            # Model parsing
```

### Unified Memory Pool

All simulation state is stored in Metal buffers with `storageModeShared`:

- Direct CPU read/write without copies
- Direct GPU access via compute shaders
- Automatic cache coherency on Apple Silicon

### Compute Pipeline

Each physics step executes these kernels in sequence:

1. **apply_actions**: Map control inputs to joint torques
2. **forward_kinematics**: Compute body transforms from joint state
3. **compute_forces**: Gravity, springs, damping
4. **integrate**: Semi-implicit Euler position/velocity update
5. **broad_phase**: Spatial hashing for collision pairs
6. **narrow_phase**: Exact contact detection
7. **solve_contacts**: PBD constraint projection (iterated)
8. **read_sensors**: Populate observation buffer

## Memory Layout

All arrays use Structure of Arrays (SoA) layout for coalesced GPU access:

```
positions[num_envs * num_bodies]     # float4 aligned
velocities[num_envs * num_bodies]    # float4 aligned
quaternions[num_envs * num_bodies]   # float4 aligned
joint_positions[num_envs * num_joints]
joint_velocities[num_envs * num_joints]
actions[num_envs * num_actuators]
observations[num_envs * obs_dim]
contacts[num_envs * max_contacts]    # Padded struct
```

**Indexing convention:**
```
body_index = env_id * num_bodies + body_id
joint_index = env_id * num_joints + joint_id
```

## Physics Model

### Integration

Semi-implicit Euler:
```
v(t+dt) = v(t) + a(t) * dt
x(t+dt) = x(t) + v(t+dt) * dt
```

Angular integration uses quaternion exponential map for stability.

### Constraints

Position-Based Dynamics (PBD) with fixed iteration count:
- Avoid convergence checks that cause GPU thread divergence
- Trade accuracy for parallel efficiency
- 4-8 iterations typical for RL workloads

### Collision Detection

**Broad phase**: Spatial hashing with Morton encoding
- O(n) update, O(1) query per cell
- Grid size auto-tuned to body dimensions

**Narrow phase**: Primitive-specific algorithms
- Sphere-sphere: Analytic distance
- Sphere-capsule: Point-segment distance
- Capsule-capsule: Segment-segment distance
- Box-plane: Vertex-plane SAT

## Thread Model

Each kernel is dispatched with threads proportional to work:

| Kernel | Threads |
|--------|---------|
| apply_actions | num_envs × num_actuators |
| forward_kinematics | num_envs × num_bodies |
| integrate | num_envs × num_bodies |
| broad_phase | num_envs × num_geoms |
| narrow_phase | num_envs × max_contacts |
| solve_contacts | num_envs × max_contacts |
| read_sensors | num_envs × num_sensors |

Threadgroup size is auto-tuned to GPU capabilities (typically 256).

## Performance Considerations

### Memory Bandwidth

- Unified memory eliminates CPU↔GPU transfer cost
- SoA layout maximizes GPU cache utilization
- float4 alignment enables SIMD loads

### Compute Efficiency

- Fixed iteration counts prevent thread divergence
- No dynamic memory allocation during stepping
- Atomic operations only for contact counting

### Scalability

- Linear scaling with environment count up to GPU limits
- 16K environments typical maximum
- Memory-limited at ~32MB for 16K Ant environments
