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

Full-step velocity snapshots and acceleration derivation are also encoded as
Metal kernels in the same command buffer. Steady-state stepping does not walk
the batched body-velocity arrays on the CPU. Synchronous `step` commits and
waits once; `stepAsync` commits the identical command stream and returns, while
`waitStep` performs completion/error checking and makes shared-memory outputs
safe to consume. Each world permits one in-flight step so actions, dispatch
metadata, and state cannot be overwritten while its GPU work is pending.

The action buffer is also exposed as a writable shared-memory view.
`stepCurrentActionsAsync` submits the existing contents directly, avoiding the
normal NumPy-to-action-buffer memcpy when a policy can target that storage.
Like other shared views it is only CPU-safe while no Metal step is pending;
Zeno does not currently expose an MPSGraph, PyTorch MPS, or JAX device-buffer
handle with cross-command synchronization.

`stepSubset` compacts its byte mask into a short list of active environment IDs
in shared memory, then dispatches the physics kernels over that list. Each
kernel maps its compact dispatch slot back to the stable physical environment
index before accessing state. Inactive body, joint, observation, contact, RNG,
warm-start constraint, and temporal contact-cache state is therefore preserved
by omission: it is neither simulated nor backed up and restored. Simulation
state remains GPU-resident; the only per-call CPU work is scanning the control
mask and writing active IDs.

When configured, `compute_task_outputs` runs after the final physics substep in
the same Metal command buffer. It derives locomotion reward, health/horizon
termination, and per-environment episode clocks directly from GPU-resident
state and actions. The contract is deliberately parameterized by root body,
forward axis, weights, height range, and horizon; it is a reusable primitive,
not an assertion that a named Gymnasium or MuJoCo task has identical semantics.

Reset templates are also Metal-visible. `reset_bodies` restores body and
temporal state in parallel, while `reset_env_aux` clears joint, action, contact,
constraint, reward, done, and episode-clock state. Masked reset uses the same
compact physical-environment indirection as masked stepping, so inactive
contact/cache state is not cleared accidentally. Sensor observations are
regenerated before the single reset command buffer completes.

The Gymnasium `NEXT_STEP` path can prepend that compact reset to the next full
physics step in one encoder and one command buffer. Inline reset dispatch
parameters are snapshotted separately from the full-step dispatch metadata, and
the reset kernel preserves the already-copied next action batch. A reset plus a
compact subset step is intentionally not exposed because both operations would
need different physical-ID lists in the current single compact-ID buffer.

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
rewards[num_envs]
dones[num_envs]
episode_steps[num_envs]
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

Contacts use one position-correction pass per narrow-phase penetration sample,
then reconstruct body velocities from the corrected XPBD poses. Cached normal
and two-axis tangent impulses are warm-started after reconstruction, followed by
fixed-count projected velocity iterations for restitution and a Coulomb friction
disk. Full-step acceleration is derived after those impulses so the public state
includes contact response.

### Collision Detection

**Broad phase**: Spatial hashing with Morton encoding
- O(n) update, O(1) query per cell
- Cell size is auto-tuned from shape-aware bounding radii
- Infinite planes are paired explicitly with compatible finite geoms, because
  an unbounded surface cannot be represented by one hash cell

**Metal narrow phase**: currently verified primitive-specific pairs
- Sphere-sphere: analytic distance
- Sphere-capsule: closest point on the oriented capsule segment
- Capsule-capsule: closest points between two oriented segments
- Sphere-box: closest point in the oriented box frame
- Sphere-cylinder: closest point on the exact oriented finite cylinder
- Capsule-box: exact piecewise-quadratic segment/AABB distance in box space
- Box-box: 15-axis oriented separating-axis test with one support contact
- Sphere-plane: analytic signed distance
- Capsule-plane: endpoint distance
- Oriented box-plane: projected half extent and deepest support point
- Oriented cylinder-plane: analytic finite-cylinder support point

The standalone CPU narrow-phase module contains additional algorithms, but
`World.step` does not call that module. The sphere/capsule/box/plane matrix is
covered on Metal (excluding the meaningless plane-plane pair). Sphere-cylinder
and cylinder-plane are also covered with exact finite-cylinder tests, including
the bundled Pusher object/table interaction. Other cylinder pairings, mesh, and
heightfield geometries can be parsed and enter broad phase, yet their contacts
are not currently resolved by the Metal kernel. Body and local geometry
transforms are composed on device for supported pairs, and MJCF capsule
`fromto` is converted to a local axis quaternion. This remains an explicit
support boundary, not implied coverage from the parser enum.

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
