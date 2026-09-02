# Status and scope

Zeno is an Apple-Silicon research engine for batched reinforcement-learning
simulation. It favors parallel throughput and direct access to unified memory.
It does not attempt to reproduce every MuJoCo feature or numerical choice.

## Ready to use

- Batched structure-of-arrays state in shared `MTLBuffer` storage
- A staged Metal step: actions, forces, integration, contacts, constraints,
  sensors, observations, and optional task outputs
- Synchronous and split-submit asynchronous stepping
- Compact masked stepping and fused reset-before-step
- Zero-copy NumPy views for state, actions, and outputs
- MJCF loading for the documented subset
- Python, Gymnasium vector, Stable-Baselines3, Zig, and C interfaces
- Full-world batch runs verified from 64 through 16,384 environments

## Bounded physics support

The GPU contact path covers every pair among sphere, capsule, box, and plane.
It also covers sphere-cylinder and cylinder-plane contacts used by the bundled
Pusher model. Other cylinder pairs, meshes, and heightfields are outside that
contact boundary.

Regression tests cover narrow cases such as two-sphere settling,
sliding-to-rolling friction, driven Pendulum joint drift, collision-free motion,
and a ballistic comparison. They do not establish general accuracy or parity
with another simulator.

The exact cases and error bounds are in [Performance and
evidence](reference/performance.md). Parser coverage is in [MJCF
subset](guide/mjcf.md).

## Experimental

These systems exist in the Zig API and have tests, but remain prototypes:

- PBD cloth and volumetric soft bodies
- SPH fluids
- PBR material and rendering support
- Tendons beyond the main batched RL path
- The swarm grid, graph, and message bus

## Not promised

- MuJoCo-equivalent dynamics, rewards, termination, or solver behavior
- Engineering-grade physical validation
- Determinism across independent GPU runs with contact atomics
- CUDA, Vulkan, Linux, or Intel Mac support
- Zero-copy PyTorch MPS or JAX device-tensor interop
- Differentiable simulation

Use MuJoCo when trusted reference dynamics matter more than native Mac GPU
batching. Use Zeno when the workload fits the boundary above and throughput on
Apple hardware is the point.
