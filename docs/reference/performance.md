# Performance

Zeno targets high-throughput batched reinforcement-learning workloads on Apple
Silicon. Performance results must be interpreted together with the workload and
measurement method; throughput alone does not establish physics equivalence.

## Measured engine-pipeline throughput

These results use real repository MJCF models and the full `World.step` path on
an Apple M4 Pro (14-core CPU, 20-core GPU), with 1,024 environments and 1,000
steps. Values are medians of five runs from the July 2026 re-baseline.

| Environment | Wall time | Environment steps/second |
|-------------|----------:|-------------------------:|
| Pendulum | 345 ms | 2.97M |
| Cartpole | 375 ms | 2.73M |
| Ant | 699 ms | 1.47M |
| Humanoid | 1,367 ms | 0.75M |

The benchmark measures host-observed completion of Zeno's own pipeline. The
per-stage profiling counters currently measure CPU command-encoding time, not
GPU stage duration.

## Synthetic kernel scaling

`bench_envs` exercises simplified standalone Metal kernels. It is useful for
studying batch scaling and dispatch overhead, but it does not run the complete
engine and must not be compared directly with the table above.

| Workload | Environments | Wall time for 1,000 iterations |
|----------|-------------:|-------------------------------:|
| Pendulum-like | 1,024 | 15 ms |
| Cartpole-like | 1,024 | 50 ms |
| Ant-like | 1,024 | 45 ms |
| Humanoid-like | 1,024 | 69 ms |
| Ant-like | 4,096 | 138 ms |
| Ant-like | 16,384 | 833 ms |

Run the benchmarks locally to record results for the current compiler, OS, and
GPU:

```bash
zig build bench
```

Benchmark output identifies whether a workload is synthetic or uses
`World.step`, and prints the Metal device where available.

## Why batching helps

- Unified memory avoids explicit CPU-to-GPU state copies on Apple Silicon.
- One command stream processes many environments, amortizing dispatch cost.
- Structure-of-arrays buffers provide contiguous, aligned GPU access.
- Fixed solver iteration counts keep the workload predictable.

The best batch size depends on the model, contact density, solver settings,
device, and available memory. Measure the actual training workload instead of
assuming a universal optimum.

## Configuration tradeoffs

Smaller timesteps, additional substeps, and more contact iterations generally
increase computation and can improve stability, but they do not by themselves
guarantee accuracy. Validate each task against invariants or a trusted reference.

```python
# Throughput-oriented configuration for a simple task
env = zeno.make(
    "ant.xml",
    num_envs=1024,
    timestep=0.01,
    contact_iterations=2,
)

# More conservative configuration for contact-rich testing
env = zeno.make(
    "ant.xml",
    num_envs=1024,
    timestep=0.001,
    contact_iterations=8,
    substeps=4,
)
```

## Profiling

The built-in profiling API is useful for host-side regression comparisons:

```python
env = zeno.make("ant.xml", num_envs=1024, enable_profiling=True)
for _ in range(100):
    env.step(actions)
print(env.get_timing())
```

Use Xcode Instruments with Metal System Trace for actual GPU-stage timing.

## Cross-simulator comparisons

Zeno does not publish a MuJoCo, Newton, Isaac, or Brax speedup claim. A valid
comparison requires matched model semantics, integration timestep, solver work,
contacts, observations, termination behavior, hardware provenance, warm-up, and
synchronization. The current repository does not yet contain that full matrix.

`benchmarks/compare_mujoco.py` can run both implementations on one machine as a
throughput experiment, but its result is not evidence of physics equivalence.
`benchmarks/validate_physics.py` runs a live free-body ballistic trajectory in
both engines and fails when its explicit error threshold is exceeded.

## Supported hardware

- macOS 13 or newer
- Apple Silicon with Metal support
- Zig 0.16.x for source builds

The published re-baseline used an Apple M4 Pro with 48 GB unified memory. Results
from other systems should include the Metal device, macOS version, Zig version,
configuration, warm-up, repetition count, and aggregation method.
