# Performance

Zeno targets high-throughput batched reinforcement-learning workloads on Apple
Silicon. Performance results must be interpreted together with the workload and
measurement method; throughput alone does not establish physics equivalence.

## Measured engine-pipeline throughput

These results use real repository MJCF models and the full `World.step` path on
an Apple M4 Pro (12-core CPU, 16-core GPU, 24 GiB unified memory), with 1,024
environments and 1,000 steps. Values are medians of five runs on September 2,
2026 using macOS 26.7 (25G227) and Zig 0.16.0.

| Environment | Wall time | Environment steps/second |
|-------------|----------:|-------------------------:|
| Pendulum | 416 ms | 2.46M |
| Cartpole | 409 ms | 2.50M |
| Pusher | 741 ms | 1.38M |
| Ant | 712 ms | 1.44M |
| Humanoid | 3,014 ms | 0.34M |

The benchmark measures host-observed completion of Zeno's own pipeline. The
per-stage profiling counters currently measure CPU command-encoding time, not
GPU stage duration.

This baseline follows the geometry-transform correction that composes body and
local geom transforms, derives capsule orientation from MJCF `fromto`, fixes
sphere–sphere normal direction, and completes the sphere/capsule/box/plane
contact matrix. It also includes the exact sphere-cylinder and cylinder-plane
work used by Pusher, plus the split position/velocity contact solver whose
friction impulses survive XPBD velocity reconstruction. Humanoid is substantially slower than the
earlier, invalid baseline because correctly oriented capsules generate contacts
the previous Z-aligned approximation missed. The lower number is the
authoritative result. The latest repetition ranges were 369–489 ms for
Pendulum, 400–421 ms for Cartpole, 701–753 ms for Pusher, 684–730 ms for Ant,
and 2,954–3,099 ms for Humanoid.

## Bounded long-horizon correctness results

The native integration suite includes deliberately narrow invariants rather
than a general stability claim:

| Case | Horizon | Verified result |
|------|--------:|-----------------|
| Two 0.1 m spheres stacked on a plane | 2,000 × 0.002 s | Settles at 0.095/0.289 m, zero final speed, and never exceeds initial height |
| 0.1 m sphere sliding at 1 m/s | 1,000 × 0.002 s | μ=0 retains 1.000 m/s; μ=1 reaches rolling at 0.708 m/s with ω≈7.11 rad/s |
| Driven bundled Pendulum | 5,000 × 0.002 s | Maximum hinge-anchor drift 24.5 mm and weld-length error 4.3 mm in the measured run; gates are 30/6 mm |
| Collision-free free sphere | 10,000 × 0.002 s | Position error 1.42 mm; quaternion norm error 1.2×10⁻⁷; linear/angular speed² drift 0.000129/0.00186 |

The stack regression caught and now prevents an earlier failure that launched
the two bodies above 2 m by applying the same stale penetration correction once
per solver iteration. The Pendulum result is evidence of a bounded tested case,
not exact rigidity. The free-body speed proxies cover only zero gravity,
collision-free motion with gyroscopic forces disabled; they are not a general
energy-conservation result.

## Full-engine batch scaling

The same isolated benchmark runs the complete Ant `World.step` pipeline across
five batch sizes. Every point uses 10 warm-up steps, 1,000 measured steps, five
fresh worlds, and the median; latency is wall time per synchronized batch step.

| Environments | Batch-step latency | Environment steps/second | Reported state memory | Throughput vs 64 |
|-------------:|-------------------:|-------------------------:|----------------------:|-----------------:|
| 64 | 0.326 ms | 0.196M | 0.4 MB | 1.00× |
| 256 | 0.448 ms | 0.571M | 1.5 MB | 2.91× |
| 1,024 | 0.712 ms | 1.438M | 6.0 MB | 7.33× |
| 4,096 | 0.940 ms | 4.359M | 24.0 MB | 22.20× |
| 16,384 | 3.129 ms | 5.237M | 95.9 MB | 26.68× |

Throughput keeps increasing through 16,384 environments on this machine, while
the marginal gain falls sharply after 4,096. Memory grows approximately
linearly. This supports the verified batch range; it does not imply that 16,384
is optimal for every model. The checked summary and all sorted samples are at
`benchmarks/results/m4-pro-full-world-scaling-2026-09-02.json`.

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

The aggregate target may execute independent benchmark programs concurrently.
For publishable engine-pipeline measurements, run the full simulator workload
in isolation and repeat it from an otherwise idle machine:

```bash
zig build bench-full-physics -Doptimize=ReleaseFast
```

Masked stepping has a separate compact-dispatch benchmark. It compares native
active-environment dispatch with the former strategy of running the full batch
and restoring inactive public state through CPU memcpy. It covers 1,024 and
4,096 Ant environments at several active fractions:

```bash
zig build bench-masked-step -Doptimize=ReleaseFast
```

Compact dispatch leaves every inactive buffer untouched, including solver
warm-start and contact-cache state. The legacy reference restores the 13 public
state buffers supported by the former implementation, so it is useful as a
historical performance baseline but has weaker inactive-state semantics.

### Compact masked-step result

Measured on September 2, 2026 using an Apple M4 Pro with 24 GiB unified memory,
macOS 26.7 (25G227), and Zig 0.16.0. Each variant used the Ant model, one fixed
physics substep, 10 warm-up steps, 200 measured steps, and the median of five
fresh-world repetitions. The dense GPU column is an unmasked `World.step`
control measured by the same harness.

| Batch | Active | Compact wall time | Dense GPU wall time | Legacy wall time | Compact vs dense | Compact vs legacy |
|------:|-------:|------------------:|--------------------:|-----------------:|-----------------:|------------------:|
| 1,024 envs | 50% | 113.0 ms | 153.8 ms | 201.3 ms | 1.36× | 1.78× |
| 4,096 envs | 50% | 171.4 ms | 196.1 ms | 427.8 ms | 1.14× | 2.50× |
| 4,096 envs | 25% | 155.9 ms | 189.4 ms | 481.1 ms | 1.22× | 3.09× |
| 4,096 envs | 10% | 104.1 ms | 194.1 ms | 602.8 ms | 1.86× | 5.79× |

“Batch-equivalent” counts allocated environments per completed masked step so
the ratio is directly comparable with the legacy full-batch path; it is not a
claim that inactive environments were simulated. At 10% active, compact
dispatch processed 0.788M actual active environment-steps/s (7.87M
batch-equivalent). These results
establish the benefit of active-environment indirection for this workload and
machine, not a general Zeno-to-Isaac Gym or Zeno-to-MuJoCo speed comparison.

### GPU task-output result

`bench-task-outputs` compares an otherwise identical fixed-substep Ant pipeline
with task evaluation disabled, with reward/termination evaluated by Metal, and
with the same arithmetic performed by a host loop after every synchronized
step:

```bash
zig build bench-task-outputs -Doptimize=ReleaseFast
```

On the same M4 Pro provenance, using 10 warm-up steps, 300 measured steps, and
five interleaved fresh-world repetitions with rotating variant order, the Metal
and CPU paths produced identical final reward/done checksums.

| Batch | Physics only | Metal task | CPU task | Metal overhead | Metal vs CPU task path |
|------:|-------------:|-----------:|---------:|---------------:|-----------------------:|
| 1,024 envs | 1.36M env-steps/s | 1.35M | 1.34M | 0.54% | 1.01× |
| 4,096 envs | 4.21M env-steps/s | 4.17M | 4.10M | 0.92% | 1.02× |

This is an end-to-end Zeno comparison, not isolated kernel timing. The bounded
result shows that task evaluation can stay in the existing command stream at
0.5–0.9% measured overhead. The 1,024-environment Metal/CPU difference is
within run noise; the 4,096 case showed a small throughput advantage. The
architectural gain is removal of the per-step host task loop, not a universal
speedup or reference-task reward-equivalence claim.

### Asynchronous submission and overlap result

`bench-async-step` compares sequential fixed-substep Ant physics plus 250 µs of
synthetic CPU work against committing the same Metal step first and performing
that CPU work before `step_wait`:

```bash
zig build bench-async-step -Doptimize=ReleaseFast
```

Measured on the same M4 Pro using 10 warm-up steps, 300 measured steps, and five
interleaved fresh-world repetitions:

| Batch | Sequential throughput | Overlapped throughput | Mean submission | Overlap ratio |
|------:|----------------------:|----------------------:|----------------:|--------------:|
| 1,024 envs | 1.00M env-steps/s | 1.11M env-steps/s | 55.9 µs | 1.11× |
| 4,096 envs | 3.33M env-steps/s | 3.63M env-steps/s | 116.3 µs | 1.09× |

The benchmark's independent fresh-world final-state checksums differed by
0.010% and 0.0088%, respectively; contact atomics are not deterministic across
independent runs. A 25-step same-action regression separately verifies
synchronous and split-submit state equivalence within `1e-5`. The 4,096 case
shows modest overlap for this synthetic CPU workload at both batch sizes.
Neither is a guaranteed policy-training gain.

### Fused autoreset result

`bench-autoreset` compares a complete compact Metal reset followed by a full
step as two synchronized submissions against Gymnasium-style reset-before-step
encoded in one command buffer:

```bash
zig build bench-autoreset -Doptimize=ReleaseFast
```

Measured on September 2, 2026 using the same Apple M4 Pro with 24 GiB unified
memory, macOS 26.7 (25G227), and Zig 0.16.0. Each case used Ant, one fixed
substep, 10 warm-up steps, 200 measured steps, and the median of five interleaved
fresh-world repetitions.

| Batch | Reset fraction | Separate throughput | Fused throughput | Speedup |
|------:|---------------:|--------------------:|-----------------:|--------:|
| 1,024 envs | 50% | 1.05M env-steps/s | 1.22M | 1.17× |
| 4,096 envs | 50% | 3.11M env-steps/s | 3.58M | 1.15× |
| 4,096 envs | 10% | 3.40M env-steps/s | 3.89M | 1.15× |

The 50% final-state checksums matched exactly; the 10% fresh-world checksums
differed by 0.0069%, consistent with independent contact-atomic scheduling. A
separate pendulum regression compares the fused and two-submission paths within
`1e-5` and verifies that fused reset does not erase the next action batch. These
figures isolate command-submission removal, not end-to-end trainer speedup.

### Shared action-buffer result

`bench-action-staging` compares the ordinary per-step copy into Metal shared
memory with submitting actions already written through the shared action view:

```bash
zig build bench-action-staging -Doptimize=ReleaseFast
```

On the same M4 Pro provenance, fixed-action Ant runs used one substep, 10 warm-up
steps, 500 measured steps, and five interleaved fresh-world repetitions.

| Batch | Copied throughput | Shared-buffer throughput | Ratio |
|------:|------------------:|-------------------------:|------:|
| 1,024 envs | 1.36M env-steps/s | 1.35M | 0.99× |
| 4,096 envs | 4.35M env-steps/s | 4.24M | 0.97× |

The difference is within run noise: removing a 32 KiB or 128 KiB host memcpy is
too small relative to this full physics workload to establish a speedup. The
useful result is the writable destination contract for policies that can emit
directly into existing NumPy storage; Zeno does not claim zero-copy PyTorch MPS
or JAX device-tensor interop. Independent contact-atomic checksum differences
were 0.266% and 0.617%; a deterministic pendulum regression separately verifies
copied and shared-action submission within `1e-5`.

### Python zero-copy output result

On the same machine and software snapshot, the Ant Python benchmark at 1,024
environments used 50 warm-up steps, three trials of 200 steps, and identical
preallocated actions. Enabling `zero_copy_outputs` increased observed mean
throughput from 1.336M to 1.348M environment-steps/s (1.01×). At 64 Ant
environments over 500 measured steps, median step latency changed from 0.359 ms
to 0.399 ms. These small sequential-run differences are sensitive to system
noise; the durable result is removal of the output memcpy, not a guaranteed
speedup ratio.

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

### Measured Metal versus sequential CPU throughput

The improved comparison harness uses fixed seeded controls, symmetric warm-up
and reset discipline, five fresh repetitions, medians plus full ranges, and
machine-readable Apple hardware/software provenance. On the M4 Pro configuration
above, its 1,024-environment Pendulum run measured:

| Execution path | Median wall time | Range | Environment steps/second |
|----------------|-----------------:|------:|-------------------------:|
| Zeno batched Metal Python API | 0.455 s | 0.437–0.513 s | 2.252M |
| MuJoCo sequential CPU Python API | 2.309 s | 2.178–2.524 s | 0.443M |

The observed throughput ratio was 5.08× for these execution strategies. It is
not a physics-equivalence or matched-simulator speedup claim: the engines use
different solver, contact, and observation semantics, and the CPU reference is
an intentionally sequential `MjData` loop. The complete five-sample record is
stored at `benchmarks/results/m4-pro-pendulum-cpu-comparison-2026-09-02.json`
in the repository.

Reproduce it from the repository root with:

```bash
PYTHONPATH=python python3 benchmarks/compare_mujoco.py \
  --model assets/pendulum.xml --envs 1024 --steps 1000 \
  --warmup 10 --repeats 5 --seed 42 --json-output result.json
```

## Supported hardware

- macOS 13 or newer
- Apple Silicon with Metal support
- Zig 0.16.x for source builds

The published re-baseline used an Apple M4 Pro with 24 GiB unified memory. Results
from other systems should include the Metal device, macOS version, Zig version,
configuration, warm-up, repetition count, and aggregation method.
