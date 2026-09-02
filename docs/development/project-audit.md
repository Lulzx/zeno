# Project Audit

Last verified: 2026-09-02 on Apple Silicon with Zig 0.16.0.

## Verified release surface

- `zig build test --summary all`: 250 tests across 14 native suites.
- `PYTHONPATH=python python3 -m pytest tests/test_python_integration.py -q`: 87
  Python, Gymnasium, rendering-contract, and
  Stable-Baselines3 integration tests.
- `tests/check_abi_contract.py`: all 64 public C exports match the CFFI surface.
- `zig build -Doptimize=ReleaseFast` and `zig build metallib`.
- `zig build bench`: synthetic and engine-pipeline suites complete; their output
  is explicitly separated and makes no cross-simulator equivalence claim.
- `zig build bench-full-physics -Doptimize=ReleaseFast`: the real Ant
  `World.step` pipeline scales from 64 through 16,384 environments, with five
  sorted samples, device provenance, latency, throughput, and memory per point.
- `benchmarks/validate_physics.py`: live Zeno/MuJoCo free-body ballistic check,
  with a 0.01 m maximum-error gate.
- `benchmarks/compare_mujoco.py`: fixed-seed, five-repeat Metal versus
  sequential-CPU throughput experiment with JSON provenance and full samples;
  its ratio is explicitly not a matched-semantics speedup claim.
- `mkdocs build --strict`.
- A clean macOS 13 ARM64 wheel contains `libzeno.dylib` and reference assets;
  an isolated install can create, reset, and step native and Gym environments.

## Objective evidence matrix

| Requested outcome | Current authoritative evidence | Audit conclusion |
|-------------------|--------------------------------|------------------|
| Isaac Gym-style batched simulation on Macs | `World` dispatches one Metal command stream over environment-major SoA buffers; the full Ant pipeline is measured from 64 through 16,384 environments | Achieved for the documented Zeno physics subset |
| Apple-Silicon GPU-native stepping and resident state | Dynamics, collision, constraints, sensors, task outputs, reset, and autoreset are Metal kernels; state and scene data use GPU-addressable `MTLBuffer` storage rather than per-step host simulation | Achieved; the host still submits commands, synchronizes requested outputs, and may prepare actions/masks |
| Exploit unified memory | Writable action views and optional observation/reward/done views directly expose shared Metal storage; copied and zero-copy contracts have native and Python tests | Achieved for NumPy/CPU-visible shared memory; no zero-copy PyTorch-MPS or JAX device-tensor claim |
| Vector Gymnasium and SB3 integration | All registered IDs resolve and step; Gymnasium 0.29–1.x behavior is tested; the SB3 `VecEnv` adapter runs PPO and delegates split submission to native async stepping | Achieved for the registered Zeno task semantics |
| Aggressive batched execution | Compact active-env dispatch, immediate async submission, fused reset-before-step, GPU task evaluation, shared action staging, and zero-copy outputs have dedicated correctness tests and isolated M4 Pro benchmarks | Achieved with measured rather than universal speedup claims |
| Reproducible correctness | 250 native tests, 87 Python tests, 64-symbol ABI validation, ballistic comparison, primitive-pair responses, >512-geom broad phase, and bounded long-horizon stack/friction/joint/free-body gates | Achieved for the explicitly tested cases; not proof of general MuJoCo parity |
| Apple-GPU provenance, scaling, latency, and throughput | Benchmarks print the Metal device and unified-memory status; checked JSON records include OS/build, CPU/GPU configuration, compiler, repetitions, samples, workload, and claim boundary | Achieved on the recorded 12-CPU/16-GPU M4 Pro configuration |
| Comparison against an existing CPU path | Fixed-seed five-repeat Pendulum experiment records Zeno's batched Metal Python API and a sequential MuJoCo CPU `MjData` loop on the same machine | Achieved as an execution-throughput comparison only |
| Honest support boundaries | README, MJCF guide, architecture, and performance reference distinguish parsing from active Metal contacts, Zeno tasks from reference rewards, synthetic from full-engine measurements, and validation from equivalence | Achieved; unsupported limits remain listed below |

The requested platform outcome is therefore satisfied as a high-throughput
Apple-Silicon research simulator for its promised subset. This conclusion does
not expand the subset into universal MJCF, MuJoCo, Isaac Gym, or production
physics equivalence.

## Correctness and integration fixes in this audit

- Native ImageIO texture decoding and validated Metal upload.
- Gymnasium 0.29 through 1.x vector API compatibility and a real SB3 `VecEnv`
  adapter, exercised by a small PPO training run.
- Idempotent native-resource cleanup and context-manager support.
- MJCF `<freejoint/>` parsing, which restored motion in generated swarm models.
- Defined overflow behavior when a scene needs more constraint colors than the
  runtime representation supports.
- Per-body gravity scale, linear damping, and angular damping are now consumed
  by the integration shader instead of a hard-coded velocity multiplier.
- Contact position correction runs once per narrow-phase depth instead of once
  per solver iteration; normal, restitution, and two-axis Coulomb impulses now
  run after XPBD velocity reconstruction, and public acceleration is derived
  afterward. Bounded regressions cover a 2,000-step two-sphere stack and a
  1,000-step sliding-to-rolling friction transition.
- A 5,000-step driven Pendulum regression bounds hinge-anchor and welded-child
  drift for that model; a separate 10,000-step collision-free free-body case
  bounds linear/angular speed proxies, position error, and quaternion norm.
- Honest environment-step accounting and removal of invented benchmark targets.
- Offline `.metallib` compilation, install, and CI validation.
- Platform-correct Python wheels that build the native library for macOS 13
  ARM64 and bundle the built-in models.
- Immediate asynchronous submission, fused compact Gymnasium autoreset, and
  writable unified-memory actions, each with bounded M4 Pro measurements.
- All ten registered Gymnasium IDs resolve their bundled assets and step under
  Gymnasium 1.x. Six locomotion models use explicit Zeno-owned Metal task
  presets; unsupported objective families remain raw physics.

## Supported boundary

Zeno is a macOS/Apple-Silicon research engine, not a drop-in MuJoCo replacement.
It supports the MJCF and simulation subset listed in the public guide. Parsing a
model does not imply numerically identical solver, contact, reward, observation,
or termination semantics. The bounded ballistic comparison is evidence for that
case only.

The following are explicit research limitations, not unfinished release claims:

- contact updates use CAS float atomics, so contact-heavy runs are not promised
  to be bit-exact across executions;
- stage profiling measures CPU encoding time rather than GPU counter samples;
- Gym's `rgb_array` mode is a diagnostic body-center projection; interactive
  `human` rendering is not advertised;
- runtime Metal-source compilation remains the portable default, while
  `zig build metallib` produces an optional offline artifact;
- composite/flexcomp MJCF and advanced MuJoCo solver modes are unsupported.
- the Metal `World.step` narrow phase covers every meaningful pairing among
  sphere, capsule, box, and plane, using a single support contact for box–box;
  exact sphere-cylinder and cylinder-plane contacts cover the bundled Pusher
  interaction, but other cylinder pairs and broader CPU mesh/heightfield
  functions do not establish GPU contact support;
- scenes above 512 geoms use the spatial hash; its cell reach is shape-aware,
  and compatible finite geoms are explicitly paired with infinite planes even
  when their hash cells are distant;

Future work can broaden this boundary, but it must arrive with new acceptance
tests and must not be described as already supported.
