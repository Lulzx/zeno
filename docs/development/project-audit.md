# Project Audit

Last verified: 2026-09-01 on Apple Silicon with Zig 0.16.0.

## Verified release surface

- `zig build test --summary all`: 225 tests across 14 native suites.
- `scripts/test_python.sh`: 68 Python, Gymnasium, rendering-contract, and
  Stable-Baselines3 integration tests.
- `tests/check_abi_contract.py`: all 55 public C exports match the CFFI surface.
- `zig build -Doptimize=ReleaseFast` and `zig build metallib`.
- `zig build bench`: synthetic and engine-pipeline suites complete; their output
  is explicitly separated and makes no cross-simulator equivalence claim.
- `benchmarks/validate_physics.py`: live Zeno/MuJoCo free-body ballistic check,
  with a 0.01 m maximum-error gate.
- `mkdocs build --strict`.
- A clean macOS 13 ARM64 wheel contains `libzeno.dylib` and reference assets;
  an isolated install can create, reset, and step native and Gym environments.

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
- Honest environment-step accounting and removal of invented benchmark targets.
- Offline `.metallib` compilation, install, and CI validation.
- Platform-correct Python wheels that build the native library for macOS 13
  ARM64 and bundle the built-in models.

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

Future work can broaden this boundary, but it must arrive with new acceptance
tests and must not be described as already supported.
