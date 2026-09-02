# Zeno documentation

Zeno runs batches of rigid-body worlds with Metal on Apple Silicon. State stays
in shared memory. Zig owns the runtime; Python supplies the common RL-facing
interface.

Start with the shortest path that matches the work:

| Need | Read |
|---|---|
| Build Zeno | [Installation](getting-started/installation.md) |
| Run a model | [Quick start](getting-started/quickstart.md) |
| Decide whether Zeno fits | [Status and scope](status.md) |
| Use Python | [Python API](guide/python-api.md) |
| Use Gymnasium or SB3 | [Gymnasium](guide/gymnasium.md) |
| Use Zig | [Zig API](guide/zig-api.md) |
| Understand the pipeline | [Architecture](reference/architecture.md) |
| Interpret benchmark claims | [Performance](reference/performance.md) |
| Inspect the equations | [Physics model](reference/physics.md) |

## Operating rule

Parsing an MJCF file proves that Zeno can read it. Running it proves that the
pipeline executes. Neither proves MuJoCo-equivalent dynamics. Performance and
correctness claims in these docs name the model, hardware, workload, and test
boundary.

## Repository map

| Path | Contents |
|---|---|
| `src/world/` | world state and step orchestration |
| `src/shaders/` | Metal compute kernels |
| `src/physics/` | constraints and experimental systems |
| `src/collision/` | contact generation |
| `src/mjcf/` | MJCF parser |
| `python/zeno/` | Python, Gymnasium, and SB3 bindings |
| `benchmarks/` | benchmark programs and recorded results |
| `tests/` | Zig and Python regressions |

This page is the complete documentation index.
