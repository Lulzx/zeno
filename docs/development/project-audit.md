# Project Audit

This audit reflects the repository state after the recent physics, swarm, and visualization commits.

## Current Status

- Zig CI target: 0.14.1.
- Verified locally with Zig 0.14.1: `zig build test` passes.
- Verified native ABI: `python tests/check_abi_contract.py` passes with 54 exported `zeno_*` functions.
- Verified Python integration: `51 passed, 7 skipped` with `PYTHONPATH=python` and `DYLD_LIBRARY_PATH=zig-out/lib`.
- Zig 0.16 is not currently supported. The code still uses older stdlib APIs in runtime modules, even though the build script now tolerates old and new framework-linking APIs.

## What Went Wrong

1. The documented toolchain drifted from the tested toolchain.
   README and installation docs claimed Zig 0.15+, CI used Zig 0.14.0 before this audit, and local `zig` resolved to Zig 0.16.0. This made failures look like code regressions when part of the problem was an unsupported compiler.

2. Recent swarm work mixed Zig `ArrayList` APIs across versions.
   `ReplayRecorder` and the replay serialization test used unmanaged-style calls against `std.ArrayList`, which fails on Zig 0.14. The fix is to use `std.ArrayListUnmanaged` consistently when passing allocators explicitly.

3. A newer alignment literal entered 0.14-targeted code.
   `alignedAlloc([4]f32, .@"16", len)` is accepted by newer Zig APIs but fails in Zig 0.14. The code now uses the 0.14-compatible alignment value.

4. The Python test script did not match the repository layout.
   `scripts/test_python.sh` attempted to run `python/tests/`, but the actual integration test file is `tests/test_python_integration.py`. The script also did not expose the editable local package when it was not installed.

5. Visualization scripts were tied to one checkout path.
   `viz_swarm.py` and `visualize_demo.py` inserted `/Users/lulzx/work/zeno/python` directly into `sys.path`. They now derive the project root from the script path.

6. Commits were too broad.
   The swarm commits bundled native runtime code, Python API, docs, benchmark code, scenarios, tests, replay, and visualization. This made compiler-version regressions and script-path bugs harder to isolate.

## Recent Commit Analysis

- `7833996 feat: add swarm platform with GPU spatial hash broad phase`
  Added the core swarm platform, GPU spatial hashing, Python bindings, and tests. This was a large feature commit with high integration risk because it touched GPU shaders, world stepping, C ABI, and Python wrappers together.

- `262ce4e feat: add swarm message realism, tasks, attacks, replay, and docs`
  Expanded swarm realism, tasks, replay, docs, examples, and benchmarks. This commit introduced the replay `ArrayList` API mismatch and added a Python test script with the wrong test path.

- `2566ca3 feat: add 3D swarm visualization and fix atomic read in broad_phase_detect`
  Fixed an atomic-read issue in the shader and added visualization. The visualization used an absolute local path, making the script non-portable.

- Earlier performance commits (`3ba9b86`, `fc39f18`) reduced GPU submission overhead. These are useful, but should be protected by correctness tests around contact generation, constraint ordering, and broad-phase determinism because they changed synchronization boundaries.

## Fixes Applied

- Made `build.zig` framework/libc linking compatible with both old and new Zig build APIs.
- Converted replay storage and its serialization test to `std.ArrayListUnmanaged`.
- Restored Zig 0.14-compatible aligned allocation.
- Fixed `scripts/test_python.sh` to run the actual Python integration test and set `PYTHONPATH`.
- Replaced hardcoded visualization paths with script-relative project-root detection.
- Corrected docs and CI to state/use the currently supported Zig 0.14.1 toolchain.

## Next Improvements

1. Keep `.tool-versions`, CI, and the installation docs in sync whenever the Zig compiler is upgraded.
2. Split future feature work into smaller commits: native runtime, C ABI, Python API, tests, docs, and visualization should land separately when possible.
3. Add a CI step that runs `scripts/test_python.sh` so helper scripts cannot silently rot.
4. Add a dedicated Zig 0.16 migration branch if newer compiler support matters. Treat it as a real migration because filesystem I/O, time APIs, allocators, and list initialization all changed.
5. Re-baseline performance tables after the swarm and broad-phase changes, and store the hardware/compiler metadata with the benchmark output.
6. Replace the placeholder texture loader in `src/render/material.zig` with a real image decoder or remove the claim until it is implemented.
7. Add smoke tests for visualization scripts that at least import and create their output paths without requiring long video rendering.
