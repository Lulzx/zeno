# Fix Plan (2026-07-22 review)

> **Status update (same day):** P0-A (Zig 0.16 migration), P0-B items 1-4 plus
> the sort-key/origin/cell-size/zero-dispatch/friction follow-ups, all of
> P1-A, and P1-B items 1-6 are **done** — `zig build test` (36+ suites, incl.
> a new contact-count regression test), ReleaseFast build, ABI check (both
> directions, 55 fns), and the Python suite (56 passed incl. 3 new swarm
> tests) are green on Zig 0.16.0. CI now runs Zig tests under
> `MTL_DEBUG_LAYER=1`. Replay format bumped to v2 (metrics serialized).
>
> Known residuals: (1) `solve_contacts` now uses CAS float atomics — no lost
> impulses, but float addition order is nondeterministic at ULP level;
> bit-exact contact determinism needs contact graph coloring or a per-body
> gather pass. (2) Per-stage profiling timers still measure CPU encode time,
> not GPU time. (3) The performance re-baseline and texture decoder are done;
> optional precompiled `.metallib` artifacts remain open.

Findings from a full review of the working tree, the swarm platform, the FFI boundary, and the GPU pipeline. Ordered by priority. Items marked **CONFIRMED** were verified by reading the code paths end-to-end (and, for P0-A, by direct reproduction); the rest were reported by review with quoted code and are high-confidence.

Status: the README has already been rewritten (honest scope, no cross-simulator speedup headline). Everything below is still open.

---

## P0-A — Toolchain: local builds are completely broken (CONFIRMED by repro)

Zig 0.14.1 cannot link **any** native binary on this machine (macOS 26.6 beta, Xcode SDK 26.5). Reproduced with a hello-world: every libSystem symbol is undefined even though the verbose link line is correct (`-syslibroot .../MacOSX26.5.sdk ... -lSystem`). Explicit targets (`-target aarch64-macos.26.6`) link fine — so this is 0.14's native-libc path failing to consume the macOS 26 SDK's `libSystem.tbd` stubs. `SDKROOT`, `--sysroot`, and `--libc` do not help. Because `zig build`'s build runner is always compiled for native, **`zig build` cannot work at all with 0.14.x here**. Zig 0.16.0 (`/opt/homebrew/bin/zig`) links natively without issue.

Consequence: the audit doc's "verified locally: `zig build test` passes" is stale; every `zig build` invocation in this session failed at the build-runner link step. CI (GitHub macOS runners on older SDKs) is currently the only way to run the Zig test suite.

**Action: migrate the codebase to Zig 0.16.** This is now a prerequisite for local development, not a nice-to-have. Known migration surface (from the audit + code): `std.ArrayList` → new unmanaged-by-default API, `alignedAlloc` alignment enum (`.@"16"`), fs/io API changes, `std.time`, build-system `linkFramework` moves to `root_module` (build.zig already handles both via `@hasDecl`). Update `.tool-versions`, CI, README, and installation docs in the same change. Until the migration lands, verify all Zig changes via CI, not locally.

## P0-B — GPU pipeline correctness (4 bugs, first two CONFIRMED in source)

1. **`contact_counts` is never zeroed during stepping** — only `World.reset()` zeroes it (world.zig:1313). Every `broad_phase`/`broad_phase_detect` invocation `atomic_fetch_add`s (all_shaders.metal:526, 693); no kernel and no per-step CPU path ever resets it. Counts grow monotonically; once `count >= max_contacts`, the guard rejects all *new* pairs forever and the contact slots freeze on the first `max_contacts` pairs ever seen — later-approaching bodies never collide, and stale contacts are re-solved indefinitely. Fix: a small GPU clear kernel at the top of each substep (CPU `.zero()` is insufficient — see item 2). Mind the interaction with `cache_contacts`/`prev_contact_counts` warm-start ordering.

2. **`sh_cell_counts.zero()` is a CPU memset executed at encode time** (world.zig:1052) but all substeps encode into one command buffer committed later (world.zig:715-733). At GPU execution time the buffer is zeroed exactly once; substeps 2..N run `broad_phase_count_cells` on top of stale counts → prefix-sum offsets ~2× too large → the scatter in `broad_phase_detect` writes past each env's `sorted_geoms` region (GPU memory corruption). Same fix shape as item 1: GPU-side clear each substep.

3. **`threadgroup_barrier` used as a grid-wide barrier** in `broad_phase_detect` (all_shaders.metal:639-640) between the scatter and query phases. It only syncs one threadgroup; with >1 threadgroup (always, for the >512-geom scenes this path serves), queries read partially scattered data → nondeterministic missed/bogus contacts. Fix: split scatter and detect into two dispatches with `encoder.memoryBarrier(.buffers)` between them.

4. **`solve_contacts` does non-atomic `+=` on shared body velocities/positions** (all_shaders.metal:1740-1752, 1780-1787). One thread per contact; contacts sharing a body race and lose impulses (a body resting on 4 contacts intermittently receives 1-3 of them). Fix: either `atomic_add_float` CAS (already exists for `apply_joint_forces`) or extend graph coloring to contacts.

Also in this area (lower severity): `sort_contacts` key ignores geom IDs so same-body-pair contacts stay in race order (add geom ids as tiebreaker, all_shaders.metal:724-729); spatial hash has no origin offset (negative coordinates all clamp into boundary cells → O(n²) degradation for origin-centered scenes) and `cell_size=1.0` is never tuned, so geom pairs larger than a cell can be missed entirely where the O(n²) path would find them (world.zig:553-561, shaders :655-657); `dispatch1D` with 0 sensors or 0 actuators dispatches with `threadsPerThreadgroup=0` (invalid) — guard like the other zero-able dispatches (world.zig:957, 1266); friction "accumulated impulse" stores only the last iteration's delta (`impulses.x = j_n` vs `j_n_accumulated`, all_shaders.metal:1757/1792); per-stage profiling timers measure CPU encode time, not GPU time.

## P1-A — Swarm platform (verified by review with quoted code)

1. **Use-after-free in `computeFragmentation`** for >1024 agents (metrics.zig:59-67): the heap-fallback `defer page_alloc.free(...)` statements sit inside the `else` block, so the buffers are freed before use. Move allocation/defers to function scope.
2. **Dropout attack corrupts the CSR graph** (attacks.zig:79): `row_ptr[agent+1] = row_ptr[agent]` breaks monotonicity and gives agent+1 the dropped agent's neighbors. Needs a proper row compaction (or a per-agent "disabled" mask consulted by readers).
3. **The message pipeline can never deliver** : `stepOnce` order is deliver → policy dispatch (queues outbox) → `clearStep()` which wipes outbox counts (dispatcher.zig:94, message_bus.zig:323-330). Policy-sent messages are erased before any deliver sees them, and `computeMetrics`/`recordFrame` run after the clear so `message_count`/`bytes_sent`/replay message stats are always 0. Reorder: clear at the *start* of a step (or keep per-step metrics separate from queue clearing).
4. **External actions are silently discarded** (dispatcher.zig:62-64: `if (external_actions) |actions| { _ = actions; }`), even though `zeno_swarm_step` and the Python `ZenoSwarm.step(actions)` docstring promise they're applied — and passing actions also disables the built-in policy path. When wiring them through, fix the buffer-length math in main.zig:796-801 (`num_agents * num_actuators` over-counts; `num_actuators` is per-env) and the length-1 `dummy_actions` slice passed to Zig policies (dispatcher.zig:76-77).
5. **Jamming forges phantom inbox messages instead of blocking delivery** (attacks.zig:50 sets `inbox_counts[agent] = max_messages_per_step` *after* deliver): jammed agents read stale/zeroed slots as real messages. The test at test_swarm.zig:698 asserts the broken mechanism — fix both.
6. **`max_inbox_per_agent` can never limit below `max_messages_per_step`** (message_bus.zig:62-65 uses `@max`) — the only regime where a limit matters.
7. Replay: missing `errdefer` on `pos_copy`/`vel_copy` in `recordFrame` (leaks silently every step under pressure, swallowed by `catch {}` at swarm.zig:141); `readFrom` trusts `num_agents` from the stream (~64 GB alloc attempt on corrupt files); `writeTo` never serializes `frame.metrics` so round-trips zero it.
8. Numeric safety: `tasks.zig:134-135` `@intFromFloat` on caller-supplied inverted bounds is UB in ReleaseFast; coverage silently caps at 4096 cells while dividing by 4096 (biased score). `grid.zig:55-57,117-119` `@intFromFloat` on huge/NaN positions is UB — clamp on the float side first.
9. Message-bus pending-ring overflow drops messages without incrementing `total_messages_dropped` (message_bus.zig:247-250).

## P1-B — FFI boundary

1. **No validation of `num_agents` vs world bodies** at `zeno_swarm_create`/Python `ZenoSwarm.__init__` → OOB read of GPU-shared memory in ReleaseFast (grid.zig:77-79 indexes `positions[body_offset + i]`). Validate `body_offset + num_agents <= num_bodies * num_envs` on both sides.
2. **`max_bodies_per_env`/`max_joints_per_env`/`max_geoms_per_env` are accepted in `ZenoConfig` and silently dropped** — `zeno_world_create` never forwards them and `WorldConfig` has no such fields (main.zig:156-164). Either wire them through or remove them from the ABI/Python (note `create_swarm_world` relies on them today, illusorily).
3. **Zero-copy lifetime guard is dead code** (_ffi.py:600-620): the `ZeroCopyArray` weakref wrapper is cached but the *bare* `np.frombuffer` view is returned, so a GC'd world leaves dangling views. Return the guarded object (or attach the world ref to the array via `.base`).
4. Error codes ignored: `zeno_swarm_get_metrics`/`get_neighbor_counts` (swarm.py:158, 177), `zeno_world_get_info` (_ffi.py:531, 1147) — invalid handles silently return zeroed data. Raise on nonzero return.
5. `get_info` reports `gpu_memory_usage = memory_usage` (main.zig:476-477) — duplicate, not GPU memory.
6. Hardening: `check_abi_contract.py` compares names only, one direction — extend to struct sizes/field offsets (e.g. autogenerate a layout dump from Zig) and check cdef↔export in both directions. `tests/test_python_integration.py` has zero swarm coverage — add at least create/step/metrics/destroy round-trips.

## P2 — Presentation & process

- ~~Re-baseline the performance tables after the P0-B fixes land~~ **DONE (2026-07-22).** README + `docs/index.md` tables re-measured on M4 Pro post-fix (medians of 5 runs): Pendulum 2.97M, Cartpole 2.73M, Ant 1.47M, Humanoid 0.75M env-steps/sec — ~40–75% below the pre-fix figures, which had been inflated by the broad-phase bugs skipping collision work. Timers still measure CPU encode time (see below). Also fixed a bench-scene bug where capsule-capsule was placed 0.212 apart with combined radius 0.2, reporting a misleading 0 collisions.
- Benchmarks: label synthetic vs engine-pipeline numbers in the bench output itself, and store hardware/compiler metadata with results.
- Add CI steps for `scripts/test_python.sh` and a Metal API-validation (debug) test run, which would have caught the zero-thread dispatch and barrier issues.
- ~~`src/render/material.zig` texture loading is stubbed.~~ **DONE (2026-09-01).** Native ImageIO decoding produces straight-alpha RGBA8, validates upload sizes, and has decode plus Metal-upload coverage.
- Precompiled `.metallib` artifact support alongside runtime `newLibraryWithSource`.

## Suggested execution order for Opus

1. Zig 0.16 migration (P0-A) — unblocks local `zig build test` for everything else.
2. GPU clear-kernel fix for `contact_counts` + `cell_counts` (P0-B 1-2, one change), then the barrier split (3), then contact-solve atomics/coloring (4). Add a determinism regression test (same seed, two runs, byte-identical contacts) after 3.
3. Swarm P1-A items 1-4 (crash/corruption/dead-feature class), then 5-9.
4. FFI P1-B items 1-4, then test hardening.
5. P2 as follow-up.
