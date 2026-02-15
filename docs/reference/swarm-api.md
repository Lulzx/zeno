# Swarm API Reference

## Python API

### `zeno.swarm.create_swarm_world()`

Create a world with agent bodies and a swarm instance.

```python
create_swarm_world(
    num_agents: int,
    agent_radius: float = 0.1,
    num_envs: int = 1,
    layout: str = "grid",
    spacing: float = 0.5,
    communication_range: float = 10.0,
    max_contacts_per_env: int = 256,
) -> Tuple[ZenoWorld, ZenoSwarm]
```

### `SwarmConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_agents` | `int` | `0` | Number of agents |
| `communication_range` | `float` | `10.0` | Neighbor detection range |
| `max_neighbors` | `int` | `32` | Max neighbors per agent |
| `max_message_bytes` | `int` | `48` | Max payload bytes per message |
| `max_messages_per_step` | `int` | `4` | Max messages per agent per step |
| `grid_cell_size` | `float` | `10.0` | Spatial hash cell size |
| `seed` | `int` | `42` | Random seed |
| `enable_physics` | `bool` | `True` | Enable physics integration |
| `latency_ticks` | `int` | `0` | Message delivery delay in steps |
| `drop_prob` | `float` | `0.0` | Message drop probability [0, 1] |
| `max_broadcast_recipients` | `int` | `0xFFFFFFFF` | Broadcast fan-out limit |
| `max_inbox_per_agent` | `int` | `0` | Per-agent inbox limit (0 = default) |
| `strict_determinism` | `bool` | `True` | Deterministic dropout via seeded RNG |

### `ZenoSwarm`

```python
ZenoSwarm(world: ZenoWorld, config: SwarmConfig, body_offset: int = 0)
```

#### `step(actions=None)`

Execute one swarm step: grid rebuild, graph build, message delivery, metrics.

#### `get_metrics() -> SwarmMetrics`

Return metrics from the most recent step.

#### `get_neighbor_counts() -> np.ndarray`

Return per-agent neighbor counts as a uint32 numpy array.

#### `set_body_offset(offset: int)`

Set the index of the first agent body in the world's body array.

#### `evaluate_task(task_type: str, **params) -> TaskResult`

Evaluate a cooperative task. Task types: `"formation"`, `"coverage"`, `"pursuit"`, `"tracking"`.

**Formation params:** `center_x`, `center_y`, `target_radius`, `formation_type` (0=circle, 1=line, 2=grid)

**Coverage params:** `x_min`, `y_min`, `x_max`, `y_max`, `cell_size`

**Pursuit params:** `num_pursuers`, `capture_radius`

**Tracking params:** `target_x`, `target_y`, `target_z`, `track_radius`

#### `apply_attack(attack_type, intensity=0.0, target_agents=None, seed=0)`

Apply an adversarial attack. Types: `"none"`, `"jamming"`, `"dropout"`, `"byzantine"`, `"partition"`.

#### `start_recording()`

Begin recording replay frames.

#### `stop_recording()`

Stop recording replay frames.

#### `get_replay_stats() -> ReplayStats`

Return frame count, total bytes, and recording status.

### `SwarmMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `connectivity_ratio` | `float` | Fraction of possible edges that exist |
| `fragmentation_score` | `float` | Connected components (1 = fully connected) |
| `collision_count` | `int` | Physics collisions this step |
| `message_count` | `int` | Messages delivered this step |
| `bytes_sent` | `int` | Total payload bytes sent |
| `total_edges` | `int` | Total neighbor edges |
| `avg_neighbors` | `float` | Mean neighbors per agent |
| `messages_dropped` | `int` | Messages lost to dropout/jamming |
| `convergence_time_ms` | `float` | Time to reach task objective (ms) |
| `near_miss_count` | `int` | Agents within 2x collision radius |
| `task_success` | `float` | Latest task evaluation score |

### `TaskResult`

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Normalized success [0, 1] |
| `complete` | `bool` | Whether objective is met |
| `detail` | `tuple[float, ...]` | Task-specific detail values (4 floats) |

### `ReplayStats`

| Field | Type | Description |
|-------|------|-------------|
| `frame_count` | `int` | Number of recorded frames |
| `total_bytes` | `int` | Total bytes of recorded data |
| `recording` | `bool` | Whether recording is active |

---

## C ABI

All C functions use opaque handles and return `ZenoError` (0 = success).

### Core

```c
ZenoSwarmHandle zeno_swarm_create(ZenoWorldHandle world, const ZenoSwarmConfig* config);
void            zeno_swarm_destroy(ZenoSwarmHandle handle);
ZenoError       zeno_swarm_step(ZenoSwarmHandle handle, ZenoWorldHandle world, const float* actions);
ZenoError       zeno_swarm_get_metrics(ZenoSwarmHandle handle, ZenoSwarmMetrics* out);
ZenoError       zeno_swarm_get_neighbor_counts(ZenoSwarmHandle handle, uint32_t* out, uint32_t max_agents);
void            zeno_swarm_set_body_offset(ZenoSwarmHandle handle, uint32_t offset);
```

### Tasks

```c
ZenoError zeno_swarm_evaluate_task(
    ZenoSwarmHandle handle,
    ZenoWorldHandle world,
    uint32_t task_type,          // 0=formation, 1=coverage, 2=pursuit, 3=tracking
    const float params[8],
    ZenoTaskResult* result
);
```

### Attacks

```c
ZenoError zeno_swarm_apply_attack(ZenoSwarmHandle handle, const ZenoAttackConfig* config);
```

### Zero-Copy Graph Access

```c
uint32_t* zeno_swarm_get_neighbor_index_ptr(ZenoSwarmHandle handle);  // CSR column indices
uint32_t* zeno_swarm_get_neighbor_row_ptr(ZenoSwarmHandle handle);    // CSR row offsets
```

### Replay

```c
ZenoError zeno_swarm_start_recording(ZenoSwarmHandle handle);
ZenoError zeno_swarm_stop_recording(ZenoSwarmHandle handle);
ZenoError zeno_swarm_get_replay_stats(ZenoSwarmHandle handle, ZenoReplayStats* out);
```

### C Structs

```c
typedef struct {
    uint32_t num_agents;
    float    communication_range;
    uint32_t max_neighbors;
    uint32_t max_message_bytes;
    uint32_t max_messages_per_step;
    float    grid_cell_size;
    uint64_t seed;
    bool     enable_physics;
    uint32_t latency_ticks;
    float    drop_prob;
    uint32_t max_broadcast_recipients;
    uint32_t max_inbox_per_agent;
    bool     strict_determinism;
    uint8_t  _pad[2];
} ZenoSwarmConfig;

typedef struct {
    float    connectivity_ratio;
    float    fragmentation_score;
    uint32_t collision_count;
    uint32_t message_count;
    uint32_t bytes_sent;
    uint32_t total_edges;
    float    avg_neighbors;
    uint32_t messages_dropped;
    float    convergence_time_ms;
    uint32_t near_miss_count;
    float    task_success;
    uint32_t _pad;
} ZenoSwarmMetrics;

typedef struct {
    float score;
    bool  complete;
    uint8_t _pad1[3];
    float detail[4];
} ZenoTaskResult;

typedef struct {
    uint32_t attack_type;  // 0=none, 1=jamming, 2=dropout, 3=byzantine, 4=partition
    float    intensity;
    uint32_t target_agents[16];
    uint32_t num_targets;
    uint32_t seed;
    uint32_t _pad[2];
} ZenoAttackConfig;

typedef struct {
    uint64_t frame_count;
    uint64_t total_bytes;
    bool     recording;
    uint8_t  _pad[7];
} ZenoReplayStats;
```

---

## Zig API

### `src/swarm/swarm.zig` — `Swarm`

```zig
pub fn init(allocator: Allocator, config: SwarmConfig) !Swarm
pub fn deinit(self: *Swarm) void
pub fn step(self: *Swarm, positions: [][4]f32, velocities: [][4]f32, actions: ?[*]const f32, action_dim: u32) void
pub fn setBodyOffset(self: *Swarm, offset: u32) void
pub fn getMetrics(self: *const Swarm) SwarmMetrics
pub fn getNeighborCounts(self: *const Swarm, out: []u32) void
pub fn evaluateTask(self: *const Swarm, positions: [][4]f32, velocities: [][4]f32, task_type: u32, params: [8]f32) TaskResult
pub fn setAttack(self: *Swarm, config: AttackConfig) void
pub fn clearAttack(self: *Swarm) void
pub fn startRecording(self: *Swarm) void
pub fn stopRecording(self: *Swarm) void
pub fn getReplayStats(self: *const Swarm) ReplayStats
```

### `src/swarm/metrics.zig`

```zig
pub fn computeMetrics(graph: *const AdjacencyGraph, bus: *const MessageBus, num_agents: u32) SwarmMetrics
pub fn computeFragmentation(graph: *const AdjacencyGraph, num_agents: u32, allocator: ?Allocator) f32
pub fn computeNearMisses(positions: [][4]f32, body_offset: u32, num_agents: u32, collision_radius: f32) u32
pub fn computeConvergence(history: []const SwarmMetrics, threshold: f32) f32
```

### `src/swarm/tasks.zig`

```zig
pub fn evaluateTask(task_type: u32, positions: [][4]f32, velocities: [][4]f32, graph: *const AdjacencyGraph, agent_states: []const AgentState, body_offset: u32, num_agents: u32, params: [8]f32) TaskResult
```

### `src/swarm/attacks.zig`

```zig
pub fn applyAttack(config: *const AttackConfig, bus: *MessageBus, graph: *AdjacencyGraph, num_agents: u32, step_count: u64) void
```

### `src/swarm/replay.zig`

```zig
pub const ReplayRecorder = struct {
    pub fn init(allocator: Allocator) ReplayRecorder
    pub fn deinit(self: *ReplayRecorder) void
    pub fn startRecording(self: *ReplayRecorder) void
    pub fn stopRecording(self: *ReplayRecorder) void
    pub fn recordFrame(self: *ReplayRecorder, step: u64, positions: [][4]f32, velocities: [][4]f32, bus: *const MessageBus, metrics: SwarmMetrics) !void
    pub fn getFrame(self: *const ReplayRecorder, index: usize) ?ReplayFrame
    pub fn frameCount(self: *const ReplayRecorder) usize
    pub fn verifyDeterminism(self: *const ReplayRecorder, other: *const ReplayRecorder) bool
    pub fn getStats(self: *const ReplayRecorder) ReplayStats
};
```
