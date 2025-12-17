# Zig API

For maximum performance and control, use Zeno directly from Zig.

## World Creation

### Loading from MJCF

```zig
const std = @import("std");
const zeno = @import("zeno");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Parse MJCF file
    var scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    defer scene.deinit();

    // Create simulation world
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 1024,
        .timestep = 0.002,
    });
    defer world.deinit();
}
```

### SimParams Configuration

```zig
const params = zeno.world.SimParams{
    .num_envs = 1024,
    .timestep = 0.002,
    .gravity = .{ 0, 0, -9.81 },
    .contact_iterations = 4,
    .max_contacts_per_env = 32,
};

var world = try zeno.World.init(allocator, scene, params);
```

## Simulation Loop

### Basic Stepping

```zig
// Prepare actions (num_envs * action_dim)
var actions: [1024 * 8]f32 = undefined;
for (&actions) |*a| a.* = 0.0;

// Step simulation
for (0..1000) |_| {
    try world.step(&actions, 0);
}
```

### With Substeps

For improved accuracy with stiff contacts:

```zig
const substeps = 4;
try world.step(&actions, substeps);
```

## State Access

### Observations

```zig
// Zero-copy access to observation buffer
const obs = world.getObservations();
// obs is []f32 of length num_envs * obs_dim

const obs_dim = world.scene.observation_dim;
for (0..world.params.num_envs) |env_id| {
    const env_obs = obs[env_id * obs_dim ..][0..obs_dim];
    // Process observations for this environment
}
```

### Rewards and Dones

```zig
const rewards = world.getRewards();   // []f32, length num_envs
const dones = world.getDones();       // []u8, length num_envs
```

### Body State

```zig
// Access position buffer directly
const positions = world.state.positions.getData();
// positions is []f32 with layout: [env_id * num_bodies + body_id] * 4

// Helper function
fn getBodyPosition(world: *World, env_id: usize, body_id: usize) [3]f32 {
    const idx = (env_id * world.scene.bodies.len + body_id) * 4;
    const data = world.state.positions.getData();
    return .{ data[idx], data[idx + 1], data[idx + 2] };
}
```

## Environment Reset

### Reset All

```zig
try world.reset(null);
```

### Reset Specific Environments

```zig
var mask: [1024]u8 = undefined;
for (&mask, 0..) |*m, i| {
    m.* = if (dones[i] != 0) 1 else 0;
}
try world.reset(&mask);
```

## Scene Structure

### Accessing Scene Data

```zig
const scene = world.scene;

// Bodies
for (scene.bodies) |body| {
    std.debug.print("Body: {s}, mass: {d}\n", .{ body.name, body.mass });
}

// Joints
for (scene.joints) |joint| {
    std.debug.print("Joint: {s}, type: {}\n", .{ joint.name, joint.joint_type });
}

// Geometries
for (scene.geoms) |geom| {
    std.debug.print("Geom: {s}, type: {}\n", .{ geom.name, geom.geom_type });
}
```

## Metal Buffer Access

For advanced use cases, access the underlying Metal buffers:

```zig
// Get raw MTLBuffer pointer
const position_buffer = world.state.positions.buffer;

// Synchronize if needed (usually automatic)
try world.command.synchronize();
```

## Error Handling

All Zeno functions that can fail return error unions:

```zig
const world = try zeno.World.init(allocator, scene, params);
// Possible errors:
// - error.MetalInitFailed
// - error.OutOfMemory
// - error.InvalidScene
```

## Complete Example

```zig
const std = @import("std");
const zeno = @import("zeno");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load scene
    var scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    defer scene.deinit();

    // Create world
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 1024,
        .timestep = 0.002,
    });
    defer world.deinit();

    // Initialize actions
    const action_dim = scene.actuators.len;
    var actions = try allocator.alloc(f32, 1024 * action_dim);
    defer allocator.free(actions);
    @memset(actions, 0);

    // Training loop
    var total_reward: f32 = 0;
    for (0..10000) |step| {
        // Random actions
        for (actions) |*a| {
            a.* = randomFloat() * 2 - 1;
        }

        // Step
        try world.step(actions.ptr, 0);

        // Accumulate rewards
        const rewards = world.getRewards();
        for (rewards) |r| total_reward += r;

        // Reset done environments
        const dones = world.getDones();
        var any_done = false;
        for (dones) |d| {
            if (d != 0) {
                any_done = true;
                break;
            }
        }
        if (any_done) {
            try world.reset(dones);
        }

        if (step % 1000 == 0) {
            std.debug.print("Step {}: avg reward = {d:.3}\n", .{
                step,
                total_reward / @as(f32, @floatFromInt((step + 1) * 1024)),
            });
        }
    }
}

fn randomFloat() f32 {
    var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    return rng.random().float(f32);
}
```
