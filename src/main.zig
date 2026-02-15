//! Zeno Physics Engine - C ABI Exports
//! Main entry point for the shared library.

const std = @import("std");

// Module imports
pub const objc = @import("objc.zig");
pub const metal = struct {
    pub const device = @import("metal/device.zig");
    pub const buffer = @import("metal/buffer.zig");
    pub const pipeline = @import("metal/pipeline.zig");
    pub const command = @import("metal/command.zig");
};
pub const physics = struct {
    pub const constants = @import("physics/constants.zig");
    pub const body = @import("physics/body.zig");
    pub const joint = @import("physics/joint.zig");
    pub const contact = @import("physics/contact.zig");
    pub const state = @import("physics/state.zig");
    pub const soft_body = @import("physics/soft_body.zig");
    pub const fluid = @import("physics/fluid.zig");
    pub const xpbd = @import("physics/xpbd.zig");
    pub const tendon = @import("physics/tendon.zig");
};
pub const collision = struct {
    pub const primitives = @import("collision/primitives.zig");
    pub const broad_phase = @import("collision/broad_phase.zig");
    pub const narrow_phase = @import("collision/narrow_phase.zig");
    pub const gjk = @import("collision/gjk.zig");
};
pub const world = struct {
    pub const scene = @import("world/scene.zig");
    pub const sensors = @import("world/sensors.zig");
    pub const world_mod = @import("world/world.zig");
};
pub const mjcf = struct {
    pub const parser = @import("mjcf/parser.zig");
    pub const schema = @import("mjcf/schema.zig");
};
pub const swarm = struct {
    pub const types = @import("swarm/types.zig");
    pub const grid = @import("swarm/grid.zig");
    pub const graph = @import("swarm/graph.zig");
    pub const message_bus = @import("swarm/message_bus.zig");
    pub const policy_mod = @import("swarm/policy.zig");
    pub const dispatcher = @import("swarm/dispatcher.zig");
    pub const swarm_mod = @import("swarm/swarm.zig");
    pub const metrics = @import("swarm/metrics.zig");
    pub const tasks = @import("swarm/tasks.zig");
    pub const attacks = @import("swarm/attacks.zig");
    pub const replay = @import("swarm/replay.zig");
};

// Re-exports for convenience
pub const World = world.world_mod.World;
pub const WorldConfig = world.world_mod.WorldConfig;
pub const Scene = world.scene.Scene;
pub const SceneBuilder = world.scene.SceneBuilder;
pub const Swarm = swarm.swarm_mod.Swarm;
pub const SwarmConfig = swarm.types.SwarmConfig;
pub const SwarmMetrics = swarm.types.SwarmMetrics;
pub const AgentState = swarm.types.AgentState;
pub const TaskResult = swarm.types.TaskResult;
pub const AttackConfig = swarm.types.AttackConfig;
pub const ReplayStats = swarm.types.ReplayStats;

// ============================================================================
// C ABI Types
// ============================================================================

/// Opaque handle to a Zeno world.
pub const ZenoWorldHandle = ?*anyopaque;

/// Configuration for world creation.
pub const ZenoConfig = extern struct {
    num_envs: u32 = 1,
    timestep: f32 = 0, // 0 = use MJCF timestep
    contact_iterations: u32 = 4,
    max_contacts_per_env: u32 = 64,
    seed: u64 = 42,
    substeps: u32 = 1,
    enable_profiling: bool = false,
    max_bodies_per_env: u32 = 0,
    max_joints_per_env: u32 = 0,
    max_geoms_per_env: u32 = 0,
};

/// World information.
pub const ZenoInfo = extern struct {
    num_envs: u32,
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    num_sensors: u32,
    num_geoms: u32,
    obs_dim: u32,
    action_dim: u32,
    timestep: f32,
    memory_usage: u64,
    gpu_memory_usage: u64,
    metal_available: bool,
};

/// Per-step profiling data.
pub const ZenoProfilingData = extern struct {
    integrate_ns: f32 = 0,
    collision_broad_ns: f32 = 0,
    collision_narrow_ns: f32 = 0,
    constraint_solve_ns: f32 = 0,
    total_step_ns: f32 = 0,
    num_contacts: u32 = 0,
    num_active_constraints: u32 = 0,
};

/// Error codes.
pub const ZenoError = enum(i32) {
    success = 0,
    invalid_handle = -1,
    file_not_found = -2,
    parse_error = -3,
    metal_error = -4,
    out_of_memory = -5,
    invalid_argument = -6,
    not_implemented = -7,
};

// Global allocator for C API
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

fn uploadParams(world_ptr: *World) void {
    world_ptr.params_buffer.getSlice(world.world_mod.SimParams)[0] = world_ptr.params;
}

// ============================================================================
// C ABI Functions
// ============================================================================

/// Create a new world from an MJCF file.
export fn zeno_world_create(
    mjcf_path: [*:0]const u8,
    config: *const ZenoConfig,
) ZenoWorldHandle {
    const allocator = gpa.allocator();

    // Parse MJCF file
    const path = std.mem.span(mjcf_path);
    const scene = mjcf.parser.parseFile(allocator, path) catch |err| {
        std.log.err("Failed to parse MJCF: {}", .{err});
        return null;
    };

    // Resolve timestep: use MJCF value if config doesn't specify one
    const timestep = if (config.timestep > 0) config.timestep else scene.physics_config.timestep;

    // Create world config
    const world_config = WorldConfig{
        .num_envs = config.num_envs,
        .timestep = timestep,
        .contact_iterations = config.contact_iterations,
        .max_contacts_per_env = config.max_contacts_per_env,
        .seed = config.seed,
        .substeps = config.substeps,
        .enable_profiling = config.enable_profiling,
    };

    // Create world
    const world_ptr = allocator.create(World) catch {
        return null;
    };

    world_ptr.* = World.init(allocator, scene, world_config) catch |err| {
        std.log.err("Failed to create world: {}", .{err});
        allocator.destroy(world_ptr);
        return null;
    };

    return @ptrCast(world_ptr);
}

/// Create a world from MJCF string.
export fn zeno_world_create_from_string(
    mjcf_string: [*:0]const u8,
    config: *const ZenoConfig,
) ZenoWorldHandle {
    const allocator = gpa.allocator();

    const xml = std.mem.span(mjcf_string);
    const scene = mjcf.parser.parseString(allocator, xml) catch {
        return null;
    };

    // Resolve timestep: use MJCF value if config doesn't specify one
    const timestep = if (config.timestep > 0) config.timestep else scene.physics_config.timestep;

    const world_config = WorldConfig{
        .num_envs = config.num_envs,
        .timestep = timestep,
        .contact_iterations = config.contact_iterations,
        .max_contacts_per_env = config.max_contacts_per_env,
        .seed = config.seed,
        .substeps = config.substeps,
        .enable_profiling = config.enable_profiling,
    };

    const world_ptr = allocator.create(World) catch {
        return null;
    };

    world_ptr.* = World.init(allocator, scene, world_config) catch {
        allocator.destroy(world_ptr);
        return null;
    };

    return @ptrCast(world_ptr);
}

/// Destroy a world and free resources.
export fn zeno_world_destroy(handle: ZenoWorldHandle) void {
    if (handle == null) return;

    const allocator = gpa.allocator();
    const world_ptr: *World = @ptrCast(@alignCast(handle));
    world_ptr.deinit();
    allocator.destroy(world_ptr);
}

/// Step the simulation.
export fn zeno_world_step(
    handle: ZenoWorldHandle,
    actions: [*]const f32,
    substeps: u32,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const action_count = world_ptr.config.num_envs * world_ptr.params.num_actuators;
    const actions_slice = actions[0..action_count];

    world_ptr.step(actions_slice, substeps) catch {
        return .metal_error;
    };

    return .success;
}

/// Step only a subset of environments.
export fn zeno_world_step_subset(
    handle: ZenoWorldHandle,
    actions: [*]const f32,
    env_mask: [*]const u8,
    substeps: u32,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const action_count = world_ptr.config.num_envs * world_ptr.params.num_actuators;
    const actions_slice = actions[0..action_count];
    const mask_slice = env_mask[0..world_ptr.config.num_envs];

    world_ptr.stepSubset(actions_slice, mask_slice, substeps) catch |err| {
        return switch (err) {
            error.InvalidSize => .invalid_argument,
            else => .metal_error,
        };
    };

    return .success;
}

/// Reset environments.
export fn zeno_world_reset(
    handle: ZenoWorldHandle,
    env_mask: ?[*]const u8,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));

    if (env_mask) |mask| {
        const mask_slice = mask[0..world_ptr.config.num_envs];
        world_ptr.reset(mask_slice);
    } else {
        world_ptr.reset(null);
    }

    return .success;
}

/// Reset environments to externally provided state.
export fn zeno_world_reset_to_state(
    handle: ZenoWorldHandle,
    positions: [*]const f32,
    quaternions: [*]const f32,
    velocities: ?[*]const f32,
    angular_velocities: ?[*]const f32,
    env_mask: ?[*]const u8,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const num_envs: usize = @intCast(world_ptr.config.num_envs);
    const num_bodies: usize = @intCast(world_ptr.params.num_bodies);
    const total_floats = num_envs * num_bodies * 4;

    const pos_slice = positions[0..total_floats];
    var mask_slice: ?[]const u8 = null;
    if (env_mask) |m| {
        mask_slice = m[0..num_envs];
    }

    world_ptr.setBodyPositions(pos_slice, mask_slice) catch {
        return .invalid_argument;
    };

    const quat_slice = quaternions[0..total_floats];
    const quat_dest = world_ptr.state.getQuaternions();
    if (mask_slice) |mask| {
        for (mask, 0..) |should_set, env_id| {
            if (should_set != 0) {
                for (0..num_bodies) |b| {
                    const idx = env_id * num_bodies + b;
                    const base = idx * 4;
                    quat_dest[idx] = .{
                        quat_slice[base + 0],
                        quat_slice[base + 1],
                        quat_slice[base + 2],
                        quat_slice[base + 3],
                    };
                }
            }
        }
    } else {
        for (0..quat_dest.len) |idx| {
            const base = idx * 4;
            quat_dest[idx] = .{
                quat_slice[base + 0],
                quat_slice[base + 1],
                quat_slice[base + 2],
                quat_slice[base + 3],
            };
        }
    }

    if (velocities) |vel_ptr| {
        const vel_slice = vel_ptr[0..total_floats];
        world_ptr.setBodyVelocities(vel_slice, mask_slice) catch {
            return .invalid_argument;
        };
    }

    if (angular_velocities) |ang_ptr| {
        const ang_slice = ang_ptr[0..total_floats];
        const ang_dest = world_ptr.state.getAngularVelocities();

        if (mask_slice) |mask| {
            for (mask, 0..) |should_set, env_id| {
                if (should_set != 0) {
                    for (0..num_bodies) |b| {
                        const idx = env_id * num_bodies + b;
                        const base = idx * 4;
                        ang_dest[idx] = .{
                            ang_slice[base + 0],
                            ang_slice[base + 1],
                            ang_slice[base + 2],
                            ang_slice[base + 3],
                        };
                    }
                }
            }
        } else {
            for (0..ang_dest.len) |idx| {
                const base = idx * 4;
                ang_dest[idx] = .{
                    ang_slice[base + 0],
                    ang_slice[base + 1],
                    ang_slice[base + 2],
                    ang_slice[base + 3],
                };
            }
        }
    }

    world_ptr.state.contact_counts_buffer.zero() catch {};
    return .success;
}

/// Get pointer to observations buffer (zero-copy).
export fn zeno_world_get_observations(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getObservationsPtr();
}

/// Get pointer to rewards buffer.
export fn zeno_world_get_rewards(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getRewardsPtr();
}

/// Get pointer to dones buffer.
export fn zeno_world_get_dones(handle: ZenoWorldHandle) ?[*]u8 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getDonesPtr();
}

/// Get number of environments.
export fn zeno_world_num_envs(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.config.num_envs;
}

/// Get number of bodies per environment.
export fn zeno_world_num_bodies(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.params.num_bodies;
}

/// Get number of joints per environment.
export fn zeno_world_num_joints(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.params.num_joints;
}

/// Get number of sensors per environment.
export fn zeno_world_num_sensors(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.params.num_sensors;
}

/// Get observation dimension.
export fn zeno_world_obs_dim(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.state.obs_dim;
}

/// Get action dimension.
export fn zeno_world_action_dim(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.params.num_actuators;
}

/// Get world information.
export fn zeno_world_get_info(handle: ZenoWorldHandle, info: *ZenoInfo) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const world_info = world_ptr.getInfo();

    info.* = .{
        .num_envs = world_info.num_envs,
        .num_bodies = world_info.num_bodies,
        .num_joints = world_info.num_joints,
        .num_actuators = world_info.num_actuators,
        .num_sensors = world_info.num_sensors,
        .num_geoms = world_info.num_geoms,
        .obs_dim = world_info.obs_dim,
        .action_dim = world_info.action_dim,
        .timestep = world_info.timestep,
        .memory_usage = @intCast(world_info.memory_usage),
        .gpu_memory_usage = @intCast(world_info.memory_usage),
        .metal_available = true,
    };

    return .success;
}

/// Get body positions (for visualization).
export fn zeno_world_get_body_positions(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const positions = world_ptr.getBodyPositions();
    if (positions.len == 0) return null;

    return @ptrCast(&positions[0]);
}

/// Get body quaternions (for visualization).
export fn zeno_world_get_body_quaternions(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const quats = world_ptr.getBodyQuaternions();
    if (quats.len == 0) return null;

    return @ptrCast(&quats[0]);
}

/// Get body velocities.
export fn zeno_world_get_body_velocities(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getBodyVelocitiesPtr();
}

/// Get body angular velocities.
export fn zeno_world_get_body_angular_velocities(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getBodyAngularVelocitiesPtr();
}

/// Get body accelerations.
export fn zeno_world_get_body_accelerations(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getBodyAccelerationsPtr();
}

/// Get joint positions.
export fn zeno_world_get_joint_positions(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getJointPositionsPtr();
}

/// Get joint velocities.
export fn zeno_world_get_joint_velocities(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getJointVelocitiesPtr();
}

/// Get joint forces/torques.
export fn zeno_world_get_joint_forces(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const slice = world_ptr.state.joint_torques_buffer.getSlice(f32);
    if (slice.len == 0) return null;
    return slice.ptr;
}

/// Get contact counts.
export fn zeno_world_get_contact_counts(handle: ZenoWorldHandle) ?[*]u32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getContactCountsPtr();
}

/// Get sensor data.
export fn zeno_world_get_sensor_data(handle: ZenoWorldHandle) ?[*]f32 {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getSensorDataPtr();
}

/// Get contacts buffer.
export fn zeno_world_get_contacts(handle: ZenoWorldHandle) ?*anyopaque {
    if (handle == null) return null;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.getContactsPtr();
}

/// Set body positions.
export fn zeno_world_set_body_positions(
    handle: ZenoWorldHandle,
    positions: [*]const f32,
    env_mask: ?[*]const u8,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const total_floats = world_ptr.config.num_envs * world_ptr.params.num_bodies * 4;

    // Create slice from pointer
    const pos_slice = positions[0..total_floats];

    // Create mask slice if present
    var mask_slice: ?[]const u8 = null;
    if (env_mask) |m| {
        mask_slice = m[0..world_ptr.config.num_envs];
    }

    world_ptr.setBodyPositions(pos_slice, mask_slice) catch {
        return .metal_error;
    };

    return .success;
}

/// Set body velocities.
export fn zeno_world_set_body_velocities(
    handle: ZenoWorldHandle,
    velocities: [*]const f32,
    env_mask: ?[*]const u8,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const total_floats = world_ptr.config.num_envs * world_ptr.params.num_bodies * 4;

    const vel_slice = velocities[0..total_floats];

    var mask_slice: ?[]const u8 = null;
    if (env_mask) |m| {
        mask_slice = m[0..world_ptr.config.num_envs];
    }

    world_ptr.setBodyVelocities(vel_slice, mask_slice) catch {
        return .metal_error;
    };

    return .success;
}

/// Set gravity vector at runtime.
export fn zeno_world_set_gravity(
    handle: ZenoWorldHandle,
    gx: f32,
    gy: f32,
    gz: f32,
) ZenoError {
    if (handle == null) return .invalid_handle;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    world_ptr.params.gravity_x = gx;
    world_ptr.params.gravity_y = gy;
    world_ptr.params.gravity_z = gz;
    uploadParams(world_ptr);
    return .success;
}

/// Set world timestep at runtime.
export fn zeno_world_set_timestep(
    handle: ZenoWorldHandle,
    timestep: f32,
) ZenoError {
    if (handle == null) return .invalid_handle;
    if (timestep <= 0) return .invalid_argument;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    world_ptr.config.timestep = timestep;
    const substeps = @max(world_ptr.config.substeps, 1);
    world_ptr.params.dt = timestep / @as(f32, @floatFromInt(substeps));
    uploadParams(world_ptr);
    return .success;
}

/// Get profiling data for the most recent step.
export fn zeno_world_get_profiling(
    handle: ZenoWorldHandle,
    data: *ZenoProfilingData,
) ZenoError {
    if (handle == null) return .invalid_handle;
    const world_ptr: *World = @ptrCast(@alignCast(handle));
    const profiling = world_ptr.getProfilingData();
    data.* = .{
        .integrate_ns = profiling.integrate_ns,
        .collision_broad_ns = profiling.collision_broad_ns,
        .collision_narrow_ns = profiling.collision_narrow_ns,
        .constraint_solve_ns = profiling.constraint_solve_ns,
        .total_step_ns = profiling.total_step_ns,
        .num_contacts = profiling.num_contacts,
        .num_active_constraints = profiling.num_active_constraints,
    };
    return .success;
}

/// Reset profiling counters.
export fn zeno_world_reset_profiling(handle: ZenoWorldHandle) void {
    if (handle == null) return;
    const world_ptr: *World = @ptrCast(@alignCast(handle));
    world_ptr.resetProfiling();
}

/// Get approximate memory usage in bytes.
export fn zeno_world_memory_usage(handle: ZenoWorldHandle) u64 {
    if (handle == null) return 0;

    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return @intCast(world_ptr.state.memoryUsage());
}

/// Hint memory compaction (currently no-op).
export fn zeno_world_compact_memory(handle: ZenoWorldHandle) void {
    _ = handle;
}

/// Convert error code to human-readable string.
export fn zeno_error_string(error_code: i32) [*:0]const u8 {
    return switch (error_code) {
        0 => "success",
        -1 => "invalid handle",
        -2 => "file not found",
        -3 => "parse error",
        -4 => "metal error",
        -5 => "out of memory",
        -6 => "invalid argument",
        -7 => "not implemented",
        else => "unknown error",
    };
}

/// Get library version.
export fn zeno_version() [*:0]const u8 {
    return "0.1.0";
}

/// Get the max contacts per environment for the world.
export fn zeno_world_max_contacts_per_env(handle: ZenoWorldHandle) u32 {
    if (handle == null) return 0;
    const world_ptr: *World = @ptrCast(@alignCast(handle));
    return world_ptr.config.max_contacts_per_env;
}

/// Check if Metal is available.
export fn zeno_metal_available() bool {
    const device = objc.createSystemDefaultDevice();
    return device != null;
}

// ============================================================================
// Swarm C ABI Types
// ============================================================================

pub const ZenoSwarmHandle = ?*anyopaque;

pub const ZenoSwarmConfig = swarm.types.SwarmConfig;
pub const ZenoSwarmMetrics = swarm.types.SwarmMetrics;
pub const ZenoAgentState = swarm.types.AgentState;
pub const ZenoTaskResult = swarm.types.TaskResult;
pub const ZenoAttackConfig = swarm.types.AttackConfig;
pub const ZenoReplayStats = swarm.types.ReplayStats;

// ============================================================================
// Swarm C ABI Functions
// ============================================================================

/// Create a new swarm instance attached to a world.
export fn zeno_swarm_create(
    world_handle: ZenoWorldHandle,
    config: *const ZenoSwarmConfig,
) ZenoSwarmHandle {
    if (world_handle == null) return null;
    _ = @as(*World, @ptrCast(@alignCast(world_handle)));

    const allocator = gpa.allocator();
    const swarm_ptr = allocator.create(Swarm) catch return null;
    swarm_ptr.* = Swarm.init(allocator, config.*) catch {
        allocator.destroy(swarm_ptr);
        return null;
    };
    return @ptrCast(swarm_ptr);
}

/// Destroy a swarm and free resources.
export fn zeno_swarm_destroy(handle: ZenoSwarmHandle) void {
    if (handle == null) return;
    const allocator = gpa.allocator();
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.deinit();
    allocator.destroy(swarm_ptr);
}

/// Step the swarm (grid rebuild, graph build, message delivery).
/// Physics stepping should be done separately via zeno_world_step.
export fn zeno_swarm_step(
    handle: ZenoSwarmHandle,
    world_handle: ZenoWorldHandle,
    actions_ptr: ?[*]f32,
) ZenoError {
    if (handle == null or world_handle == null) return .invalid_handle;

    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    const world_ptr: *World = @ptrCast(@alignCast(world_handle));

    const positions = world_ptr.state.getPositions();
    const velocities = world_ptr.state.getVelocities();

    const action_dim: u32 = world_ptr.params.num_actuators;
    const num_agents = swarm_ptr.config.num_agents;

    var actions: ?[]f32 = null;
    if (actions_ptr) |ap| {
        actions = ap[0 .. num_agents * action_dim];
    }

    swarm_ptr.step(positions, velocities, actions, action_dim);

    return .success;
}

/// Get swarm metrics.
export fn zeno_swarm_get_metrics(handle: ZenoSwarmHandle, out: *ZenoSwarmMetrics) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    out.* = swarm_ptr.metrics;
    return .success;
}

/// Get pointer to agent states array.
export fn zeno_swarm_get_agent_states(handle: ZenoSwarmHandle) ?[*]ZenoAgentState {
    if (handle == null) return null;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    if (swarm_ptr.agent_states.len == 0) return null;
    return swarm_ptr.agent_states.ptr;
}

/// Get neighbor counts for all agents (caller provides buffer of size num_agents).
export fn zeno_swarm_get_neighbor_counts(handle: ZenoSwarmHandle, out: [*]u32, count: u32) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.getNeighborCounts(out[0..count]);
    return .success;
}

/// Set the body offset (index of first agent body in world body array).
export fn zeno_swarm_set_body_offset(handle: ZenoSwarmHandle, offset: u32) void {
    if (handle == null) return;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.setBodyOffset(offset);
}

/// Evaluate a task objective.
export fn zeno_swarm_evaluate_task(
    handle: ZenoSwarmHandle,
    world_handle: ZenoWorldHandle,
    task_type: u32,
    params: *const [8]f32,
    result: *ZenoTaskResult,
) ZenoError {
    if (handle == null or world_handle == null) return .invalid_handle;

    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    const world_ptr: *World = @ptrCast(@alignCast(world_handle));

    const positions = world_ptr.state.getPositions();
    const velocities = world_ptr.state.getVelocities();

    const tt = std.meta.intToEnum(swarm.tasks.TaskType, task_type) catch return .invalid_argument;
    result.* = swarm_ptr.evaluateTask(tt, positions, velocities, params.*);
    return .success;
}

/// Apply an adversarial attack to the swarm.
export fn zeno_swarm_apply_attack(
    handle: ZenoSwarmHandle,
    config: *const ZenoAttackConfig,
) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.setAttack(config.*);
    return .success;
}

/// Get raw pointer to CSR neighbor_ids for zero-copy Python access.
export fn zeno_swarm_get_neighbor_index_ptr(handle: ZenoSwarmHandle) ?[*]u32 {
    if (handle == null) return null;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    if (swarm_ptr.graph.neighbor_ids.len == 0) return null;
    return swarm_ptr.graph.neighbor_ids.ptr;
}

/// Get raw pointer to CSR row_ptr for zero-copy Python access.
export fn zeno_swarm_get_neighbor_row_ptr(handle: ZenoSwarmHandle) ?[*]u32 {
    if (handle == null) return null;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    if (swarm_ptr.graph.row_ptr.len == 0) return null;
    return swarm_ptr.graph.row_ptr.ptr;
}

/// Start recording replay frames.
export fn zeno_swarm_start_recording(handle: ZenoSwarmHandle) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.startRecording();
    return .success;
}

/// Stop recording replay frames.
export fn zeno_swarm_stop_recording(handle: ZenoSwarmHandle) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    swarm_ptr.stopRecording();
    return .success;
}

/// Get replay statistics.
export fn zeno_swarm_get_replay_stats(handle: ZenoSwarmHandle, out: *ZenoReplayStats) ZenoError {
    if (handle == null) return .invalid_handle;
    const swarm_ptr: *Swarm = @ptrCast(@alignCast(handle));
    out.* = swarm_ptr.getReplayStats();
    return .success;
}

// ============================================================================
// Test entry point
// ============================================================================

test "basic world creation" {
    const allocator = std.testing.allocator;

    // Create a simple scene programmatically
    var builder = SceneBuilder.init(allocator);
    _ = builder.ground();
    _ = builder.body("ball", .{ 0, 0, 1 }).mass(1.0).sphere(0.1).freeJoint();
    var scene = builder.build();
    defer scene.deinit();

    const config = WorldConfig{
        .num_envs = 1,
        .timestep = 0.002,
    };

    var w = try World.init(allocator, scene, config);
    defer w.deinit();

    try std.testing.expectEqual(@as(u32, 1), w.config.num_envs);
}
