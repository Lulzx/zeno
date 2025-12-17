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
};
pub const collision = struct {
    pub const primitives = @import("collision/primitives.zig");
    pub const broad_phase = @import("collision/broad_phase.zig");
    pub const narrow_phase = @import("collision/narrow_phase.zig");
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

// Re-exports for convenience
pub const World = world.world_mod.World;
pub const WorldConfig = world.world_mod.WorldConfig;
pub const Scene = world.scene.Scene;
pub const SceneBuilder = world.scene.SceneBuilder;

// ============================================================================
// C ABI Types
// ============================================================================

/// Opaque handle to a Zeno world.
pub const ZenoWorldHandle = ?*anyopaque;

/// Configuration for world creation.
pub const ZenoConfig = extern struct {
    num_envs: u32 = 1,
    timestep: f32 = 0.002,
    contact_iterations: u32 = 4,
    max_contacts_per_env: u32 = 64,
    seed: u64 = 42,
    substeps: u32 = 1,
};

/// World information.
pub const ZenoInfo = extern struct {
    num_envs: u32,
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    obs_dim: u32,
    action_dim: u32,
    timestep: f32,
    memory_usage: u64,
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
};

// Global allocator for C API
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

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

    // Create world config
    const world_config = WorldConfig{
        .num_envs = config.num_envs,
        .timestep = config.timestep,
        .contact_iterations = config.contact_iterations,
        .max_contacts_per_env = config.max_contacts_per_env,
        .seed = config.seed,
        .substeps = config.substeps,
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

    const world_config = WorldConfig{
        .num_envs = config.num_envs,
        .timestep = config.timestep,
        .contact_iterations = config.contact_iterations,
        .max_contacts_per_env = config.max_contacts_per_env,
        .seed = config.seed,
        .substeps = config.substeps,
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
        .obs_dim = world_info.obs_dim,
        .action_dim = world_info.action_dim,
        .timestep = world_info.timestep,
        .memory_usage = @intCast(world_info.memory_usage),
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

/// Get library version.
export fn zeno_version() [*:0]const u8 {
    return "0.1.0";
}

/// Check if Metal is available.
export fn zeno_metal_available() bool {
    const device = objc.createSystemDefaultDevice();
    return device != null;
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
