//! Main simulation world - orchestrates physics stepping and GPU execution.

const std = @import("std");
const objc = @import("../objc.zig");
const Device = @import("../metal/device.zig").Device;
const Buffer = @import("../metal/buffer.zig").Buffer;
const BufferOptions = @import("../metal/buffer.zig").BufferOptions;
const ComputePipeline = @import("../metal/pipeline.zig").ComputePipeline;
const PipelineManager = @import("../metal/pipeline.zig").PipelineManager;
const CommandBuffer = @import("../metal/command.zig").CommandBuffer;
const ComputeEncoder = @import("../metal/command.zig").ComputeEncoder;
const State = @import("../physics/state.zig").State;
const Scene = @import("scene.zig").Scene;
const constants = @import("../physics/constants.zig");
const body_mod = @import("../physics/body.zig");
const joint_mod = @import("../physics/joint.zig");
const primitives = @import("../collision/primitives.zig");
const contact = @import("../physics/contact.zig");
const xpbd_mod = @import("../physics/xpbd.zig");
const XPBDConstraint = xpbd_mod.XPBDConstraint;

/// Simulation configuration.
pub const WorldConfig = struct {
    num_envs: u32 = 1,
    timestep: f32 = constants.DEFAULT_TIMESTEP,
    contact_iterations: u32 = constants.DEFAULT_CONTACT_ITERATIONS,
    max_contacts_per_env: u32 = constants.DEFAULT_MAX_CONTACTS_PER_ENV,
    seed: u64 = 42,
    substeps: u32 = 1,
    enable_profiling: bool = false,
};

/// GPU parameters passed to shaders.
pub const SimParams = extern struct {
    num_envs: u32 align(16),
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    num_geoms: u32,
    num_sensors: u32,
    max_contacts: u32,
    contact_iterations: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    friction: f32,
    restitution: f32,
    baumgarte: f32,
    slop: f32,
    target_color: u32, // For constraint graph coloring (solve_joints)
    num_constraints: u32, // Total constraints to process for current color
    constraint_offset: u32, // Offset into constraint buffer for current color
    obs_dim: u32, // Observation dimension per environment
};

/// GPU spatial hash parameters (matches Metal SpatialHashParams struct).
pub const SpatialHashParamsGPU = extern struct {
    grid_dim_x: u32 align(16),
    grid_dim_y: u32,
    grid_dim_z: u32,
    total_cells: u32,
    cell_size: f32,
    inv_cell_size: f32,
    num_geoms: u32,
    num_envs: u32,
    max_contacts: u32,
    /// World-space position of grid cell (0,0,0). Without this offset every
    /// geom at negative coordinates clamps into the boundary cells.
    origin_x: f32 = 0,
    origin_y: f32 = 0,
    origin_z: f32 = 0,
};

/// Maps compact dispatch slots to stable physical environment indices.
pub const EnvDispatchParams = extern struct {
    num_envs: u32 align(16),
    dispatch_envs: u32,
    use_active_env_ids: u32,
    _pad: u32 = 0,
};

/// Optional, bounded locomotion task evaluated on the GPU after each world
/// step. This is a reusable primitive, not a claim of MuJoCo task equivalence.
pub const TaskParams = extern struct {
    enabled: u32 align(16) = 0,
    root_body: u32 = 0,
    forward_axis: u32 = 0,
    max_episode_steps: u32 = 0,
    forward_reward_weight: f32 = 0,
    control_cost_weight: f32 = 0,
    healthy_bonus: f32 = 0,
    healthy_z_min: f32 = -std.math.inf(f32),
    healthy_z_max: f32 = std.math.inf(f32),
    terminate_when_unhealthy: u32 = 0,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

/// Profiling data for the most recent step call.
pub const ProfilingData = struct {
    integrate_ns: f32 = 0,
    collision_broad_ns: f32 = 0,
    collision_narrow_ns: f32 = 0,
    constraint_solve_ns: f32 = 0,
    total_step_ns: f32 = 0,
    num_contacts: u32 = 0,
    num_active_constraints: u32 = 0,
};

const ProfilingDataRaw = struct {
    integrate_ns: u64 = 0,
    collision_broad_ns: u64 = 0,
    collision_narrow_ns: u64 = 0,
    constraint_solve_ns: u64 = 0,
    total_step_ns: u64 = 0,
    num_contacts: u32 = 0,
    num_active_constraints: u32 = 0,

    fn toPublic(self: ProfilingDataRaw) ProfilingData {
        return .{
            .integrate_ns = @floatFromInt(self.integrate_ns),
            .collision_broad_ns = @floatFromInt(self.collision_broad_ns),
            .collision_narrow_ns = @floatFromInt(self.collision_narrow_ns),
            .constraint_solve_ns = @floatFromInt(self.constraint_solve_ns),
            .total_step_ns = @floatFromInt(self.total_step_ns),
            .num_contacts = self.num_contacts,
            .num_active_constraints = self.num_active_constraints,
        };
    }
};

const PendingStep = struct {
    command: CommandBuffer,
    active_env_count: ?u32,
    dispatch_envs: u32,
    profile_enabled: bool,
    submit_elapsed_ns: u64,
    profile: ProfilingDataRaw,
};

/// Main simulation world.
pub const World = struct {
    // Metal resources
    device: Device,
    pipelines: PipelineManager,

    // Simulation state
    state: State,
    scene: Scene,

    // Configuration
    config: WorldConfig,
    params: SimParams,

    // GPU buffers for scene data
    body_data_buffer: Buffer,
    joint_data_buffer: Buffer,
    geom_data_buffer: Buffer,
    actuator_data_buffer: Buffer,
    sensor_data_buffer: Buffer,
    params_buffer: Buffer,
    constraints_buffer: Buffer,
    num_constraints_per_env: u32,

    // Constraint graph coloring for race-free parallel solving
    num_colors: u8,
    color_ranges: ?[][2]u32, // [color] = (start_offset, count)

    // GPU spatial hash buffers for broad phase (allocated when num_geoms > 512)
    spatial_hash_cell_counts: ?Buffer = null,
    spatial_hash_cell_offsets: ?Buffer = null,
    spatial_hash_sorted_geoms: ?Buffer = null,
    spatial_hash_cell_ids: ?Buffer = null,
    spatial_hash_params_buffer: ?Buffer = null,
    use_spatial_hash: bool = false,

    // Contact caching buffers for temporal coherence
    prev_contacts_buffer: Buffer,
    prev_contact_counts_buffer: Buffer,

    // Warm start factor buffer (single float passed to GPU)
    warm_start_factor_buffer: Buffer,

    // Compact environment dispatch. Full steps use identity indexing without
    // reading active_env_ids; subset steps populate only active physical IDs.
    active_env_ids_buffer: Buffer,
    env_dispatch_params_buffer: Buffer,

    // Optional GPU-side task evaluation and per-environment episode clocks.
    task_params: TaskParams,
    task_params_buffer: Buffer,
    episode_steps_buffer: Buffer,

    // GPU-resident reset templates (one environment, broadcast on reset).
    initial_positions_buffer: Buffer,
    initial_quaternions_buffer: Buffer,
    initial_velocities_buffer: Buffer,
    initial_angular_velocities_buffer: Buffer,
    initial_joint_positions_buffer: Buffer,
    initial_joint_velocities_buffer: Buffer,

    // Adaptive substeps state
    adaptive_substeps: u32,
    max_adaptive_substeps: u32,
    violation_threshold: f32,
    profiling: ProfilingDataRaw = .{},
    pending_step: ?PendingStep = null,

    // Allocator
    allocator: std.mem.Allocator,

    fn nowNanos() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    // Helper functions for quaternion math
    fn quatConjugate(q: [4]f32) [4]f32 {
        return .{ -q[0], -q[1], -q[2], q[3] };
    }

    fn quatMultiply(a: [4]f32, b: [4]f32) [4]f32 {
        return .{
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        };
    }

    /// Rotate a vector by a quaternion: v' = q * v * q^-1
    fn rotateByQuat(v: [3]f32, q: [4]f32) [3]f32 {
        // Optimized formula: v' = v + 2*q.w*(q.xyz × v) + 2*(q.xyz × (q.xyz × v))
        const qv = [3]f32{ q[0], q[1], q[2] };
        const qw = q[3];

        // t = 2 * (qv × v)
        const t = [3]f32{
            2.0 * (qv[1] * v[2] - qv[2] * v[1]),
            2.0 * (qv[2] * v[0] - qv[0] * v[2]),
            2.0 * (qv[0] * v[1] - qv[1] * v[0]),
        };

        // v' = v + qw*t + (qv × t)
        return .{
            v[0] + qw * t[0] + (qv[1] * t[2] - qv[2] * t[1]),
            v[1] + qw * t[1] + (qv[2] * t[0] - qv[0] * t[2]),
            v[2] + qw * t[2] + (qv[0] * t[1] - qv[1] * t[0]),
        };
    }

    /// Create a new simulation world.
    pub fn init(
        allocator: std.mem.Allocator,
        scene: Scene,
        config: WorldConfig,
    ) !World {
        // Initialize Metal device
        var device = try Device.init(allocator);

        // Load shader library
        try device.loadLibraryFromSource(@embedFile("../shaders/all_shaders.metal"));

        // Create pipeline manager
        var pipelines = PipelineManager.init(allocator, device.device, device.library);

        // Preload compute pipelines
        pipelines.preload(&.{
            "apply_actions",
            "apply_joint_forces",
            "forward_kinematics",
            "update_kinematic",
            "compute_forces",
            "integrate",
            "broad_phase",
            "broad_phase_count_cells",
            "broad_phase_prefix_sum",
            "broad_phase_scatter",
            "broad_phase_detect",
            "clear_uint_buffer",
            "clear_active_uint_slices",
            "sort_contacts",
            "narrow_phase",
            "solve_contacts",
            "warm_start_contacts",
            "solve_contact_velocities",
            "solve_joints",
            "update_joint_states",
            "read_sensors",
            "reset_bodies",
            "reset_env_aux",
            "warm_start_constraints",
            "store_lambda_prev",
            "cache_contacts",
            "match_cached_contacts",
            "save_prev_state",
            "xpbd_update_velocities",
            "derive_accelerations",
            "compute_task_outputs",
        }) catch |err| {
            std.log.err("World initialization failed while preloading pipelines: {}", .{err});
            return err;
        };

        // Calculate dimensions
        const num_bodies = scene.numBodies();
        const num_joints = scene.numJoints();
        const num_actuators = scene.numActuators();
        const num_geoms = scene.numGeoms();
        const obs_dim = scene.obsDim();

        // Create simulation state
        var state = try State.init(
            allocator,
            device.device,
            config.num_envs,
            num_bodies,
            num_joints,
            num_actuators,
            obs_dim,
            config.max_contacts_per_env,
        );

        // Initialize RNG
        state.initRng(config.seed);

        // Create GPU buffers for scene data
        const opts = BufferOptions{ .storage_mode = .shared };

        const body_data_buffer = try Buffer.init(
            device.device,
            num_bodies * @sizeOf(BodyDataGPU),
            opts,
        );

        const joint_data_buffer = try Buffer.init(
            device.device,
            @max(num_joints, 1) * @sizeOf(JointDataGPU),
            opts,
        );

        const geom_data_buffer = try Buffer.init(
            device.device,
            num_geoms * @sizeOf(primitives.GeomGPU),
            opts,
        );

        const actuator_data_buffer = try Buffer.init(
            device.device,
            @max(num_actuators, 1) * @sizeOf(ActuatorDataGPU),
            opts,
        );

        const sensor_data_buffer = try Buffer.init(
            device.device,
            @max(scene.sensor_config.count(), 1) * @sizeOf(@import("sensors.zig").SensorGPU),
            opts,
        );

        // Generate constraints from joints
        var template_constraints: std.ArrayListUnmanaged(XPBDConstraint) = .empty;
        defer template_constraints.deinit(allocator);

        for (scene.joints.items) |joint| {
            // Decompose joint into primitives
            const primitives_list = joint_mod.decomposeJoint(&joint, allocator) catch |err| {
                std.log.err("World initialization failed while decomposing joints: {}", .{err});
                return err;
            };
            defer allocator.free(primitives_list);

            for (primitives_list) |jc| {
                var c: XPBDConstraint = undefined;
                // Default compliance 0 (rigid)
                const compliance: f32 = if (jc.params.compliance > 0) jc.params.compliance else 0.0;

                switch (jc.constraint_type) {
                    .point => {
                        c = xpbd_mod.createPositionalConstraint(jc.body_a, jc.body_b, 0, // env 0
                            jc.local_anchor_a, jc.local_anchor_b, compliance);
                        try template_constraints.append(allocator, c);
                    },
                    .hinge => {
                        const body_a = scene.bodies.items[jc.body_a];
                        const body_b = scene.bodies.items[jc.body_b];

                        // Rotate axis from A's local frame to world, then to B's local frame
                        // axis_world = rotate(axis_a, q_a)
                        // axis_b = rotate(axis_world, q_b^-1)
                        const axis_world = rotateByQuat(jc.axis, body_a.quaternion);
                        const q_b_inv = quatConjugate(body_b.quaternion);
                        const axis_b = rotateByQuat(axis_world, q_b_inv);

                        c = xpbd_mod.createAngularConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.axis,
                            axis_b,
                            compliance,
                            jc.params.damping,
                        );
                        try template_constraints.append(allocator, c);
                    },
                    .weld => {
                        const body_a = scene.bodies.items[jc.body_a];
                        const body_b = scene.bodies.items[jc.body_b];

                        // rel_quat = q_a^-1 * q_b
                        const q_a_inv = quatConjugate(body_a.quaternion);
                        const rel_quat = quatMultiply(q_a_inv, body_b.quaternion);

                        c = xpbd_mod.createWeldConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.local_anchor_a,
                            jc.local_anchor_b,
                            rel_quat,
                            compliance,
                        );
                        try template_constraints.append(allocator, c);
                    },
                    .angular_limit => {
                        // Angular limit - limits rotation around an axis between bounds
                        c = xpbd_mod.createAngularLimitConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.axis,
                            jc.params.lower,
                            jc.params.upper,
                            compliance,
                        );
                        try template_constraints.append(allocator, c);
                    },
                    .linear_limit => {
                        // Linear limit - limits translation along an axis
                        c = .{
                            .indices = .{ jc.body_a, jc.body_b, 0, @intFromEnum(xpbd_mod.ConstraintType.linear_limit) },
                            .anchor_a = .{ jc.axis[0], jc.axis[1], jc.axis[2], compliance },
                            .anchor_b = .{ jc.local_anchor_b[0], jc.local_anchor_b[1], jc.local_anchor_b[2], 0 },
                            .axis_target = .{ 0, 0, 0, 0 },
                            .limits = .{ jc.params.lower, jc.params.upper, 0, 0 },
                            .state = .{ 0, 0, 0, 0 },
                        };
                        try template_constraints.append(allocator, c);
                    },
                    .slider => {
                        // Slider constraint - allows only translation along axis
                        // Handles both perpendicular positional lock (2 DOFs) and angular weld (3 DOFs)
                        // Reference relative quaternion stored in limits field
                        const body_a_data = scene.bodies.items[jc.body_a];
                        const body_b_data = scene.bodies.items[jc.body_b];
                        const q_a_inv = quatConjugate(body_a_data.quaternion);
                        const rel_quat = quatMultiply(q_a_inv, body_b_data.quaternion);

                        c = .{
                            .indices = .{ jc.body_a, jc.body_b, 0, @intFromEnum(xpbd_mod.ConstraintType.slider) },
                            .anchor_a = .{ jc.local_anchor_a[0], jc.local_anchor_a[1], jc.local_anchor_a[2], compliance },
                            .anchor_b = .{ jc.local_anchor_b[0], jc.local_anchor_b[1], jc.local_anchor_b[2], jc.params.damping },
                            .axis_target = .{ jc.axis[0], jc.axis[1], jc.axis[2], 0 },
                            .limits = .{ rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3] },
                            .state = .{ 0, 0, 0, 0 },
                        };
                        try template_constraints.append(allocator, c);
                    },
                    .cone_limit => {
                        // Cone limit - limits rotation angle from reference axis
                        // Use angular limit with symmetric bounds
                        c = xpbd_mod.createAngularLimitConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.axis,
                            -jc.params.upper, // Symmetric cone
                            jc.params.upper,
                            compliance,
                        );
                        try template_constraints.append(allocator, c);
                    },
                    .distance => {
                        // Distance constraint - keeps points at fixed distance
                        c = xpbd_mod.createConnectConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.local_anchor_a,
                            jc.local_anchor_b,
                            jc.params.target, // target distance
                            compliance,
                        );
                        try template_constraints.append(allocator, c);
                    },
                }
            }
        }

        const num_constraints_per_env = @as(u32, @intCast(template_constraints.items.len));
        const total_constraints = num_constraints_per_env * config.num_envs;

        const constraints_buffer = try Buffer.init(
            device.device,
            @max(total_constraints, 1) * @sizeOf(XPBDConstraint),
            opts,
        );

        // Contact caching buffers: mirror the contacts buffer for previous frame
        const contact_count = config.num_envs * config.max_contacts_per_env;
        var prev_contacts_buffer = try Buffer.init(
            device.device,
            contact_count * @sizeOf(contact.ContactGPU),
            opts,
        );
        var prev_contact_counts_buffer = try Buffer.init(
            device.device,
            config.num_envs * 4,
            opts,
        );
        try prev_contacts_buffer.zero();
        try prev_contact_counts_buffer.zero();

        // Warm start factor buffer (single float)
        const warm_start_factor_buffer = try Buffer.init(
            device.device,
            @sizeOf(f32),
            opts,
        );
        const ws_ptr = warm_start_factor_buffer.getSlice(f32);
        ws_ptr[0] = 0.8; // Default warm start factor

        // Apply graph coloring to template constraints to avoid race conditions
        // Constraints sharing bodies get different colors, allowing parallel solving per color
        var num_colors: u8 = 0;
        var color_ranges: ?[][2]u32 = null;

        if (num_constraints_per_env > 0) {
            num_colors = try xpbd_mod.colorConstraints(template_constraints.items, num_bodies, allocator);
            xpbd_mod.sortConstraintsByColor(template_constraints.items);
            color_ranges = try xpbd_mod.getColorRanges(template_constraints.items, num_colors, allocator);
        }

        // Populate for all envs (constraints are sorted by color)
        if (num_constraints_per_env > 0) {
            const constraints_ptr = constraints_buffer.getSlice(XPBDConstraint);
            for (0..config.num_envs) |env_id| {
                for (template_constraints.items, 0..) |tmpl, i| {
                    var c = tmpl;
                    c.indices[2] = @intCast(env_id); // Update env_id
                    constraints_ptr[env_id * num_constraints_per_env + i] = c;
                }
            }
        }

        const params_buffer = try Buffer.init(
            device.device,
            @sizeOf(SimParams),
            opts,
        );

        const active_env_ids_buffer = try Buffer.init(
            device.device,
            config.num_envs * @sizeOf(u32),
            opts,
        );
        const active_env_ids = active_env_ids_buffer.getSlice(u32);
        for (active_env_ids, 0..) |*env_id, i| env_id.* = @intCast(i);

        const env_dispatch_params_buffer = try Buffer.init(
            device.device,
            @sizeOf(EnvDispatchParams),
            opts,
        );
        env_dispatch_params_buffer.getSlice(EnvDispatchParams)[0] = .{
            .num_envs = config.num_envs,
            .dispatch_envs = config.num_envs,
            .use_active_env_ids = 0,
        };

        const task_params: TaskParams = .{};
        const task_params_buffer = try Buffer.init(
            device.device,
            @sizeOf(TaskParams),
            opts,
        );
        task_params_buffer.getSlice(TaskParams)[0] = task_params;

        var episode_steps_buffer = try Buffer.init(
            device.device,
            config.num_envs * @sizeOf(u32),
            opts,
        );
        try episode_steps_buffer.zero();

        const initial_positions_buffer = try Buffer.init(device.device, @max(num_bodies, 1) * 16, opts);
        const initial_quaternions_buffer = try Buffer.init(device.device, @max(num_bodies, 1) * 16, opts);
        const initial_velocities_buffer = try Buffer.init(device.device, @max(num_bodies, 1) * 16, opts);
        const initial_angular_velocities_buffer = try Buffer.init(device.device, @max(num_bodies, 1) * 16, opts);
        const initial_joint_positions_buffer = try Buffer.init(device.device, @max(num_joints, 1) * 4, opts);
        const initial_joint_velocities_buffer = try Buffer.init(device.device, @max(num_joints, 1) * 4, opts);

        // Build params
        const params = SimParams{
            .num_envs = config.num_envs,
            .num_bodies = num_bodies,
            .num_joints = num_joints,
            .num_actuators = num_actuators,
            .num_geoms = num_geoms,
            .num_sensors = scene.sensor_config.count(),
            .max_contacts = config.max_contacts_per_env,
            .contact_iterations = config.contact_iterations,
            .dt = config.timestep / @as(f32, @floatFromInt(config.substeps)),
            .gravity_x = scene.physics_config.gravity[0],
            .gravity_y = scene.physics_config.gravity[1],
            .gravity_z = scene.physics_config.gravity[2],
            .friction = scene.physics_config.friction,
            .restitution = scene.physics_config.restitution,
            .baumgarte = scene.physics_config.baumgarte_factor,
            .slop = scene.physics_config.penetration_slop,
            .target_color = 0,
            .num_constraints = 0,
            .constraint_offset = 0,
            .obs_dim = obs_dim,
        };

        // Allocate spatial hash buffers for large scenes
        const use_spatial_hash = num_geoms > 512;
        var sh_cell_counts: ?Buffer = null;
        var sh_cell_offsets: ?Buffer = null;
        var sh_sorted_geoms: ?Buffer = null;
        var sh_cell_ids: ?Buffer = null;
        var sh_params_buf: ?Buffer = null;

        if (use_spatial_hash) {
            // Grid dimensions: use a reasonable default (64^3 = 262144 cells)
            const grid_dim: u32 = 64;
            const total_cells = grid_dim * grid_dim * grid_dim;
            const total_geoms = config.num_envs * num_geoms;

            sh_cell_counts = try Buffer.init(device.device, config.num_envs * total_cells * @sizeOf(u32), opts);
            sh_cell_offsets = try Buffer.init(device.device, config.num_envs * total_cells * @sizeOf(u32), opts);
            sh_sorted_geoms = try Buffer.init(device.device, total_geoms * @sizeOf(u32), opts);
            sh_cell_ids = try Buffer.init(device.device, total_geoms * @sizeOf(u32), opts);
            sh_params_buf = try Buffer.init(device.device, @sizeOf(SpatialHashParamsGPU), opts);

            // Cell size must exceed the largest pairwise reach or the ±1-cell
            // neighbor query misses overlapping geoms in non-adjacent cells.
            var max_reach: f32 = 0;
            for (scene.geoms.items) |g| {
                if (g.geom_type == .plane) continue;
                const reach = switch (g.geom_type) {
                    .sphere => g.size[0],
                    .capsule => g.size[0] + g.size[1],
                    .box, .heightfield => @sqrt(
                        g.size[0] * g.size[0] +
                            g.size[1] * g.size[1] +
                            g.size[2] * g.size[2],
                    ),
                    .cylinder => @sqrt(
                        g.size[0] * g.size[0] + g.size[1] * g.size[1],
                    ),
                    .mesh => g.size[0],
                    .plane => unreachable,
                };
                max_reach = @max(max_reach, reach);
            }
            const cell_size: f32 = @max(1.0, 2.0 * max_reach + 0.2);

            // Center the grid on the origin — scenes are origin-centered, and an
            // unshifted grid clamps all negative coordinates into boundary cells.
            const half_extent = 0.5 * cell_size * @as(f32, @floatFromInt(grid_dim));

            const sh_params_ptr = sh_params_buf.?.getSlice(SpatialHashParamsGPU);
            sh_params_ptr[0] = .{
                .grid_dim_x = grid_dim,
                .grid_dim_y = grid_dim,
                .grid_dim_z = grid_dim,
                .total_cells = total_cells,
                .cell_size = cell_size,
                .inv_cell_size = 1.0 / cell_size,
                .num_geoms = num_geoms,
                .num_envs = config.num_envs,
                .max_contacts = config.max_contacts_per_env,
                .origin_x = -half_extent,
                .origin_y = -half_extent,
                .origin_z = -half_extent,
            };

            try sh_cell_counts.?.zero();
            try sh_cell_offsets.?.zero();
        }

        var world = World{
            .device = device,
            .pipelines = pipelines,
            .state = state,
            .scene = scene,
            .config = config,
            .params = params,
            .body_data_buffer = body_data_buffer,
            .joint_data_buffer = joint_data_buffer,
            .geom_data_buffer = geom_data_buffer,
            .actuator_data_buffer = actuator_data_buffer,
            .sensor_data_buffer = sensor_data_buffer,
            .params_buffer = params_buffer,
            .constraints_buffer = constraints_buffer,
            .num_constraints_per_env = num_constraints_per_env,
            .num_colors = num_colors,
            .color_ranges = color_ranges,
            .spatial_hash_cell_counts = sh_cell_counts,
            .spatial_hash_cell_offsets = sh_cell_offsets,
            .spatial_hash_sorted_geoms = sh_sorted_geoms,
            .spatial_hash_cell_ids = sh_cell_ids,
            .spatial_hash_params_buffer = sh_params_buf,
            .use_spatial_hash = use_spatial_hash,
            .prev_contacts_buffer = prev_contacts_buffer,
            .prev_contact_counts_buffer = prev_contact_counts_buffer,
            .warm_start_factor_buffer = warm_start_factor_buffer,
            .active_env_ids_buffer = active_env_ids_buffer,
            .env_dispatch_params_buffer = env_dispatch_params_buffer,
            .task_params = task_params,
            .task_params_buffer = task_params_buffer,
            .episode_steps_buffer = episode_steps_buffer,
            .initial_positions_buffer = initial_positions_buffer,
            .initial_quaternions_buffer = initial_quaternions_buffer,
            .initial_velocities_buffer = initial_velocities_buffer,
            .initial_angular_velocities_buffer = initial_angular_velocities_buffer,
            .initial_joint_positions_buffer = initial_joint_positions_buffer,
            .initial_joint_velocities_buffer = initial_joint_velocities_buffer,
            .adaptive_substeps = config.substeps,
            .max_adaptive_substeps = @max(config.substeps * 4, 8),
            .violation_threshold = 0.01,
            .allocator = allocator,
        };

        // Upload scene data to GPU
        try world.uploadSceneData();

        // Initialize state from scene
        try world.initializeState();

        // Capture a single-environment template in Metal-visible shared
        // buffers. Reset kernels broadcast this template without host state
        // walks, and the template remains stable as environment 0 evolves.
        @memcpy(
            world.initial_positions_buffer.getAlignedSlice([4]f32, 16)[0..num_bodies],
            world.state.getPositions()[0..num_bodies],
        );
        @memcpy(
            world.initial_quaternions_buffer.getAlignedSlice([4]f32, 16)[0..num_bodies],
            world.state.getQuaternions()[0..num_bodies],
        );
        @memcpy(
            world.initial_velocities_buffer.getAlignedSlice([4]f32, 16)[0..num_bodies],
            world.state.getVelocities()[0..num_bodies],
        );
        @memcpy(
            world.initial_angular_velocities_buffer.getAlignedSlice([4]f32, 16)[0..num_bodies],
            world.state.getAngularVelocities()[0..num_bodies],
        );
        @memcpy(
            world.initial_joint_positions_buffer.getSlice(f32)[0..num_joints],
            world.state.getJointPositions()[0..num_joints],
        );
        @memcpy(
            world.initial_joint_velocities_buffer.getSlice(f32)[0..num_joints],
            world.state.getJointVelocities()[0..num_joints],
        );

        return world;
    }

    /// Upload scene definition to GPU buffers.
    fn uploadSceneData(self: *World) !void {
        // Upload body data
        const body_data = self.body_data_buffer.getSlice(BodyDataGPU);
        for (self.scene.bodies.items, 0..) |body_def, i| {
            body_data[i] = BodyDataGPU.fromBodyDef(&body_def);
        }

        // Upload joint data
        if (self.scene.joints.items.len > 0) {
            const joint_data = self.joint_data_buffer.getSlice(JointDataGPU);
            for (self.scene.joints.items, 0..) |joint_def, i| {
                joint_data[i] = JointDataGPU.fromJointDef(&joint_def);
            }
        }

        // Upload geom data
        const geom_data = self.geom_data_buffer.getSlice(primitives.GeomGPU);
        for (self.scene.geoms.items, 0..) |geom, i| {
            geom_data[i] = primitives.GeomGPU.fromGeom(&geom);
        }

        // Upload actuator data
        if (self.scene.actuators.items.len > 0) {
            const actuator_data = self.actuator_data_buffer.getSlice(ActuatorDataGPU);
            for (self.scene.actuators.items, 0..) |actuator, i| {
                actuator_data[i] = ActuatorDataGPU.fromActuatorDef(&actuator);
            }
        }

        // Upload sensor data
        if (self.scene.sensor_config.sensors.items.len > 0) {
            const sensors = @import("sensors.zig");
            const sensor_data = self.sensor_data_buffer.getSlice(sensors.SensorGPU);
            var offset: u32 = 0;
            for (self.scene.sensor_config.sensors.items, 0..) |sensor, i| {
                sensor_data[i] = sensors.SensorGPU.fromSensor(&sensor, offset);
                offset += sensor.dim;
            }
        }

        // Upload params
        const params_ptr = self.params_buffer.getSlice(SimParams);
        params_ptr[0] = self.params;
    }

    /// Initialize state from scene definition.
    fn initializeState(self: *World) !void {
        const positions = self.state.getPositions();
        const quaternions = self.state.getQuaternions();
        const velocities = self.state.getVelocities();
        const angular_vels = self.state.getAngularVelocities();
        const inv_mass_inertia = self.state.inv_mass_inertia_buffer.getSlice([4]f32);

        // Initialize all environments with the same initial state
        for (0..self.config.num_envs) |env| {
            for (self.scene.bodies.items, 0..) |body_def, b| {
                const idx = self.state.bodyIndex(@intCast(env), @intCast(b));

                positions[idx] = .{ body_def.position[0], body_def.position[1], body_def.position[2], 0 };
                quaternions[idx] = body_def.quaternion;
                velocities[idx] = .{ body_def.linear_velocity[0], body_def.linear_velocity[1], body_def.linear_velocity[2], 0 };
                angular_vels[idx] = .{ body_def.angular_velocity[0], body_def.angular_velocity[1], body_def.angular_velocity[2], 0 };

                const inv_mass = body_def.invMass();
                const inv_inertia = body_def.invInertia();
                inv_mass_inertia[idx] = .{ inv_mass, inv_inertia[0], inv_inertia[1], inv_inertia[2] };
            }
        }

        // Zero velocities and forces
        try self.state.forces_buffer.zero();
        try self.state.torques_buffer.zero();
        try self.state.joint_positions_buffer.zero();
        try self.state.joint_velocities_buffer.zero();
    }

    /// Step the simulation with adaptive substeps.
    pub fn step(self: *World, actions: []const f32, substeps: u32) !void {
        try self.stepAsync(actions, substeps);
        try self.waitStep();
    }

    /// Encode and commit a simulation step without waiting for GPU completion.
    pub fn stepAsync(self: *World, actions: []const f32, substeps: u32) !void {
        try self.submitStepInternal(actions, substeps, null, null);
    }

    /// Commit a full step using actions already written into shared memory.
    pub fn stepCurrentActionsAsync(self: *World, substeps: u32) !void {
        try self.submitStepInternal(null, substeps, null, null);
    }

    fn submitStepInternal(
        self: *World,
        actions: ?[]const f32,
        substeps: u32,
        active_env_count: ?u32,
        reset_env_count: ?u32,
    ) !void {
        if (self.pending_step != null) return error.StepPending;

        // Callers with a writable unified-memory action view can skip this copy.
        if (actions) |new_actions| try self.state.setActions(new_actions);

        // Use adaptive substep count if no explicit override
        const actual_substeps = if (substeps == 0) self.adaptive_substeps else substeps;
        const dt = self.config.timestep / @as(f32, @floatFromInt(@max(actual_substeps, 1)));
        self.params.dt = dt;
        self.params_buffer.getSlice(SimParams)[0] = self.params;
        const dispatch_envs = active_env_count orelse self.config.num_envs;
        self.env_dispatch_params_buffer.getSlice(EnvDispatchParams)[0] = .{
            .num_envs = self.config.num_envs,
            .dispatch_envs = dispatch_envs,
            .use_active_env_ids = @intFromBool(active_env_count != null),
        };

        var step_profile: ProfilingDataRaw = .{};
        const profile_enabled = self.config.enable_profiling;
        const step_start = if (profile_enabled) nowNanos() else 0;

        // ONE command buffer + ONE encoder for ALL substeps — single GPU submission,
        // using memory barriers instead of separate encoder create/destroy per stage.
        var cmd = try CommandBuffer.init(self.device.command_queue);
        var encoder = try cmd.computeEncoder();

        // Gymnasium NEXT_STEP autoreset can restore the selected environments
        // before physics in the same command buffer. The new actions were copied
        // above, so the fused reset deliberately preserves the action buffer.
        if (reset_env_count) |count| {
            try self.encodeReset(&encoder, count, count != self.config.num_envs, true, false);
            encoder.memoryBarrier(.buffers);
        }

        encoder.setBuffer(&self.env_dispatch_params_buffer, 0, 29);
        encoder.setBuffer(&self.active_env_ids_buffer, 0, 30);

        for (0..actual_substeps) |substep_index| {
            var substep_profile: ProfilingDataRaw = .{};
            try self.stepOnce(
                &encoder,
                if (profile_enabled) &substep_profile else null,
                substep_index == 0,
                substep_index + 1 == actual_substeps,
                dispatch_envs,
            );

            if (profile_enabled) {
                step_profile.integrate_ns += substep_profile.integrate_ns;
                step_profile.collision_broad_ns += substep_profile.collision_broad_ns;
                step_profile.collision_narrow_ns += substep_profile.collision_narrow_ns;
                step_profile.constraint_solve_ns += substep_profile.constraint_solve_ns;
            }

            encoder.memoryBarrier(.buffers); // barrier between substeps
        }

        // Task outputs are derived from the final state once per public step,
        // not once per physics substep, and stay in shared Metal buffers.
        if (self.task_params.enabled != 0) {
            const pipeline = try self.pipelines.getPipeline("compute_task_outputs");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.actions_buffer, 0, 2);
            encoder.setBuffer(&self.state.rewards_buffer, 0, 3);
            encoder.setBuffer(&self.state.dones_buffer, 0, 4);
            encoder.setBuffer(&self.episode_steps_buffer, 0, 5);
            encoder.setBuffer(&self.task_params_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.dispatch1D(pipeline, dispatch_envs);
        }

        encoder.endEncoding();
        // Command buffers returned by Metal are autoreleased. Keep an explicit
        // reference across the public async boundary, then release in wait.
        _ = objc.retain(cmd.buffer);
        cmd.commit();
        self.pending_step = .{
            .command = cmd,
            .active_env_count = active_env_count,
            .dispatch_envs = dispatch_envs,
            .profile_enabled = profile_enabled,
            .submit_elapsed_ns = if (profile_enabled) nowNanos() - step_start else 0,
            .profile = step_profile,
        };
    }

    /// Wait for the in-flight step and make shared-memory outputs safe to read.
    pub fn waitStep(self: *World) !void {
        var pending = self.pending_step orelse return error.NoPendingStep;
        defer {
            objc.release(pending.command.buffer);
            self.pending_step = null;
        }

        const wait_start = nowNanos();
        pending.command.waitUntilCompleted();
        const wait_elapsed_ns = nowNanos() - wait_start;
        if (!pending.command.succeeded()) return error.ExecutionFailed;

        var step_profile = pending.profile;
        const profile_enabled = pending.profile_enabled;
        const active_env_count = pending.active_env_count;
        const dispatch_envs = pending.dispatch_envs;

        if (profile_enabled) {
            // Host cost excludes useful CPU work performed between submission
            // and wait; this remains a host-side metric, not GPU stage timing.
            step_profile.total_step_ns = pending.submit_elapsed_ns + wait_elapsed_ns;

            // Aggregate active contact count across environments.
            var total_contacts: u64 = 0;
            const contact_counts = self.state.contact_counts_buffer.getSlice(u32);
            if (active_env_count) |active_count| {
                const active_ids = self.active_env_ids_buffer.getSlice(u32);
                for (active_ids[0..active_count]) |env_id| {
                    total_contacts += contact_counts[env_id];
                }
            } else {
                for (contact_counts) |count| total_contacts += count;
            }
            step_profile.num_contacts = @intCast(@min(total_contacts, std.math.maxInt(u32)));
            step_profile.num_active_constraints = self.num_constraints_per_env * dispatch_envs;
            self.profiling = step_profile;
        } else {
            self.profiling = .{};
        }

        // Adaptive substep detection: check max constraint violation after solving.
        // If violations exceed threshold, increase substeps for next frame.
        // If violations are well below threshold, decrease substeps to save compute.
        if (self.num_constraints_per_env > 0 and dispatch_envs > 0) {
            const constraints_slice = self.constraints_buffer.getSlice(XPBDConstraint);
            var max_violation: f32 = 0.0;

            // Sample one stepped environment. Checking every environment would
            // be expensive, but a subset step must not adapt from stale state
            // in physical environment 0 when that environment was inactive.
            const sample_env_id: usize = if (active_env_count != null)
                self.active_env_ids_buffer.getSlice(u32)[0]
            else
                0;
            const constraint_start = sample_env_id * self.num_constraints_per_env;
            for (0..self.num_constraints_per_env) |ci| {
                const violation = @abs(constraints_slice[constraint_start + ci].state[2]);
                if (violation > max_violation) {
                    max_violation = violation;
                }
            }

            if (max_violation > self.violation_threshold) {
                // Increase substeps (up to max) when constraints are violated
                if (self.adaptive_substeps < self.max_adaptive_substeps) {
                    self.adaptive_substeps += 1;
                }
            } else if (max_violation < self.violation_threshold * 0.25) {
                // Decrease substeps when well within tolerance
                if (self.adaptive_substeps > self.config.substeps) {
                    self.adaptive_substeps -= 1;
                }
            }
        }
    }

    pub fn hasPendingStep(self: *const World) bool {
        return self.pending_step != null;
    }

    fn prepareActiveEnvIds(self: *World, env_mask: []const u8) u32 {
        const ids = self.active_env_ids_buffer.getSlice(u32);
        var active_count: u32 = 0;
        for (env_mask, 0..) |active, physical_env_id| {
            if (active == 0) continue;
            ids[active_count] = @intCast(physical_env_id);
            active_count += 1;
        }
        return active_count;
    }

    /// Step only the masked environments while keeping others unchanged.
    /// The CPU compacts the small mask into active physical environment IDs;
    /// Metal kernels dispatch only active work while retaining stable state indices.
    pub fn stepSubset(self: *World, actions: []const f32, env_mask: []const u8, substeps: u32) !void {
        try self.stepSubsetAsync(actions, env_mask, substeps);
        try self.waitStep();
    }

    /// Commit a compact masked step without waiting for GPU completion.
    pub fn stepSubsetAsync(self: *World, actions: []const f32, env_mask: []const u8, substeps: u32) !void {
        if (self.pending_step != null) return error.StepPending;
        const num_envs: usize = @intCast(self.config.num_envs);
        if (env_mask.len != num_envs) return error.InvalidSize;

        const compact_count = self.prepareActiveEnvIds(env_mask);
        if (compact_count == num_envs) return self.submitStepInternal(actions, substeps, null, null);
        try self.submitStepInternal(actions, substeps, compact_count, null);
    }

    /// Reset selected environments and then step the full batch in one Metal
    /// command buffer. This is intentionally full-step only: the compact ID
    /// buffer holds the reset selection for the duration of the submission.
    pub fn stepWithResetAsync(
        self: *World,
        actions: []const f32,
        reset_mask: []const u8,
        substeps: u32,
    ) !void {
        if (self.pending_step != null) return error.StepPending;
        const num_envs: usize = @intCast(self.config.num_envs);
        if (reset_mask.len != num_envs) return error.InvalidSize;

        const reset_count = self.prepareActiveEnvIds(reset_mask);
        try self.submitStepInternal(
            actions,
            substeps,
            null,
            if (reset_count == 0) null else reset_count,
        );
    }

    /// Fused autoreset variant for actions already written into shared memory.
    pub fn stepWithResetCurrentActionsAsync(
        self: *World,
        reset_mask: []const u8,
        substeps: u32,
    ) !void {
        if (self.pending_step != null) return error.StepPending;
        const num_envs: usize = @intCast(self.config.num_envs);
        if (reset_mask.len != num_envs) return error.InvalidSize;

        const reset_count = self.prepareActiveEnvIds(reset_mask);
        try self.submitStepInternal(
            null,
            substeps,
            null,
            if (reset_count == 0) null else reset_count,
        );
    }

    /// Execute one physics substep.
    /// Encodes all GPU work into the provided compute encoder using memory barriers
    /// between dependent dispatch groups, instead of creating separate encoders.
    fn stepOnce(
        self: *World,
        encoder: *ComputeEncoder,
        profile: ?*ProfilingDataRaw,
        first_substep: bool,
        final_substep: bool,
        dispatch_envs: u32,
    ) !void {

        // GROUP 0: GPU-side count clears. CPU memsets execute at encode time —
        // before any GPU work runs — so per-substep resets must be GPU kernels:
        // contact counts would otherwise accumulate across substeps until they
        // saturate max_contacts, and spatial-hash cell counts would double on
        // every substep after the first.
        {
            const pipeline = try self.pipelines.getPipeline("clear_active_uint_slices");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 0);
            var elements_per_env: u32 = 1;
            encoder.setBytes(std.mem.asBytes(&elements_per_env), 1);
            encoder.dispatch1D(pipeline, dispatch_envs);

            if (self.use_spatial_hash) {
                var sh_cell_counts = self.spatial_hash_cell_counts.?;
                const sh_params = self.spatial_hash_params_buffer.?.getSlice(SpatialHashParamsGPU)[0];
                const cell_clear_count: u32 = sh_params.total_cells * dispatch_envs;
                encoder.setBuffer(&sh_cell_counts, 0, 0);
                elements_per_env = sh_params.total_cells;
                encoder.setBytes(std.mem.asBytes(&elements_per_env), 1);
                encoder.dispatch1D(pipeline, cell_clear_count);
            }
        }

        encoder.memoryBarrier(.buffers);

        // GROUP A: warm_start_constraints + apply_actions
        // (independent — write to different buffers: constraints vs joint_torques)
        if (self.num_constraints_per_env > 0) {
            var local_params = self.params_buffer.getSlice(SimParams)[0];
            local_params.target_color = self.num_constraints_per_env;

            const pipeline = try self.pipelines.getPipeline("warm_start_constraints");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.constraints_buffer, 0, 0);
            encoder.setBytes(std.mem.asBytes(&local_params), 1);
            encoder.setBuffer(&self.warm_start_factor_buffer, 0, 2);
            encoder.dispatch1D(pipeline, dispatch_envs * self.num_constraints_per_env);
        }

        {
            const pipeline = try self.pipelines.getPipeline("apply_actions");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.actions_buffer, 0, 0);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 1);
            encoder.setBuffer(&self.actuator_data_buffer, 0, 2);
            encoder.setBuffer(&self.params_buffer, 0, 3);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_actuators);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP B: compute_forces + update_kinematic
        // (independent — forces/torques vs kinematic positions/quaternions)
        {
            const pipeline = try self.pipelines.getPipeline("compute_forces");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.forces_buffer, 0, 2);
            encoder.setBuffer(&self.state.torques_buffer, 0, 3);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 4);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 5);
            encoder.setBuffer(&self.params_buffer, 0, 6);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 7);
            encoder.setBuffer(&self.body_data_buffer, 0, 8);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }

        {
            const pipeline = try self.pipelines.getPipeline("update_kinematic");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.body_data_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP C: apply_joint_forces + save_prev_state
        // (independent — atomic forces/torques vs prev_positions/prev_quaternions)
        if (self.params.num_joints > 0) {
            const pipeline = try self.pipelines.getPipeline("apply_joint_forces");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.torques_buffer, 0, 0);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 1);
            encoder.setBuffer(&self.joint_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.setBuffer(&self.state.forces_buffer, 0, 5);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_joints);
        }

        {
            const pipeline = try self.pipelines.getPipeline("save_prev_state");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.state.prev_positions_buffer, 0, 2);
            encoder.setBuffer(&self.state.prev_quaternions_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 5);
            encoder.setBuffer(&self.state.prev_velocities_buffer, 0, 6);
            var snapshot_full_step: u32 = @intFromBool(first_substep);
            encoder.setBytes(std.mem.asBytes(&snapshot_full_step), 7);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP D: integrate
        const integrate_start = if (profile != null) nowNanos() else 0;
        {
            const pipeline = try self.pipelines.getPipeline("integrate");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.forces_buffer, 0, 4);
            encoder.setBuffer(&self.state.torques_buffer, 0, 5);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.setBuffer(&self.body_data_buffer, 0, 8);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }
        if (profile) |p| {
            p.integrate_ns += nowNanos() - integrate_start;
        }

        encoder.memoryBarrier(.buffers);

        // GROUP E: broad_phase (O(n²) fallback or 3-pass spatial hash)
        const broad_start = if (profile != null) nowNanos() else 0;
        if (self.use_spatial_hash) {
            // 3-pass GPU spatial hash broad phase
            var sh_cell_counts = self.spatial_hash_cell_counts.?;
            var sh_cell_offsets = self.spatial_hash_cell_offsets.?;
            var sh_sorted_geoms = self.spatial_hash_sorted_geoms.?;
            var sh_cell_ids = self.spatial_hash_cell_ids.?;
            var sh_params_buf = self.spatial_hash_params_buffer.?;

            // Pass 1: Count cells
            {
                const pipeline = try self.pipelines.getPipeline("broad_phase_count_cells");
                encoder.setPipeline(pipeline);
                encoder.setBuffer(&self.state.positions_buffer, 0, 0);
                encoder.setBuffer(&self.geom_data_buffer, 0, 1);
                encoder.setBuffer(&sh_cell_counts, 0, 2);
                encoder.setBuffer(&sh_cell_ids, 0, 3);
                encoder.setBuffer(&sh_params_buf, 0, 4);
                encoder.setBuffer(&self.params_buffer, 0, 5);
                encoder.setBuffer(&self.state.quaternions_buffer, 0, 6);
                encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_geoms);
            }

            encoder.memoryBarrier(.buffers);

            // Pass 2: Prefix sum
            {
                const pipeline = try self.pipelines.getPipeline("broad_phase_prefix_sum");
                encoder.setPipeline(pipeline);
                encoder.setBuffer(&sh_cell_counts, 0, 0);
                encoder.setBuffer(&sh_cell_offsets, 0, 1);
                encoder.setBuffer(&sh_params_buf, 0, 2);
                encoder.dispatch1D(pipeline, dispatch_envs);
            }

            encoder.memoryBarrier(.buffers);

            // Pass 3a: Scatter geoms into sorted order. Separate dispatch from
            // detection — a threadgroup_barrier cannot synchronize the whole
            // grid, so scatter must fully complete before any thread queries.
            {
                const pipeline = try self.pipelines.getPipeline("broad_phase_scatter");
                encoder.setPipeline(pipeline);
                encoder.setBuffer(&sh_cell_ids, 0, 0);
                encoder.setBuffer(&sh_cell_offsets, 0, 1);
                encoder.setBuffer(&sh_cell_counts, 0, 2);
                encoder.setBuffer(&sh_sorted_geoms, 0, 3);
                encoder.setBuffer(&sh_params_buf, 0, 4);
                encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_geoms);
            }

            encoder.memoryBarrier(.buffers);

            // Pass 3b: Detect collisions
            {
                const pipeline = try self.pipelines.getPipeline("broad_phase_detect");
                encoder.setPipeline(pipeline);
                encoder.setBuffer(&self.state.positions_buffer, 0, 0);
                encoder.setBuffer(&self.geom_data_buffer, 0, 1);
                encoder.setBuffer(&sh_cell_ids, 0, 2);
                encoder.setBuffer(&sh_cell_offsets, 0, 3);
                encoder.setBuffer(&sh_cell_counts, 0, 4);
                encoder.setBuffer(&sh_sorted_geoms, 0, 5);
                encoder.setBuffer(&self.state.contacts_buffer, 0, 6);
                encoder.setBuffer(&self.state.contact_counts_buffer, 0, 7);
                encoder.setBuffer(&sh_params_buf, 0, 8);
                encoder.setBuffer(&self.params_buffer, 0, 9);
                encoder.setBuffer(&self.state.quaternions_buffer, 0, 10);
                encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_geoms);
            }
        } else {
            // O(n²) fallback for small scenes
            const pipeline = try self.pipelines.getPipeline("broad_phase");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.geom_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 3);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_geoms);
        }
        if (profile) |p| {
            p.collision_broad_ns += nowNanos() - broad_start;
        }

        encoder.memoryBarrier(.buffers);

        // GROUP E.5: sort_contacts (deterministic ordering)
        {
            const pipeline = try self.pipelines.getPipeline("sort_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 0);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 1);
            encoder.setBuffer(&self.params_buffer, 0, 2);
            encoder.dispatch1D(pipeline, dispatch_envs);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP F: narrow_phase
        const narrow_start = if (profile != null) nowNanos() else 0;
        {
            const pipeline = try self.pipelines.getPipeline("narrow_phase");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.geom_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 3);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
        }
        if (profile) |p| {
            p.collision_narrow_ns += nowNanos() - narrow_start;
        }

        encoder.memoryBarrier(.buffers);

        // GROUP G: match_cached_contacts
        {
            const pipeline = try self.pipelines.getPipeline("match_cached_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 0);
            encoder.setBuffer(&self.prev_contacts_buffer, 0, 1);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 2);
            encoder.setBuffer(&self.prev_contact_counts_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP H: solve_joints loop
        // (barrier between each color; barrier between each iteration)
        const solve_start = if (profile != null) nowNanos() else 0;
        if (self.num_constraints_per_env > 0 and self.color_ranges != null) {
            const params_ptr = self.params_buffer.getSlice(SimParams);
            const ranges = self.color_ranges.?;

            for (0..self.config.contact_iterations) |_| {
                for (0..self.num_colors) |color| {
                    const range = ranges[color];
                    const offset = range[0];
                    const count = range[1];

                    if (count == 0) continue;

                    var local_params = params_ptr[0];
                    local_params.constraint_offset = offset;
                    local_params.num_constraints = count;
                    local_params.target_color = self.num_constraints_per_env;

                    const pipeline = try self.pipelines.getPipeline("solve_joints");
                    encoder.setPipeline(pipeline);
                    encoder.setBuffer(&self.state.positions_buffer, 0, 0);
                    encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
                    encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
                    encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
                    encoder.setBuffer(&self.constraints_buffer, 0, 4);
                    encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 5);
                    encoder.setBytes(std.mem.asBytes(&local_params), 6);
                    encoder.setBuffer(&self.body_data_buffer, 0, 7);
                    encoder.dispatch1D(pipeline, dispatch_envs * count);

                    encoder.memoryBarrier(.buffers);
                }
            }
        }

        // GROUP I: one position-correction pass. Narrow phase computes one
        // penetration depth per substep, so repeating the same correction
        // would inject energy when velocities are reconstructed below.
        {
            const pipeline = try self.pipelines.getPipeline("solve_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 4);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 5);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
        }
        encoder.memoryBarrier(.buffers);
        if (profile) |p| {
            p.constraint_solve_ns += nowNanos() - solve_start;
        }

        // GROUP J: xpbd_update_velocities
        {
            const pipeline = try self.pipelines.getPipeline("xpbd_update_velocities");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.prev_positions_buffer, 0, 4);
            encoder.setBuffer(&self.state.prev_quaternions_buffer, 0, 5);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.setBuffer(&self.state.prev_velocities_buffer, 0, 8);
            encoder.setBuffer(&self.state.accelerations_buffer, 0, 9);
            const inv_full_step: f32 = if (self.config.timestep > 0) 1.0 / self.config.timestep else 0.0;
            // Contact impulses run after this kernel, so full-step acceleration
            // is derived in a dedicated final-substep pass below.
            var derive_acceleration: u32 = 0;
            encoder.setBytes(std.mem.asBytes(&inv_full_step), 10);
            encoder.setBytes(std.mem.asBytes(&derive_acceleration), 11);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP J.5: warm start and iteratively solve velocity-level contact
        // impulses after XPBD velocity reconstruction so normal restitution
        // and Coulomb friction are not overwritten.
        {
            const pipeline = try self.pipelines.getPipeline("warm_start_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 4);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 5);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
        }
        encoder.memoryBarrier(.buffers);

        for (0..self.config.contact_iterations) |iteration_index| {
            const pipeline = try self.pipelines.getPipeline("solve_contact_velocities");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 4);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 5);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            var velocity_iteration: u32 = @intCast(iteration_index);
            encoder.setBytes(std.mem.asBytes(&velocity_iteration), 8);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
            encoder.memoryBarrier(.buffers);
        }

        if (final_substep) {
            const pipeline = try self.pipelines.getPipeline("derive_accelerations");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 0);
            encoder.setBuffer(&self.state.prev_velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.accelerations_buffer, 0, 2);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            const inv_full_step: f32 = if (self.config.timestep > 0)
                1.0 / self.config.timestep
            else
                0.0;
            encoder.setBytes(std.mem.asBytes(&inv_full_step), 5);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
            encoder.memoryBarrier(.buffers);
        }

        // GROUP K: update_joint_states
        if (self.params.num_joints > 0) {
            const pipeline = try self.pipelines.getPipeline("update_joint_states");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.joint_data_buffer, 0, 4);
            encoder.setBuffer(&self.state.joint_positions_buffer, 0, 5);
            encoder.setBuffer(&self.state.joint_velocities_buffer, 0, 6);
            encoder.setBuffer(&self.params_buffer, 0, 7);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_joints);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP L: read_sensors + cache_contacts
        // (independent — observations vs prev_contacts)
        {
            const pipeline = try self.pipelines.getPipeline("read_sensors");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.joint_positions_buffer, 0, 4);
            encoder.setBuffer(&self.state.joint_velocities_buffer, 0, 5);
            encoder.setBuffer(&self.sensor_data_buffer, 0, 6);
            encoder.setBuffer(&self.state.observations_buffer, 0, 7);
            encoder.setBuffer(&self.params_buffer, 0, 8);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_sensors);
        }

        {
            const pipeline = try self.pipelines.getPipeline("cache_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 0);
            encoder.setBuffer(&self.prev_contacts_buffer, 0, 1);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 2);
            encoder.setBuffer(&self.prev_contact_counts_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.dispatch1D(pipeline, dispatch_envs * self.config.max_contacts_per_env);
        }

        encoder.memoryBarrier(.buffers);

        // GROUP M: store_lambda_prev
        if (self.num_constraints_per_env > 0) {
            var local_params = self.params_buffer.getSlice(SimParams)[0];
            local_params.target_color = self.num_constraints_per_env;

            const pipeline = try self.pipelines.getPipeline("store_lambda_prev");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.constraints_buffer, 0, 0);
            encoder.setBytes(std.mem.asBytes(&local_params), 1);
            encoder.dispatch1D(pipeline, dispatch_envs * self.num_constraints_per_env);
        }
    }

    fn encodeReset(
        self: *World,
        encoder: *ComputeEncoder,
        dispatch_envs: u32,
        use_active_ids: bool,
        preserve_actions: bool,
        refresh_observations: bool,
    ) !void {
        var reset_dispatch: EnvDispatchParams = .{
            .num_envs = self.config.num_envs,
            .dispatch_envs = dispatch_envs,
            .use_active_env_ids = @intFromBool(use_active_ids),
        };
        encoder.setBytes(std.mem.asBytes(&reset_dispatch), 29);
        encoder.setBuffer(&self.active_env_ids_buffer, 0, 30);

        {
            const pipeline = try self.pipelines.getPipeline("reset_bodies");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.accelerations_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 4);
            encoder.setBuffer(&self.state.forces_buffer, 0, 5);
            encoder.setBuffer(&self.state.torques_buffer, 0, 6);
            encoder.setBuffer(&self.state.prev_positions_buffer, 0, 7);
            encoder.setBuffer(&self.state.prev_quaternions_buffer, 0, 8);
            encoder.setBuffer(&self.state.prev_velocities_buffer, 0, 9);
            encoder.setBuffer(&self.initial_positions_buffer, 0, 10);
            encoder.setBuffer(&self.initial_velocities_buffer, 0, 11);
            encoder.setBuffer(&self.initial_quaternions_buffer, 0, 12);
            encoder.setBuffer(&self.initial_angular_velocities_buffer, 0, 13);
            encoder.setBuffer(&self.params_buffer, 0, 14);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_bodies);
        }

        {
            const pipeline = try self.pipelines.getPipeline("reset_env_aux");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.joint_positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.joint_velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 2);
            encoder.setBuffer(&self.state.actions_buffer, 0, 3);
            encoder.setBuffer(&self.state.observations_buffer, 0, 4);
            encoder.setBuffer(&self.state.rewards_buffer, 0, 5);
            encoder.setBuffer(&self.state.dones_buffer, 0, 6);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 7);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 8);
            encoder.setBuffer(&self.prev_contacts_buffer, 0, 9);
            encoder.setBuffer(&self.prev_contact_counts_buffer, 0, 10);
            encoder.setBuffer(&self.constraints_buffer, 0, 11);
            encoder.setBuffer(&self.episode_steps_buffer, 0, 12);
            encoder.setBuffer(&self.initial_joint_positions_buffer, 0, 13);
            encoder.setBuffer(&self.initial_joint_velocities_buffer, 0, 14);
            var reset_params = self.params;
            reset_params.target_color = self.num_constraints_per_env;
            // reset_env_aux does not otherwise use constraint_offset. Reuse it
            // as a private per-dispatch flag without changing the public ABI.
            reset_params.constraint_offset = @intFromBool(preserve_actions);
            encoder.setBytes(std.mem.asBytes(&reset_params), 15);
            encoder.dispatch1D(pipeline, dispatch_envs);
        }

        if (refresh_observations and self.params.num_sensors > 0) {
            encoder.memoryBarrier(.buffers);
            const pipeline = try self.pipelines.getPipeline("read_sensors");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.state.joint_positions_buffer, 0, 4);
            encoder.setBuffer(&self.state.joint_velocities_buffer, 0, 5);
            encoder.setBuffer(&self.sensor_data_buffer, 0, 6);
            encoder.setBuffer(&self.state.observations_buffer, 0, 7);
            encoder.setBuffer(&self.params_buffer, 0, 8);
            encoder.dispatch1D(pipeline, dispatch_envs * self.params.num_sensors);
        }
    }

    /// Reset selected environments from GPU-resident templates and refresh
    /// observations in a single Metal command buffer.
    pub fn reset(self: *World, env_mask: ?[]const u8) !void {
        if (self.pending_step != null) return error.StepPending;
        var dispatch_envs = self.config.num_envs;
        var use_active_ids = false;
        if (env_mask) |mask| {
            if (mask.len != self.config.num_envs) return error.InvalidSize;
            dispatch_envs = self.prepareActiveEnvIds(mask);
            if (dispatch_envs == 0) return;
            use_active_ids = dispatch_envs != self.config.num_envs;
        }

        var cmd = try CommandBuffer.init(self.device.command_queue);
        var encoder = try cmd.computeEncoder();
        try self.encodeReset(&encoder, dispatch_envs, use_active_ids, false, true);

        encoder.endEncoding();
        cmd.commitAndWait();
    }

    /// Configure or disable GPU-side locomotion reward and termination.
    pub fn configureTask(self: *World, params: TaskParams) !void {
        if (self.pending_step != null) return error.StepPending;
        if (params.enabled != 0) {
            if (params.root_body >= self.params.num_bodies or params.forward_axis > 2) {
                return error.InvalidTaskConfig;
            }
            if (!std.math.isFinite(params.forward_reward_weight) or
                !std.math.isFinite(params.control_cost_weight) or
                !std.math.isFinite(params.healthy_bonus) or
                !std.math.isFinite(params.healthy_z_min) or
                !std.math.isFinite(params.healthy_z_max) or
                params.healthy_z_min > params.healthy_z_max)
            {
                return error.InvalidTaskConfig;
            }
        }
        self.task_params = params;
        self.task_params.enabled = @intFromBool(params.enabled != 0);
        self.task_params.terminate_when_unhealthy = @intFromBool(params.terminate_when_unhealthy != 0);
        self.task_params_buffer.getSlice(TaskParams)[0] = self.task_params;
        @memset(self.state.getRewards(), 0);
        @memset(self.state.getDones(), 0);
        @memset(self.episode_steps_buffer.getSlice(u32), 0);
    }

    /// Get observations (zero-copy pointer to GPU buffer).
    pub fn getObservations(self: *World) []f32 {
        return self.state.getObservations();
    }

    /// Get observations pointer for FFI.
    pub fn getObservationsPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        return self.state.getObservationsPtr();
    }

    /// Writable shared-memory action pointer. It must not be accessed while a
    /// step is pending because Metal may be consuming it.
    pub fn getActionsPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const actions = self.state.getActions();
        if (actions.len == 0) return null;
        return actions.ptr;
    }

    /// Get rewards.
    pub fn getRewards(self: *World) []f32 {
        return self.state.getRewards();
    }

    /// Get rewards pointer for FFI.
    pub fn getRewardsPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        return self.state.getRewardsPtr();
    }

    /// Get dones.
    pub fn getDones(self: *World) []u8 {
        return self.state.getDones();
    }

    /// Get dones pointer for FFI.
    pub fn getDonesPtr(self: *World) ?[*]u8 {
        if (self.pending_step != null) return null;
        return self.state.getDonesPtr();
    }

    /// Get body positions for visualization.
    pub fn getBodyPositions(self: *World) [][4]f32 {
        return self.state.getPositions();
    }

    /// Get body quaternions for visualization.
    pub fn getBodyQuaternions(self: *World) [][4]f32 {
        return self.state.getQuaternions();
    }

    /// Get body velocities pointer.
    pub fn getBodyVelocitiesPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const slice = self.state.getVelocities();
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Get body accelerations pointer.
    pub fn getBodyAccelerationsPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const slice = self.state.getAccelerations();
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Get body angular velocities pointer.
    pub fn getBodyAngularVelocitiesPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const slice = self.state.getAngularVelocities();
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Get joint positions pointer.
    pub fn getJointPositionsPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const slice = self.state.getJointPositions();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get joint velocities pointer.
    pub fn getJointVelocitiesPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        const slice = self.state.getJointVelocities();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get contact counts pointer.
    pub fn getContactCountsPtr(self: *World) ?[*]u32 {
        if (self.pending_step != null) return null;
        const slice = self.state.contact_counts_buffer.getSlice(u32);
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get sensor data pointer.
    pub fn getSensorDataPtr(self: *World) ?[*]f32 {
        if (self.pending_step != null) return null;
        // Sensor readings are written into the observations buffer by the
        // read_sensors kernel. Return that data for FFI consumers.
        return self.state.getObservationsPtr();
    }

    /// Get contacts buffer pointer.
    pub fn getContactsPtr(self: *World) ?*anyopaque {
        if (self.pending_step != null) return null;
        const slice = self.state.contacts_buffer.getSlice(u8);
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Set body positions from external buffer.
    pub fn setBodyPositions(self: *World, positions: []const f32, mask: ?[]const u8) !void {
        if (self.pending_step != null) return error.StepPending;
        const dest = self.state.getPositions();
        const expected = self.config.num_envs * self.params.num_bodies * 4;
        if (positions.len < expected) return error.InvalidSize;

        if (mask) |m| {
            for (m, 0..) |should_set, env_id| {
                if (should_set != 0) {
                    for (0..self.params.num_bodies) |b| {
                        const idx = self.state.bodyIndex(@intCast(env_id), @intCast(b));
                        const base = idx * 4;
                        dest[idx] = .{
                            positions[base + 0],
                            positions[base + 1],
                            positions[base + 2],
                            positions[base + 3],
                        };
                    }
                }
            }
        } else {
            for (0..dest.len) |idx| {
                const base = idx * 4;
                dest[idx] = .{
                    positions[base + 0],
                    positions[base + 1],
                    positions[base + 2],
                    positions[base + 3],
                };
            }
        }
    }

    /// Set body velocities from external buffer.
    pub fn setBodyVelocities(self: *World, velocities: []const f32, mask: ?[]const u8) !void {
        if (self.pending_step != null) return error.StepPending;
        const dest = self.state.getVelocities();
        const expected = self.config.num_envs * self.params.num_bodies * 4;
        if (velocities.len < expected) return error.InvalidSize;

        if (mask) |m| {
            for (m, 0..) |should_set, env_id| {
                if (should_set != 0) {
                    for (0..self.params.num_bodies) |b| {
                        const idx = self.state.bodyIndex(@intCast(env_id), @intCast(b));
                        const base = idx * 4;
                        dest[idx] = .{
                            velocities[base + 0],
                            velocities[base + 1],
                            velocities[base + 2],
                            velocities[base + 3],
                        };
                    }
                }
            }
        } else {
            for (0..dest.len) |idx| {
                const base = idx * 4;
                dest[idx] = .{
                    velocities[base + 0],
                    velocities[base + 1],
                    velocities[base + 2],
                    velocities[base + 3],
                };
            }
        }
    }

    /// Get simulation info.
    pub fn getInfo(self: *const World) WorldInfo {
        return .{
            .num_envs = self.config.num_envs,
            .num_bodies = self.params.num_bodies,
            .num_joints = self.params.num_joints,
            .num_actuators = self.params.num_actuators,
            .num_sensors = self.params.num_sensors,
            .num_geoms = self.params.num_geoms,
            .obs_dim = self.state.obs_dim,
            .action_dim = self.params.num_actuators,
            .timestep = self.config.timestep,
            .memory_usage = self.state.memoryUsage(),
        };
    }

    /// Get latest profiling data.
    pub fn getProfilingData(self: *const World) ProfilingData {
        return self.profiling.toPublic();
    }

    /// Reset profiling data.
    pub fn resetProfiling(self: *World) void {
        self.profiling = .{};
    }

    pub fn deinit(self: *World) void {
        if (self.pending_step != null) self.waitStep() catch {};
        // Free color ranges if allocated
        if (self.color_ranges) |ranges| {
            self.allocator.free(ranges);
        }

        self.body_data_buffer.deinit();
        self.joint_data_buffer.deinit();
        self.geom_data_buffer.deinit();
        self.actuator_data_buffer.deinit();
        self.sensor_data_buffer.deinit();
        self.params_buffer.deinit();
        self.constraints_buffer.deinit();

        // Free spatial hash buffers
        if (self.spatial_hash_cell_counts) |*b| b.deinit();
        if (self.spatial_hash_cell_offsets) |*b| b.deinit();
        if (self.spatial_hash_sorted_geoms) |*b| b.deinit();
        if (self.spatial_hash_cell_ids) |*b| b.deinit();
        if (self.spatial_hash_params_buffer) |*b| b.deinit();

        self.prev_contacts_buffer.deinit();
        self.prev_contact_counts_buffer.deinit();
        self.warm_start_factor_buffer.deinit();
        self.active_env_ids_buffer.deinit();
        self.env_dispatch_params_buffer.deinit();
        self.task_params_buffer.deinit();
        self.episode_steps_buffer.deinit();
        self.initial_positions_buffer.deinit();
        self.initial_quaternions_buffer.deinit();
        self.initial_velocities_buffer.deinit();
        self.initial_angular_velocities_buffer.deinit();
        self.initial_joint_positions_buffer.deinit();
        self.initial_joint_velocities_buffer.deinit();

        self.state.deinit();
        self.pipelines.deinit();
        self.scene.deinit();
        self.device.deinit();
    }
};

/// World information struct.
pub const WorldInfo = struct {
    num_envs: u32,
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    num_sensors: u32,
    num_geoms: u32,
    obs_dim: u32,
    action_dim: u32,
    timestep: f32,
    memory_usage: usize,
};

// GPU data structures

const BodyDataGPU = extern struct {
    position: [4]f32 align(16),
    quaternion: [4]f32 align(16),
    inv_mass_inertia: [4]f32 align(16),
    params: [4]f32 align(16), // parent_id, body_type, gravity_scale, damping
    com_offset: [4]f32 align(16), // center of mass offset from body frame origin

    pub fn fromBodyDef(def: *const body_mod.BodyDef) BodyDataGPU {
        const inv_mass = def.invMass();
        const inv_inertia = def.invInertia();

        return .{
            .position = .{ def.position[0], def.position[1], def.position[2], 0 },
            .quaternion = def.quaternion,
            .inv_mass_inertia = .{ inv_mass, inv_inertia[0], inv_inertia[1], inv_inertia[2] },
            .params = .{
                @floatFromInt(def.parent_id),
                @floatFromInt(@intFromEnum(def.body_type)),
                def.gravity_scale,
                def.linear_damping,
            },
            .com_offset = .{ def.com_offset[0], def.com_offset[1], def.com_offset[2], def.angular_damping },
        };
    }
};

const JointDataGPU = extern struct {
    anchor_parent: [4]f32 align(16),
    anchor_child: [4]f32 align(16),
    axis: [4]f32 align(16),
    params: [4]f32 align(16), // type, parent, child, limit_lower
    params2: [4]f32 align(16), // limit_upper, damping, stiffness, ref_pos

    pub fn fromJointDef(def: *const joint_mod.JointDef) JointDataGPU {
        return .{
            .anchor_parent = .{ def.anchor_parent[0], def.anchor_parent[1], def.anchor_parent[2], 0 },
            .anchor_child = .{ def.anchor_child[0], def.anchor_child[1], def.anchor_child[2], 0 },
            .axis = .{ def.axis[0], def.axis[1], def.axis[2], 0 },
            .params = .{
                @floatFromInt(@intFromEnum(def.joint_type)),
                @floatFromInt(def.parent_body),
                @floatFromInt(def.child_body),
                def.limit_lower,
            },
            .params2 = .{ def.limit_upper, def.damping, def.stiffness, def.ref_position },
        };
    }
};

const ActuatorDataGPU = extern struct {
    params: [4]f32 align(16), // joint, ctrl_min, ctrl_max, gear
    params2: [4]f32 align(16), // force_min, force_max, kp, kv

    pub fn fromActuatorDef(def: *const joint_mod.ActuatorDef) ActuatorDataGPU {
        return .{
            .params = .{
                @floatFromInt(def.joint),
                def.ctrl_min,
                def.ctrl_max,
                def.gear,
            },
            .params2 = .{ def.force_min, def.force_max, def.kp, def.kv },
        };
    }
};
