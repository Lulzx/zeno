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
const InitialState = @import("../physics/state.zig").InitialState;
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
    target_color: u32,      // For constraint graph coloring (solve_joints)
    num_constraints: u32,   // Total constraints to process for current color
    constraint_offset: u32, // Offset into constraint buffer for current color
    _padding: u32,          // Alignment padding
};

/// Main simulation world.
pub const World = struct {
    // Metal resources
    device: Device,
    pipelines: PipelineManager,

    // Simulation state
    state: State,
    initial_state: ?InitialState,
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

    // Contact caching buffers for temporal coherence
    prev_contacts_buffer: Buffer,
    prev_contact_counts_buffer: Buffer,

    // Warm start factor buffer (single float passed to GPU)
    warm_start_factor_buffer: Buffer,

    // Adaptive substeps state
    adaptive_substeps: u32,
    max_adaptive_substeps: u32,
    violation_threshold: f32,

    // Allocator
    allocator: std.mem.Allocator,

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
        try pipelines.preload(&.{
            "apply_actions",
            "apply_joint_forces",
            "forward_kinematics",
            "update_kinematic",
            "compute_forces",
            "integrate",
            "broad_phase",
            "narrow_phase",
            "solve_contacts",
            "solve_joints",
            "update_joint_states",
            "read_sensors",
            "reset_env",
            "warm_start_constraints",
            "store_lambda_prev",
            "cache_contacts",
            "match_cached_contacts",
        });

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
        var template_constraints: std.ArrayListUnmanaged(XPBDConstraint) = .{};
        defer template_constraints.deinit(allocator);

        for (scene.joints.items) |joint| {
            // Decompose joint into primitives
            const primitives_list = try joint_mod.decomposeJoint(&joint, allocator);
            defer allocator.free(primitives_list);

            for (primitives_list) |jc| {
                var c: XPBDConstraint = undefined;
                // Default compliance 0 (rigid)
                const compliance: f32 = if (jc.params.compliance > 0) jc.params.compliance else 0.0;
                
                switch (jc.constraint_type) {
                    .point => {
                        c = xpbd_mod.createPositionalConstraint(
                            jc.body_a, jc.body_b, 0, // env 0
                            jc.local_anchor_a, jc.local_anchor_b,
                            compliance
                        );
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
                        // This needs both angular locking and linear DOF
                        // For now, treat as a positional constraint with axis freedom
                        c = xpbd_mod.createPositionalConstraint(
                            jc.body_a,
                            jc.body_b,
                            0,
                            jc.local_anchor_a,
                            jc.local_anchor_b,
                            compliance,
                        );
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
            ._padding = 0,
        };

        var world = World{
            .device = device,
            .pipelines = pipelines,
            .state = state,
            .initial_state = null,
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
            .prev_contacts_buffer = prev_contacts_buffer,
            .prev_contact_counts_buffer = prev_contact_counts_buffer,
            .warm_start_factor_buffer = warm_start_factor_buffer,
            .adaptive_substeps = config.substeps,
            .max_adaptive_substeps = @max(config.substeps * 4, 8),
            .violation_threshold = 0.01,
            .allocator = allocator,
        };

        // Upload scene data to GPU
        try world.uploadSceneData();

        // Initialize state from scene
        try world.initializeState();

        // Capture initial state for reset
        world.initial_state = try InitialState.capture(allocator, &world.state);

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
            geom_data[i] = primitives.GeomGPU.fromGeom(&geom, @intCast(i));
        }

        // Upload actuator data
        if (self.scene.actuators.items.len > 0) {
            const actuator_data = self.actuator_data_buffer.getSlice(ActuatorDataGPU);
            for (self.scene.actuators.items, 0..) |actuator, i| {
                actuator_data[i] = ActuatorDataGPU.fromActuatorDef(&actuator);
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
        // Copy actions to GPU buffer
        try self.state.setActions(actions);

        // Use adaptive substep count if no explicit override
        const actual_substeps = if (substeps == 0) self.adaptive_substeps else substeps;

        for (0..actual_substeps) |_| {
            try self.stepOnce();
        }

        // Adaptive substep detection: check max constraint violation after solving.
        // If violations exceed threshold, increase substeps for next frame.
        // If violations are well below threshold, decrease substeps to save compute.
        if (self.num_constraints_per_env > 0) {
            const constraints_slice = self.constraints_buffer.getSlice(XPBDConstraint);
            var max_violation: f32 = 0.0;

            // Sample a subset of constraints to detect violation level
            // (checking all would be expensive; sample from env 0 only)
            for (0..self.num_constraints_per_env) |ci| {
                const violation = @abs(constraints_slice[ci].state[2]);
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

    /// Execute one physics substep.
    fn stepOnce(self: *World) !void {
        var cmd = try CommandBuffer.init(self.device.command_queue);

        // Stage 0a: Warm start constraints from previous frame's lambda values
        if (self.num_constraints_per_env > 0) {
            const params_ptr = self.params_buffer.getSlice(SimParams);
            params_ptr[0].target_color = self.num_constraints_per_env;

            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("warm_start_constraints");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.constraints_buffer, 0, 0);
            encoder.setBuffer(&self.params_buffer, 0, 1);
            encoder.setBuffer(&self.warm_start_factor_buffer, 0, 2);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.num_constraints_per_env);
        }

        // Stage 1: Apply actions to joint torques
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("apply_actions");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.actions_buffer, 0, 0);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 1);
            encoder.setBuffer(&self.actuator_data_buffer, 0, 2);
            encoder.setBuffer(&self.params_buffer, 0, 3);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_actuators);
        }

        // Stage 1.5: Apply joint forces (convert joint torques to body torques)
        if (self.params.num_joints > 0) {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("apply_joint_forces");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.torques_buffer, 0, 0);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 1);
            encoder.setBuffer(&self.joint_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_joints);
        }

        // Stage 2: Forward kinematics
        // Disabled: We use maximal coordinates + constraints

        // Stage 2.5: Update kinematic bodies
        // Kinematic bodies follow their velocities but aren't affected by forces
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("update_kinematic");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
            encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
            encoder.setBuffer(&self.body_data_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_bodies);
        }

        // Stage 3: Compute forces
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("compute_forces");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
            encoder.setBuffer(&self.state.forces_buffer, 0, 2);
            encoder.setBuffer(&self.state.torques_buffer, 0, 3);
            encoder.setBuffer(&self.state.joint_torques_buffer, 0, 4);
            encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 5);
            encoder.setBuffer(&self.params_buffer, 0, 6);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_bodies);
        }

        // Stage 4: Integration
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

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
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_bodies);
        }

        // Stage 5: Broad phase collision detection
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("broad_phase");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.geom_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 3);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_geoms);
        }

        // Stage 6: Narrow phase collision detection
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("narrow_phase");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.geom_data_buffer, 0, 2);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 3);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 4);
            encoder.setBuffer(&self.params_buffer, 0, 5);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.config.max_contacts_per_env);
        }

        // Stage 6.5: Match new contacts against cached contacts from previous frame
        // Transfer accumulated impulses for temporal coherence
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("match_cached_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 0);
            encoder.setBuffer(&self.prev_contacts_buffer, 0, 1);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 2);
            encoder.setBuffer(&self.prev_contact_counts_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.config.max_contacts_per_env);
        }

        // Stage 7: Joint solver (XPBD iterations with graph coloring)
        // Process constraints by color to avoid race conditions when constraints share bodies
        if (self.num_constraints_per_env > 0 and self.color_ranges != null) {
            const params_ptr = self.params_buffer.getSlice(SimParams);
            const ranges = self.color_ranges.?;

            for (0..self.config.contact_iterations) |_| {
                // Process each color sequentially (constraints within a color don't share bodies)
                for (0..self.num_colors) |color| {
                    const range = ranges[color];
                    const offset = range[0];
                    const count = range[1];

                    if (count == 0) continue;

                    // Update params for this color group
                    // target_color is repurposed to pass constraints_per_env to shader
                    params_ptr[0].constraint_offset = offset;
                    params_ptr[0].num_constraints = count;
                    params_ptr[0].target_color = self.num_constraints_per_env;

                    var encoder = try cmd.computeEncoder();
                    defer encoder.endEncoding();

                    const pipeline = try self.pipelines.getPipeline("solve_joints");
                    encoder.setPipeline(pipeline);
                    encoder.setBuffer(&self.state.positions_buffer, 0, 0);
                    encoder.setBuffer(&self.state.velocities_buffer, 0, 1);
                    encoder.setBuffer(&self.state.quaternions_buffer, 0, 2);
                    encoder.setBuffer(&self.state.angular_velocities_buffer, 0, 3);
                    encoder.setBuffer(&self.constraints_buffer, 0, 4);
                    encoder.setBuffer(&self.state.inv_mass_inertia_buffer, 0, 5);
                    encoder.setBuffer(&self.params_buffer, 0, 6);
                    // Dispatch count * num_envs threads (one per constraint instance)
                    encoder.dispatch1D(pipeline, self.config.num_envs * count);
                }
            }
        }

        // Stage 8: Contact solver (PBD iterations)
        for (0..self.config.contact_iterations) |_| {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

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
            encoder.dispatch1D(pipeline, self.config.num_envs * self.config.max_contacts_per_env);
        }

        // Stage 7.5: Update joint states (Inverse Kinematics for observations)
        if (self.params.num_joints > 0) {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

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
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_joints);
        }

        // Stage 8: Read sensors
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

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
            encoder.dispatch1D(pipeline, self.config.num_envs * self.params.num_sensors);
        }

        // Stage 9: Cache contacts for next frame's temporal coherence
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("cache_contacts");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.contacts_buffer, 0, 0);
            encoder.setBuffer(&self.prev_contacts_buffer, 0, 1);
            encoder.setBuffer(&self.state.contact_counts_buffer, 0, 2);
            encoder.setBuffer(&self.prev_contact_counts_buffer, 0, 3);
            encoder.setBuffer(&self.params_buffer, 0, 4);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.config.max_contacts_per_env);
        }

        // Stage 10: Store lambda_prev for next frame's warm starting
        if (self.num_constraints_per_env > 0) {
            const params_ptr = self.params_buffer.getSlice(SimParams);
            params_ptr[0].target_color = self.num_constraints_per_env;

            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("store_lambda_prev");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.constraints_buffer, 0, 0);
            encoder.setBuffer(&self.params_buffer, 0, 1);
            encoder.dispatch1D(pipeline, self.config.num_envs * self.num_constraints_per_env);
        }

        // Execute and wait
        cmd.commitAndWait();

        // Clear forces for next step
        try self.state.forces_buffer.zero();
        try self.state.torques_buffer.zero();
    }

    /// Reset specific environments.
    pub fn reset(self: *World, env_mask: ?[]const u8) void {
        if (self.initial_state == null) return;

        if (env_mask) |mask| {
            for (mask, 0..) |should_reset, i| {
                if (should_reset != 0) {
                    self.initial_state.?.restore(&self.state, @intCast(i));
                }
            }
        } else {
            // Reset all environments
            for (0..self.config.num_envs) |i| {
                self.initial_state.?.restore(&self.state, @intCast(i));
            }
        }

        // Clear contact counts
        self.state.contact_counts_buffer.zero() catch {};
    }

    /// Get observations (zero-copy pointer to GPU buffer).
    pub fn getObservations(self: *World) []f32 {
        return self.state.getObservations();
    }

    /// Get observations pointer for FFI.
    pub fn getObservationsPtr(self: *World) ?[*]f32 {
        return self.state.getObservationsPtr();
    }

    /// Get rewards.
    pub fn getRewards(self: *World) []f32 {
        return self.state.getRewards();
    }

    /// Get rewards pointer for FFI.
    pub fn getRewardsPtr(self: *World) ?[*]f32 {
        return self.state.getRewardsPtr();
    }

    /// Get dones.
    pub fn getDones(self: *World) []u8 {
        return self.state.getDones();
    }

    /// Get dones pointer for FFI.
    pub fn getDonesPtr(self: *World) ?[*]u8 {
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
        const slice = self.state.getVelocities();
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Get body angular velocities pointer.
    pub fn getBodyAngularVelocitiesPtr(self: *World) ?[*]f32 {
        const slice = self.state.getAngularVelocities();
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Get joint positions pointer.
    pub fn getJointPositionsPtr(self: *World) ?[*]f32 {
        const slice = self.state.getJointPositions();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get joint velocities pointer.
    pub fn getJointVelocitiesPtr(self: *World) ?[*]f32 {
        const slice = self.state.getJointVelocities();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get contact counts pointer.
    pub fn getContactCountsPtr(self: *World) ?[*]u32 {
        const slice = self.state.contact_counts_buffer.getSlice(u32);
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get sensor data pointer.
    pub fn getSensorDataPtr(self: *World) ?[*]f32 {
        const slice = self.sensor_data_buffer.getSlice(f32);
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get contacts buffer pointer.
    pub fn getContactsPtr(self: *World) ?*anyopaque {
        const slice = self.state.contacts_buffer.getSlice(u8);
        if (slice.len == 0) return null;
        return @ptrCast(slice.ptr);
    }

    /// Set body positions from external buffer.
    pub fn setBodyPositions(self: *World, positions: []const f32, mask: ?[]const u8) !void {
        const dest = self.state.getPositions();
        const src: []const [4]f32 = @alignCast(@ptrCast(positions));

        if (mask) |m| {
            for (m, 0..) |should_set, env_id| {
                if (should_set != 0) {
                    for (0..self.params.num_bodies) |b| {
                        const idx = self.state.bodyIndex(@intCast(env_id), @intCast(b));
                        dest[idx] = src[idx];
                    }
                }
            }
        } else {
            @memcpy(dest, src);
        }
    }

    /// Set body velocities from external buffer.
    pub fn setBodyVelocities(self: *World, velocities: []const f32, mask: ?[]const u8) !void {
        const dest = self.state.getVelocities();
        const src: []const [4]f32 = @alignCast(@ptrCast(velocities));

        if (mask) |m| {
            for (m, 0..) |should_set, env_id| {
                if (should_set != 0) {
                    for (0..self.params.num_bodies) |b| {
                        const idx = self.state.bodyIndex(@intCast(env_id), @intCast(b));
                        dest[idx] = src[idx];
                    }
                }
            }
        } else {
            @memcpy(dest, src);
        }
    }

    /// Get simulation info.
    pub fn getInfo(self: *const World) WorldInfo {
        return .{
            .num_envs = self.config.num_envs,
            .num_bodies = self.params.num_bodies,
            .num_joints = self.params.num_joints,
            .num_actuators = self.params.num_actuators,
            .obs_dim = self.state.obs_dim,
            .action_dim = self.params.num_actuators,
            .timestep = self.config.timestep,
            .memory_usage = self.state.memoryUsage(),
        };
    }

    pub fn deinit(self: *World) void {
        if (self.initial_state) |*init_state| {
            init_state.deinit();
        }

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
        self.prev_contacts_buffer.deinit();
        self.prev_contact_counts_buffer.deinit();
        self.warm_start_factor_buffer.deinit();

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
