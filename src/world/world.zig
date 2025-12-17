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

    // Allocator
    allocator: std.mem.Allocator,

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
            "forward_kinematics",
            "compute_forces",
            "integrate",
            "broad_phase",
            "narrow_phase",
            "solve_contacts",
            "read_sensors",
            "reset_env",
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

    /// Step the simulation.
    pub fn step(self: *World, actions: []const f32, substeps: u32) !void {
        // Copy actions to GPU buffer
        try self.state.setActions(actions);

        const actual_substeps = if (substeps == 0) self.config.substeps else substeps;

        for (0..actual_substeps) |_| {
            try self.stepOnce();
        }
    }

    /// Execute one physics substep.
    fn stepOnce(self: *World) !void {
        var cmd = try CommandBuffer.init(self.device.command_queue);

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

        // Stage 2: Forward kinematics
        {
            var encoder = try cmd.computeEncoder();
            defer encoder.endEncoding();

            const pipeline = try self.pipelines.getPipeline("forward_kinematics");
            encoder.setPipeline(pipeline);
            encoder.setBuffer(&self.state.positions_buffer, 0, 0);
            encoder.setBuffer(&self.state.quaternions_buffer, 0, 1);
            encoder.setBuffer(&self.state.joint_positions_buffer, 0, 2);
            encoder.setBuffer(&self.joint_data_buffer, 0, 3);
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

        // Stage 7: Contact solver (PBD iterations)
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

        self.body_data_buffer.deinit();
        self.joint_data_buffer.deinit();
        self.geom_data_buffer.deinit();
        self.actuator_data_buffer.deinit();
        self.sensor_data_buffer.deinit();
        self.params_buffer.deinit();

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
