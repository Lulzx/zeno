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
            "apply_joint_forces",
            "forward_kinematics",
            "compute_forces",
            "integrate",
            "broad_phase",
            "narrow_phase",
            "solve_contacts",
            "solve_joints",
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
                         
                         // Rotate axis from A to World
                         // q * v * q_inv
                         const q_a = body_a.quaternion;
                         const v = jc.axis;
                         
                         // q_a * (v, 0)
                         
                         // ... Wait, quaternion math is verbose inline.
                         // Use approximate logic:
                         // Hinge axis in B is derived from initial pose.
                         // Assuming bodies are initially aligned such that axis_a_world == axis_b_world.
                         // We can compute axis_b_local by transforming axis_a_local through initial relative transform.
                         
                         // Or just use the simple rotation helpers from `primitives` if I could import them.
                         // Let's implement minimal rotate here.
                         
                         // v_world = rotate(v_a, q_a)
                         // v_b = rotate(v_world, conjugate(q_b))
                         
                         // Inline rotate:
                         // t = 2 * cross(q.xyz, v)
                         // v' = v + q.w * t + cross(q.xyz, t)
                         
                         const q_xyz = [3]f32{q_a[0], q_a[1], q_a[2]};
                         const t = [3]f32{
                             2.0 * (q_xyz[1]*v[2] - q_xyz[2]*v[1]),
                             2.0 * (q_xyz[2]*v[0] - q_xyz[0]*v[2]),
                             2.0 * (q_xyz[0]*v[1] - q_xyz[1]*v[0])
                         };
                         const v_world = [3]f32{
                             v[0] + q_a[3]*t[0] + (q_xyz[1]*t[2] - q_xyz[2]*t[1]),
                             v[1] + q_a[3]*t[1] + (q_xyz[2]*t[0] - q_xyz[0]*t[2]),
                             v[2] + q_a[3]*t[2] + (q_xyz[0]*t[1] - q_xyz[1]*t[0])
                         };
                         
                         // Rotate inverse B
                         const q_b = body_b.quaternion;
                         const q_b_inv = [4]f32{-q_b[0], -q_b[1], -q_b[2], q_b[3]};
                         const q_b_xyz = [3]f32{q_b_inv[0], q_b_inv[1], q_b_inv[2]};
                         
                         const t2 = [3]f32{
                             2.0 * (q_b_xyz[1]*v_world[2] - q_b_xyz[2]*v_world[1]),
                             2.0 * (q_b_xyz[2]*v_world[0] - q_b_xyz[0]*v_world[2]),
                             2.0 * (q_b_xyz[0]*v_world[1] - q_b_xyz[1]*v_world[0])
                         };
                         const axis_b = [3]f32{
                             v_world[0] + q_b_inv[3]*t2[0] + (q_b_xyz[1]*t2[2] - q_b_xyz[2]*t2[1]),
                             v_world[1] + q_b_inv[3]*t2[1] + (q_b_xyz[2]*t2[0] - q_b_xyz[0]*t2[2]),
                             v_world[2] + q_b_inv[3]*t2[2] + (q_b_xyz[0]*t2[1] - q_b_xyz[1]*t2[0])
                         };

                         c = xpbd_mod.createAngularConstraint(
                            jc.body_a, jc.body_b, 0,
                            jc.axis, axis_b,
                            compliance, jc.params.damping
                         );
                         try template_constraints.append(allocator, c);
                    },
                    else => {},
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

        // Populate for all envs
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
        // TODO: Bodies marked as 'kinematic' (fixed children) are now static unless
        // converted to dynamic bodies with Fixed/Weld constraints in Scene/Parser.

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

        // Stage 7: Joint solver (XPBD iterations)
        if (self.num_constraints_per_env > 0) {
            for (0..self.config.contact_iterations) |_| {
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
                encoder.dispatch1D(pipeline, self.config.num_envs * self.num_constraints_per_env);
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

        self.body_data_buffer.deinit();
        self.joint_data_buffer.deinit();
        self.geom_data_buffer.deinit();
        self.actuator_data_buffer.deinit();
        self.sensor_data_buffer.deinit();
        self.params_buffer.deinit();
        self.constraints_buffer.deinit();

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
