//! Batched simulation state in Structure of Arrays (SoA) layout.
//! Optimized for GPU coalesced memory access.

const std = @import("std");
const objc = @import("../objc.zig");
const Buffer = @import("../metal/buffer.zig").Buffer;
const BufferOptions = @import("../metal/buffer.zig").BufferOptions;
const constants = @import("constants.zig");
const body = @import("body.zig");
const contact = @import("contact.zig");

/// Batched simulation state stored in Metal buffers.
pub const State = struct {
    // Buffer handles
    positions_buffer: Buffer,
    velocities_buffer: Buffer,
    angular_velocities_buffer: Buffer,
    quaternions_buffer: Buffer,
    forces_buffer: Buffer,
    torques_buffer: Buffer,
    inv_mass_inertia_buffer: Buffer,

    joint_positions_buffer: Buffer,
    joint_velocities_buffer: Buffer,
    joint_torques_buffer: Buffer,

    actions_buffer: Buffer,
    observations_buffer: Buffer,
    rewards_buffer: Buffer,
    dones_buffer: Buffer,

    contacts_buffer: Buffer,
    contact_counts_buffer: Buffer,

    rng_state_buffer: Buffer,

    // Dimensions
    num_envs: u32,
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    obs_dim: u32,
    max_contacts: u32,

    // Device reference
    device: objc.id,
    allocator: std.mem.Allocator,

    /// Initialize state buffers for the given dimensions.
    pub fn init(
        allocator: std.mem.Allocator,
        device: objc.id,
        num_envs: u32,
        num_bodies: u32,
        num_joints: u32,
        num_actuators: u32,
        obs_dim: u32,
        max_contacts: u32,
    ) !State {
        const opts = BufferOptions{ .storage_mode = .shared };

        const body_count = num_envs * num_bodies;
        const joint_count = num_envs * num_joints;
        const actuator_count = num_envs * num_actuators;
        const obs_count = num_envs * obs_dim;
        const contact_count = num_envs * max_contacts;

        return State{
            // Body state (float4 aligned)
            .positions_buffer = try Buffer.init(device, body_count * 16, opts),
            .velocities_buffer = try Buffer.init(device, body_count * 16, opts),
            .angular_velocities_buffer = try Buffer.init(device, body_count * 16, opts),
            .quaternions_buffer = try Buffer.init(device, body_count * 16, opts),
            .forces_buffer = try Buffer.init(device, body_count * 16, opts),
            .torques_buffer = try Buffer.init(device, body_count * 16, opts),
            .inv_mass_inertia_buffer = try Buffer.init(device, body_count * 16, opts),

            // Joint state
            .joint_positions_buffer = try Buffer.init(device, joint_count * 4, opts),
            .joint_velocities_buffer = try Buffer.init(device, joint_count * 4, opts),
            .joint_torques_buffer = try Buffer.init(device, joint_count * 4, opts),

            // Control/observation
            .actions_buffer = try Buffer.init(device, actuator_count * 4, opts),
            .observations_buffer = try Buffer.init(device, obs_count * 4, opts),
            .rewards_buffer = try Buffer.init(device, num_envs * 4, opts),
            .dones_buffer = try Buffer.init(device, num_envs, opts),

            // Contacts
            .contacts_buffer = try Buffer.init(device, contact_count * @sizeOf(contact.ContactGPU), opts),
            .contact_counts_buffer = try Buffer.init(device, num_envs * 4, opts),

            // RNG state (xoroshiro128+)
            .rng_state_buffer = try Buffer.init(device, num_envs * 16, opts),

            .num_envs = num_envs,
            .num_bodies = num_bodies,
            .num_joints = num_joints,
            .num_actuators = num_actuators,
            .obs_dim = obs_dim,
            .max_contacts = max_contacts,
            .device = device,
            .allocator = allocator,
        };
    }

    /// Get positions as a slice (zero-copy).
    pub fn getPositions(self: *State) [][4]f32 {
        return self.positions_buffer.getAlignedSlice([4]f32, 16);
    }

    /// Get quaternions as a slice (zero-copy).
    pub fn getQuaternions(self: *State) [][4]f32 {
        return self.quaternions_buffer.getAlignedSlice([4]f32, 16);
    }

    /// Get velocities as a slice (zero-copy).
    pub fn getVelocities(self: *State) [][4]f32 {
        return self.velocities_buffer.getAlignedSlice([4]f32, 16);
    }

    /// Get angular velocities as a slice (zero-copy).
    pub fn getAngularVelocities(self: *State) [][4]f32 {
        return self.angular_velocities_buffer.getAlignedSlice([4]f32, 16);
    }

    /// Get joint positions as a slice.
    pub fn getJointPositions(self: *State) []f32 {
        return self.joint_positions_buffer.getSlice(f32);
    }

    /// Get joint velocities as a slice.
    pub fn getJointVelocities(self: *State) []f32 {
        return self.joint_velocities_buffer.getSlice(f32);
    }

    /// Get actions buffer as a slice.
    pub fn getActions(self: *State) []f32 {
        return self.actions_buffer.getSlice(f32);
    }

    /// Get observations buffer as a slice (zero-copy for Python).
    pub fn getObservations(self: *State) []f32 {
        return self.observations_buffer.getSlice(f32);
    }

    /// Get rewards buffer.
    pub fn getRewards(self: *State) []f32 {
        return self.rewards_buffer.getSlice(f32);
    }

    /// Get dones buffer.
    pub fn getDones(self: *State) []u8 {
        return self.dones_buffer.getSlice(u8);
    }

    /// Get raw pointer to observations (for Python FFI).
    pub fn getObservationsPtr(self: *State) ?[*]f32 {
        const slice = self.getObservations();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get raw pointer to rewards.
    pub fn getRewardsPtr(self: *State) ?[*]f32 {
        const slice = self.getRewards();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Get raw pointer to dones.
    pub fn getDonesPtr(self: *State) ?[*]u8 {
        const slice = self.getDones();
        if (slice.len == 0) return null;
        return slice.ptr;
    }

    /// Set actions from external buffer.
    pub fn setActions(self: *State, actions: []const f32) !void {
        const dest = self.getActions();
        if (actions.len > dest.len) return error.InvalidSize;
        @memcpy(dest[0..actions.len], actions);
    }

    /// Get body index for env_id and body_id.
    pub fn bodyIndex(self: *const State, env_id: u32, body_id: u32) u32 {
        return env_id * self.num_bodies + body_id;
    }

    /// Get joint index for env_id and joint_id.
    pub fn jointIndex(self: *const State, env_id: u32, joint_id: u32) u32 {
        return env_id * self.num_joints + joint_id;
    }

    /// Get observation index.
    pub fn obsIndex(self: *const State, env_id: u32, obs_id: u32) u32 {
        return env_id * self.obs_dim + obs_id;
    }

    /// Initialize RNG state with seed.
    pub fn initRng(self: *State, seed: u64) void {
        const rng_state = self.rng_state_buffer.getSlice([4]u32);

        for (0..self.num_envs) |i| {
            // Use splitmix64 to generate initial state
            var s = seed +% @as(u64, @intCast(i)) *% 0x9E3779B97F4A7C15;

            s = (s ^ (s >> 30)) *% 0xBF58476D1CE4E5B9;
            s = (s ^ (s >> 27)) *% 0x94D049BB133111EB;
            s = s ^ (s >> 31);

            rng_state[i][0] = @truncate(s);
            rng_state[i][1] = @truncate(s >> 32);

            s = seed +% @as(u64, @intCast(i)) *% 0x9E3779B97F4A7C15 +% 1;
            s = (s ^ (s >> 30)) *% 0xBF58476D1CE4E5B9;
            s = (s ^ (s >> 27)) *% 0x94D049BB133111EB;
            s = s ^ (s >> 31);

            rng_state[i][2] = @truncate(s);
            rng_state[i][3] = @truncate(s >> 32);
        }
    }

    /// Zero all state buffers.
    pub fn zero(self: *State) !void {
        try self.positions_buffer.zero();
        try self.velocities_buffer.zero();
        try self.angular_velocities_buffer.zero();
        try self.quaternions_buffer.zero();
        try self.forces_buffer.zero();
        try self.torques_buffer.zero();

        try self.joint_positions_buffer.zero();
        try self.joint_velocities_buffer.zero();
        try self.joint_torques_buffer.zero();

        try self.actions_buffer.zero();
        try self.observations_buffer.zero();
        try self.rewards_buffer.zero();
        try self.dones_buffer.zero();

        try self.contacts_buffer.zero();
        try self.contact_counts_buffer.zero();
    }

    /// Copy state from one environment to another.
    pub fn copyEnv(self: *State, src_env: u32, dst_env: u32) void {
        const positions = self.getPositions();
        const quaternions = self.getQuaternions();
        const velocities = self.getVelocities();
        const angular_vels = self.getAngularVelocities();
        const joint_pos = self.getJointPositions();
        const joint_vel = self.getJointVelocities();

        for (0..self.num_bodies) |b| {
            const src_idx = self.bodyIndex(src_env, @intCast(b));
            const dst_idx = self.bodyIndex(dst_env, @intCast(b));
            positions[dst_idx] = positions[src_idx];
            quaternions[dst_idx] = quaternions[src_idx];
            velocities[dst_idx] = velocities[src_idx];
            angular_vels[dst_idx] = angular_vels[src_idx];
        }

        for (0..self.num_joints) |j| {
            const src_idx = self.jointIndex(src_env, @intCast(j));
            const dst_idx = self.jointIndex(dst_env, @intCast(j));
            joint_pos[dst_idx] = joint_pos[src_idx];
            joint_vel[dst_idx] = joint_vel[src_idx];
        }
    }

    /// Get total memory usage in bytes.
    pub fn memoryUsage(self: *const State) usize {
        return self.positions_buffer.size +
            self.velocities_buffer.size +
            self.angular_velocities_buffer.size +
            self.quaternions_buffer.size +
            self.forces_buffer.size +
            self.torques_buffer.size +
            self.inv_mass_inertia_buffer.size +
            self.joint_positions_buffer.size +
            self.joint_velocities_buffer.size +
            self.joint_torques_buffer.size +
            self.actions_buffer.size +
            self.observations_buffer.size +
            self.rewards_buffer.size +
            self.dones_buffer.size +
            self.contacts_buffer.size +
            self.contact_counts_buffer.size +
            self.rng_state_buffer.size;
    }

    /// Release all buffers.
    pub fn deinit(self: *State) void {
        self.positions_buffer.deinit();
        self.velocities_buffer.deinit();
        self.angular_velocities_buffer.deinit();
        self.quaternions_buffer.deinit();
        self.forces_buffer.deinit();
        self.torques_buffer.deinit();
        self.inv_mass_inertia_buffer.deinit();

        self.joint_positions_buffer.deinit();
        self.joint_velocities_buffer.deinit();
        self.joint_torques_buffer.deinit();

        self.actions_buffer.deinit();
        self.observations_buffer.deinit();
        self.rewards_buffer.deinit();
        self.dones_buffer.deinit();

        self.contacts_buffer.deinit();
        self.contact_counts_buffer.deinit();

        self.rng_state_buffer.deinit();
    }
};

/// Initial state snapshot for reset.
pub const InitialState = struct {
    positions: [][4]f32,
    quaternions: [][4]f32,
    velocities: [][4]f32,
    angular_velocities: [][4]f32,
    joint_positions: []f32,
    joint_velocities: []f32,
    allocator: std.mem.Allocator,

    pub fn capture(allocator: std.mem.Allocator, state: *State) !InitialState {
        // Only capture for env 0 (template)
        const num_bodies = state.num_bodies;
        const num_joints = state.num_joints;

        const init_state = InitialState{
            .positions = try allocator.alloc([4]f32, num_bodies),
            .quaternions = try allocator.alloc([4]f32, num_bodies),
            .velocities = try allocator.alloc([4]f32, num_bodies),
            .angular_velocities = try allocator.alloc([4]f32, num_bodies),
            .joint_positions = try allocator.alloc(f32, num_joints),
            .joint_velocities = try allocator.alloc(f32, num_joints),
            .allocator = allocator,
        };

        const positions = state.getPositions();
        const quaternions = state.getQuaternions();
        const velocities = state.getVelocities();
        const angular_vels = state.getAngularVelocities();
        const joint_pos = state.getJointPositions();
        const joint_vel = state.getJointVelocities();

        @memcpy(init_state.positions, positions[0..num_bodies]);
        @memcpy(init_state.quaternions, quaternions[0..num_bodies]);
        @memcpy(init_state.velocities, velocities[0..num_bodies]);
        @memcpy(init_state.angular_velocities, angular_vels[0..num_bodies]);
        @memcpy(init_state.joint_positions, joint_pos[0..num_joints]);
        @memcpy(init_state.joint_velocities, joint_vel[0..num_joints]);

        return init_state;
    }

    pub fn restore(self: *const InitialState, state: *State, env_id: u32) void {
        const positions = state.getPositions();
        const quaternions = state.getQuaternions();
        const velocities = state.getVelocities();
        const angular_vels = state.getAngularVelocities();
        const joint_pos = state.getJointPositions();
        const joint_vel = state.getJointVelocities();

        for (0..state.num_bodies) |b| {
            const idx = state.bodyIndex(env_id, @intCast(b));
            positions[idx] = self.positions[b];
            quaternions[idx] = self.quaternions[b];
            velocities[idx] = self.velocities[b];
            angular_vels[idx] = self.angular_velocities[b];
        }

        for (0..state.num_joints) |j| {
            const idx = state.jointIndex(env_id, @intCast(j));
            joint_pos[idx] = self.joint_positions[j];
            joint_vel[idx] = self.joint_velocities[j];
        }
    }

    pub fn deinit(self: *InitialState) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.quaternions);
        self.allocator.free(self.velocities);
        self.allocator.free(self.angular_velocities);
        self.allocator.free(self.joint_positions);
        self.allocator.free(self.joint_velocities);
    }
};
