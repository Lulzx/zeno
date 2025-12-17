//! Optimized state representation for XPBD solver.
//!
//! Memory layout optimized based on M4 Pro bandwidth benchmarks:
//! - All vectors use float4 alignment for coalesced GPU access
//! - SoA layout for cache-efficient iteration over bodies
//! - Unified BodyState struct matches Metal shader layout

const std = @import("std");
const constants = @import("constants.zig");

/// GPU-optimized body state structure.
/// Matches the BodyState struct in xpbd_solver.metal exactly.
/// Total: 80 bytes per body (5 × float4)
pub const BodyState = extern struct {
    /// Position (xyz) + inverse mass (w)
    position: [4]f32 align(16) = .{ 0, 0, 0, 0 },

    /// Orientation quaternion (x, y, z, w)
    quaternion: [4]f32 align(16) = .{ 0, 0, 0, 1 },

    /// Linear velocity (xyz) + unused (w)
    velocity: [4]f32 align(16) = .{ 0, 0, 0, 0 },

    /// Angular velocity (xyz) + unused (w)
    angular_vel: [4]f32 align(16) = .{ 0, 0, 0, 0 },

    /// Inverse inertia tensor diagonal (xyz) + unused (w)
    /// For non-diagonal inertia, would need full 3x3 matrix
    inv_inertia: [4]f32 align(16) = .{ 0, 0, 0, 0 },

    pub fn setPosition(self: *BodyState, pos: [3]f32) void {
        self.position[0] = pos[0];
        self.position[1] = pos[1];
        self.position[2] = pos[2];
    }

    pub fn getPosition(self: *const BodyState) [3]f32 {
        return .{ self.position[0], self.position[1], self.position[2] };
    }

    pub fn setInvMass(self: *BodyState, inv_mass: f32) void {
        self.position[3] = inv_mass;
    }

    pub fn getInvMass(self: *const BodyState) f32 {
        return self.position[3];
    }

    pub fn setQuaternion(self: *BodyState, q: [4]f32) void {
        self.quaternion = q;
    }

    pub fn getQuaternion(self: *const BodyState) [4]f32 {
        return self.quaternion;
    }

    pub fn setVelocity(self: *BodyState, vel: [3]f32) void {
        self.velocity[0] = vel[0];
        self.velocity[1] = vel[1];
        self.velocity[2] = vel[2];
    }

    pub fn getVelocity(self: *const BodyState) [3]f32 {
        return .{ self.velocity[0], self.velocity[1], self.velocity[2] };
    }

    pub fn setAngularVel(self: *BodyState, omega: [3]f32) void {
        self.angular_vel[0] = omega[0];
        self.angular_vel[1] = omega[1];
        self.angular_vel[2] = omega[2];
    }

    pub fn getAngularVel(self: *const BodyState) [3]f32 {
        return .{ self.angular_vel[0], self.angular_vel[1], self.angular_vel[2] };
    }

    pub fn setInvInertia(self: *BodyState, inv_I: [3]f32) void {
        self.inv_inertia[0] = inv_I[0];
        self.inv_inertia[1] = inv_I[1];
        self.inv_inertia[2] = inv_I[2];
    }

    pub fn getInvInertia(self: *const BodyState) [3]f32 {
        return .{ self.inv_inertia[0], self.inv_inertia[1], self.inv_inertia[2] };
    }

    /// Check if body is static (infinite mass).
    pub fn isStatic(self: *const BodyState) bool {
        return self.position[3] < 1e-8;
    }

    /// Create a static body state.
    pub fn staticBody(pos: [3]f32, quat: [4]f32) BodyState {
        return .{
            .position = .{ pos[0], pos[1], pos[2], 0 }, // inv_mass = 0
            .quaternion = quat,
            .velocity = .{ 0, 0, 0, 0 },
            .angular_vel = .{ 0, 0, 0, 0 },
            .inv_inertia = .{ 0, 0, 0, 0 },
        };
    }

    /// Create a dynamic body state.
    pub fn dynamicBody(pos: [3]f32, quat: [4]f32, mass: f32, inertia: [3]f32) BodyState {
        const inv_mass = if (mass > 0) 1.0 / mass else 0;
        const inv_I: [3]f32 = .{
            if (inertia[0] > 0) 1.0 / inertia[0] else 0,
            if (inertia[1] > 0) 1.0 / inertia[1] else 0,
            if (inertia[2] > 0) 1.0 / inertia[2] else 0,
        };
        return .{
            .position = .{ pos[0], pos[1], pos[2], inv_mass },
            .quaternion = quat,
            .velocity = .{ 0, 0, 0, 0 },
            .angular_vel = .{ 0, 0, 0, 0 },
            .inv_inertia = .{ inv_I[0], inv_I[1], inv_I[2], 0 },
        };
    }
};

/// Solver parameters for GPU dispatch.
/// Matches SolverParams struct in xpbd_solver.metal.
pub const SolverParams = extern struct {
    num_envs: u32 align(16) = 0,
    max_constraints: u32 = 0,
    num_bodies: u32 = 0,
    iteration: u32 = 0,

    dt: f32 align(16) = 0.002,
    inv_dt: f32 = 500.0,
    inv_dt_sq: f32 = 250000.0,
    relaxation: f32 = 1.0,

    gravity: [4]f32 align(16) = .{ 0, 0, -9.81, 0 },

    pub fn init(
        num_envs: u32,
        num_bodies: u32,
        max_constraints: u32,
        dt: f32,
        gravity: [3]f32,
    ) SolverParams {
        return .{
            .num_envs = num_envs,
            .max_constraints = max_constraints,
            .num_bodies = num_bodies,
            .iteration = 0,
            .dt = dt,
            .inv_dt = 1.0 / dt,
            .inv_dt_sq = 1.0 / (dt * dt),
            .relaxation = 1.0,
            .gravity = .{ gravity[0], gravity[1], gravity[2], 0 },
        };
    }
};

/// Batched state buffer for all environments.
pub const BatchedState = struct {
    /// Body states for all environments (num_envs × num_bodies)
    bodies: []BodyState,

    /// Previous positions for velocity update (num_envs × num_bodies)
    prev_positions: [][4]f32,

    /// Previous quaternions for angular velocity update
    prev_quaternions: [][4]f32,

    /// Joint positions (num_envs × num_joints)
    joint_positions: []f32,

    /// Joint velocities
    joint_velocities: []f32,

    /// Joint torques (from actuators)
    joint_torques: []f32,

    /// Observations (num_envs × obs_dim)
    observations: []f32,

    /// Rewards (num_envs)
    rewards: []f32,

    /// Done flags (num_envs)
    dones: []u8,

    /// Configuration
    num_envs: u32,
    num_bodies: u32,
    num_joints: u32,
    obs_dim: u32,

    /// Allocator
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        num_envs: u32,
        num_bodies: u32,
        num_joints: u32,
        obs_dim: u32,
    ) !BatchedState {
        const total_bodies = num_envs * num_bodies;
        const total_joints = num_envs * num_joints;

        return .{
            .bodies = try allocator.alloc(BodyState, total_bodies),
            .prev_positions = try allocator.alloc([4]f32, total_bodies),
            .prev_quaternions = try allocator.alloc([4]f32, total_bodies),
            .joint_positions = try allocator.alloc(f32, total_joints),
            .joint_velocities = try allocator.alloc(f32, total_joints),
            .joint_torques = try allocator.alloc(f32, total_joints),
            .observations = try allocator.alloc(f32, num_envs * obs_dim),
            .rewards = try allocator.alloc(f32, num_envs),
            .dones = try allocator.alloc(u8, num_envs),
            .num_envs = num_envs,
            .num_bodies = num_bodies,
            .num_joints = num_joints,
            .obs_dim = obs_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchedState) void {
        self.allocator.free(self.bodies);
        self.allocator.free(self.prev_positions);
        self.allocator.free(self.prev_quaternions);
        self.allocator.free(self.joint_positions);
        self.allocator.free(self.joint_velocities);
        self.allocator.free(self.joint_torques);
        self.allocator.free(self.observations);
        self.allocator.free(self.rewards);
        self.allocator.free(self.dones);
    }

    /// Get body state for a specific environment and body.
    pub fn getBody(self: *const BatchedState, env_id: u32, body_id: u32) *const BodyState {
        return &self.bodies[env_id * self.num_bodies + body_id];
    }

    /// Get mutable body state.
    pub fn getBodyMut(self: *BatchedState, env_id: u32, body_id: u32) *BodyState {
        return &self.bodies[env_id * self.num_bodies + body_id];
    }

    /// Get joint position for environment.
    pub fn getJointPosition(self: *const BatchedState, env_id: u32, joint_id: u32) f32 {
        return self.joint_positions[env_id * self.num_joints + joint_id];
    }

    /// Set joint torque.
    pub fn setJointTorque(self: *BatchedState, env_id: u32, joint_id: u32, torque: f32) void {
        self.joint_torques[env_id * self.num_joints + joint_id] = torque;
    }

    /// Store current positions for velocity computation.
    pub fn storePreviousState(self: *BatchedState) void {
        for (self.bodies, 0..) |body, i| {
            self.prev_positions[i] = body.position;
            self.prev_quaternions[i] = body.quaternion;
        }
    }

    /// Memory size in bytes.
    pub fn memorySize(self: *const BatchedState) usize {
        const bodies_size = self.bodies.len * @sizeOf(BodyState);
        const prev_size = (self.prev_positions.len + self.prev_quaternions.len) * @sizeOf([4]f32);
        const joints_size = (self.joint_positions.len + self.joint_velocities.len + self.joint_torques.len) * @sizeOf(f32);
        const obs_size = self.observations.len * @sizeOf(f32);
        const misc_size = self.rewards.len * @sizeOf(f32) + self.dones.len;
        return bodies_size + prev_size + joints_size + obs_size + misc_size;
    }

    /// Reset a single environment to initial state.
    pub fn resetEnv(self: *BatchedState, env_id: u32, initial_bodies: []const BodyState) void {
        const offset = env_id * self.num_bodies;
        for (initial_bodies, 0..) |body, i| {
            self.bodies[offset + i] = body;
        }

        // Clear joint state
        const joint_offset = env_id * self.num_joints;
        for (0..self.num_joints) |i| {
            self.joint_positions[joint_offset + i] = 0;
            self.joint_velocities[joint_offset + i] = 0;
            self.joint_torques[joint_offset + i] = 0;
        }

        // Clear done flag and reward
        self.dones[env_id] = 0;
        self.rewards[env_id] = 0;
    }
};

// Compile-time size verification
comptime {
    // Ensure BodyState is exactly 80 bytes for GPU alignment
    if (@sizeOf(BodyState) != 80) {
        @compileError("BodyState must be 80 bytes");
    }

    // Ensure SolverParams matches expected size
    if (@sizeOf(SolverParams) != 48) {
        @compileError("SolverParams must be 48 bytes");
    }
}
