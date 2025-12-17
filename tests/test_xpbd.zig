//! Tests for XPBD constraint solver.

const std = @import("std");
const testing = std.testing;

// Local definitions for testing (mirrors src/physics/ modules)
const xpbd = struct {
    pub const ConstraintType = enum(u8) {
        contact_normal = 0,
        contact_friction = 1,
        positional = 2,
        angular = 3,
        angular_limit = 4,
        linear_limit = 5,
    };

    pub const XPBDConstraint = extern struct {
        indices: [4]u32 align(16),
        anchor_a: [4]f32 align(16),
        anchor_b: [4]f32 align(16),
        axis_target: [4]f32 align(16),
        limits: [4]f32 align(16),
        state: [4]f32 align(16),

        pub fn getBodyA(self: *const XPBDConstraint) u32 {
            return self.indices[0];
        }
        pub fn getBodyB(self: *const XPBDConstraint) u32 {
            return self.indices[1];
        }
        pub fn getEnvId(self: *const XPBDConstraint) u32 {
            return self.indices[2];
        }
        pub fn getType(self: *const XPBDConstraint) ConstraintType {
            return @enumFromInt(self.indices[3]);
        }
        pub fn getCompliance(self: *const XPBDConstraint) f32 {
            return self.anchor_a[3];
        }
        pub fn getLambda(self: *const XPBDConstraint) f32 {
            return self.state[0];
        }
    };

    pub const XPBDConfig = struct {
        iterations: u32 = 4,
        contact_compliance: f32 = 0.0,
        joint_compliance: f32 = 0.0,
        warm_start: bool = true,
        relaxation: f32 = 1.0,
        velocity_damping: f32 = 0.0,
        substeps: u32 = 1,

        pub fn forRL() XPBDConfig {
            return .{
                .iterations = 4,
                .contact_compliance = 1e-9,
                .warm_start = true,
            };
        }

        pub fn forAccuracy() XPBDConfig {
            return .{
                .iterations = 8,
                .substeps = 4,
                .warm_start = true,
            };
        }
    };

    pub fn createContactConstraint(
        body_a: u32,
        body_b: u32,
        env_id: u32,
        position: [3]f32,
        normal: [3]f32,
        _: f32,
        _: f32,
        _: f32,
        compliance: f32,
    ) XPBDConstraint {
        return .{
            .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.contact_normal) },
            .anchor_a = .{ position[0], position[1], position[2], compliance },
            .anchor_b = .{ 0, 0, 0, 0 },
            .axis_target = .{ normal[0], normal[1], normal[2], 0 },
            .limits = .{ 0, 0, 0, 0 },
            .state = .{ 0, 0, 0, 0 },
        };
    }

    pub fn createPositionalConstraint(
        body_a: u32,
        body_b: u32,
        env_id: u32,
        local_a: [3]f32,
        local_b: [3]f32,
        compliance: f32,
    ) XPBDConstraint {
        return .{
            .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.positional) },
            .anchor_a = .{ local_a[0], local_a[1], local_a[2], compliance },
            .anchor_b = .{ local_b[0], local_b[1], local_b[2], 0 },
            .axis_target = .{ 0, 0, 0, 0 },
            .limits = .{ 0, 0, 0, 0 },
            .state = .{ 0, 0, 0, 0 },
        };
    }

    pub fn computeEffectiveMass(
        inv_mass_a: f32,
        inv_mass_b: f32,
        inv_inertia_a: [3]f32,
        inv_inertia_b: [3]f32,
        r_a: [3]f32,
        r_b: [3]f32,
        normal: [3]f32,
    ) f32 {
        const rn_a = cross(r_a, normal);
        const rn_b = cross(r_b, normal);
        const angular_a = inv_inertia_a[0] * rn_a[0] * rn_a[0] +
            inv_inertia_a[1] * rn_a[1] * rn_a[1] +
            inv_inertia_a[2] * rn_a[2] * rn_a[2];
        const angular_b = inv_inertia_b[0] * rn_b[0] * rn_b[0] +
            inv_inertia_b[1] * rn_b[1] * rn_b[1] +
            inv_inertia_b[2] * rn_b[2] * rn_b[2];
        return inv_mass_a + inv_mass_b + angular_a + angular_b;
    }

    fn cross(a: [3]f32, b: [3]f32) [3]f32 {
        return .{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    }
};

const state_v2 = struct {
    pub const BodyState = extern struct {
        position: [4]f32 align(16) = .{ 0, 0, 0, 0 },
        quaternion: [4]f32 align(16) = .{ 0, 0, 0, 1 },
        velocity: [4]f32 align(16) = .{ 0, 0, 0, 0 },
        angular_vel: [4]f32 align(16) = .{ 0, 0, 0, 0 },
        inv_inertia: [4]f32 align(16) = .{ 0, 0, 0, 0 },

        pub fn getInvMass(self: *const BodyState) f32 {
            return self.position[3];
        }
        pub fn isStatic(self: *const BodyState) bool {
            return self.position[3] < 1e-8;
        }
        pub fn staticBody(pos: [3]f32, quat: [4]f32) BodyState {
            return .{
                .position = .{ pos[0], pos[1], pos[2], 0 },
                .quaternion = quat,
            };
        }
        pub fn dynamicBody(pos: [3]f32, quat: [4]f32, mass: f32, inertia: [3]f32) BodyState {
            const inv_mass = if (mass > 0) 1.0 / mass else 0;
            _ = inertia;
            return .{
                .position = .{ pos[0], pos[1], pos[2], inv_mass },
                .quaternion = quat,
            };
        }
    };

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
                .dt = dt,
                .inv_dt = 1.0 / dt,
                .inv_dt_sq = 1.0 / (dt * dt),
                .gravity = .{ gravity[0], gravity[1], gravity[2], 0 },
            };
        }
    };

    pub const BatchedState = struct {
        bodies: []BodyState,
        num_envs: u32,
        num_bodies: u32,
        num_joints: u32,
        obs_dim: u32,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, num_envs: u32, num_bodies: u32, num_joints: u32, obs_dim: u32) !BatchedState {
            return .{
                .bodies = try allocator.alloc(BodyState, num_envs * num_bodies),
                .num_envs = num_envs,
                .num_bodies = num_bodies,
                .num_joints = num_joints,
                .obs_dim = obs_dim,
                .allocator = allocator,
            };
        }
        pub fn deinit(self: *BatchedState) void {
            self.allocator.free(self.bodies);
        }
    };
};

const contact_buffer = struct {
    pub const CompactContact = extern struct {
        position_penetration: [4]f32 align(16),
        normal_friction: [4]f32 align(16),
        indices: [4]u32 align(16),
        solver_state: [4]f32 align(16),

        pub fn init(
            position: [3]f32,
            normal: [3]f32,
            penetration: f32,
            body_a: u32,
            body_b: u32,
            env_id: u32,
            friction: f32,
            restitution: f32,
        ) CompactContact {
            return .{
                .position_penetration = .{ position[0], position[1], position[2], penetration },
                .normal_friction = .{ normal[0], normal[1], normal[2], friction },
                .indices = .{ body_a, body_b, env_id, 1 },
                .solver_state = .{ 0, 0, 0, restitution },
            };
        }

        pub fn getPenetration(self: *const CompactContact) f32 {
            return self.position_penetration[3];
        }
        pub fn isActive(self: *const CompactContact) bool {
            return (self.indices[3] & 1) != 0;
        }
        pub fn setInactive(self: *CompactContact) void {
            self.indices[3] &= ~@as(u32, 1);
        }

        pub fn toXPBDConstraint(self: *const CompactContact, compliance: f32) xpbd.XPBDConstraint {
            return .{
                .indices = .{ self.indices[0], self.indices[1], self.indices[2], @intFromEnum(xpbd.ConstraintType.contact_normal) },
                .anchor_a = .{ self.position_penetration[0], self.position_penetration[1], self.position_penetration[2], compliance },
                .anchor_b = .{ 0, 0, 0, 0 },
                .axis_target = .{ self.normal_friction[0], self.normal_friction[1], self.normal_friction[2], 0 },
                .limits = .{ 0, std.math.inf(f32), self.normal_friction[3], self.solver_state[3] },
                .state = .{ self.solver_state[0], 0, self.position_penetration[3], 0 },
            };
        }
    };

    pub const ContactBufferManager = struct {
        contacts: []CompactContact,
        counts: []u32,
        offsets: []u32,
        num_envs: u32,
        max_contacts_per_env: u32,
        total_contacts: u32,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, num_envs: u32, max_contacts_per_env: u32) !ContactBufferManager {
            var self = ContactBufferManager{
                .contacts = try allocator.alloc(CompactContact, num_envs * max_contacts_per_env),
                .counts = try allocator.alloc(u32, num_envs),
                .offsets = try allocator.alloc(u32, num_envs),
                .num_envs = num_envs,
                .max_contacts_per_env = max_contacts_per_env,
                .total_contacts = 0,
                .allocator = allocator,
            };
            @memset(self.counts, 0);
            for (0..num_envs) |i| {
                self.offsets[i] = @intCast(i * max_contacts_per_env);
            }
            return self;
        }

        pub fn deinit(self: *ContactBufferManager) void {
            self.allocator.free(self.contacts);
            self.allocator.free(self.counts);
            self.allocator.free(self.offsets);
        }

        pub fn addContact(self: *ContactBufferManager, env_id: u32, contact: CompactContact) bool {
            const count = self.counts[env_id];
            if (count >= self.max_contacts_per_env) return false;
            self.contacts[self.offsets[env_id] + count] = contact;
            self.counts[env_id] = count + 1;
            self.total_contacts += 1;
            return true;
        }

        pub fn getContactMut(self: *ContactBufferManager, env_id: u32, contact_idx: u32) ?*CompactContact {
            if (contact_idx >= self.counts[env_id]) return null;
            return &self.contacts[self.offsets[env_id] + contact_idx];
        }

        pub fn iterEnvContacts(self: *const ContactBufferManager, env_id: u32) []const CompactContact {
            const offset = self.offsets[env_id];
            const count = self.counts[env_id];
            return self.contacts[offset..][0..count];
        }

        pub fn compactInPlace(self: *ContactBufferManager) void {
            for (0..self.num_envs) |env_id| {
                self.compactEnv(@intCast(env_id));
            }
        }

        fn compactEnv(self: *ContactBufferManager, env_id: u32) void {
            const offset = self.offsets[env_id];
            const count = self.counts[env_id];
            var write_idx: u32 = 0;
            var read_idx: u32 = 0;
            while (read_idx < count) {
                if (self.contacts[offset + read_idx].isActive()) {
                    if (write_idx != read_idx) {
                        self.contacts[offset + write_idx] = self.contacts[offset + read_idx];
                    }
                    write_idx += 1;
                } else {
                    self.total_contacts -= 1;
                }
                read_idx += 1;
            }
            self.counts[env_id] = write_idx;
        }
    };

    pub const ContactBufferLayout = struct {
        contacts_offset: u32,
        counts_offset: u32,
        contacts_size: u32,
        counts_size: u32,
        total_size: u32,

        pub fn compute(num_envs: u32, max_contacts_per_env: u32) ContactBufferLayout {
            const contacts_size = num_envs * max_contacts_per_env * @sizeOf(CompactContact);
            const counts_size = num_envs * @sizeOf(u32);
            const aligned_contacts = (contacts_size + 15) & ~@as(u32, 15);
            return .{
                .contacts_offset = 0,
                .counts_offset = aligned_contacts,
                .contacts_size = contacts_size,
                .counts_size = counts_size,
                .total_size = aligned_contacts + counts_size,
            };
        }
    };
};

test "XPBDConstraint size is 96 bytes" {
    try testing.expectEqual(@as(usize, 96), @sizeOf(xpbd.XPBDConstraint));
}

test "CompactContact size is 64 bytes" {
    try testing.expectEqual(@as(usize, 64), @sizeOf(contact_buffer.CompactContact));
}

test "BodyState size is 80 bytes" {
    try testing.expectEqual(@as(usize, 80), @sizeOf(state_v2.BodyState));
}

test "SolverParams size is 48 bytes" {
    try testing.expectEqual(@as(usize, 48), @sizeOf(state_v2.SolverParams));
}

test "create contact constraint" {
    const constraint = xpbd.createContactConstraint(
        0, // body_a
        1, // body_b
        0, // env_id
        .{ 0, 0, 0.5 }, // position
        .{ 0, 0, 1 }, // normal (up)
        0.1, // penetration
        0.5, // friction
        0.0, // restitution
        0.0, // compliance (rigid)
    );

    try testing.expectEqual(@as(u32, 0), constraint.getBodyA());
    try testing.expectEqual(@as(u32, 1), constraint.getBodyB());
    try testing.expectEqual(@as(u32, 0), constraint.getEnvId());
    try testing.expectEqual(xpbd.ConstraintType.contact_normal, constraint.getType());
    try testing.expectApproxEqAbs(@as(f32, 0.0), constraint.getCompliance(), 0.001);
}

test "create positional constraint" {
    const constraint = xpbd.createPositionalConstraint(
        0, // body_a
        1, // body_b
        0, // env_id
        .{ 0.1, 0, 0 }, // local_a
        .{ -0.1, 0, 0 }, // local_b
        1e-6, // compliance
    );

    try testing.expectEqual(xpbd.ConstraintType.positional, constraint.getType());
    try testing.expectApproxEqAbs(@as(f32, 1e-6), constraint.getCompliance(), 1e-9);
}

test "effective mass computation" {
    // Two unit mass bodies
    const inv_mass_a: f32 = 1.0;
    const inv_mass_b: f32 = 1.0;
    const inv_inertia_a: [3]f32 = .{ 1, 1, 1 };
    const inv_inertia_b: [3]f32 = .{ 1, 1, 1 };

    // Contact at body centers
    const r_a: [3]f32 = .{ 0, 0, 0 };
    const r_b: [3]f32 = .{ 0, 0, 0 };
    const normal: [3]f32 = .{ 0, 0, 1 };

    const w = xpbd.computeEffectiveMass(
        inv_mass_a,
        inv_mass_b,
        inv_inertia_a,
        inv_inertia_b,
        r_a,
        r_b,
        normal,
    );

    // When contact is at center, only linear mass contributes
    // w = inv_mass_a + inv_mass_b = 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), w, 0.001);
}

test "effective mass with offset contact" {
    const inv_mass_a: f32 = 1.0;
    const inv_mass_b: f32 = 0.0; // static
    const inv_inertia_a: [3]f32 = .{ 1, 1, 1 };
    const inv_inertia_b: [3]f32 = .{ 0, 0, 0 }; // static

    // Contact offset from center
    const r_a: [3]f32 = .{ 1, 0, 0 }; // 1m to the right
    const r_b: [3]f32 = .{ 0, 0, 0 };
    const normal: [3]f32 = .{ 0, 0, 1 }; // up

    const w = xpbd.computeEffectiveMass(
        inv_mass_a,
        inv_mass_b,
        inv_inertia_a,
        inv_inertia_b,
        r_a,
        r_b,
        normal,
    );

    // r × n = (1,0,0) × (0,0,1) = (0,-1,0)
    // angular contribution = |r × n|² * inv_I = 1 * 1 = 1
    // w = 1 (linear) + 1 (angular) = 2
    try testing.expectApproxEqAbs(@as(f32, 2.0), w, 0.001);
}

test "contact buffer add and iterate" {
    const allocator = testing.allocator;
    var buffer = try contact_buffer.ContactBufferManager.init(allocator, 4, 8);
    defer buffer.deinit();

    // Add contacts to environment 0
    _ = buffer.addContact(0, contact_buffer.CompactContact.init(
        .{ 0, 0, 0 },
        .{ 0, 0, 1 },
        0.1,
        0,
        1,
        0,
        0.5,
        0.0,
    ));

    _ = buffer.addContact(0, contact_buffer.CompactContact.init(
        .{ 1, 0, 0 },
        .{ 0, 0, 1 },
        0.2,
        0,
        2,
        0,
        0.5,
        0.0,
    ));

    // Add contact to environment 1
    _ = buffer.addContact(1, contact_buffer.CompactContact.init(
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
        0.15,
        1,
        2,
        1,
        0.5,
        0.0,
    ));

    try testing.expectEqual(@as(u32, 2), buffer.counts[0]);
    try testing.expectEqual(@as(u32, 1), buffer.counts[1]);
    try testing.expectEqual(@as(u32, 0), buffer.counts[2]);
    try testing.expectEqual(@as(u32, 3), buffer.total_contacts);

    // Iterate environment 0
    const contacts_0 = buffer.iterEnvContacts(0);
    try testing.expectEqual(@as(usize, 2), contacts_0.len);
    try testing.expectApproxEqAbs(@as(f32, 0.1), contacts_0[0].getPenetration(), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.2), contacts_0[1].getPenetration(), 0.001);
}

test "contact buffer compaction" {
    const allocator = testing.allocator;
    var buffer = try contact_buffer.ContactBufferManager.init(allocator, 2, 4);
    defer buffer.deinit();

    // Add 3 contacts to env 0
    _ = buffer.addContact(0, contact_buffer.CompactContact.init(.{ 0, 0, 0 }, .{ 0, 0, 1 }, 0.1, 0, 1, 0, 0.5, 0.0));
    _ = buffer.addContact(0, contact_buffer.CompactContact.init(.{ 1, 0, 0 }, .{ 0, 0, 1 }, 0.2, 0, 1, 0, 0.5, 0.0));
    _ = buffer.addContact(0, contact_buffer.CompactContact.init(.{ 2, 0, 0 }, .{ 0, 0, 1 }, 0.3, 0, 1, 0, 0.5, 0.0));

    try testing.expectEqual(@as(u32, 3), buffer.counts[0]);

    // Mark middle contact as inactive
    if (buffer.getContactMut(0, 1)) |c| {
        c.setInactive();
    }

    // Compact
    buffer.compactInPlace();

    // Should have 2 contacts now
    try testing.expectEqual(@as(u32, 2), buffer.counts[0]);
    try testing.expectEqual(@as(u32, 2), buffer.total_contacts);

    // Verify remaining contacts
    const contacts = buffer.iterEnvContacts(0);
    try testing.expectApproxEqAbs(@as(f32, 0.1), contacts[0].getPenetration(), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.3), contacts[1].getPenetration(), 0.001);
}

test "batched state initialization" {
    const allocator = testing.allocator;
    var state = try state_v2.BatchedState.init(allocator, 4, 8, 6, 32);
    defer state.deinit();

    try testing.expectEqual(@as(u32, 4), state.num_envs);
    try testing.expectEqual(@as(u32, 8), state.num_bodies);
    try testing.expectEqual(@as(u32, 6), state.num_joints);
    try testing.expectEqual(@as(u32, 32), state.obs_dim);

    // Initialize a dynamic body
    const body = state_v2.BodyState.dynamicBody(
        .{ 0, 0, 1 },
        .{ 0, 0, 0, 1 },
        1.0, // 1 kg
        .{ 0.1, 0.1, 0.1 }, // inertia
    );

    state.bodies[0] = body;

    try testing.expectApproxEqAbs(@as(f32, 1.0), state.bodies[0].getInvMass(), 0.001);
    try testing.expect(!state.bodies[0].isStatic());
}

test "static body has zero inverse mass" {
    const body = state_v2.BodyState.staticBody(
        .{ 0, 0, 0 },
        .{ 0, 0, 0, 1 },
    );

    try testing.expectApproxEqAbs(@as(f32, 0.0), body.getInvMass(), 1e-9);
    try testing.expect(body.isStatic());
}

test "solver params initialization" {
    const params = state_v2.SolverParams.init(
        1024, // num_envs
        8, // num_bodies
        64, // max_constraints
        0.002, // dt
        .{ 0, 0, -9.81 }, // gravity
    );

    try testing.expectEqual(@as(u32, 1024), params.num_envs);
    try testing.expectEqual(@as(u32, 8), params.num_bodies);
    try testing.expectApproxEqAbs(@as(f32, 0.002), params.dt, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 500.0), params.inv_dt, 0.1);
    try testing.expectApproxEqAbs(@as(f32, -9.81), params.gravity[2], 0.01);
}

test "XPBDConfig presets" {
    const rl_config = xpbd.XPBDConfig.forRL();
    try testing.expectEqual(@as(u32, 4), rl_config.iterations);
    try testing.expect(rl_config.warm_start);

    const accurate_config = xpbd.XPBDConfig.forAccuracy();
    try testing.expectEqual(@as(u32, 8), accurate_config.iterations);
    try testing.expectEqual(@as(u32, 4), accurate_config.substeps);
}

test "contact to xpbd constraint conversion" {
    const contact = contact_buffer.CompactContact.init(
        .{ 0, 0, 0.5 },
        .{ 0, 0, 1 },
        0.1,
        0,
        1,
        0,
        0.5,
        0.2,
    );

    const constraint = contact.toXPBDConstraint(1e-6);

    try testing.expectEqual(@as(u32, 0), constraint.getBodyA());
    try testing.expectEqual(@as(u32, 1), constraint.getBodyB());
    try testing.expectEqual(xpbd.ConstraintType.contact_normal, constraint.getType());
    try testing.expectApproxEqAbs(@as(f32, 1e-6), constraint.getCompliance(), 1e-9);
}

test "contact buffer layout computation" {
    const layout = contact_buffer.ContactBufferLayout.compute(1024, 32);

    // contacts: 1024 * 32 * 64 = 2 MB
    try testing.expectEqual(@as(u32, 0), layout.contacts_offset);
    try testing.expectEqual(@as(u32, 1024 * 32 * 64), layout.contacts_size);

    // counts: 1024 * 4 = 4 KB (aligned to 16)
    try testing.expectEqual(@as(u32, 1024 * 4), layout.counts_size);

    // Total should be contacts + aligned counts
    try testing.expect(layout.total_size >= layout.contacts_size + layout.counts_size);
}
