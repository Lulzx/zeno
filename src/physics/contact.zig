//! Contact data structures for collision response.

const std = @import("std");
const constants = @import("constants.zig");

/// Contact point between two bodies.
pub const Contact = struct {
    /// Contact point in world coordinates.
    position: [3]f32,
    /// Contact normal (from body A to body B).
    normal: [3]f32,
    /// Penetration depth (positive = overlapping).
    penetration: f32,
    /// Body A index.
    body_a: u32,
    /// Body B index.
    body_b: u32,
    /// Geometry A index.
    geom_a: u32,
    /// Geometry B index.
    geom_b: u32,
    /// Local contact point on body A.
    local_a: [3]f32,
    /// Local contact point on body B.
    local_b: [3]f32,
    /// Combined friction coefficient.
    friction: f32,
    /// Combined restitution coefficient.
    restitution: f32,
    /// Accumulated normal impulse (for warm starting).
    impulse_normal: f32 = 0.0,
    /// Accumulated tangent impulse 1.
    impulse_tangent1: f32 = 0.0,
    /// Accumulated tangent impulse 2.
    impulse_tangent2: f32 = 0.0,
    /// Tangent direction 1.
    tangent1: [3]f32 = .{ 0, 0, 0 },
    /// Tangent direction 2.
    tangent2: [3]f32 = .{ 0, 0, 0 },
    /// Contact age (frames since contact started).
    age: u32 = 0,
    /// Is this contact active?
    active: bool = true,

    /// Initialize a new contact.
    pub fn init(
        pos: [3]f32,
        normal: [3]f32,
        penetration: f32,
        body_a: u32,
        body_b: u32,
    ) Contact {
        var contact = Contact{
            .position = pos,
            .normal = normal,
            .penetration = penetration,
            .body_a = body_a,
            .body_b = body_b,
            .geom_a = 0,
            .geom_b = 0,
            .local_a = .{ 0, 0, 0 },
            .local_b = .{ 0, 0, 0 },
            .friction = constants.DEFAULT_FRICTION,
            .restitution = constants.DEFAULT_RESTITUTION,
        };
        contact.computeTangents();
        return contact;
    }

    /// Compute tangent vectors from normal.
    pub fn computeTangents(self: *Contact) void {
        // Find a vector not parallel to normal
        const ref: [3]f32 = if (@abs(self.normal[0]) < 0.9)
            .{ 1, 0, 0 }
        else
            .{ 0, 1, 0 };

        // First tangent = normal × ref
        self.tangent1 = cross(self.normal, ref);
        self.tangent1 = normalize(self.tangent1);

        // Second tangent = normal × tangent1
        self.tangent2 = cross(self.normal, self.tangent1);
    }

    /// Get relative velocity at contact point.
    pub fn relativeVelocity(
        self: *const Contact,
        vel_a: [3]f32,
        omega_a: [3]f32,
        vel_b: [3]f32,
        omega_b: [3]f32,
    ) [3]f32 {
        // v_rel = (v_a + ω_a × r_a) - (v_b + ω_b × r_b)
        const v_a = add(vel_a, cross(omega_a, self.local_a));
        const v_b = add(vel_b, cross(omega_b, self.local_b));
        return sub(v_a, v_b);
    }

    /// Get normal component of relative velocity.
    pub fn normalVelocity(
        self: *const Contact,
        vel_a: [3]f32,
        omega_a: [3]f32,
        vel_b: [3]f32,
        omega_b: [3]f32,
    ) f32 {
        const v_rel = self.relativeVelocity(vel_a, omega_a, vel_b, omega_b);
        return dot(v_rel, self.normal);
    }

    /// Check if contact is separating.
    pub fn isSeparating(
        self: *const Contact,
        vel_a: [3]f32,
        omega_a: [3]f32,
        vel_b: [3]f32,
        omega_b: [3]f32,
    ) bool {
        return self.normalVelocity(vel_a, omega_a, vel_b, omega_b) > constants.EPSILON;
    }
};

/// Contact manifold (multiple contact points between a pair of bodies).
pub const ContactManifold = struct {
    /// Contact points.
    points: [constants.MAX_CONTACTS_PER_PAIR]Contact = undefined,
    /// Number of active contact points.
    count: u32 = 0,
    /// Body A index.
    body_a: u32 = 0,
    /// Body B index.
    body_b: u32 = 0,
    /// Is manifold active?
    active: bool = false,

    /// Add a contact point to the manifold.
    pub fn addContact(self: *ContactManifold, contact: Contact) void {
        if (self.count >= constants.MAX_CONTACTS_PER_PAIR) {
            // Replace the contact with smallest penetration
            var min_idx: u32 = 0;
            var min_pen = self.points[0].penetration;
            for (1..self.count) |i| {
                if (self.points[i].penetration < min_pen) {
                    min_pen = self.points[i].penetration;
                    min_idx = @intCast(i);
                }
            }
            if (contact.penetration > min_pen) {
                self.points[min_idx] = contact;
            }
        } else {
            self.points[self.count] = contact;
            self.count += 1;
        }
    }

    /// Remove contacts that are no longer valid.
    pub fn prune(self: *ContactManifold, threshold: f32) void {
        var i: u32 = 0;
        while (i < self.count) {
            if (self.points[i].penetration < -threshold or !self.points[i].active) {
                // Remove by swapping with last
                self.count -= 1;
                if (i < self.count) {
                    self.points[i] = self.points[self.count];
                }
            } else {
                i += 1;
            }
        }
        self.active = self.count > 0;
    }

    /// Clear all contacts.
    pub fn clear(self: *ContactManifold) void {
        self.count = 0;
        self.active = false;
    }

    /// Get average contact point.
    pub fn averagePosition(self: *const ContactManifold) [3]f32 {
        if (self.count == 0) return .{ 0, 0, 0 };

        var sum: [3]f32 = .{ 0, 0, 0 };
        for (0..self.count) |i| {
            sum[0] += self.points[i].position[0];
            sum[1] += self.points[i].position[1];
            sum[2] += self.points[i].position[2];
        }
        const n: f32 = @floatFromInt(self.count);
        return .{ sum[0] / n, sum[1] / n, sum[2] / n };
    }

    /// Get average normal.
    pub fn averageNormal(self: *const ContactManifold) [3]f32 {
        if (self.count == 0) return .{ 0, 0, 1 };

        var sum: [3]f32 = .{ 0, 0, 0 };
        for (0..self.count) |i| {
            sum[0] += self.points[i].normal[0];
            sum[1] += self.points[i].normal[1];
            sum[2] += self.points[i].normal[2];
        }
        return normalize(sum);
    }
};

/// GPU-friendly contact data for compute shaders.
pub const ContactGPU = extern struct {
    /// Position xyz + penetration.
    position_pen: [4]f32 align(16),
    /// Normal xyz + friction.
    normal_friction: [4]f32 align(16),
    /// Body indices (a, b) and geom indices.
    indices: [4]u32 align(16),
    /// Impulses (normal, tangent1, tangent2, restitution).
    impulses: [4]f32 align(16),

    pub fn fromContact(c: *const Contact) ContactGPU {
        return .{
            .position_pen = .{ c.position[0], c.position[1], c.position[2], c.penetration },
            .normal_friction = .{ c.normal[0], c.normal[1], c.normal[2], c.friction },
            .indices = .{ c.body_a, c.body_b, c.geom_a, c.geom_b },
            .impulses = .{ c.impulse_normal, c.impulse_tangent1, c.impulse_tangent2, c.restitution },
        };
    }

    pub fn toContact(self: *const ContactGPU) Contact {
        return .{
            .position = .{ self.position_pen[0], self.position_pen[1], self.position_pen[2] },
            .normal = .{ self.normal_friction[0], self.normal_friction[1], self.normal_friction[2] },
            .penetration = self.position_pen[3],
            .friction = self.normal_friction[3],
            .body_a = self.indices[0],
            .body_b = self.indices[1],
            .geom_a = self.indices[2],
            .geom_b = self.indices[3],
            .impulse_normal = self.impulses[0],
            .impulse_tangent1 = self.impulses[1],
            .impulse_tangent2 = self.impulses[2],
            .restitution = self.impulses[3],
            .local_a = .{ 0, 0, 0 },
            .local_b = .{ 0, 0, 0 },
        };
    }
};

/// Contact buffer for GPU simulation.
pub const ContactBuffer = struct {
    /// Maximum contacts per environment.
    max_contacts: u32,
    /// Number of environments.
    num_envs: u32,

    /// Get buffer size in bytes for contact data.
    pub fn dataSize(self: *const ContactBuffer) usize {
        return self.num_envs * self.max_contacts * @sizeOf(ContactGPU);
    }

    /// Get buffer size for contact counts.
    pub fn countSize(self: *const ContactBuffer) usize {
        return self.num_envs * @sizeOf(u32);
    }

    /// Get index for contact in linear buffer.
    pub fn index(self: *const ContactBuffer, env_id: u32, contact_id: u32) u32 {
        return env_id * self.max_contacts + contact_id;
    }
};

// Vector math helpers

fn dot(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn add(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

fn sub(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn normalize(v: [3]f32) [3]f32 {
    const len = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < constants.EPSILON) return .{ 0, 0, 1 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}
