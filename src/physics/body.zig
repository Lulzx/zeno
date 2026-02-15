//! Rigid body definition and properties.

const std = @import("std");
const constants = @import("constants.zig");

/// Body type classification.
pub const BodyType = enum(u8) {
    /// Static body - never moves, infinite mass.
    static = 0,
    /// Kinematic body - moves but not affected by forces.
    kinematic = 1,
    /// Dynamic body - fully simulated.
    dynamic = 2,
};

/// Rigid body definition (scene description, not simulation state).
pub const BodyDef = struct {
    /// Unique name for this body.
    name: []const u8 = "",
    /// Body type.
    body_type: BodyType = .dynamic,
    /// Initial position in world coordinates.
    position: [3]f32 = .{ 0, 0, 0 },
    /// Initial orientation as quaternion (x, y, z, w).
    quaternion: [4]f32 = .{ 0, 0, 0, 1 },
    /// Initial linear velocity.
    linear_velocity: [3]f32 = .{ 0, 0, 0 },
    /// Initial angular velocity.
    angular_velocity: [3]f32 = .{ 0, 0, 0 },
    /// Mass in kg (0 = infinite mass / static).
    mass: f32 = 1.0,
    /// Inertia tensor diagonal (computed from geoms if zero).
    inertia: [3]f32 = .{ 0, 0, 0 },
    /// Index of parent body (-1 for root).
    parent_id: i32 = -1,
    /// Linear damping coefficient.
    linear_damping: f32 = 0.0,
    /// Angular damping coefficient.
    angular_damping: f32 = 0.0,
    /// Gravity scale (0 = no gravity).
    gravity_scale: f32 = 1.0,
    /// Enable collision detection for this body.
    collision_enabled: bool = true,
    /// Collision group (bodies in same group don't collide).
    collision_group: u32 = 0,
    /// Collision mask (bitfield for which groups to collide with).
    collision_mask: u32 = 0xFFFFFFFF,
    /// Center of mass offset from body frame origin (local coordinates).
    com_offset: [3]f32 = .{ 0, 0, 0 },

    /// Calculate inverse mass.
    /// Static and kinematic bodies have infinite mass (inv_mass = 0) so forces don't affect them.
    pub fn invMass(self: *const BodyDef) f32 {
        if (self.body_type == .static or self.body_type == .kinematic or self.mass <= 0.0) {
            return 0.0;
        }
        return 1.0 / self.mass;
    }

    /// Calculate inverse inertia tensor diagonal.
    /// Static and kinematic bodies have infinite inertia so torques don't affect them.
    pub fn invInertia(self: *const BodyDef) [3]f32 {
        if (self.body_type == .static or self.body_type == .kinematic) {
            return .{ 0, 0, 0 };
        }
        var inv: [3]f32 = undefined;
        for (0..3) |i| {
            if (self.inertia[i] > constants.EPSILON) {
                inv[i] = 1.0 / self.inertia[i];
            } else {
                inv[i] = 0.0;
            }
        }
        return inv;
    }

    /// Compute inertia tensor from mass assuming uniform density sphere.
    pub fn setInertiaSphere(self: *BodyDef, radius: f32) void {
        const i = 0.4 * self.mass * radius * radius;
        self.inertia = .{ i, i, i };
    }

    /// Compute inertia tensor from mass assuming uniform density box.
    pub fn setInertiaBox(self: *BodyDef, half_extents: [3]f32) void {
        const m = self.mass / 12.0;
        const x2 = half_extents[0] * half_extents[0] * 4.0;
        const y2 = half_extents[1] * half_extents[1] * 4.0;
        const z2 = half_extents[2] * half_extents[2] * 4.0;
        self.inertia = .{
            m * (y2 + z2),
            m * (x2 + z2),
            m * (x2 + y2),
        };
    }

    /// Compute inertia tensor from mass assuming uniform density capsule.
    pub fn setInertiaCapsule(self: *BodyDef, radius: f32, height: f32) void {
        const r2 = radius * radius;
        const h = height - 2.0 * radius; // Cylinder height
        const h2 = h * h;

        // Cylinder part
        const m_cyl = self.mass * h / (h + 4.0 / 3.0 * radius);
        const i_cyl_axial = 0.5 * m_cyl * r2;
        const i_cyl_trans = m_cyl * (3.0 * r2 + h2) / 12.0;

        // Hemisphere parts
        const m_hem = (self.mass - m_cyl) * 0.5;
        const i_hem = 0.4 * m_hem * r2;
        const offset = h * 0.5 + 3.0 / 8.0 * radius;
        const i_hem_trans = i_hem + m_hem * offset * offset;

        self.inertia = .{
            i_cyl_trans + 2.0 * i_hem_trans,
            i_cyl_trans + 2.0 * i_hem_trans,
            i_cyl_axial + 2.0 * i_hem,
        };
    }
};

/// Body state for GPU simulation (SoA layout).
/// These are the actual fields stored in GPU buffers.
pub const BodyStateFields = struct {
    /// Position (xyz) + padding for alignment.
    pub const Position = [4]f32;
    /// Orientation quaternion (xyzw).
    pub const Quaternion = [4]f32;
    /// Linear velocity (xyz) + padding.
    pub const LinearVelocity = [4]f32;
    /// Angular velocity (xyz) + padding.
    pub const AngularVelocity = [4]f32;
    /// Accumulated force (xyz) + padding.
    pub const Force = [4]f32;
    /// Accumulated torque (xyz) + padding.
    pub const Torque = [4]f32;
    /// Inverse mass and inverse inertia (inv_mass, inv_Ixx, inv_Iyy, inv_Izz).
    pub const InvMassInertia = [4]f32;
};

/// Transform representation.
pub const Transform = struct {
    position: [3]f32,
    quaternion: [4]f32,

    pub const IDENTITY = Transform{
        .position = .{ 0, 0, 0 },
        .quaternion = .{ 0, 0, 0, 1 },
    };

    /// Create from position and quaternion arrays.
    pub fn init(pos: [3]f32, quat: [4]f32) Transform {
        return .{ .position = pos, .quaternion = quat };
    }

    /// Create from position only (identity rotation).
    pub fn fromPosition(pos: [3]f32) Transform {
        return .{ .position = pos, .quaternion = .{ 0, 0, 0, 1 } };
    }

    /// Create from axis-angle rotation.
    pub fn fromAxisAngle(pos: [3]f32, axis: [3]f32, angle: f32) Transform {
        const half_angle = angle * 0.5;
        const s = @sin(half_angle);
        const c = @cos(half_angle);
        return .{
            .position = pos,
            .quaternion = .{ axis[0] * s, axis[1] * s, axis[2] * s, c },
        };
    }

    /// Create from Euler angles (ZYX convention).
    pub fn fromEuler(pos: [3]f32, roll: f32, pitch: f32, yaw: f32) Transform {
        const cr = @cos(roll * 0.5);
        const sr = @sin(roll * 0.5);
        const cp = @cos(pitch * 0.5);
        const sp = @sin(pitch * 0.5);
        const cy = @cos(yaw * 0.5);
        const sy = @sin(yaw * 0.5);

        return .{
            .position = pos,
            .quaternion = .{
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            },
        };
    }

    /// Transform a point from local to world coordinates.
    pub fn transformPoint(self: *const Transform, local: [3]f32) [3]f32 {
        const rotated = rotateByQuat(local, self.quaternion);
        return .{
            rotated[0] + self.position[0],
            rotated[1] + self.position[1],
            rotated[2] + self.position[2],
        };
    }

    /// Transform a vector (rotation only, no translation).
    pub fn transformVector(self: *const Transform, local: [3]f32) [3]f32 {
        return rotateByQuat(local, self.quaternion);
    }

    /// Inverse transform a point from world to local coordinates.
    pub fn inverseTransformPoint(self: *const Transform, world: [3]f32) [3]f32 {
        const rel: [3]f32 = .{
            world[0] - self.position[0],
            world[1] - self.position[1],
            world[2] - self.position[2],
        };
        return rotateByQuatInverse(rel, self.quaternion);
    }

    /// Compose transforms (self * other).
    pub fn multiply(self: *const Transform, other: *const Transform) Transform {
        return .{
            .position = self.transformPoint(other.position),
            .quaternion = quatMultiply(self.quaternion, other.quaternion),
        };
    }

    /// Get inverse transform.
    pub fn inverse(self: *const Transform) Transform {
        const inv_quat = quatConjugate(self.quaternion);
        const inv_pos = rotateByQuat(.{
            -self.position[0],
            -self.position[1],
            -self.position[2],
        }, inv_quat);
        return .{
            .position = inv_pos,
            .quaternion = inv_quat,
        };
    }
};

// Quaternion math helpers

fn quatMultiply(a: [4]f32, b: [4]f32) [4]f32 {
    return .{
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    };
}

fn quatConjugate(q: [4]f32) [4]f32 {
    return .{ -q[0], -q[1], -q[2], q[3] };
}

fn rotateByQuat(v: [3]f32, q: [4]f32) [3]f32 {
    // Rodrigues' rotation formula optimized
    const qv: [3]f32 = .{ q[0], q[1], q[2] };
    const uv = cross(qv, v);
    const uuv = cross(qv, uv);
    return .{
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    };
}

fn rotateByQuatInverse(v: [3]f32, q: [4]f32) [3]f32 {
    return rotateByQuat(v, quatConjugate(q));
}

fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

/// AABB (Axis-Aligned Bounding Box) for collision detection.
pub const AABB = struct {
    min: [3]f32,
    max: [3]f32,

    pub fn init(min: [3]f32, max: [3]f32) AABB {
        return .{ .min = min, .max = max };
    }

    pub fn fromCenterExtents(ctr: [3]f32, half_extents: [3]f32) AABB {
        return .{
            .min = .{
                ctr[0] - half_extents[0],
                ctr[1] - half_extents[1],
                ctr[2] - half_extents[2],
            },
            .max = .{
                ctr[0] + half_extents[0],
                ctr[1] + half_extents[1],
                ctr[2] + half_extents[2],
            },
        };
    }

    pub fn intersects(self: *const AABB, other: *const AABB) bool {
        return self.min[0] <= other.max[0] and self.max[0] >= other.min[0] and
            self.min[1] <= other.max[1] and self.max[1] >= other.min[1] and
            self.min[2] <= other.max[2] and self.max[2] >= other.min[2];
    }

    pub fn contains(self: *const AABB, point: [3]f32) bool {
        return point[0] >= self.min[0] and point[0] <= self.max[0] and
            point[1] >= self.min[1] and point[1] <= self.max[1] and
            point[2] >= self.min[2] and point[2] <= self.max[2];
    }

    pub fn center(self: *const AABB) [3]f32 {
        return .{
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        };
    }

    pub fn extents(self: *const AABB) [3]f32 {
        return .{
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        };
    }

    pub fn expand(self: *AABB, margin: f32) void {
        self.min[0] -= margin;
        self.min[1] -= margin;
        self.min[2] -= margin;
        self.max[0] += margin;
        self.max[1] += margin;
        self.max[2] += margin;
    }

    pub fn merge(self: *const AABB, other: *const AABB) AABB {
        return .{
            .min = .{
                @min(self.min[0], other.min[0]),
                @min(self.min[1], other.min[1]),
                @min(self.min[2], other.min[2]),
            },
            .max = .{
                @max(self.max[0], other.max[0]),
                @max(self.max[1], other.max[1]),
                @max(self.max[2], other.max[2]),
            },
        };
    }
};
