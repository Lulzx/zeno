//! Tendon and equality constraint definitions for MuJoCo-compatible simulation.
//!
//! Tendons connect multiple joints or sites, useful for modeling:
//! - Muscles and cables
//! - Pulleys and routing
//! - Coupled joint motion
//!
//! Equality constraints enforce exact relationships:
//! - Weld (fixed relative pose)
//! - Connect (distance between bodies)
//! - Joint (force joint to target)
//! - Tendon (force tendon length)

const std = @import("std");
const constants = @import("constants.zig");

/// Tendon type enumeration.
pub const TendonType = enum(u8) {
    /// Fixed tendon: linear combination of joint positions.
    fixed = 0,
    /// Spatial tendon: passes through via-points (sites).
    spatial = 1,
};

/// Tendon wrapping object type.
pub const WrapType = enum(u8) {
    /// No wrapping.
    none = 0,
    /// Wrap around sphere.
    sphere = 1,
    /// Wrap around cylinder.
    cylinder = 2,
};

/// Element in a tendon path (for spatial tendons).
pub const TendonPathElement = struct {
    /// Site index for via-point, or joint index for fixed tendon.
    index: u32 = 0,
    /// Coefficient for fixed tendon (joint contribution to length).
    coef: f32 = 1.0,
    /// Wrapping object type.
    wrap_type: WrapType = .none,
    /// Wrapping object index (geom index).
    wrap_obj: u32 = 0,
};

/// Tendon definition.
pub const TendonDef = struct {
    /// Unique name for this tendon.
    name: []const u8 = "",
    /// Tendon type.
    tendon_type: TendonType = .fixed,
    /// Path elements (joints for fixed, sites for spatial).
    path: []const TendonPathElement = &.{},
    /// Stiffness coefficient.
    stiffness: f32 = 0.0,
    /// Damping coefficient.
    damping: f32 = 0.0,
    /// Lower length limit.
    limit_lower: f32 = -std.math.inf(f32),
    /// Upper length limit.
    limit_upper: f32 = std.math.inf(f32),
    /// Enable length limits.
    limited: bool = false,
    /// Rest length (for spring behavior).
    rest_length: f32 = 0.0,
    /// Constraint margin.
    margin: f32 = 0.0,
    /// Tendon width (for visualization).
    width: f32 = 0.003,
    /// RGBA color for visualization.
    rgba: [4]f32 = .{ 0.5, 0.4, 0.3, 1.0 },

    /// Compute tendon length from joint positions (for fixed tendon).
    pub fn computeLength(self: *const TendonDef, joint_positions: []const f32) f32 {
        if (self.tendon_type != .fixed) return 0.0;

        var length: f32 = 0.0;
        for (self.path) |elem| {
            if (elem.index < joint_positions.len) {
                length += elem.coef * joint_positions[elem.index];
            }
        }
        return length;
    }

    /// Check if length is within limits.
    pub fn isWithinLimits(self: *const TendonDef, length: f32) bool {
        if (!self.limited) return true;
        return length >= self.limit_lower and length <= self.limit_upper;
    }

    /// Get limit violation (positive if outside limits).
    pub fn limitViolation(self: *const TendonDef, length: f32) f32 {
        if (!self.limited) return 0.0;
        if (length < self.limit_lower) return self.limit_lower - length;
        if (length > self.limit_upper) return length - self.limit_upper;
        return 0.0;
    }

    /// Compute spring force (toward rest length).
    pub fn computeSpringForce(self: *const TendonDef, length: f32, velocity: f32) f32 {
        const displacement = length - self.rest_length;
        return -self.stiffness * displacement - self.damping * velocity;
    }
};

/// Equality constraint type.
pub const EqualityType = enum(u8) {
    /// Connect: maintain distance between two body points.
    connect = 0,
    /// Weld: fix relative pose between bodies.
    weld = 1,
    /// Joint: force joint to match target position.
    joint = 2,
    /// Tendon: force tendon length to target.
    tendon = 3,
    /// Distance: maintain distance between two bodies.
    distance = 4,
};

/// Equality constraint definition.
pub const EqualityDef = struct {
    /// Unique name.
    name: []const u8 = "",
    /// Constraint type.
    equality_type: EqualityType = .connect,
    /// First body/joint/tendon index.
    obj1: u32 = 0,
    /// Second body/joint/tendon index (for connect/weld).
    obj2: u32 = 0,
    /// Anchor point on first object (body-local coordinates).
    anchor1: [3]f32 = .{ 0, 0, 0 },
    /// Anchor point on second object (body-local coordinates).
    anchor2: [3]f32 = .{ 0, 0, 0 },
    /// Target value (distance, angle, length depending on type).
    target: f32 = 0.0,
    /// Relative quaternion for weld constraints (wxyz).
    relpose: [4]f32 = .{ 1, 0, 0, 0 },
    /// Joint position offset for joint constraints.
    polycoef: [5]f32 = .{ 0, 1, 0, 0, 0 },
    /// Constraint is active.
    active: bool = true,
    /// Solver tolerance.
    solimp: [5]f32 = .{ 0.9, 0.95, 0.001, 0.5, 2 },
    /// Solver reference.
    solref: [2]f32 = .{ 0.02, 1 },
};

/// GPU-optimized tendon constraint data.
pub const TendonGPU = extern struct {
    // --- 16 bytes ---
    /// (tendon_id, type, num_elements, flags)
    header: [4]u32 align(16),
    // --- 16 bytes ---
    /// (stiffness, damping, limit_lower, limit_upper)
    params: [4]f32 align(16),
    // --- 16 bytes ---
    /// (rest_length, margin, current_length, velocity)
    state: [4]f32 align(16),
    // --- 16 bytes ---
    /// (lambda, lambda_prev, violation, effective_mass)
    solver: [4]f32 align(16),

    pub fn fromDef(def: *const TendonDef, id: u32) TendonGPU {
        const flags: u32 = if (def.limited) 1 else 0;
        return .{
            .header = .{ id, @intFromEnum(def.tendon_type), @intCast(def.path.len), flags },
            .params = .{ def.stiffness, def.damping, def.limit_lower, def.limit_upper },
            .state = .{ def.rest_length, def.margin, 0, 0 },
            .solver = .{ 0, 0, 0, 0 },
        };
    }
};

/// GPU-optimized equality constraint data.
pub const EqualityGPU = extern struct {
    // --- 16 bytes ---
    /// (eq_id, type, obj1, obj2)
    header: [4]u32 align(16),
    // --- 16 bytes ---
    /// anchor1 (xyz) + target
    anchor1_target: [4]f32 align(16),
    // --- 16 bytes ---
    /// anchor2 (xyz) + active flag
    anchor2_active: [4]f32 align(16),
    // --- 16 bytes ---
    /// relpose quaternion (wxyz)
    relpose: [4]f32 align(16),
    // --- 16 bytes ---
    /// (lambda, lambda_prev, violation, effective_mass)
    solver: [4]f32 align(16),

    pub fn fromDef(def: *const EqualityDef, id: u32) EqualityGPU {
        return .{
            .header = .{ id, @intFromEnum(def.equality_type), def.obj1, def.obj2 },
            .anchor1_target = .{ def.anchor1[0], def.anchor1[1], def.anchor1[2], def.target },
            .anchor2_active = .{ def.anchor2[0], def.anchor2[1], def.anchor2[2], if (def.active) 1.0 else 0.0 },
            .relpose = def.relpose,
            .solver = .{ 0, 0, 0, 0 },
        };
    }
};

/// Tendon path element for GPU (fixed size).
pub const TendonPathGPU = extern struct {
    // --- 16 bytes ---
    /// (joint/site index, wrap_type, wrap_obj, pad)
    indices: [4]u32 align(16),
    // --- 16 bytes ---
    /// (coefficient, site_pos x, site_pos y, site_pos z)
    data: [4]f32 align(16),
};

/// Create a fixed tendon from joint indices and coefficients.
pub fn createFixedTendon(
    allocator: std.mem.Allocator,
    name: []const u8,
    joints: []const u32,
    coefs: []const f32,
) !TendonDef {
    std.debug.assert(joints.len == coefs.len);

    const path = try allocator.alloc(TendonPathElement, joints.len);
    for (joints, coefs, 0..) |joint, coef, i| {
        path[i] = .{
            .index = joint,
            .coef = coef,
            .wrap_type = .none,
            .wrap_obj = 0,
        };
    }

    return TendonDef{
        .name = name,
        .tendon_type = .fixed,
        .path = path,
    };
}

/// Create a spatial tendon from site indices.
pub fn createSpatialTendon(
    allocator: std.mem.Allocator,
    name: []const u8,
    sites: []const u32,
) !TendonDef {
    const path = try allocator.alloc(TendonPathElement, sites.len);
    for (sites, 0..) |site, i| {
        path[i] = .{
            .index = site,
            .coef = 1.0,
            .wrap_type = .none,
            .wrap_obj = 0,
        };
    }

    return TendonDef{
        .name = name,
        .tendon_type = .spatial,
        .path = path,
    };
}

/// Create a connect equality constraint.
pub fn createConnectConstraint(
    name: []const u8,
    body1: u32,
    body2: u32,
    anchor1: [3]f32,
    anchor2: [3]f32,
) EqualityDef {
    return EqualityDef{
        .name = name,
        .equality_type = .connect,
        .obj1 = body1,
        .obj2 = body2,
        .anchor1 = anchor1,
        .anchor2 = anchor2,
    };
}

/// Create a weld equality constraint.
pub fn createWeldConstraint(
    name: []const u8,
    body1: u32,
    body2: u32,
    anchor1: [3]f32,
    anchor2: [3]f32,
    relpose: [4]f32,
) EqualityDef {
    return EqualityDef{
        .name = name,
        .equality_type = .weld,
        .obj1 = body1,
        .obj2 = body2,
        .anchor1 = anchor1,
        .anchor2 = anchor2,
        .relpose = relpose,
    };
}

/// Create a joint equality constraint.
pub fn createJointConstraint(
    name: []const u8,
    joint1: u32,
    joint2: u32,
    polycoef: [5]f32,
) EqualityDef {
    return EqualityDef{
        .name = name,
        .equality_type = .joint,
        .obj1 = joint1,
        .obj2 = joint2,
        .polycoef = polycoef,
    };
}

/// Create a tendon equality constraint.
pub fn createTendonConstraint(
    name: []const u8,
    tendon: u32,
    target: f32,
) EqualityDef {
    return EqualityDef{
        .name = name,
        .equality_type = .tendon,
        .obj1 = tendon,
        .target = target,
    };
}
