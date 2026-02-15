//! XPBD (Extended Position-Based Dynamics) constraint solver.
//!
//! XPBD unifies position and velocity correction into a single compliant framework:
//!   λ = (-C - α̃ * λ_prev) / (w + α̃)
//!   Δx = w * ∇C * λ
//!
//! Where:
//!   - C is the constraint violation
//!   - α̃ = compliance / dt² is the time-scaled compliance
//!   - w = Σ(m_inv * |∇C|²) is the generalized inverse mass
//!   - λ is the Lagrange multiplier (accumulated impulse)
//!
//! Benefits:
//!   - Stable at large timesteps
//!   - Unified treatment of joints and contacts
//!   - Soft constraints via compliance parameter
//!   - Better energy conservation than standard PBD

const std = @import("std");
const constants = @import("constants.zig");

/// XPBD constraint types (unified for joints and contacts).
pub const ConstraintType = enum(u8) {
    /// Contact non-penetration constraint
    contact_normal = 0,
    /// Contact friction constraint
    contact_friction = 1,
    /// Point-to-point (ball joint anchor)
    positional = 2,
    /// Angular constraint (hinge alignment)
    angular = 3,
    /// Angular limit (joint range)
    angular_limit = 4,
    /// Linear limit (prismatic range)
    linear_limit = 5,
    /// Tendon length constraint
    tendon = 6,
    /// Weld constraint (fixed relative pose)
    weld = 7,
    /// Connect constraint (distance between body points)
    connect = 8,
    /// Joint equality constraint
    joint_equality = 9,
    /// Slider constraint (prismatic: constrain perpendicular translation, free along axis)
    slider = 10,
};

/// GPU-optimized XPBD constraint data.
/// Layout optimized for coalesced memory access in compute shaders.
pub const XPBDConstraint = extern struct {
    // --- 16 bytes ---
    /// Body indices: (body_a, body_b, env_id, type_and_color)
    /// type_and_color format: (type & 0xFF) | (color << 8)
    indices: [4]u32 align(16),

    // --- 16 bytes ---
    /// Local anchor on body A (xyz) + compliance
    anchor_a: [4]f32 align(16),

    // --- 16 bytes ---
    /// Local anchor on body B (xyz) + damping
    anchor_b: [4]f32 align(16),

    // --- 16 bytes ---
    /// Constraint axis/normal (xyz) + target value
    axis_target: [4]f32 align(16),

    // --- 16 bytes ---
    /// Limits: (lower, upper, friction, restitution)
    limits: [4]f32 align(16),

    // --- 16 bytes ---
    /// Solver state: (lambda, lambda_prev, violation, effective_mass)
    state: [4]f32 align(16),

    // Total: 96 bytes per constraint (6 cache lines)

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
        return @enumFromInt(self.indices[3] & 0xFF);
    }

    pub fn getColor(self: *const XPBDConstraint) u8 {
        return @truncate(self.indices[3] >> 8);
    }

    pub fn setColor(self: *XPBDConstraint, color: u8) void {
        self.indices[3] = (self.indices[3] & 0xFF) | (@as(u32, color) << 8);
    }

    pub fn getCompliance(self: *const XPBDConstraint) f32 {
        return self.anchor_a[3];
    }

    pub fn getDamping(self: *const XPBDConstraint) f32 {
        return self.anchor_b[3];
    }

    pub fn getTarget(self: *const XPBDConstraint) f32 {
        return self.axis_target[3];
    }

    pub fn getLambda(self: *const XPBDConstraint) f32 {
        return self.state[0];
    }

    pub fn setLambda(self: *XPBDConstraint, lambda: f32) void {
        self.state[0] = lambda;
    }
};

/// XPBD solver configuration.
pub const XPBDConfig = struct {
    /// Number of solver iterations.
    iterations: u32 = 4,
    /// Global position compliance (0 = rigid).
    position_compliance: f32 = 0.0,
    /// Contact compliance (soft contacts).
    contact_compliance: f32 = 0.0,
    /// Joint compliance.
    joint_compliance: f32 = 0.0,
    /// Velocity damping factor.
    damping: f32 = 0.0,
    /// Enable warm starting from previous frame.
    warm_start: bool = true,
    /// Substep count for stability.
    substeps: u32 = 1,
    /// Relaxation factor (1.0 = standard, <1.0 = under-relaxed).
    relaxation: f32 = 1.0,

    pub fn forRL() XPBDConfig {
        return .{
            .iterations = 4,
            .position_compliance = 0.0,
            .contact_compliance = 1e-9, // Slightly soft contacts
            .joint_compliance = 0.0,
            .damping = 0.005,
            .warm_start = true,
            .substeps = 1,
            .relaxation = 1.0,
        };
    }

    pub fn forAccuracy() XPBDConfig {
        return .{
            .iterations = 8,
            .position_compliance = 0.0,
            .contact_compliance = 0.0,
            .joint_compliance = 0.0,
            .damping = 0.001,
            .warm_start = true,
            .substeps = 4,
            .relaxation = 1.0,
        };
    }
};

/// Pre-computed values for XPBD solve step.
pub const XPBDSolveData = extern struct {
    // --- 16 bytes ---
    /// Scaled compliance: α̃ = compliance / dt²
    alpha_tilde: f32 align(16),
    /// Inverse timestep squared: 1/dt²
    inv_dt_sq: f32,
    /// Timestep
    dt: f32,
    /// Iteration index
    iteration: u32,

    // --- 16 bytes ---
    /// Gravity (xyz) + unused
    gravity: [4]f32 align(16),

    pub fn init(dt: f32, compliance: f32, gravity: [3]f32) XPBDSolveData {
        const dt_sq = dt * dt;
        return .{
            .alpha_tilde = compliance / dt_sq,
            .inv_dt_sq = 1.0 / dt_sq,
            .dt = dt,
            .iteration = 0,
            .gravity = .{ gravity[0], gravity[1], gravity[2], 0 },
        };
    }
};

/// Create a contact constraint.
pub fn createContactConstraint(
    body_a: u32,
    body_b: u32,
    env_id: u32,
    position: [3]f32,
    normal: [3]f32,
    penetration: f32,
    friction: f32,
    restitution: f32,
    compliance: f32,
) XPBDConstraint {
    return .{
        .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.contact_normal) },
        .anchor_a = .{ position[0], position[1], position[2], compliance },
        .anchor_b = .{ 0, 0, 0, 0 }, // Computed from world position
        .axis_target = .{ normal[0], normal[1], normal[2], 0 },
        .limits = .{ 0, std.math.inf(f32), friction, restitution },
        .state = .{ 0, 0, penetration, 0 },
    };
}

/// Create a positional (point-to-point) constraint.
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

/// Create an angular constraint (hinge axis alignment).
pub fn createAngularConstraint(
    body_a: u32,
    body_b: u32,
    env_id: u32,
    axis_a: [3]f32,
    axis_b: [3]f32,
    compliance: f32,
    damping: f32,
) XPBDConstraint {
    return .{
        .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.angular) },
        .anchor_a = .{ axis_a[0], axis_a[1], axis_a[2], compliance },
        .anchor_b = .{ axis_b[0], axis_b[1], axis_b[2], damping },
        .axis_target = .{ 0, 0, 0, 0 },
        .limits = .{ 0, 0, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Create an angular limit constraint.
pub fn createAngularLimitConstraint(
    body_a: u32,
    body_b: u32,
    env_id: u32,
    axis: [3]f32,
    lower: f32,
    upper: f32,
    compliance: f32,
) XPBDConstraint {
    return .{
        .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.angular_limit) },
        .anchor_a = .{ axis[0], axis[1], axis[2], compliance },
        .anchor_b = .{ 0, 0, 0, 0 },
        .axis_target = .{ 0, 0, 0, 0 },
        .limits = .{ lower, upper, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Create a tendon length constraint.
pub fn createTendonConstraint(
    tendon_id: u32,
    env_id: u32,
    rest_length: f32,
    stiffness: f32,
    damping: f32,
    lower: f32,
    upper: f32,
) XPBDConstraint {
    const compliance = if (stiffness > 0) 1.0 / stiffness else 0.0;
    return .{
        .indices = .{ tendon_id, 0, env_id, @intFromEnum(ConstraintType.tendon) },
        .anchor_a = .{ rest_length, 0, 0, compliance },
        .anchor_b = .{ 0, 0, 0, damping },
        .axis_target = .{ 0, 0, 0, 0 },
        .limits = .{ lower, upper, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Create a weld (fixed relative pose) constraint.
pub fn createWeldConstraint(
    body_a: u32,
    body_b: u32,
    env_id: u32,
    local_a: [3]f32,
    local_b: [3]f32,
    rel_quat: [4]f32,
    compliance: f32,
) XPBDConstraint {
    return .{
        .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.weld) },
        .anchor_a = .{ local_a[0], local_a[1], local_a[2], compliance },
        .anchor_b = .{ local_b[0], local_b[1], local_b[2], 0 },
        .axis_target = rel_quat,
        .limits = .{ 0, 0, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Create a connect (distance between body points) constraint.
pub fn createConnectConstraint(
    body_a: u32,
    body_b: u32,
    env_id: u32,
    local_a: [3]f32,
    local_b: [3]f32,
    target_distance: f32,
    compliance: f32,
) XPBDConstraint {
    return .{
        .indices = .{ body_a, body_b, env_id, @intFromEnum(ConstraintType.connect) },
        .anchor_a = .{ local_a[0], local_a[1], local_a[2], compliance },
        .anchor_b = .{ local_b[0], local_b[1], local_b[2], 0 },
        .axis_target = .{ 0, 0, 0, target_distance },
        .limits = .{ 0, 0, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Create a joint equality constraint (coupled joints).
pub fn createJointEqualityConstraint(
    joint_a: u32,
    joint_b: u32,
    env_id: u32,
    polycoef: [5]f32,
    compliance: f32,
) XPBDConstraint {
    return .{
        .indices = .{ joint_a, joint_b, env_id, @intFromEnum(ConstraintType.joint_equality) },
        .anchor_a = .{ polycoef[0], polycoef[1], polycoef[2], compliance },
        .anchor_b = .{ polycoef[3], polycoef[4], 0, 0 },
        .axis_target = .{ 0, 0, 0, 0 },
        .limits = .{ 0, 0, 0, 0 },
        .state = .{ 0, 0, 0, 0 },
    };
}

/// Constraint buffer for GPU solver.
pub const ConstraintBuffer = struct {
    /// Maximum constraints per environment.
    max_constraints: u32,
    /// Number of environments.
    num_envs: u32,
    /// Constraint counts per type per environment.
    counts: []u32,
    /// Allocator.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_envs: u32, max_constraints: u32) !ConstraintBuffer {
        const counts = try allocator.alloc(u32, num_envs * @typeInfo(ConstraintType).@"enum".fields.len);
        @memset(counts, 0);

        return .{
            .max_constraints = max_constraints,
            .num_envs = num_envs,
            .counts = counts,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConstraintBuffer) void {
        self.allocator.free(self.counts);
    }

    /// Total buffer size in bytes.
    pub fn dataSize(self: *const ConstraintBuffer) usize {
        return self.num_envs * self.max_constraints * @sizeOf(XPBDConstraint);
    }

    /// Get linear index for constraint.
    pub fn index(self: *const ConstraintBuffer, env_id: u32, constraint_id: u32) u32 {
        return env_id * self.max_constraints + constraint_id;
    }
};

/// XPBD solver kernel parameters (for Metal dispatch).
pub const SolverKernelParams = extern struct {
    num_envs: u32 align(16),
    max_constraints: u32,
    num_bodies: u32,
    iteration: u32,

    dt: f32 align(16),
    inv_dt: f32,
    inv_dt_sq: f32,
    relaxation: f32,

    gravity: [4]f32 align(16),
};

/// Compute effective mass for a constraint.
/// w = m_a^-1 + m_b^-1 + (r_a × n)ᵀ I_a^-1 (r_a × n) + (r_b × n)ᵀ I_b^-1 (r_b × n)
pub fn computeEffectiveMass(
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_inertia_a: [3]f32,
    inv_inertia_b: [3]f32,
    r_a: [3]f32,
    r_b: [3]f32,
    normal: [3]f32,
) f32 {
    // r × n for each body
    const rn_a = cross(r_a, normal);
    const rn_b = cross(r_b, normal);

    // (r × n)ᵀ I^-1 (r × n) = Σ I_i^-1 * (r × n)_i²
    // For diagonal inertia tensor
    const angular_a = inv_inertia_a[0] * rn_a[0] * rn_a[0] +
        inv_inertia_a[1] * rn_a[1] * rn_a[1] +
        inv_inertia_a[2] * rn_a[2] * rn_a[2];

    const angular_b = inv_inertia_b[0] * rn_b[0] * rn_b[0] +
        inv_inertia_b[1] * rn_b[1] * rn_b[1] +
        inv_inertia_b[2] * rn_b[2] * rn_b[2];

    return inv_mass_a + inv_mass_b + angular_a + angular_b;
}

/// Solve a single XPBD constraint (CPU reference implementation).
pub fn solveConstraint(
    constraint: *XPBDConstraint,
    positions: [][3]f32,
    velocities: [][3]f32,
    inv_masses: []f32,
    dt: f32,
    iteration: u32,
) void {
    const body_a = constraint.getBodyA();
    const body_b = constraint.getBodyB();
    const constraint_type = constraint.getType();

    // Compute constraint violation C
    var violation: f32 = 0;
    var gradient: [3]f32 = .{ 0, 0, 0 };

    switch (constraint_type) {
        .contact_normal => {
            // C = penetration (stored in state[2])
            violation = constraint.state[2];
            gradient = .{
                constraint.axis_target[0],
                constraint.axis_target[1],
                constraint.axis_target[2],
            };
        },
        .positional => {
            // C = |p_a + r_a - p_b - r_b|
            const world_a = positions[body_a];
            const world_b = positions[body_b];
            const diff = sub(world_a, world_b);
            violation = length(diff);
            if (violation > constants.EPSILON) {
                gradient = scale(diff, 1.0 / violation);
            }
        },
        else => {},
    }

    if (@abs(violation) < constants.EPSILON) return;

    // Compute effective mass
    const w = inv_masses[body_a] + inv_masses[body_b];
    if (w < constants.EPSILON) return;

    // XPBD update: λ = (-C - α̃ * λ_prev) / (w + α̃)
    const compliance = constraint.getCompliance();
    const alpha_tilde = compliance / (dt * dt);
    const lambda_prev = if (iteration == 0) 0.0 else constraint.getLambda();

    const delta_lambda = (-violation - alpha_tilde * lambda_prev) / (w + alpha_tilde);
    constraint.setLambda(lambda_prev + delta_lambda);

    // Apply position correction: Δx = w * ∇C * Δλ
    const impulse = scale(gradient, delta_lambda);

    positions[body_a] = add(positions[body_a], scale(impulse, inv_masses[body_a]));
    positions[body_b] = sub(positions[body_b], scale(impulse, inv_masses[body_b]));

    // Update velocities (implicit)
    const vel_correction = scale(impulse, 1.0 / dt);
    velocities[body_a] = add(velocities[body_a], scale(vel_correction, inv_masses[body_a]));
    velocities[body_b] = sub(velocities[body_b], scale(vel_correction, inv_masses[body_b]));
}

// Vector helpers

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

fn scale(v: [3]f32, s: f32) [3]f32 {
    return .{ v[0] * s, v[1] * s, v[2] * s };
}

fn length(v: [3]f32) f32 {
    return @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// Assign colors to constraints using greedy graph coloring.
/// Constraints sharing the same body get different colors.
/// Returns the number of colors used.
pub fn colorConstraints(constraints: []XPBDConstraint, num_bodies: u32, allocator: std.mem.Allocator) !u8 {
    const n = constraints.len;
    if (n == 0) return 0;

    // Build adjacency: for each body, track which constraints reference it
    var body_constraints = try allocator.alloc(std.ArrayListUnmanaged(u32), num_bodies);
    defer {
        for (body_constraints) |*list| {
            list.deinit(allocator);
        }
        allocator.free(body_constraints);
    }
    for (body_constraints) |*list| {
        list.* = .{};
    }

    // Populate body -> constraint mappings
    for (constraints, 0..) |c, i| {
        const body_a = c.getBodyA();
        const body_b = c.getBodyB();
        if (body_a < num_bodies) {
            try body_constraints[body_a].append(allocator, @intCast(i));
        }
        if (body_b < num_bodies and body_b != body_a) {
            try body_constraints[body_b].append(allocator, @intCast(i));
        }
    }

    // Greedy coloring
    var max_color: u8 = 0;
    var neighbor_colors = try allocator.alloc(bool, 256);
    defer allocator.free(neighbor_colors);

    for (constraints, 0..) |*c, i| {
        // Reset neighbor colors
        @memset(neighbor_colors, false);

        // Find colors used by neighboring constraints (those sharing a body)
        const body_a = c.getBodyA();
        const body_b = c.getBodyB();

        if (body_a < num_bodies) {
            for (body_constraints[body_a].items) |neighbor_idx| {
                if (neighbor_idx < i) {
                    const neighbor_color = constraints[neighbor_idx].getColor();
                    neighbor_colors[neighbor_color] = true;
                }
            }
        }

        if (body_b < num_bodies and body_b != body_a) {
            for (body_constraints[body_b].items) |neighbor_idx| {
                if (neighbor_idx < i) {
                    const neighbor_color = constraints[neighbor_idx].getColor();
                    neighbor_colors[neighbor_color] = true;
                }
            }
        }

        // Find smallest available color
        var color: u8 = 0;
        while (color < 255 and neighbor_colors[color]) {
            color += 1;
        }

        c.setColor(color);
        if (color > max_color) {
            max_color = color;
        }
    }

    return max_color + 1;
}

/// Sort constraints by color for efficient GPU dispatch.
/// After sorting, constraints of the same color are contiguous.
pub fn sortConstraintsByColor(constraints: []XPBDConstraint) void {
    std.mem.sort(XPBDConstraint, constraints, {}, struct {
        fn lessThan(_: void, a: XPBDConstraint, b: XPBDConstraint) bool {
            return a.getColor() < b.getColor();
        }
    }.lessThan);
}

/// Get the start index and count for each color group.
/// Returns array of (start, count) pairs indexed by color.
pub fn getColorRanges(constraints: []const XPBDConstraint, num_colors: u8, allocator: std.mem.Allocator) ![][2]u32 {
    var ranges = try allocator.alloc([2]u32, num_colors);
    @memset(ranges, .{ 0, 0 });

    if (constraints.len == 0) return ranges;

    var current_color: u8 = constraints[0].getColor();
    var start: u32 = 0;

    for (constraints, 0..) |c, i| {
        const color = c.getColor();
        if (color != current_color) {
            ranges[current_color] = .{ start, @intCast(i - start) };
            current_color = color;
            start = @intCast(i);
        }
    }
    // Final color group
    ranges[current_color] = .{ start, @intCast(constraints.len - start) };

    return ranges;
}
