//! Joint types and constraint definitions for articulated bodies.

const std = @import("std");
const constants = @import("constants.zig");

/// Joint type enumeration.
pub const JointType = enum(u8) {
    /// Fixed joint - no degrees of freedom.
    fixed = 0,
    /// Hinge/revolute joint - 1 rotational DOF.
    revolute = 1,
    /// Prismatic/slide joint - 1 translational DOF.
    prismatic = 2,
    /// Ball/spherical joint - 3 rotational DOF.
    ball = 3,
    /// Free joint - 6 DOF (no constraint).
    free = 4,
    /// Universal joint - 2 rotational DOF.
    universal = 5,
};

/// Joint definition (scene description).
pub const JointDef = struct {
    /// Unique name for this joint.
    name: []const u8 = "",
    /// Type of joint.
    joint_type: JointType = .revolute,
    /// Parent body index.
    parent_body: u32 = 0,
    /// Child body index.
    child_body: u32 = 0,
    /// Anchor point in parent body local coordinates.
    anchor_parent: [3]f32 = .{ 0, 0, 0 },
    /// Anchor point in child body local coordinates.
    anchor_child: [3]f32 = .{ 0, 0, 0 },
    /// Joint axis in parent body local coordinates (for revolute/prismatic).
    axis: [3]f32 = .{ 0, 0, 1 },
    /// Secondary axis for universal joints.
    axis2: [3]f32 = .{ 0, 1, 0 },
    /// Lower limit (radians for revolute, meters for prismatic).
    limit_lower: f32 = -std.math.inf(f32),
    /// Upper limit.
    limit_upper: f32 = std.math.inf(f32),
    /// Joint damping coefficient.
    damping: f32 = constants.DEFAULT_JOINT_DAMPING,
    /// Joint stiffness (for spring-like behavior).
    stiffness: f32 = constants.DEFAULT_JOINT_STIFFNESS,
    /// Armature inertia (added to motor).
    armature: f32 = constants.DEFAULT_JOINT_ARMATURE,
    /// Reference position (equilibrium point).
    ref_position: f32 = 0.0,
    /// Enable limits.
    limited: bool = false,
    /// Enable motor/actuator.
    actuated: bool = false,

    /// Get degrees of freedom for this joint type.
    pub fn dof(self: *const JointDef) u32 {
        return switch (self.joint_type) {
            .fixed => 0,
            .revolute => 1,
            .prismatic => 1,
            .ball => 3,
            .free => 6,
            .universal => 2,
        };
    }

    /// Check if joint has limits.
    pub fn hasLimits(self: *const JointDef) bool {
        return self.limited and self.limit_lower < self.limit_upper;
    }

    /// Clamp position to limits.
    pub fn clampPosition(self: *const JointDef, pos: f32) f32 {
        if (!self.hasLimits()) return pos;
        return std.math.clamp(pos, self.limit_lower, self.limit_upper);
    }

    /// Calculate limit violation (positive if outside limits).
    pub fn limitViolation(self: *const JointDef, pos: f32) f32 {
        if (!self.hasLimits()) return 0.0;
        if (pos < self.limit_lower) return self.limit_lower - pos;
        if (pos > self.limit_upper) return pos - self.limit_upper;
        return 0.0;
    }
};

/// Joint state for GPU simulation (SoA layout fields).
pub const JointStateFields = struct {
    /// Joint position (angle for revolute, distance for prismatic).
    pub const Position = f32;
    /// Joint velocity.
    pub const Velocity = f32;
    /// Applied torque/force from actuator.
    pub const Torque = f32;
    /// Constraint lambda (Lagrange multiplier).
    pub const Lambda = f32;
};

/// Actuator definition.
pub const ActuatorDef = struct {
    /// Actuator name.
    name: []const u8 = "",
    /// Target joint index.
    joint: u32 = 0,
    /// Control range minimum.
    ctrl_min: f32 = -1.0,
    /// Control range maximum.
    ctrl_max: f32 = 1.0,
    /// Force/torque range minimum.
    force_min: f32 = -std.math.inf(f32),
    /// Force/torque range maximum.
    force_max: f32 = std.math.inf(f32),
    /// Gear ratio (torque = gear * control).
    gear: f32 = 1.0,
    /// Actuator type.
    actuator_type: ActuatorType = .motor,
    /// Position gain for position actuators.
    kp: f32 = 0.0,
    /// Velocity gain.
    kv: f32 = 0.0,

    /// Map control input to torque.
    pub fn controlToTorque(self: *const ActuatorDef, ctrl: f32, joint_pos: f32, joint_vel: f32) f32 {
        const clamped = std.math.clamp(ctrl, self.ctrl_min, self.ctrl_max);

        var torque: f32 = switch (self.actuator_type) {
            .motor => clamped * self.gear,
            .position => self.kp * (clamped - joint_pos) - self.kv * joint_vel,
            .velocity => self.kv * (clamped - joint_vel),
        };

        torque = std.math.clamp(torque, self.force_min, self.force_max);
        return torque;
    }
};

pub const ActuatorType = enum(u8) {
    /// Direct torque control.
    motor = 0,
    /// Position servo.
    position = 1,
    /// Velocity servo.
    velocity = 2,
};

/// Joint constraint solver data.
pub const JointConstraint = struct {
    /// Constraint type.
    constraint_type: ConstraintType,
    /// Body indices.
    body_a: u32,
    body_b: u32,
    /// Local anchor points.
    local_anchor_a: [3]f32,
    local_anchor_b: [3]f32,
    /// Constraint axis in world space.
    axis: [3]f32,
    /// Constraint parameters.
    params: ConstraintParams,

    /// Accumulated impulse for warm starting.
    impulse: f32 = 0.0,
    /// Position correction impulse.
    position_impulse: f32 = 0.0,
};

pub const ConstraintType = enum(u8) {
    /// Distance constraint (keep points at fixed distance).
    distance = 0,
    /// Point constraint (keep points coincident).
    point = 1,
    /// Hinge constraint (allow rotation around axis only).
    hinge = 2,
    /// Slider constraint (allow translation along axis only).
    slider = 3,
    /// Cone limit (limit rotation angle from axis).
    cone_limit = 4,
    /// Angular limit (limit rotation around axis).
    angular_limit = 5,
    /// Linear limit (limit translation along axis).
    linear_limit = 6,
};

pub const ConstraintParams = struct {
    /// Target value (distance, angle, etc.).
    target: f32 = 0.0,
    /// Compliance (inverse stiffness).
    compliance: f32 = 0.0,
    /// Damping.
    damping: f32 = 0.0,
    /// Lower limit.
    lower: f32 = 0.0,
    /// Upper limit.
    upper: f32 = 0.0,
};

/// Decompose joint into primitive constraints.
pub fn decomposeJoint(joint: *const JointDef, allocator: std.mem.Allocator) ![]JointConstraint {
    var constraints = std.ArrayList(JointConstraint).init(allocator);

    switch (joint.joint_type) {
        .fixed => {
            // Fixed joint = 3 point constraints + 3 angular constraints
            // For simplicity, use a single rigid constraint
            try constraints.append(.{
                .constraint_type = .point,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = .{ 0, 0, 0 },
                .params = .{},
            });
        },
        .revolute => {
            // Point constraint at anchor
            try constraints.append(.{
                .constraint_type = .point,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = .{ 0, 0, 0 },
                .params = .{},
            });
            // Hinge constraint around axis
            try constraints.append(.{
                .constraint_type = .hinge,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = joint.axis,
                .params = .{
                    .damping = joint.damping,
                },
            });
            // Angular limits if enabled
            if (joint.hasLimits()) {
                try constraints.append(.{
                    .constraint_type = .angular_limit,
                    .body_a = joint.parent_body,
                    .body_b = joint.child_body,
                    .local_anchor_a = joint.anchor_parent,
                    .local_anchor_b = joint.anchor_child,
                    .axis = joint.axis,
                    .params = .{
                        .lower = joint.limit_lower,
                        .upper = joint.limit_upper,
                    },
                });
            }
        },
        .prismatic => {
            // Slider constraint along axis
            try constraints.append(.{
                .constraint_type = .slider,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = joint.axis,
                .params = .{
                    .damping = joint.damping,
                },
            });
            // Linear limits if enabled
            if (joint.hasLimits()) {
                try constraints.append(.{
                    .constraint_type = .linear_limit,
                    .body_a = joint.parent_body,
                    .body_b = joint.child_body,
                    .local_anchor_a = joint.anchor_parent,
                    .local_anchor_b = joint.anchor_child,
                    .axis = joint.axis,
                    .params = .{
                        .lower = joint.limit_lower,
                        .upper = joint.limit_upper,
                    },
                });
            }
        },
        .ball => {
            // Point constraint only (free rotation)
            try constraints.append(.{
                .constraint_type = .point,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = .{ 0, 0, 0 },
                .params = .{},
            });
            // Cone limit if specified
            if (joint.hasLimits()) {
                try constraints.append(.{
                    .constraint_type = .cone_limit,
                    .body_a = joint.parent_body,
                    .body_b = joint.child_body,
                    .local_anchor_a = joint.anchor_parent,
                    .local_anchor_b = joint.anchor_child,
                    .axis = joint.axis,
                    .params = .{
                        .upper = joint.limit_upper,
                    },
                });
            }
        },
        .free => {
            // No constraints
        },
        .universal => {
            // Point constraint
            try constraints.append(.{
                .constraint_type = .point,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = .{ 0, 0, 0 },
                .params = .{},
            });
            // Two perpendicular hinge constraints
            try constraints.append(.{
                .constraint_type = .hinge,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = joint.axis,
                .params = .{},
            });
            try constraints.append(.{
                .constraint_type = .hinge,
                .body_a = joint.parent_body,
                .body_b = joint.child_body,
                .local_anchor_a = joint.anchor_parent,
                .local_anchor_b = joint.anchor_child,
                .axis = joint.axis2,
                .params = .{},
            });
        },
    }

    return constraints.toOwnedSlice();
}
