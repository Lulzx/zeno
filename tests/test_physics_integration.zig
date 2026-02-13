//! Integration tests for physics system.
//! Tests XPBD constraint solver, joint decomposition, and kinematic body handling.

const std = @import("std");
const testing = std.testing;

const body = @import("zeno").physics.body;
const joint = @import("zeno").physics.joint;
const xpbd = @import("zeno").physics.xpbd;
const constants = @import("zeno").physics.constants;

// ============================================================================
// Joint Decomposition Tests
// ============================================================================

test "revolute joint decomposes to point and hinge constraints" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .revolute,
        .parent_body = 0,
        .child_body = 1,
        .anchor_parent = .{ 0, 0, 1 },
        .anchor_child = .{ 0, 0, -1 },
        .axis = .{ 0, 0, 1 },
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Revolute should produce point + hinge constraints
    try testing.expectEqual(@as(usize, 2), constraints.len);
    try testing.expectEqual(joint.ConstraintType.point, constraints[0].constraint_type);
    try testing.expectEqual(joint.ConstraintType.hinge, constraints[1].constraint_type);
}

test "revolute joint with limits produces angular limit constraint" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .revolute,
        .parent_body = 0,
        .child_body = 1,
        .axis = .{ 0, 0, 1 },
        .limit_lower = -1.57,
        .limit_upper = 1.57,
        .limited = true,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Revolute with limits: point + hinge + angular_limit
    try testing.expectEqual(@as(usize, 3), constraints.len);
    try testing.expectEqual(joint.ConstraintType.point, constraints[0].constraint_type);
    try testing.expectEqual(joint.ConstraintType.hinge, constraints[1].constraint_type);
    try testing.expectEqual(joint.ConstraintType.angular_limit, constraints[2].constraint_type);

    // Check limit values
    try testing.expectApproxEqAbs(@as(f32, -1.57), constraints[2].params.lower, 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.57), constraints[2].params.upper, 0.01);
}

test "fixed joint decomposes to weld constraint" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .fixed,
        .parent_body = 0,
        .child_body = 1,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Fixed joint should produce single weld constraint
    try testing.expectEqual(@as(usize, 1), constraints.len);
    try testing.expectEqual(joint.ConstraintType.weld, constraints[0].constraint_type);
}

test "ball joint decomposes to point constraint only" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .ball,
        .parent_body = 0,
        .child_body = 1,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Ball joint: just point constraint (free rotation)
    try testing.expectEqual(@as(usize, 1), constraints.len);
    try testing.expectEqual(joint.ConstraintType.point, constraints[0].constraint_type);
}

test "ball joint with cone limit" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .ball,
        .parent_body = 0,
        .child_body = 1,
        .axis = .{ 0, 0, 1 },
        .limit_upper = 0.5, // 0.5 radians cone angle
        .limited = true,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Ball with limits: point + cone_limit
    try testing.expectEqual(@as(usize, 2), constraints.len);
    try testing.expectEqual(joint.ConstraintType.point, constraints[0].constraint_type);
    try testing.expectEqual(joint.ConstraintType.cone_limit, constraints[1].constraint_type);
}

test "prismatic joint decomposes to slider constraint" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .prismatic,
        .parent_body = 0,
        .child_body = 1,
        .axis = .{ 1, 0, 0 },
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Prismatic: slider constraint
    try testing.expectEqual(@as(usize, 1), constraints.len);
    try testing.expectEqual(joint.ConstraintType.slider, constraints[0].constraint_type);
}

test "prismatic joint with limits produces linear limit constraint" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .prismatic,
        .parent_body = 0,
        .child_body = 1,
        .axis = .{ 1, 0, 0 },
        .limit_lower = -1.0,
        .limit_upper = 1.0,
        .limited = true,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Prismatic with limits: slider + linear_limit
    try testing.expectEqual(@as(usize, 2), constraints.len);
    try testing.expectEqual(joint.ConstraintType.slider, constraints[0].constraint_type);
    try testing.expectEqual(joint.ConstraintType.linear_limit, constraints[1].constraint_type);
}

test "universal joint produces point and two hinge constraints" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .universal,
        .parent_body = 0,
        .child_body = 1,
        .axis = .{ 1, 0, 0 },
        .axis2 = .{ 0, 1, 0 },
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Universal: point + 2 hinges
    try testing.expectEqual(@as(usize, 3), constraints.len);
    try testing.expectEqual(joint.ConstraintType.point, constraints[0].constraint_type);
    try testing.expectEqual(joint.ConstraintType.hinge, constraints[1].constraint_type);
    try testing.expectEqual(joint.ConstraintType.hinge, constraints[2].constraint_type);
}

test "free joint produces no constraints" {
    const allocator = testing.allocator;

    const j = joint.JointDef{
        .joint_type = .free,
        .parent_body = 0,
        .child_body = 1,
    };

    const constraints = try joint.decomposeJoint(&j, allocator);
    defer allocator.free(constraints);

    // Free joint: no constraints
    try testing.expectEqual(@as(usize, 0), constraints.len);
}

// ============================================================================
// XPBD Constraint Structure Tests
// ============================================================================

test "XPBD constraint size and alignment" {
    // Ensure GPU-friendly layout
    try testing.expectEqual(@as(usize, 96), @sizeOf(xpbd.XPBDConstraint));
    try testing.expectEqual(@as(usize, 16), @alignOf(xpbd.XPBDConstraint));
}

test "create positional constraint" {
    const c = xpbd.createPositionalConstraint(
        0,
        1,
        0, // env_id
        .{ 0, 0, 1 },
        .{ 0, 0, -1 },
        0.0, // compliance (rigid)
    );

    try testing.expectEqual(@as(u32, 0), c.getBodyA());
    try testing.expectEqual(@as(u32, 1), c.getBodyB());
    try testing.expectEqual(@as(u32, 0), c.getEnvId());
    try testing.expectEqual(xpbd.ConstraintType.positional, c.getType());
    try testing.expectApproxEqAbs(@as(f32, 0.0), c.getCompliance(), 0.001);
}

test "create angular constraint" {
    const c = xpbd.createAngularConstraint(
        0,
        1,
        0,
        .{ 0, 0, 1 }, // axis in body A
        .{ 0, 0, 1 }, // axis in body B
        0.0,
        0.01, // damping
    );

    try testing.expectEqual(xpbd.ConstraintType.angular, c.getType());
    try testing.expectApproxEqAbs(@as(f32, 0.01), c.getDamping(), 0.001);
}

test "create weld constraint" {
    const c = xpbd.createWeldConstraint(
        0,
        1,
        0,
        .{ 0, 0, 0.5 },
        .{ 0, 0, -0.5 },
        .{ 0, 0, 0, 1 }, // identity relative quaternion
        0.0,
    );

    try testing.expectEqual(xpbd.ConstraintType.weld, c.getType());
    // Check relative quaternion stored in axis_target
    try testing.expectApproxEqAbs(@as(f32, 0.0), c.axis_target[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.axis_target[3], 0.001);
}

test "create angular limit constraint" {
    const c = xpbd.createAngularLimitConstraint(
        0,
        1,
        0,
        .{ 0, 0, 1 }, // axis
        -1.57, // lower
        1.57, // upper
        0.0,
    );

    try testing.expectEqual(xpbd.ConstraintType.angular_limit, c.getType());
    try testing.expectApproxEqAbs(@as(f32, -1.57), c.limits[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.57), c.limits[1], 0.01);
}

test "create contact constraint" {
    const c = xpbd.createContactConstraint(
        0,
        1,
        0,
        .{ 1, 0, 0 }, // position
        .{ 0, 0, 1 }, // normal
        0.05, // penetration
        0.5, // friction
        0.3, // restitution
        1e-9, // compliance
    );

    try testing.expectEqual(xpbd.ConstraintType.contact_normal, c.getType());
    try testing.expectApproxEqAbs(@as(f32, 0.05), c.state[2], 0.001); // penetration stored in state
    try testing.expectApproxEqAbs(@as(f32, 0.5), c.limits[2], 0.001); // friction
    try testing.expectApproxEqAbs(@as(f32, 0.3), c.limits[3], 0.001); // restitution
}

test "create tendon constraint" {
    const c = xpbd.createTendonConstraint(
        0, // tendon_id
        0, // env_id
        1.0, // rest_length
        100.0, // stiffness
        0.1, // damping
        0.5, // lower
        1.5, // upper
    );

    try testing.expectEqual(xpbd.ConstraintType.tendon, c.getType());
    try testing.expectApproxEqAbs(@as(f32, 0.01), c.getCompliance(), 0.001); // 1/100
}

test "create connect constraint" {
    const c = xpbd.createConnectConstraint(
        0,
        1,
        0,
        .{ 0, 0, 0 },
        .{ 0, 0, 0 },
        2.0, // target distance
        0.0,
    );

    try testing.expectEqual(xpbd.ConstraintType.connect, c.getType());
    try testing.expectApproxEqAbs(@as(f32, 2.0), c.getTarget(), 0.001);
}

// ============================================================================
// Body Definition Tests for Kinematic Bodies
// ============================================================================

test "kinematic body has zero inverse mass" {
    var def = body.BodyDef{};
    def.body_type = .kinematic;
    def.mass = 10.0;

    // Kinematic bodies should have infinite mass (inv_mass = 0)
    try testing.expectApproxEqAbs(@as(f32, 0.0), def.invMass(), 0.001);
}

test "kinematic body has zero inverse inertia" {
    var def = body.BodyDef{};
    def.body_type = .kinematic;
    def.mass = 10.0;
    def.inertia = .{ 1, 1, 1 };

    const inv_inertia = def.invInertia();

    // Kinematic bodies should have infinite inertia
    try testing.expectApproxEqAbs(@as(f32, 0.0), inv_inertia[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), inv_inertia[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), inv_inertia[2], 0.001);
}

test "static body has zero inverse mass" {
    var def = body.BodyDef{};
    def.body_type = .static;
    def.mass = 10.0;

    try testing.expectApproxEqAbs(@as(f32, 0.0), def.invMass(), 0.001);
}

test "dynamic body has finite inverse mass" {
    var def = body.BodyDef{};
    def.body_type = .dynamic;
    def.mass = 2.0;

    try testing.expectApproxEqAbs(@as(f32, 0.5), def.invMass(), 0.001);
}

// ============================================================================
// XPBD Config Tests
// ============================================================================

test "RL config preset" {
    const config = xpbd.XPBDConfig.forRL();

    try testing.expectEqual(@as(u32, 4), config.iterations);
    try testing.expectEqual(@as(u32, 1), config.substeps);
    try testing.expect(config.warm_start);
    try testing.expectApproxEqAbs(@as(f32, 1e-9), config.contact_compliance, 1e-10);
}

test "accuracy config preset" {
    const config = xpbd.XPBDConfig.forAccuracy();

    try testing.expectEqual(@as(u32, 8), config.iterations);
    try testing.expectEqual(@as(u32, 4), config.substeps);
    try testing.expect(config.warm_start);
    try testing.expectApproxEqAbs(@as(f32, 0.0), config.contact_compliance, 1e-10);
}

// ============================================================================
// XPBD Solve Data Tests
// ============================================================================

test "XPBD solve data initialization" {
    const dt = 0.001;
    const compliance = 1e-6;
    const gravity = [3]f32{ 0, 0, -9.81 };

    const solve_data = xpbd.XPBDSolveData.init(dt, compliance, gravity);

    try testing.expectApproxEqAbs(@as(f32, 0.001), solve_data.dt, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), solve_data.alpha_tilde, 0.1); // 1e-6 / (0.001)^2 = 1.0
    try testing.expectApproxEqAbs(@as(f32, -9.81), solve_data.gravity[2], 0.01);
}

// ============================================================================
// Effective Mass Computation Tests
// ============================================================================

test "effective mass for equal masses" {
    const inv_mass_a: f32 = 1.0;
    const inv_mass_b: f32 = 1.0;
    const inv_inertia_a = [3]f32{ 1, 1, 1 };
    const inv_inertia_b = [3]f32{ 1, 1, 1 };
    const r_a = [3]f32{ 0, 0, 0 }; // At center of mass
    const r_b = [3]f32{ 0, 0, 0 };
    const normal = [3]f32{ 1, 0, 0 };

    const w = xpbd.computeEffectiveMass(
        inv_mass_a,
        inv_mass_b,
        inv_inertia_a,
        inv_inertia_b,
        r_a,
        r_b,
        normal,
    );

    // At center of mass, no angular contribution: w = inv_mass_a + inv_mass_b = 2
    try testing.expectApproxEqAbs(@as(f32, 2.0), w, 0.01);
}

test "effective mass with offset contact point" {
    const inv_mass_a: f32 = 1.0;
    const inv_mass_b: f32 = 1.0;
    const inv_inertia_a = [3]f32{ 1, 1, 1 };
    const inv_inertia_b = [3]f32{ 1, 1, 1 };
    const r_a = [3]f32{ 0, 1, 0 }; // Offset from center
    const r_b = [3]f32{ 0, -1, 0 };
    const normal = [3]f32{ 1, 0, 0 };

    const w = xpbd.computeEffectiveMass(
        inv_mass_a,
        inv_mass_b,
        inv_inertia_a,
        inv_inertia_b,
        r_a,
        r_b,
        normal,
    );

    // With offset, angular contribution increases effective mass
    // r_a x n = (0,1,0) x (1,0,0) = (0,0,-1)
    // angular_a = 1 * 0^2 + 1 * 0^2 + 1 * 1^2 = 1
    // Total: 2 + 1 + 1 = 4
    try testing.expectApproxEqAbs(@as(f32, 4.0), w, 0.01);
}

test "effective mass with static body" {
    const inv_mass_a: f32 = 0.0; // Static
    const inv_mass_b: f32 = 1.0;
    const inv_inertia_a = [3]f32{ 0, 0, 0 };
    const inv_inertia_b = [3]f32{ 1, 1, 1 };
    const r_a = [3]f32{ 0, 0, 0 };
    const r_b = [3]f32{ 0, 0, 0 };
    const normal = [3]f32{ 0, 0, 1 };

    const w = xpbd.computeEffectiveMass(
        inv_mass_a,
        inv_mass_b,
        inv_inertia_a,
        inv_inertia_b,
        r_a,
        r_b,
        normal,
    );

    // Static body contributes zero: w = inv_mass_b = 1
    try testing.expectApproxEqAbs(@as(f32, 1.0), w, 0.01);
}

// ============================================================================
// Constraint Buffer Tests
// ============================================================================

test "constraint buffer initialization" {
    const allocator = testing.allocator;
    const num_envs: u32 = 4;
    const max_constraints: u32 = 32;

    var buffer = try xpbd.ConstraintBuffer.init(allocator, num_envs, max_constraints);
    defer buffer.deinit();

    try testing.expectEqual(num_envs, buffer.num_envs);
    try testing.expectEqual(max_constraints, buffer.max_constraints);

    // Data size should be num_envs * max_constraints * sizeof(XPBDConstraint)
    const expected_size = num_envs * max_constraints * @sizeOf(xpbd.XPBDConstraint);
    try testing.expectEqual(expected_size, buffer.dataSize());
}

test "constraint buffer indexing" {
    const allocator = testing.allocator;
    const num_envs: u32 = 4;
    const max_constraints: u32 = 32;

    var buffer = try xpbd.ConstraintBuffer.init(allocator, num_envs, max_constraints);
    defer buffer.deinit();

    // Test linear indexing
    try testing.expectEqual(@as(u32, 0), buffer.index(0, 0));
    try testing.expectEqual(@as(u32, 1), buffer.index(0, 1));
    try testing.expectEqual(@as(u32, 32), buffer.index(1, 0));
    try testing.expectEqual(@as(u32, 33), buffer.index(1, 1));
    try testing.expectEqual(@as(u32, 64), buffer.index(2, 0));
}

// ============================================================================
// Actuator Tests
// ============================================================================

test "motor actuator direct torque" {
    const act = joint.ActuatorDef{
        .actuator_type = .motor,
        .gear = 10.0,
        .ctrl_min = -1.0,
        .ctrl_max = 1.0,
        .force_min = -100.0,
        .force_max = 100.0,
    };

    // Motor: torque = ctrl * gear
    const torque = act.controlToTorque(0.5, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 5.0), torque, 0.01);
}

test "position servo actuator" {
    const act = joint.ActuatorDef{
        .actuator_type = .position,
        .kp = 100.0,
        .kv = 10.0,
        .ctrl_min = -3.14,
        .ctrl_max = 3.14,
        .force_min = -50.0,
        .force_max = 50.0,
    };

    // Position: torque = kp * (ctrl - pos) - kv * vel
    const joint_pos: f32 = 0.5;
    const joint_vel: f32 = 0.1;
    const target: f32 = 1.0;

    const torque = act.controlToTorque(target, joint_pos, joint_vel);
    // Expected: 100 * (1.0 - 0.5) - 10 * 0.1 = 50 - 1 = 49
    try testing.expectApproxEqAbs(@as(f32, 49.0), torque, 0.1);
}

test "velocity servo actuator" {
    const act = joint.ActuatorDef{
        .actuator_type = .velocity,
        .kv = 10.0,
        .ctrl_min = -10.0,
        .ctrl_max = 10.0,
        .force_min = -50.0,
        .force_max = 50.0,
    };

    // Velocity: torque = kv * (ctrl - vel)
    const joint_vel: f32 = 2.0;
    const target_vel: f32 = 5.0;

    const torque = act.controlToTorque(target_vel, 0, joint_vel);
    // Expected: 10 * (5.0 - 2.0) = 30
    try testing.expectApproxEqAbs(@as(f32, 30.0), torque, 0.1);
}

test "actuator torque clamping" {
    const act = joint.ActuatorDef{
        .actuator_type = .motor,
        .gear = 1000.0, // High gear
        .ctrl_min = -1.0,
        .ctrl_max = 1.0,
        .force_min = -50.0,
        .force_max = 50.0,
    };

    // Without clamping: 1.0 * 1000 = 1000
    // With clamping: 50.0
    const torque = act.controlToTorque(1.0, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), torque, 0.1);
}

test "actuator control input clamping" {
    const act = joint.ActuatorDef{
        .actuator_type = .motor,
        .gear = 10.0,
        .ctrl_min = -1.0,
        .ctrl_max = 1.0,
        .force_min = -100.0,
        .force_max = 100.0,
    };

    // Control input 5.0 should be clamped to 1.0
    const torque = act.controlToTorque(5.0, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 10.0), torque, 0.1);
}

// Regression test for motor actuator bug where compute_forces kernel zeroed
// the body torques buffer, erasing actuator torques applied by apply_joint_forces.
// See: src/shaders/all_shaders.metal compute_forces kernel.
test "motor actuator with high gear produces nonzero torque" {
    const act = joint.ActuatorDef{
        .actuator_type = .motor,
        .gear = 100.0,
        .ctrl_min = -1.0,
        .ctrl_max = 1.0,
        .force_min = -std.math.inf(f32),
        .force_max = std.math.inf(f32),
    };

    // With ctrl=1.0, gear=100: torque should be 100
    const torque = act.controlToTorque(1.0, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 100.0), torque, 0.01);
    try testing.expect(torque > 0.0); // Must be nonzero

    // With ctrl=0.5: torque should be 50
    const torque2 = act.controlToTorque(0.5, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), torque2, 0.01);

    // Negative control should produce negative torque
    const torque3 = act.controlToTorque(-1.0, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, -100.0), torque3, 0.01);
    try testing.expect(torque3 < 0.0);
}
