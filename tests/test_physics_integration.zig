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

// ============================================================================
// GPU Pipeline Regression Tests
// ============================================================================

test "Metal box-plane contact composes body and local geom transforms" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="oriented-box-contact">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <geom name="ground" type="plane" size="5 5 0.1"/>
        \\    <body name="box" pos="0 0 0.2" quat="0.9238795 0 0.3826834 0">
        \\      <freejoint name="root"/>
        \\      <geom name="offset-box" type="box" pos="0.1 0 0" size="0.2 0.1 0.3" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 2,
        .timestep = 0.001,
        .substeps = 1,
        .max_contacts_per_env = 8,
    });
    defer world.deinit();

    try world.step(&.{}, 1);
    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        // Rotating the 0.1 local +X offset by body +45° around Y lowers the
        // box center, while projected oriented half-extents reach farther.
        try testing.expectApproxEqAbs(@as(f32, 0.224264), contact.position_pen[3], 1e-4);
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectEqual([4]u32{ 0, 1, 0, 1 }, contact.indices);
        const box_body = world.state.bodyIndex(@intCast(env_id), 1);
        try testing.expect(world.state.getPositions()[box_body][2] > 0.2);
    }
}

test "Metal sphere contacts use solver-consistent normals" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="sphere-normal">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="left" pos="-0.25 0 0">
        \\      <freejoint name="left-root"/>
        \\      <geom type="sphere" size="0.4" mass="1"/>
        \\    </body>
        \\    <body name="right" pos="0.25 0 0">
        \\      <freejoint name="right-root"/>
        \\      <geom type="sphere" size="0.4" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[0], 1e-5);
        const left = world.state.bodyIndex(@intCast(env_id), 1);
        const right = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[left][0] < -0.25);
        try testing.expect(world.state.getPositions()[right][0] > 0.25);
    }
}

test "Metal sphere-capsule contact uses MJCF fromto orientation" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="sphere-capsule">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="bar" pos="0 0 0.1">
        \\      <geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.1" mass="0"/>
        \\    </body>
        \\    <body name="ball" pos="0 0 0.25">
        \\      <freejoint name="ball-root"/>
        \\      <geom type="sphere" size="0.2" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.15), contact.position_pen[3], 1e-5);
        const ball = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[ball][2] > 0.25);
    }
}

test "Metal capsule-capsule contact resolves oriented segments" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="capsule-capsule">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="lower" pos="0 0 0.1">
        \\      <geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.1" mass="0"/>
        \\    </body>
        \\    <body name="upper" pos="0 0 0.25">
        \\      <freejoint name="upper-root"/>
        \\      <geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.1" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.05), contact.position_pen[3], 1e-5);
        const upper = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[upper][2] > 0.25);
    }
}

test "Metal sphere-box contact resolves an oriented box" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="sphere-box">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="box" quat="0.9238795 0 0.3826834 0">
        \\      <geom type="box" size="0.5 0.2 0.1" mass="0"/>
        \\    </body>
        \\    <body name="sphere" pos="0.155563 0 0.155563">
        \\      <freejoint name="sphere-root"/>
        \\      <geom type="sphere" size="0.15" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    const outward = @as(f32, 0.70710677);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(-outward, contact.normal_friction[0], 1e-5);
        try testing.expectApproxEqAbs(-outward, contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.03), contact.position_pen[3], 1e-4);
        const sphere = world.state.bodyIndex(@intCast(env_id), 2);
        const sphere_pos = world.state.getPositions()[sphere];
        try testing.expect(sphere_pos[0] * outward + sphere_pos[2] * outward > 0.22);
    }
}

test "Metal sphere-cylinder contact resolves an oriented finite cylinder" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="sphere-cylinder">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="cylinder" quat="0.7071068 0 0.7071068 0">
        \\      <geom type="cylinder" size="0.1 0.4" mass="0"/>
        \\    </body>
        \\    <body name="sphere" pos="0 0 0.15">
        \\      <freejoint name="sphere-root"/>
        \\      <geom type="sphere" size="0.1" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.05), contact.position_pen[3], 1e-5);
        const sphere_body = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[sphere_body][2] > 0.15);
    }
}

test "Metal cylinder-plane contact uses oriented cylinder support" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="cylinder-plane">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <geom name="ground" type="plane" size="5 5 0.1"/>
        \\    <body name="cylinder" pos="0 0 0.05" quat="0.7071068 0 0.7071068 0">
        \\      <freejoint name="root"/>
        \\      <geom type="cylinder" size="0.1 0.4" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.05), contact.position_pen[3], 1e-5);
        const cylinder_body = world.state.bodyIndex(@intCast(env_id), 1);
        try testing.expect(world.state.getPositions()[cylinder_body][2] > 0.05);
    }
}

test "bundled Pusher object cylinder contacts its table on Metal" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pusher.xml");
    const table_geom = scene.getGeomByName("table") orelse return error.MissingTableGeom;
    const object_geom = scene.getGeomByName("object") orelse return error.MissingObjectGeom;
    const object_body = scene.getBodyByName("object") orelse return error.MissingObjectBody;
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 2,
        .substeps = 1,
        .max_contacts_per_env = 64,
    });
    defer world.deinit();

    const actions = try allocator.alloc(f32, world.config.num_envs * world.params.num_actuators);
    defer allocator.free(actions);
    @memset(actions, 0);
    try world.step(actions, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        var found_object_table = false;
        const contact_base = env_id * world.config.max_contacts_per_env;
        for (contacts[contact_base .. contact_base + @min(counts[env_id], world.config.max_contacts_per_env)]) |contact| {
            const direct = contact.indices[2] == table_geom and contact.indices[3] == object_geom;
            const reverse = contact.indices[2] == object_geom and contact.indices[3] == table_geom;
            if ((direct or reverse) and contact.position_pen[3] >= 0) {
                found_object_table = true;
                break;
            }
        }
        try testing.expect(found_object_table);
        const object_index = world.state.bodyIndex(@intCast(env_id), object_body);
        try testing.expect(world.state.getPositions()[object_index][2] >= 0.029);
    }
}

test "Metal contact solver keeps a two-sphere stack bounded for 2000 steps" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="stack-stability">
        \\  <option timestep="0.002" gravity="0 0 -9.81"/>
        \\  <worldbody>
        \\    <geom type="plane" size="5 5 0.1" friction="1 0 0"/>
        \\    <body name="lower" pos="0 0 0.1">
        \\      <freejoint/><geom type="sphere" size="0.1" mass="1" friction="1 0 0"/>
        \\    </body>
        \\    <body name="upper" pos="0 0 0.3">
        \\      <freejoint/><geom type="sphere" size="0.1" mass="1" friction="1 0 0"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 8,
        .timestep = 0.002,
        .contact_iterations = 8,
        .substeps = 1,
        .max_contacts_per_env = 16,
    });
    defer world.deinit();

    var minimum_lower_z: f32 = 0.1;
    var minimum_upper_z: f32 = 0.3;
    var maximum_upper_z: f32 = 0.3;
    for (0..2000) |_| {
        try world.step(&.{}, 1);
        const positions = world.state.getPositions();
        for (0..world.config.num_envs) |env_id| {
            const lower = world.state.bodyIndex(@intCast(env_id), 1);
            const upper = world.state.bodyIndex(@intCast(env_id), 2);
            minimum_lower_z = @min(minimum_lower_z, positions[lower][2]);
            minimum_upper_z = @min(minimum_upper_z, positions[upper][2]);
            maximum_upper_z = @max(maximum_upper_z, positions[upper][2]);
        }
    }

    try testing.expect(minimum_lower_z > 0.09);
    try testing.expect(minimum_upper_z > 0.28);
    try testing.expect(maximum_upper_z < 0.301);
    const positions = world.state.getPositions();
    const velocities = world.state.getVelocities();
    for (0..world.config.num_envs) |env_id| {
        const lower = world.state.bodyIndex(@intCast(env_id), 1);
        const upper = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expectApproxEqAbs(@as(f32, 0.095), positions[lower][2], 0.002);
        try testing.expectApproxEqAbs(@as(f32, 0.289), positions[upper][2], 0.003);
        for (0..3) |axis| {
            try testing.expect(@abs(velocities[lower][axis]) < 0.01);
            try testing.expect(@abs(velocities[upper][axis]) < 0.01);
        }
    }
}

test "Metal Coulomb friction transitions a sliding sphere to rolling" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const frictionless_xml =
        \\<mujoco model="frictionless-slide">
        \\  <option timestep="0.002" gravity="0 0 -9.81"/>
        \\  <worldbody>
        \\    <geom type="plane" size="5 5 0.1" friction="0 0 0"/>
        \\    <body pos="0 0 0.1"><freejoint/>
        \\      <geom type="sphere" size="0.1" mass="1" friction="0 0 0"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;
    const friction_xml =
        \\<mujoco model="frictional-slide">
        \\  <option timestep="0.002" gravity="0 0 -9.81"/>
        \\  <worldbody>
        \\    <geom type="plane" size="5 5 0.1" friction="1 0 0"/>
        \\    <body pos="0 0 0.1"><freejoint/>
        \\      <geom type="sphere" size="0.1" mass="1" friction="1 0 0"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;
    const frictionless_scene = try zeno.mjcf.parser.parseString(allocator, frictionless_xml);
    const friction_scene = try zeno.mjcf.parser.parseString(allocator, friction_xml);
    const config = zeno.WorldConfig{
        .num_envs = 4,
        .timestep = 0.002,
        .contact_iterations = 8,
        .substeps = 1,
        .max_contacts_per_env = 8,
    };
    var frictionless = try zeno.World.init(allocator, frictionless_scene, config);
    defer frictionless.deinit();
    var frictional = try zeno.World.init(allocator, friction_scene, config);
    defer frictional.deinit();
    for (0..config.num_envs) |env_id| {
        const body_index = frictionless.state.bodyIndex(@intCast(env_id), 1);
        frictionless.state.getVelocities()[body_index][0] = 1;
        frictional.state.getVelocities()[body_index][0] = 1;
    }

    try frictionless.step(&.{}, 1);
    try frictional.step(&.{}, 1);
    const first_rolling_velocity = frictional.state.getVelocities()[1][0];
    const first_rolling_acceleration = frictional.state.getAccelerations()[1];
    try testing.expectApproxEqAbs(
        (first_rolling_velocity - 1.0) / config.timestep,
        first_rolling_acceleration[0],
        1e-4,
    );
    try testing.expect(first_rolling_acceleration[0] < -1);
    try testing.expectApproxEqAbs(@as(f32, 0), first_rolling_acceleration[2], 1e-4);

    for (1..1000) |_| {
        try frictionless.step(&.{}, 1);
        try frictional.step(&.{}, 1);
    }
    const free_velocities = frictionless.state.getVelocities();
    const rolling_velocities = frictional.state.getVelocities();
    const rolling_angular = frictional.state.getAngularVelocities();
    for (0..config.num_envs) |env_id| {
        const body_index = frictionless.state.bodyIndex(@intCast(env_id), 1);
        try testing.expect(free_velocities[body_index][0] > 0.99);
        try testing.expect(rolling_velocities[body_index][0] > 0.65);
        try testing.expect(rolling_velocities[body_index][0] < 0.75);
        try testing.expect(rolling_angular[body_index][1] > 6.5);
        try testing.expectApproxEqAbs(
            rolling_velocities[body_index][0],
            rolling_angular[body_index][1] * 0.1,
            0.02,
        );
    }
}

test "driven Pendulum hinge and weld drift stay bounded for 5000 steps" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 8,
        .timestep = 0.002,
        .contact_iterations = 8,
        .substeps = 1,
        .max_contacts_per_env = 16,
    });
    defer world.deinit();

    const action_count = world.config.num_envs * world.params.num_actuators;
    const actions = try allocator.alloc(f32, action_count);
    defer allocator.free(actions);
    const initial_base = world.state.getPositions()[1];
    var maximum_base_drift: f32 = 0;
    var maximum_hinge_drift: f32 = 0;
    var maximum_weld_length_error: f32 = 0;

    for (0..5000) |step_index| {
        const phase = @as(f32, @floatFromInt(step_index)) * 0.01;
        @memset(actions, 2.0 * @sin(phase));
        try world.step(actions, 1);
        const positions = world.state.getPositions();
        const velocities = world.state.getVelocities();
        for (0..world.config.num_envs) |env_id| {
            const base = world.state.bodyIndex(@intCast(env_id), 1);
            const pole = world.state.bodyIndex(@intCast(env_id), 2);
            const bob = world.state.bodyIndex(@intCast(env_id), 3);
            var base_drift_squared: f32 = 0;
            var hinge_drift_squared: f32 = 0;
            var weld_length_squared: f32 = 0;
            for (0..3) |axis| {
                const base_delta = positions[base][axis] - initial_base[axis];
                const hinge_delta = positions[pole][axis] - positions[base][axis];
                const weld_delta = positions[bob][axis] - positions[pole][axis];
                base_drift_squared += base_delta * base_delta;
                hinge_drift_squared += hinge_delta * hinge_delta;
                weld_length_squared += weld_delta * weld_delta;
            }
            const base_drift = @sqrt(base_drift_squared);
            const hinge_drift = @sqrt(hinge_drift_squared);
            const weld_length = @sqrt(weld_length_squared);
            maximum_base_drift = @max(maximum_base_drift, base_drift);
            maximum_hinge_drift = @max(maximum_hinge_drift, hinge_drift);
            maximum_weld_length_error = @max(maximum_weld_length_error, @abs(weld_length - 1));
            for (0..3) |axis| {
                try testing.expect(std.math.isFinite(positions[bob][axis]));
                try testing.expect(std.math.isFinite(velocities[bob][axis]));
            }
        }
    }

    // These bounds describe this driven 1 m model; they are not a universal
    // rigidity claim for arbitrary joint graphs or timesteps.
    try testing.expect(maximum_base_drift < 0.01);
    try testing.expect(maximum_hinge_drift < 0.03);
    try testing.expect(maximum_weld_length_error < 0.006);
}

test "collision-free free body conserves bounded energy proxies for 10000 steps" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="free-body-energy">
        \\  <option timestep="0.002" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="free" pos="1 2 3"><freejoint/>
        \\      <geom type="sphere" size="0.2" mass="2" contype="0" conaffinity="0"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;
    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 8,
        .timestep = 0.002,
        .substeps = 1,
        .max_contacts_per_env = 4,
    });
    defer world.deinit();

    const initial_velocity = [3]f32{ 0.7, -0.4, 0.2 };
    const initial_angular_velocity = [3]f32{ 0.3, -0.2, 0.5 };
    const initial_position = world.state.getPositions()[1];
    for (0..world.config.num_envs) |env_id| {
        const body_index = world.state.bodyIndex(@intCast(env_id), 1);
        for (0..3) |axis| {
            world.state.getVelocities()[body_index][axis] = initial_velocity[axis];
            world.state.getAngularVelocities()[body_index][axis] = initial_angular_velocity[axis];
        }
    }

    var maximum_quaternion_norm_error: f32 = 0;
    for (0..10000) |_| {
        try world.step(&.{}, 1);
        const quaternions = world.state.getQuaternions();
        for (0..world.config.num_envs) |env_id| {
            const body_index = world.state.bodyIndex(@intCast(env_id), 1);
            const quaternion = quaternions[body_index];
            const norm = @sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
            maximum_quaternion_norm_error = @max(maximum_quaternion_norm_error, @abs(norm - 1));
        }
    }

    const elapsed: f32 = 10000 * 0.002;
    const initial_speed_squared: f32 = 0.7 * 0.7 + 0.4 * 0.4 + 0.2 * 0.2;
    const initial_angular_speed_squared: f32 = 0.3 * 0.3 + 0.2 * 0.2 + 0.5 * 0.5;
    for (0..world.config.num_envs) |env_id| {
        const body_index = world.state.bodyIndex(@intCast(env_id), 1);
        const position = world.state.getPositions()[body_index];
        const velocity = world.state.getVelocities()[body_index];
        const angular_velocity = world.state.getAngularVelocities()[body_index];
        var position_error_squared: f32 = 0;
        var speed_squared: f32 = 0;
        var angular_speed_squared: f32 = 0;
        for (0..3) |axis| {
            const expected = initial_position[axis] + initial_velocity[axis] * elapsed;
            const error_value = position[axis] - expected;
            position_error_squared += error_value * error_value;
            speed_squared += velocity[axis] * velocity[axis];
            angular_speed_squared += angular_velocity[axis] * angular_velocity[axis];
        }
        try testing.expect(@sqrt(position_error_squared) < 0.003);
        try testing.expect(@abs(speed_squared - initial_speed_squared) < 0.001);
        try testing.expect(@abs(angular_speed_squared - initial_angular_speed_squared) < 0.005);
    }
    try testing.expect(maximum_quaternion_norm_error < 1e-5);
}

test "Metal capsule-box contact uses exact segment AABB distance" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="capsule-box">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="box" quat="0.9238795 0 0 0.3826834">
        \\      <geom type="box" size="0.2 0.1 0.2" mass="0"/>
        \\    </body>
        \\    <body name="capsule" pos="0 0 0.28">
        \\      <freejoint name="capsule-root"/>
        \\      <geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.1" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[2], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.02), contact.position_pen[3], 1e-5);
        const capsule = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[capsule][2] > 0.28);
    }
}

test "Metal box-box contact uses oriented SAT response" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const xml =
        \\<mujoco model="box-box">
        \\  <option timestep="0.001" gravity="0 0 0"/>
        \\  <worldbody>
        \\    <body name="fixed" quat="0.9238795 0 0 0.3826834">
        \\      <geom type="box" size="0.2 0.2 0.2" mass="0"/>
        \\    </body>
        \\    <body name="moving" pos="0.4 0 0">
        \\      <freejoint name="moving-root"/>
        \\      <geom type="box" size="0.2 0.2 0.2" mass="1"/>
        \\    </body>
        \\  </worldbody>
        \\</mujoco>
    ;

    const scene = try zeno.mjcf.parser.parseString(allocator, xml);
    var world = try zeno.World.init(allocator, scene, .{ .num_envs = 2, .substeps = 1 });
    defer world.deinit();
    try world.step(&.{}, 1);

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    for (0..world.config.num_envs) |env_id| {
        try testing.expectEqual(@as(u32, 1), counts[env_id]);
        const contact = contacts[env_id * world.config.max_contacts_per_env];
        try testing.expectApproxEqAbs(@as(f32, -1), contact.normal_friction[0], 1e-5);
        try testing.expectApproxEqAbs(@as(f32, 0.0828427), contact.position_pen[3], 1e-4);
        const moving = world.state.bodyIndex(@intCast(env_id), 2);
        try testing.expect(world.state.getPositions()[moving][0] > 0.4);
    }
}

test "Metal spatial hash pairs finite geoms with infinite planes across cells" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const primitives = zeno.collision.primitives;

    var scene = zeno.Scene.init(allocator);
    scene.physics_config.gravity = .{ 0, 0, 0 };

    const plane_body = try scene.addBody(.{ .body_type = .static });
    var plane = primitives.Geom.plane(.{ 0, 0, 1 }, 0);
    plane.body_id = plane_body;
    plane.group = 1;
    plane.mask = 1;
    _ = try scene.addGeom(plane);

    // World switches to the spatial broad phase above 512 geoms. These inert
    // fillers force that path without creating collision candidates.
    for (0..511) |i| {
        const filler_body = try scene.addBody(.{
            .body_type = .static,
            .position = .{ @as(f32, @floatFromInt(i % 16)), @as(f32, @floatFromInt(i / 16)), 10 },
        });
        var filler = primitives.Geom.sphere(0.01);
        filler.body_id = filler_body;
        filler.group = 0;
        filler.mask = 0;
        _ = try scene.addGeom(filler);
    }

    const sphere_body = try scene.addBody(.{
        .body_type = .dynamic,
        .position = .{ 20, 0, 0.05 },
        .mass = 1,
        .inertia = .{ 0.004, 0.004, 0.004 },
    });
    var sphere = primitives.Geom.sphere(0.1);
    sphere.body_id = sphere_body;
    sphere.group = 1;
    sphere.mask = 1;
    const sphere_geom = try scene.addGeom(sphere);

    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 1,
        .timestep = 0.001,
        .substeps = 1,
        .max_contacts_per_env = 8,
    });
    defer world.deinit();

    try world.step(&.{}, 1);
    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    try testing.expectEqual(@as(u32, 1), counts[0]);
    try testing.expectEqual([4]u32{ plane_body, sphere_body, 0, sphere_geom }, contacts[0].indices);
    try testing.expectApproxEqAbs(@as(f32, -1), contacts[0].normal_friction[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.05), contacts[0].position_pen[3], 1e-5);
    try testing.expect(world.state.getPositions()[sphere_body][2] > 0.05);
}

test "contact counts do not accumulate across steps" {
    // Regression: contact_counts was never cleared during stepping, so counts
    // grew monotonically until they saturated max_contacts and froze the
    // contact slots on the first pairs ever seen.
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");

    const config = zeno.WorldConfig{
        .num_envs = 2,
        .timestep = 0.005,
        .substeps = 2,
        .max_contacts_per_env = 64,
    };

    var world = try zeno.World.init(allocator, scene, config);
    defer world.deinit();

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, config.num_envs * info.action_dim);
    defer allocator.free(actions);
    @memset(actions, 0);

    // An ant resting on the plane keeps a persistent set of contacts. With the
    // accumulation bug, counts hit max_contacts within a few steps.
    for (0..50) |_| {
        try world.step(actions, 0);
    }

    const counts_ptr = world.getContactCountsPtr() orelse return error.NoContactCounts;
    for (0..config.num_envs) |env| {
        const count = counts_ptr[env];
        try testing.expect(count < config.max_contacts_per_env);
    }
}

test "batched GPU acceleration matches full-step velocity delta" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const config = zeno.WorldConfig{
        .num_envs = 17,
        .timestep = 0.004,
        .substeps = 2,
        .max_contacts_per_env = 16,
    };

    var world = try zeno.World.init(allocator, scene, config);
    defer world.deinit();

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, config.num_envs * info.action_dim);
    defer allocator.free(actions);
    for (actions, 0..) |*action, i| {
        action.* = @as(f32, @floatFromInt(i % 7)) * 0.03 - 0.09;
    }

    const velocity_count: usize = @intCast(config.num_envs * info.num_bodies);
    const before = try allocator.alloc([4]f32, velocity_count);
    defer allocator.free(before);
    @memcpy(before, world.state.getVelocities());

    try world.step(actions, 0);

    const after = world.state.getVelocities();
    const acceleration = world.state.getAccelerations();
    const inv_step: f32 = 1.0 / config.timestep;
    for (0..velocity_count) |i| {
        for (0..3) |axis| {
            const expected = (after[i][axis] - before[i][axis]) * inv_step;
            try testing.expectApproxEqAbs(expected, acceleration[i][axis], 1e-5);
        }
        try testing.expectEqual(@as(f32, 0), acceleration[i][3]);
    }
}

test "GPU masked stepping preserves complete inactive environment state" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");
    const Buffer = zeno.metal.buffer.Buffer;

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const config = zeno.WorldConfig{
        .num_envs = 4,
        .timestep = 0.004,
        .substeps = 2,
        .max_contacts_per_env = 16,
    };

    var world = try zeno.World.init(allocator, scene, config);
    defer world.deinit();

    const Check = struct {
        buffer: *const Buffer,
        snapshot: []u8 = &.{},
    };
    var checks = [_]Check{
        .{ .buffer = &world.state.positions_buffer },
        .{ .buffer = &world.state.quaternions_buffer },
        .{ .buffer = &world.state.velocities_buffer },
        .{ .buffer = &world.state.accelerations_buffer },
        .{ .buffer = &world.state.angular_velocities_buffer },
        .{ .buffer = &world.state.forces_buffer },
        .{ .buffer = &world.state.torques_buffer },
        .{ .buffer = &world.state.joint_positions_buffer },
        .{ .buffer = &world.state.joint_velocities_buffer },
        .{ .buffer = &world.state.joint_torques_buffer },
        .{ .buffer = &world.state.observations_buffer },
        .{ .buffer = &world.state.rewards_buffer },
        .{ .buffer = &world.state.dones_buffer },
        .{ .buffer = &world.state.contacts_buffer },
        .{ .buffer = &world.state.contact_counts_buffer },
        .{ .buffer = &world.state.rng_state_buffer },
        .{ .buffer = &world.state.prev_positions_buffer },
        .{ .buffer = &world.state.prev_quaternions_buffer },
        .{ .buffer = &world.state.prev_velocities_buffer },
        .{ .buffer = &world.prev_contacts_buffer },
        .{ .buffer = &world.prev_contact_counts_buffer },
        .{ .buffer = &world.constraints_buffer },
    };
    defer for (checks) |check| allocator.free(check.snapshot);

    for (&checks) |*check| {
        check.snapshot = try allocator.dupe(u8, check.buffer.getSlice(u8));
    }

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, config.num_envs * info.action_dim);
    defer allocator.free(actions);
    @memset(actions, 0.5);
    const mask = [_]u8{ 1, 0, 1, 0 };

    try world.stepSubset(actions, &mask, 0);

    for (checks) |check| {
        const current = check.buffer.getSlice(u8);
        const bytes_per_env = current.len / config.num_envs;
        for ([_]usize{ 1, 3 }) |env_id| {
            const start = env_id * bytes_per_env;
            try testing.expectEqualSlices(
                u8,
                check.snapshot[start .. start + bytes_per_env],
                current[start .. start + bytes_per_env],
            );
        }
    }

    const positions = world.state.getPositions();
    const bodies_per_env: usize = @intCast(info.num_bodies);
    try testing.expect(!std.mem.eql(
        u8,
        checks[0].snapshot[0 .. bodies_per_env * @sizeOf([4]f32)],
        std.mem.sliceAsBytes(positions[0..bodies_per_env]),
    ));
}

test "masked-step profiling excludes inactive contact counts" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 3,
        .enable_profiling = true,
        .substeps = 1,
    });
    defer world.deinit();

    const counts = world.state.contact_counts_buffer.getSlice(u32);
    counts[0] = 77;
    counts[1] = 0;
    counts[2] = 88;

    const actions = try allocator.alloc(f32, world.params.num_envs * world.params.num_actuators);
    defer allocator.free(actions);
    @memset(actions, 0);

    const mask = [_]u8{ 0, 1, 0 };
    try world.stepSubset(actions, &mask, 1);

    try testing.expectEqual(counts[1], world.getProfilingData().num_contacts);
    try testing.expectEqual(@as(u32, 77), counts[0]);
    try testing.expectEqual(@as(u32, 88), counts[2]);
}

test "compacted active environments match dense reference stepping" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const batched_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    const reference_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    const common = zeno.WorldConfig{
        .num_envs = 4,
        .timestep = 0.002,
        .substeps = 1,
        .contact_iterations = 4,
        .max_contacts_per_env = 32,
    };
    var batched = try zeno.World.init(allocator, batched_scene, common);
    defer batched.deinit();

    var reference_config = common;
    reference_config.num_envs = 2;
    var reference = try zeno.World.init(allocator, reference_scene, reference_config);
    defer reference.deinit();

    const action_dim: usize = @intCast(batched.params.num_actuators);
    const actions = try allocator.alloc(f32, common.num_envs * action_dim);
    defer allocator.free(actions);
    for (actions, 0..) |*action, i| action.* = @as(f32, @floatFromInt(i % 11)) * 0.04 - 0.2;

    const reference_actions = try allocator.alloc(f32, 2 * action_dim);
    defer allocator.free(reference_actions);
    @memcpy(reference_actions[0..action_dim], actions[action_dim .. 2 * action_dim]);
    @memcpy(reference_actions[action_dim .. 2 * action_dim], actions[3 * action_dim .. 4 * action_dim]);

    const mask = [_]u8{ 0, 1, 0, 1 };
    for (0..12) |_| {
        try batched.stepSubset(actions, &mask, 1);
        try reference.step(reference_actions, 1);
    }

    const dispatch = batched.env_dispatch_params_buffer.getSlice(zeno.world.world_mod.EnvDispatchParams)[0];
    try testing.expectEqual(@as(u32, 2), dispatch.dispatch_envs);
    try testing.expectEqual(@as(u32, 1), dispatch.use_active_env_ids);

    const bodies_per_env: usize = @intCast(batched.params.num_bodies);
    const batched_positions = batched.state.getPositions();
    const reference_positions = reference.state.getPositions();
    const batched_velocities = batched.state.getVelocities();
    const reference_velocities = reference.state.getVelocities();

    for ([_]usize{ 1, 3 }, 0..) |physical_env, dense_env| {
        for (0..bodies_per_env) |body_id| {
            const batched_idx = physical_env * bodies_per_env + body_id;
            const reference_idx = dense_env * bodies_per_env + body_id;
            for (0..4) |axis| {
                try testing.expectApproxEqAbs(reference_positions[reference_idx][axis], batched_positions[batched_idx][axis], 1e-5);
                try testing.expectApproxEqAbs(reference_velocities[reference_idx][axis], batched_velocities[batched_idx][axis], 1e-5);
            }
        }
    }

    const obs_per_env: usize = @intCast(batched.state.obs_dim);
    const batched_obs = batched.state.getObservations();
    const reference_obs = reference.state.getObservations();
    for ([_]usize{ 1, 3 }, 0..) |physical_env, dense_env| {
        for (0..obs_per_env) |obs_id| {
            try testing.expectApproxEqAbs(
                reference_obs[dense_env * obs_per_env + obs_id],
                batched_obs[physical_env * obs_per_env + obs_id],
                1e-5,
            );
        }
    }
}

test "GPU task outputs compute rewards horizons and masked episode clocks" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 3,
        .timestep = 0.002,
        .substeps = 1,
    });
    defer world.deinit();

    try world.configureTask(.{
        .enabled = 1,
        .root_body = 1,
        .forward_axis = 0,
        .max_episode_steps = 2,
        .forward_reward_weight = 2.0,
        .control_cost_weight = 0.25,
        .healthy_bonus = 1.5,
        .healthy_z_min = -100,
        .healthy_z_max = 100,
        .terminate_when_unhealthy = 1,
    });

    const action_count = world.params.num_envs * world.params.num_actuators;
    const actions = try allocator.alloc(f32, action_count);
    defer allocator.free(actions);
    @memset(actions, 0.5);

    try world.step(actions, 1);
    const rewards = world.state.getRewards();
    const dones = world.state.getDones();
    const episode_steps = world.episode_steps_buffer.getSlice(u32);
    const velocities = world.state.getVelocities();
    const action_cost = @as(f32, @floatFromInt(world.params.num_actuators)) * 0.25;
    for (0..world.params.num_envs) |env_id| {
        const root = world.state.bodyIndex(@intCast(env_id), 1);
        const expected = 2.0 * velocities[root][0] - 0.25 * action_cost + 1.5;
        try testing.expectApproxEqAbs(expected, rewards[env_id], 1e-5);
        try testing.expectEqual(@as(u8, 0), dones[env_id]);
        try testing.expectEqual(@as(u32, 1), episode_steps[env_id]);
    }

    try world.step(actions, 1);
    for (0..world.params.num_envs) |env_id| {
        try testing.expectEqual(@as(u8, 1), dones[env_id]);
        try testing.expectEqual(@as(u32, 2), episode_steps[env_id]);
    }

    const reset_mask = [_]u8{ 0, 1, 0 };
    try world.reset(&reset_mask);
    try testing.expectEqual(@as(f32, 0), rewards[1]);
    try testing.expectEqual(@as(u8, 0), dones[1]);
    try testing.expectEqual(@as(u32, 0), episode_steps[1]);

    try world.stepSubset(actions, &reset_mask, 1);
    try testing.expectEqual(@as(u32, 2), episode_steps[0]);
    try testing.expectEqual(@as(u32, 1), episode_steps[1]);
    try testing.expectEqual(@as(u32, 2), episode_steps[2]);
    try testing.expectEqual(@as(u8, 1), dones[0]);
    try testing.expectEqual(@as(u8, 0), dones[1]);
    try testing.expectEqual(@as(u8, 1), dones[2]);
}

test "GPU task configuration rejects invalid body axis and health range" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    var world = try zeno.World.init(allocator, scene, .{});
    defer world.deinit();

    try testing.expectError(error.InvalidTaskConfig, world.configureTask(.{
        .enabled = 1,
        .root_body = world.params.num_bodies,
        .healthy_z_min = 0,
        .healthy_z_max = 1,
    }));
    try testing.expectError(error.InvalidTaskConfig, world.configureTask(.{
        .enabled = 1,
        .forward_axis = 3,
        .healthy_z_min = 0,
        .healthy_z_max = 1,
    }));
    try testing.expectError(error.InvalidTaskConfig, world.configureTask(.{
        .enabled = 1,
        .healthy_z_min = 2,
        .healthy_z_max = 1,
    }));
}

test "GPU masked reset restores complete active state without touching inactive auxiliaries" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = 3,
        .max_contacts_per_env = 4,
    });
    defer world.deinit();

    const active_env: u32 = 1;
    const body_start = active_env * world.params.num_bodies;
    const joint_start = active_env * world.params.num_joints;
    for (0..world.params.num_bodies) |body_id| {
        const index = body_start + body_id;
        world.state.getPositions()[index] = .{ 9, 8, 7, 0 };
        world.state.getVelocities()[index] = .{ 6, 5, 4, 0 };
        world.state.getAccelerations()[index] = .{ 3, 2, 1, 0 };
        world.state.getAngularVelocities()[index] = .{ 4, 3, 2, 0 };
        world.state.forces_buffer.getAlignedSlice([4]f32, 16)[index] = .{ 1, 1, 1, 0 };
        world.state.torques_buffer.getAlignedSlice([4]f32, 16)[index] = .{ 2, 2, 2, 0 };
    }
    for (0..world.params.num_joints) |joint_id| {
        world.state.getJointPositions()[joint_start + joint_id] = 9;
        world.state.getJointVelocities()[joint_start + joint_id] = 8;
        world.state.joint_torques_buffer.getSlice(f32)[joint_start + joint_id] = 7;
    }

    const contacts = world.state.contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    const prev_contacts = world.prev_contacts_buffer.getAlignedSlice(zeno.physics.contact.ContactGPU, 16);
    const contact_start = active_env * world.config.max_contacts_per_env;
    for (0..world.config.max_contacts_per_env) |contact_id| {
        contacts[contact_start + contact_id].position_pen[3] = 4;
        prev_contacts[contact_start + contact_id].position_pen[3] = 5;
    }
    const counts = world.state.contact_counts_buffer.getSlice(u32);
    const prev_counts = world.prev_contact_counts_buffer.getSlice(u32);
    counts[0] = 77;
    counts[1] = 66;
    counts[2] = 55;
    prev_counts[0] = 44;
    prev_counts[1] = 33;
    prev_counts[2] = 22;
    world.state.getRewards()[1] = 12;
    world.state.getDones()[1] = 1;
    world.episode_steps_buffer.getSlice(u32)[1] = 99;

    const inactive_position = world.state.getPositions()[0];
    const mask = [_]u8{ 0, 1, 0 };
    try world.reset(&mask);

    try testing.expectEqual(inactive_position, world.state.getPositions()[0]);
    try testing.expectEqual(@as(u32, 77), counts[0]);
    try testing.expectEqual(@as(u32, 55), counts[2]);
    try testing.expectEqual(@as(u32, 44), prev_counts[0]);
    try testing.expectEqual(@as(u32, 22), prev_counts[2]);
    try testing.expectEqual(@as(u32, 0), counts[1]);
    try testing.expectEqual(@as(u32, 0), prev_counts[1]);
    try testing.expectEqual(@as(f32, 0), world.state.getRewards()[1]);
    try testing.expectEqual(@as(u8, 0), world.state.getDones()[1]);
    try testing.expectEqual(@as(u32, 0), world.episode_steps_buffer.getSlice(u32)[1]);

    const initial_positions = world.initial_positions_buffer.getAlignedSlice([4]f32, 16);
    const initial_velocities = world.initial_velocities_buffer.getAlignedSlice([4]f32, 16);
    for (0..world.params.num_bodies) |body_id| {
        const index = body_start + body_id;
        try testing.expectEqual(initial_positions[body_id], world.state.getPositions()[index]);
        try testing.expectEqual(initial_velocities[body_id], world.state.getVelocities()[index]);
        try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, world.state.getAccelerations()[index]);
        try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, world.state.forces_buffer.getAlignedSlice([4]f32, 16)[index]);
        try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, world.state.torques_buffer.getAlignedSlice([4]f32, 16)[index]);
    }
    for (0..world.config.max_contacts_per_env) |contact_id| {
        try testing.expectEqual(@as(f32, -1), contacts[contact_start + contact_id].position_pen[3]);
        try testing.expectEqual(@as(f32, -1), prev_contacts[contact_start + contact_id].position_pen[3]);
    }
}

test "async Metal step commits before wait and enforces one in-flight command" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const async_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    const reference_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    const config = zeno.WorldConfig{
        .num_envs = 8,
        .timestep = 0.002,
        .substeps = 1,
        .max_contacts_per_env = 16,
    };
    var async_world = try zeno.World.init(allocator, async_scene, config);
    defer async_world.deinit();
    var reference = try zeno.World.init(allocator, reference_scene, config);
    defer reference.deinit();

    const actions = try allocator.alloc(f32, config.num_envs * async_world.params.num_actuators);
    defer allocator.free(actions);
    for (actions, 0..) |*action, i| action.* = @as(f32, @floatFromInt(i % 7)) * 0.03 - 0.09;

    try async_world.stepAsync(actions, 1);
    try testing.expect(async_world.hasPendingStep());
    try testing.expectError(error.StepPending, async_world.stepAsync(actions, 1));
    try testing.expectError(error.StepPending, async_world.reset(null));
    try testing.expectError(error.StepPending, async_world.configureTask(.{}));

    try reference.step(actions, 1);
    try async_world.waitStep();
    try testing.expect(!async_world.hasPendingStep());
    try testing.expectError(error.NoPendingStep, async_world.waitStep());

    for (0..24) |_| {
        try async_world.stepAsync(actions, 1);
        try reference.step(actions, 1);
        try async_world.waitStep();
    }

    const async_positions = async_world.state.getPositions();
    const reference_positions = reference.state.getPositions();
    for (async_positions, reference_positions) |actual, expected| {
        for (0..4) |axis| try testing.expectApproxEqAbs(expected[axis], actual[axis], 1e-5);
    }

    const before_noop = try allocator.dupe([4]f32, async_positions);
    defer allocator.free(before_noop);
    const no_envs = [_]u8{0} ** config.num_envs;
    try async_world.stepSubsetAsync(actions, &no_envs, 1);
    try testing.expect(async_world.hasPendingStep());
    try async_world.waitStep();
    try testing.expectEqualSlices([4]f32, before_noop, async_world.state.getPositions());
}

test "fused masked reset and full step matches separate submissions" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const separate_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const fused_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const config = zeno.WorldConfig{
        .num_envs = 4,
        .timestep = 0.002,
        .substeps = 1,
        .max_contacts_per_env = 8,
    };
    var separate = try zeno.World.init(allocator, separate_scene, config);
    defer separate.deinit();
    var fused = try zeno.World.init(allocator, fused_scene, config);
    defer fused.deinit();

    const actions = try allocator.alloc(f32, config.num_envs * fused.params.num_actuators);
    defer allocator.free(actions);
    for (actions, 0..) |*action, i| action.* = 0.15 + @as(f32, @floatFromInt(i % 5)) * 0.02;

    // Move both worlds away from their templates so selected reset behavior is
    // observable and inactive environments must continue from evolved state.
    for (0..8) |_| {
        try separate.step(actions, 1);
        try fused.step(actions, 1);
    }

    const reset_mask = [_]u8{ 0, 1, 0, 1 };
    try separate.reset(&reset_mask);
    try separate.step(actions, 1);
    try fused.stepWithResetAsync(actions, &reset_mask, 1);
    try testing.expect(fused.hasPendingStep());
    try fused.waitStep();

    // The reset kernel must preserve the next episode's actions when fused.
    try testing.expectEqualSlices(f32, actions, fused.state.getActions());

    for (separate.state.getPositions(), fused.state.getPositions()) |expected, actual| {
        for (0..4) |axis| try testing.expectApproxEqAbs(expected[axis], actual[axis], 1e-5);
    }
    for (separate.state.getVelocities(), fused.state.getVelocities()) |expected, actual| {
        for (0..4) |axis| try testing.expectApproxEqAbs(expected[axis], actual[axis], 1e-5);
    }
    try testing.expectEqualSlices(f32, separate.state.getJointPositions(), fused.state.getJointPositions());
    try testing.expectEqualSlices(f32, separate.state.getJointVelocities(), fused.state.getJointVelocities());
    try testing.expectEqualSlices(f32, separate.state.getObservations(), fused.state.getObservations());
    try testing.expectEqualSlices(u32, separate.episode_steps_buffer.getSlice(u32), fused.episode_steps_buffer.getSlice(u32));
}

test "shared action buffer submission matches copied actions" {
    const allocator = testing.allocator;
    const zeno = @import("zeno");

    const copied_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const shared_scene = try zeno.mjcf.parser.parseFile(allocator, "assets/pendulum.xml");
    const config = zeno.WorldConfig{ .num_envs = 8, .substeps = 1 };
    var copied = try zeno.World.init(allocator, copied_scene, config);
    defer copied.deinit();
    var shared = try zeno.World.init(allocator, shared_scene, config);
    defer shared.deinit();

    const actions = try allocator.alloc(f32, config.num_envs * copied.params.num_actuators);
    defer allocator.free(actions);
    for (actions, 0..) |*action, i| action.* = @as(f32, @floatFromInt(i + 1)) * 0.04;
    @memcpy(shared.state.getActions(), actions);

    try copied.step(actions, 1);
    try shared.stepCurrentActionsAsync(1);
    try testing.expect(shared.hasPendingStep());
    try shared.waitStep();

    try testing.expectEqualSlices(f32, actions, shared.state.getActions());
    for (copied.state.getPositions(), shared.state.getPositions()) |expected, actual| {
        for (0..4) |axis| try testing.expectApproxEqAbs(expected[axis], actual[axis], 1e-5);
    }
}
