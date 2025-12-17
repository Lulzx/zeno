//! Tests for physics module.

const std = @import("std");
const testing = std.testing;

const body = @import("zeno").physics.body;
const joint = @import("zeno").physics.joint;
const contact = @import("zeno").physics.contact;
const constants = @import("zeno").physics.constants;

test "body def default values" {
    const def = body.BodyDef{};

    try testing.expectEqual(body.BodyType.dynamic, def.body_type);
    try testing.expectApproxEqAbs(@as(f32, 1.0), def.mass, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), def.gravity_scale, 0.001);
}

test "body inverse mass" {
    var def = body.BodyDef{};
    def.mass = 2.0;

    try testing.expectApproxEqAbs(@as(f32, 0.5), def.invMass(), 0.001);

    // Static body should have zero inverse mass
    def.body_type = .static;
    try testing.expectApproxEqAbs(@as(f32, 0.0), def.invMass(), 0.001);
}

test "transform identity" {
    const t = body.Transform.IDENTITY;

    try testing.expectApproxEqAbs(@as(f32, 0.0), t.position[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), t.quaternion[3], 0.001);
}

test "transform point" {
    const t = body.Transform{
        .position = .{ 1.0, 2.0, 3.0 },
        .quaternion = .{ 0, 0, 0, 1 }, // Identity rotation
    };

    const local = [3]f32{ 1.0, 0.0, 0.0 };
    const world = t.transformPoint(local);

    try testing.expectApproxEqAbs(@as(f32, 2.0), world[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), world[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), world[2], 0.001);
}

test "aabb intersection" {
    const a = body.AABB{
        .min = .{ 0, 0, 0 },
        .max = .{ 2, 2, 2 },
    };

    const b = body.AABB{
        .min = .{ 1, 1, 1 },
        .max = .{ 3, 3, 3 },
    };

    const c = body.AABB{
        .min = .{ 5, 5, 5 },
        .max = .{ 6, 6, 6 },
    };

    try testing.expect(a.intersects(&b));
    try testing.expect(!a.intersects(&c));
}

test "aabb contains point" {
    const aabb = body.AABB{
        .min = .{ 0, 0, 0 },
        .max = .{ 2, 2, 2 },
    };

    try testing.expect(aabb.contains(.{ 1, 1, 1 }));
    try testing.expect(!aabb.contains(.{ 3, 3, 3 }));
}

test "joint dof" {
    const revolute = joint.JointDef{ .joint_type = .revolute };
    const ball = joint.JointDef{ .joint_type = .ball };
    const free = joint.JointDef{ .joint_type = .free };
    const fixed = joint.JointDef{ .joint_type = .fixed };

    try testing.expectEqual(@as(u32, 1), revolute.dof());
    try testing.expectEqual(@as(u32, 3), ball.dof());
    try testing.expectEqual(@as(u32, 6), free.dof());
    try testing.expectEqual(@as(u32, 0), fixed.dof());
}

test "joint limits" {
    var j = joint.JointDef{
        .limit_lower = -1.5,
        .limit_upper = 1.5,
        .limited = true,
    };

    try testing.expect(j.hasLimits());
    try testing.expectApproxEqAbs(@as(f32, 1.5), j.clampPosition(2.0), 0.001);
    try testing.expectApproxEqAbs(@as(f32, -1.5), j.clampPosition(-2.0), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.5), j.limitViolation(2.0), 0.001);
}

test "contact initialization" {
    const c = contact.Contact.init(
        .{ 0, 0, 0 },
        .{ 0, 0, 1 },
        0.1,
        0,
        1,
    );

    try testing.expect(c.active);
    try testing.expectApproxEqAbs(@as(f32, 0.1), c.penetration, 0.001);
    try testing.expectEqual(@as(u32, 0), c.body_a);
    try testing.expectEqual(@as(u32, 1), c.body_b);
}

test "physics config validation" {
    const valid = constants.PhysicsConfig{};
    try testing.expect(valid.validate());

    const invalid = constants.PhysicsConfig{
        .timestep = -1.0,
    };
    try testing.expect(!invalid.validate());
}
