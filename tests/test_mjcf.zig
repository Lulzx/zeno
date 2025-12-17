//! Tests for MJCF parser.

const std = @import("std");
const testing = std.testing;

const parser = @import("zeno").mjcf.parser;
const schema = @import("zeno").mjcf.schema;

const simple_mjcf =
    \\<mujoco model="test">
    \\    <option timestep="0.01" gravity="0 0 -10"/>
    \\    <worldbody>
    \\        <geom type="plane" size="5 5 0.1"/>
    \\        <body name="ball" pos="0 0 1">
    \\            <joint type="free"/>
    \\            <geom type="sphere" size="0.1" mass="1"/>
    \\        </body>
    \\    </worldbody>
    \\</mujoco>
;

const pendulum_mjcf =
    \\<mujoco model="pendulum">
    \\    <option timestep="0.02"/>
    \\    <worldbody>
    \\        <geom type="plane" size="5 5 0.1"/>
    \\        <body name="base" pos="0 0 1.5">
    \\            <body name="pole" pos="0 0 0">
    \\                <joint name="hinge" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
    \\                <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.02" mass="1"/>
    \\            </body>
    \\        </body>
    \\    </worldbody>
    \\    <actuator>
    \\        <motor joint="hinge" ctrlrange="-5 5" gear="1"/>
    \\    </actuator>
    \\</mujoco>
;

test "parse simple mjcf" {
    var scene = try parser.parseString(testing.allocator, simple_mjcf);
    defer scene.deinit();

    // Should have bodies
    try testing.expect(scene.numBodies() > 0);

    // Check physics config
    try testing.expectApproxEqAbs(@as(f32, 0.01), scene.physics_config.timestep, 0.001);
    try testing.expectApproxEqAbs(@as(f32, -10.0), scene.physics_config.gravity[2], 0.001);
}

test "parse pendulum mjcf" {
    var scene = try parser.parseString(testing.allocator, pendulum_mjcf);
    defer scene.deinit();

    // Should have joint
    try testing.expect(scene.numJoints() > 0);

    // Should have actuator
    try testing.expect(scene.numActuators() > 0);

    // Check joint by name
    const joint_id = scene.getJointByName("hinge");
    try testing.expect(joint_id != null);
}

test "scene validation" {
    var scene = try parser.parseString(testing.allocator, simple_mjcf);
    defer scene.deinit();

    try testing.expect(scene.validate());
}

test "geom type parsing" {
    try testing.expectEqual(schema.GeomType.sphere, schema.GeomType.fromString("sphere").?);
    try testing.expectEqual(schema.GeomType.capsule, schema.GeomType.fromString("capsule").?);
    try testing.expectEqual(schema.GeomType.box, schema.GeomType.fromString("box").?);
    try testing.expectEqual(schema.GeomType.plane, schema.GeomType.fromString("plane").?);
    try testing.expect(schema.GeomType.fromString("invalid") == null);
}

test "joint type parsing" {
    try testing.expectEqual(schema.JointType.hinge, schema.JointType.fromString("hinge").?);
    try testing.expectEqual(schema.JointType.slide, schema.JointType.fromString("slide").?);
    try testing.expectEqual(schema.JointType.ball, schema.JointType.fromString("ball").?);
    try testing.expectEqual(schema.JointType.free, schema.JointType.fromString("free").?);
}
