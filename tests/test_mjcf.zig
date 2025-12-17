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

// Edge case tests for MJCF parser

test "empty worldbody" {
    const empty_mjcf =
        \\<mujoco model="empty">
        \\    <worldbody>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, empty_mjcf);
    defer scene.deinit();

    // Should have no bodies (or just ground body)
    try testing.expect(scene.numBodies() == 0);
}

test "deeply nested bodies" {
    const nested_mjcf =
        \\<mujoco model="nested">
        \\    <worldbody>
        \\        <body name="b1" pos="0 0 1">
        \\            <geom type="sphere" size="0.1"/>
        \\            <body name="b2" pos="0 0 1">
        \\                <geom type="sphere" size="0.1"/>
        \\                <body name="b3" pos="0 0 1">
        \\                    <geom type="sphere" size="0.1"/>
        \\                    <body name="b4" pos="0 0 1">
        \\                        <geom type="sphere" size="0.1"/>
        \\                    </body>
        \\                </body>
        \\            </body>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, nested_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numBodies() >= 4);
    try testing.expect(scene.getBodyByName("b4") != null);
}

test "multiple joints on body" {
    const multi_joint_mjcf =
        \\<mujoco model="multi_joint">
        \\    <worldbody>
        \\        <body name="arm" pos="0 0 1">
        \\            <joint name="j1" type="hinge" axis="1 0 0"/>
        \\            <joint name="j2" type="hinge" axis="0 1 0"/>
        \\            <joint name="j3" type="hinge" axis="0 0 1"/>
        \\            <geom type="box" size="0.1 0.1 0.5"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, multi_joint_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numJoints() >= 3);
}

test "geom fromto syntax" {
    const fromto_mjcf =
        \\<mujoco model="fromto">
        \\    <worldbody>
        \\        <body name="link" pos="0 0 0">
        \\            <geom type="capsule" fromto="0 0 0 0 0 1" size="0.1"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, fromto_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numGeoms() >= 1);
}

test "actuator types" {
    const actuator_mjcf =
        \\<mujoco model="actuators">
        \\    <worldbody>
        \\        <body name="arm" pos="0 0 1">
        \\            <joint name="motor_joint" type="hinge"/>
        \\            <joint name="pos_joint" type="hinge"/>
        \\            <joint name="vel_joint" type="hinge"/>
        \\            <geom type="sphere" size="0.1"/>
        \\        </body>
        \\    </worldbody>
        \\    <actuator>
        \\        <motor joint="motor_joint" ctrlrange="-1 1"/>
        \\        <position joint="pos_joint" ctrlrange="-3.14 3.14" kp="100"/>
        \\        <velocity joint="vel_joint" ctrlrange="-10 10" kv="10"/>
        \\    </actuator>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, actuator_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numActuators() >= 3);
}

test "sensor types" {
    const sensor_mjcf =
        \\<mujoco model="sensors">
        \\    <worldbody>
        \\        <body name="robot" pos="0 0 1">
        \\            <joint name="j1" type="hinge"/>
        \\            <geom type="sphere" size="0.1"/>
        \\        </body>
        \\    </worldbody>
        \\    <sensor>
        \\        <jointpos joint="j1"/>
        \\        <jointvel joint="j1"/>
        \\    </sensor>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, sensor_mjcf);
    defer scene.deinit();

    try testing.expect(scene.sensor_config.count() >= 2);
}

test "default class inheritance" {
    const default_mjcf =
        \\<mujoco model="defaults">
        \\    <default>
        \\        <geom friction="0.5 0.1 0.1"/>
        \\        <joint damping="0.5"/>
        \\    </default>
        \\    <worldbody>
        \\        <body name="b1" pos="0 0 1">
        \\            <joint type="hinge"/>
        \\            <geom type="sphere" size="0.1"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, default_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numBodies() >= 1);
}

test "euler angles" {
    const euler_mjcf =
        \\<mujoco model="euler">
        \\    <worldbody>
        \\        <body name="rotated" pos="0 0 1" euler="90 0 0">
        \\            <geom type="box" size="0.5 0.1 0.1"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, euler_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numBodies() >= 1);
}

test "quaternion orientation" {
    const quat_mjcf =
        \\<mujoco model="quat">
        \\    <worldbody>
        \\        <body name="rotated" pos="0 0 1" quat="0.707 0 0.707 0">
        \\            <geom type="box" size="0.5 0.1 0.1"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, quat_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numBodies() >= 1);
}

test "geom density vs mass" {
    const mass_mjcf =
        \\<mujoco model="mass">
        \\    <worldbody>
        \\        <body name="b1" pos="0 0 1">
        \\            <geom type="sphere" size="0.1" mass="1.5"/>
        \\        </body>
        \\        <body name="b2" pos="1 0 1">
        \\            <geom type="sphere" size="0.1" density="1000"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, mass_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numBodies() >= 2);
}

test "joint limits" {
    const limits_mjcf =
        \\<mujoco model="limits">
        \\    <worldbody>
        \\        <body name="arm" pos="0 0 1">
        \\            <joint name="limited" type="hinge" range="-1.57 1.57"/>
        \\            <geom type="capsule" size="0.05 0.5"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, limits_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numJoints() >= 1);
    const joint_id = scene.getJointByName("limited");
    try testing.expect(joint_id != null);
}

test "all geom types" {
    const geom_types_mjcf =
        \\<mujoco model="geom_types">
        \\    <worldbody>
        \\        <geom type="plane" size="10 10 0.1"/>
        \\        <body name="shapes" pos="0 0 1">
        \\            <geom name="sphere" type="sphere" size="0.1" pos="0 0 0"/>
        \\            <geom name="box" type="box" size="0.1 0.1 0.1" pos="1 0 0"/>
        \\            <geom name="capsule" type="capsule" size="0.05 0.2" pos="2 0 0"/>
        \\            <geom name="cylinder" type="cylinder" size="0.05 0.2" pos="3 0 0"/>
        \\        </body>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, geom_types_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numGeoms() >= 5);
}

test "actuator gear ratio" {
    const gear_mjcf =
        \\<mujoco model="gear">
        \\    <worldbody>
        \\        <body name="arm" pos="0 0 1">
        \\            <joint name="j1" type="hinge"/>
        \\            <geom type="capsule" size="0.05 0.5"/>
        \\        </body>
        \\    </worldbody>
        \\    <actuator>
        \\        <motor joint="j1" gear="100"/>
        \\    </actuator>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, gear_mjcf);
    defer scene.deinit();

    try testing.expect(scene.numActuators() >= 1);
}

test "option element" {
    const option_mjcf =
        \\<mujoco model="option">
        \\    <option timestep="0.001" gravity="0 0 -10"/>
        \\    <worldbody>
        \\    </worldbody>
        \\</mujoco>
    ;

    var scene = try parser.parseString(testing.allocator, option_mjcf);
    defer scene.deinit();

    try testing.expectApproxEqAbs(@as(f32, 0.001), scene.physics_config.timestep, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, -10.0), scene.physics_config.gravity[2], 0.1);
}
