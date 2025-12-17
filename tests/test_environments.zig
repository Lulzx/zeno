//! Tests for all environment MJCF files.
//!
//! Ensures all 10 standard environments parse correctly and have valid configurations.

const std = @import("std");
const testing = std.testing;

const parser = @import("zeno").mjcf.parser;
const Scene = @import("zeno").world.scene.Scene;

// All environment MJCF files
const ENV_FILES = [_]struct {
    name: []const u8,
    file: []const u8,
    min_bodies: u32,
    min_joints: u32,
    min_actuators: u32,
}{
    .{ .name = "Pendulum", .file = "assets/pendulum.xml", .min_bodies = 2, .min_joints = 1, .min_actuators = 1 },
    .{ .name = "Cartpole", .file = "assets/cartpole.xml", .min_bodies = 2, .min_joints = 1, .min_actuators = 1 },
    .{ .name = "Ant", .file = "assets/ant.xml", .min_bodies = 5, .min_joints = 5, .min_actuators = 4 },
    .{ .name = "Humanoid", .file = "assets/humanoid.xml", .min_bodies = 10, .min_joints = 10, .min_actuators = 10 },
    .{ .name = "HalfCheetah", .file = "assets/cheetah.xml", .min_bodies = 5, .min_joints = 5, .min_actuators = 5 },
    .{ .name = "Hopper", .file = "assets/hopper.xml", .min_bodies = 4, .min_joints = 3, .min_actuators = 3 },
    .{ .name = "Walker2d", .file = "assets/walker.xml", .min_bodies = 5, .min_joints = 5, .min_actuators = 5 },
    .{ .name = "Swimmer", .file = "assets/swimmer.xml", .min_bodies = 3, .min_joints = 2, .min_actuators = 2 },
    .{ .name = "Reacher", .file = "assets/reacher.xml", .min_bodies = 3, .min_joints = 2, .min_actuators = 2 },
    .{ .name = "Pusher", .file = "assets/pusher.xml", .min_bodies = 4, .min_joints = 3, .min_actuators = 3 },
};

test "parse all environment mjcf files" {
    for (ENV_FILES) |env| {
        var scene = parser.parseFile(testing.allocator, env.file) catch |err| {
            std.debug.print("Failed to parse {s}: {}\n", .{ env.name, err });
            return err;
        };
        defer scene.deinit();

        // Check minimum counts
        try testing.expect(scene.numBodies() >= env.min_bodies);
        try testing.expect(scene.numJoints() >= env.min_joints);
        try testing.expect(scene.numActuators() >= env.min_actuators);
    }
}

test "all environments have valid physics config" {
    for (ENV_FILES) |env| {
        var scene = try parser.parseFile(testing.allocator, env.file);
        defer scene.deinit();

        // Check timestep is reasonable
        try testing.expect(scene.physics_config.timestep > 0.0);
        try testing.expect(scene.physics_config.timestep <= 0.1);
    }
}

test "all environments validate" {
    for (ENV_FILES) |env| {
        var scene = try parser.parseFile(testing.allocator, env.file);
        defer scene.deinit();

        try testing.expect(scene.validate());
    }
}

test "all environments have named bodies" {
    for (ENV_FILES) |env| {
        var scene = try parser.parseFile(testing.allocator, env.file);
        defer scene.deinit();

        // At least one body should be accessible by name
        // Use the body_names hash map directly instead of iterating bodies
        const has_named = scene.body_names.count() > 0;
        try testing.expect(has_named);
    }
}

test "pendulum environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/pendulum.xml");
    defer scene.deinit();

    // Pendulum should have a hinge joint
    try testing.expect(scene.numJoints() >= 1);
    try testing.expect(scene.numActuators() == 1);
}

test "cartpole environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/cartpole.xml");
    defer scene.deinit();

    // Cartpole has slide joint for cart
    try testing.expect(scene.numJoints() >= 2);
    try testing.expect(scene.numActuators() == 1);
}

test "ant environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/ant.xml");
    defer scene.deinit();

    // Ant has many joints and actuators for legs
    try testing.expect(scene.numJoints() >= 8);
    try testing.expect(scene.numActuators() >= 8);

    // Should have torso
    try testing.expect(scene.getBodyByName("torso") != null);
}

test "reacher environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/reacher.xml");
    defer scene.deinit();

    // Reacher is a 2-link arm
    try testing.expect(scene.numJoints() == 2);
    try testing.expect(scene.numActuators() == 2);

    // Should have fingertip and target
    try testing.expect(scene.getBodyByName("fingertip") != null);
    try testing.expect(scene.getBodyByName("target") != null);
}

test "pusher environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/pusher.xml");
    defer scene.deinit();

    // Pusher has 3-DOF arm plus object joints
    try testing.expect(scene.numJoints() >= 3);
    try testing.expect(scene.numActuators() == 3);

    // Should have object and goal
    try testing.expect(scene.getBodyByName("object") != null);
    try testing.expect(scene.getBodyByName("goal") != null);
}

test "humanoid environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/humanoid.xml");
    defer scene.deinit();

    // Humanoid is the most complex
    try testing.expect(scene.numBodies() >= 13);
    try testing.expect(scene.numJoints() >= 13);
    try testing.expect(scene.numActuators() >= 13);
}

test "swimmer environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/swimmer.xml");
    defer scene.deinit();

    // Swimmer has 2 joints connecting 3 segments
    try testing.expect(scene.numJoints() >= 2);
    try testing.expect(scene.numActuators() >= 2);
}

test "hopper environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/hopper.xml");
    defer scene.deinit();

    // Hopper has 3 controlled joints
    try testing.expect(scene.numJoints() >= 3);
    try testing.expect(scene.numActuators() >= 3);
}

test "walker2d environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/walker.xml");
    defer scene.deinit();

    // Walker2d has 6 controlled joints
    try testing.expect(scene.numJoints() >= 6);
    try testing.expect(scene.numActuators() >= 6);
}

test "halfcheetah environment details" {
    var scene = try parser.parseFile(testing.allocator, "assets/cheetah.xml");
    defer scene.deinit();

    // HalfCheetah has 6 controlled joints
    try testing.expect(scene.numJoints() >= 6);
    try testing.expect(scene.numActuators() >= 6);
}
