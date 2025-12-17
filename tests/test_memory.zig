//! Memory leak detection tests.
//!
//! These tests use Zig's GeneralPurposeAllocator with leak detection enabled
//! to ensure all allocated memory is properly freed.

const std = @import("std");
const testing = std.testing;

const parser = @import("zeno").mjcf.parser;
const Scene = @import("zeno").world.scene.Scene;
const body_mod = @import("zeno").physics.body;
const joint_mod = @import("zeno").physics.joint;
const primitives = @import("zeno").collision.primitives;

/// Create a leak-detecting allocator for testing.
fn createLeakDetectingAllocator() std.heap.GeneralPurposeAllocator(.{
    .enable_memory_limit = false,
    .safety = true,
}) {
    return .{};
}

test "scene allocation and deallocation - no leaks" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in scene test!");
        }
    }
    const allocator = gpa.allocator();

    // Create and destroy scene multiple times
    for (0..10) |_| {
        var scene = Scene.init(allocator);

        // Add bodies
        _ = try scene.addBody(.{
            .name = "body1",
            .position = .{ 0, 0, 1 },
        });
        _ = try scene.addBody(.{
            .name = "body2",
            .position = .{ 1, 0, 1 },
        });
        _ = try scene.addBody(.{
            .name = "body3",
            .position = .{ 2, 0, 1 },
        });

        // Add joints
        _ = try scene.addJoint(.{
            .name = "joint1",
            .parent_body = 0,
            .child_body = 1,
            .joint_type = .revolute,
        });

        // Add geoms
        _ = try scene.addNamedGeom("geom1", .{
            .geom_type = .sphere,
            .size = .{ 0.5, 0, 0 },
            .body_id = 1,
        });

        // Add actuator
        _ = try scene.addActuator(.{
            .name = "actuator1",
            .joint = 0,
        });

        scene.deinit();
    }
}

test "mjcf parser allocation - no leaks" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in MJCF parser test!");
        }
    }
    const allocator = gpa.allocator();

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

    // Parse and destroy multiple times
    for (0..10) |_| {
        var scene = try parser.parseString(allocator, simple_mjcf);
        scene.deinit();
    }
}

test "complex mjcf allocation - no leaks" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in complex MJCF test!");
        }
    }
    const allocator = gpa.allocator();

    const pendulum_mjcf =
        \\<mujoco model="pendulum">
        \\    <option timestep="0.02"/>
        \\    <worldbody>
        \\        <geom type="plane" size="5 5 0.1"/>
        \\        <body name="base" pos="0 0 1.5">
        \\            <geom type="sphere" size="0.1"/>
        \\            <body name="pole" pos="0 0 0">
        \\                <joint name="hinge" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        \\                <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.02" mass="1"/>
        \\                <body name="tip" pos="0 0 -1">
        \\                    <geom type="sphere" size="0.05"/>
        \\                </body>
        \\            </body>
        \\        </body>
        \\    </worldbody>
        \\    <actuator>
        \\        <motor joint="hinge" ctrlrange="-5 5" gear="1"/>
        \\    </actuator>
        \\</mujoco>
    ;

    // Parse and destroy multiple times
    for (0..5) |_| {
        var scene = try parser.parseString(allocator, pendulum_mjcf);

        // Access all data to ensure it's valid
        _ = scene.numBodies();
        _ = scene.numJoints();
        _ = scene.numGeoms();
        _ = scene.numActuators();
        _ = scene.getBodyByName("base");
        _ = scene.getJointByName("hinge");

        scene.deinit();
    }
}

test "scene name lookup allocation - no leaks" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in name lookup test!");
        }
    }
    const allocator = gpa.allocator();

    var scene = Scene.init(allocator);
    defer scene.deinit();

    // Add many named items
    for (0..100) |i| {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "body_{d}", .{i}) catch unreachable;
        _ = try scene.addBody(.{
            .name = name,
            .position = .{ @floatFromInt(i), 0, 0 },
        });
    }

    // Look up all items
    for (0..100) |i| {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "body_{d}", .{i}) catch unreachable;
        const id = scene.getBodyByName(name);
        try testing.expect(id != null);
        try testing.expectEqual(@as(u32, @intCast(i)), id.?);
    }
}

test "sensor configuration allocation - no leaks" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in sensor config test!");
        }
    }
    const allocator = gpa.allocator();

    const sensor_mjcf =
        \\<mujoco model="sensors">
        \\    <worldbody>
        \\        <body name="robot" pos="0 0 1">
        \\            <joint name="slide" type="slide" axis="1 0 0"/>
        \\            <geom type="box" size="0.1 0.1 0.1"/>
        \\        </body>
        \\    </worldbody>
        \\    <sensor>
        \\        <jointpos joint="slide"/>
        \\        <jointvel joint="slide"/>
        \\    </sensor>
        \\</mujoco>
    ;

    for (0..5) |_| {
        var scene = try parser.parseString(allocator, sensor_mjcf);
        try testing.expect(scene.sensor_config.count() > 0);
        scene.deinit();
    }
}

test "error handling allocation - no leaks on parse failure" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in error handling test!");
        }
    }
    const allocator = gpa.allocator();

    // Try parsing malformed XML - should not leak
    const malformed_xml = "<mujoco><body><body></mujoco>";
    const result = parser.parseString(allocator, malformed_xml);
    if (result) |scene| {
        // Even if parsing succeeded, ensure cleanup happens
        var s = scene;
        s.deinit();
    } else |_| {
        // Expected to fail, but should not leak
        // GPA deinit will verify no leaks
    }
}

test "repeated parse and cleanup - stress test" {
    var gpa = createLeakDetectingAllocator();
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            @panic("Memory leak detected in stress test!");
        }
    }
    const allocator = gpa.allocator();

    const ant_like_mjcf =
        \\<mujoco model="ant">
        \\    <option timestep="0.01"/>
        \\    <worldbody>
        \\        <geom type="plane" size="10 10 0.1"/>
        \\        <body name="torso" pos="0 0 0.75">
        \\            <geom type="sphere" size="0.25"/>
        \\            <body name="leg1" pos="0.25 0 0">
        \\                <joint name="hip1" type="hinge" axis="0 0 1"/>
        \\                <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.05"/>
        \\                <body name="ankle1" pos="0.2 0 0">
        \\                    <joint name="ankle1" type="hinge" axis="0 1 0"/>
        \\                    <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04"/>
        \\                </body>
        \\            </body>
        \\            <body name="leg2" pos="-0.25 0 0">
        \\                <joint name="hip2" type="hinge" axis="0 0 1"/>
        \\                <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.05"/>
        \\            </body>
        \\        </body>
        \\    </worldbody>
        \\    <actuator>
        \\        <motor joint="hip1" gear="50"/>
        \\        <motor joint="ankle1" gear="50"/>
        \\        <motor joint="hip2" gear="50"/>
        \\    </actuator>
        \\</mujoco>
    ;

    // Parse many times to catch accumulating leaks
    for (0..20) |_| {
        var scene = try parser.parseString(allocator, ant_like_mjcf);
        scene.deinit();
    }
}
