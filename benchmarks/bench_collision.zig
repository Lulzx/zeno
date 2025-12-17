//! Benchmark for collision detection performance.

const std = @import("std");
const zeno = @import("zeno");

const primitives = zeno.collision.primitives;
const narrow_phase = zeno.collision.narrow_phase;
const broad_phase = zeno.collision.broad_phase;
const body = zeno.physics.body;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n=== Zeno Collision Benchmark ===\n\n", .{});

    // Narrow phase benchmarks
    benchmarkSphereSphere();
    benchmarkSphereCapsule();
    benchmarkCapsuleCapsule();

    // Broad phase benchmarks
    try benchmarkSpatialHash(allocator);
}

fn benchmarkSphereSphere() void {
    const iterations = 1_000_000;

    var geom_a = primitives.Geom.sphere(0.5);
    var geom_b = primitives.Geom.sphere(0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 0.8, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    var timer = std.time.Timer.start() catch unreachable;

    var collisions: u32 = 0;
    for (0..iterations) |_| {
        if (narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b)) |_| {
            collisions += 1;
        }
    }

    const elapsed = timer.read();
    const ns_per_test = elapsed / iterations;
    const tests_per_sec = @as(f64, 1e9) / @as(f64, @floatFromInt(ns_per_test));

    std.debug.print("Sphere-Sphere Collision:\n", .{});
    std.debug.print("  {d:.0} tests/sec ({d:.0} ns/test)\n", .{ tests_per_sec, @as(f64, @floatFromInt(ns_per_test)) });
    std.debug.print("  Collisions: {}/{}\n\n", .{ collisions, iterations });
}

fn benchmarkSphereCapsule() void {
    const iterations = 1_000_000;

    var geom_a = primitives.Geom.sphere(0.5);
    var geom_b = primitives.Geom.capsule(0.1, 0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 0.5, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    var timer = std.time.Timer.start() catch unreachable;

    var collisions: u32 = 0;
    for (0..iterations) |_| {
        if (narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b)) |_| {
            collisions += 1;
        }
    }

    const elapsed = timer.read();
    const ns_per_test = elapsed / iterations;
    const tests_per_sec = @as(f64, 1e9) / @as(f64, @floatFromInt(ns_per_test));

    std.debug.print("Sphere-Capsule Collision:\n", .{});
    std.debug.print("  {d:.0} tests/sec ({d:.0} ns/test)\n", .{ tests_per_sec, @as(f64, @floatFromInt(ns_per_test)) });
    std.debug.print("  Collisions: {}/{}\n\n", .{ collisions, iterations });
}

fn benchmarkCapsuleCapsule() void {
    const iterations = 1_000_000;

    var geom_a = primitives.Geom.capsule(0.1, 0.5);
    var geom_b = primitives.Geom.capsule(0.1, 0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 0.15, 0.15, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    var timer = std.time.Timer.start() catch unreachable;

    var collisions: u32 = 0;
    for (0..iterations) |_| {
        if (narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b)) |_| {
            collisions += 1;
        }
    }

    const elapsed = timer.read();
    const ns_per_test = elapsed / iterations;
    const tests_per_sec = @as(f64, 1e9) / @as(f64, @floatFromInt(ns_per_test));

    std.debug.print("Capsule-Capsule Collision:\n", .{});
    std.debug.print("  {d:.0} tests/sec ({d:.0} ns/test)\n", .{ tests_per_sec, @as(f64, @floatFromInt(ns_per_test)) });
    std.debug.print("  Collisions: {}/{}\n\n", .{ collisions, iterations });
}

fn benchmarkSpatialHash(allocator: std.mem.Allocator) !void {
    const num_objects = 10000;
    const iterations = 1000;

    std.debug.print("Spatial Hash ({} objects):\n", .{num_objects});

    var hash = try broad_phase.SpatialHash.init(allocator, num_objects, 1.0, 64);
    defer hash.deinit();

    // Generate random positions
    var positions: [num_objects][3]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (&positions) |*pos| {
        pos.* = .{
            random.float(f32) * 100.0,
            random.float(f32) * 100.0,
            random.float(f32) * 100.0,
        };
    }

    // Benchmark update
    var timer = std.time.Timer.start() catch unreachable;

    for (0..iterations) |_| {
        hash.update(&positions, num_objects);
    }

    const elapsed = timer.read();
    const ms_per_update = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations)) / 1e6;

    std.debug.print("  Update: {d:.3} ms ({d:.0} updates/sec)\n", .{
        ms_per_update,
        1000.0 / ms_per_update,
    });

    std.debug.print("\n", .{});
}
