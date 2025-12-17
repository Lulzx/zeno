//! Tests for collision detection.

const std = @import("std");
const testing = std.testing;

const primitives = @import("zeno").collision.primitives;
const narrow_phase = @import("zeno").collision.narrow_phase;
const broad_phase = @import("zeno").collision.broad_phase;
const body = @import("zeno").physics.body;

test "sphere geom creation" {
    const sphere = primitives.Geom.sphere(0.5);

    try testing.expectEqual(primitives.GeomType.sphere, sphere.geom_type);
    try testing.expectApproxEqAbs(@as(f32, 0.5), sphere.getRadius(), 0.001);
}

test "capsule geom creation" {
    const capsule = primitives.Geom.capsule(0.1, 0.5);

    try testing.expectEqual(primitives.GeomType.capsule, capsule.geom_type);
    try testing.expectApproxEqAbs(@as(f32, 0.1), capsule.getRadius(), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.5), capsule.getHalfLength(), 0.001);
}

test "box geom creation" {
    const geom = primitives.Geom.box(1.0, 2.0, 3.0);

    try testing.expectEqual(primitives.GeomType.box, geom.geom_type);
    const extents = geom.getHalfExtents();
    try testing.expectApproxEqAbs(@as(f32, 1.0), extents[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), extents[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), extents[2], 0.001);
}

test "sphere local aabb" {
    var sphere = primitives.Geom.sphere(1.0);
    sphere.local_pos = .{ 0, 0, 0 };

    const aabb = sphere.computeLocalAABB();

    try testing.expectApproxEqAbs(@as(f32, -1.0), aabb.min[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aabb.max[0], 0.001);
}

test "sphere volume" {
    const sphere = primitives.Geom.sphere(1.0);
    const vol = sphere.volume();

    // Volume of unit sphere = 4/3 * pi * r^3
    const expected = 4.0 / 3.0 * std.math.pi;
    try testing.expectApproxEqAbs(expected, vol, 0.01);
}

test "sphere-sphere collision" {
    var geom_a = primitives.Geom.sphere(0.5);
    var geom_b = primitives.Geom.sphere(0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 0.8, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    const result = narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b);

    try testing.expect(result != null);
    if (result) |r| {
        try testing.expectApproxEqAbs(@as(f32, 0.2), r.penetration, 0.01);
        try testing.expectApproxEqAbs(@as(f32, 1.0), r.normal[0], 0.01);
    }
}

test "sphere-sphere no collision" {
    var geom_a = primitives.Geom.sphere(0.5);
    var geom_b = primitives.Geom.sphere(0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 2.0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    const result = narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b);

    try testing.expect(result == null);
}

test "spatial hash cell id" {
    var hash = try broad_phase.SpatialHash.init(testing.allocator, 100, 1.0, 16);
    defer hash.deinit();

    const id1 = hash.cellId(.{ 0.5, 0.5, 0.5 });
    const id2 = hash.cellId(.{ 1.5, 0.5, 0.5 });
    const id3 = hash.cellId(.{ 0.5, 0.5, 0.5 });

    // Same position should give same ID
    try testing.expectEqual(id1, id3);
    // Different cells should give different IDs
    try testing.expect(id1 != id2);
}

test "morton encoding" {
    // Morton code should interleave bits
    const m1 = broad_phase.mortonEncode(0, 0, 0);
    const m2 = broad_phase.mortonEncode(1, 0, 0);
    const m3 = broad_phase.mortonEncode(0, 1, 0);

    try testing.expectEqual(@as(u32, 0), m1);
    try testing.expect(m2 > 0);
    try testing.expect(m3 > 0);
    try testing.expect(m2 != m3);
}
