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

// Additional collision tests for better coverage

test "cylinder geom creation" {
    const cyl = primitives.Geom.cylinder(0.5, 1.0);

    try testing.expectEqual(primitives.GeomType.cylinder, cyl.geom_type);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cyl.getRadius(), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), cyl.getHalfLength(), 0.001);
}

test "plane geom creation" {
    const plane = primitives.Geom.plane(.{ 0, 0, 1 }, 0.0);

    try testing.expectEqual(primitives.GeomType.plane, plane.geom_type);
    try testing.expectApproxEqAbs(@as(f32, 0.0), plane.getPlaneOffset(), 0.001);
}

test "box local aabb" {
    var geom = primitives.Geom.box(1.0, 2.0, 3.0);
    geom.local_pos = .{ 0, 0, 0 };

    const aabb = geom.computeLocalAABB();

    try testing.expectApproxEqAbs(@as(f32, -1.0), aabb.min[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aabb.max[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -2.0), aabb.min[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), aabb.max[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -3.0), aabb.min[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), aabb.max[2], 0.001);
}

test "capsule local aabb" {
    var geom = primitives.Geom.capsule(0.5, 1.0);
    geom.local_pos = .{ 0, 0, 0 };

    const aabb = geom.computeLocalAABB();

    // Capsule with radius 0.5, half-length 1.0
    try testing.expectApproxEqAbs(@as(f32, -0.5), aabb.min[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.5), aabb.max[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, -1.5), aabb.min[2], 0.001); // half_length + radius
    try testing.expectApproxEqAbs(@as(f32, 1.5), aabb.max[2], 0.001);
}

test "box volume" {
    const geom = primitives.Geom.box(1.0, 2.0, 3.0);
    const vol = geom.volume();

    // Volume = 8 * half_x * half_y * half_z
    const expected: f32 = 8.0 * 1.0 * 2.0 * 3.0;
    try testing.expectApproxEqAbs(expected, vol, 0.01);
}

test "capsule volume" {
    const geom = primitives.Geom.capsule(1.0, 1.0);
    const vol = geom.volume();

    // Volume = cylinder + 2 hemispheres = pi*r^2*2h + 4/3*pi*r^3
    const cylinder_vol = std.math.pi * 1.0 * 1.0 * 2.0;
    const sphere_vol = 4.0 / 3.0 * std.math.pi * 1.0;
    const expected = cylinder_vol + sphere_vol;
    try testing.expectApproxEqAbs(expected, vol, 0.1);
}

test "sphere-plane collision" {
    var sphere = primitives.Geom.sphere(0.5);
    var plane = primitives.Geom.plane(.{ 0, 0, 1 }, 0.0);

    const transform_sphere = body.Transform{ .position = .{ 0, 0, 0.3 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_plane = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    const result = narrow_phase.detectCollision(&sphere, &transform_sphere, &plane, &transform_plane);

    try testing.expect(result != null);
    if (result) |r| {
        // Sphere at z=0.3 with radius 0.5 penetrates plane at z=0 by 0.2
        try testing.expectApproxEqAbs(@as(f32, 0.2), r.penetration, 0.01);
    }
}

test "sphere-plane no collision" {
    var sphere = primitives.Geom.sphere(0.5);
    var plane = primitives.Geom.plane(.{ 0, 0, 1 }, 0.0);

    const transform_sphere = body.Transform{ .position = .{ 0, 0, 1.0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_plane = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    const result = narrow_phase.detectCollision(&sphere, &transform_sphere, &plane, &transform_plane);

    try testing.expect(result == null);
}

test "geom default friction and restitution" {
    const sphere = primitives.Geom.sphere(1.0);

    try testing.expect(sphere.friction > 0.0);
    try testing.expect(sphere.friction <= 1.0);
    try testing.expect(sphere.restitution >= 0.0);
    try testing.expect(sphere.restitution <= 1.0);
}

test "geom collision groups" {
    var geom = primitives.Geom.sphere(1.0);
    geom.group = 1;
    geom.mask = 0xFFFFFFFE; // Don't collide with group 0

    try testing.expectEqual(@as(u32, 1), geom.group);
    try testing.expectEqual(@as(u32, 0xFFFFFFFE), geom.mask);
}

test "sphere touching exactly" {
    var geom_a = primitives.Geom.sphere(0.5);
    var geom_b = primitives.Geom.sphere(0.5);

    const transform_a = body.Transform{ .position = .{ 0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };
    const transform_b = body.Transform{ .position = .{ 1.0, 0, 0 }, .quaternion = .{ 0, 0, 0, 1 } };

    const result = narrow_phase.detectCollision(&geom_a, &transform_a, &geom_b, &transform_b);

    // Exactly touching - penetration should be 0 or very small
    if (result) |r| {
        try testing.expect(r.penetration <= 0.01);
    }
}

test "spatial hash update" {
    var hash = try broad_phase.SpatialHash.init(testing.allocator, 10, 1.0, 16);
    defer hash.deinit();

    // Add some positions
    const positions = [_][3]f32{
        .{ 0.5, 0.5, 0.5 },
        .{ 1.0, 1.0, 1.0 },
        .{ 10.5, 10.5, 10.5 },
    };

    hash.update(&positions, 3);

    // Should have updated cell IDs
    try testing.expect(hash.num_geoms == 3);
}

test "aabb intersects" {
    const a = body.AABB{ .min = .{ 0, 0, 0 }, .max = .{ 2, 2, 2 } };
    const b = body.AABB{ .min = .{ 1, 1, 1 }, .max = .{ 3, 3, 3 } };
    const c = body.AABB{ .min = .{ 5, 5, 5 }, .max = .{ 6, 6, 6 } };

    try testing.expect(a.intersects(&b));
    try testing.expect(b.intersects(&a));
    try testing.expect(!a.intersects(&c));
    try testing.expect(!c.intersects(&a));
}

test "aabb expand" {
    var aabb = body.AABB{ .min = .{ 0, 0, 0 }, .max = .{ 1, 1, 1 } };
    aabb.expand(0.5);

    try testing.expectApproxEqAbs(@as(f32, -0.5), aabb.min[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.5), aabb.max[0], 0.001);
}

test "aabb contains point" {
    const aabb = body.AABB{ .min = .{ 0, 0, 0 }, .max = .{ 1, 1, 1 } };

    try testing.expect(aabb.contains(.{ 0.5, 0.5, 0.5 }));
    try testing.expect(!aabb.contains(.{ 2.0, 0.5, 0.5 }));
    try testing.expect(!aabb.contains(.{ -0.1, 0.5, 0.5 }));
}

test "aabb merge" {
    const a = body.AABB{ .min = .{ 0, 0, 0 }, .max = .{ 1, 1, 1 } };
    const b = body.AABB{ .min = .{ 2, 2, 2 }, .max = .{ 3, 3, 3 } };

    const merged = a.merge(&b);

    try testing.expectApproxEqAbs(@as(f32, 0.0), merged.min[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3.0), merged.max[0], 0.001);
}
