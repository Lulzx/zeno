//! Narrow phase collision detection algorithms.
//! Computes exact contact points between geometric primitives.

const std = @import("std");
const primitives = @import("primitives.zig");
const contact = @import("../physics/contact.zig");
const body = @import("../physics/body.zig");
const constants = @import("../physics/constants.zig");

/// Contact result from narrow phase.
pub const ContactResult = struct {
    /// Contact point in world space.
    point: [3]f32,
    /// Contact normal (from A to B).
    normal: [3]f32,
    /// Penetration depth (positive = overlapping).
    penetration: f32,
    /// Local contact point on A.
    local_a: [3]f32,
    /// Local contact point on B.
    local_b: [3]f32,
    /// Is contact valid?
    valid: bool = true,
};

/// Detect collision between two geometries.
pub fn detectCollision(
    geom_a: *const primitives.Geom,
    transform_a: *const body.Transform,
    geom_b: *const primitives.Geom,
    transform_b: *const body.Transform,
) ?ContactResult {
    // Get world-space positions and orientations
    const pos_a = geom_a.getWorldCenter(transform_a);
    const pos_b = geom_b.getWorldCenter(transform_b);

    // Dispatch based on geometry types
    return switch (geom_a.geom_type) {
        .sphere => switch (geom_b.geom_type) {
            .sphere => sphereSphere(pos_a, geom_a.getRadius(), pos_b, geom_b.getRadius()),
            .capsule => sphereCapsule(pos_a, geom_a.getRadius(), pos_b, geom_b, transform_b),
            .box => sphereBox(pos_a, geom_a.getRadius(), pos_b, geom_b, transform_b),
            .plane => spherePlane(pos_a, geom_a.getRadius(), geom_b, transform_b),
            else => null,
        },
        .capsule => switch (geom_b.geom_type) {
            .sphere => blk: {
                if (sphereCapsule(pos_b, geom_b.getRadius(), pos_a, geom_a, transform_a)) |r| {
                    break :blk flipContact(r);
                }
                break :blk null;
            },
            .capsule => capsuleCapsule(pos_a, geom_a, transform_a, pos_b, geom_b, transform_b),
            .plane => capsulePlane(pos_a, geom_a, transform_a, geom_b, transform_b),
            else => null,
        },
        .box => switch (geom_b.geom_type) {
            .sphere => blk: {
                if (sphereBox(pos_b, geom_b.getRadius(), pos_a, geom_a, transform_a)) |r| {
                    break :blk flipContact(r);
                }
                break :blk null;
            },
            .plane => boxPlane(pos_a, geom_a, transform_a, geom_b, transform_b),
            .box => boxBox(pos_a, geom_a, transform_a, pos_b, geom_b, transform_b),
            else => null,
        },
        .plane => switch (geom_b.geom_type) {
            .sphere => blk: {
                if (spherePlane(pos_b, geom_b.getRadius(), geom_a, transform_a)) |r| {
                    break :blk flipContact(r);
                }
                break :blk null;
            },
            .capsule => blk: {
                if (capsulePlane(pos_b, geom_b, transform_b, geom_a, transform_a)) |r| {
                    break :blk flipContact(r);
                }
                break :blk null;
            },
            .box => blk: {
                if (boxPlane(pos_b, geom_b, transform_b, geom_a, transform_a)) |r| {
                    break :blk flipContact(r);
                }
                break :blk null;
            },
            else => null,
        },
        else => null,
    };
}

/// Sphere-sphere collision.
fn sphereSphere(pos_a: [3]f32, radius_a: f32, pos_b: [3]f32, radius_b: f32) ?ContactResult {
    const diff = sub(pos_b, pos_a);
    const dist_sq = dot(diff, diff);
    const min_dist = radius_a + radius_b;

    if (dist_sq >= min_dist * min_dist) {
        return null;
    }

    const dist = @sqrt(dist_sq);
    const normal = if (dist > constants.EPSILON)
        scale(diff, 1.0 / dist)
    else
        [3]f32{ 0, 0, 1 };

    const penetration = min_dist - dist;
    const point = add(pos_a, scale(normal, radius_a - penetration * 0.5));

    return ContactResult{
        .point = point,
        .normal = normal,
        .penetration = penetration,
        .local_a = scale(normal, radius_a),
        .local_b = scale(normal, -radius_b),
    };
}

/// Sphere-capsule collision.
fn sphereCapsule(
    sphere_pos: [3]f32,
    sphere_radius: f32,
    capsule_pos: [3]f32,
    capsule: *const primitives.Geom,
    capsule_transform: *const body.Transform,
) ?ContactResult {
    const capsule_radius = capsule.getRadius();
    const capsule_half_len = capsule.getHalfLength();

    // Capsule axis in world space (assuming Z-aligned locally)
    const axis = capsule_transform.transformVector(.{ 0, 0, 1 });

    // Capsule endpoints
    const p1 = sub(capsule_pos, scale(axis, capsule_half_len));
    const p2 = add(capsule_pos, scale(axis, capsule_half_len));

    // Find closest point on segment to sphere center
    const closest = closestPointOnSegment(sphere_pos, p1, p2);
    const diff = sub(sphere_pos, closest);
    const dist_sq = dot(diff, diff);
    const min_dist = sphere_radius + capsule_radius;

    if (dist_sq >= min_dist * min_dist) {
        return null;
    }

    const dist = @sqrt(dist_sq);
    const normal = if (dist > constants.EPSILON)
        scale(diff, 1.0 / dist)
    else
        [3]f32{ 0, 0, 1 };

    const penetration = min_dist - dist;
    const point = add(closest, scale(normal, capsule_radius - penetration * 0.5));

    return ContactResult{
        .point = point,
        .normal = normal,
        .penetration = penetration,
        .local_a = scale(normal, sphere_radius),
        .local_b = scale(normal, -capsule_radius),
    };
}

/// Sphere-plane collision.
fn spherePlane(
    sphere_pos: [3]f32,
    sphere_radius: f32,
    plane: *const primitives.Geom,
    plane_transform: *const body.Transform,
) ?ContactResult {
    // Plane normal in world space (Z-up locally)
    const normal = plane_transform.transformVector(.{ 0, 0, 1 });
    const plane_point = plane_transform.transformPoint(plane.local_pos);
    const plane_offset = plane.getPlaneOffset();

    // Signed distance from sphere center to plane
    const signed_dist = dot(sub(sphere_pos, plane_point), normal) - plane_offset;

    if (signed_dist >= sphere_radius) {
        return null;
    }

    const penetration = sphere_radius - signed_dist;
    const point = sub(sphere_pos, scale(normal, signed_dist + penetration * 0.5));

    return ContactResult{
        .point = point,
        .normal = normal,
        .penetration = penetration,
        .local_a = scale(normal, -sphere_radius),
        .local_b = .{ 0, 0, 0 },
    };
}

/// Sphere-box collision.
fn sphereBox(
    sphere_pos: [3]f32,
    sphere_radius: f32,
    _: [3]f32, // box_pos unused - we use box_transform
    box: *const primitives.Geom,
    box_transform: *const body.Transform,
) ?ContactResult {
    // Transform sphere center to box local space
    const inv_transform = box_transform.inverse();
    const local_sphere = inv_transform.transformPoint(sphere_pos);

    // Clamp to box extents
    const half_extents = box.getHalfExtents();
    const clamped: [3]f32 = .{
        std.math.clamp(local_sphere[0], -half_extents[0], half_extents[0]),
        std.math.clamp(local_sphere[1], -half_extents[1], half_extents[1]),
        std.math.clamp(local_sphere[2], -half_extents[2], half_extents[2]),
    };

    // Distance from clamped point to sphere center
    const diff = sub(local_sphere, clamped);
    const dist_sq = dot(diff, diff);

    if (dist_sq >= sphere_radius * sphere_radius) {
        return null;
    }

    const dist = @sqrt(dist_sq);

    // Normal in local space
    const local_normal = if (dist > constants.EPSILON)
        scale(diff, 1.0 / dist)
    else
        boxFaceNormal(local_sphere, half_extents);

    // Transform back to world space
    const world_closest = box_transform.transformPoint(clamped);
    const normal = box_transform.transformVector(local_normal);
    const penetration = sphere_radius - dist;
    const point = add(world_closest, scale(normal, penetration * 0.5));

    return ContactResult{
        .point = point,
        .normal = normal,
        .penetration = penetration,
        .local_a = scale(normal, -sphere_radius),
        .local_b = sub(clamped, box.local_pos),
    };
}

/// Capsule-capsule collision.
fn capsuleCapsule(
    pos_a: [3]f32,
    capsule_a: *const primitives.Geom,
    transform_a: *const body.Transform,
    pos_b: [3]f32,
    capsule_b: *const primitives.Geom,
    transform_b: *const body.Transform,
) ?ContactResult {
    const radius_a = capsule_a.getRadius();
    const radius_b = capsule_b.getRadius();
    const half_len_a = capsule_a.getHalfLength();
    const half_len_b = capsule_b.getHalfLength();

    // Capsule axes
    const axis_a = transform_a.transformVector(.{ 0, 0, 1 });
    const axis_b = transform_b.transformVector(.{ 0, 0, 1 });

    // Endpoints
    const a1 = sub(pos_a, scale(axis_a, half_len_a));
    const a2 = add(pos_a, scale(axis_a, half_len_a));
    const b1 = sub(pos_b, scale(axis_b, half_len_b));
    const b2 = add(pos_b, scale(axis_b, half_len_b));

    // Find closest points between segments
    const closest = closestPointsBetweenSegments(a1, a2, b1, b2);

    const diff = sub(closest.point_b, closest.point_a);
    const dist_sq = dot(diff, diff);
    const min_dist = radius_a + radius_b;

    if (dist_sq >= min_dist * min_dist) {
        return null;
    }

    const dist = @sqrt(dist_sq);
    const normal = if (dist > constants.EPSILON)
        scale(diff, 1.0 / dist)
    else
        [3]f32{ 0, 0, 1 };

    const penetration = min_dist - dist;
    const point_a_world = add(closest.point_a, scale(normal, radius_a));
    const point_b_world = sub(closest.point_b, scale(normal, radius_b));
    const point = scale(add(point_a_world, point_b_world), 0.5);

    return ContactResult{
        .point = point,
        .normal = normal,
        .penetration = penetration,
        .local_a = scale(normal, radius_a),
        .local_b = scale(normal, -radius_b),
    };
}

/// Capsule-plane collision.
fn capsulePlane(
    capsule_pos: [3]f32,
    capsule: *const primitives.Geom,
    capsule_transform: *const body.Transform,
    plane: *const primitives.Geom,
    plane_transform: *const body.Transform,
) ?ContactResult {
    const radius = capsule.getRadius();
    const half_len = capsule.getHalfLength();

    const axis = capsule_transform.transformVector(.{ 0, 0, 1 });
    const plane_normal = plane_transform.transformVector(.{ 0, 0, 1 });
    const plane_point = plane_transform.transformPoint(plane.local_pos);

    // Check both endpoints
    const p1 = sub(capsule_pos, scale(axis, half_len));
    const p2 = add(capsule_pos, scale(axis, half_len));

    const d1 = dot(sub(p1, plane_point), plane_normal) - radius;
    const d2 = dot(sub(p2, plane_point), plane_normal) - radius;

    if (d1 >= 0 and d2 >= 0) {
        return null;
    }

    // Use deepest point
    const deepest = if (d1 < d2) p1 else p2;
    const dist = @min(d1, d2);

    const point = sub(deepest, scale(plane_normal, dist * 0.5 + radius));

    return ContactResult{
        .point = point,
        .normal = plane_normal,
        .penetration = -dist,
        .local_a = scale(plane_normal, -radius),
        .local_b = .{ 0, 0, 0 },
    };
}

/// Box-plane collision.
fn boxPlane(
    _: [3]f32, // box_pos unused - we use box_transform
    box: *const primitives.Geom,
    box_transform: *const body.Transform,
    plane: *const primitives.Geom,
    plane_transform: *const body.Transform,
) ?ContactResult {
    const half_extents = box.getHalfExtents();
    const plane_normal = plane_transform.transformVector(.{ 0, 0, 1 });
    const plane_point = plane_transform.transformPoint(plane.local_pos);

    // Check all 8 corners
    const corners = [8][3]f32{
        .{ -half_extents[0], -half_extents[1], -half_extents[2] },
        .{ half_extents[0], -half_extents[1], -half_extents[2] },
        .{ -half_extents[0], half_extents[1], -half_extents[2] },
        .{ half_extents[0], half_extents[1], -half_extents[2] },
        .{ -half_extents[0], -half_extents[1], half_extents[2] },
        .{ half_extents[0], -half_extents[1], half_extents[2] },
        .{ -half_extents[0], half_extents[1], half_extents[2] },
        .{ half_extents[0], half_extents[1], half_extents[2] },
    };

    var min_dist: f32 = std.math.inf(f32);
    var deepest_corner: [3]f32 = undefined;
    var deepest_world: [3]f32 = undefined;

    for (corners) |corner| {
        const world_corner = box_transform.transformPoint(corner);
        const dist = dot(sub(world_corner, plane_point), plane_normal);

        if (dist < min_dist) {
            min_dist = dist;
            deepest_corner = corner;
            deepest_world = world_corner;
        }
    }

    if (min_dist >= 0) {
        return null;
    }

    return ContactResult{
        .point = deepest_world,
        .normal = plane_normal,
        .penetration = -min_dist,
        .local_a = deepest_corner,
        .local_b = .{ 0, 0, 0 },
    };
}

/// Box-box collision using SAT.
fn boxBox(
    pos_a: [3]f32,
    box_a: *const primitives.Geom,
    transform_a: *const body.Transform,
    pos_b: [3]f32,
    box_b: *const primitives.Geom,
    transform_b: *const body.Transform,
) ?ContactResult {
    const half_a = box_a.getHalfExtents();
    const half_b = box_b.getHalfExtents();

    // Box axes in world space
    const axes_a = [3][3]f32{
        transform_a.transformVector(.{ 1, 0, 0 }),
        transform_a.transformVector(.{ 0, 1, 0 }),
        transform_a.transformVector(.{ 0, 0, 1 }),
    };
    const axes_b = [3][3]f32{
        transform_b.transformVector(.{ 1, 0, 0 }),
        transform_b.transformVector(.{ 0, 1, 0 }),
        transform_b.transformVector(.{ 0, 0, 1 }),
    };

    const d = sub(pos_b, pos_a);

    var min_overlap: f32 = std.math.inf(f32);
    var best_axis: [3]f32 = .{ 0, 0, 1 };

    // Test 15 separating axes
    // 3 face axes of A
    for (0..3) |i| {
        if (testAxis(axes_a[i], d, axes_a, half_a, axes_b, half_b)) |overlap| {
            if (overlap < min_overlap) {
                min_overlap = overlap;
                best_axis = axes_a[i];
            }
        } else return null;
    }

    // 3 face axes of B
    for (0..3) |i| {
        if (testAxis(axes_b[i], d, axes_a, half_a, axes_b, half_b)) |overlap| {
            if (overlap < min_overlap) {
                min_overlap = overlap;
                best_axis = axes_b[i];
            }
        } else return null;
    }

    // 9 edge-edge axes
    for (0..3) |i| {
        for (0..3) |j| {
            const edge_axis = cross(axes_a[i], axes_b[j]);
            const len = @sqrt(dot(edge_axis, edge_axis));
            if (len < constants.EPSILON) continue;

            const normalized = scale(edge_axis, 1.0 / len);
            if (testAxis(normalized, d, axes_a, half_a, axes_b, half_b)) |overlap| {
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    best_axis = normalized;
                }
            } else return null;
        }
    }

    // Ensure normal points from A to B
    if (dot(best_axis, d) < 0) {
        best_axis = scale(best_axis, -1);
    }

    // Find contact point (simplified: use center between closest points)
    const point = add(pos_a, scale(d, 0.5));

    return ContactResult{
        .point = point,
        .normal = best_axis,
        .penetration = min_overlap,
        .local_a = .{ 0, 0, 0 },
        .local_b = .{ 0, 0, 0 },
    };
}

// Helper functions

fn testAxis(
    axis: [3]f32,
    d: [3]f32,
    axes_a: [3][3]f32,
    half_a: [3]f32,
    axes_b: [3][3]f32,
    half_b: [3]f32,
) ?f32 {
    const proj_a = half_a[0] * @abs(dot(axis, axes_a[0])) +
        half_a[1] * @abs(dot(axis, axes_a[1])) +
        half_a[2] * @abs(dot(axis, axes_a[2]));

    const proj_b = half_b[0] * @abs(dot(axis, axes_b[0])) +
        half_b[1] * @abs(dot(axis, axes_b[1])) +
        half_b[2] * @abs(dot(axis, axes_b[2]));

    const dist = @abs(dot(d, axis));

    if (dist > proj_a + proj_b) {
        return null; // Separating axis found
    }

    return proj_a + proj_b - dist;
}

fn closestPointOnSegment(point: [3]f32, seg_start: [3]f32, seg_end: [3]f32) [3]f32 {
    const seg = sub(seg_end, seg_start);
    const len_sq = dot(seg, seg);

    if (len_sq < constants.EPSILON) {
        return seg_start;
    }

    const t = std.math.clamp(dot(sub(point, seg_start), seg) / len_sq, 0.0, 1.0);
    return add(seg_start, scale(seg, t));
}

const ClosestPoints = struct {
    point_a: [3]f32,
    point_b: [3]f32,
};

fn closestPointsBetweenSegments(a1: [3]f32, a2: [3]f32, b1: [3]f32, b2: [3]f32) ClosestPoints {
    const d1 = sub(a2, a1);
    const d2 = sub(b2, b1);
    const r = sub(a1, b1);

    const a = dot(d1, d1);
    const e = dot(d2, d2);
    const f = dot(d2, r);

    var s: f32 = 0;
    var t: f32 = 0;

    if (a < constants.EPSILON and e < constants.EPSILON) {
        return .{ .point_a = a1, .point_b = b1 };
    }

    if (a < constants.EPSILON) {
        t = std.math.clamp(f / e, 0, 1);
    } else {
        const c = dot(d1, r);
        if (e < constants.EPSILON) {
            s = std.math.clamp(-c / a, 0, 1);
        } else {
            const b = dot(d1, d2);
            const denom = a * e - b * b;

            if (denom != 0) {
                s = std.math.clamp((b * f - c * e) / denom, 0, 1);
            }

            t = (b * s + f) / e;

            if (t < 0) {
                t = 0;
                s = std.math.clamp(-c / a, 0, 1);
            } else if (t > 1) {
                t = 1;
                s = std.math.clamp((b - c) / a, 0, 1);
            }
        }
    }

    return .{
        .point_a = add(a1, scale(d1, s)),
        .point_b = add(b1, scale(d2, t)),
    };
}

fn boxFaceNormal(point: [3]f32, half_extents: [3]f32) [3]f32 {
    // Find which face the point is closest to
    var max_dist: f32 = -std.math.inf(f32);
    var normal: [3]f32 = .{ 0, 0, 1 };

    for (0..3) |i| {
        const d_pos = half_extents[i] - point[i];
        const d_neg = half_extents[i] + point[i];

        if (d_pos > max_dist) {
            max_dist = d_pos;
            normal = .{ 0, 0, 0 };
            normal[i] = 1;
        }
        if (d_neg > max_dist) {
            max_dist = d_neg;
            normal = .{ 0, 0, 0 };
            normal[i] = -1;
        }
    }

    return normal;
}

fn flipContact(c: ContactResult) ContactResult {
    return .{
        .point = c.point,
        .normal = scale(c.normal, -1),
        .penetration = c.penetration,
        .local_a = c.local_b,
        .local_b = c.local_a,
    };
}

// Vector math helpers
fn dot(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn cross(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn add(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

fn sub(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn scale(v: [3]f32, s: f32) [3]f32 {
    return .{ v[0] * s, v[1] * s, v[2] * s };
}
