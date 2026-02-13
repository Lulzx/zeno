//! GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm)
//! for convex collision detection and penetration depth computation.

const std = @import("std");
const constants = @import("../physics/constants.zig");
const body = @import("../physics/body.zig");
const primitives = @import("primitives.zig");
const mesh_mod = @import("mesh.zig");

/// Maximum GJK iterations before giving up.
const GJK_MAX_ITERATIONS: u32 = 64;
/// Maximum EPA iterations before giving up.
const EPA_MAX_ITERATIONS: u32 = 64;
/// Maximum faces in the EPA polytope.
const EPA_MAX_FACES: usize = 128;
/// Tolerance for GJK termination.
const GJK_TOLERANCE: f32 = 1e-6;
/// Tolerance for EPA termination.
const EPA_TOLERANCE: f32 = 1e-4;

/// Result from GJK algorithm.
pub const GjkResult = struct {
    /// Whether the shapes intersect.
    intersect: bool,
    /// Simplex vertices (in Minkowski difference space).
    simplex: [4][3]f32,
    /// Number of simplex vertices (1-4).
    simplex_count: u32,
    /// Support points on shape A corresponding to simplex vertices.
    support_a: [4][3]f32,
    /// Support points on shape B corresponding to simplex vertices.
    support_b: [4][3]f32,
};

/// Result from EPA algorithm.
pub const EpaResult = struct {
    /// Contact normal (from A to B).
    normal: [3]f32,
    /// Penetration depth.
    depth: f32,
    /// Contact point on shape A.
    point_a: [3]f32,
    /// Contact point on shape B.
    point_b: [3]f32,
};

/// Shape interface for GJK support function.
pub const ConvexShape = struct {
    shape_type: ShapeType,
    /// Pointer to geometry data.
    geom: *const primitives.Geom,
    /// World transform.
    transform: *const body.Transform,

    pub const ShapeType = enum {
        sphere,
        capsule,
        box_shape,
        mesh,
    };

    /// Compute the support point (furthest point in a given direction).
    pub fn support(self: *const ConvexShape, dir: [3]f32) [3]f32 {
        return switch (self.shape_type) {
            .sphere => sphereSupport(self.geom, self.transform, dir),
            .capsule => capsuleSupport(self.geom, self.transform, dir),
            .box_shape => boxSupport(self.geom, self.transform, dir),
            .mesh => meshSupport(self.geom, self.transform, dir),
        };
    }
};

/// Sphere support function.
fn sphereSupport(geom: *const primitives.Geom, transform: *const body.Transform, dir: [3]f32) [3]f32 {
    const center = geom.getWorldCenter(transform);
    const radius = geom.getRadius();
    const n = safeNormalize(dir);
    return .{
        center[0] + n[0] * radius,
        center[1] + n[1] * radius,
        center[2] + n[2] * radius,
    };
}

/// Capsule support function.
fn capsuleSupport(geom: *const primitives.Geom, transform: *const body.Transform, dir: [3]f32) [3]f32 {
    const center = geom.getWorldCenter(transform);
    const radius = geom.getRadius();
    const half_len = geom.getHalfLength();
    const axis = transform.transformVector(.{ 0, 0, 1 });
    const n = safeNormalize(dir);

    // Pick the endpoint furthest along dir
    const d = dot3(axis, dir);
    const endpoint = if (d >= 0)
        add3(center, scale3(axis, half_len))
    else
        sub3(center, scale3(axis, half_len));

    return .{
        endpoint[0] + n[0] * radius,
        endpoint[1] + n[1] * radius,
        endpoint[2] + n[2] * radius,
    };
}

/// Box support function.
fn boxSupport(geom: *const primitives.Geom, transform: *const body.Transform, dir: [3]f32) [3]f32 {
    const half = geom.getHalfExtents();

    // Transform direction to local space
    const inv = transform.inverse();
    const local_dir = inv.transformVector(dir);

    // Pick the corner furthest along the local direction
    const local_support: [3]f32 = .{
        if (local_dir[0] >= 0) half[0] else -half[0],
        if (local_dir[1] >= 0) half[1] else -half[1],
        if (local_dir[2] >= 0) half[2] else -half[2],
    };

    return transform.transformPoint(local_support);
}

/// Mesh support function (uses precomputed convex hull).
fn meshSupport(geom: *const primitives.Geom, transform: *const body.Transform, dir: [3]f32) [3]f32 {
    const m = geom.mesh_ptr orelse return transform.transformPoint(geom.local_pos);

    // Transform direction to mesh local space
    const inv = transform.inverse();
    const local_dir = inv.transformVector(dir);

    // Find furthest hull vertex in this direction
    const local_support = m.support(local_dir);

    return transform.transformPoint(local_support);
}

/// Compute support point in Minkowski difference (A - B).
fn minkowskiSupport(
    shape_a: *const ConvexShape,
    shape_b: *const ConvexShape,
    dir: [3]f32,
) struct { point: [3]f32, sa: [3]f32, sb: [3]f32 } {
    const sa = shape_a.support(dir);
    const sb = shape_b.support(negate3(dir));
    return .{
        .point = sub3(sa, sb),
        .sa = sa,
        .sb = sb,
    };
}

/// Run GJK algorithm to determine if two convex shapes intersect.
pub fn gjk(shape_a: *const ConvexShape, shape_b: *const ConvexShape) GjkResult {
    var result = GjkResult{
        .intersect = false,
        .simplex = undefined,
        .simplex_count = 0,
        .support_a = undefined,
        .support_b = undefined,
    };

    // Initial direction: from center of A to center of B
    const center_a = shape_a.geom.getWorldCenter(shape_a.transform);
    const center_b = shape_b.geom.getWorldCenter(shape_b.transform);
    var dir = sub3(center_b, center_a);
    if (dot3(dir, dir) < GJK_TOLERANCE) {
        dir = .{ 1, 0, 0 };
    }

    // Get first support point
    const first = minkowskiSupport(shape_a, shape_b, dir);
    result.simplex[0] = first.point;
    result.support_a[0] = first.sa;
    result.support_b[0] = first.sb;
    result.simplex_count = 1;

    // Search direction toward origin
    dir = negate3(first.point);

    for (0..GJK_MAX_ITERATIONS) |_| {
        const new_support = minkowskiSupport(shape_a, shape_b, dir);

        // If new point doesn't pass the origin, no intersection
        if (dot3(new_support.point, dir) < -GJK_TOLERANCE) {
            result.intersect = false;
            return result;
        }

        // Add point to simplex
        const idx = result.simplex_count;
        result.simplex[idx] = new_support.point;
        result.support_a[idx] = new_support.sa;
        result.support_b[idx] = new_support.sb;
        result.simplex_count += 1;

        // Evolve simplex
        switch (result.simplex_count) {
            2 => {
                if (doSimplex2(&result.simplex, &result.support_a, &result.support_b, &result.simplex_count, &dir)) {
                    result.intersect = true;
                    return result;
                }
            },
            3 => {
                if (doSimplex3(&result.simplex, &result.support_a, &result.support_b, &result.simplex_count, &dir)) {
                    result.intersect = true;
                    return result;
                }
            },
            4 => {
                if (doSimplex4(&result.simplex, &result.support_a, &result.support_b, &result.simplex_count, &dir)) {
                    result.intersect = true;
                    return result;
                }
            },
            else => unreachable,
        }
    }

    // Max iterations reached; assume no intersection
    return result;
}

/// Line case: simplex has 2 points (A is newest).
fn doSimplex2(
    simplex: *[4][3]f32,
    support_a: *[4][3]f32,
    support_b: *[4][3]f32,
    count: *u32,
    dir: *[3]f32,
) bool {
    _ = support_a;
    _ = support_b;
    const a = simplex[1]; // newest
    const b = simplex[0];

    const ab = sub3(b, a);
    const ao = negate3(a);

    if (dot3(ab, ao) > 0) {
        // Origin is in the region of AB
        dir.* = tripleProduct(ab, ao, ab);
        if (dot3(dir.*, dir.*) < GJK_TOLERANCE) {
            // Degenerate: origin is on the line segment
            dir.* = perpendicular(ab);
        }
    } else {
        // Origin is in the region of A
        simplex[0] = a;
        count.* = 1;
        dir.* = ao;
    }

    return false;
}

/// Triangle case: simplex has 3 points (A is newest).
fn doSimplex3(
    simplex: *[4][3]f32,
    support_a: *[4][3]f32,
    support_b: *[4][3]f32,
    count: *u32,
    dir: *[3]f32,
) bool {
    const a = simplex[2]; // newest
    const b = simplex[1];
    const c = simplex[0];

    const ab = sub3(b, a);
    const ac = sub3(c, a);
    const ao = negate3(a);
    const abc_normal = cross3(ab, ac);

    // Check if origin is outside edge AB
    const ab_perp = cross3(ab, abc_normal);
    if (dot3(ab_perp, ao) > 0) {
        if (dot3(ab, ao) > 0) {
            // Region AB
            simplex[0] = b;
            simplex[1] = a;
            support_a[0] = support_a[1];
            support_b[0] = support_b[1];
            support_a[1] = support_a[2];
            support_b[1] = support_b[2];
            count.* = 2;
            dir.* = tripleProduct(ab, ao, ab);
            if (dot3(dir.*, dir.*) < GJK_TOLERANCE) {
                dir.* = perpendicular(ab);
            }
        } else {
            // Region A
            simplex[0] = a;
            support_a[0] = support_a[2];
            support_b[0] = support_b[2];
            count.* = 1;
            dir.* = ao;
        }
        return false;
    }

    // Check if origin is outside edge AC
    const ac_perp = cross3(abc_normal, ac);
    if (dot3(ac_perp, ao) > 0) {
        if (dot3(ac, ao) > 0) {
            // Region AC
            simplex[0] = c;
            simplex[1] = a;
            support_a[0] = support_a[0]; // c stays
            support_b[0] = support_b[0];
            support_a[1] = support_a[2];
            support_b[1] = support_b[2];
            count.* = 2;
            dir.* = tripleProduct(ac, ao, ac);
            if (dot3(dir.*, dir.*) < GJK_TOLERANCE) {
                dir.* = perpendicular(ac);
            }
        } else {
            // Region A
            simplex[0] = a;
            support_a[0] = support_a[2];
            support_b[0] = support_b[2];
            count.* = 1;
            dir.* = ao;
        }
        return false;
    }

    // Origin is above or below the triangle
    if (dot3(abc_normal, ao) > 0) {
        // Above triangle - keep winding
        dir.* = abc_normal;
    } else {
        // Below triangle - flip winding
        const tmp = simplex[0];
        simplex[0] = simplex[1];
        simplex[1] = tmp;
        const tmp_sa = support_a[0];
        support_a[0] = support_a[1];
        support_a[1] = tmp_sa;
        const tmp_sb = support_b[0];
        support_b[0] = support_b[1];
        support_b[1] = tmp_sb;
        dir.* = negate3(abc_normal);
    }

    return false;
}

/// Tetrahedron case: simplex has 4 points (A is newest).
fn doSimplex4(
    simplex: *[4][3]f32,
    support_a: *[4][3]f32,
    support_b: *[4][3]f32,
    count: *u32,
    dir: *[3]f32,
) bool {
    const a = simplex[3]; // newest
    const b = simplex[2];
    const c = simplex[1];
    const d = simplex[0];

    const ab = sub3(b, a);
    const ac = sub3(c, a);
    const ad = sub3(d, a);
    const ao = negate3(a);

    // Check each face of the tetrahedron
    const abc = cross3(ab, ac);
    const acd = cross3(ac, ad);
    const adb = cross3(ad, ab);

    // Face ABC
    if (dot3(abc, ao) > 0) {
        // Origin is outside face ABC - reduce to triangle ABC
        simplex[0] = c;
        simplex[1] = b;
        simplex[2] = a;
        support_a[0] = support_a[1];
        support_b[0] = support_b[1];
        support_a[1] = support_a[2];
        support_b[1] = support_b[2];
        support_a[2] = support_a[3];
        support_b[2] = support_b[3];
        count.* = 3;
        dir.* = abc;
        return false;
    }

    // Face ACD
    if (dot3(acd, ao) > 0) {
        simplex[0] = d;
        simplex[1] = c;
        simplex[2] = a;
        support_a[0] = support_a[0]; // d stays
        support_b[0] = support_b[0];
        support_a[1] = support_a[1]; // c stays
        support_b[1] = support_b[1];
        support_a[2] = support_a[3];
        support_b[2] = support_b[3];
        count.* = 3;
        dir.* = acd;
        return false;
    }

    // Face ADB
    if (dot3(adb, ao) > 0) {
        simplex[0] = b;
        simplex[1] = d;
        simplex[2] = a;
        support_a[0] = support_a[2]; // b
        support_b[0] = support_b[2];
        support_a[1] = support_a[0]; // d
        support_b[1] = support_b[0];
        support_a[2] = support_a[3]; // a
        support_b[2] = support_b[3];
        count.* = 3;
        dir.* = adb;
        return false;
    }

    // Origin is inside the tetrahedron
    return true;
}

/// EPA face for the expanding polytope.
const EpaFace = struct {
    indices: [3]u32,
    normal: [3]f32,
    distance: f32,
};

/// Run EPA to find penetration depth and contact information.
/// Requires that GJK found an intersection (result has a tetrahedron simplex).
pub fn epa(
    shape_a: *const ConvexShape,
    shape_b: *const ConvexShape,
    gjk_result: *const GjkResult,
) ?EpaResult {
    if (!gjk_result.intersect or gjk_result.simplex_count < 4) return null;

    // Initialize polytope from GJK simplex
    var vertices: [256][3]f32 = undefined;
    var vert_a: [256][3]f32 = undefined;
    var vert_b: [256][3]f32 = undefined;
    var num_verts: u32 = 4;

    for (0..4) |i| {
        vertices[i] = gjk_result.simplex[i];
        vert_a[i] = gjk_result.support_a[i];
        vert_b[i] = gjk_result.support_b[i];
    }

    // Initialize faces from tetrahedron (4 triangles)
    var faces: [EPA_MAX_FACES]EpaFace = undefined;
    var num_faces: u32 = 0;

    // Create initial faces with correct winding (normals pointing outward)
    const tet_faces = [4][3]u32{
        .{ 0, 1, 2 },
        .{ 0, 3, 1 },
        .{ 0, 2, 3 },
        .{ 1, 3, 2 },
    };

    for (tet_faces) |fi| {
        const n = faceNormal(vertices[fi[0]], vertices[fi[1]], vertices[fi[2]]);
        const d = dot3(n, vertices[fi[0]]);

        // Ensure normal points outward (away from origin)
        if (d < 0) {
            faces[num_faces] = .{
                .indices = .{ fi[0], fi[2], fi[1] },
                .normal = negate3(n),
                .distance = -d,
            };
        } else {
            faces[num_faces] = .{
                .indices = fi,
                .normal = n,
                .distance = d,
            };
        }
        num_faces += 1;
    }

    for (0..EPA_MAX_ITERATIONS) |_| {
        // Find the closest face to the origin
        var closest_face: u32 = 0;
        var closest_dist: f32 = std.math.inf(f32);

        for (0..num_faces) |i| {
            if (faces[i].distance < closest_dist) {
                closest_dist = faces[i].distance;
                closest_face = @intCast(i);
            }
        }

        // Get new support point in direction of closest face normal
        const search_dir = faces[closest_face].normal;
        const new_support = minkowskiSupport(shape_a, shape_b, search_dir);
        const new_dist = dot3(new_support.point, search_dir);

        // Check convergence
        if (new_dist - closest_dist < EPA_TOLERANCE) {
            // Compute contact points using barycentric coordinates on the closest face
            const face = faces[closest_face];
            const bary = barycentricOriginProjection(
                vertices[face.indices[0]],
                vertices[face.indices[1]],
                vertices[face.indices[2]],
            );

            const pa = add3(add3(
                scale3(vert_a[face.indices[0]], bary[0]),
                scale3(vert_a[face.indices[1]], bary[1]),
            ), scale3(vert_a[face.indices[2]], bary[2]));

            const pb = add3(add3(
                scale3(vert_b[face.indices[0]], bary[0]),
                scale3(vert_b[face.indices[1]], bary[1]),
            ), scale3(vert_b[face.indices[2]], bary[2]));

            return EpaResult{
                .normal = face.normal,
                .depth = closest_dist,
                .point_a = pa,
                .point_b = pb,
            };
        }

        // Add new vertex
        if (num_verts >= 256) break;
        vertices[num_verts] = new_support.point;
        vert_a[num_verts] = new_support.sa;
        vert_b[num_verts] = new_support.sb;
        const new_idx: u32 = num_verts;
        num_verts += 1;

        // Find and remove faces visible from the new point
        // An edge list to track the horizon
        var edges: [256][2]u32 = undefined;
        var num_edges: u32 = 0;

        var i: u32 = 0;
        while (i < num_faces) {
            if (dot3(faces[i].normal, sub3(new_support.point, vertices[faces[i].indices[0]])) > 0) {
                // Face is visible from new point - add edges to horizon
                addEdge(&edges, &num_edges, faces[i].indices[0], faces[i].indices[1]);
                addEdge(&edges, &num_edges, faces[i].indices[1], faces[i].indices[2]);
                addEdge(&edges, &num_edges, faces[i].indices[2], faces[i].indices[0]);

                // Remove face by swapping with last
                faces[i] = faces[num_faces - 1];
                num_faces -= 1;
            } else {
                i += 1;
            }
        }

        // Create new faces from horizon edges to new point
        for (0..num_edges) |ei| {
            if (num_faces >= EPA_MAX_FACES) break;

            const e = edges[ei];
            const n = faceNormal(vertices[e[0]], vertices[e[1]], vertices[new_idx]);
            const d = dot3(n, vertices[e[0]]);

            if (d < 0) {
                faces[num_faces] = .{
                    .indices = .{ e[1], e[0], new_idx },
                    .normal = negate3(n),
                    .distance = -d,
                };
            } else {
                faces[num_faces] = .{
                    .indices = .{ e[0], e[1], new_idx },
                    .normal = n,
                    .distance = d,
                };
            }
            num_faces += 1;
        }

        if (num_faces == 0) break;
    }

    // Fallback: return best guess from current polytope
    if (num_faces > 0) {
        var closest_face: u32 = 0;
        var closest_dist: f32 = std.math.inf(f32);
        for (0..num_faces) |fi| {
            if (faces[fi].distance < closest_dist) {
                closest_dist = faces[fi].distance;
                closest_face = @intCast(fi);
            }
        }
        const face = faces[closest_face];
        const bary = barycentricOriginProjection(
            vertices[face.indices[0]],
            vertices[face.indices[1]],
            vertices[face.indices[2]],
        );
        const pa = add3(add3(
            scale3(vert_a[face.indices[0]], bary[0]),
            scale3(vert_a[face.indices[1]], bary[1]),
        ), scale3(vert_a[face.indices[2]], bary[2]));
        const pb = add3(add3(
            scale3(vert_b[face.indices[0]], bary[0]),
            scale3(vert_b[face.indices[1]], bary[1]),
        ), scale3(vert_b[face.indices[2]], bary[2]));

        return EpaResult{
            .normal = face.normal,
            .depth = closest_dist,
            .point_a = pa,
            .point_b = pb,
        };
    }

    return null;
}

/// Add an edge to the horizon edge list. If the reverse edge already exists, remove both.
fn addEdge(edges: *[256][2]u32, num_edges: *u32, a: u32, b: u32) void {
    // Check if reverse edge exists
    var i: u32 = 0;
    while (i < num_edges.*) {
        if (edges[i][0] == b and edges[i][1] == a) {
            // Remove by swapping with last
            edges[i] = edges[num_edges.* - 1];
            num_edges.* -= 1;
            return;
        }
        i += 1;
    }

    // Add new edge
    if (num_edges.* < 256) {
        edges[num_edges.*] = .{ a, b };
        num_edges.* += 1;
    }
}

/// Compute the normal of a triangle face (normalized).
fn faceNormal(a: [3]f32, b: [3]f32, c: [3]f32) [3]f32 {
    const ab = sub3(b, a);
    const ac = sub3(c, a);
    return safeNormalize(cross3(ab, ac));
}

/// Compute barycentric coordinates of the origin's projection onto a triangle.
fn barycentricOriginProjection(a: [3]f32, b: [3]f32, c: [3]f32) [3]f32 {
    const ab = sub3(b, a);
    const ac = sub3(c, a);
    const ao = negate3(a);

    const d00 = dot3(ab, ab);
    const d01 = dot3(ab, ac);
    const d11 = dot3(ac, ac);
    const d20 = dot3(ao, ab);
    const d21 = dot3(ao, ac);

    const denom = d00 * d11 - d01 * d01;
    if (@abs(denom) < GJK_TOLERANCE) {
        // Degenerate triangle
        return .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    }

    const v = (d11 * d20 - d01 * d21) / denom;
    const w = (d00 * d21 - d01 * d20) / denom;
    const u = 1.0 - v - w;

    // Clamp to valid range
    const cu = std.math.clamp(u, 0.0, 1.0);
    const cv = std.math.clamp(v, 0.0, 1.0);
    const cw = std.math.clamp(w, 0.0, 1.0);

    // Renormalize
    const total = cu + cv + cw;
    if (total < GJK_TOLERANCE) {
        return .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    }

    return .{ cu / total, cv / total, cw / total };
}

// ==== Vector math helpers ====

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn cross3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn add3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

fn sub3(a: [3]f32, b: [3]f32) [3]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn scale3(v: [3]f32, s: f32) [3]f32 {
    return .{ v[0] * s, v[1] * s, v[2] * s };
}

fn negate3(v: [3]f32) [3]f32 {
    return .{ -v[0], -v[1], -v[2] };
}

fn safeNormalize(v: [3]f32) [3]f32 {
    const len = @sqrt(dot3(v, v));
    if (len < GJK_TOLERANCE) return .{ 0, 0, 1 };
    return scale3(v, 1.0 / len);
}

/// Triple product: (a x b) x c = b(c.a) - a(c.b)
fn tripleProduct(a: [3]f32, b: [3]f32, c: [3]f32) [3]f32 {
    return cross3(cross3(a, b), c);
}

/// Find a vector perpendicular to the given one.
fn perpendicular(v: [3]f32) [3]f32 {
    if (@abs(v[0]) < @abs(v[1])) {
        return safeNormalize(cross3(v, .{ 1, 0, 0 }));
    } else {
        return safeNormalize(cross3(v, .{ 0, 1, 0 }));
    }
}

// ==== Tests ====

test "gjk sphere-sphere overlap" {
    const t_a = body.Transform.fromPosition(.{ 0, 0, 0 });
    const t_b = body.Transform.fromPosition(.{ 1, 0, 0 });

    var geom_a = primitives.Geom.sphere(1.0);
    geom_a.local_pos = .{ 0, 0, 0 };
    var geom_b = primitives.Geom.sphere(1.0);
    geom_b.local_pos = .{ 0, 0, 0 };

    const sa = ConvexShape{ .shape_type = .sphere, .geom = &geom_a, .transform = &t_a };
    const sb = ConvexShape{ .shape_type = .sphere, .geom = &geom_b, .transform = &t_b };

    const result = gjk(&sa, &sb);
    try std.testing.expect(result.intersect);
}

test "gjk sphere-sphere no overlap" {
    const t_a = body.Transform.fromPosition(.{ 0, 0, 0 });
    const t_b = body.Transform.fromPosition(.{ 5, 0, 0 });

    var geom_a = primitives.Geom.sphere(1.0);
    geom_a.local_pos = .{ 0, 0, 0 };
    var geom_b = primitives.Geom.sphere(1.0);
    geom_b.local_pos = .{ 0, 0, 0 };

    const sa = ConvexShape{ .shape_type = .sphere, .geom = &geom_a, .transform = &t_a };
    const sb = ConvexShape{ .shape_type = .sphere, .geom = &geom_b, .transform = &t_b };

    const result = gjk(&sa, &sb);
    try std.testing.expect(!result.intersect);
}

test "gjk+epa sphere-sphere penetration" {
    const t_a = body.Transform.fromPosition(.{ 0, 0, 0 });
    const t_b = body.Transform.fromPosition(.{ 1, 0, 0 });

    var geom_a = primitives.Geom.sphere(1.0);
    geom_a.local_pos = .{ 0, 0, 0 };
    var geom_b = primitives.Geom.sphere(1.0);
    geom_b.local_pos = .{ 0, 0, 0 };

    const sa = ConvexShape{ .shape_type = .sphere, .geom = &geom_a, .transform = &t_a };
    const sb = ConvexShape{ .shape_type = .sphere, .geom = &geom_b, .transform = &t_b };

    const gjk_result = gjk(&sa, &sb);
    try std.testing.expect(gjk_result.intersect);

    if (epa(&sa, &sb, &gjk_result)) |epa_result| {
        // Penetration depth should be approximately 1.0 (2 * radius - distance)
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), epa_result.depth, 0.1);
        // Normal should roughly point along X axis
        try std.testing.expect(@abs(epa_result.normal[0]) > 0.8);
    }
}

test "gjk box-sphere overlap" {
    const t_a = body.Transform.fromPosition(.{ 0, 0, 0 });
    const t_b = body.Transform.fromPosition(.{ 1.2, 0, 0 });

    var geom_a = primitives.Geom.box(1.0, 1.0, 1.0);
    geom_a.local_pos = .{ 0, 0, 0 };
    var geom_b = primitives.Geom.sphere(0.5);
    geom_b.local_pos = .{ 0, 0, 0 };

    const sa = ConvexShape{ .shape_type = .box_shape, .geom = &geom_a, .transform = &t_a };
    const sb = ConvexShape{ .shape_type = .sphere, .geom = &geom_b, .transform = &t_b };

    const result = gjk(&sa, &sb);
    try std.testing.expect(result.intersect);
}

test "gjk box-sphere no overlap" {
    const t_a = body.Transform.fromPosition(.{ 0, 0, 0 });
    const t_b = body.Transform.fromPosition(.{ 3, 0, 0 });

    var geom_a = primitives.Geom.box(1.0, 1.0, 1.0);
    geom_a.local_pos = .{ 0, 0, 0 };
    var geom_b = primitives.Geom.sphere(0.5);
    geom_b.local_pos = .{ 0, 0, 0 };

    const sa = ConvexShape{ .shape_type = .box_shape, .geom = &geom_a, .transform = &t_a };
    const sb = ConvexShape{ .shape_type = .sphere, .geom = &geom_b, .transform = &t_b };

    const result = gjk(&sa, &sb);
    try std.testing.expect(!result.intersect);
}
