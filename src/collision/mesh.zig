//! Mesh geometry support for collision detection.
//! Uses convex hull approximation for efficient collision.

const std = @import("std");
const body = @import("../physics/body.zig");

/// A mesh composed of vertices and triangles.
pub const Mesh = struct {
    /// Vertex positions (xyz).
    vertices: []const [3]f32,
    /// Triangle indices (3 indices per triangle).
    indices: []const u32,
    /// Precomputed convex hull vertices for collision.
    hull_vertices: []const [3]f32,
    /// Precomputed AABB.
    aabb: body.AABB,
    /// Center of mass.
    center: [3]f32,
    /// Scale factor applied to vertices.
    scale: [3]f32,

    /// Allocator for cleanup.
    allocator: std.mem.Allocator,

    /// Load mesh from arrays (takes ownership of data).
    pub fn init(
        allocator: std.mem.Allocator,
        vertices: []const [3]f32,
        indices: []const u32,
        scale: [3]f32,
    ) !Mesh {
        // Copy and scale vertices
        const scaled_verts = try allocator.alloc([3]f32, vertices.len);
        errdefer allocator.free(scaled_verts);

        var min_pt: [3]f32 = .{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
        var max_pt: [3]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
        var center: [3]f32 = .{ 0, 0, 0 };

        for (vertices, 0..) |v, i| {
            scaled_verts[i] = .{
                v[0] * scale[0],
                v[1] * scale[1],
                v[2] * scale[2],
            };

            min_pt[0] = @min(min_pt[0], scaled_verts[i][0]);
            min_pt[1] = @min(min_pt[1], scaled_verts[i][1]);
            min_pt[2] = @min(min_pt[2], scaled_verts[i][2]);
            max_pt[0] = @max(max_pt[0], scaled_verts[i][0]);
            max_pt[1] = @max(max_pt[1], scaled_verts[i][1]);
            max_pt[2] = @max(max_pt[2], scaled_verts[i][2]);

            center[0] += scaled_verts[i][0];
            center[1] += scaled_verts[i][1];
            center[2] += scaled_verts[i][2];
        }

        const n = @as(f32, @floatFromInt(vertices.len));
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;

        // Copy indices
        const indices_copy = try allocator.dupe(u32, indices);
        errdefer allocator.free(indices_copy);

        // Compute convex hull (simplified: use extremal vertices)
        const hull = try computeConvexHull(allocator, scaled_verts);

        return Mesh{
            .vertices = scaled_verts,
            .indices = indices_copy,
            .hull_vertices = hull,
            .aabb = .{ .min = min_pt, .max = max_pt },
            .center = center,
            .scale = scale,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Mesh) void {
        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
        self.allocator.free(self.hull_vertices);
    }

    /// Get support point in given direction (for GJK).
    pub fn support(self: *const Mesh, direction: [3]f32) [3]f32 {
        var max_dot: f32 = -std.math.inf(f32);
        var best_vertex: [3]f32 = self.hull_vertices[0];

        for (self.hull_vertices) |v| {
            const d = v[0] * direction[0] + v[1] * direction[1] + v[2] * direction[2];
            if (d > max_dot) {
                max_dot = d;
                best_vertex = v;
            }
        }

        return best_vertex;
    }

    /// Compute approximate volume using AABB.
    pub fn volume(self: *const Mesh) f32 {
        const dx = self.aabb.max[0] - self.aabb.min[0];
        const dy = self.aabb.max[1] - self.aabb.min[1];
        const dz = self.aabb.max[2] - self.aabb.min[2];
        // Approximate as 60% of bounding box volume
        return 0.6 * dx * dy * dz;
    }

    /// Compute approximate inertia using bounding box.
    pub fn computeInertia(self: *const Mesh, mass: f32) [3]f32 {
        const dx = self.aabb.max[0] - self.aabb.min[0];
        const dy = self.aabb.max[1] - self.aabb.min[1];
        const dz = self.aabb.max[2] - self.aabb.min[2];
        const m12 = mass / 12.0;

        return .{
            m12 * (dy * dy + dz * dz),
            m12 * (dx * dx + dz * dz),
            m12 * (dx * dx + dy * dy),
        };
    }
};

/// Compute a simplified convex hull using extremal vertices.
fn computeConvexHull(allocator: std.mem.Allocator, vertices: []const [3]f32) ![]const [3]f32 {
    if (vertices.len == 0) {
        return try allocator.alloc([3]f32, 0);
    }

    // Find extremal vertices along each axis (6 points)
    var extremals: [6]usize = .{ 0, 0, 0, 0, 0, 0 };
    var extremal_values: [6]f32 = .{
        vertices[0][0], vertices[0][0], // min/max X
        vertices[0][1], vertices[0][1], // min/max Y
        vertices[0][2], vertices[0][2], // min/max Z
    };

    for (vertices, 0..) |v, i| {
        if (v[0] < extremal_values[0]) {
            extremal_values[0] = v[0];
            extremals[0] = i;
        }
        if (v[0] > extremal_values[1]) {
            extremal_values[1] = v[0];
            extremals[1] = i;
        }
        if (v[1] < extremal_values[2]) {
            extremal_values[2] = v[1];
            extremals[2] = i;
        }
        if (v[1] > extremal_values[3]) {
            extremal_values[3] = v[1];
            extremals[3] = i;
        }
        if (v[2] < extremal_values[4]) {
            extremal_values[4] = v[2];
            extremals[4] = i;
        }
        if (v[2] > extremal_values[5]) {
            extremal_values[5] = v[2];
            extremals[5] = i;
        }
    }

    // Also add diagonal extremals for better approximation
    var hull_list = std.ArrayList([3]f32).init(allocator);
    errdefer hull_list.deinit();

    // Add unique extremal vertices
    for (extremals) |idx| {
        const v = vertices[idx];
        var is_duplicate = false;
        for (hull_list.items) |existing| {
            if (approxEq(v, existing)) {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate) {
            try hull_list.append(v);
        }
    }

    // Add some additional vertices for better hull approximation
    // Sample diagonals
    const diag_dirs: [8][3]f32 = .{
        .{ 1, 1, 1 },
        .{ 1, 1, -1 },
        .{ 1, -1, 1 },
        .{ 1, -1, -1 },
        .{ -1, 1, 1 },
        .{ -1, 1, -1 },
        .{ -1, -1, 1 },
        .{ -1, -1, -1 },
    };

    for (diag_dirs) |dir| {
        var max_dot: f32 = -std.math.inf(f32);
        var best_idx: usize = 0;

        for (vertices, 0..) |v, i| {
            const d = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
            if (d > max_dot) {
                max_dot = d;
                best_idx = i;
            }
        }

        const v = vertices[best_idx];
        var is_duplicate = false;
        for (hull_list.items) |existing| {
            if (approxEq(v, existing)) {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate) {
            try hull_list.append(v);
        }
    }

    return try hull_list.toOwnedSlice();
}

fn approxEq(a: [3]f32, b: [3]f32) bool {
    const eps: f32 = 1e-6;
    return @abs(a[0] - b[0]) < eps and
        @abs(a[1] - b[1]) < eps and
        @abs(a[2] - b[2]) < eps;
}

/// Mesh asset storage for sharing meshes between geoms.
pub const MeshAsset = struct {
    name: []const u8,
    mesh: Mesh,
};

/// Load STL file (ASCII or binary).
pub fn loadSTL(allocator: std.mem.Allocator, path: []const u8, scale: [3]f32) !Mesh {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 50 * 1024 * 1024); // 50MB max
    defer allocator.free(content);

    // Check if binary or ASCII
    if (content.len > 80 and !std.mem.startsWith(u8, content, "solid")) {
        return loadBinarySTL(allocator, content, scale);
    } else {
        return loadAsciiSTL(allocator, content, scale);
    }
}

fn loadBinarySTL(allocator: std.mem.Allocator, content: []const u8, scale: [3]f32) !Mesh {
    if (content.len < 84) return error.InvalidFormat;

    // Skip 80-byte header
    const num_triangles = std.mem.readInt(u32, content[80..84], .little);
    const expected_size = 84 + num_triangles * 50;
    if (content.len < expected_size) return error.InvalidFormat;

    var vertices = std.ArrayList([3]f32).init(allocator);
    errdefer vertices.deinit();
    var indices = std.ArrayList(u32).init(allocator);
    errdefer indices.deinit();

    var offset: usize = 84;
    for (0..num_triangles) |_| {
        // Skip normal (12 bytes)
        offset += 12;

        // Read 3 vertices
        for (0..3) |_| {
            const x = @as(f32, @bitCast(std.mem.readInt(u32, content[offset..][0..4], .little)));
            const y = @as(f32, @bitCast(std.mem.readInt(u32, content[offset + 4 ..][0..4], .little)));
            const z = @as(f32, @bitCast(std.mem.readInt(u32, content[offset + 8 ..][0..4], .little)));
            offset += 12;

            const idx: u32 = @intCast(vertices.items.len);
            try vertices.append(.{ x, y, z });
            try indices.append(idx);
        }

        // Skip attribute byte count (2 bytes)
        offset += 2;
    }

    const owned_verts = try vertices.toOwnedSlice();
    errdefer allocator.free(owned_verts);
    const owned_indices = try indices.toOwnedSlice();
    errdefer allocator.free(owned_indices);

    const mesh = try Mesh.init(allocator, owned_verts, owned_indices, scale);
    allocator.free(owned_verts);
    allocator.free(owned_indices);

    return mesh;
}

fn loadAsciiSTL(allocator: std.mem.Allocator, content: []const u8, scale: [3]f32) !Mesh {
    var vertices = std.ArrayList([3]f32).init(allocator);
    errdefer vertices.deinit();
    var indices = std.ArrayList(u32).init(allocator);
    errdefer indices.deinit();

    var lines = std.mem.splitAny(u8, content, "\r\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "vertex")) {
            // Parse "vertex x y z"
            var parts = std.mem.tokenizeAny(u8, trimmed, " \t");
            _ = parts.next(); // Skip "vertex"

            const x = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;
            const y = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;
            const z = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;

            const idx: u32 = @intCast(vertices.items.len);
            try vertices.append(.{ x, y, z });
            try indices.append(idx);
        }
    }

    const owned_verts = try vertices.toOwnedSlice();
    errdefer allocator.free(owned_verts);
    const owned_indices = try indices.toOwnedSlice();
    errdefer allocator.free(owned_indices);

    const mesh = try Mesh.init(allocator, owned_verts, owned_indices, scale);
    allocator.free(owned_verts);
    allocator.free(owned_indices);

    return mesh;
}

/// Load simple OBJ file (vertices and faces only).
pub fn loadOBJ(allocator: std.mem.Allocator, path: []const u8, scale: [3]f32) !Mesh {
    const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 50 * 1024 * 1024);
    defer allocator.free(content);

    var vertices = std.ArrayList([3]f32).init(allocator);
    errdefer vertices.deinit();
    var indices = std.ArrayList(u32).init(allocator);
    errdefer indices.deinit();

    var lines = std.mem.splitAny(u8, content, "\r\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        if (std.mem.startsWith(u8, trimmed, "v ")) {
            // Parse vertex "v x y z"
            var parts = std.mem.tokenizeAny(u8, trimmed, " \t");
            _ = parts.next(); // Skip "v"

            const x = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;
            const y = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;
            const z = std.fmt.parseFloat(f32, parts.next() orelse continue) catch continue;

            try vertices.append(.{ x, y, z });
        } else if (std.mem.startsWith(u8, trimmed, "f ")) {
            // Parse face "f v1 v2 v3" or "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
            var parts = std.mem.tokenizeAny(u8, trimmed, " \t");
            _ = parts.next(); // Skip "f"

            var face_indices: [3]u32 = undefined;
            var idx_count: usize = 0;

            while (parts.next()) |part| {
                if (idx_count >= 3) break;

                // Handle "v" or "v/vt" or "v/vt/vn" or "v//vn" formats
                var sub_parts = std.mem.splitScalar(u8, part, '/');
                const v_str = sub_parts.next() orelse continue;
                const v_idx = std.fmt.parseInt(i32, v_str, 10) catch continue;

                // OBJ indices are 1-based and can be negative
                const abs_idx: u32 = if (v_idx > 0)
                    @intCast(v_idx - 1)
                else
                    @intCast(@as(i32, @intCast(vertices.items.len)) + v_idx);

                face_indices[idx_count] = abs_idx;
                idx_count += 1;
            }

            if (idx_count >= 3) {
                try indices.append(face_indices[0]);
                try indices.append(face_indices[1]);
                try indices.append(face_indices[2]);
            }
        }
    }

    const owned_verts = try vertices.toOwnedSlice();
    errdefer allocator.free(owned_verts);
    const owned_indices = try indices.toOwnedSlice();
    errdefer allocator.free(owned_indices);

    const mesh = try Mesh.init(allocator, owned_verts, owned_indices, scale);
    allocator.free(owned_verts);
    allocator.free(owned_indices);

    return mesh;
}
