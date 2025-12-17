//! Collision primitive types and geometry definitions.

const std = @import("std");
const body = @import("../physics/body.zig");
const constants = @import("../physics/constants.zig");

const mesh_mod = @import("mesh.zig");

/// Heightfield terrain data.
/// Stores height samples in a regular grid for efficient terrain collision.
pub const Heightfield = struct {
    /// Height data (row-major, size = rows * cols).
    data: []const f32,
    /// Number of rows (Y direction).
    rows: u32,
    /// Number of columns (X direction).
    cols: u32,
    /// Grid spacing in X direction.
    spacing_x: f32,
    /// Grid spacing in Y direction.
    spacing_y: f32,
    /// Base height offset.
    base_height: f32 = 0.0,
    /// Scale factor for height values.
    height_scale: f32 = 1.0,

    /// Sample height at a world XY position.
    /// Returns interpolated height or null if outside bounds.
    pub fn sampleHeight(self: *const Heightfield, world_x: f32, world_y: f32) ?f32 {
        // Convert world coords to grid coords
        const half_width = @as(f32, @floatFromInt(self.cols - 1)) * self.spacing_x * 0.5;
        const half_height = @as(f32, @floatFromInt(self.rows - 1)) * self.spacing_y * 0.5;

        const grid_x = (world_x + half_width) / self.spacing_x;
        const grid_y = (world_y + half_height) / self.spacing_y;

        // Check bounds
        if (grid_x < 0 or grid_y < 0) return null;
        if (grid_x >= @as(f32, @floatFromInt(self.cols - 1)) or
            grid_y >= @as(f32, @floatFromInt(self.rows - 1)))
        {
            return null;
        }

        // Bilinear interpolation
        const ix: u32 = @intFromFloat(grid_x);
        const iy: u32 = @intFromFloat(grid_y);
        const fx = grid_x - @as(f32, @floatFromInt(ix));
        const fy = grid_y - @as(f32, @floatFromInt(iy));

        const h00 = self.getHeight(ix, iy);
        const h10 = self.getHeight(ix + 1, iy);
        const h01 = self.getHeight(ix, iy + 1);
        const h11 = self.getHeight(ix + 1, iy + 1);

        // Bilinear interpolation
        const h0 = h00 * (1 - fx) + h10 * fx;
        const h1 = h01 * (1 - fx) + h11 * fx;
        const height = h0 * (1 - fy) + h1 * fy;

        return height * self.height_scale + self.base_height;
    }

    /// Sample surface normal at a world XY position.
    pub fn sampleNormal(self: *const Heightfield, world_x: f32, world_y: f32) ?[3]f32 {
        const half_width = @as(f32, @floatFromInt(self.cols - 1)) * self.spacing_x * 0.5;
        const half_height = @as(f32, @floatFromInt(self.rows - 1)) * self.spacing_y * 0.5;

        const grid_x = (world_x + half_width) / self.spacing_x;
        const grid_y = (world_y + half_height) / self.spacing_y;

        if (grid_x < 1 or grid_y < 1) return null;
        if (grid_x >= @as(f32, @floatFromInt(self.cols - 2)) or
            grid_y >= @as(f32, @floatFromInt(self.rows - 2)))
        {
            return null;
        }

        const ix: u32 = @intFromFloat(grid_x);
        const iy: u32 = @intFromFloat(grid_y);

        // Compute gradient using central differences
        const hL = self.getHeight(ix - 1, iy) * self.height_scale;
        const hR = self.getHeight(ix + 1, iy) * self.height_scale;
        const hD = self.getHeight(ix, iy - 1) * self.height_scale;
        const hU = self.getHeight(ix, iy + 1) * self.height_scale;

        const dx = (hR - hL) / (2.0 * self.spacing_x);
        const dy = (hU - hD) / (2.0 * self.spacing_y);

        // Normal = normalize(-dx, -dy, 1)
        const len = @sqrt(dx * dx + dy * dy + 1.0);
        return .{ -dx / len, -dy / len, 1.0 / len };
    }

    /// Get height at grid coordinates (clamped to bounds).
    pub fn getHeight(self: *const Heightfield, x: u32, y: u32) f32 {
        const cx = @min(x, self.cols - 1);
        const cy = @min(y, self.rows - 1);
        return self.data[cy * self.cols + cx];
    }

    /// Get world-space bounds of the heightfield.
    pub fn getBounds(self: *const Heightfield) struct { min: [3]f32, max: [3]f32 } {
        const half_width = @as(f32, @floatFromInt(self.cols - 1)) * self.spacing_x * 0.5;
        const half_depth = @as(f32, @floatFromInt(self.rows - 1)) * self.spacing_y * 0.5;

        // Find min/max heights
        var min_h: f32 = std.math.inf(f32);
        var max_h: f32 = -std.math.inf(f32);
        for (self.data) |h| {
            min_h = @min(min_h, h);
            max_h = @max(max_h, h);
        }

        return .{
            .min = .{ -half_width, -half_depth, min_h * self.height_scale + self.base_height },
            .max = .{ half_width, half_depth, max_h * self.height_scale + self.base_height },
        };
    }
};

/// Geometry type enumeration.
pub const GeomType = enum(u8) {
    sphere = 0,
    capsule = 1,
    box = 2,
    plane = 3,
    cylinder = 4,
    mesh = 5,
    heightfield = 6,
};

/// Geometry definition.
pub const Geom = struct {
    /// Geometry type.
    geom_type: GeomType = .sphere,
    /// Parent body index.
    body_id: u32 = 0,
    /// Local position offset from body.
    local_pos: [3]f32 = .{ 0, 0, 0 },
    /// Local rotation (quaternion xyzw).
    local_quat: [4]f32 = .{ 0, 0, 0, 1 },
    /// Size parameters (meaning depends on type).
    size: [3]f32 = .{ 1, 1, 1 },
    /// Friction coefficient.
    friction: f32 = constants.DEFAULT_FRICTION,
    /// Restitution coefficient.
    restitution: f32 = constants.DEFAULT_RESTITUTION,
    /// Collision group.
    group: u32 = 0,
    /// Collision mask.
    mask: u32 = 0xFFFFFFFF,
    /// Is this geometry used for collision?
    collide: bool = true,
    /// Mass (for inertia calculation).
    mass: f32 = 1.0,
    /// Mesh asset index (for mesh type).
    mesh_id: u32 = 0,
    /// Optional pointer to mesh data (not owned).
    mesh_ptr: ?*const mesh_mod.Mesh = null,
    /// Heightfield asset index (for heightfield type).
    heightfield_id: u32 = 0,
    /// Optional pointer to heightfield data (not owned).
    heightfield_ptr: ?*const Heightfield = null,

    /// Create a sphere geometry.
    pub fn sphere(radius: f32) Geom {
        return .{
            .geom_type = .sphere,
            .size = .{ radius, 0, 0 },
        };
    }

    /// Create a capsule geometry.
    pub fn capsule(radius: f32, half_length: f32) Geom {
        return .{
            .geom_type = .capsule,
            .size = .{ radius, half_length, 0 },
        };
    }

    /// Create a box geometry.
    pub fn box(half_x: f32, half_y: f32, half_z: f32) Geom {
        return .{
            .geom_type = .box,
            .size = .{ half_x, half_y, half_z },
        };
    }

    /// Create a plane geometry (infinite).
    pub fn plane(normal: [3]f32, offset: f32) Geom {
        return .{
            .geom_type = .plane,
            .size = .{ offset, 0, 0 },
            .local_quat = normalToQuat(normal),
        };
    }

    /// Create a cylinder geometry.
    pub fn cylinder(radius: f32, half_height: f32) Geom {
        return .{
            .geom_type = .cylinder,
            .size = .{ radius, half_height, 0 },
        };
    }

    /// Create a heightfield geometry.
    pub fn heightfield(hf: *const Heightfield) Geom {
        const bounds = hf.getBounds();
        return .{
            .geom_type = .heightfield,
            .size = .{
                (bounds.max[0] - bounds.min[0]) * 0.5,
                (bounds.max[1] - bounds.min[1]) * 0.5,
                (bounds.max[2] - bounds.min[2]) * 0.5,
            },
            .heightfield_ptr = hf,
        };
    }

    /// Get the radius (for sphere, capsule, cylinder).
    pub fn getRadius(self: *const Geom) f32 {
        return self.size[0];
    }

    /// Get half length (for capsule, cylinder).
    pub fn getHalfLength(self: *const Geom) f32 {
        return self.size[1];
    }

    /// Get half extents (for box).
    pub fn getHalfExtents(self: *const Geom) [3]f32 {
        return self.size;
    }

    /// Get plane offset.
    pub fn getPlaneOffset(self: *const Geom) f32 {
        return self.size[0];
    }

    /// Compute AABB in local space.
    pub fn computeLocalAABB(self: *const Geom) body.AABB {
        return switch (self.geom_type) {
            .sphere => body.AABB.fromCenterExtents(
                self.local_pos,
                .{ self.size[0], self.size[0], self.size[0] },
            ),
            .capsule => blk: {
                const r = self.size[0];
                const h = self.size[1];
                // Axis-aligned capsule (assumes Z-aligned in local space)
                break :blk body.AABB.fromCenterExtents(
                    self.local_pos,
                    .{ r, r, h + r },
                );
            },
            .box => body.AABB.fromCenterExtents(self.local_pos, self.size),
            .plane => body.AABB{
                .min = .{ -1e10, -1e10, -1e10 },
                .max = .{ 1e10, 1e10, 1e10 },
            },
            .cylinder => blk: {
                const r = self.size[0];
                const h = self.size[1];
                break :blk body.AABB.fromCenterExtents(
                    self.local_pos,
                    .{ r, r, h },
                );
            },
            .mesh => if (self.mesh_ptr) |m|
                body.AABB{
                    .min = .{
                        self.local_pos[0] + m.aabb.min[0],
                        self.local_pos[1] + m.aabb.min[1],
                        self.local_pos[2] + m.aabb.min[2],
                    },
                    .max = .{
                        self.local_pos[0] + m.aabb.max[0],
                        self.local_pos[1] + m.aabb.max[1],
                        self.local_pos[2] + m.aabb.max[2],
                    },
                }
            else
                body.AABB{
                    .min = .{ -1, -1, -1 },
                    .max = .{ 1, 1, 1 },
                },
            .heightfield => if (self.heightfield_ptr) |hf| blk: {
                const bounds = hf.getBounds();
                break :blk body.AABB{
                    .min = .{
                        self.local_pos[0] + bounds.min[0],
                        self.local_pos[1] + bounds.min[1],
                        self.local_pos[2] + bounds.min[2],
                    },
                    .max = .{
                        self.local_pos[0] + bounds.max[0],
                        self.local_pos[1] + bounds.max[1],
                        self.local_pos[2] + bounds.max[2],
                    },
                };
            } else body.AABB{
                .min = .{ -10, -10, -1 },
                .max = .{ 10, 10, 1 },
            },
        };
    }

    /// Compute world-space AABB given body transform.
    pub fn computeWorldAABB(self: *const Geom, transform: *const body.Transform) body.AABB {
        const local_aabb = self.computeLocalAABB();

        // Transform AABB corners and expand
        const corners = [8][3]f32{
            .{ local_aabb.min[0], local_aabb.min[1], local_aabb.min[2] },
            .{ local_aabb.max[0], local_aabb.min[1], local_aabb.min[2] },
            .{ local_aabb.min[0], local_aabb.max[1], local_aabb.min[2] },
            .{ local_aabb.max[0], local_aabb.max[1], local_aabb.min[2] },
            .{ local_aabb.min[0], local_aabb.min[1], local_aabb.max[2] },
            .{ local_aabb.max[0], local_aabb.min[1], local_aabb.max[2] },
            .{ local_aabb.min[0], local_aabb.max[1], local_aabb.max[2] },
            .{ local_aabb.max[0], local_aabb.max[1], local_aabb.max[2] },
        };

        var world_aabb = body.AABB{
            .min = .{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) },
            .max = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) },
        };

        for (corners) |corner| {
            const world_pt = transform.transformPoint(corner);
            world_aabb.min[0] = @min(world_aabb.min[0], world_pt[0]);
            world_aabb.min[1] = @min(world_aabb.min[1], world_pt[1]);
            world_aabb.min[2] = @min(world_aabb.min[2], world_pt[2]);
            world_aabb.max[0] = @max(world_aabb.max[0], world_pt[0]);
            world_aabb.max[1] = @max(world_aabb.max[1], world_pt[1]);
            world_aabb.max[2] = @max(world_aabb.max[2], world_pt[2]);
        }

        return world_aabb;
    }

    /// Compute volume.
    pub fn volume(self: *const Geom) f32 {
        return switch (self.geom_type) {
            .sphere => 4.0 / 3.0 * std.math.pi * std.math.pow(f32, self.size[0], 3),
            .capsule => blk: {
                const r = self.size[0];
                const h = self.size[1] * 2.0;
                const sphere_vol = 4.0 / 3.0 * std.math.pi * std.math.pow(f32, r, 3);
                const cyl_vol = std.math.pi * r * r * h;
                break :blk sphere_vol + cyl_vol;
            },
            .box => 8.0 * self.size[0] * self.size[1] * self.size[2],
            .cylinder => blk: {
                const r = self.size[0];
                const h = self.size[1] * 2.0;
                break :blk std.math.pi * r * r * h;
            },
            .plane => 0.0,
            .mesh => if (self.mesh_ptr) |m| m.volume() else 0.0,
            .heightfield => 0.0, // Heightfield is static terrain, no volume
        };
    }

    /// Compute inertia tensor diagonal (assuming uniform density).
    pub fn computeInertia(self: *const Geom, mass: f32) [3]f32 {
        return switch (self.geom_type) {
            .sphere => blk: {
                const i = 0.4 * mass * self.size[0] * self.size[0];
                break :blk .{ i, i, i };
            },
            .capsule => blk: {
                const r = self.size[0];
                const h = self.size[1] * 2.0;
                const r2 = r * r;
                const h2 = h * h;

                // Approximate using cylinder + hemispheres
                const i_x = mass * (3.0 * r2 + h2) / 12.0;
                const i_z = 0.5 * mass * r2;
                break :blk .{ i_x, i_x, i_z };
            },
            .box => blk: {
                const m = mass / 12.0;
                const x2 = 4.0 * self.size[0] * self.size[0];
                const y2 = 4.0 * self.size[1] * self.size[1];
                const z2 = 4.0 * self.size[2] * self.size[2];
                break :blk .{
                    m * (y2 + z2),
                    m * (x2 + z2),
                    m * (x2 + y2),
                };
            },
            .cylinder => blk: {
                const r = self.size[0];
                const h = self.size[1] * 2.0;
                const r2 = r * r;
                const h2 = h * h;

                const i_x = mass * (3.0 * r2 + h2) / 12.0;
                const i_z = 0.5 * mass * r2;
                break :blk .{ i_x, i_x, i_z };
            },
            .plane => .{ 0, 0, 0 },
            .mesh => if (self.mesh_ptr) |m| m.computeInertia(mass) else .{ 0, 0, 0 },
            .heightfield => .{ 0, 0, 0 }, // Heightfield is static terrain
        };
    }

    /// Get world-space center position.
    pub fn getWorldCenter(self: *const Geom, body_transform: *const body.Transform) [3]f32 {
        return body_transform.transformPoint(self.local_pos);
    }

    /// Get world-space orientation.
    pub fn getWorldQuat(self: *const Geom, body_transform: *const body.Transform) [4]f32 {
        return quatMultiply(body_transform.quaternion, self.local_quat);
    }
};

/// GPU-friendly geometry data.
pub const GeomGPU = extern struct {
    /// Type and body ID packed.
    type_body: [4]u32 align(16),
    /// Local position xyz + radius/size[0].
    pos_size0: [4]f32 align(16),
    /// Local quat xyzw.
    quat: [4]f32 align(16),
    /// Size[1], size[2], friction, restitution.
    params: [4]f32 align(16),

    pub fn fromGeom(g: *const Geom, idx: u32) GeomGPU {
        return .{
            .type_body = .{ @intFromEnum(g.geom_type), g.body_id, idx, g.group },
            .pos_size0 = .{ g.local_pos[0], g.local_pos[1], g.local_pos[2], g.size[0] },
            .quat = g.local_quat,
            .params = .{ g.size[1], g.size[2], g.friction, g.restitution },
        };
    }
};

// Helper functions

fn normalToQuat(normal: [3]f32) [4]f32 {
    // Convert normal to quaternion that rotates Z-axis to normal
    const z: [3]f32 = .{ 0, 0, 1 };
    const n = normalize(normal);

    const d = dot(z, n);
    if (d > 0.99999) {
        return .{ 0, 0, 0, 1 };
    }
    if (d < -0.99999) {
        return .{ 1, 0, 0, 0 };
    }

    const axis = normalize(cross(z, n));
    const angle = std.math.acos(d);
    const s = @sin(angle * 0.5);
    const c = @cos(angle * 0.5);

    return .{ axis[0] * s, axis[1] * s, axis[2] * s, c };
}

fn quatMultiply(a: [4]f32, b: [4]f32) [4]f32 {
    return .{
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    };
}

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

fn normalize(v: [3]f32) [3]f32 {
    const len = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-8) return .{ 0, 0, 1 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}
