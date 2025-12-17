//! Spatial hashing broad phase for collision detection.
//! Efficiently finds potentially colliding pairs.

const std = @import("std");
const body = @import("../physics/body.zig");
const primitives = @import("primitives.zig");
const constants = @import("../physics/constants.zig");

/// Collision pair from broad phase.
pub const CollisionPair = struct {
    geom_a: u32,
    geom_b: u32,
    env_id: u32,
};

/// Spatial hash cell.
pub const Cell = struct {
    /// Start index in sorted list.
    start: u32 = 0,
    /// End index in sorted list.
    end: u32 = 0,
};

/// Spatial hash grid for broad phase collision detection.
pub const SpatialHash = struct {
    /// Cell size (should be >= 2x max body radius).
    cell_size: f32,
    /// Grid dimensions.
    grid_size: u32,
    /// Cell data.
    cells: []Cell,
    /// Sorted geometry indices.
    sorted_geoms: []u32,
    /// Cell IDs for each geometry.
    cell_ids: []u32,
    /// Number of geometries.
    num_geoms: u32,
    /// Allocator.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_geoms: u32, cell_size: f32, grid_size: u32) !SpatialHash {
        const num_cells = grid_size * grid_size * grid_size;

        return SpatialHash{
            .cell_size = cell_size,
            .grid_size = grid_size,
            .cells = try allocator.alloc(Cell, num_cells),
            .sorted_geoms = try allocator.alloc(u32, max_geoms),
            .cell_ids = try allocator.alloc(u32, max_geoms),
            .num_geoms = 0,
            .allocator = allocator,
        };
    }

    /// Compute cell ID from position.
    pub fn cellId(self: *const SpatialHash, pos: [3]f32) u32 {
        const ix = @as(u32, @intFromFloat(@max(0, @floor(pos[0] / self.cell_size)))) % self.grid_size;
        const iy = @as(u32, @intFromFloat(@max(0, @floor(pos[1] / self.cell_size)))) % self.grid_size;
        const iz = @as(u32, @intFromFloat(@max(0, @floor(pos[2] / self.cell_size)))) % self.grid_size;

        return ix + iy * self.grid_size + iz * self.grid_size * self.grid_size;
    }

    /// Hash position to cell ID with spatial coherence.
    pub fn hashPosition(self: *const SpatialHash, pos: [3]f32) u32 {
        // Morton code for better spatial coherence
        const ix = @as(u32, @intFromFloat(@max(0, @floor(pos[0] / self.cell_size)))) % self.grid_size;
        const iy = @as(u32, @intFromFloat(@max(0, @floor(pos[1] / self.cell_size)))) % self.grid_size;
        const iz = @as(u32, @intFromFloat(@max(0, @floor(pos[2] / self.cell_size)))) % self.grid_size;

        return mortonEncode(ix, iy, iz);
    }

    /// Update the spatial hash with new positions.
    pub fn update(
        self: *SpatialHash,
        positions: []const [3]f32,
        num_geoms: u32,
    ) void {
        self.num_geoms = num_geoms;

        // Clear cells
        for (self.cells) |*cell| {
            cell.start = 0;
            cell.end = 0;
        }

        // Compute cell IDs for each geometry
        for (0..num_geoms) |i| {
            self.cell_ids[i] = self.cellId(positions[i]);
        }

        // Count geometries per cell
        for (0..num_geoms) |i| {
            const cid = self.cell_ids[i];
            self.cells[cid].end += 1;
        }

        // Compute cell start indices (prefix sum)
        var sum: u32 = 0;
        for (self.cells) |*cell| {
            cell.start = sum;
            sum += cell.end;
            cell.end = cell.start;
        }

        // Sort geometries into cells
        for (0..num_geoms) |i| {
            const cid = self.cell_ids[i];
            self.sorted_geoms[self.cells[cid].end] = @intCast(i);
            self.cells[cid].end += 1;
        }
    }

    /// Query potential collisions for an AABB.
    pub fn query(
        self: *const SpatialHash,
        aabb: *const body.AABB,
        result: *std.ArrayList(u32),
    ) void {
        const min_cell: [3]i32 = .{
            @intFromFloat(@floor(aabb.min[0] / self.cell_size)),
            @intFromFloat(@floor(aabb.min[1] / self.cell_size)),
            @intFromFloat(@floor(aabb.min[2] / self.cell_size)),
        };
        const max_cell: [3]i32 = .{
            @intFromFloat(@ceil(aabb.max[0] / self.cell_size)),
            @intFromFloat(@ceil(aabb.max[1] / self.cell_size)),
            @intFromFloat(@ceil(aabb.max[2] / self.cell_size)),
        };

        var iz = min_cell[2];
        while (iz <= max_cell[2]) : (iz += 1) {
            var iy = min_cell[1];
            while (iy <= max_cell[1]) : (iy += 1) {
                var ix = min_cell[0];
                while (ix <= max_cell[0]) : (ix += 1) {
                    const ux: u32 = @intCast(@mod(ix, @as(i32, @intCast(self.grid_size))));
                    const uy: u32 = @intCast(@mod(iy, @as(i32, @intCast(self.grid_size))));
                    const uz: u32 = @intCast(@mod(iz, @as(i32, @intCast(self.grid_size))));

                    const cid = ux + uy * self.grid_size + uz * self.grid_size * self.grid_size;
                    const cell = self.cells[cid];

                    for (cell.start..cell.end) |idx| {
                        result.append(self.sorted_geoms[idx]) catch {};
                    }
                }
            }
        }
    }

    /// Find all potential collision pairs.
    pub fn findPairs(
        self: *const SpatialHash,
        aabbs: []const body.AABB,
        pairs: *std.ArrayList(CollisionPair),
        env_id: u32,
    ) void {
        // Check each geometry against cells it overlaps
        for (0..self.num_geoms) |i| {
            const aabb = &aabbs[i];

            var candidates = std.ArrayList(u32).init(self.allocator);
            defer candidates.deinit();

            self.query(aabb, &candidates);

            // Check against candidates
            for (candidates.items) |j| {
                if (j <= i) continue; // Avoid duplicates

                if (aabb.intersects(&aabbs[j])) {
                    pairs.append(.{
                        .geom_a = @intCast(i),
                        .geom_b = j,
                        .env_id = env_id,
                    }) catch {};
                }
            }
        }
    }

    pub fn deinit(self: *SpatialHash) void {
        self.allocator.free(self.cells);
        self.allocator.free(self.sorted_geoms);
        self.allocator.free(self.cell_ids);
    }
};

/// GPU-friendly broad phase data.
pub const BroadPhaseGPU = struct {
    /// Cell start indices.
    cell_starts: []u32,
    /// Cell end indices.
    cell_ends: []u32,
    /// Sorted geom indices.
    sorted_geoms: []u32,
    /// AABB min points (xyz + pad).
    aabb_mins: [][4]f32,
    /// AABB max points (xyz + pad).
    aabb_maxs: [][4]f32,
    /// Output pairs.
    pairs: []CollisionPairGPU,
    /// Number of pairs found.
    pair_count: []u32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_envs: u32, max_geoms: u32, grid_cells: u32, max_pairs: u32) !BroadPhaseGPU {
        const total_geoms = num_envs * max_geoms;

        return .{
            .cell_starts = try allocator.alloc(u32, num_envs * grid_cells),
            .cell_ends = try allocator.alloc(u32, num_envs * grid_cells),
            .sorted_geoms = try allocator.alloc(u32, total_geoms),
            .aabb_mins = try allocator.alloc([4]f32, total_geoms),
            .aabb_maxs = try allocator.alloc([4]f32, total_geoms),
            .pairs = try allocator.alloc(CollisionPairGPU, num_envs * max_pairs),
            .pair_count = try allocator.alloc(u32, num_envs),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BroadPhaseGPU) void {
        self.allocator.free(self.cell_starts);
        self.allocator.free(self.cell_ends);
        self.allocator.free(self.sorted_geoms);
        self.allocator.free(self.aabb_mins);
        self.allocator.free(self.aabb_maxs);
        self.allocator.free(self.pairs);
        self.allocator.free(self.pair_count);
    }
};

/// GPU collision pair.
pub const CollisionPairGPU = extern struct {
    geom_a: u32,
    geom_b: u32,
    env_id: u32,
    _pad: u32 = 0,
};

/// Morton encoding for 3D spatial hashing.
pub fn mortonEncode(x: u32, y: u32, z: u32) u32 {
    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

fn expandBits(v: u32) u32 {
    var x = v & 0x3FF; // 10 bits
    x = (x | (x << 16)) & 0x30000FF;
    x = (x | (x << 8)) & 0x300F00F;
    x = (x | (x << 4)) & 0x30C30C3;
    x = (x | (x << 2)) & 0x9249249;
    return x;
}

/// Simple brute-force broad phase for small scenes.
pub const BruteForceBroadPhase = struct {
    /// Find all overlapping AABB pairs.
    pub fn findPairs(
        allocator: std.mem.Allocator,
        aabbs: []const body.AABB,
        groups: []const u32,
        masks: []const u32,
    ) ![]CollisionPair {
        var pairs = std.ArrayList(CollisionPair).init(allocator);

        for (0..aabbs.len) |i| {
            for ((i + 1)..aabbs.len) |j| {
                // Check collision masks
                if ((groups[i] & masks[j]) == 0 and (groups[j] & masks[i]) == 0) {
                    continue;
                }

                if (aabbs[i].intersects(&aabbs[j])) {
                    try pairs.append(.{
                        .geom_a = @intCast(i),
                        .geom_b = @intCast(j),
                        .env_id = 0,
                    });
                }
            }
        }

        return pairs.toOwnedSlice();
    }
};

test "morton encoding" {
    const m = mortonEncode(1, 2, 3);
    try std.testing.expect(m > 0);
}

test "spatial hash" {
    const allocator = std.testing.allocator;

    var hash = try SpatialHash.init(allocator, 100, 1.0, 16);
    defer hash.deinit();

    var positions: [10][3]f32 = undefined;
    for (0..10) |i| {
        positions[i] = .{ @floatFromInt(i), 0, 0 };
    }

    hash.update(&positions, 10);

    try std.testing.expectEqual(@as(u32, 10), hash.num_geoms);
}
