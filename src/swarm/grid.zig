//! Uniform spatial grid for O(N) neighbor search.
//! Used by the swarm platform to find nearby agents efficiently.

const std = @import("std");

/// Uniform spatial grid for fast neighbor iteration.
/// Rebuild is O(N) via counting sort, queries are O(1) per cell.
pub const UniformGrid = struct {
    cell_size: f32,
    inv_cell_size: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    total_cells: u32,

    /// Number of geoms per cell (used during rebuild, then reset).
    cell_counts: []u32,
    /// Prefix sum offsets for each cell.
    cell_offsets: []u32,
    /// Agent indices sorted by cell.
    sorted_agents: []u32,
    /// Cell ID for each agent.
    cell_ids: []u32,

    num_agents: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_agents: u32, cell_size: f32, grid_dim: u32) !UniformGrid {
        const total_cells = grid_dim * grid_dim * grid_dim;
        return .{
            .cell_size = cell_size,
            .inv_cell_size = 1.0 / cell_size,
            .dim_x = grid_dim,
            .dim_y = grid_dim,
            .dim_z = grid_dim,
            .total_cells = total_cells,
            .cell_counts = try allocator.alloc(u32, total_cells),
            .cell_offsets = try allocator.alloc(u32, total_cells),
            .sorted_agents = try allocator.alloc(u32, max_agents),
            .cell_ids = try allocator.alloc(u32, max_agents),
            .num_agents = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *UniformGrid) void {
        self.allocator.free(self.cell_counts);
        self.allocator.free(self.cell_offsets);
        self.allocator.free(self.sorted_agents);
        self.allocator.free(self.cell_ids);
    }

    /// Compute cell ID from a 3D position.
    fn cellIdFromPos(self: *const UniformGrid, x: f32, y: f32, z: f32) u32 {
        const ix = @as(u32, @intFromFloat(@max(0, @floor(x * self.inv_cell_size)))) % self.dim_x;
        const iy = @as(u32, @intFromFloat(@max(0, @floor(y * self.inv_cell_size)))) % self.dim_y;
        const iz = @as(u32, @intFromFloat(@max(0, @floor(z * self.inv_cell_size)))) % self.dim_z;
        return ix + iy * self.dim_x + iz * self.dim_x * self.dim_y;
    }

    /// Rebuild the grid from positions. O(N) counting sort.
    /// `positions` is the world positions buffer (float4 per body).
    /// `body_offset` is the index of the first agent body in the positions array.
    /// `num_agents` is how many agent bodies to index.
    pub fn rebuild(
        self: *UniformGrid,
        positions: [][4]f32,
        body_offset: u32,
        num_agents: u32,
    ) void {
        self.num_agents = num_agents;

        // Clear cell counts
        @memset(self.cell_counts, 0);

        // Assign cell IDs
        for (0..num_agents) |i| {
            const pos = positions[body_offset + i];
            const cid = self.cellIdFromPos(pos[0], pos[1], pos[2]);
            self.cell_ids[i] = cid;
            self.cell_counts[cid] += 1;
        }

        // Prefix sum → offsets
        var sum: u32 = 0;
        for (0..self.total_cells) |c| {
            self.cell_offsets[c] = sum;
            sum += self.cell_counts[c];
            self.cell_counts[c] = 0; // Reset for scatter
        }

        // Scatter agents into sorted order
        for (0..num_agents) |i| {
            const cid = self.cell_ids[i];
            const slot = self.cell_offsets[cid] + self.cell_counts[cid];
            self.sorted_agents[slot] = @intCast(i);
            self.cell_counts[cid] += 1;
        }
    }

    /// Query neighbors of an agent within range_sq (squared distance).
    /// Returns the number of neighbors written to out_buffer.
    pub fn queryNeighbors(
        self: *const UniformGrid,
        agent_id: u32,
        positions: [][4]f32,
        body_offset: u32,
        range_sq: f32,
        out_buffer: []u32,
    ) u32 {
        const pos = positions[body_offset + agent_id];
        const px = pos[0];
        const py = pos[1];
        const pz = pos[2];

        // Compute cell range to search (own cell + neighbors)
        const cx = @as(i32, @intFromFloat(@max(0, @floor(px * self.inv_cell_size))));
        const cy = @as(i32, @intFromFloat(@max(0, @floor(py * self.inv_cell_size))));
        const cz = @as(i32, @intFromFloat(@max(0, @floor(pz * self.inv_cell_size))));

        var count: u32 = 0;
        const max_out = @as(u32, @intCast(out_buffer.len));

        var dz: i32 = -1;
        while (dz <= 1) : (dz += 1) {
            var dy: i32 = -1;
            while (dy <= 1) : (dy += 1) {
                var dx: i32 = -1;
                while (dx <= 1) : (dx += 1) {
                    const nx = cx + dx;
                    const ny = cy + dy;
                    const nz = cz + dz;

                    if (nx < 0 or ny < 0 or nz < 0) continue;
                    const ux = @as(u32, @intCast(nx)) % self.dim_x;
                    const uy = @as(u32, @intCast(ny)) % self.dim_y;
                    const uz = @as(u32, @intCast(nz)) % self.dim_z;

                    const cid = ux + uy * self.dim_x + uz * self.dim_x * self.dim_y;
                    const start = self.cell_offsets[cid];
                    const end = start + self.cell_counts[cid];

                    for (start..end) |s| {
                        const other = self.sorted_agents[s];
                        if (other == agent_id) continue;

                        const opos = positions[body_offset + other];
                        const ddx = opos[0] - px;
                        const ddy = opos[1] - py;
                        const ddz = opos[2] - pz;
                        const dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                        if (dist_sq <= range_sq) {
                            if (count < max_out) {
                                out_buffer[count] = other;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        return count;
    }
};

test "uniform grid rebuild and query" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 100, 2.0, 16);
    defer grid.deinit();

    // Create some positions
    var positions: [4][4]f32 = .{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 }, // within range of 0
        .{ 10, 10, 10, 0 }, // far away
        .{ 0.5, 0.5, 0, 0 }, // within range of 0
    };

    grid.rebuild(&positions, 0, 4);
    try std.testing.expectEqual(@as(u32, 4), grid.num_agents);

    var neighbors: [10]u32 = undefined;
    const count = grid.queryNeighbors(0, &positions, 0, 4.0, &neighbors);
    try std.testing.expect(count >= 2); // agents 1 and 3 should be neighbors
}
