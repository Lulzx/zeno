//! CSR (Compressed Sparse Row) adjacency graph for swarm agents.
//! Built from the uniform grid each step. Neighbor lists are sorted
//! by agent_id for deterministic iteration.

const std = @import("std");
const UniformGrid = @import("grid.zig").UniformGrid;

/// CSR adjacency graph. Rebuilt each step from the spatial grid.
pub const AdjacencyGraph = struct {
    /// row_ptr[i] = start index in neighbor_ids for agent i.
    /// row_ptr[num_agents] = total edges.
    row_ptr: []u32,
    /// Flat array of neighbor agent IDs.
    neighbor_ids: []u32,
    /// Distances to neighbors (parallel to neighbor_ids).
    neighbor_dists: []f32,
    /// Number of agents.
    num_agents: u32,
    /// Total edges currently stored.
    total_edges: u32,
    /// Max neighbors per agent (capacity constraint).
    max_neighbors: u32,
    /// Scratch buffer for queryNeighbors.
    scratch: []u32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_agents: u32, max_neighbors: u32) !AdjacencyGraph {
        const max_edges = max_agents * max_neighbors;
        return .{
            .row_ptr = try allocator.alloc(u32, max_agents + 1),
            .neighbor_ids = try allocator.alloc(u32, max_edges),
            .neighbor_dists = try allocator.alloc(f32, max_edges),
            .num_agents = 0,
            .total_edges = 0,
            .max_neighbors = max_neighbors,
            .scratch = try allocator.alloc(u32, max_neighbors),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdjacencyGraph) void {
        self.allocator.free(self.row_ptr);
        self.allocator.free(self.neighbor_ids);
        self.allocator.free(self.neighbor_dists);
        self.allocator.free(self.scratch);
    }

    /// Build adjacency from grid. O(N * max_neighbors * log(max_neighbors)).
    pub fn buildFromGrid(
        self: *AdjacencyGraph,
        grid: *const UniformGrid,
        positions: [][4]f32,
        body_offset: u32,
        num_agents: u32,
        range_sq: f32,
    ) void {
        self.num_agents = num_agents;
        self.total_edges = 0;

        for (0..num_agents) |agent_id| {
            self.row_ptr[agent_id] = self.total_edges;

            const count = grid.queryNeighbors(
                @intCast(agent_id),
                positions,
                body_offset,
                range_sq,
                self.scratch,
            );

            const actual = @min(count, self.max_neighbors);

            // Sort neighbors by ID for determinism
            if (actual > 0) {
                sortU32Slice(self.scratch[0..actual]);
            }

            // Copy sorted neighbors and compute distances
            const pos = positions[body_offset + agent_id];
            for (0..actual) |n| {
                const nid = self.scratch[n];
                const npos = positions[body_offset + nid];
                const dx = npos[0] - pos[0];
                const dy = npos[1] - pos[1];
                const dz = npos[2] - pos[2];

                self.neighbor_ids[self.total_edges] = nid;
                self.neighbor_dists[self.total_edges] = @sqrt(dx * dx + dy * dy + dz * dz);
                self.total_edges += 1;
            }
        }

        self.row_ptr[num_agents] = self.total_edges;
    }

    /// Get the neighbor list for a given agent.
    pub fn getNeighbors(self: *const AdjacencyGraph, agent_id: u32) []const u32 {
        const start = self.row_ptr[agent_id];
        const end = self.row_ptr[agent_id + 1];
        return self.neighbor_ids[start..end];
    }

    /// Get neighbor distances for a given agent.
    pub fn getNeighborDists(self: *const AdjacencyGraph, agent_id: u32) []const f32 {
        const start = self.row_ptr[agent_id];
        const end = self.row_ptr[agent_id + 1];
        return self.neighbor_dists[start..end];
    }

    /// Get neighbor count for a given agent.
    pub fn neighborCount(self: *const AdjacencyGraph, agent_id: u32) u32 {
        return self.row_ptr[agent_id + 1] - self.row_ptr[agent_id];
    }
};

/// Simple insertion sort for small u32 slices.
fn sortU32Slice(slice: []u32) void {
    if (slice.len <= 1) return;
    for (1..slice.len) |i| {
        const key = slice[i];
        var j: usize = i;
        while (j > 0 and slice[j - 1] > key) {
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = key;
    }
}

test "adjacency graph from grid" {
    const allocator = std.testing.allocator;
    const UniformGridType = @import("grid.zig").UniformGrid;

    var grid = try UniformGridType.init(allocator, 100, 2.0, 16);
    defer grid.deinit();

    var positions: [3][4]f32 = .{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 100, 100, 100, 0 },
    };

    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 100, 32);
    defer graph.deinit();

    graph.buildFromGrid(&grid, &positions, 0, 3, 4.0);

    // Agent 0 and 1 should be neighbors
    const n0 = graph.getNeighbors(0);
    try std.testing.expect(n0.len >= 1);
    try std.testing.expectEqual(@as(u32, 1), n0[0]);

    // Agent 2 should have no neighbors
    const n2 = graph.getNeighbors(2);
    try std.testing.expectEqual(@as(usize, 0), n2.len);
}
