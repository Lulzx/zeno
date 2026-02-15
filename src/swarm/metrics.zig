//! Swarm metrics computation module.
//! Provides connectivity, fragmentation, and near-miss analysis.

const std = @import("std");
const types = @import("types.zig");
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageBus = @import("message_bus.zig").MessageBus;
const SwarmMetrics = types.SwarmMetrics;

/// Compute swarm metrics from current state.
pub fn computeMetrics(
    graph: *const AdjacencyGraph,
    message_bus: *const MessageBus,
    num_agents: u32,
) SwarmMetrics {
    if (num_agents == 0) return .{};

    const max_possible_edges = num_agents * (num_agents - 1);
    const connectivity = if (max_possible_edges > 0)
        @as(f32, @floatFromInt(graph.total_edges)) / @as(f32, @floatFromInt(max_possible_edges))
    else
        0;

    const avg_neighbors = if (num_agents > 0)
        @as(f32, @floatFromInt(graph.total_edges)) / @as(f32, @floatFromInt(num_agents))
    else
        0;

    return .{
        .connectivity_ratio = connectivity,
        .fragmentation_score = @floatFromInt(computeFragmentation(graph, num_agents)),
        .collision_count = 0,
        .message_count = message_bus.total_messages_delivered,
        .bytes_sent = message_bus.total_bytes_sent,
        .total_edges = graph.total_edges,
        .avg_neighbors = avg_neighbors,
        .messages_dropped = message_bus.total_messages_dropped,
    };
}

/// Compute number of connected components via Union-Find.
/// Returns 1 if fully connected, N if all disconnected.
pub fn computeFragmentation(graph: *const AdjacencyGraph, num_agents: u32) u32 {
    if (num_agents == 0) return 0;
    if (num_agents == 1) return 1;

    // Stack-allocated Union-Find for small swarms, heap for large
    const MAX_STACK: u32 = 1024;
    var stack_parent: [MAX_STACK]u32 = undefined;
    var stack_rank: [MAX_STACK]u8 = undefined;

    var parent: []u32 = undefined;
    var rank: []u8 = undefined;

    if (num_agents <= MAX_STACK) {
        parent = stack_parent[0..num_agents];
        rank = stack_rank[0..num_agents];
    } else {
        // For large swarms, fall back to page allocator
        const page_alloc = std.heap.page_allocator;
        parent = page_alloc.alloc(u32, num_agents) catch return num_agents;
        rank = page_alloc.alloc(u8, num_agents) catch {
            page_alloc.free(parent);
            return num_agents;
        };
        defer page_alloc.free(parent);
        defer page_alloc.free(rank);
    }

    // Initialize: each node is its own parent
    for (0..num_agents) |i| {
        parent[i] = @intCast(i);
        rank[i] = 0;
    }

    // Union edges from adjacency graph
    for (0..num_agents) |i| {
        const neighbors = graph.getNeighbors(@intCast(i));
        for (neighbors) |n| {
            union_sets(parent, rank, @intCast(i), n);
        }
    }

    // Count distinct roots
    var components: u32 = 0;
    for (0..num_agents) |i| {
        if (find(parent, @intCast(i)) == @as(u32, @intCast(i))) {
            components += 1;
        }
    }
    return components;
}

fn find(parent: []u32, x: u32) u32 {
    var curr = x;
    while (parent[curr] != curr) {
        parent[curr] = parent[parent[curr]]; // path compression
        curr = parent[curr];
    }
    return curr;
}

fn union_sets(parent: []u32, rank: []u8, a: u32, b: u32) void {
    const ra = find(parent, a);
    const rb = find(parent, b);
    if (ra == rb) return;
    if (rank[ra] < rank[rb]) {
        parent[ra] = rb;
    } else if (rank[ra] > rank[rb]) {
        parent[rb] = ra;
    } else {
        parent[rb] = ra;
        rank[ra] += 1;
    }
}

/// Count near-miss pairs: agents within 2x collision_radius but not colliding.
pub fn computeNearMisses(
    positions: [][4]f32,
    body_offset: u32,
    num_agents: u32,
    collision_radius: f32,
) u32 {
    const near_radius = collision_radius * 2.0;
    const near_sq = near_radius * near_radius;
    const col_sq = collision_radius * collision_radius;

    var count: u32 = 0;
    for (0..num_agents) |i| {
        const pi = positions[body_offset + i];
        for (i + 1..num_agents) |j| {
            const pj = positions[body_offset + j];
            const dx = pj[0] - pi[0];
            const dy = pj[1] - pi[1];
            const dz = pj[2] - pi[2];
            const dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq <= near_sq and dist_sq > col_sq) {
                count += 1;
            }
        }
    }
    return count;
}

/// Compute convergence time from metrics history.
/// Returns the step index where task_success first exceeded threshold, or 0 if never.
pub fn computeConvergence(metrics_history: []const SwarmMetrics, threshold: f32) f32 {
    for (metrics_history, 0..) |m, i| {
        if (m.task_success >= threshold) {
            return @floatFromInt(i);
        }
    }
    return 0;
}
