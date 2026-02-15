//! Tests for the Zeno swarm platform.

const std = @import("std");
const zeno = @import("zeno");

const UniformGrid = zeno.swarm.grid.UniformGrid;
const AdjacencyGraph = zeno.swarm.graph.AdjacencyGraph;
const MessageBus = zeno.swarm.message_bus.MessageBus;
const Swarm = zeno.swarm.swarm_mod.Swarm;
const SwarmConfig = zeno.swarm.types.SwarmConfig;
const MessageSlot = zeno.swarm.types.MessageSlot;
const dispatcher = zeno.swarm.dispatcher;

// ============================================================================
// Grid Tests
// ============================================================================

test "grid: rebuild and query basic" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 10, 10, 10, 0 },
    };

    grid.rebuild(&positions, 0, 4);
    try std.testing.expectEqual(@as(u32, 4), grid.num_agents);

    var neighbors: [10]u32 = undefined;
    const count = grid.queryNeighbors(0, &positions, 0, 4.0, &neighbors);

    // Agents 1 and 2 are within range sqrt(1) and sqrt(1) < 2
    try std.testing.expect(count >= 2);
}

test "grid: empty grid" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{};
    grid.rebuild(&positions, 0, 0);
    try std.testing.expectEqual(@as(u32, 0), grid.num_agents);
}

test "grid: single agent has no neighbors" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 5, 5, 5, 0 },
    };

    grid.rebuild(&positions, 0, 1);

    var neighbors: [10]u32 = undefined;
    const count = grid.queryNeighbors(0, &positions, 0, 4.0, &neighbors);
    try std.testing.expectEqual(@as(u32, 0), count);
}

// ============================================================================
// Graph Tests
// ============================================================================

test "graph: build from grid" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 100, 100, 100, 0 },
    };

    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();

    graph.buildFromGrid(&grid, &positions, 0, 3, 4.0);

    // Agent 0 should have agent 1 as neighbor
    const n0 = graph.getNeighbors(0);
    try std.testing.expect(n0.len >= 1);
    try std.testing.expectEqual(@as(u32, 1), n0[0]);

    // Agent 2 should have no neighbors
    const n2 = graph.getNeighbors(2);
    try std.testing.expectEqual(@as(usize, 0), n2.len);
}

test "graph: neighbor count" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 0.5, 0, 0, 0 },
        .{ 1.0, 0, 0, 0 },
    };

    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();

    graph.buildFromGrid(&grid, &positions, 0, 3, 4.0);

    // All agents are close, each should have 2 neighbors
    try std.testing.expectEqual(@as(u32, 2), graph.neighborCount(0));
    try std.testing.expectEqual(@as(u32, 2), graph.neighborCount(1));
    try std.testing.expectEqual(@as(u32, 2), graph.neighborCount(2));
}

test "graph: neighbors sorted by id" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 2.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 0.5, 0, 0, 0 },
        .{ 1.0, 0, 0, 0 },
    };

    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();

    graph.buildFromGrid(&grid, &positions, 0, 3, 4.0);

    // Neighbors of agent 0 should be [1, 2] (sorted)
    const n0 = graph.getNeighbors(0);
    if (n0.len >= 2) {
        try std.testing.expect(n0[0] < n0[1]);
    }
}

// ============================================================================
// Message Bus Tests
// ============================================================================

test "message bus: send and budget" {
    const allocator = std.testing.allocator;

    var bus = try MessageBus.init(allocator, 4, 2, 48);
    defer bus.deinit();
    bus.clearStep();

    // Send 2 messages (up to budget)
    try std.testing.expect(bus.send(0, 1, 1, "hi"));
    try std.testing.expect(bus.send(0, 2, 1, "hey"));

    // Third message exceeds budget
    try std.testing.expect(!bus.send(0, 3, 1, "overflow"));

    try std.testing.expectEqual(@as(u32, 2), bus.outbox_counts[0]);
}

test "message bus: deliver point-to-point" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 3, 25.0);

    var bus = try MessageBus.init(allocator, 3, 4, 48);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 sends to agent 1
    try std.testing.expect(bus.send(0, 1, 42, "test"));

    // Deliver
    bus.deliver(&graph);

    // Agent 1 should have 1 message
    const inbox = bus.getInbox(1);
    try std.testing.expectEqual(@as(usize, 1), inbox.len);
    try std.testing.expectEqual(@as(u32, 42), inbox[0].message_type);
}

test "message bus: broadcast" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 3, 25.0);

    var bus = try MessageBus.init(allocator, 3, 4, 48);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 broadcasts
    try std.testing.expect(bus.send(0, MessageSlot.BROADCAST, 99, "bcast"));

    bus.deliver(&graph);

    // Agents 1 and 2 should each have 1 message
    try std.testing.expectEqual(@as(usize, 1), bus.getInbox(1).len);
    try std.testing.expectEqual(@as(usize, 1), bus.getInbox(2).len);
    // Agent 0 should NOT receive its own broadcast
    try std.testing.expectEqual(@as(usize, 0), bus.getInbox(0).len);
}

// ============================================================================
// Swarm Lifecycle Tests
// ============================================================================

test "swarm: init and deinit" {
    const allocator = std.testing.allocator;

    const config = SwarmConfig{
        .num_agents = 10,
        .communication_range = 5.0,
        .max_neighbors = 8,
        .max_message_bytes = 48,
        .max_messages_per_step = 4,
        .grid_cell_size = 5.0,
    };

    var s = try Swarm.init(allocator, config);
    defer s.deinit();

    try std.testing.expectEqual(@as(u32, 10), s.config.num_agents);
    try std.testing.expectEqual(@as(u64, 0), s.step_count);
}

test "swarm: step with positions" {
    const allocator = std.testing.allocator;

    const config = SwarmConfig{
        .num_agents = 3,
        .communication_range = 5.0,
        .max_neighbors = 8,
        .max_message_bytes = 48,
        .max_messages_per_step = 4,
        .grid_cell_size = 5.0,
    };

    var s = try Swarm.init(allocator, config);
    defer s.deinit();

    // Mock positions and velocities
    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 }, // ground
        .{ 0, 0, 0.1, 0 }, // agent 0
        .{ 1, 0, 0.1, 0 }, // agent 1
        .{ 0, 1, 0.1, 0 }, // agent 2
    };
    var velocities = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };

    s.setBodyOffset(1); // skip ground

    s.step(&positions, &velocities, null, 0);

    try std.testing.expectEqual(@as(u64, 1), s.step_count);
    try std.testing.expect(s.metrics.total_edges > 0);
}

// ============================================================================
// Dispatcher Tests
// ============================================================================

test "dispatcher: compute metrics" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 100, 100, 100, 0 },
    };
    grid.rebuild(&positions, 0, 3);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 3, 4.0);

    var bus = try MessageBus.init(allocator, 3, 4, 48);
    defer bus.deinit();
    bus.clearStep();

    const metrics = dispatcher.computeMetrics(&graph, &bus, 3);
    try std.testing.expect(metrics.connectivity_ratio > 0);
    try std.testing.expect(metrics.avg_neighbors > 0);
}
