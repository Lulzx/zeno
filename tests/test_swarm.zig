//! Tests for the Zeno swarm platform.

const std = @import("std");
const zeno = @import("zeno");

const UniformGrid = zeno.swarm.grid.UniformGrid;
const AdjacencyGraph = zeno.swarm.graph.AdjacencyGraph;
const MessageBus = zeno.swarm.message_bus.MessageBus;
const Swarm = zeno.swarm.swarm_mod.Swarm;
const SwarmConfig = zeno.swarm.types.SwarmConfig;
const SwarmMetrics = zeno.swarm.types.SwarmMetrics;
const MessageSlot = zeno.swarm.types.MessageSlot;
const AttackConfig = zeno.swarm.types.AttackConfig;
const TaskResult = zeno.swarm.types.TaskResult;
const dispatcher = zeno.swarm.dispatcher;
const metrics_mod = zeno.swarm.metrics;
const tasks_mod = zeno.swarm.tasks;
const attacks_mod = zeno.swarm.attacks;
const replay_mod = zeno.swarm.replay;

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

    const config = SwarmConfig{
        .num_agents = 4,
        .max_messages_per_step = 2,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
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

    const config = SwarmConfig{
        .num_agents = 3,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 sends to agent 1
    try std.testing.expect(bus.send(0, 1, 42, "test"));

    // Deliver
    bus.deliver(&graph, 0);

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

    const config = SwarmConfig{
        .num_agents = 3,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 broadcasts
    try std.testing.expect(bus.send(0, MessageSlot.BROADCAST, 99, "bcast"));

    bus.deliver(&graph, 0);

    // Agents 1 and 2 should each have 1 message
    try std.testing.expectEqual(@as(usize, 1), bus.getInbox(1).len);
    try std.testing.expectEqual(@as(usize, 1), bus.getInbox(2).len);
    // Agent 0 should NOT receive its own broadcast
    try std.testing.expectEqual(@as(usize, 0), bus.getInbox(0).len);
}

// ============================================================================
// Message Bus Realism Tests
// ============================================================================

test "message bus: dropout drops messages" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 2);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 2, 25.0);

    const config = SwarmConfig{
        .num_agents = 2,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
        .drop_prob = 1.0, // drop everything
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    try std.testing.expect(bus.send(0, 1, 1, "hello"));
    bus.deliver(&graph, 0);

    // All messages dropped
    try std.testing.expectEqual(@as(u32, 0), bus.total_messages_delivered);
    try std.testing.expect(bus.total_messages_dropped > 0);
}

test "message bus: latency delays messages" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 2);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 2, 25.0);

    const config = SwarmConfig{
        .num_agents = 2,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
        .latency_ticks = 2,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Send at step 0
    try std.testing.expect(bus.send(0, 1, 1, "hello"));
    bus.deliver(&graph, 0);
    // Should NOT be delivered yet at step 0
    try std.testing.expectEqual(@as(u32, 0), bus.total_messages_delivered);
    bus.clearStep();

    // Step 1: still pending
    bus.deliver(&graph, 1);
    try std.testing.expectEqual(@as(u32, 0), bus.total_messages_delivered);
    bus.clearStep();

    // Step 2: should be delivered now
    bus.deliver(&graph, 2);
    try std.testing.expectEqual(@as(u32, 1), bus.total_messages_delivered);
}

test "message bus: broadcast cap" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 3, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 4);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 4, 100.0);

    const config = SwarmConfig{
        .num_agents = 4,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
        .max_broadcast_recipients = 1,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 broadcasts — should only reach 1 neighbor
    try std.testing.expect(bus.send(0, MessageSlot.BROADCAST, 99, "bcast"));
    bus.deliver(&graph, 0);

    try std.testing.expectEqual(@as(u32, 1), bus.total_messages_delivered);
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

    const config = SwarmConfig{
        .num_agents = 3,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    const m = dispatcher.computeMetrics(&graph, &bus, 3);
    try std.testing.expect(m.connectivity_ratio > 0);
    try std.testing.expect(m.avg_neighbors > 0);
}

// ============================================================================
// Metrics Module Tests
// ============================================================================

test "metrics: fragmentation fully connected" {
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

    const components = metrics_mod.computeFragmentation(&graph, 3);
    try std.testing.expectEqual(@as(u32, 1), components);
}

test "metrics: fragmentation disconnected" {
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

    const components = metrics_mod.computeFragmentation(&graph, 3);
    try std.testing.expectEqual(@as(u32, 2), components); // {0,1} and {2}
}

test "metrics: near misses" {
    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 0.15, 0, 0, 0 }, // within 2x radius but not colliding (dist=0.15, col_radius=0.1)
        .{ 100, 0, 0, 0 },  // far away
    };

    const near_miss = metrics_mod.computeNearMisses(&positions, 0, 3, 0.1);
    try std.testing.expectEqual(@as(u32, 1), near_miss);
}

// ============================================================================
// Task Evaluator Tests
// ============================================================================

test "tasks: formation circle score" {
    var positions = [_][4]f32{
        // 4 agents on a circle of radius 5, centered at (0,0)
        .{ 5, 0, 0, 0 },
        .{ 0, 5, 0, 0 },
        .{ -5, 0, 0, 0 },
        .{ 0, -5, 0, 0 },
    };

    const allocator = std.testing.allocator;
    var grid = try UniformGrid.init(allocator, 10, 20.0, 16);
    defer grid.deinit();
    grid.rebuild(&positions, 0, 4);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 4, 400.0);

    var agent_states = [_]zeno.swarm.types.AgentState{
        .{ .agent_id = 0, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 1, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 2, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 3, .team_id = 0, .status = 1, .flags = 0 },
    };

    const params = [8]f32{ 0, 0, 5, 0, 0, 0, 0, 0 }; // center=(0,0), radius=5, circle
    const result = tasks_mod.evaluateTask(
        .formation,
        &positions,
        &positions, // velocities unused
        &graph,
        &agent_states,
        0,
        4,
        params,
    );

    // Agents are exactly on the target circle, score should be high
    try std.testing.expect(result.score > 0.5);
}

test "tasks: coverage score" {
    var positions = [_][4]f32{
        .{ 1, 1, 0, 0 },
        .{ 3, 3, 0, 0 },
        .{ 5, 5, 0, 0 },
        .{ 7, 7, 0, 0 },
    };

    const allocator = std.testing.allocator;
    var grid = try UniformGrid.init(allocator, 10, 20.0, 16);
    defer grid.deinit();
    grid.rebuild(&positions, 0, 4);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 4, 400.0);

    var agent_states = [_]zeno.swarm.types.AgentState{
        .{ .agent_id = 0, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 1, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 2, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 3, .team_id = 0, .status = 1, .flags = 0 },
    };

    const params = [8]f32{ 0, 0, 10, 10, 2.0, 0, 0, 0 }; // area 10x10, cell_size=2
    const result = tasks_mod.evaluateTask(
        .coverage,
        &positions,
        &positions,
        &graph,
        &agent_states,
        0,
        4,
        params,
    );

    try std.testing.expect(result.score > 0);
    try std.testing.expect(result.score <= 1.0);
}

test "tasks: pursuit all captured" {
    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 }, // pursuer 0
        .{ 5, 0, 0, 0 }, // pursuer 1
        .{ 0, 0.05, 0, 0 }, // evader at same position as pursuer 0
        .{ 5, 0.05, 0, 0 }, // evader at same position as pursuer 1
    };

    const allocator = std.testing.allocator;
    var grid = try UniformGrid.init(allocator, 10, 20.0, 16);
    defer grid.deinit();
    grid.rebuild(&positions, 0, 4);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 4, 400.0);

    var agent_states = [_]zeno.swarm.types.AgentState{
        .{ .agent_id = 0, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 1, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 2, .team_id = 0, .status = 1, .flags = 0 },
        .{ .agent_id = 3, .team_id = 0, .status = 1, .flags = 0 },
    };

    const params = [8]f32{ 2, 0.1, 0, 0, 0, 0, 0, 0 }; // 2 pursuers, capture_radius=0.1
    const result = tasks_mod.evaluateTask(
        .pursuit,
        &positions,
        &positions,
        &graph,
        &agent_states,
        0,
        4,
        params,
    );

    try std.testing.expect(result.score >= 1.0);
    try std.testing.expect(result.complete);
}

// ============================================================================
// Attack Tests
// ============================================================================

test "attack: jamming blocks communication" {
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

    const config = SwarmConfig{
        .num_agents = 3,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 sends to agent 1
    try std.testing.expect(bus.send(0, 1, 1, "hello"));
    bus.deliver(&graph, 0);

    // Now apply jamming to agent 1
    var attack_config = AttackConfig{
        .attack_type = .jamming,
        .num_targets = 1,
    };
    attack_config.target_agents[0] = 1;
    attacks_mod.applyJamming(&bus, &attack_config);

    // Agent 1's inbox should be maxed out (blocked)
    try std.testing.expectEqual(bus.max_messages_per_step, bus.inbox_counts[1]);
}

test "attack: partition splits graph" {
    const allocator = std.testing.allocator;

    var grid = try UniformGrid.init(allocator, 10, 5.0, 16);
    defer grid.deinit();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 3, 0, 0, 0 },
    };
    grid.rebuild(&positions, 0, 4);

    var graph = try AdjacencyGraph.init(allocator, 10, 32);
    defer graph.deinit();
    graph.buildFromGrid(&grid, &positions, 0, 4, 100.0);

    const edges_before = graph.total_edges;
    try std.testing.expect(edges_before > 0);

    var attack_config = AttackConfig{
        .attack_type = .partition,
        .intensity = 0.5, // split at agent index 2
    };
    attacks_mod.applyPartition(&graph, &attack_config);

    // Should have fewer edges (cross-partition edges removed)
    try std.testing.expect(graph.total_edges < edges_before);
}

test "attack: byzantine corrupts payloads" {
    const allocator = std.testing.allocator;

    const config = SwarmConfig{
        .num_agents = 3,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Agent 0 sends a known payload
    const payload = "hello world!";
    try std.testing.expect(bus.send(0, 1, 1, payload));

    // Apply byzantine to agent 0
    var attack_config = AttackConfig{
        .attack_type = .byzantine,
        .num_targets = 1,
        .seed = 42,
    };
    attack_config.target_agents[0] = 0;
    attacks_mod.applyByzantine(&bus, &attack_config, 0);

    // Verify payload was corrupted
    const slot = bus.outbox[0];
    const original = "hello world!";
    var different = false;
    for (0..original.len) |i| {
        if (slot.payload[i] != original[i]) {
            different = true;
            break;
        }
    }
    try std.testing.expect(different);
}

// ============================================================================
// Replay Tests
// ============================================================================

test "replay: record and verify" {
    const allocator = std.testing.allocator;

    var recorder = replay_mod.ReplayRecorder.init(allocator);
    defer recorder.deinit();

    recorder.startRecording();

    var positions = [_][4]f32{
        .{ 0, 0, 0, 0 }, // ground
        .{ 1, 2, 3, 0 },
        .{ 4, 5, 6, 0 },
    };
    var velocities = [_][4]f32{
        .{ 0, 0, 0, 0 },
        .{ 0.1, 0.2, 0.3, 0 },
        .{ 0.4, 0.5, 0.6, 0 },
    };

    const bus_config = SwarmConfig{
        .num_agents = 2,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, bus_config);
    defer bus.deinit();

    try recorder.recordFrame(0, &positions, &velocities, 1, 2, &bus, .{});
    try recorder.recordFrame(1, &positions, &velocities, 1, 2, &bus, .{});

    try std.testing.expectEqual(@as(usize, 2), recorder.frameCount());

    // Verify determinism with a copy
    var recorder2 = replay_mod.ReplayRecorder.init(allocator);
    defer recorder2.deinit();
    recorder2.startRecording();
    try recorder2.recordFrame(0, &positions, &velocities, 1, 2, &bus, .{});
    try recorder2.recordFrame(1, &positions, &velocities, 1, 2, &bus, .{});

    try std.testing.expect(recorder.verifyDeterminism(&recorder2));
}

test "replay: binary serialization roundtrip" {
    const allocator = std.testing.allocator;

    var recorder = replay_mod.ReplayRecorder.init(allocator);
    defer recorder.deinit();
    recorder.startRecording();

    var positions = [_][4]f32{
        .{ 1, 2, 3, 0 },
        .{ 4, 5, 6, 0 },
    };
    var velocities = [_][4]f32{
        .{ 0.1, 0.2, 0.3, 0 },
        .{ 0.4, 0.5, 0.6, 0 },
    };

    const bus_config = SwarmConfig{
        .num_agents = 2,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, bus_config);
    defer bus.deinit();

    try recorder.recordFrame(0, &positions, &velocities, 0, 2, &bus, .{});

    // Serialize to buffer
    var buf: std.ArrayList(u8) = .{};
    defer buf.deinit(allocator);
    try recorder.writeTo(buf.writer(allocator));

    // Deserialize
    var stream = std.io.fixedBufferStream(buf.items);
    var recorder2 = try replay_mod.ReplayRecorder.readFrom(stream.reader(), allocator);
    defer recorder2.deinit();

    try std.testing.expectEqual(@as(usize, 1), recorder2.frameCount());
    try std.testing.expect(recorder.verifyDeterminism(&recorder2));
}
