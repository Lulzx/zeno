//! Swarm step orchestration.
//! Coordinates grid rebuild, graph construction, message delivery,
//! policy dispatch, and physics stepping.

const std = @import("std");
const types = @import("types.zig");
const UniformGrid = @import("grid.zig").UniformGrid;
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageBus = @import("message_bus.zig").MessageBus;
const policy_mod = @import("policy.zig");
const PolicyVtable = policy_mod.PolicyVtable;
const OutboxWriter = policy_mod.OutboxWriter;
const SwarmMetrics = types.SwarmMetrics;
const AgentState = types.AgentState;

/// Execute one swarm step, orchestrating all subsystems.
pub fn stepOnce(
    grid: *UniformGrid,
    graph: *AdjacencyGraph,
    message_bus: *MessageBus,
    agent_states: []AgentState,
    positions: [][4]f32,
    velocities: [][4]f32,
    body_offset: u32,
    num_agents: u32,
    range_sq: f32,
    policy: ?PolicyVtable,
    external_actions: ?[]f32,
    action_dim: u32,
) void {
    // 1. Rebuild spatial grid from current positions
    grid.rebuild(positions, body_offset, num_agents);

    // 2. Build adjacency graph from grid
    graph.buildFromGrid(grid, positions, body_offset, num_agents, range_sq);

    // 3. Deliver messages queued from the previous step
    message_bus.deliver(graph);

    // 4. Dispatch policy or apply external actions
    if (external_actions) |actions| {
        // External actions provided (from Python) — nothing extra to do here,
        // the caller will pass actions to World.step().
        _ = actions;
    } else if (policy) |vtable| {
        // Zig-native policy: call step_fn for each agent
        for (0..num_agents) |i| {
            const agent_id: u32 = @intCast(i);
            const neighbors = graph.getNeighbors(agent_id);
            const inbox = message_bus.getInbox(agent_id);

            // Create per-agent outbox writer
            var writer = OutboxWriter{
                .bus = message_bus,
                .sender_id = agent_id,
            };

            // Action slice for this agent (output)
            // For now we don't have a separate action buffer for native policies
            // Policies using vtable should write to their own buffer
            var dummy_actions: [1]f32 = .{0};
            const action_slice = if (action_dim > 0) dummy_actions[0..@min(action_dim, 1)] else dummy_actions[0..0];

            vtable.step_fn(
                agent_id,
                positions,
                velocities,
                neighbors,
                inbox,
                &agent_states[i],
                action_slice,
                &writer,
            );
        }
    }

    // 5. Physics step is done by the caller (World.step())
    // 6. Clear message bus for next step
    message_bus.clearStep();
}

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
        .fragmentation_score = 0, // TODO: compute connected components
        .collision_count = 0,
        .message_count = message_bus.total_messages_delivered,
        .bytes_sent = message_bus.total_bytes_sent,
        .total_edges = graph.total_edges,
        .avg_neighbors = avg_neighbors,
    };
}
