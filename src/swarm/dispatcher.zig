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
const metrics_mod = @import("metrics.zig");
const attacks_mod = @import("attacks.zig");
const SwarmMetrics = types.SwarmMetrics;
const AgentState = types.AgentState;
const AttackConfig = types.AttackConfig;

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
    step_count: u64,
    attack_config: ?*const AttackConfig,
) void {
    // 1. Rebuild spatial grid from current positions
    grid.rebuild(positions, body_offset, num_agents);

    // 2. Build adjacency graph from grid
    graph.buildFromGrid(grid, positions, body_offset, num_agents, range_sq);

    // 3. Apply pre-delivery attacks (byzantine corrupts outbox before delivery)
    if (attack_config) |ac| {
        if (ac.attack_type == .byzantine) {
            attacks_mod.applyByzantine(message_bus, ac, step_count);
        }
    }

    // 4. Deliver messages queued from the previous step
    message_bus.deliver(graph, step_count);

    // 5. Apply post-delivery attacks (jamming, dropout, partition)
    if (attack_config) |ac| {
        switch (ac.attack_type) {
            .jamming => attacks_mod.applyJamming(message_bus, ac),
            .dropout => attacks_mod.applyDropout(message_bus, graph, ac, step_count),
            .partition => attacks_mod.applyPartition(graph, ac),
            else => {},
        }
    }

    // 6. Dispatch policy or apply external actions
    if (external_actions) |actions| {
        _ = actions;
    } else if (policy) |vtable| {
        for (0..num_agents) |i| {
            const agent_id: u32 = @intCast(i);
            const neighbors = graph.getNeighbors(agent_id);
            const inbox = message_bus.getInbox(agent_id);

            var writer = OutboxWriter{
                .bus = message_bus,
                .sender_id = agent_id,
            };

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

    // 7. Physics step is done by the caller (World.step())
    // 8. Clear message bus for next step
    message_bus.clearStep();
}

/// Compute swarm metrics from current state. Delegates to metrics module.
pub fn computeMetrics(
    graph: *const AdjacencyGraph,
    message_bus: *const MessageBus,
    num_agents: u32,
) SwarmMetrics {
    return metrics_mod.computeMetrics(graph, message_bus, num_agents);
}
