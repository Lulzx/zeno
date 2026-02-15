//! Adversarial attack simulation for swarm resilience testing.
//! Applies perturbations to the message bus and graph.

const std = @import("std");
const types = @import("types.zig");
const MessageBus = @import("message_bus.zig").MessageBus;
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const AttackType = types.AttackType;
const AttackConfig = types.AttackConfig;

/// Simple xorshift32 PRNG for deterministic attacks.
fn xorshift32(state: *u32) u32 {
    var s = state.*;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    state.* = s;
    return s;
}

fn xorshiftFloat(state: *u32) f32 {
    return @as(f32, @floatFromInt(xorshift32(state) & 0x7FFFFF)) / @as(f32, 0x7FFFFF);
}

/// Apply attack based on config.
pub fn applyAttack(
    bus: *MessageBus,
    graph: *AdjacencyGraph,
    config: *const AttackConfig,
    step_count: u64,
) void {
    switch (config.attack_type) {
        .none => {},
        .jamming => applyJamming(bus, config),
        .dropout => applyDropout(bus, graph, config, step_count),
        .byzantine => applyByzantine(bus, config, step_count),
        .partition => applyPartition(graph, config),
    }
}

/// Jamming: targeted agents can't send or receive.
pub fn applyJamming(bus: *MessageBus, config: *const AttackConfig) void {
    for (0..config.num_targets) |i| {
        const agent = config.target_agents[i];
        if (agent >= bus.num_agents) continue;

        // Block sending: zero outbox
        bus.outbox_counts[agent] = 0;
        // Block receiving: fill inbox to max
        bus.inbox_counts[agent] = bus.max_messages_per_step;
    }
}

/// Dropout: randomly disconnect agents with probability `intensity`.
pub fn applyDropout(
    bus: *MessageBus,
    graph: *AdjacencyGraph,
    config: *const AttackConfig,
    step_count: u64,
) void {
    var rng_state: u32 = config.seed +% @as(u32, @truncate(step_count));
    if (rng_state == 0) rng_state = 1;

    for (0..bus.num_agents) |agent_idx| {
        const agent: u32 = @intCast(agent_idx);
        const r = xorshiftFloat(&rng_state);
        if (r < config.intensity) {
            // Clear inbox
            bus.inbox_counts[agent] = 0;
            // Zero graph row (disconnect from all neighbors)
            const start = graph.row_ptr[agent];
            const end = graph.row_ptr[agent + 1];
            // We can't actually remove edges from CSR without rebuild,
            // so we mark the row as empty by setting row_ptr[agent+1] = row_ptr[agent]
            // This is safe because the graph is rebuilt every step.
            _ = start;
            _ = end;
            // Instead, zero the neighbor count by adjusting total_edges tracking
            graph.row_ptr[agent + 1] = graph.row_ptr[agent];
        }
    }
}

/// Byzantine: corrupt message payloads from targeted agents.
pub fn applyByzantine(
    bus: *MessageBus,
    config: *const AttackConfig,
    step_count: u64,
) void {
    var rng_state: u32 = config.seed +% @as(u32, @truncate(step_count)) +% 0xBEEF;
    if (rng_state == 0) rng_state = 1;

    for (0..config.num_targets) |t| {
        const agent = config.target_agents[t];
        if (agent >= bus.num_agents) continue;

        const msg_count = bus.outbox_counts[agent];
        for (0..msg_count) |m| {
            const slot_idx = agent * bus.max_messages_per_step + @as(u32, @intCast(m));
            const slot = &bus.outbox[slot_idx];
            // Corrupt payload bytes, preserve header
            for (0..slot.payload_len) |b| {
                slot.payload[b] = @truncate(xorshift32(&rng_state));
            }
        }
    }
}

/// Partition: remove all edges between two groups.
/// Group A = agents [0..threshold), Group B = agents [threshold..num_agents)
pub fn applyPartition(graph: *AdjacencyGraph, config: *const AttackConfig) void {
    const threshold: u32 = @intFromFloat(@as(f32, @floatFromInt(graph.num_agents)) * std.math.clamp(config.intensity, 0, 1));
    if (threshold == 0 or threshold >= graph.num_agents) return;

    // For each agent, filter out cross-partition neighbors
    // We do this by compacting the neighbor list in-place
    var new_total: u32 = 0;
    for (0..graph.num_agents) |agent_idx| {
        const agent: u32 = @intCast(agent_idx);
        const start = graph.row_ptr[agent];
        const end = graph.row_ptr[agent + 1];
        const agent_in_a = agent < threshold;

        graph.row_ptr[agent] = new_total;
        for (start..end) |e| {
            const neighbor = graph.neighbor_ids[e];
            const neighbor_in_a = neighbor < threshold;
            if (agent_in_a == neighbor_in_a) {
                // Same partition, keep edge
                graph.neighbor_ids[new_total] = neighbor;
                graph.neighbor_dists[new_total] = graph.neighbor_dists[e];
                new_total += 1;
            }
        }
    }
    graph.row_ptr[graph.num_agents] = new_total;
    graph.total_edges = new_total;
}
