//! Benchmarks for the Zeno swarm platform.
//! Measures grid rebuild, graph build, message delivery, and full step times.

const std = @import("std");
const zeno = @import("zeno");

const UniformGrid = zeno.swarm.grid.UniformGrid;
const AdjacencyGraph = zeno.swarm.graph.AdjacencyGraph;
const MessageBus = zeno.swarm.message_bus.MessageBus;
const Swarm = zeno.swarm.swarm_mod.Swarm;
const SwarmConfig = zeno.swarm.types.SwarmConfig;
const MessageSlot = zeno.swarm.types.MessageSlot;

fn benchmarkGridRebuild(allocator: std.mem.Allocator, num_agents: u32) !void {
    var grid = try UniformGrid.init(allocator, num_agents, 10.0, 64);
    defer grid.deinit();

    const positions = try allocator.alloc([4]f32, num_agents);
    defer allocator.free(positions);

    // Scatter agents randomly
    var rng_state: u32 = 12345;
    for (positions) |*pos| {
        pos.* = .{
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 10.0,
            0,
        };
    }

    const iters: u32 = 100;
    var timer = try std.time.Timer.start();

    for (0..iters) |_| {
        grid.rebuild(positions, 0, num_agents);
    }

    const elapsed_ns = timer.read();
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    std.debug.print("  Grid rebuild ({d:>6} agents): {d:>8.1} us\n", .{ num_agents, per_iter_us });
}

fn benchmarkGraphBuild(allocator: std.mem.Allocator, num_agents: u32) !void {
    var grid = try UniformGrid.init(allocator, num_agents, 10.0, 64);
    defer grid.deinit();

    var graph = try AdjacencyGraph.init(allocator, num_agents, 32);
    defer graph.deinit();

    const positions = try allocator.alloc([4]f32, num_agents);
    defer allocator.free(positions);

    var rng_state: u32 = 12345;
    for (positions) |*pos| {
        pos.* = .{
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 10.0,
            0,
        };
    }

    grid.rebuild(positions, 0, num_agents);

    const iters: u32 = 100;
    var timer = try std.time.Timer.start();

    for (0..iters) |_| {
        graph.buildFromGrid(&grid, positions, 0, num_agents, 100.0);
    }

    const elapsed_ns = timer.read();
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    std.debug.print("  Graph build  ({d:>6} agents): {d:>8.1} us\n", .{ num_agents, per_iter_us });
}

fn benchmarkMessageDelivery(allocator: std.mem.Allocator, num_agents: u32) !void {
    var grid = try UniformGrid.init(allocator, num_agents, 10.0, 64);
    defer grid.deinit();

    var graph = try AdjacencyGraph.init(allocator, num_agents, 32);
    defer graph.deinit();

    const config = SwarmConfig{
        .num_agents = num_agents,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();

    const positions = try allocator.alloc([4]f32, num_agents);
    defer allocator.free(positions);

    var rng_state: u32 = 12345;
    for (positions) |*pos| {
        pos.* = .{
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 10.0,
            0,
        };
    }

    grid.rebuild(positions, 0, num_agents);
    graph.buildFromGrid(&grid, positions, 0, num_agents, 100.0);

    const iters: u32 = 100;
    var timer = try std.time.Timer.start();

    for (0..iters) |iter| {
        bus.clearStep();
        // Each agent broadcasts
        for (0..@min(num_agents, 1000)) |i| {
            _ = bus.send(@intCast(i), MessageSlot.BROADCAST, 1, "bench");
        }
        bus.deliver(&graph, iter);
    }

    const elapsed_ns = timer.read();
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    std.debug.print("  Msg delivery ({d:>6} agents): {d:>8.1} us\n", .{ num_agents, per_iter_us });
}

fn benchmarkFullStep(allocator: std.mem.Allocator, num_agents: u32) !void {
    const config = SwarmConfig{
        .num_agents = num_agents,
        .communication_range = 10.0,
        .max_neighbors = 32,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
        .grid_cell_size = 10.0,
    };

    var swarm = try Swarm.init(allocator, config);
    defer swarm.deinit();

    const total_bodies = num_agents + 1; // +1 for ground
    const positions = try allocator.alloc([4]f32, total_bodies);
    defer allocator.free(positions);
    const velocities = try allocator.alloc([4]f32, total_bodies);
    defer allocator.free(velocities);

    positions[0] = .{ 0, 0, 0, 0 }; // ground
    velocities[0] = .{ 0, 0, 0, 0 };

    var rng_state: u32 = 12345;
    for (1..total_bodies) |i| {
        positions[i] = .{
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 100.0,
            randFloat(&rng_state) * 10.0,
            0,
        };
        velocities[i] = .{ 0, 0, 0, 0 };
    }

    swarm.setBodyOffset(1);

    const iters: u32 = 100;
    var timer = try std.time.Timer.start();

    for (0..iters) |_| {
        swarm.step(positions, velocities, null, 0);
    }

    const elapsed_ns = timer.read();
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    std.debug.print("  Full step    ({d:>6} agents): {d:>8.1} us\n", .{ num_agents, per_iter_us });
}

fn randFloat(state: *u32) f32 {
    var s = state.*;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    state.* = s;
    return @as(f32, @floatFromInt(s & 0x7FFFFF)) / @as(f32, 0x7FFFFF);
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n=== Zeno Swarm Benchmark ===\n\n", .{});

    const agent_counts = [_]u32{ 100, 1000, 10000 };

    std.debug.print("Grid Rebuild:\n", .{});
    for (agent_counts) |n| {
        try benchmarkGridRebuild(allocator, n);
    }

    std.debug.print("\nGraph Build:\n", .{});
    for (agent_counts) |n| {
        try benchmarkGraphBuild(allocator, n);
    }

    std.debug.print("\nMessage Delivery:\n", .{});
    for (agent_counts) |n| {
        try benchmarkMessageDelivery(allocator, n);
    }

    std.debug.print("\nFull Swarm Step:\n", .{});
    for (agent_counts) |n| {
        try benchmarkFullStep(allocator, n);
    }

    std.debug.print("\nDone.\n", .{});
}
