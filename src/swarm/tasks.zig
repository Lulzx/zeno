//! Task evaluators for swarm cooperative objectives.
//! Stateless functions that score swarm performance on specific tasks.

const std = @import("std");
const types = @import("types.zig");
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const AgentState = types.AgentState;
const TaskResult = types.TaskResult;

/// Task types.
pub const TaskType = enum(u32) {
    formation = 0,
    coverage = 1,
    pursuit = 2,
    tracking = 3,
};

/// Evaluate a task by type.
pub fn evaluateTask(
    task_type: TaskType,
    positions: [][4]f32,
    velocities: [][4]f32,
    graph: *const AdjacencyGraph,
    agent_states: []const AgentState,
    body_offset: u32,
    num_agents: u32,
    params: [8]f32,
) TaskResult {
    _ = velocities;
    return switch (task_type) {
        .formation => evaluateFormation(positions, body_offset, num_agents, params),
        .coverage => evaluateCoverage(positions, body_offset, num_agents, params),
        .pursuit => evaluatePursuit(positions, body_offset, num_agents, params),
        .tracking => evaluateTrackingDefense(positions, graph, agent_states, body_offset, num_agents, params),
    };
}

/// Formation task: agents form a target shape.
/// params: [center_x, center_y, target_radius, formation_type, 0, 0, 0, 0]
/// formation_type: 0=circle, 1=line, 2=grid
fn evaluateFormation(
    positions: [][4]f32,
    body_offset: u32,
    num_agents: u32,
    params: [8]f32,
) TaskResult {
    if (num_agents == 0) return .{};

    const center_x = params[0];
    const center_y = params[1];
    const target_radius = params[2];
    const formation_type: u32 = @intFromFloat(@max(0, params[3]));

    var total_error: f32 = 0;
    var detail = [4]f32{ 0, 0, 0, 0 };

    for (0..num_agents) |i| {
        const pos = positions[body_offset + i];
        const target = formationTarget(
            @intCast(i),
            num_agents,
            center_x,
            center_y,
            target_radius,
            formation_type,
        );
        const dx = pos[0] - target[0];
        const dy = pos[1] - target[1];
        const err = @sqrt(dx * dx + dy * dy);
        total_error += err;
    }

    const mean_error = total_error / @as(f32, @floatFromInt(num_agents));
    const score = std.math.clamp(1.0 - (mean_error / @max(target_radius, 0.001)), 0, 1);

    detail[0] = mean_error;
    detail[1] = @floatFromInt(num_agents);

    return .{
        .score = score,
        .complete = score > 0.95,
        .detail = detail,
    };
}

fn formationTarget(
    agent_id: u32,
    num_agents: u32,
    center_x: f32,
    center_y: f32,
    radius: f32,
    formation_type: u32,
) [2]f32 {
    const n = @as(f32, @floatFromInt(num_agents));
    const i = @as(f32, @floatFromInt(agent_id));
    return switch (formation_type) {
        0 => .{ // circle
            center_x + radius * @cos(2.0 * std.math.pi * i / n),
            center_y + radius * @sin(2.0 * std.math.pi * i / n),
        },
        1 => .{ // line
            center_x - radius + 2.0 * radius * i / @max(n - 1.0, 1.0),
            center_y,
        },
        else => blk: { // grid
            const side = @ceil(@sqrt(n));
            const row = @floor(i / side);
            const col = i - row * side;
            const spacing = 2.0 * radius / @max(side - 1.0, 1.0);
            break :blk .{
                center_x - radius + col * spacing,
                center_y - radius + row * spacing,
            };
        },
    };
}

/// Coverage task: measure spatial coverage of a rectangular area.
/// params: [x_min, y_min, x_max, y_max, cell_size, 0, 0, 0]
fn evaluateCoverage(
    positions: [][4]f32,
    body_offset: u32,
    num_agents: u32,
    params: [8]f32,
) TaskResult {
    if (num_agents == 0) return .{};

    const x_min = params[0];
    const y_min = params[1];
    const x_max = params[2];
    const y_max = params[3];
    const cell_size = @max(params[4], 0.1);

    const nx: u32 = @intFromFloat(@ceil((x_max - x_min) / cell_size));
    const ny: u32 = @intFromFloat(@ceil((y_max - y_min) / cell_size));
    const total_cells = nx * ny;

    if (total_cells == 0) return .{};

    // Use a bitfield for covered cells (max 4096 cells = 512 bytes)
    const MAX_CELLS: u32 = 4096;
    const actual_cells = @min(total_cells, MAX_CELLS);
    var covered = [_]u8{0} ** (MAX_CELLS / 8 + 1);

    var covered_count: u32 = 0;
    for (0..num_agents) |i| {
        const pos = positions[body_offset + i];
        if (pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max) continue;
        const cx: u32 = @min(@as(u32, @intFromFloat((pos[0] - x_min) / cell_size)), nx - 1);
        const cy: u32 = @min(@as(u32, @intFromFloat((pos[1] - y_min) / cell_size)), ny - 1);
        const cell_idx = cy * nx + cx;
        if (cell_idx < actual_cells) {
            const byte_idx = cell_idx / 8;
            const bit_idx: u3 = @intCast(cell_idx % 8);
            if (covered[byte_idx] & (@as(u8, 1) << bit_idx) == 0) {
                covered[byte_idx] |= @as(u8, 1) << bit_idx;
                covered_count += 1;
            }
        }
    }

    const score = @as(f32, @floatFromInt(covered_count)) / @as(f32, @floatFromInt(actual_cells));
    return .{
        .score = score,
        .complete = score > 0.9,
        .detail = .{
            @floatFromInt(covered_count),
            @floatFromInt(actual_cells),
            0,
            0,
        },
    };
}

/// Pursuit-evasion task: N pursuers chase M evaders.
/// params: [num_pursuers, capture_radius, 0, 0, 0, 0, 0, 0]
fn evaluatePursuit(
    positions: [][4]f32,
    body_offset: u32,
    num_agents: u32,
    params: [8]f32,
) TaskResult {
    if (num_agents == 0) return .{};

    const num_pursuers: u32 = @intFromFloat(@max(1, @min(params[0], @as(f32, @floatFromInt(num_agents - 1)))));
    const capture_radius = @max(params[1], 0.01);
    const cap_sq = capture_radius * capture_radius;

    const num_evaders = num_agents - num_pursuers;
    if (num_evaders == 0) return .{ .score = 1.0, .complete = true };

    var captured: u32 = 0;
    for (num_pursuers..num_agents) |e| {
        const epos = positions[body_offset + e];
        for (0..num_pursuers) |p| {
            const ppos = positions[body_offset + p];
            const dx = epos[0] - ppos[0];
            const dy = epos[1] - ppos[1];
            const dz = epos[2] - ppos[2];
            if (dx * dx + dy * dy + dz * dz <= cap_sq) {
                captured += 1;
                break;
            }
        }
    }

    const score = @as(f32, @floatFromInt(captured)) / @as(f32, @floatFromInt(num_evaders));
    return .{
        .score = score,
        .complete = captured == num_evaders,
        .detail = .{
            @floatFromInt(captured),
            @floatFromInt(num_evaders),
            @floatFromInt(num_pursuers),
            0,
        },
    };
}

/// Tracking-defense task: defenders maintain connectivity while tracking a target.
/// params: [target_x, target_y, target_z, track_radius, 0, 0, 0, 0]
fn evaluateTrackingDefense(
    positions: [][4]f32,
    graph: *const AdjacencyGraph,
    agent_states: []const AgentState,
    body_offset: u32,
    num_agents: u32,
    params: [8]f32,
) TaskResult {
    _ = agent_states;
    if (num_agents == 0) return .{};

    const target_x = params[0];
    const target_y = params[1];
    const target_z = params[2];
    const track_radius = @max(params[3], 0.01);
    const track_sq = track_radius * track_radius;

    // Count agents within tracking radius
    var tracking_count: u32 = 0;
    for (0..num_agents) |i| {
        const pos = positions[body_offset + i];
        const dx = pos[0] - target_x;
        const dy = pos[1] - target_y;
        const dz = pos[2] - target_z;
        if (dx * dx + dy * dy + dz * dz <= track_sq) {
            tracking_count += 1;
        }
    }

    // Compute connectivity ratio
    const max_edges = num_agents * (num_agents - 1);
    const connectivity = if (max_edges > 0)
        @as(f32, @floatFromInt(graph.total_edges)) / @as(f32, @floatFromInt(max_edges))
    else
        0;

    const track_frac = @as(f32, @floatFromInt(tracking_count)) / @as(f32, @floatFromInt(num_agents));
    const score = track_frac * connectivity;

    return .{
        .score = score,
        .complete = score > 0.8,
        .detail = .{
            track_frac,
            connectivity,
            @floatFromInt(tracking_count),
            0,
        },
    };
}
