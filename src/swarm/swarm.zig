//! Main Swarm struct — composes all swarm subsystems.
//! Provides the primary API for creating and stepping a swarm simulation.

const std = @import("std");
const types = @import("types.zig");
const UniformGrid = @import("grid.zig").UniformGrid;
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageBus = @import("message_bus.zig").MessageBus;
const PolicyVtable = @import("policy.zig").PolicyVtable;
const dispatcher = @import("dispatcher.zig");

const SwarmConfig = types.SwarmConfig;
const AgentState = types.AgentState;
const SwarmMetrics = types.SwarmMetrics;

/// Main swarm simulation struct.
pub const Swarm = struct {
    config: SwarmConfig,
    grid: UniformGrid,
    graph: AdjacencyGraph,
    message_bus: MessageBus,
    agent_states: []AgentState,

    /// Index of the first agent body in the world's body array (skip ground plane etc).
    body_offset: u32,
    /// Optional Zig-native policy.
    policy: ?PolicyVtable,
    /// Number of steps executed.
    step_count: u64,
    /// Most recent metrics.
    metrics: SwarmMetrics,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SwarmConfig) !Swarm {
        const num_agents = config.num_agents;
        const grid_dim: u32 = 64; // Default grid resolution

        var grid = try UniformGrid.init(allocator, num_agents, config.grid_cell_size, grid_dim);
        errdefer grid.deinit();

        var graph = try AdjacencyGraph.init(allocator, num_agents, config.max_neighbors);
        errdefer graph.deinit();

        var message_bus = try MessageBus.init(
            allocator,
            num_agents,
            config.max_messages_per_step,
            config.max_message_bytes,
        );
        errdefer message_bus.deinit();

        const agent_states = try allocator.alloc(AgentState, num_agents);
        for (0..num_agents) |i| {
            agent_states[i] = .{
                .agent_id = @intCast(i),
                .team_id = 0,
                .status = 1, // active
                .flags = 0,
            };
        }

        return .{
            .config = config,
            .grid = grid,
            .graph = graph,
            .message_bus = message_bus,
            .agent_states = agent_states,
            .body_offset = 0,
            .policy = null,
            .step_count = 0,
            .metrics = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Swarm) void {
        self.grid.deinit();
        self.graph.deinit();
        self.message_bus.deinit();
        self.allocator.free(self.agent_states);
    }

    /// Execute one swarm step.
    /// `positions` and `velocities` are from the World's state buffers.
    /// `external_actions` is optional (for Python-driven policies).
    /// The caller is responsible for calling World.step() after this.
    pub fn step(
        self: *Swarm,
        positions: [][4]f32,
        velocities: [][4]f32,
        external_actions: ?[]f32,
        action_dim: u32,
    ) void {
        const range_sq = self.config.communication_range * self.config.communication_range;

        dispatcher.stepOnce(
            &self.grid,
            &self.graph,
            &self.message_bus,
            self.agent_states,
            positions,
            velocities,
            self.body_offset,
            self.config.num_agents,
            range_sq,
            self.policy,
            external_actions,
            action_dim,
        );

        self.metrics = dispatcher.computeMetrics(
            &self.graph,
            &self.message_bus,
            self.config.num_agents,
        );

        self.step_count += 1;
    }

    /// Set the body offset (first agent body index).
    pub fn setBodyOffset(self: *Swarm, offset: u32) void {
        self.body_offset = offset;
    }

    /// Get neighbor counts for all agents.
    pub fn getNeighborCounts(self: *const Swarm, out: []u32) void {
        const n = @min(out.len, self.config.num_agents);
        for (0..n) |i| {
            out[i] = self.graph.neighborCount(@intCast(i));
        }
    }
};
