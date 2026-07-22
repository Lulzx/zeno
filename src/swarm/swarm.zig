//! Main Swarm struct — composes all swarm subsystems.
//! Provides the primary API for creating and stepping a swarm simulation.

const std = @import("std");
const types = @import("types.zig");
const UniformGrid = @import("grid.zig").UniformGrid;
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageBus = @import("message_bus.zig").MessageBus;
const PolicyVtable = @import("policy.zig").PolicyVtable;
const dispatcher = @import("dispatcher.zig");
const tasks_mod = @import("tasks.zig");
const replay_mod = @import("replay.zig");
const metrics_mod = @import("metrics.zig");

const SwarmConfig = types.SwarmConfig;
const AgentState = types.AgentState;
const SwarmMetrics = types.SwarmMetrics;
const AttackConfig = types.AttackConfig;
const TaskResult = types.TaskResult;

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
    /// Optional attack configuration.
    attack_config: ?AttackConfig,
    /// Optional replay recorder.
    recorder: ?replay_mod.ReplayRecorder,
    /// Swarm-owned actions buffer (num_agents * action_dim), filled by the
    /// policy or by external actions each step. Read back via getActions().
    actions: []f32,
    /// Per-agent action dimension the buffer was sized for.
    action_dim: u32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SwarmConfig) !Swarm {
        const num_agents = config.num_agents;
        const grid_dim: u32 = 64; // Default grid resolution

        var grid = try UniformGrid.init(allocator, num_agents, config.grid_cell_size, grid_dim);
        errdefer grid.deinit();

        var graph = try AdjacencyGraph.init(allocator, num_agents, config.max_neighbors);
        errdefer graph.deinit();

        var message_bus = try MessageBus.init(allocator, config);
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
            .attack_config = null,
            .recorder = null,
            .actions = &.{},
            .action_dim = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Swarm) void {
        self.grid.deinit();
        self.graph.deinit();
        self.message_bus.deinit();
        self.allocator.free(self.agent_states);
        if (self.actions.len > 0) self.allocator.free(self.actions);
        if (self.recorder) |*rec| rec.deinit();
    }

    /// Ensure the actions buffer matches num_agents * action_dim.
    fn ensureActions(self: *Swarm, action_dim: u32) []f32 {
        const needed: usize = @as(usize, self.config.num_agents) * action_dim;
        if (self.actions.len != needed) {
            const new_buf = self.allocator.alloc(f32, needed) catch return self.actions[0..0];
            if (self.actions.len > 0) self.allocator.free(self.actions);
            self.actions = new_buf;
            self.action_dim = action_dim;
        }
        @memset(self.actions, 0);
        return self.actions;
    }

    /// Actions produced by the last step (policy output or external actions).
    pub fn getActions(self: *const Swarm) []const f32 {
        return self.actions;
    }

    /// Execute one swarm step.
    pub fn step(
        self: *Swarm,
        positions: [][4]f32,
        velocities: [][4]f32,
        external_actions: ?[]const f32,
        action_dim: u32,
    ) void {
        const range_sq = self.config.communication_range * self.config.communication_range;

        const ac_ptr: ?*const AttackConfig = if (self.attack_config) |*ac| ac else null;
        const actions_out = self.ensureActions(action_dim);

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
            actions_out,
            action_dim,
            self.step_count,
            ac_ptr,
        );

        self.metrics = dispatcher.computeMetrics(
            &self.graph,
            &self.message_bus,
            self.config.num_agents,
        );

        // Update fragmentation score
        self.metrics.fragmentation_score = @floatFromInt(
            metrics_mod.computeFragmentation(&self.graph, self.config.num_agents),
        );

        // Record frame if recording
        if (self.recorder) |*rec| {
            rec.recordFrame(
                self.step_count,
                positions,
                velocities,
                self.body_offset,
                self.config.num_agents,
                &self.message_bus,
                self.metrics,
            ) catch {};
        }

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

    /// Set attack configuration.
    pub fn setAttack(self: *Swarm, config: AttackConfig) void {
        self.attack_config = config;
    }

    /// Clear attack configuration.
    pub fn clearAttack(self: *Swarm) void {
        self.attack_config = null;
    }

    /// Evaluate a task.
    pub fn evaluateTask(
        self: *const Swarm,
        task_type: tasks_mod.TaskType,
        positions: [][4]f32,
        velocities: [][4]f32,
        params: [8]f32,
    ) TaskResult {
        return tasks_mod.evaluateTask(
            task_type,
            positions,
            velocities,
            &self.graph,
            self.agent_states,
            self.body_offset,
            self.config.num_agents,
            params,
        );
    }

    /// Start recording replay frames.
    pub fn startRecording(self: *Swarm) void {
        if (self.recorder == null) {
            self.recorder = replay_mod.ReplayRecorder.init(self.allocator);
        }
        if (self.recorder) |*rec| rec.startRecording();
    }

    /// Stop recording replay frames.
    pub fn stopRecording(self: *Swarm) void {
        if (self.recorder) |*rec| rec.stopRecording();
    }

    /// Get replay stats.
    pub fn getReplayStats(self: *const Swarm) types.ReplayStats {
        if (self.recorder) |*rec| return rec.getStats();
        return .{};
    }
};
