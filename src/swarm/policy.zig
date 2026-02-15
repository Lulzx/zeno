//! Vtable interface for Zig-native swarm policies.
//! Python policies bypass this and pass bulk action arrays through the C API.

const types = @import("types.zig");
const MessageSlot = types.MessageSlot;
const AgentState = types.AgentState;

/// Function signature for a single-agent policy step.
/// Reads positions, velocities, neighbors, inbox, and agent state.
/// Writes to out_actions (force vector) and optionally queues outbox messages.
pub const PolicyFn = *const fn (
    agent_id: u32,
    positions: [][4]f32,
    velocities: [][4]f32,
    neighbors: []const u32,
    inbox: []const MessageSlot,
    agent_state: *AgentState,
    out_actions: []f32,
    outbox: *OutboxWriter,
) void;

/// Function signature for policy reset (called when env resets).
pub const ResetFn = *const fn (agent_id: u32, agent_state: *AgentState) void;

/// Vtable for a Zig-native swarm policy.
pub const PolicyVtable = struct {
    step_fn: PolicyFn,
    reset_fn: ?ResetFn = null,
    name: []const u8 = "unnamed",
};

/// Helper for policies to queue outgoing messages.
pub const OutboxWriter = struct {
    bus: *@import("message_bus.zig").MessageBus,
    sender_id: u32,

    /// Send a message to a specific agent or broadcast (receiver=0xFFFFFFFF).
    pub fn send(self: *OutboxWriter, receiver: u32, msg_type: u32, payload: []const u8) bool {
        return self.bus.send(self.sender_id, receiver, msg_type, payload);
    }

    /// Broadcast a message to all neighbors.
    pub fn broadcast(self: *OutboxWriter, msg_type: u32, payload: []const u8) bool {
        return self.bus.send(self.sender_id, MessageSlot.BROADCAST, msg_type, payload);
    }
};
