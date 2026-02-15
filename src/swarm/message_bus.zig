//! Budgeted inter-agent communication system.
//! Outbox/inbox pattern with per-agent message count and byte budget.

const std = @import("std");
const types = @import("types.zig");
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageSlot = types.MessageSlot;

/// Budgeted message bus for swarm communication.
pub const MessageBus = struct {
    /// Outbox: max_messages_per_step messages per agent.
    outbox: []MessageSlot,
    /// Number of messages queued per agent in outbox.
    outbox_counts: []u32,
    /// Inbox: max_messages_per_step messages per agent.
    inbox: []MessageSlot,
    /// Number of messages delivered per agent in inbox.
    inbox_counts: []u32,
    /// Bytes sent per agent this step.
    bytes_sent: []u32,

    num_agents: u32,
    max_messages_per_step: u32,
    max_message_bytes: u32,
    allocator: std.mem.Allocator,

    // Step-level metrics
    total_messages_delivered: u32 = 0,
    total_bytes_sent: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        num_agents: u32,
        max_messages_per_step: u32,
        max_message_bytes: u32,
    ) !MessageBus {
        const slots_per_agent = max_messages_per_step;
        const total_slots = num_agents * slots_per_agent;

        const outbox = try allocator.alloc(MessageSlot, total_slots);
        @memset(outbox, std.mem.zeroes(MessageSlot));

        const inbox = try allocator.alloc(MessageSlot, total_slots);
        @memset(inbox, std.mem.zeroes(MessageSlot));

        const outbox_counts = try allocator.alloc(u32, num_agents);
        @memset(outbox_counts, 0);
        const inbox_counts = try allocator.alloc(u32, num_agents);
        @memset(inbox_counts, 0);
        const bytes_sent_buf = try allocator.alloc(u32, num_agents);
        @memset(bytes_sent_buf, 0);

        return .{
            .outbox = outbox,
            .outbox_counts = outbox_counts,
            .inbox = inbox,
            .inbox_counts = inbox_counts,
            .bytes_sent = bytes_sent_buf,
            .num_agents = num_agents,
            .max_messages_per_step = max_messages_per_step,
            .max_message_bytes = max_message_bytes,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MessageBus) void {
        self.allocator.free(self.outbox);
        self.allocator.free(self.outbox_counts);
        self.allocator.free(self.inbox);
        self.allocator.free(self.inbox_counts);
        self.allocator.free(self.bytes_sent);
    }

    /// Queue a message from sender to receiver. Returns false if budget exceeded.
    pub fn send(
        self: *MessageBus,
        sender: u32,
        receiver: u32,
        msg_type: u32,
        payload: []const u8,
    ) bool {
        if (sender >= self.num_agents) return false;

        const count = self.outbox_counts[sender];
        if (count >= self.max_messages_per_step) return false;

        const payload_len: u32 = @intCast(@min(payload.len, self.max_message_bytes));
        if (payload_len > 48) return false;

        // Check byte budget
        if (self.bytes_sent[sender] + payload_len > self.max_message_bytes * self.max_messages_per_step) return false;

        const slot_idx = sender * self.max_messages_per_step + count;
        self.outbox[slot_idx] = .{
            .sender_id = sender,
            .receiver_id = receiver,
            .message_type = msg_type,
            .payload_len = payload_len,
        };
        if (payload_len > 0) {
            @memcpy(self.outbox[slot_idx].payload[0..payload_len], payload[0..payload_len]);
        }

        self.outbox_counts[sender] = count + 1;
        self.bytes_sent[sender] += payload_len;
        return true;
    }

    /// Deliver messages based on neighbor adjacency.
    /// Only delivers to agents that are neighbors (within communication range).
    pub fn deliver(self: *MessageBus, graph: *const AdjacencyGraph) void {
        // Clear inbox counts
        @memset(self.inbox_counts, 0);
        self.total_messages_delivered = 0;
        self.total_bytes_sent = 0;

        for (0..self.num_agents) |sender_idx| {
            const sender: u32 = @intCast(sender_idx);
            const msg_count = self.outbox_counts[sender];
            if (msg_count == 0) continue;

            for (0..msg_count) |m| {
                const slot_idx = sender * self.max_messages_per_step + @as(u32, @intCast(m));
                const msg = self.outbox[slot_idx];

                if (msg.receiver_id == MessageSlot.BROADCAST) {
                    // Broadcast to all neighbors
                    const neighbors = graph.getNeighbors(sender);
                    for (neighbors) |neighbor| {
                        self.deliverToAgent(neighbor, &msg);
                    }
                } else {
                    // Point-to-point: only deliver if receiver is a neighbor
                    const neighbors = graph.getNeighbors(sender);
                    for (neighbors) |neighbor| {
                        if (neighbor == msg.receiver_id) {
                            self.deliverToAgent(msg.receiver_id, &msg);
                            break;
                        }
                    }
                }
            }
        }
    }

    fn deliverToAgent(self: *MessageBus, receiver: u32, msg: *const MessageSlot) void {
        if (receiver >= self.num_agents) return;

        const count = self.inbox_counts[receiver];
        if (count >= self.max_messages_per_step) return;

        const inbox_idx = receiver * self.max_messages_per_step + count;
        self.inbox[inbox_idx] = msg.*;
        self.inbox_counts[receiver] = count + 1;
        self.total_messages_delivered += 1;
        self.total_bytes_sent += msg.payload_len;
    }

    /// Clear all outbox/inbox state for the next step.
    pub fn clearStep(self: *MessageBus) void {
        @memset(self.outbox_counts, 0);
        @memset(self.inbox_counts, 0);
        @memset(self.bytes_sent, 0);
        self.total_messages_delivered = 0;
        self.total_bytes_sent = 0;
    }

    /// Get inbox messages for a specific agent.
    pub fn getInbox(self: *const MessageBus, agent_id: u32) []const MessageSlot {
        if (agent_id >= self.num_agents) return &.{};
        const count = self.inbox_counts[agent_id];
        const start = agent_id * self.max_messages_per_step;
        return self.inbox[start .. start + count];
    }
};

test "message bus send and deliver" {
    const allocator = std.testing.allocator;

    var bus = try MessageBus.init(allocator, 4, 4, 48);
    defer bus.deinit();
    bus.clearStep();

    // Send a message from agent 0 to agent 1
    const payload = "hello";
    const ok = bus.send(0, 1, 1, payload);
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(u32, 1), bus.outbox_counts[0]);
}
