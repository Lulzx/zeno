//! Budgeted inter-agent communication system.
//! Outbox/inbox pattern with per-agent message count and byte budget.
//! Supports latency, dropout, broadcast caps, and inbox limits.

const std = @import("std");
const types = @import("types.zig");
const AdjacencyGraph = @import("graph.zig").AdjacencyGraph;
const MessageSlot = types.MessageSlot;
const SwarmConfig = types.SwarmConfig;

/// Simple xorshift32 PRNG for deterministic dropout.
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

/// Budgeted message bus for swarm communication.
pub const MessageBus = struct {
    /// Outbox: max_messages_per_step messages per agent.
    outbox: []MessageSlot,
    /// Number of messages queued per agent in outbox.
    outbox_counts: []u32,
    /// Inbox: effective_inbox_capacity messages per agent.
    inbox: []MessageSlot,
    /// Number of messages delivered per agent in inbox.
    inbox_counts: []u32,
    /// Bytes sent per agent this step.
    bytes_sent: []u32,

    /// Pending queue for latency (null if latency_ticks == 0).
    pending_queue: ?[]MessageSlot,
    /// Pending counts: num_agents * (latency_ticks + 1) ring buffer.
    pending_counts: ?[]u32,

    num_agents: u32,
    max_messages_per_step: u32,
    max_message_bytes: u32,
    effective_inbox_capacity: u32,
    latency_ticks: u32,
    drop_prob: f32,
    max_broadcast_recipients: u32,
    strict_determinism: bool,
    seed: u64,
    allocator: std.mem.Allocator,

    // Step-level metrics
    total_messages_delivered: u32 = 0,
    total_bytes_sent: u32 = 0,
    total_messages_dropped: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, config: SwarmConfig) !MessageBus {
        const num_agents = config.num_agents;
        const max_mps = config.max_messages_per_step;
        const inbox_cap = if (config.max_inbox_per_agent > 0)
            @max(config.max_inbox_per_agent, max_mps)
        else
            max_mps;

        const outbox_total = num_agents * max_mps;
        const inbox_total = num_agents * inbox_cap;

        const outbox = try allocator.alloc(MessageSlot, outbox_total);
        @memset(outbox, std.mem.zeroes(MessageSlot));

        const inbox = try allocator.alloc(MessageSlot, inbox_total);
        @memset(inbox, std.mem.zeroes(MessageSlot));

        const outbox_counts = try allocator.alloc(u32, num_agents);
        @memset(outbox_counts, 0);
        const inbox_counts = try allocator.alloc(u32, num_agents);
        @memset(inbox_counts, 0);
        const bytes_sent_buf = try allocator.alloc(u32, num_agents);
        @memset(bytes_sent_buf, 0);

        // Latency queue: only allocate if latency_ticks > 0
        var pending_queue: ?[]MessageSlot = null;
        var pending_counts: ?[]u32 = null;

        if (config.latency_ticks > 0) {
            const ring_size = config.latency_ticks + 1;
            const pending_total = num_agents * max_mps * ring_size;
            const pq = try allocator.alloc(MessageSlot, pending_total);
            @memset(pq, std.mem.zeroes(MessageSlot));
            pending_queue = pq;

            const pc = try allocator.alloc(u32, num_agents * ring_size);
            @memset(pc, 0);
            pending_counts = pc;
        }

        return .{
            .outbox = outbox,
            .outbox_counts = outbox_counts,
            .inbox = inbox,
            .inbox_counts = inbox_counts,
            .bytes_sent = bytes_sent_buf,
            .pending_queue = pending_queue,
            .pending_counts = pending_counts,
            .num_agents = num_agents,
            .max_messages_per_step = max_mps,
            .max_message_bytes = config.max_message_bytes,
            .effective_inbox_capacity = inbox_cap,
            .latency_ticks = config.latency_ticks,
            .drop_prob = config.drop_prob,
            .max_broadcast_recipients = config.max_broadcast_recipients,
            .strict_determinism = config.strict_determinism,
            .seed = config.seed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MessageBus) void {
        self.allocator.free(self.outbox);
        self.allocator.free(self.outbox_counts);
        self.allocator.free(self.inbox);
        self.allocator.free(self.inbox_counts);
        self.allocator.free(self.bytes_sent);
        if (self.pending_queue) |pq| self.allocator.free(pq);
        if (self.pending_counts) |pc| self.allocator.free(pc);
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

    /// Deliver messages based on neighbor adjacency with latency/dropout/broadcast caps.
    pub fn deliver(self: *MessageBus, graph: *const AdjacencyGraph, step_count: u64) void {
        // Clear inbox counts
        @memset(self.inbox_counts, 0);
        self.total_messages_delivered = 0;
        self.total_bytes_sent = 0;
        self.total_messages_dropped = 0;

        // Init dropout RNG if needed
        var rng_state: u32 = 0;
        if (self.drop_prob > 0) {
            rng_state = if (self.strict_determinism)
                @as(u32, @truncate(self.seed)) +% @as(u32, @truncate(step_count))
            else
                @as(u32, @truncate(step_count)) ^ 0xDEAD;
            if (rng_state == 0) rng_state = 1;
        }

        if (self.latency_ticks == 0) {
            // Fast path: no latency
            self.deliverImmediate(graph, &rng_state);
        } else {
            // Latency path: enqueue into pending, dequeue from ring
            self.deliverWithLatency(graph, &rng_state, step_count);
        }
    }

    fn deliverImmediate(self: *MessageBus, graph: *const AdjacencyGraph, rng_state: *u32) void {
        for (0..self.num_agents) |sender_idx| {
            const sender: u32 = @intCast(sender_idx);
            const msg_count = self.outbox_counts[sender];
            if (msg_count == 0) continue;

            for (0..msg_count) |m| {
                const slot_idx = sender * self.max_messages_per_step + @as(u32, @intCast(m));
                const msg = self.outbox[slot_idx];

                if (msg.receiver_id == MessageSlot.BROADCAST) {
                    const neighbors = graph.getNeighbors(sender);
                    var recipients: u32 = 0;
                    for (neighbors) |neighbor| {
                        if (recipients >= self.max_broadcast_recipients) break;
                        if (self.shouldDrop(rng_state)) continue;
                        self.deliverToAgent(neighbor, &msg);
                        recipients += 1;
                    }
                } else {
                    const neighbors = graph.getNeighbors(sender);
                    for (neighbors) |neighbor| {
                        if (neighbor == msg.receiver_id) {
                            if (!self.shouldDrop(rng_state)) {
                                self.deliverToAgent(msg.receiver_id, &msg);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    fn deliverWithLatency(self: *MessageBus, graph: *const AdjacencyGraph, rng_state: *u32, step_count: u64) void {
        const ring_size = self.latency_ticks + 1;
        const pq = self.pending_queue.?;
        const pc = self.pending_counts.?;

        // Enqueue current outbox messages into the pending slot for (step_count + latency_ticks)
        const enqueue_slot = @as(u32, @intCast((step_count + self.latency_ticks) % ring_size));

        for (0..self.num_agents) |sender_idx| {
            const sender: u32 = @intCast(sender_idx);
            const msg_count = self.outbox_counts[sender];
            if (msg_count == 0) continue;

            for (0..msg_count) |m| {
                const slot_idx = sender * self.max_messages_per_step + @as(u32, @intCast(m));
                const msg = self.outbox[slot_idx];

                // Store in pending queue
                const pending_base = (sender * ring_size + enqueue_slot) * self.max_messages_per_step;
                const pending_idx = sender * ring_size + enqueue_slot;
                const count = pc[pending_idx];
                if (count < self.max_messages_per_step) {
                    pq[pending_base + count] = msg;
                    pc[pending_idx] = count + 1;
                }
            }
        }

        // Dequeue from current step's ring slot
        const dequeue_slot = @as(u32, @intCast(step_count % ring_size));

        for (0..self.num_agents) |sender_idx| {
            const sender: u32 = @intCast(sender_idx);
            const pending_idx = sender * ring_size + dequeue_slot;
            const pending_msg_count = pc[pending_idx];
            if (pending_msg_count == 0) continue;

            const pending_base = (sender * ring_size + dequeue_slot) * self.max_messages_per_step;

            for (0..pending_msg_count) |pm| {
                const msg = pq[pending_base + pm];

                if (msg.receiver_id == MessageSlot.BROADCAST) {
                    const neighbors = graph.getNeighbors(sender);
                    var recipients: u32 = 0;
                    for (neighbors) |neighbor| {
                        if (recipients >= self.max_broadcast_recipients) break;
                        if (self.shouldDrop(rng_state)) continue;
                        self.deliverToAgent(neighbor, &msg);
                        recipients += 1;
                    }
                } else {
                    const neighbors = graph.getNeighbors(sender);
                    for (neighbors) |neighbor| {
                        if (neighbor == msg.receiver_id) {
                            if (!self.shouldDrop(rng_state)) {
                                self.deliverToAgent(msg.receiver_id, &msg);
                            }
                            break;
                        }
                    }
                }
            }

            // Clear this ring slot
            pc[pending_idx] = 0;
        }
    }

    fn shouldDrop(self: *MessageBus, rng_state: *u32) bool {
        if (self.drop_prob <= 0) return false;
        if (self.drop_prob >= 1.0) {
            self.total_messages_dropped += 1;
            return true;
        }
        const r = xorshiftFloat(rng_state);
        if (r < self.drop_prob) {
            self.total_messages_dropped += 1;
            return true;
        }
        return false;
    }

    fn deliverToAgent(self: *MessageBus, receiver: u32, msg: *const MessageSlot) void {
        if (receiver >= self.num_agents) return;

        const count = self.inbox_counts[receiver];
        if (count >= self.effective_inbox_capacity) return;

        const inbox_idx = receiver * self.effective_inbox_capacity + count;
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
        self.total_messages_dropped = 0;
    }

    /// Get inbox messages for a specific agent.
    pub fn getInbox(self: *const MessageBus, agent_id: u32) []const MessageSlot {
        if (agent_id >= self.num_agents) return &.{};
        const count = self.inbox_counts[agent_id];
        const start = agent_id * self.effective_inbox_capacity;
        return self.inbox[start .. start + count];
    }
};

test "message bus send and deliver" {
    const allocator = std.testing.allocator;

    const config = SwarmConfig{
        .num_agents = 4,
        .max_messages_per_step = 4,
        .max_message_bytes = 48,
    };
    var bus = try MessageBus.init(allocator, config);
    defer bus.deinit();
    bus.clearStep();

    // Send a message from agent 0 to agent 1
    const payload = "hello";
    const ok = bus.send(0, 1, 1, payload);
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(u32, 1), bus.outbox_counts[0]);
}
