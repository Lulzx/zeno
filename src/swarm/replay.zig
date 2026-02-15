//! Deterministic replay recording and verification.
//! Records swarm execution traces for debugging and analysis.

const std = @import("std");
const types = @import("types.zig");
const SwarmMetrics = types.SwarmMetrics;
const MessageBus = @import("message_bus.zig").MessageBus;

/// A single recorded frame of swarm state.
pub const ReplayFrame = struct {
    step: u64,
    num_agents: u32,
    positions: [][4]f32,
    velocities: [][4]f32,
    messages_sent: u32,
    messages_delivered: u32,
    metrics: SwarmMetrics,
    checksum: u32,
};

const REPLAY_MAGIC: u32 = 0x5A454E4F; // "ZENO"
const REPLAY_VERSION: u32 = 1;

/// Records swarm frames for replay and determinism verification.
pub const ReplayRecorder = struct {
    frames: std.ArrayList(ReplayFrame),
    recording: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ReplayRecorder {
        return .{
            .frames = .{},
            .recording = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReplayRecorder) void {
        for (self.frames.items) |frame| {
            self.allocator.free(frame.positions);
            self.allocator.free(frame.velocities);
        }
        self.frames.deinit(self.allocator);
    }

    pub fn startRecording(self: *ReplayRecorder) void {
        self.recording = true;
    }

    pub fn stopRecording(self: *ReplayRecorder) void {
        self.recording = false;
    }

    pub fn recordFrame(
        self: *ReplayRecorder,
        step: u64,
        positions: [][4]f32,
        velocities: [][4]f32,
        body_offset: u32,
        num_agents: u32,
        bus: *const MessageBus,
        metrics: SwarmMetrics,
    ) !void {
        if (!self.recording) return;

        // Copy position/velocity data for the agents
        const pos_copy = try self.allocator.alloc([4]f32, num_agents);
        const vel_copy = try self.allocator.alloc([4]f32, num_agents);

        if (num_agents > 0) {
            @memcpy(pos_copy, positions[body_offset .. body_offset + num_agents]);
            @memcpy(vel_copy, velocities[body_offset .. body_offset + num_agents]);
        }

        const checksum = computeCrc32(pos_copy);

        // Sum outbox counts for messages_sent
        var total_sent: u32 = 0;
        for (0..num_agents) |i| {
            total_sent += bus.outbox_counts[i];
        }

        try self.frames.append(self.allocator, .{
            .step = step,
            .num_agents = num_agents,
            .positions = pos_copy,
            .velocities = vel_copy,
            .messages_sent = total_sent,
            .messages_delivered = bus.total_messages_delivered,
            .metrics = metrics,
            .checksum = checksum,
        });
    }

    pub fn getFrame(self: *const ReplayRecorder, step: u64) ?*const ReplayFrame {
        for (self.frames.items) |*frame| {
            if (frame.step == step) return frame;
        }
        return null;
    }

    pub fn frameCount(self: *const ReplayRecorder) usize {
        return self.frames.items.len;
    }

    /// Verify bitwise determinism between two recordings by comparing checksums.
    pub fn verifyDeterminism(self: *const ReplayRecorder, other: *const ReplayRecorder) bool {
        if (self.frames.items.len != other.frames.items.len) return false;
        for (self.frames.items, other.frames.items) |a, b| {
            if (a.checksum != b.checksum) return false;
            if (a.step != b.step) return false;
        }
        return true;
    }

    /// Get replay stats for C ABI.
    pub fn getStats(self: *const ReplayRecorder) types.ReplayStats {
        var total_bytes: u64 = 0;
        for (self.frames.items) |frame| {
            // positions + velocities + overhead
            total_bytes += @as(u64, frame.num_agents) * 32 + 64;
        }
        return .{
            .frame_count = self.frames.items.len,
            .total_bytes = total_bytes,
            .recording = self.recording,
        };
    }

    /// Write all frames to a binary stream.
    pub fn writeTo(self: *const ReplayRecorder, writer: anytype) !void {
        try writer.writeInt(u32, REPLAY_MAGIC, .little);
        try writer.writeInt(u32, REPLAY_VERSION, .little);
        try writer.writeInt(u64, self.frames.items.len, .little);

        for (self.frames.items) |frame| {
            try writer.writeInt(u64, frame.step, .little);
            try writer.writeInt(u32, frame.num_agents, .little);

            // Write positions
            for (frame.positions) |pos| {
                for (pos) |v| {
                    try writer.writeInt(u32, @bitCast(v), .little);
                }
            }
            // Write velocities
            for (frame.velocities) |vel| {
                for (vel) |v| {
                    try writer.writeInt(u32, @bitCast(v), .little);
                }
            }

            try writer.writeInt(u32, frame.messages_sent, .little);
            try writer.writeInt(u32, frame.messages_delivered, .little);
            try writer.writeInt(u32, frame.checksum, .little);
        }
    }

    /// Read frames from a binary stream.
    pub fn readFrom(reader: anytype, allocator: std.mem.Allocator) !ReplayRecorder {
        const magic = try reader.readInt(u32, .little);
        if (magic != REPLAY_MAGIC) return error.InvalidFormat;

        const version = try reader.readInt(u32, .little);
        if (version != REPLAY_VERSION) return error.UnsupportedVersion;

        const num_frames = try reader.readInt(u64, .little);

        var recorder = ReplayRecorder.init(allocator);
        errdefer recorder.deinit();

        for (0..num_frames) |_| {
            const step = try reader.readInt(u64, .little);
            const num_agents = try reader.readInt(u32, .little);

            const pos_copy = try allocator.alloc([4]f32, num_agents);
            errdefer allocator.free(pos_copy);
            for (pos_copy) |*pos| {
                for (pos) |*v| {
                    v.* = @bitCast(try reader.readInt(u32, .little));
                }
            }

            const vel_copy = try allocator.alloc([4]f32, num_agents);
            errdefer allocator.free(vel_copy);
            for (vel_copy) |*vel| {
                for (vel) |*v| {
                    v.* = @bitCast(try reader.readInt(u32, .little));
                }
            }

            const messages_sent = try reader.readInt(u32, .little);
            const messages_delivered = try reader.readInt(u32, .little);
            const checksum = try reader.readInt(u32, .little);

            try recorder.frames.append(allocator, .{
                .step = step,
                .num_agents = num_agents,
                .positions = pos_copy,
                .velocities = vel_copy,
                .messages_sent = messages_sent,
                .messages_delivered = messages_delivered,
                .metrics = .{},
                .checksum = checksum,
            });
        }

        return recorder;
    }
};

/// Compute CRC32 over positions buffer for determinism verification.
pub fn computeCrc32(positions: [][4]f32) u32 {
    const bytes = std.mem.sliceAsBytes(positions);
    return std.hash.Crc32.hash(bytes);
}
