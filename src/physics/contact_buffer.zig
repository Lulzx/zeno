//! Optimized contact buffer with per-environment compaction.
//!
//! Design based on M4 Pro bandwidth benchmarks:
//! - Compact contact struct (64 bytes vs 96 bytes)
//! - Per-environment contact counts for efficient iteration
//! - Support for contact streaming (no over-allocation)

const std = @import("std");
const xpbd = @import("xpbd.zig");
const constants = @import("constants.zig");

/// Compact contact struct for GPU solver.
/// 64 bytes total (4 × float4) - fits in single cache line.
pub const CompactContact = extern struct {
    // --- 16 bytes ---
    /// World position (xyz) + penetration depth (w)
    position_penetration: [4]f32 align(16),

    // --- 16 bytes ---
    /// Contact normal (xyz) + friction coefficient (w)
    normal_friction: [4]f32 align(16),

    // --- 16 bytes ---
    /// Body indices: (body_a, body_b, env_id, flags)
    /// flags: bit 0 = active, bits 1-7 = contact age
    indices: [4]u32 align(16),

    // --- 16 bytes ---
    /// Solver state: (lambda_n, lambda_t1, lambda_t2, restitution)
    solver_state: [4]f32 align(16),

    pub fn init(
        position: [3]f32,
        normal: [3]f32,
        penetration: f32,
        body_a: u32,
        body_b: u32,
        env_id: u32,
        friction: f32,
        restitution: f32,
    ) CompactContact {
        return .{
            .position_penetration = .{ position[0], position[1], position[2], penetration },
            .normal_friction = .{ normal[0], normal[1], normal[2], friction },
            .indices = .{ body_a, body_b, env_id, 1 }, // flags = 1 (active)
            .solver_state = .{ 0, 0, 0, restitution },
        };
    }

    pub fn getPosition(self: *const CompactContact) [3]f32 {
        return .{ self.position_penetration[0], self.position_penetration[1], self.position_penetration[2] };
    }

    pub fn getPenetration(self: *const CompactContact) f32 {
        return self.position_penetration[3];
    }

    pub fn getNormal(self: *const CompactContact) [3]f32 {
        return .{ self.normal_friction[0], self.normal_friction[1], self.normal_friction[2] };
    }

    pub fn getFriction(self: *const CompactContact) f32 {
        return self.normal_friction[3];
    }

    pub fn getBodyA(self: *const CompactContact) u32 {
        return self.indices[0];
    }

    pub fn getBodyB(self: *const CompactContact) u32 {
        return self.indices[1];
    }

    pub fn getEnvId(self: *const CompactContact) u32 {
        return self.indices[2];
    }

    pub fn isActive(self: *const CompactContact) bool {
        return (self.indices[3] & 1) != 0;
    }

    pub fn setInactive(self: *CompactContact) void {
        self.indices[3] &= ~@as(u32, 1);
    }

    pub fn getAge(self: *const CompactContact) u8 {
        return @truncate((self.indices[3] >> 1) & 0x7F);
    }

    pub fn incrementAge(self: *CompactContact) void {
        const age = self.getAge();
        if (age < 127) {
            self.indices[3] = (self.indices[3] & 1) | (@as(u32, age + 1) << 1);
        }
    }

    pub fn getLambdaN(self: *const CompactContact) f32 {
        return self.solver_state[0];
    }

    pub fn setLambdaN(self: *CompactContact, lambda: f32) void {
        self.solver_state[0] = lambda;
    }

    pub fn getRestitution(self: *const CompactContact) f32 {
        return self.solver_state[3];
    }

    /// Convert to XPBD constraint format.
    pub fn toXPBDConstraint(self: *const CompactContact, compliance: f32) xpbd.XPBDConstraint {
        return .{
            .indices = .{
                self.indices[0], // body_a
                self.indices[1], // body_b
                self.indices[2], // env_id
                @intFromEnum(xpbd.ConstraintType.contact_normal),
            },
            .anchor_a = .{ self.position_penetration[0], self.position_penetration[1], self.position_penetration[2], compliance },
            .anchor_b = .{ 0, 0, 0, 0 },
            .axis_target = .{ self.normal_friction[0], self.normal_friction[1], self.normal_friction[2], 0 },
            .limits = .{ 0, std.math.inf(f32), self.normal_friction[3], self.solver_state[3] },
            .state = .{ self.solver_state[0], 0, self.position_penetration[3], 0 },
        };
    }
};

/// Contact buffer manager with compaction support.
pub const ContactBufferManager = struct {
    /// Contact storage (num_envs × max_contacts_per_env)
    contacts: []CompactContact,

    /// Per-environment contact counts
    counts: []u32,

    /// Per-environment contact offsets for compacted iteration
    offsets: []u32,

    /// Configuration
    num_envs: u32,
    max_contacts_per_env: u32,

    /// Statistics
    total_contacts: u32,
    peak_contacts: u32,

    /// Allocator
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_envs: u32, max_contacts_per_env: u32) !ContactBufferManager {
        const total = num_envs * max_contacts_per_env;

        var self = ContactBufferManager{
            .contacts = try allocator.alloc(CompactContact, total),
            .counts = try allocator.alloc(u32, num_envs),
            .offsets = try allocator.alloc(u32, num_envs),
            .num_envs = num_envs,
            .max_contacts_per_env = max_contacts_per_env,
            .total_contacts = 0,
            .peak_contacts = 0,
            .allocator = allocator,
        };

        // Initialize counts to zero
        @memset(self.counts, 0);

        // Initialize offsets (strided layout)
        for (0..num_envs) |i| {
            self.offsets[i] = @intCast(i * max_contacts_per_env);
        }

        return self;
    }

    pub fn deinit(self: *ContactBufferManager) void {
        self.allocator.free(self.contacts);
        self.allocator.free(self.counts);
        self.allocator.free(self.offsets);
    }

    /// Clear all contacts for a frame.
    pub fn clearAll(self: *ContactBufferManager) void {
        @memset(self.counts, 0);
        self.total_contacts = 0;
    }

    /// Clear contacts for a specific environment.
    pub fn clearEnv(self: *ContactBufferManager, env_id: u32) void {
        self.total_contacts -= self.counts[env_id];
        self.counts[env_id] = 0;
    }

    /// Add a contact to an environment.
    /// Returns true if contact was added, false if buffer full.
    pub fn addContact(self: *ContactBufferManager, env_id: u32, contact: CompactContact) bool {
        const count = self.counts[env_id];
        if (count >= self.max_contacts_per_env) {
            return false;
        }

        const idx = self.offsets[env_id] + count;
        self.contacts[idx] = contact;
        self.counts[env_id] = count + 1;
        self.total_contacts += 1;
        self.peak_contacts = @max(self.peak_contacts, self.total_contacts);

        return true;
    }

    /// Get contact for environment.
    pub fn getContact(self: *const ContactBufferManager, env_id: u32, contact_idx: u32) ?*const CompactContact {
        if (contact_idx >= self.counts[env_id]) return null;
        return &self.contacts[self.offsets[env_id] + contact_idx];
    }

    /// Get mutable contact.
    pub fn getContactMut(self: *ContactBufferManager, env_id: u32, contact_idx: u32) ?*CompactContact {
        if (contact_idx >= self.counts[env_id]) return null;
        return &self.contacts[self.offsets[env_id] + contact_idx];
    }

    /// Iterate over contacts for an environment.
    pub fn iterEnvContacts(self: *const ContactBufferManager, env_id: u32) []const CompactContact {
        const offset = self.offsets[env_id];
        const count = self.counts[env_id];
        return self.contacts[offset..][0..count];
    }

    /// Remove inactive contacts (compaction within each environment).
    pub fn compactInPlace(self: *ContactBufferManager) void {
        for (0..self.num_envs) |env_id| {
            self.compactEnv(@intCast(env_id));
        }
    }

    fn compactEnv(self: *ContactBufferManager, env_id: u32) void {
        const offset = self.offsets[env_id];
        var count = self.counts[env_id];

        var write_idx: u32 = 0;
        var read_idx: u32 = 0;

        while (read_idx < count) {
            if (self.contacts[offset + read_idx].isActive()) {
                if (write_idx != read_idx) {
                    self.contacts[offset + write_idx] = self.contacts[offset + read_idx];
                }
                write_idx += 1;
            } else {
                self.total_contacts -= 1;
            }
            read_idx += 1;
        }

        self.counts[env_id] = write_idx;
    }

    /// Get total memory usage.
    pub fn memorySize(self: *const ContactBufferManager) usize {
        return self.contacts.len * @sizeOf(CompactContact) +
            self.counts.len * @sizeOf(u32) +
            self.offsets.len * @sizeOf(u32);
    }

    /// Get buffer utilization.
    pub fn utilization(self: *const ContactBufferManager) f32 {
        const capacity = self.num_envs * self.max_contacts_per_env;
        return @as(f32, @floatFromInt(self.total_contacts)) / @as(f32, @floatFromInt(capacity));
    }

    /// Print statistics.
    pub fn printStats(self: *const ContactBufferManager) void {
        std.debug.print("Contact Buffer Stats:\n", .{});
        std.debug.print("  Total contacts: {}\n", .{self.total_contacts});
        std.debug.print("  Peak contacts: {}\n", .{self.peak_contacts});
        std.debug.print("  Utilization: {d:.1}%\n", .{self.utilization() * 100});
        std.debug.print("  Memory: {} KB\n", .{self.memorySize() / 1024});

        // Distribution
        var min_count: u32 = std.math.maxInt(u32);
        var max_count: u32 = 0;
        var sum: u64 = 0;

        for (self.counts) |c| {
            min_count = @min(min_count, c);
            max_count = @max(max_count, c);
            sum += c;
        }

        const avg = @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(self.num_envs));
        std.debug.print("  Per-env: min={}, max={}, avg={d:.1}\n", .{ min_count, max_count, avg });
    }
};

/// GPU buffer layout info for Metal dispatch.
pub const ContactBufferLayout = struct {
    /// Byte offset to contacts array
    contacts_offset: u32,
    /// Byte offset to counts array
    counts_offset: u32,
    /// Byte size of contacts array
    contacts_size: u32,
    /// Byte size of counts array
    counts_size: u32,
    /// Total buffer size
    total_size: u32,

    pub fn compute(num_envs: u32, max_contacts_per_env: u32) ContactBufferLayout {
        const contacts_size = num_envs * max_contacts_per_env * @sizeOf(CompactContact);
        const counts_size = num_envs * @sizeOf(u32);

        // Align counts to 16 bytes
        const aligned_contacts = (contacts_size + 15) & ~@as(u32, 15);

        return .{
            .contacts_offset = 0,
            .counts_offset = aligned_contacts,
            .contacts_size = contacts_size,
            .counts_size = counts_size,
            .total_size = aligned_contacts + counts_size,
        };
    }
};

// Compile-time verification
comptime {
    if (@sizeOf(CompactContact) != 64) {
        @compileError("CompactContact must be 64 bytes");
    }
}
