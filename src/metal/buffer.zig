//! Metal buffer management for unified memory GPU/CPU access.
//! All buffers use storageModeShared for zero-copy data transfer.

const std = @import("std");
const objc = @import("../objc.zig");

pub const BufferError = error{
    AllocationFailed,
    InvalidSize,
    InvalidAlignment,
};

/// Storage mode for Metal buffers.
pub const StorageMode = enum(u64) {
    /// Shared between CPU and GPU (unified memory).
    shared = 0 << 4,
    /// Managed by Metal, requires explicit sync.
    managed = 1 << 4,
    /// Private to GPU only.
    private = 2 << 4,
};

/// Buffer options combining storage mode and CPU cache mode.
pub const BufferOptions = struct {
    storage_mode: StorageMode = .shared,
    cpu_cache_mode_write_combined: bool = false,
    hazard_tracking_untracked: bool = false,

    pub fn toMTLResourceOptions(self: BufferOptions) u64 {
        var options: u64 = @intFromEnum(self.storage_mode);
        if (self.cpu_cache_mode_write_combined) {
            options |= 1; // MTLResourceCPUCacheModeWriteCombined
        }
        if (self.hazard_tracking_untracked) {
            options |= 1 << 8; // MTLResourceHazardTrackingModeUntracked
        }
        return options;
    }
};

/// GPU buffer with unified memory support.
pub const Buffer = struct {
    buffer: objc.id,
    size: usize,
    alignment: usize,

    /// Create a new buffer with specified size and options.
    pub fn init(device: objc.id, size: usize, options: BufferOptions) BufferError!Buffer {
        if (size == 0) {
            return BufferError.InvalidSize;
        }

        const mtl_options = options.toMTLResourceOptions();
        const buffer = objc.msgSend(
            device,
            objc.sel("newBufferWithLength:options:"),
            .{ @as(u64, size), mtl_options },
        );

        if (buffer == null) {
            return BufferError.AllocationFailed;
        }

        return Buffer{
            .buffer = buffer.?,
            .size = size,
            .alignment = 16, // Default Metal alignment
        };
    }

    /// Create a buffer initialized with data.
    pub fn initWithData(device: objc.id, data: []const u8, options: BufferOptions) BufferError!Buffer {
        if (data.len == 0) {
            return BufferError.InvalidSize;
        }

        const mtl_options = options.toMTLResourceOptions();
        const buffer = objc.msgSend(
            device,
            objc.sel("newBufferWithBytes:length:options:"),
            .{ data.ptr, @as(u64, data.len), mtl_options },
        );

        if (buffer == null) {
            return BufferError.AllocationFailed;
        }

        return Buffer{
            .buffer = buffer.?,
            .size = data.len,
            .alignment = 16,
        };
    }

    /// Create an aligned buffer for SIMD operations.
    pub fn initAligned(device: objc.id, size: usize, alignment: usize, options: BufferOptions) BufferError!Buffer {
        if (size == 0) {
            return BufferError.InvalidSize;
        }
        if (alignment == 0 or (alignment & (alignment - 1)) != 0) {
            return BufferError.InvalidAlignment;
        }

        // Round up size to alignment
        const aligned_size = (size + alignment - 1) & ~(alignment - 1);

        var buf = try init(device, aligned_size, options);
        buf.alignment = alignment;
        return buf;
    }

    /// Get raw pointer to buffer contents (CPU accessible for shared mode).
    pub fn contents(self: *const Buffer) ?*anyopaque {
        return objc.msgSend(self.buffer, objc.sel("contents"), .{});
    }

    /// Get typed slice of buffer contents.
    pub fn getSlice(self: *const Buffer, comptime T: type) []T {
        const ptr = self.contents() orelse return &[_]T{};
        const count = self.size / @sizeOf(T);
        return @as([*]T, @ptrCast(@alignCast(ptr)))[0..count];
    }

    /// Get typed aligned slice of buffer contents.
    pub fn getAlignedSlice(self: *const Buffer, comptime T: type, comptime alignment: usize) []align(alignment) T {
        const ptr = self.contents() orelse return &[_]T{};
        const count = self.size / @sizeOf(T);
        const aligned_ptr: [*]align(alignment) T = @ptrCast(@alignCast(ptr));
        return aligned_ptr[0..count];
    }

    /// Copy data to buffer.
    pub fn write(self: *Buffer, data: []const u8, offset: usize) BufferError!void {
        if (offset + data.len > self.size) {
            return BufferError.InvalidSize;
        }

        const ptr = self.contents() orelse return BufferError.AllocationFailed;
        const dest = @as([*]u8, @ptrCast(ptr)) + offset;
        @memcpy(dest[0..data.len], data);
    }

    /// Copy typed data to buffer.
    pub fn writeTyped(self: *Buffer, comptime T: type, data: []const T, offset: usize) BufferError!void {
        const byte_offset = offset * @sizeOf(T);
        const bytes = std.mem.sliceAsBytes(data);
        return self.write(bytes, byte_offset);
    }

    /// Read data from buffer.
    pub fn read(self: *const Buffer, dest: []u8, offset: usize) BufferError!void {
        if (offset + dest.len > self.size) {
            return BufferError.InvalidSize;
        }

        const ptr = self.contents() orelse return BufferError.AllocationFailed;
        const src = @as([*]const u8, @ptrCast(ptr)) + offset;
        @memcpy(dest, src[0..dest.len]);
    }

    /// Fill buffer with a value.
    pub fn fill(self: *Buffer, value: u8) BufferError!void {
        const ptr = self.contents() orelse return BufferError.AllocationFailed;
        const dest = @as([*]u8, @ptrCast(ptr));
        @memset(dest[0..self.size], value);
    }

    /// Fill buffer with zeros.
    pub fn zero(self: *Buffer) BufferError!void {
        return self.fill(0);
    }

    /// Get Metal buffer handle for shader binding.
    pub fn getHandle(self: *const Buffer) objc.id {
        return self.buffer;
    }

    /// For managed storage mode, signal that CPU has finished writing.
    pub fn didModifyRange(self: *Buffer, offset: usize, length: usize) void {
        const range = objc.MTLRegion{
            .origin = .{ .x = offset, .y = 0, .z = 0 },
            .size = .{ .width = length, .height = 1, .depth = 1 },
        };
        _ = range;
        objc.msgSendVoid(
            self.buffer,
            objc.sel("didModifyRange:"),
            .{@as(u64, offset) | (@as(u64, length) << 32)},
        );
    }

    /// Release the buffer.
    pub fn deinit(self: *Buffer) void {
        objc.release(self.buffer);
    }
};

/// Pool of pre-allocated buffers for efficient memory management.
pub const BufferPool = struct {
    device: objc.id,
    buffers: std.ArrayList(Buffer),
    allocator: std.mem.Allocator,
    default_options: BufferOptions,

    pub fn init(allocator: std.mem.Allocator, device: objc.id) BufferPool {
        return .{
            .device = device,
            .buffers = std.ArrayList(Buffer).init(allocator),
            .allocator = allocator,
            .default_options = .{ .storage_mode = .shared },
        };
    }

    /// Allocate a buffer from the pool or create a new one.
    pub fn acquire(self: *BufferPool, size: usize) BufferError!*Buffer {
        // Try to find an existing buffer that fits
        for (self.buffers.items, 0..) |*buf, i| {
            if (buf.size >= size) {
                // Remove from pool and return
                const buffer = self.buffers.orderedRemove(i);
                const new_buf = try self.allocator.create(Buffer);
                new_buf.* = buffer;
                return new_buf;
            }
        }

        // Create new buffer
        const new_buf = try self.allocator.create(Buffer);
        new_buf.* = try Buffer.init(self.device, size, self.default_options);
        return new_buf;
    }

    /// Return a buffer to the pool.
    pub fn release(self: *BufferPool, buffer: *Buffer) void {
        self.buffers.append(buffer.*) catch {
            buffer.deinit();
        };
        self.allocator.destroy(buffer);
    }

    /// Clear all buffers in the pool.
    pub fn clear(self: *BufferPool) void {
        for (self.buffers.items) |*buf| {
            buf.deinit();
        }
        self.buffers.clearRetainingCapacity();
    }

    pub fn deinit(self: *BufferPool) void {
        self.clear();
        self.buffers.deinit();
    }
};

/// Typed buffer wrapper for convenient typed access.
pub fn TypedBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        buffer: Buffer,
        count: usize,

        pub fn init(device: objc.id, count: usize, options: BufferOptions) BufferError!Self {
            const size = count * @sizeOf(T);
            return Self{
                .buffer = try Buffer.init(device, size, options),
                .count = count,
            };
        }

        pub fn initWithData(device: objc.id, data: []const T, options: BufferOptions) BufferError!Self {
            const bytes = std.mem.sliceAsBytes(data);
            return Self{
                .buffer = try Buffer.initWithData(device, bytes, options),
                .count = data.len,
            };
        }

        pub fn getSlice(self: *const Self) []T {
            return self.buffer.getSlice(T);
        }

        pub fn write(self: *Self, data: []const T, offset: usize) BufferError!void {
            return self.buffer.writeTyped(T, data, offset);
        }

        pub fn getHandle(self: *const Self) objc.id {
            return self.buffer.getHandle();
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit();
        }
    };
}

// Common typed buffer aliases
pub const Float4Buffer = TypedBuffer([4]f32);
pub const FloatBuffer = TypedBuffer(f32);
pub const UInt32Buffer = TypedBuffer(u32);
pub const UInt8Buffer = TypedBuffer(u8);
