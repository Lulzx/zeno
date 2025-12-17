//! Metal command buffer and encoder management.
//! Handles GPU command submission and synchronization.

const std = @import("std");
const objc = @import("../objc.zig");
const Buffer = @import("buffer.zig").Buffer;
const ComputePipeline = @import("pipeline.zig").ComputePipeline;

pub const CommandError = error{
    CommandBufferCreationFailed,
    EncoderCreationFailed,
    ExecutionFailed,
    Timeout,
};

/// Command buffer wrapper for GPU command recording.
pub const CommandBuffer = struct {
    buffer: objc.id,
    queue: objc.id,

    /// Create a new command buffer from the command queue.
    pub fn init(queue: objc.id) CommandError!CommandBuffer {
        const buffer = objc.msgSend(queue, objc.sel("commandBuffer"), .{});
        if (buffer == null) {
            return CommandError.CommandBufferCreationFailed;
        }
        return CommandBuffer{
            .buffer = buffer.?,
            .queue = queue,
        };
    }

    /// Create a new command buffer with unretained references (faster).
    pub fn initUnretained(queue: objc.id) CommandError!CommandBuffer {
        const buffer = objc.msgSend(queue, objc.sel("commandBufferWithUnretainedReferences"), .{});
        if (buffer == null) {
            return CommandError.CommandBufferCreationFailed;
        }
        return CommandBuffer{
            .buffer = buffer.?,
            .queue = queue,
        };
    }

    /// Create a compute command encoder.
    pub fn computeEncoder(self: *CommandBuffer) CommandError!ComputeEncoder {
        const encoder = objc.msgSend(self.buffer, objc.sel("computeCommandEncoder"), .{});
        if (encoder == null) {
            return CommandError.EncoderCreationFailed;
        }
        return ComputeEncoder{ .encoder = encoder.? };
    }

    /// Create a compute encoder with dispatch type.
    pub fn computeEncoderWithDispatchType(self: *CommandBuffer, concurrent: bool) CommandError!ComputeEncoder {
        const dispatch_type: u64 = if (concurrent) 1 else 0; // MTLDispatchTypeConcurrent or Serial
        const encoder = objc.msgSend(
            self.buffer,
            objc.sel("computeCommandEncoderWithDispatchType:"),
            .{dispatch_type},
        );
        if (encoder == null) {
            return CommandError.EncoderCreationFailed;
        }
        return ComputeEncoder{ .encoder = encoder.? };
    }

    /// Commit the command buffer for execution.
    pub fn commit(self: *CommandBuffer) void {
        objc.msgSendVoid(self.buffer, objc.sel("commit"), .{});
    }

    /// Wait for command buffer execution to complete.
    pub fn waitUntilCompleted(self: *CommandBuffer) void {
        objc.msgSendVoid(self.buffer, objc.sel("waitUntilCompleted"), .{});
    }

    /// Wait for the command buffer to be scheduled.
    pub fn waitUntilScheduled(self: *CommandBuffer) void {
        objc.msgSendVoid(self.buffer, objc.sel("waitUntilScheduled"), .{});
    }

    /// Commit and wait for completion.
    pub fn commitAndWait(self: *CommandBuffer) void {
        self.commit();
        self.waitUntilCompleted();
    }

    /// Add a completion handler.
    pub fn addCompletedHandler(self: *CommandBuffer, handler: *const fn (objc.id) void) void {
        const block = createBlock(handler);
        objc.msgSendVoid(self.buffer, objc.sel("addCompletedHandler:"), .{block});
    }

    /// Check command buffer status.
    pub fn getStatus(self: *const CommandBuffer) CommandBufferStatus {
        const status = objc.msgSendInt(u64, self.buffer, objc.sel("status"), .{});
        return @enumFromInt(status);
    }

    /// Get error if command buffer failed.
    pub fn getError(self: *const CommandBuffer) ?[]const u8 {
        const error_obj = objc.msgSend(self.buffer, objc.sel("error"), .{});
        if (error_obj == null) return null;

        const desc = objc.msgSend(error_obj, objc.sel("localizedDescription"), .{});
        return objc.getNSStringContents(desc);
    }

    /// Check if execution completed successfully.
    pub fn succeeded(self: *const CommandBuffer) bool {
        return self.getStatus() == .completed and self.getError() == null;
    }

    /// Present a drawable (for rendering, not typically used for compute).
    pub fn presentDrawable(self: *CommandBuffer, drawable: objc.id) void {
        objc.msgSendVoid(self.buffer, objc.sel("presentDrawable:"), .{drawable});
    }

    /// Encode a wait for an event.
    pub fn encodeWait(self: *CommandBuffer, event: objc.id, value: u64) void {
        objc.msgSendVoid(self.buffer, objc.sel("encodeWaitForEvent:value:"), .{ event, value });
    }

    /// Encode a signal for an event.
    pub fn encodeSignal(self: *CommandBuffer, event: objc.id, value: u64) void {
        objc.msgSendVoid(self.buffer, objc.sel("encodeSignalEvent:value:"), .{ event, value });
    }
};

pub const CommandBufferStatus = enum(u64) {
    not_enqueued = 0,
    enqueued = 1,
    committed = 2,
    scheduled = 3,
    completed = 4,
    failed = 5,
};

/// Compute command encoder for dispatching compute work.
pub const ComputeEncoder = struct {
    encoder: objc.id,

    /// Set the compute pipeline state.
    pub fn setPipeline(self: *ComputeEncoder, pipeline: *const ComputePipeline) void {
        objc.msgSendVoid(self.encoder, objc.sel("setComputePipelineState:"), .{pipeline.pipeline});
    }

    /// Set a buffer at an index.
    pub fn setBuffer(self: *ComputeEncoder, buffer: *const Buffer, offset: u32, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setBuffer:offset:atIndex:"),
            .{ buffer.buffer, @as(u64, offset), @as(u64, index) },
        );
    }

    /// Set a raw Metal buffer at an index.
    pub fn setBufferHandle(self: *ComputeEncoder, buffer: objc.id, offset: u32, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setBuffer:offset:atIndex:"),
            .{ buffer, @as(u64, offset), @as(u64, index) },
        );
    }

    /// Set bytes directly (for small constant data).
    pub fn setBytes(self: *ComputeEncoder, data: []const u8, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setBytes:length:atIndex:"),
            .{ data.ptr, @as(u64, data.len), @as(u64, index) },
        );
    }

    /// Set typed constant data.
    pub fn setConstant(self: *ComputeEncoder, comptime T: type, value: *const T, index: u32) void {
        const bytes = std.mem.asBytes(value);
        self.setBytes(bytes, index);
    }

    /// Set threadgroup memory length.
    pub fn setThreadgroupMemoryLength(self: *ComputeEncoder, length: u32, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setThreadgroupMemoryLength:atIndex:"),
            .{ @as(u64, length), @as(u64, index) },
        );
    }

    /// Set a texture at an index.
    pub fn setTexture(self: *ComputeEncoder, texture: objc.id, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setTexture:atIndex:"),
            .{ texture, @as(u64, index) },
        );
    }

    /// Set a sampler state at an index.
    pub fn setSamplerState(self: *ComputeEncoder, sampler: objc.id, index: u32) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("setSamplerState:atIndex:"),
            .{ sampler, @as(u64, index) },
        );
    }

    /// Dispatch threadgroups with explicit size.
    pub fn dispatchThreadgroups(
        self: *ComputeEncoder,
        threadgroups: objc.MTLSize,
        threads_per_threadgroup: objc.MTLSize,
    ) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            .{ threadgroups, threads_per_threadgroup },
        );
    }

    /// Dispatch with total thread count (Metal 2+).
    pub fn dispatchThreads(
        self: *ComputeEncoder,
        threads: objc.MTLSize,
        threads_per_threadgroup: objc.MTLSize,
    ) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("dispatchThreads:threadsPerThreadgroup:"),
            .{ threads, threads_per_threadgroup },
        );
    }

    /// Dispatch 1D work.
    pub fn dispatch1D(self: *ComputeEncoder, pipeline: *const ComputePipeline, count: u32) void {
        const tg_size = pipeline.optimalThreadgroupSize(count);
        const grid = objc.MTLSize.make1D(count);
        self.dispatchThreads(grid, tg_size);
    }

    /// Dispatch 1D work with explicit threadgroup size.
    pub fn dispatch1DWithThreadgroup(self: *ComputeEncoder, count: u32, threadgroup_size: u32) void {
        const num_groups = (count + threadgroup_size - 1) / threadgroup_size;
        self.dispatchThreadgroups(
            objc.MTLSize.make1D(num_groups),
            objc.MTLSize.make1D(threadgroup_size),
        );
    }

    /// Dispatch 2D work.
    pub fn dispatch2D(
        self: *ComputeEncoder,
        width: u32,
        height: u32,
        threadgroup_width: u32,
        threadgroup_height: u32,
    ) void {
        const grid = objc.MTLSize.make(width, height, 1);
        const tg_size = objc.MTLSize.make(threadgroup_width, threadgroup_height, 1);
        self.dispatchThreads(grid, tg_size);
    }

    /// Insert a memory barrier.
    pub fn memoryBarrier(self: *ComputeEncoder, scope: MemoryBarrierScope) void {
        objc.msgSendVoid(self.encoder, objc.sel("memoryBarrierWithScope:"), .{@intFromEnum(scope)});
    }

    /// Insert a barrier for specific buffers.
    pub fn memoryBarrierWithResources(self: *ComputeEncoder, buffers: []const *const Buffer) void {
        var handles: [32]objc.id = undefined;
        const count = @min(buffers.len, 32);
        for (buffers[0..count], 0..) |buf, i| {
            handles[i] = buf.buffer;
        }
        objc.msgSendVoid(
            self.encoder,
            objc.sel("memoryBarrierWithResources:count:"),
            .{ &handles, @as(u64, count) },
        );
    }

    /// Use a resource for read/write access.
    pub fn useResource(self: *ComputeEncoder, resource: objc.id, usage: ResourceUsage) void {
        objc.msgSendVoid(
            self.encoder,
            objc.sel("useResource:usage:"),
            .{ resource, @as(u64, @intFromEnum(usage)) },
        );
    }

    /// Push a debug group.
    pub fn pushDebugGroup(self: *ComputeEncoder, label: []const u8) void {
        const ns_label = objc.createNSString(label);
        defer objc.release(ns_label);
        objc.msgSendVoid(self.encoder, objc.sel("pushDebugGroup:"), .{ns_label});
    }

    /// Pop a debug group.
    pub fn popDebugGroup(self: *ComputeEncoder) void {
        objc.msgSendVoid(self.encoder, objc.sel("popDebugGroup"), .{});
    }

    /// End encoding.
    pub fn endEncoding(self: *ComputeEncoder) void {
        objc.msgSendVoid(self.encoder, objc.sel("endEncoding"), .{});
    }
};

pub const MemoryBarrierScope = enum(u64) {
    buffers = 1,
    textures = 2,
    render_targets = 4,
};

pub const ResourceUsage = enum(u64) {
    read = 1,
    write = 2,
    sample = 4,
};

/// Shared event for synchronization between command buffers.
pub const SharedEvent = struct {
    event: objc.id,

    pub fn init(device: objc.id) SharedEvent {
        const event = objc.msgSend(device, objc.sel("newSharedEvent"), .{});
        return .{ .event = event.? };
    }

    pub fn getValue(self: *const SharedEvent) u64 {
        return objc.msgSendInt(u64, self.event, objc.sel("signaledValue"), .{});
    }

    pub fn getHandle(self: *const SharedEvent) objc.id {
        return self.event;
    }

    pub fn deinit(self: *SharedEvent) void {
        objc.release(self.event);
    }
};

/// Fence for synchronization within an encoder.
pub const Fence = struct {
    fence: objc.id,

    pub fn init(device: objc.id) Fence {
        const fence = objc.msgSend(device, objc.sel("newFence"), .{});
        return .{ .fence = fence.? };
    }

    pub fn getHandle(self: *const Fence) objc.id {
        return self.fence;
    }

    pub fn deinit(self: *Fence) void {
        objc.release(self.fence);
    }
};

/// Helper to create a simple block for callbacks.
fn createBlock(comptime handler: *const fn (objc.id) void) objc.id {
    // Simplified block creation - in production would need full block runtime support
    _ = handler;
    return null;
}

/// Command queue with multiple in-flight buffers.
pub const CommandQueue = struct {
    queue: objc.id,
    device: objc.id,
    in_flight_semaphore: std.Thread.Semaphore,
    max_in_flight: u32,

    pub fn init(device: objc.id, max_in_flight: u32) CommandQueue {
        const queue = objc.msgSend(device, objc.sel("newCommandQueue"), .{});
        return .{
            .queue = queue.?,
            .device = device,
            .in_flight_semaphore = std.Thread.Semaphore{},
            .max_in_flight = max_in_flight,
        };
    }

    pub fn initWithMaxBuffers(device: objc.id, max_buffers: u32) CommandQueue {
        const queue = objc.msgSend(
            device,
            objc.sel("newCommandQueueWithMaxCommandBufferCount:"),
            .{@as(u64, max_buffers)},
        );
        return .{
            .queue = queue.?,
            .device = device,
            .in_flight_semaphore = std.Thread.Semaphore{},
            .max_in_flight = max_buffers,
        };
    }

    pub fn createCommandBuffer(self: *CommandQueue) CommandError!CommandBuffer {
        return CommandBuffer.init(self.queue);
    }

    pub fn getHandle(self: *const CommandQueue) objc.id {
        return self.queue;
    }

    pub fn deinit(self: *CommandQueue) void {
        objc.release(self.queue);
    }
};
