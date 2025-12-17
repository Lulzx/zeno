//! Metal compute pipeline management for shader execution.
//! Handles compute pipeline state creation and configuration.

const std = @import("std");
const objc = @import("../objc.zig");
const Buffer = @import("buffer.zig").Buffer;

pub const PipelineError = error{
    FunctionNotFound,
    PipelineCreationFailed,
    InvalidThreadgroupSize,
    OutOfMemory,
};

/// Compute pipeline state wrapper.
pub const ComputePipeline = struct {
    pipeline: objc.id,
    function: objc.id,
    max_total_threads_per_threadgroup: u32,
    threadgroup_memory_length: u32,
    static_threadgroup_memory_length: u32,

    /// Create a compute pipeline from a library function.
    pub fn init(device: objc.id, library: objc.id, function_name: []const u8) PipelineError!ComputePipeline {
        const ns_name = objc.createNSString(function_name);
        defer objc.release(ns_name);

        const function = objc.msgSend(library, objc.sel("newFunctionWithName:"), .{ns_name});
        if (function == null) {
            return PipelineError.FunctionNotFound;
        }

        var error_ptr: objc.id = null;
        const pipeline = objc.msgSend(
            device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ function.?, &error_ptr },
        );

        if (pipeline == null) {
            if (error_ptr != null) {
                const desc = objc.msgSend(error_ptr, objc.sel("localizedDescription"), .{});
                const cstr = objc.getNSStringContents(desc);
                if (cstr) |c| {
                    std.log.err("Pipeline error: {s}", .{c});
                }
            }
            objc.release(function.?);
            return PipelineError.PipelineCreationFailed;
        }

        const max_threads = objc.msgSendInt(u32, pipeline.?, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
        const tg_mem = objc.msgSendInt(u32, pipeline.?, objc.sel("threadExecutionWidth"), .{});

        return ComputePipeline{
            .pipeline = pipeline.?,
            .function = function.?,
            .max_total_threads_per_threadgroup = max_threads,
            .threadgroup_memory_length = 0,
            .static_threadgroup_memory_length = tg_mem,
        };
    }

    /// Create a compute pipeline with reflection.
    pub fn initWithReflection(
        device: objc.id,
        library: objc.id,
        function_name: []const u8,
        constants: ?objc.id,
    ) PipelineError!ComputePipeline {
        const ns_name = objc.createNSString(function_name);
        defer objc.release(ns_name);

        var function: objc.id = null;
        if (constants != null) {
            var error_ptr: objc.id = null;
            function = objc.msgSend(
                library,
                objc.sel("newFunctionWithName:constantValues:error:"),
                .{ ns_name, constants.?, &error_ptr },
            );
        } else {
            function = objc.msgSend(library, objc.sel("newFunctionWithName:"), .{ns_name});
        }

        if (function == null) {
            return PipelineError.FunctionNotFound;
        }

        // Create descriptor for more control
        const descriptor_class = objc.getClass("MTLComputePipelineDescriptor");
        const descriptor = objc.msgSend(descriptor_class, objc.sel("new"), .{});
        defer objc.release(descriptor);

        objc.msgSendVoid(descriptor, objc.sel("setComputeFunction:"), .{function.?});

        var error_ptr: objc.id = null;
        var reflection: objc.id = null;
        const options: u64 = 0; // MTLPipelineOptionNone

        const pipeline = objc.msgSend(
            device,
            objc.sel("newComputePipelineStateWithDescriptor:options:reflection:error:"),
            .{ descriptor, options, &reflection, &error_ptr },
        );

        if (pipeline == null) {
            objc.release(function.?);
            return PipelineError.PipelineCreationFailed;
        }

        const max_threads = objc.msgSendInt(u32, pipeline.?, objc.sel("maxTotalThreadsPerThreadgroup"), .{});

        return ComputePipeline{
            .pipeline = pipeline.?,
            .function = function.?,
            .max_total_threads_per_threadgroup = max_threads,
            .threadgroup_memory_length = 0,
            .static_threadgroup_memory_length = 0,
        };
    }

    /// Get the optimal threadgroup size for a given grid size.
    pub fn optimalThreadgroupSize(self: *const ComputePipeline, grid_size: u32) objc.MTLSize {
        const max = self.max_total_threads_per_threadgroup;
        const width = @min(grid_size, max);
        return objc.MTLSize.make1D(width);
    }

    /// Calculate grid size (number of threadgroups) for a given work item count.
    pub fn gridSize(self: *const ComputePipeline, work_items: u32) objc.MTLSize {
        const tg_size = self.optimalThreadgroupSize(work_items);
        const grid_width = (work_items + @as(u32, @intCast(tg_size.width)) - 1) / @as(u32, @intCast(tg_size.width));
        return objc.MTLSize.make1D(grid_width);
    }

    /// Get thread execution width (SIMD width).
    pub fn threadExecutionWidth(self: *const ComputePipeline) u32 {
        return objc.msgSendInt(u32, self.pipeline, objc.sel("threadExecutionWidth"), .{});
    }

    pub fn deinit(self: *ComputePipeline) void {
        objc.release(self.function);
        objc.release(self.pipeline);
    }
};

/// Manager for multiple compute pipelines.
pub const PipelineManager = struct {
    device: objc.id,
    library: objc.id,
    pipelines: std.StringHashMap(ComputePipeline),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, device: objc.id, library: objc.id) PipelineManager {
        return .{
            .device = device,
            .library = library,
            .pipelines = std.StringHashMap(ComputePipeline).init(allocator),
            .allocator = allocator,
        };
    }

    /// Get or create a pipeline for the given function.
    pub fn getPipeline(self: *PipelineManager, function_name: []const u8) PipelineError!*ComputePipeline {
        if (self.pipelines.getPtr(function_name)) |pipeline| {
            return pipeline;
        }

        const pipeline = try ComputePipeline.init(self.device, self.library, function_name);
        const name_copy = try self.allocator.dupe(u8, function_name);
        try self.pipelines.put(name_copy, pipeline);

        return self.pipelines.getPtr(function_name).?;
    }

    /// Preload multiple pipelines.
    pub fn preload(self: *PipelineManager, function_names: []const []const u8) !void {
        for (function_names) |name| {
            _ = try self.getPipeline(name);
        }
    }

    pub fn deinit(self: *PipelineManager) void {
        var iter = self.pipelines.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.pipelines.deinit();
    }
};

/// Function constants for specialization.
pub const FunctionConstants = struct {
    constants: objc.id,

    pub fn init() FunctionConstants {
        const MTLFunctionConstantValues = objc.getClass("MTLFunctionConstantValues");
        const constants = objc.msgSend(MTLFunctionConstantValues, objc.sel("new"), .{});
        return .{ .constants = constants.? };
    }

    pub fn setBool(self: *FunctionConstants, value: bool, index: u32) void {
        const bool_value: u8 = if (value) 1 else 0;
        objc.msgSendVoid(
            self.constants,
            objc.sel("setConstantValue:type:atIndex:"),
            .{ &bool_value, @as(u64, 0), @as(u64, index) }, // MTLDataTypeBool = 0
        );
    }

    pub fn setInt(self: *FunctionConstants, value: i32, index: u32) void {
        objc.msgSendVoid(
            self.constants,
            objc.sel("setConstantValue:type:atIndex:"),
            .{ &value, @as(u64, 4), @as(u64, index) }, // MTLDataTypeInt = 4
        );
    }

    pub fn setUInt(self: *FunctionConstants, value: u32, index: u32) void {
        objc.msgSendVoid(
            self.constants,
            objc.sel("setConstantValue:type:atIndex:"),
            .{ &value, @as(u64, 5), @as(u64, index) }, // MTLDataTypeUInt = 5
        );
    }

    pub fn setFloat(self: *FunctionConstants, value: f32, index: u32) void {
        objc.msgSendVoid(
            self.constants,
            objc.sel("setConstantValue:type:atIndex:"),
            .{ &value, @as(u64, 3), @as(u64, index) }, // MTLDataTypeFloat = 3
        );
    }

    pub fn getHandle(self: *const FunctionConstants) objc.id {
        return self.constants;
    }

    pub fn deinit(self: *FunctionConstants) void {
        objc.release(self.constants);
    }
};
