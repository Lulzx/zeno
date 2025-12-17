//! Metal device wrapper for GPU compute operations.
//! Provides safe Zig interface to Metal's MTLDevice and related objects.

const std = @import("std");
const objc = @import("../objc.zig");

pub const DeviceError = error{
    NoDeviceFound,
    LibraryCreationFailed,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    BufferCreationFailed,
    FunctionNotFound,
};

/// Wrapper around MTLDevice providing GPU compute capabilities.
pub const Device = struct {
    device: objc.id,
    command_queue: objc.id,
    library: objc.id,
    allocator: std.mem.Allocator,

    /// Initialize Metal device with default GPU.
    pub fn init(allocator: std.mem.Allocator) DeviceError!Device {
        // Get default Metal device
        const device = objc.createSystemDefaultDevice();
        if (device == null) {
            return DeviceError.NoDeviceFound;
        }

        // Create command queue
        const command_queue = objc.msgSend(device, objc.sel("newCommandQueue"), .{});
        if (command_queue == null) {
            return DeviceError.CommandQueueCreationFailed;
        }

        return Device{
            .device = device.?,
            .command_queue = command_queue.?,
            .library = null,
            .allocator = allocator,
        };
    }

    /// Load Metal shader library from source code.
    pub fn loadLibraryFromSource(self: *Device, source: []const u8) DeviceError!void {
        const ns_source = objc.createNSString(source);
        defer objc.release(ns_source);

        var error_ptr: objc.id = null;
        const options: objc.id = null;

        const library = objc.msgSend(
            self.device,
            objc.sel("newLibraryWithSource:options:error:"),
            .{ ns_source, options, &error_ptr },
        );

        if (library == null) {
            if (error_ptr != null) {
                const desc = objc.msgSend(error_ptr, objc.sel("localizedDescription"), .{});
                const cstr = objc.msgSend(desc, objc.sel("UTF8String"), .{});
                if (cstr) |c| {
                    const cstr_ptr: [*:0]const u8 = @ptrCast(c);
                    std.log.err("Metal library error: {s}", .{cstr_ptr});
                }
            }
            return DeviceError.LibraryCreationFailed;
        }

        if (self.library != null) {
            objc.release(self.library);
        }
        self.library = library.?;
    }

    /// Load Metal shader library from file path.
    pub fn loadLibraryFromFile(self: *Device, path: []const u8) DeviceError!void {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return DeviceError.LibraryCreationFailed;
        };
        defer file.close();

        const source = file.readToEndAlloc(self.allocator, 1024 * 1024) catch {
            return DeviceError.LibraryCreationFailed;
        };
        defer self.allocator.free(source);

        return self.loadLibraryFromSource(source);
    }

    /// Load precompiled Metal library (.metallib).
    pub fn loadLibraryFromMetallib(self: *Device, path: []const u8) DeviceError!void {
        const ns_path = objc.createNSString(path);
        defer objc.release(ns_path);

        var error_ptr: objc.id = null;

        const library = objc.msgSend(
            self.device,
            objc.sel("newLibraryWithFile:error:"),
            .{ ns_path, &error_ptr },
        );

        if (library == null) {
            return DeviceError.LibraryCreationFailed;
        }

        if (self.library != null) {
            objc.release(self.library);
        }
        self.library = library.?;
    }

    /// Get a function from the loaded library.
    pub fn getFunction(self: *Device, name: []const u8) DeviceError!objc.id {
        if (self.library == null) {
            return DeviceError.FunctionNotFound;
        }

        const ns_name = objc.createNSString(name);
        defer objc.release(ns_name);

        const function = objc.msgSend(
            self.library,
            objc.sel("newFunctionWithName:"),
            .{ns_name},
        );

        if (function == null) {
            return DeviceError.FunctionNotFound;
        }

        return function.?;
    }

    /// Create a compute pipeline state from a function.
    pub fn createComputePipeline(self: *Device, function: objc.id) DeviceError!objc.id {
        var error_ptr: objc.id = null;

        const pipeline = objc.msgSend(
            self.device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ function, &error_ptr },
        );

        if (pipeline == null) {
            return DeviceError.PipelineCreationFailed;
        }

        return pipeline.?;
    }

    /// Get the maximum threads per threadgroup for this device.
    pub fn maxThreadsPerThreadgroup(self: *const Device) u32 {
        const size = objc.msgSend(self.device, objc.sel("maxThreadsPerThreadgroup"), .{});
        return @intCast(size.width);
    }

    /// Get device name.
    pub fn getName(self: *const Device) []const u8 {
        const name = objc.msgSend(self.device, objc.sel("name"), .{});
        const cstr = objc.msgSend(name, objc.sel("UTF8String"), .{});
        if (cstr) |c| {
            const cstr_ptr: [*:0]const u8 = @ptrCast(c);
            return std.mem.span(cstr_ptr);
        }
        return "Unknown Device";
    }

    /// Check if device supports unified memory.
    pub fn hasUnifiedMemory(self: *const Device) bool {
        return objc.msgSendBool(self.device, objc.sel("hasUnifiedMemory"), .{});
    }

    /// Clean up Metal resources.
    pub fn deinit(self: *Device) void {
        if (self.library != null) {
            objc.release(self.library);
        }
        objc.release(self.command_queue);
        objc.release(self.device);
    }
};

test "device initialization" {
    const allocator = std.testing.allocator;
    var device = try Device.init(allocator);
    defer device.deinit();

    try std.testing.expect(device.hasUnifiedMemory());
}
