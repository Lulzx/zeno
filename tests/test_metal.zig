//! Tests for Metal infrastructure.

const std = @import("std");
const testing = std.testing;

// Import from main module
const zeno = @import("zeno");
const Device = zeno.metal.device.Device;
const Buffer = zeno.metal.buffer.Buffer;
const BufferOptions = zeno.metal.buffer.BufferOptions;

test "device initialization" {
    var device = Device.init(testing.allocator) catch |err| {
        std.log.warn("Metal not available: {}", .{err});
        return;
    };
    defer device.deinit();

    try testing.expect(device.device != null);
    try testing.expect(device.command_queue != null);
}

test "device has unified memory" {
    var device = Device.init(testing.allocator) catch return;
    defer device.deinit();

    // Apple Silicon should have unified memory
    try testing.expect(device.hasUnifiedMemory());
}

test "buffer creation" {
    var device = Device.init(testing.allocator) catch return;
    defer device.deinit();

    var buffer = try Buffer.init(device.device, 1024, .{ .storage_mode = .shared });
    defer buffer.deinit();

    try testing.expectEqual(@as(usize, 1024), buffer.size);
    try testing.expect(buffer.contents() != null);
}

test "buffer read write" {
    var device = Device.init(testing.allocator) catch return;
    defer device.deinit();

    var buffer = try Buffer.init(device.device, 256, .{ .storage_mode = .shared });
    defer buffer.deinit();

    // Write data
    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try buffer.write(&data, 0);

    // Read back
    var read_data: [8]u8 = undefined;
    try buffer.read(&read_data, 0);

    try testing.expectEqualSlices(u8, &data, &read_data);
}

test "typed buffer" {
    var device = Device.init(testing.allocator) catch return;
    defer device.deinit();

    const FloatBuffer = zeno.metal.buffer.TypedBuffer(f32);
    var buffer = try FloatBuffer.init(device.device, 100, .{});
    defer buffer.deinit();

    try testing.expectEqual(@as(usize, 100), buffer.count);

    // Write floats
    const floats = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try buffer.write(&floats, 0);

    // Read back
    const slice = buffer.getSlice();
    try testing.expectApproxEqAbs(@as(f32, 1.0), slice[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4.0), slice[3], 0.001);
}

test "buffer zero" {
    var device = Device.init(testing.allocator) catch return;
    defer device.deinit();

    var buffer = try Buffer.init(device.device, 64, .{});
    defer buffer.deinit();

    // Fill with ones
    try buffer.fill(0xFF);

    // Verify filled
    const slice = buffer.getSlice(u8);
    try testing.expectEqual(@as(u8, 0xFF), slice[0]);

    // Zero
    try buffer.zero();

    // Verify zeroed
    try testing.expectEqual(@as(u8, 0), slice[0]);
}
