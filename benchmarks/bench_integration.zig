//! Benchmark for physics integration performance.

const std = @import("std");
const zeno = @import("zeno");

const Device = zeno.metal.device.Device;
const Buffer = zeno.metal.buffer.Buffer;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n=== Zeno Integration Benchmark ===\n\n", .{});

    // Initialize Metal
    var device = Device.init(allocator) catch {
        std.debug.print("Metal not available\n", .{});
        return;
    };
    defer device.deinit();

    std.debug.print("Device: {s}\n", .{device.getName()});
    std.debug.print("Unified Memory: {}\n\n", .{device.hasUnifiedMemory()});

    // Benchmark buffer allocation
    benchmarkBufferAlloc(&device);

    // Benchmark buffer fill
    benchmarkBufferFill(&device);

    // Benchmark data transfer
    benchmarkDataTransfer(&device);
}

fn benchmarkBufferAlloc(device: *Device) void {
    const sizes = [_]usize{ 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024 };
    const iterations = 100;

    std.debug.print("Buffer Allocation:\n", .{});

    for (sizes) |size| {
        var timer = std.time.Timer.start() catch unreachable;

        for (0..iterations) |_| {
            var buffer = Buffer.init(device.device, size, .{}) catch continue;
            buffer.deinit();
        }

        const elapsed = timer.read();
        const avg_ns = elapsed / iterations;
        const avg_us = @as(f64, @floatFromInt(avg_ns)) / 1000.0;

        std.debug.print("  {d:>8} KB: {d:>8.2} µs\n", .{ size / 1024, avg_us });
    }

    std.debug.print("\n", .{});
}

fn benchmarkBufferFill(device: *Device) void {
    const size = 16 * 1024 * 1024; // 16 MB
    const iterations = 100;

    var buffer = Buffer.init(device.device, size, .{}) catch return;
    defer buffer.deinit();

    std.debug.print("Buffer Fill ({} MB):\n", .{size / (1024 * 1024)});

    var timer = std.time.Timer.start() catch unreachable;

    for (0..iterations) |_| {
        buffer.fill(0) catch {};
    }

    const elapsed = timer.read();
    const avg_ns = elapsed / iterations;
    const throughput = @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(avg_ns)) / 1e9) / 1e9;

    std.debug.print("  Average: {d:.2} µs ({d:.2} GB/s)\n\n", .{
        @as(f64, @floatFromInt(avg_ns)) / 1000.0,
        throughput,
    });
}

fn benchmarkDataTransfer(device: *Device) void {
    const size = 16 * 1024 * 1024; // 16 MB
    const iterations = 100;

    var buffer = Buffer.init(device.device, size, .{}) catch return;
    defer buffer.deinit();

    const data = std.heap.page_allocator.alloc(u8, size) catch return;
    defer std.heap.page_allocator.free(data);

    std.debug.print("Data Write ({} MB):\n", .{size / (1024 * 1024)});

    var timer = std.time.Timer.start() catch unreachable;

    for (0..iterations) |_| {
        buffer.write(data, 0) catch {};
    }

    const elapsed = timer.read();
    const avg_ns = elapsed / iterations;
    const throughput = @as(f64, @floatFromInt(size)) / (@as(f64, @floatFromInt(avg_ns)) / 1e9) / 1e9;

    std.debug.print("  Average: {d:.2} µs ({d:.2} GB/s)\n\n", .{
        @as(f64, @floatFromInt(avg_ns)) / 1000.0,
        throughput,
    });
}
