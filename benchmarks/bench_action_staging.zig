//! Measure the host copy removed by shared action-buffer submission.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const repeats = 5;
const warmup_steps = 10;
const measured_steps = 500;
const Variant = enum { copied, shared };
const Result = struct { elapsed_ms: f64, throughput: f64, checksum: f64 };

fn runOnce(allocator: std.mem.Allocator, num_envs: u32, variant: Variant) !Result {
    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = num_envs,
        .timestep = 0.002,
        .contact_iterations = 4,
        .max_contacts_per_env = 64,
        .substeps = 1,
    });
    defer world.deinit();
    const actions = try allocator.alloc(f32, num_envs * world.params.num_actuators);
    defer allocator.free(actions);
    @memset(actions, 0.05);
    @memcpy(world.state.getActions(), actions);

    for (0..warmup_steps) |_| try world.step(actions, 1);
    var timer = try zeno.Timer.start();
    for (0..measured_steps) |_| switch (variant) {
        .copied => try world.step(actions, 1),
        .shared => {
            try world.stepCurrentActionsAsync(1);
            try world.waitStep();
        },
    };
    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    var checksum: f64 = 0;
    for (world.state.getPositions()) |position| checksum += position[0] + position[1] + position[2];
    return .{
        .elapsed_ms = elapsed_ms,
        .throughput = @as(f64, @floatFromInt(num_envs * measured_steps)) / (elapsed_ms / 1000.0),
        .checksum = checksum,
    };
}

fn median(samples_input: [repeats]Result) Result {
    var samples = samples_input;
    for (1..samples.len) |i| {
        var j = i;
        while (j > 0 and samples[j - 1].elapsed_ms > samples[j].elapsed_ms) : (j -= 1) {
            std.mem.swap(Result, &samples[j - 1], &samples[j]);
        }
    }
    return samples[samples.len / 2];
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var device = try zeno.metal.device.Device.init(allocator);
    defer device.deinit();
    std.debug.print("\nZeno unified-memory action staging benchmark\n", .{});
    std.debug.print("Device: {s}; unified memory: {}; Zig: {s}\n", .{ device.getName(), device.hasUnifiedMemory(), builtin.zig_version_string });
    std.debug.print("Workload: Ant, fixed actions, one substep, {d} warm-up, {d} measured, {d} interleaved fresh-world repeats\n\n", .{ warmup_steps, measured_steps, repeats });

    for ([_]u32{ 1024, 4096 }) |num_envs| {
        var copied_samples: [repeats]Result = undefined;
        var shared_samples: [repeats]Result = undefined;
        for (0..repeats) |repeat| {
            if (repeat % 2 == 0) {
                copied_samples[repeat] = try runOnce(allocator, num_envs, .copied);
                shared_samples[repeat] = try runOnce(allocator, num_envs, .shared);
            } else {
                shared_samples[repeat] = try runOnce(allocator, num_envs, .shared);
                copied_samples[repeat] = try runOnce(allocator, num_envs, .copied);
            }
        }
        const copied = median(copied_samples);
        const shared = median(shared_samples);
        const delta = @abs(copied.checksum - shared.checksum);
        const scale = @max(@abs(copied.checksum), @abs(shared.checksum), 1.0);
        std.debug.print("{d} environments:\n", .{num_envs});
        std.debug.print("  Copied actions: {d:.1} ms, {d:.0} env-steps/s\n", .{ copied.elapsed_ms, copied.throughput });
        std.debug.print("  Shared actions: {d:.1} ms, {d:.0} env-steps/s\n", .{ shared.elapsed_ms, shared.throughput });
        std.debug.print("  Ratio:          {d:.2}x\n", .{copied.elapsed_ms / shared.elapsed_ms});
        std.debug.print("  Fresh-world checksum delta: {d:.6} ({d:.5}%)\n\n", .{ delta, delta / scale * 100.0 });
    }
}
