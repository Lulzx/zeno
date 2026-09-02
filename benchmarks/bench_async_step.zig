//! Measure real Metal submission latency and CPU-work overlap.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const repeats = 5;
const warmup_steps = 10;
const measured_steps = 300;
const cpu_work_ns = 250_000;

const Variant = enum { sequential, overlapped };

const Result = struct {
    elapsed_ms: f64,
    env_steps_per_second: f64,
    mean_submit_us: f64,
    checksum: f64,
};

fn cpuWork() void {
    var timer = zeno.Timer.start() catch return;
    while (timer.read() < cpu_work_ns) std.atomic.spinLoopHint();
}

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

    for (0..warmup_steps) |_| try world.step(actions, 1);

    var submit_ns: u64 = 0;
    var total = try zeno.Timer.start();
    for (0..measured_steps) |_| {
        switch (variant) {
            .sequential => {
                try world.step(actions, 1);
                cpuWork();
            },
            .overlapped => {
                var submit = try zeno.Timer.start();
                try world.stepAsync(actions, 1);
                submit_ns += submit.read();
                cpuWork();
                try world.waitStep();
            },
        }
    }
    const elapsed_ms = @as(f64, @floatFromInt(total.read())) / 1_000_000.0;
    var checksum: f64 = 0;
    for (world.state.getPositions()) |position| checksum += position[0] + position[1] + position[2];
    return .{
        .elapsed_ms = elapsed_ms,
        .env_steps_per_second = @as(f64, @floatFromInt(num_envs * measured_steps)) / (elapsed_ms / 1000.0),
        .mean_submit_us = @as(f64, @floatFromInt(submit_ns)) / measured_steps / 1000.0,
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
    std.debug.print("\nZeno asynchronous Metal-step benchmark\n", .{});
    std.debug.print("Device: {s}; unified memory: {}; Zig: {s}\n", .{ device.getName(), device.hasUnifiedMemory(), builtin.zig_version_string });
    std.debug.print("Workload: Ant, one fixed substep, {d} us synthetic CPU work/step, {d} warm-up, {d} measured, {d} interleaved repeats\n\n", .{ cpu_work_ns / 1000, warmup_steps, measured_steps, repeats });

    for ([_]u32{ 1024, 4096 }) |num_envs| {
        var sequential_samples: [repeats]Result = undefined;
        var overlap_samples: [repeats]Result = undefined;
        for (0..repeats) |repeat| {
            if (repeat % 2 == 0) {
                sequential_samples[repeat] = try runOnce(allocator, num_envs, .sequential);
                overlap_samples[repeat] = try runOnce(allocator, num_envs, .overlapped);
            } else {
                overlap_samples[repeat] = try runOnce(allocator, num_envs, .overlapped);
                sequential_samples[repeat] = try runOnce(allocator, num_envs, .sequential);
            }
        }
        const sequential = median(sequential_samples);
        const overlap = median(overlap_samples);
        const checksum_delta = @abs(sequential.checksum - overlap.checksum);
        const checksum_scale = @max(@abs(sequential.checksum), @abs(overlap.checksum), 1.0);
        std.debug.print("{d} environments:\n", .{num_envs});
        std.debug.print("  Sequential physics + CPU: {d:.1} ms, {d:.0} env-steps/s\n", .{ sequential.elapsed_ms, sequential.env_steps_per_second });
        std.debug.print("  Async overlapped:         {d:.1} ms, {d:.0} env-steps/s\n", .{ overlap.elapsed_ms, overlap.env_steps_per_second });
        std.debug.print("  Mean async submission:    {d:.1} us\n", .{overlap.mean_submit_us});
        std.debug.print("  Overlap speedup:           {d:.2}x\n", .{sequential.elapsed_ms / overlap.elapsed_ms});
        std.debug.print("  Fresh-world checksum delta: {d:.6} ({d:.5}%)\n\n", .{ checksum_delta, checksum_delta / checksum_scale * 100.0 });
    }
}
