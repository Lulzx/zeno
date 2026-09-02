//! Compare separate reset+step submissions with fused Gymnasium autoreset.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const repeats = 5;
const warmup_steps = 10;
const measured_steps = 200;

const Variant = enum { separate, fused };

const Result = struct {
    elapsed_ms: f64,
    env_steps_per_second: f64,
    checksum: f64,
};

fn runOnce(allocator: std.mem.Allocator, num_envs: u32, reset_stride: u32, variant: Variant) !Result {
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

    const reset_mask = try allocator.alloc(u8, num_envs);
    defer allocator.free(reset_mask);
    for (reset_mask, 0..) |*reset, env_id| reset.* = @intFromBool(env_id % reset_stride == 0);

    for (0..warmup_steps) |_| try world.step(actions, 1);

    var timer = try zeno.Timer.start();
    for (0..measured_steps) |_| {
        switch (variant) {
            .separate => {
                try world.reset(reset_mask);
                try world.step(actions, 1);
            },
            .fused => {
                try world.stepWithResetAsync(actions, reset_mask, 1);
                try world.waitStep();
            },
        }
    }
    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    var checksum: f64 = 0;
    for (world.state.getPositions()) |position| checksum += position[0] + position[1] + position[2];
    return .{
        .elapsed_ms = elapsed_ms,
        .env_steps_per_second = @as(f64, @floatFromInt(num_envs * measured_steps)) / (elapsed_ms / 1000.0),
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
    std.debug.print("\nZeno fused autoreset benchmark\n", .{});
    std.debug.print("Device: {s}; unified memory: {}; Zig: {s}\n", .{ device.getName(), device.hasUnifiedMemory(), builtin.zig_version_string });
    std.debug.print("Workload: Ant, one fixed substep, {d} warm-up, {d} measured, {d} interleaved fresh-world repeats\n\n", .{ warmup_steps, measured_steps, repeats });

    const cases = [_]struct { num_envs: u32, reset_stride: u32 }{
        .{ .num_envs = 1024, .reset_stride = 2 },
        .{ .num_envs = 4096, .reset_stride = 2 },
        .{ .num_envs = 4096, .reset_stride = 10 },
    };
    for (cases) |case| {
        var separate_samples: [repeats]Result = undefined;
        var fused_samples: [repeats]Result = undefined;
        for (0..repeats) |repeat| {
            if (repeat % 2 == 0) {
                separate_samples[repeat] = try runOnce(allocator, case.num_envs, case.reset_stride, .separate);
                fused_samples[repeat] = try runOnce(allocator, case.num_envs, case.reset_stride, .fused);
            } else {
                fused_samples[repeat] = try runOnce(allocator, case.num_envs, case.reset_stride, .fused);
                separate_samples[repeat] = try runOnce(allocator, case.num_envs, case.reset_stride, .separate);
            }
        }
        const separate = median(separate_samples);
        const fused = median(fused_samples);
        const checksum_delta = @abs(separate.checksum - fused.checksum);
        const checksum_scale = @max(@abs(separate.checksum), @abs(fused.checksum), 1.0);
        std.debug.print("{d} environments ({d:.0}% reset):\n", .{ case.num_envs, 100.0 / @as(f64, @floatFromInt(case.reset_stride)) });
        std.debug.print("  Separate reset + step: {d:.1} ms, {d:.0} env-steps/s\n", .{ separate.elapsed_ms, separate.env_steps_per_second });
        std.debug.print("  Fused reset + step:    {d:.1} ms, {d:.0} env-steps/s\n", .{ fused.elapsed_ms, fused.env_steps_per_second });
        std.debug.print("  Fused speedup:         {d:.2}x\n", .{separate.elapsed_ms / fused.elapsed_ms});
        std.debug.print("  Fresh-world checksum delta: {d:.6} ({d:.5}%)\n\n", .{ checksum_delta, checksum_delta / checksum_scale * 100.0 });
    }
}
