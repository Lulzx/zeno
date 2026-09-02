//! Compare compact GPU masked stepping with the legacy host memcpy strategy.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const repeats = 5;
const warmup_steps = 10;
const measured_steps = 200;

const Variant = enum { compact_gpu, dense_gpu, legacy_cpu_memcpy };

const Result = struct {
    variant: Variant,
    num_envs: u32,
    active_envs: u32,
    active_stride: u32,
    elapsed_ms: f64,
    batch_steps_per_second: f64,
    active_steps_per_second: f64,
};

const LegacyCpuBackup = struct {
    const Entry = struct {
        buffer: *zeno.metal.buffer.Buffer,
        bytes: []u8,
    };

    entries: [13]Entry,
    allocator: std.mem.Allocator,
    num_envs: usize,

    fn init(allocator: std.mem.Allocator, world: *zeno.World) !LegacyCpuBackup {
        const buffers = [_]*zeno.metal.buffer.Buffer{
            &world.state.positions_buffer,
            &world.state.quaternions_buffer,
            &world.state.velocities_buffer,
            &world.state.accelerations_buffer,
            &world.state.angular_velocities_buffer,
            &world.state.joint_positions_buffer,
            &world.state.joint_velocities_buffer,
            &world.state.joint_torques_buffer,
            &world.state.observations_buffer,
            &world.state.rewards_buffer,
            &world.state.dones_buffer,
            &world.state.contact_counts_buffer,
            &world.state.contacts_buffer,
        };

        var result: LegacyCpuBackup = .{
            .entries = undefined,
            .allocator = allocator,
            .num_envs = world.config.num_envs,
        };
        var initialized: usize = 0;
        errdefer for (result.entries[0..initialized]) |entry| allocator.free(entry.bytes);

        for (buffers, 0..) |buffer, i| {
            result.entries[i] = .{
                .buffer = buffer,
                .bytes = try allocator.alloc(u8, buffer.size),
            };
            initialized += 1;
        }
        return result;
    }

    fn deinit(self: *LegacyCpuBackup) void {
        for (self.entries) |entry| self.allocator.free(entry.bytes);
    }

    fn step(self: *LegacyCpuBackup, world: *zeno.World, actions: []const f32, mask: []const u8) !void {
        for (self.entries) |entry| {
            @memcpy(entry.bytes, entry.buffer.getSlice(u8));
        }

        try world.step(actions, 1);

        for (mask, 0..) |active, env_id| {
            if (active != 0) continue;
            for (self.entries) |entry| {
                const live = entry.buffer.getSlice(u8);
                const bytes_per_env = live.len / self.num_envs;
                const start = env_id * bytes_per_env;
                @memcpy(live[start .. start + bytes_per_env], entry.bytes[start .. start + bytes_per_env]);
            }
        }
    }
};

fn runOnce(allocator: std.mem.Allocator, num_envs: u32, active_stride: u32, variant: Variant) !Result {
    const scene = try zeno.mjcf.parser.parseFile(allocator, "assets/ant.xml");
    var world = try zeno.World.init(allocator, scene, .{
        .num_envs = num_envs,
        .timestep = 0.002,
        .contact_iterations = 4,
        .max_contacts_per_env = 64,
        .substeps = 1,
    });
    defer world.deinit();

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, num_envs * info.action_dim);
    defer allocator.free(actions);
    @memset(actions, 0.1);

    const mask = try allocator.alloc(u8, num_envs);
    defer allocator.free(mask);
    var active_envs: u32 = 0;
    for (mask, 0..) |*value, env_id| {
        value.* = @intFromBool(env_id % active_stride == 0);
        active_envs += value.*;
    }

    var legacy = try LegacyCpuBackup.init(allocator, &world);
    defer legacy.deinit();

    for (0..warmup_steps) |_| {
        switch (variant) {
            .compact_gpu => try world.stepSubset(actions, mask, 1),
            .dense_gpu => try world.step(actions, 1),
            .legacy_cpu_memcpy => try legacy.step(&world, actions, mask),
        }
    }

    var timer = try zeno.Timer.start();
    for (0..measured_steps) |_| {
        switch (variant) {
            .compact_gpu => try world.stepSubset(actions, mask, 1),
            .dense_gpu => try world.step(actions, 1),
            .legacy_cpu_memcpy => try legacy.step(&world, actions, mask),
        }
    }

    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    const elapsed_s = elapsed_ms / 1000.0;
    return .{
        .variant = variant,
        .num_envs = num_envs,
        .active_envs = active_envs,
        .active_stride = active_stride,
        .elapsed_ms = elapsed_ms,
        .batch_steps_per_second = @as(f64, @floatFromInt(num_envs * measured_steps)) / elapsed_s,
        .active_steps_per_second = @as(f64, @floatFromInt(active_envs * measured_steps)) / elapsed_s,
    };
}

fn medianResult(allocator: std.mem.Allocator, num_envs: u32, active_stride: u32, variant: Variant) !Result {
    var samples: [repeats]Result = undefined;
    for (&samples) |*sample| sample.* = try runOnce(allocator, num_envs, active_stride, variant);
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

    std.debug.print("\nZeno compact masked-step benchmark\n", .{});
    std.debug.print("Device: {s}; unified memory: {}; Zig: {s}\n", .{ device.getName(), device.hasUnifiedMemory(), builtin.zig_version_string });
    std.debug.print("Workload: Ant, one fixed substep, {d} warm-up, {d} measured steps, {d} repeats, median\n", .{ warmup_steps, measured_steps, repeats });
    std.debug.print("Compact GPU dispatch preserves all inactive state by omission; dense GPU measures an unmasked full step; legacy CPU computes the full batch and restores 13 public buffers.\n\n", .{});

    const cases = [_]struct { num_envs: u32, active_stride: u32 }{
        .{ .num_envs = 1024, .active_stride = 2 },
        .{ .num_envs = 4096, .active_stride = 2 },
        .{ .num_envs = 4096, .active_stride = 4 },
        .{ .num_envs = 4096, .active_stride = 10 },
    };
    for (cases) |case| {
        const gpu = try medianResult(allocator, case.num_envs, case.active_stride, .compact_gpu);
        const dense = try medianResult(allocator, case.num_envs, case.active_stride, .dense_gpu);
        const cpu = try medianResult(allocator, case.num_envs, case.active_stride, .legacy_cpu_memcpy);
        const dense_speedup = dense.elapsed_ms / gpu.elapsed_ms;
        const legacy_speedup = cpu.elapsed_ms / gpu.elapsed_ms;

        std.debug.print("{d} environments ({d} active, 1/{d}):\n", .{ case.num_envs, gpu.active_envs, case.active_stride });
        std.debug.print("  Compact GPU: {d:.1} ms, {d:.0} batch-equivalent env-steps/s, {d:.0} active env-steps/s\n", .{ gpu.elapsed_ms, gpu.batch_steps_per_second, gpu.active_steps_per_second });
        std.debug.print("  Dense GPU:   {d:.1} ms, {d:.0} env-steps/s\n", .{ dense.elapsed_ms, dense.batch_steps_per_second });
        std.debug.print("  CPU memcpy:  {d:.1} ms, {d:.0} batch env-steps/s, {d:.0} active env-steps/s\n", .{ cpu.elapsed_ms, cpu.batch_steps_per_second, cpu.active_steps_per_second });
        std.debug.print("  Compact speedup: {d:.2}x vs dense GPU; {d:.2}x vs CPU memcpy\n\n", .{ dense_speedup, legacy_speedup });
    }
}
