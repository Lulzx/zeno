//! End-to-end cost of GPU-resident reward/termination versus a host loop.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const repeats = 5;
const warmup_steps = 10;
const measured_steps = 300;

const Variant = enum { physics_only, gpu_task, cpu_task };

const Result = struct {
    elapsed_ms: f64,
    env_steps_per_second: f64,
    checksum: f64,
};

fn computeTaskOnCpu(world: *zeno.World, actions: []const f32, episode_steps: []u32) void {
    const velocities = world.state.getVelocities();
    const rewards = world.state.getRewards();
    const dones = world.state.getDones();
    const num_bodies: usize = @intCast(world.params.num_bodies);
    const num_actuators: usize = @intCast(world.params.num_actuators);
    for (0..world.config.num_envs) |env_id| {
        var control_cost: f32 = 0;
        for (actions[env_id * num_actuators ..][0..num_actuators]) |action| {
            control_cost += action * action;
        }
        rewards[env_id] = velocities[env_id * num_bodies][0] - 0.01 * control_cost + 1.0;
        episode_steps[env_id] += 1;
        dones[env_id] = @intFromBool(episode_steps[env_id] >= 128);
    }
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

    if (variant == .gpu_task) {
        try world.configureTask(.{
            .enabled = 1,
            .root_body = 0,
            .forward_axis = 0,
            .max_episode_steps = 128,
            .forward_reward_weight = 1.0,
            .control_cost_weight = 0.01,
            .healthy_bonus = 1.0,
            .healthy_z_min = -100,
            .healthy_z_max = 100,
            .terminate_when_unhealthy = 1,
        });
    }

    const actions = try allocator.alloc(f32, num_envs * world.params.num_actuators);
    defer allocator.free(actions);
    @memset(actions, 0.1);

    const host_episode_steps = try allocator.alloc(u32, num_envs);
    defer allocator.free(host_episode_steps);
    @memset(host_episode_steps, 0);

    for (0..warmup_steps) |_| {
        try world.step(actions, 1);
        if (variant == .cpu_task) computeTaskOnCpu(&world, actions, host_episode_steps);
    }

    var timer = try zeno.Timer.start();
    for (0..measured_steps) |_| {
        try world.step(actions, 1);
        if (variant == .cpu_task) computeTaskOnCpu(&world, actions, host_episode_steps);
    }
    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

    var checksum: f64 = 0;
    for (world.state.getRewards()) |reward| checksum += reward;
    for (world.state.getDones()) |done| checksum += done;
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

    std.debug.print("\nZeno GPU task-output benchmark\n", .{});
    std.debug.print("Device: {s}; unified memory: {}; Zig: {s}\n", .{ device.getName(), device.hasUnifiedMemory(), builtin.zig_version_string });
    std.debug.print("Workload: Ant, {d} warm-up, {d} measured steps, {d} fresh-world repeats, median\n\n", .{ warmup_steps, measured_steps, repeats });

    for ([_]u32{ 1024, 4096 }) |num_envs| {
        var base_samples: [repeats]Result = undefined;
        var gpu_samples: [repeats]Result = undefined;
        var cpu_samples: [repeats]Result = undefined;
        for (0..repeats) |repeat| {
            // Rotate order to limit thermal/systematic bias between variants.
            switch (repeat % 3) {
                0 => {
                    base_samples[repeat] = try runOnce(allocator, num_envs, .physics_only);
                    gpu_samples[repeat] = try runOnce(allocator, num_envs, .gpu_task);
                    cpu_samples[repeat] = try runOnce(allocator, num_envs, .cpu_task);
                },
                1 => {
                    gpu_samples[repeat] = try runOnce(allocator, num_envs, .gpu_task);
                    cpu_samples[repeat] = try runOnce(allocator, num_envs, .cpu_task);
                    base_samples[repeat] = try runOnce(allocator, num_envs, .physics_only);
                },
                else => {
                    cpu_samples[repeat] = try runOnce(allocator, num_envs, .cpu_task);
                    base_samples[repeat] = try runOnce(allocator, num_envs, .physics_only);
                    gpu_samples[repeat] = try runOnce(allocator, num_envs, .gpu_task);
                },
            }
        }
        const base = median(base_samples);
        const gpu = median(gpu_samples);
        const cpu = median(cpu_samples);
        const checksum_delta = @abs(gpu.checksum - cpu.checksum);
        std.debug.print("{d} environments:\n", .{num_envs});
        std.debug.print("  Physics only: {d:.1} ms, {d:.0} env-steps/s\n", .{ base.elapsed_ms, base.env_steps_per_second });
        std.debug.print("  Metal task:   {d:.1} ms, {d:.0} env-steps/s ({d:.2}% overhead)\n", .{ gpu.elapsed_ms, gpu.env_steps_per_second, (gpu.elapsed_ms / base.elapsed_ms - 1.0) * 100.0 });
        std.debug.print("  CPU task:     {d:.1} ms, {d:.0} env-steps/s ({d:.2}x Metal/CPU throughput)\n", .{ cpu.elapsed_ms, cpu.env_steps_per_second, gpu.env_steps_per_second / cpu.env_steps_per_second });
        std.debug.print("  Final task-output checksum delta: {d:.6}\n\n", .{checksum_delta});
    }
}
