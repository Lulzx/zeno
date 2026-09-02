//! Engine-pipeline benchmark using actual MJCF models and World.step.
//! This measures Zeno throughput only; it is not a semantics-matched
//! comparison with another simulator.

const std = @import("std");
const builtin = @import("builtin");
const zeno = @import("zeno");

const World = zeno.World;
const WorldConfig = zeno.WorldConfig;
const mjcf = zeno.mjcf;

const Allocator = std.mem.Allocator;
const benchmark_repeats = 5;

/// Benchmark configuration
const BenchConfig = struct {
    name: []const u8,
    mjcf_path: []const u8,
    num_envs: u32,
    num_steps: u32,
};

/// Benchmark result
const BenchResult = struct {
    name: []const u8,
    num_envs: u32,
    num_steps: u32,
    num_bodies: u32,
    num_joints: u32,
    num_actuators: u32,
    total_time_ms: f64,
    steps_per_sec: f64,
    memory_mb: f64,
};

const RepeatedBenchResult = struct {
    median: BenchResult,
    min_time_ms: f64,
    max_time_ms: f64,
    samples: [benchmark_repeats]BenchResult,
};

fn runBenchmark(allocator: Allocator, config: BenchConfig) !BenchResult {
    // Parse MJCF file
    const scene = try mjcf.parser.parseFile(allocator, config.mjcf_path);

    // Create world configuration
    const world_config = WorldConfig{
        .num_envs = config.num_envs,
        .timestep = 0.002,
        .contact_iterations = 4,
        .max_contacts_per_env = 64,
        .substeps = 1,
    };

    // Create world
    var world = try World.init(allocator, scene, world_config);
    defer world.deinit();

    const info = world.getInfo();

    // Allocate actions buffer
    const actions = try allocator.alloc(f32, config.num_envs * info.action_dim);
    defer allocator.free(actions);

    // Initialize with random actions
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (actions) |*a| {
        a.* = random.float(f32) * 2.0 - 1.0;
    }

    // Warm-up
    for (0..10) |_| {
        try world.step(actions, 1);
    }
    try world.reset(null);

    // Timed benchmark
    var timer = try zeno.Timer.start();

    for (0..config.num_steps) |_| {
        try world.step(actions, 1);
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_env_steps = @as(f64, @floatFromInt(config.num_envs)) * @as(f64, @floatFromInt(config.num_steps));
    const steps_per_sec = total_env_steps / (elapsed_ms / 1000.0);

    return BenchResult{
        .name = config.name,
        .num_envs = config.num_envs,
        .num_steps = config.num_steps,
        .num_bodies = info.num_bodies,
        .num_joints = info.num_joints,
        .num_actuators = info.num_actuators,
        .total_time_ms = elapsed_ms,
        .steps_per_sec = steps_per_sec,
        .memory_mb = @as(f64, @floatFromInt(info.memory_usage)) / (1024.0 * 1024.0),
    };
}

fn runRepeatedBenchmark(allocator: Allocator, config: BenchConfig) !RepeatedBenchResult {
    var samples: [benchmark_repeats]BenchResult = undefined;
    for (&samples) |*sample| {
        sample.* = try runBenchmark(allocator, config);
    }

    // Small fixed sample: insertion sort keeps this compatible across Zig
    // standard-library sort API changes and makes the reported median explicit.
    for (1..samples.len) |i| {
        var j = i;
        while (j > 0 and samples[j - 1].total_time_ms > samples[j].total_time_ms) : (j -= 1) {
            std.mem.swap(BenchResult, &samples[j - 1], &samples[j]);
        }
    }

    return .{
        .median = samples[samples.len / 2],
        .min_time_ms = samples[0].total_time_ms,
        .max_time_ms = samples[samples.len - 1].total_time_ms,
        .samples = samples,
    };
}

fn printHeader(allocator: Allocator) !void {
    var device = try zeno.metal.device.Device.init(allocator);
    defer device.deinit();

    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║               Zeno Engine-Pipeline Benchmark — Real MJCF Models                             ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Device: {s}; unified memory: {}\n", .{ device.getName(), device.hasUnifiedMemory() });
    std.debug.print("Zig: {s}; workload: World.step; warm-up: 10; repeats: {d}; statistic: median\n", .{ builtin.zig_version_string, benchmark_repeats });
}

fn printResult(result: RepeatedBenchResult) void {
    const median = result.median;
    std.debug.print("\n{s}:\n", .{median.name});
    std.debug.print("  Configuration: {d} envs × {d} steps\n", .{ median.num_envs, median.num_steps });
    std.debug.print("  Model: {d} bodies, {d} joints, {d} actuators\n", .{ median.num_bodies, median.num_joints, median.num_actuators });
    std.debug.print("  Median time: {d:.1} ms (range {d:.1}–{d:.1} ms)\n", .{ median.total_time_ms, result.min_time_ms, result.max_time_ms });
    std.debug.print("  Sorted samples:", .{});
    for (result.samples) |sample| std.debug.print(" {d:.1}", .{sample.total_time_ms});
    std.debug.print(" ms\n", .{});
    std.debug.print("  Median throughput: {d:.0} env-steps/sec\n", .{median.steps_per_sec});
    std.debug.print("  Memory: {d:.1} MB\n", .{median.memory_mb});
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try printHeader(allocator);

    const benchmarks = [_]BenchConfig{
        .{
            .name = "Pendulum",
            .mjcf_path = "assets/pendulum.xml",
            .num_envs = 1024,
            .num_steps = 1000,
        },
        .{
            .name = "Cartpole",
            .mjcf_path = "assets/cartpole.xml",
            .num_envs = 1024,
            .num_steps = 1000,
        },
        .{
            .name = "Pusher",
            .mjcf_path = "assets/pusher.xml",
            .num_envs = 1024,
            .num_steps = 1000,
        },
        .{
            .name = "Ant scaling (64)",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 64,
            .num_steps = 1000,
        },
        .{
            .name = "Ant scaling (256)",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 256,
            .num_steps = 1000,
        },
        .{
            .name = "Ant",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 1024,
            .num_steps = 1000,
        },
        .{
            .name = "Ant scaling (4096)",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 4096,
            .num_steps = 1000,
        },
        .{
            .name = "Ant scaling (16384)",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 16384,
            .num_steps = 1000,
        },
        .{
            .name = "Humanoid",
            .mjcf_path = "assets/humanoid.xml",
            .num_envs = 1024,
            .num_steps = 1000,
        },
    };

    var completed: usize = 0;

    for (benchmarks) |bench| {
        const result = runRepeatedBenchmark(allocator, bench) catch |err| {
            std.debug.print("\n{s}: FAILED - {}\n", .{ bench.name, err });
            continue;
        };

        printResult(result);

        completed += 1;
    }

    std.debug.print("\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});
    std.debug.print("Summary: {d}/{d} Zeno workloads completed\n", .{ completed, benchmarks.len });
    std.debug.print("No cross-simulator speedup is inferred from this benchmark.\n", .{});
}
