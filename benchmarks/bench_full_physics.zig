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
    world.reset(null);

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

fn printHeader() void {
    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║               Zeno Engine-Pipeline Benchmark — Real MJCF Models                             ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Zig: {s}; workload: World.step with repository MJCF models\n", .{builtin.zig_version_string});
}

fn printResult(result: BenchResult) void {
    std.debug.print("\n{s}:\n", .{result.name});
    std.debug.print("  Configuration: {d} envs × {d} steps\n", .{ result.num_envs, result.num_steps });
    std.debug.print("  Model: {d} bodies, {d} joints, {d} actuators\n", .{ result.num_bodies, result.num_joints, result.num_actuators });
    std.debug.print("  Time: {d:.1} ms\n", .{result.total_time_ms});
    std.debug.print("  Throughput: {d:.0} env-steps/sec\n", .{result.steps_per_sec});
    std.debug.print("  Memory: {d:.1} MB\n", .{result.memory_mb});
}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    printHeader();

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
            .name = "Ant",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 1024,
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
        const result = runBenchmark(allocator, bench) catch |err| {
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
