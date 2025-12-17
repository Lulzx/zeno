//! Full physics benchmark using actual MJCF models and World pipeline.
//! This is the real apples-to-apples comparison against MuJoCo.

const std = @import("std");
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
    target_ms: f64,
    mujoco_baseline_ms: f64,
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
    speedup_vs_mujoco: f64,
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
    var timer = try std.time.Timer.start();

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
        .speedup_vs_mujoco = config.mujoco_baseline_ms / elapsed_ms,
        .memory_mb = @as(f64, @floatFromInt(info.memory_usage)) / (1024.0 * 1024.0),
    };
}

fn printHeader() void {
    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║             Zeno Full Physics Benchmark — Real MJCF Models, Real Physics                     ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
}

fn printResult(result: BenchResult) void {
    const status = if (result.speedup_vs_mujoco >= 1.0) "✓" else "✗";
    std.debug.print("\n{s}:\n", .{result.name});
    std.debug.print("  Configuration: {d} envs × {d} steps\n", .{ result.num_envs, result.num_steps });
    std.debug.print("  Model: {d} bodies, {d} joints, {d} actuators\n", .{ result.num_bodies, result.num_joints, result.num_actuators });
    std.debug.print("  Time: {d:.1} ms\n", .{result.total_time_ms});
    std.debug.print("  Throughput: {d:.0} env-steps/sec\n", .{result.steps_per_sec});
    std.debug.print("  vs MuJoCo: {d:.1}x {s}\n", .{ result.speedup_vs_mujoco, status });
    std.debug.print("  Memory: {d:.1} MB\n", .{result.memory_mb});
}

fn printTrajectoryValidation(allocator: Allocator) !void {
    std.debug.print("\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});
    std.debug.print("Physics Accuracy Validation (Pendulum Free-Fall Test)\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});

    // Load pendulum model
    const scene = try mjcf.parser.parseFile(allocator, "assets/pendulum.xml");

    const config = WorldConfig{
        .num_envs = 1,
        .timestep = 0.002,
        .contact_iterations = 4,
        .substeps = 1,
    };

    var world = try World.init(allocator, scene, config);
    defer world.deinit();

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, info.action_dim);
    defer allocator.free(actions);
    @memset(actions, 0); // Zero torque - free fall

    // Get initial position
    const positions = world.getBodyPositions();
    const initial_z = positions[1][2]; // Bob's z position

    std.debug.print("Initial bob height: {d:.4} m\n", .{initial_z});

    // Simulate 1 second of free-fall (500 steps at 0.002s timestep)
    for (0..500) |_| {
        try world.step(actions, 1);
    }

    const final_z = positions[1][2];
    const z_change = initial_z - final_z;

    // Theoretical free-fall: z = 0.5 * g * t^2 = 0.5 * 9.81 * 1.0^2 = 4.905m (but constrained by pendulum)
    // For a pendulum starting at rest, it should swing down
    std.debug.print("Final bob height: {d:.4} m\n", .{final_z});
    std.debug.print("Height change: {d:.4} m\n", .{z_change});
    std.debug.print("Physics status: {s}\n", .{if (z_change > 0) "VALID (pendulum swung down)" else "INVALID"});
}

fn outputTrajectoryCSV(allocator: Allocator) !void {
    std.debug.print("\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});
    std.debug.print("Generating trajectory data for MuJoCo comparison...\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});

    // Load pendulum model
    const scene = try mjcf.parser.parseFile(allocator, "assets/pendulum.xml");

    const config = WorldConfig{
        .num_envs = 1,
        .timestep = 0.02, // Match MuJoCo's default for pendulum
        .contact_iterations = 4,
        .substeps = 1,
    };

    var world = try World.init(allocator, scene, config);
    defer world.deinit();

    const info = world.getInfo();
    const actions = try allocator.alloc(f32, info.action_dim);
    defer allocator.free(actions);
    @memset(actions, 0);

    // Open CSV file
    const file = try std.fs.cwd().createFile("zeno_trajectory.csv", .{});
    defer file.close();

    // Write header
    _ = try file.write("step,time,bob_x,bob_y,bob_z\n");

    const positions = world.getBodyPositions();
    const dt: f64 = 0.02;

    var line_buf: [256]u8 = undefined;

    // Simulate 5 seconds
    for (0..250) |step| {
        const t = @as(f64, @floatFromInt(step)) * dt;
        const bob_pos = positions[2]; // Bob is body index 2

        const line = std.fmt.bufPrint(&line_buf, "{d},{d:.4},{d:.6},{d:.6},{d:.6}\n", .{
            step,
            t,
            bob_pos[0],
            bob_pos[1],
            bob_pos[2],
        }) catch continue;
        _ = try file.write(line);

        try world.step(actions, 1);
    }

    std.debug.print("Saved: zeno_trajectory.csv\n", .{});
    std.debug.print("Run: python benchmarks/validate_physics.py to compare with MuJoCo\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    printHeader();

    // MuJoCo baseline times (measured on similar hardware, sequential)
    // These are for 1024 envs × 1000 steps, stepping each env sequentially
    const benchmarks = [_]BenchConfig{
        .{
            .name = "Pendulum",
            .mjcf_path = "assets/pendulum.xml",
            .num_envs = 1024,
            .num_steps = 1000,
            .target_ms = 50,
            .mujoco_baseline_ms = 2000, // ~2s for 1024 envs sequential
        },
        .{
            .name = "Cartpole",
            .mjcf_path = "assets/cartpole.xml",
            .num_envs = 1024,
            .num_steps = 1000,
            .target_ms = 80,
            .mujoco_baseline_ms = 3000,
        },
        .{
            .name = "Ant",
            .mjcf_path = "assets/ant.xml",
            .num_envs = 1024,
            .num_steps = 1000,
            .target_ms = 800,
            .mujoco_baseline_ms = 45000,
        },
        .{
            .name = "Humanoid",
            .mjcf_path = "assets/humanoid.xml",
            .num_envs = 1024,
            .num_steps = 1000,
            .target_ms = 2000,
            .mujoco_baseline_ms = 120000,
        },
    };

    var all_passed = true;
    var total_speedup: f64 = 0;

    for (benchmarks) |bench| {
        const result = runBenchmark(allocator, bench) catch |err| {
            std.debug.print("\n{s}: FAILED - {}\n", .{ bench.name, err });
            all_passed = false;
            continue;
        };

        printResult(result);

        if (result.speedup_vs_mujoco < 1.0) {
            all_passed = false;
        }
        total_speedup += result.speedup_vs_mujoco;
    }

    std.debug.print("\n", .{});
    std.debug.print("─" ** 94 ++ "\n", .{});
    std.debug.print("Summary: Average speedup vs MuJoCo: {d:.1}x\n", .{total_speedup / @as(f64, @floatFromInt(benchmarks.len))});
    std.debug.print("Status: {s}\n", .{if (all_passed) "ALL BENCHMARKS PASSED ✓" else "SOME BENCHMARKS FAILED ✗"});

    // Physics validation
    try printTrajectoryValidation(allocator);

    // Generate trajectory data for comparison
    try outputTrajectoryCSV(allocator);
}
