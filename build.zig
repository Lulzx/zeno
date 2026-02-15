const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library (dynamic)
    const lib = b.addLibrary(.{
        .name = "zeno",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
    });

    // Link against Metal and Foundation frameworks
    lib.linkFramework("Metal");
    lib.linkFramework("Foundation");
    lib.linkFramework("QuartzCore");
    lib.linkLibC();

    // Install the library
    b.installArtifact(lib);

    // Static library for testing
    const static_lib = b.addLibrary(.{
        .name = "zeno_static",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    static_lib.linkFramework("Metal");
    static_lib.linkFramework("Foundation");
    static_lib.linkFramework("QuartzCore");
    static_lib.linkLibC();

    // Bandwidth benchmark (standalone)
    const bandwidth_bench = b.addExecutable(.{
        .name = "bandwidth",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bandwidth/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    bandwidth_bench.linkFramework("Metal");
    bandwidth_bench.linkFramework("Foundation");
    bandwidth_bench.linkLibC();

    b.installArtifact(bandwidth_bench);

    const run_bandwidth = b.addRunArtifact(bandwidth_bench);
    const bandwidth_step = b.step("bandwidth", "Run memory bandwidth benchmark");
    bandwidth_step.dependOn(&run_bandwidth.step);

    // Tests
    const test_step = b.step("test", "Run unit tests");

    const test_files = [_][]const u8{
        "tests/test_metal.zig",
        "tests/test_physics.zig",
        "tests/test_physics_integration.zig",
        "tests/test_collision.zig",
        "tests/test_mjcf.zig",
        "tests/test_xpbd.zig",
        "tests/test_memory.zig",
        "tests/test_environments.zig",
        "tests/test_softbody.zig",
        "tests/test_fluid.zig",
        "tests/test_sensors.zig",
        "tests/test_tendon.zig",
        "tests/test_swarm.zig",
    };

    for (test_files) |test_file| {
        const unit_test = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(test_file),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zeno", .module = lib.root_module },
                },
            }),
        });
        unit_test.linkFramework("Metal");
        unit_test.linkFramework("Foundation");
        unit_test.linkFramework("QuartzCore");
        unit_test.linkLibC();

        const run_test = b.addRunArtifact(unit_test);
        test_step.dependOn(&run_test.step);
    }

    // Benchmarks
    const bench_step = b.step("bench", "Run benchmarks");

    const bench_files = [_][]const u8{
        "benchmarks/bench_integration.zig",
        "benchmarks/bench_collision.zig",
        "benchmarks/bench_envs.zig",
        "benchmarks/bench_full_physics.zig",
    };

    for (bench_files) |bench_file| {
        const bench = b.addExecutable(.{
            .name = std.fs.path.stem(bench_file),
            .root_module = b.createModule(.{
                .root_source_file = b.path(bench_file),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "zeno", .module = lib.root_module },
                },
            }),
        });
        bench.linkFramework("Metal");
        bench.linkFramework("Foundation");
        bench.linkFramework("QuartzCore");
        bench.linkLibC();

        const run_bench = b.addRunArtifact(bench);
        bench_step.dependOn(&run_bench.step);
    }

    // Note: Metal shaders are embedded at compile time via @embedFile in
    // src/world/world.zig. No separate shader compilation step is needed.
    // The shaders are loaded from source at runtime using Metal's
    // newLibraryWithSource API.
}
