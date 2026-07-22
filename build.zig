const std = @import("std");

fn linkAppleRuntime(artifact: *std.Build.Step.Compile) void {
    if (@hasDecl(std.Build.Step.Compile, "linkFramework")) {
        artifact.linkFramework("Metal");
        artifact.linkFramework("Foundation");
        artifact.linkFramework("QuartzCore");
        artifact.linkLibC();
    } else {
        artifact.root_module.linkFramework("Metal", .{});
        artifact.root_module.linkFramework("Foundation", .{});
        artifact.root_module.linkFramework("QuartzCore", .{});
        artifact.root_module.link_libc = true;
    }
}

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

    // Link against Metal and Foundation frameworks.
    linkAppleRuntime(lib);

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
    linkAppleRuntime(static_lib);

    // Bandwidth benchmark (standalone)
    const bandwidth_bench = b.addExecutable(.{
        .name = "bandwidth",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bandwidth/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    if (@hasDecl(std.Build.Step.Compile, "linkFramework")) {
        bandwidth_bench.linkFramework("Metal");
        bandwidth_bench.linkFramework("Foundation");
        bandwidth_bench.linkLibC();
    } else {
        bandwidth_bench.root_module.linkFramework("Metal", .{});
        bandwidth_bench.root_module.linkFramework("Foundation", .{});
        bandwidth_bench.root_module.link_libc = true;
    }

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
        linkAppleRuntime(unit_test);

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
        "benchmarks/bench_swarm.zig",
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
        linkAppleRuntime(bench);

        const run_bench = b.addRunArtifact(bench);
        bench_step.dependOn(&run_bench.step);
    }

    // Note: Metal shaders are embedded at compile time via @embedFile in
    // src/world/world.zig. No separate shader compilation step is needed.
    // The shaders are loaded from source at runtime using Metal's
    // newLibraryWithSource API.
}
