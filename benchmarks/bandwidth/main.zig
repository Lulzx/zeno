const std = @import("std");

// =============================================================================
// Metal Bandwidth Benchmark for M4 Pro
// Measures achievable memory bandwidth with physics-relevant access patterns
// =============================================================================

const THEORETICAL_BANDWIDTH_GBS: f64 = 273.0; // M4 Pro

// Objective-C runtime
const objc = struct {
    const c = @cImport({
        @cInclude("objc/runtime.h");
        @cInclude("objc/message.h");
    });

    const id = *anyopaque;
    const Class = *anyopaque;
    const SEL = *anyopaque;

    fn getClass(name: [*:0]const u8) ?Class {
        return @ptrCast(c.objc_getClass(name));
    }

    fn sel(name: [*:0]const u8) SEL {
        return @ptrCast(c.sel_registerName(name));
    }

    fn msgSend(comptime RetType: type, target: anytype, selector: SEL, args: anytype) RetType {
        const target_ptr: ?*anyopaque = @ptrCast(target);
        const ArgsType = @TypeOf(args);
        const args_info = @typeInfo(ArgsType);

        if (args_info.@"struct".fields.len == 0) {
            const FnType = *const fn (?*anyopaque, SEL) callconv(.c) RetType;
            const func: FnType = @ptrCast(&c.objc_msgSend);
            return func(target_ptr, selector);
        } else if (args_info.@"struct".fields.len == 1) {
            const FnType = *const fn (?*anyopaque, SEL, @TypeOf(args[0])) callconv(.c) RetType;
            const func: FnType = @ptrCast(&c.objc_msgSend);
            return func(target_ptr, selector, args[0]);
        } else if (args_info.@"struct".fields.len == 2) {
            const FnType = *const fn (?*anyopaque, SEL, @TypeOf(args[0]), @TypeOf(args[1])) callconv(.c) RetType;
            const func: FnType = @ptrCast(&c.objc_msgSend);
            return func(target_ptr, selector, args[0], args[1]);
        } else if (args_info.@"struct".fields.len == 3) {
            const FnType = *const fn (?*anyopaque, SEL, @TypeOf(args[0]), @TypeOf(args[1]), @TypeOf(args[2])) callconv(.c) RetType;
            const func: FnType = @ptrCast(&c.objc_msgSend);
            return func(target_ptr, selector, args[0], args[1], args[2]);
        } else {
            @compileError("Too many arguments");
        }
    }
};

// Metal framework functions
extern "c" fn MTLCreateSystemDefaultDevice() ?objc.id;

const MTLSize = extern struct {
    width: u64,
    height: u64,
    depth: u64,
};

const BenchmarkResult = struct {
    name: []const u8,
    bytes_transferred: u64,
    time_ns: u64,
    bandwidth_gbs: f64,
    efficiency: f64,

    pub fn print(self: BenchmarkResult) void {
        std.debug.print("  {s:<25} {d:>8.2} GB/s  ({d:>5.1}% of peak)\n", .{
            self.name,
            self.bandwidth_gbs,
            self.efficiency,
        });
    }
};

const Benchmark = struct {
    device: objc.id,
    queue: objc.id,
    library: objc.id,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Benchmark {
        const device = MTLCreateSystemDefaultDevice() orelse {
            std.debug.print("Error: No Metal device found\n", .{});
            return error.NoMetalDevice;
        };

        const queue = objc.msgSend(objc.id, device, objc.sel("newCommandQueue"), .{});

        // Load shader source
        const shader_source = @embedFile("bandwidth.metal");
        const ns_string_class = objc.getClass("NSString") orelse return error.NoNSString;
        const ns_source = objc.msgSend(
            objc.id,
            ns_string_class,
            objc.sel("stringWithUTF8String:"),
            .{shader_source.ptr},
        );

        // Compile library
        var error_ptr: ?objc.id = null;
        const library = objc.msgSend(
            ?objc.id,
            device,
            objc.sel("newLibraryWithSource:options:error:"),
            .{ ns_source, @as(?objc.id, null), &error_ptr },
        );

        if (library == null) {
            if (error_ptr) |err| {
                const desc = objc.msgSend(objc.id, err, objc.sel("localizedDescription"), .{});
                const cstr = objc.msgSend([*:0]const u8, desc, objc.sel("UTF8String"), .{});
                std.debug.print("Shader error: {s}\n", .{cstr});
            }
            return error.ShaderCompileFailed;
        }

        return .{
            .device = device,
            .queue = queue,
            .library = library.?,
            .allocator = allocator,
        };
    }

    fn createBuffer(self: Benchmark, size: usize) objc.id {
        return objc.msgSend(
            objc.id,
            self.device,
            objc.sel("newBufferWithLength:options:"),
            .{ size, @as(c_ulong, 0) }, // MTLResourceStorageModeShared
        );
    }

    fn getContents(buffer: objc.id) [*]u8 {
        return @ptrCast(objc.msgSend(?*anyopaque, buffer, objc.sel("contents"), .{}));
    }

    fn createPipeline(self: Benchmark, name: [*:0]const u8) !objc.id {
        const ns_string_class = objc.getClass("NSString") orelse return error.NoClass;
        const ns_name = objc.msgSend(objc.id, ns_string_class, objc.sel("stringWithUTF8String:"), .{name});
        const function = objc.msgSend(?objc.id, self.library, objc.sel("newFunctionWithName:"), .{ns_name}) orelse {
            std.debug.print("Function not found: {s}\n", .{name});
            return error.FunctionNotFound;
        };

        var error_ptr: ?objc.id = null;
        const pipeline = objc.msgSend(
            ?objc.id,
            self.device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ function, &error_ptr },
        ) orelse return error.PipelineFailed;

        return pipeline;
    }

    fn dispatchKernel(
        self: Benchmark,
        pipeline: objc.id,
        buffers: []const objc.id,
        threads: u64,
        threadgroup_size: u64,
    ) void {
        const cmd_buffer = objc.msgSend(objc.id, self.queue, objc.sel("commandBuffer"), .{});
        const encoder = objc.msgSend(objc.id, cmd_buffer, objc.sel("computeCommandEncoder"), .{});

        objc.msgSend(void, encoder, objc.sel("setComputePipelineState:"), .{pipeline});

        for (buffers, 0..) |buf, idx| {
            objc.msgSend(void, encoder, objc.sel("setBuffer:offset:atIndex:"), .{
                buf,
                @as(u64, 0),
                @as(u64, idx),
            });
        }

        const grid = MTLSize{ .width = threads, .height = 1, .depth = 1 };
        const group = MTLSize{ .width = threadgroup_size, .height = 1, .depth = 1 };

        objc.msgSend(void, encoder, objc.sel("dispatchThreads:threadsPerThreadgroup:"), .{ grid, group });
        objc.msgSend(void, encoder, objc.sel("endEncoding"), .{});
        objc.msgSend(void, cmd_buffer, objc.sel("commit"), .{});
        objc.msgSend(void, cmd_buffer, objc.sel("waitUntilCompleted"), .{});
    }

    fn timeKernel(
        self: Benchmark,
        pipeline: objc.id,
        buffers: []const objc.id,
        threads: u64,
        threadgroup_size: u64,
        iterations: u32,
    ) u64 {
        // Warmup
        self.dispatchKernel(pipeline, buffers, threads, threadgroup_size);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            self.dispatchKernel(pipeline, buffers, threads, threadgroup_size);
        }
        const end = std.time.nanoTimestamp();

        return @intCast(@divFloor(end - start, iterations));
    }

    pub fn runSequentialRead(self: Benchmark) !BenchmarkResult {
        const pipeline = try self.createPipeline("sequential_read");

        const num_threads: u64 = 16384;
        const elements_per_thread: u64 = 256;
        const total_elements = num_threads * elements_per_thread;
        const buffer_size = total_elements * 16;

        const input = self.createBuffer(buffer_size);
        const output = self.createBuffer(num_threads * 16);
        const counter = self.createBuffer(4);

        // Initialize
        const ptr: [*]f32 = @ptrCast(@alignCast(getContents(input)));
        for (0..total_elements * 4) |i| {
            ptr[i] = @floatFromInt(i % 1000);
        }

        const time_ns = self.timeKernel(
            pipeline,
            &[_]objc.id{ input, output, counter },
            num_threads,
            256,
            10,
        );

        const bytes = buffer_size + num_threads * 16;
        const bandwidth = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(time_ns));

        return .{
            .name = "Sequential Read",
            .bytes_transferred = bytes,
            .time_ns = time_ns,
            .bandwidth_gbs = bandwidth,
            .efficiency = bandwidth / THEORETICAL_BANDWIDTH_GBS * 100.0,
        };
    }

    pub fn runStridedRead(self: Benchmark) !BenchmarkResult {
        const pipeline = try self.createPipeline("strided_read");

        const stride: u32 = 12;
        const num_threads: u64 = 4096;
        const elements_per_thread: u64 = 256;
        const buffer_size = num_threads * elements_per_thread * stride * 16;

        const input = self.createBuffer(buffer_size);
        const output = self.createBuffer(num_threads * 16);
        const counter = self.createBuffer(4);
        const stride_buf = self.createBuffer(4);

        const stride_ptr: *u32 = @ptrCast(@alignCast(getContents(stride_buf)));
        stride_ptr.* = stride;

        const ptr: [*]f32 = @ptrCast(@alignCast(getContents(input)));
        for (0..buffer_size / 4) |i| {
            ptr[i] = @floatFromInt(i % 1000);
        }

        const time_ns = self.timeKernel(
            pipeline,
            &[_]objc.id{ input, output, counter, stride_buf },
            num_threads,
            256,
            10,
        );

        const bytes_read = num_threads * elements_per_thread * 16;
        const bytes = bytes_read + num_threads * 16;
        const bandwidth = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(time_ns));

        return .{
            .name = "Strided Read (stride=12)",
            .bytes_transferred = bytes,
            .time_ns = time_ns,
            .bandwidth_gbs = bandwidth,
            .efficiency = bandwidth / THEORETICAL_BANDWIDTH_GBS * 100.0,
        };
    }

    pub fn runGatherRead(self: Benchmark) !BenchmarkResult {
        const pipeline = try self.createPipeline("gather_read");

        const num_bodies: u32 = 16384 * 12;
        const num_threads: u64 = 8192;
        const pairs_per_thread: u64 = 128;
        const total_pairs = num_threads * pairs_per_thread;

        const bodies_buf = self.createBuffer(num_bodies * 16);
        const pairs_buf = self.createBuffer(total_pairs * 8);
        const output = self.createBuffer(num_threads * 16);
        const counter = self.createBuffer(4);

        const bodies_ptr: [*]f32 = @ptrCast(@alignCast(getContents(bodies_buf)));
        for (0..@as(usize, num_bodies) * 4) |i| {
            bodies_ptr[i] = @floatFromInt(i % 1000);
        }

        const pairs_ptr: [*]u32 = @ptrCast(@alignCast(getContents(pairs_buf)));
        var rng = std.Random.DefaultPrng.init(42);
        for (0..total_pairs * 2) |i| {
            pairs_ptr[i] = rng.random().int(u32) % num_bodies;
        }

        const time_ns = self.timeKernel(
            pipeline,
            &[_]objc.id{ bodies_buf, pairs_buf, output, counter },
            num_threads,
            256,
            10,
        );

        const bytes = total_pairs * 32 + total_pairs * 8 + num_threads * 16;
        const bandwidth = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(time_ns));

        return .{
            .name = "Gather (random pairs)",
            .bytes_transferred = bytes,
            .time_ns = time_ns,
            .bandwidth_gbs = bandwidth,
            .efficiency = bandwidth / THEORETICAL_BANDWIDTH_GBS * 100.0,
        };
    }

    pub fn runSolverPattern(self: Benchmark) !BenchmarkResult {
        const pipeline = try self.createPipeline("solver_pattern");

        const num_bodies: u32 = 16384 * 12;
        const num_contacts: u64 = 16384 * 8;

        const positions = self.createBuffer(num_bodies * 16);
        const velocities = self.createBuffer(num_bodies * 16);
        const pairs_buf = self.createBuffer(num_contacts * 8);
        const impulses = self.createBuffer(num_contacts * 16);
        const counter = self.createBuffer(4);

        const pos_ptr: [*]f32 = @ptrCast(@alignCast(getContents(positions)));
        const vel_ptr: [*]f32 = @ptrCast(@alignCast(getContents(velocities)));
        for (0..@as(usize, num_bodies) * 4) |i| {
            pos_ptr[i] = @floatFromInt(i % 100);
            vel_ptr[i] = 0;
        }

        const pairs_ptr: [*]u32 = @ptrCast(@alignCast(getContents(pairs_buf)));
        var rng = std.Random.DefaultPrng.init(42);
        for (0..num_contacts * 2) |i| {
            pairs_ptr[i] = rng.random().int(u32) % num_bodies;
        }

        const impulse_ptr: [*]f32 = @ptrCast(@alignCast(getContents(impulses)));
        for (0..num_contacts * 4) |i| {
            impulse_ptr[i] = 0.01;
        }

        const time_ns = self.timeKernel(
            pipeline,
            &[_]objc.id{ positions, velocities, pairs_buf, impulses, counter },
            num_contacts,
            256,
            10,
        );

        const bytes = num_contacts * 152;
        const bandwidth = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(time_ns));

        return .{
            .name = "Solver R/W (contacts)",
            .bytes_transferred = bytes,
            .time_ns = time_ns,
            .bandwidth_gbs = bandwidth,
            .efficiency = bandwidth / THEORETICAL_BANDWIDTH_GBS * 100.0,
        };
    }

    pub fn runSequentialWrite(self: Benchmark) !BenchmarkResult {
        const pipeline = try self.createPipeline("sequential_write");

        const num_threads: u64 = 16384;
        const elements_per_thread: u64 = 64;
        const total_output = num_threads * elements_per_thread;

        const input = self.createBuffer(num_threads * 16);
        const output = self.createBuffer(total_output * 16);

        const ptr: [*]f32 = @ptrCast(@alignCast(getContents(input)));
        for (0..num_threads * 4) |i| {
            ptr[i] = @floatFromInt(i);
        }

        const time_ns = self.timeKernel(pipeline, &[_]objc.id{ input, output }, num_threads, 256, 10);

        const bytes = num_threads * 16 + total_output * 16;
        const bandwidth = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(time_ns));

        return .{
            .name = "Sequential Write",
            .bytes_transferred = bytes,
            .time_ns = time_ns,
            .bandwidth_gbs = bandwidth,
            .efficiency = bandwidth / THEORETICAL_BANDWIDTH_GBS * 100.0,
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║         Metal Bandwidth Benchmark — Apple M4 Pro             ║\n", .{});
    std.debug.print("║         Theoretical Peak: {d:.0} GB/s                          ║\n", .{THEORETICAL_BANDWIDTH_GBS});
    std.debug.print("╚══════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});

    const bench = Benchmark.init(allocator) catch |err| {
        std.debug.print("Failed to initialize benchmark: {}\n", .{err});
        return;
    };

    std.debug.print("Running benchmarks (10 iterations each)...\n\n", .{});
    std.debug.print("Access Pattern                    Bandwidth     Efficiency\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});

    var results: [5]BenchmarkResult = undefined;

    results[0] = bench.runSequentialRead() catch |e| {
        std.debug.print("Sequential read failed: {}\n", .{e});
        return;
    };
    results[0].print();

    results[1] = bench.runStridedRead() catch |e| {
        std.debug.print("Strided read failed: {}\n", .{e});
        return;
    };
    results[1].print();

    results[2] = bench.runGatherRead() catch |e| {
        std.debug.print("Gather read failed: {}\n", .{e});
        return;
    };
    results[2].print();

    results[3] = bench.runSolverPattern() catch |e| {
        std.debug.print("Solver pattern failed: {}\n", .{e});
        return;
    };
    results[3].print();

    results[4] = bench.runSequentialWrite() catch |e| {
        std.debug.print("Sequential write failed: {}\n", .{e});
        return;
    };
    results[4].print();

    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("\n", .{});

    // Analysis
    const sequential = results[0].bandwidth_gbs;
    const gather = results[2].bandwidth_gbs;
    const solver = results[3].bandwidth_gbs;

    std.debug.print("Analysis:\n", .{});
    std.debug.print("  • Peak achieved: {d:.1} GB/s ({d:.1}% of theoretical {d:.0} GB/s)\n", .{
        sequential,
        results[0].efficiency,
        THEORETICAL_BANDWIDTH_GBS,
    });
    std.debug.print("  • Random gather: {d:.1}% of peak ({d:.1} GB/s)\n", .{
        gather / sequential * 100.0,
        gather,
    });
    std.debug.print("  • Solver pattern: {d:.1}% of peak ({d:.1} GB/s)\n", .{
        solver / sequential * 100.0,
        solver,
    });
    std.debug.print("\n", .{});

    // Throughput implications
    const contact_bytes: f64 = 152.0;
    const contacts_per_sec = solver * 1e9 / contact_bytes;
    const contacts_per_env: f64 = 8.0;
    const iterations: f64 = 4.0;

    const steps_per_sec = contacts_per_sec / (contacts_per_env * iterations);

    std.debug.print("Throughput Projections (16K envs, 8 contacts/env, 4 PBD iterations):\n", .{});
    std.debug.print("  • Contact solves/sec: {d:.1}M\n", .{contacts_per_sec / 1e6});
    std.debug.print("  • Physics steps/sec:  {d:.0}K\n", .{steps_per_sec / 1e3});
    std.debug.print("  • Time per step:      {d:.2} ms\n", .{1000.0 / steps_per_sec});
    std.debug.print("\n", .{});

    if (steps_per_sec > 10000) {
        std.debug.print("✓ Bandwidth supports >10K steps/sec — architecture is viable\n", .{});
    } else {
        std.debug.print("⚠ Bandwidth-limited to {d:.0}K steps/sec — optimize memory access\n", .{steps_per_sec / 1e3});
    }
    std.debug.print("\n", .{});
}
