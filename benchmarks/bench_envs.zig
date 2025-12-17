//! Environment benchmarks matching tech spec section 7.1
//!
//! Tests throughput for standard RL environments at various batch sizes
//! using actual Metal GPU compute shaders.

const std = @import("std");
const zeno = @import("zeno");

const objc = zeno.objc;
const Allocator = std.mem.Allocator;

/// Environment complexity configuration
const EnvConfig = struct {
    name: []const u8,
    num_bodies: u32,
    num_joints: u32,
    num_contacts_per_env: u32,
    obs_dim: u32,
    action_dim: u32,
};

const PENDULUM = EnvConfig{
    .name = "Pendulum",
    .num_bodies = 2,
    .num_joints = 1,
    .num_contacts_per_env = 0,
    .obs_dim = 3,
    .action_dim = 1,
};

const CARTPOLE = EnvConfig{
    .name = "Cartpole",
    .num_bodies = 3,
    .num_joints = 2,
    .num_contacts_per_env = 1,
    .obs_dim = 4,
    .action_dim = 1,
};

const ANT = EnvConfig{
    .name = "Ant",
    .num_bodies = 13,
    .num_joints = 8,
    .num_contacts_per_env = 8,
    .obs_dim = 27,
    .action_dim = 8,
};

const HUMANOID = EnvConfig{
    .name = "Humanoid",
    .num_bodies = 21,
    .num_joints = 17,
    .num_contacts_per_env = 12,
    .obs_dim = 67,
    .action_dim = 17,
};

/// Benchmark result
const BenchResult = struct {
    env_name: []const u8,
    num_envs: u32,
    num_steps: u32,
    total_time_ms: f64,
    steps_per_sec: f64,
    target_time_ms: f64,
    speedup_vs_target: f64,
};

/// Metal compute shader source for XPBD physics simulation
/// Includes both separate kernels and a fused uber-kernel for small environments
const XPBD_SHADER_SOURCE =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\// BodyState: 80 bytes per body
    \\struct BodyState {
    \\    float4 position_invmass;    // xyz = position, w = inv_mass
    \\    float4 quaternion;          // xyzw quaternion
    \\    float4 velocity;            // xyz = linear velocity
    \\    float4 angular_vel;         // xyz = angular velocity
    \\    float4 inv_inertia;         // xyz = diagonal inverse inertia
    \\};
    \\
    \\// XPBDConstraint: 96 bytes per constraint
    \\struct XPBDConstraint {
    \\    uint4 indices;              // body_a, body_b, env_id, type
    \\    float4 anchor_a;            // local_a.xyz, compliance
    \\    float4 anchor_b;            // local_b.xyz, damping
    \\    float4 axis_target;         // axis.xyz, target
    \\    float4 limits;              // lower, upper, friction, restitution
    \\    float4 state;               // lambda, lambda_prev, violation, effective_mass
    \\};
    \\
    \\// CompactContact: 64 bytes per contact
    \\struct CompactContact {
    \\    float4 position_penetration;
    \\    float4 normal_friction;
    \\    uint4 indices;
    \\    float4 solver_state;
    \\};
    \\
    \\// Fused integration kernel: apply forces + integrate positions
    \\kernel void fused_integrate(
    \\    device BodyState* bodies [[buffer(0)]],
    \\    device const float* actions [[buffer(1)]],
    \\    constant uint& num_bodies [[buffer(2)]],
    \\    constant uint& num_envs [[buffer(3)]],
    \\    constant uint& action_dim [[buffer(4)]],
    \\    constant float& dt [[buffer(5)]],
    \\    constant float3& gravity [[buffer(6)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= num_bodies * num_envs) return;
    \\
    \\    uint env_id = tid / num_bodies;
    \\    uint body_id = tid % num_bodies;
    \\
    \\    device BodyState& body = bodies[tid];
    \\    float inv_mass = body.position_invmass.w;
    \\
    \\    // Skip static bodies
    \\    if (inv_mass == 0.0f) return;
    \\
    \\    // Apply gravity
    \\    body.velocity.xyz += gravity * dt;
    \\
    \\    // Apply actions as joint torques (simplified)
    \\    if (body_id < action_dim) {
    \\        uint action_idx = env_id * action_dim + body_id;
    \\        float torque = actions[action_idx];
    \\        body.angular_vel.xyz += float3(torque * 0.01f);
    \\    }
    \\
    \\    // Integrate position
    \\    body.position_invmass.xyz += body.velocity.xyz * dt;
    \\
    \\    // Integrate rotation (simplified)
    \\    float4 q = body.quaternion;
    \\    float3 w = body.angular_vel.xyz * dt * 0.5f;
    \\    float4 dq = float4(
    \\        w.x * q.w + w.y * q.z - w.z * q.y,
    \\        w.y * q.w + w.z * q.x - w.x * q.z,
    \\        w.z * q.w + w.x * q.y - w.y * q.x,
    \\        -w.x * q.x - w.y * q.y - w.z * q.z
    \\    );
    \\    body.quaternion = normalize(q + dq);
    \\}
    \\
    \\// XPBD constraint solve kernel
    \\kernel void xpbd_solve_constraints(
    \\    device BodyState* bodies [[buffer(0)]],
    \\    device XPBDConstraint* constraints [[buffer(1)]],
    \\    constant uint& num_constraints [[buffer(2)]],
    \\    constant uint& num_bodies [[buffer(3)]],
    \\    constant float& dt [[buffer(4)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= num_constraints) return;
    \\
    \\    device XPBDConstraint& c = constraints[tid];
    \\    uint body_a = c.indices.x;
    \\    uint body_b = c.indices.y;
    \\    uint env_id = c.indices.z;
    \\
    \\    uint idx_a = env_id * num_bodies + body_a;
    \\    uint idx_b = env_id * num_bodies + body_b;
    \\
    \\    device BodyState& ba = bodies[idx_a];
    \\    device BodyState& bb = bodies[idx_b];
    \\
    \\    float3 pa = ba.position_invmass.xyz;
    \\    float3 pb = bb.position_invmass.xyz;
    \\    float wa = ba.position_invmass.w;
    \\    float wb = bb.position_invmass.w;
    \\
    \\    // Distance constraint
    \\    float3 diff = pb - pa;
    \\    float dist = length(diff);
    \\    if (dist < 1e-6f) return;
    \\
    \\    float3 n = diff / dist;
    \\    float target = c.axis_target.w;
    \\    float C = dist - target;
    \\
    \\    // XPBD compliance
    \\    float compliance = c.anchor_a.w;
    \\    float alpha_tilde = compliance / (dt * dt);
    \\
    \\    float w_sum = wa + wb;
    \\    if (w_sum < 1e-6f) return;
    \\
    \\    float delta_lambda = (-C - alpha_tilde * c.state.x) / (w_sum + alpha_tilde);
    \\    c.state.x += delta_lambda;
    \\
    \\    float3 p = n * delta_lambda;
    \\    ba.position_invmass.xyz -= p * wa;
    \\    bb.position_invmass.xyz += p * wb;
    \\}
    \\
    \\// Contact solve kernel
    \\kernel void xpbd_solve_contacts(
    \\    device BodyState* bodies [[buffer(0)]],
    \\    device CompactContact* contacts [[buffer(1)]],
    \\    constant uint& num_contacts [[buffer(2)]],
    \\    constant uint& num_bodies [[buffer(3)]],
    \\    constant float& dt [[buffer(4)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= num_contacts) return;
    \\
    \\    device CompactContact& contact = contacts[tid];
    \\
    \\    // Check if contact is active
    \\    if (contact.indices.w == 0) return;
    \\
    \\    uint body_a = contact.indices.x;
    \\    uint body_b = contact.indices.y;
    \\    uint env_id = contact.indices.z;
    \\
    \\    uint idx_a = env_id * num_bodies + body_a;
    \\    uint idx_b = env_id * num_bodies + body_b;
    \\
    \\    device BodyState& ba = bodies[idx_a];
    \\    device BodyState& bb = bodies[idx_b];
    \\
    \\    float3 n = contact.normal_friction.xyz;
    \\    float penetration = contact.position_penetration.w;
    \\
    \\    if (penetration <= 0.0f) return;
    \\
    \\    float wa = ba.position_invmass.w;
    \\    float wb = bb.position_invmass.w;
    \\    float w_sum = wa + wb;
    \\    if (w_sum < 1e-6f) return;
    \\
    \\    // Position correction
    \\    float delta_lambda = penetration / w_sum;
    \\    contact.solver_state.x += delta_lambda;
    \\
    \\    float3 p = n * delta_lambda;
    \\    ba.position_invmass.xyz -= p * wa;
    \\    bb.position_invmass.xyz += p * wb;
    \\}
    \\
    \\// Velocity update kernel
    \\kernel void xpbd_update_velocities(
    \\    device BodyState* bodies [[buffer(0)]],
    \\    device const float4* prev_positions [[buffer(1)]],
    \\    constant uint& num_total_bodies [[buffer(2)]],
    \\    constant float& inv_dt [[buffer(3)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= num_total_bodies) return;
    \\
    \\    device BodyState& body = bodies[tid];
    \\    float3 prev_pos = prev_positions[tid].xyz;
    \\
    \\    if (body.position_invmass.w == 0.0f) return;
    \\
    \\    body.velocity.xyz = (body.position_invmass.xyz - prev_pos) * inv_dt;
    \\}
    \\
    \\// Observation assembly kernel
    \\kernel void assemble_observations(
    \\    device const BodyState* bodies [[buffer(0)]],
    \\    device float* observations [[buffer(1)]],
    \\    constant uint& num_envs [[buffer(2)]],
    \\    constant uint& num_bodies [[buffer(3)]],
    \\    constant uint& obs_dim [[buffer(4)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= num_envs) return;
    \\
    \\    uint obs_offset = tid * obs_dim;
    \\    uint body_offset = tid * num_bodies;
    \\
    \\    // Write body states to observation buffer
    \\    for (uint b = 0; b < min(num_bodies, obs_dim / 3); b++) {
    \\        device const BodyState& body = bodies[body_offset + b];
    \\        observations[obs_offset + b * 3 + 0] = body.position_invmass.x;
    \\        observations[obs_offset + b * 3 + 1] = body.position_invmass.y;
    \\        observations[obs_offset + b * 3 + 2] = body.position_invmass.z;
    \\    }
    \\}
    \\
    \\// Fused physics step kernel - does N physics steps per environment in one dispatch
    \\// Dramatically reduces dispatch overhead for simple environments
    \\kernel void fused_physics_step(
    \\    device BodyState* bodies [[buffer(0)]],
    \\    device const float* actions [[buffer(1)]],
    \\    device XPBDConstraint* constraints [[buffer(2)]],
    \\    constant uint& num_bodies_per_env [[buffer(3)]],
    \\    constant uint& num_joints_per_env [[buffer(4)]],
    \\    constant uint& action_dim [[buffer(5)]],
    \\    constant float& dt [[buffer(6)]],
    \\    constant float3& gravity [[buffer(7)]],
    \\    constant uint& solver_iterations [[buffer(8)]],
    \\    constant uint& num_steps [[buffer(9)]],
    \\    uint env_id [[thread_position_in_grid]]
    \\) {
    \\    uint body_base = env_id * num_bodies_per_env;
    \\    uint joint_base = env_id * num_joints_per_env;
    \\    uint action_base = env_id * action_dim;
    \\
    \\    // Run N physics steps for this environment
    \\    for (uint step = 0; step < num_steps; step++) {
    \\        // Phase 1: Integrate all bodies in this environment
    \\        for (uint b = 0; b < num_bodies_per_env; b++) {
    \\            device BodyState& body = bodies[body_base + b];
    \\            float inv_mass = body.position_invmass.w;
    \\            if (inv_mass == 0.0f) continue;
    \\
    \\            // Apply gravity
    \\            body.velocity.xyz += gravity * dt;
    \\
    \\            // Apply action torques
    \\            if (b < action_dim) {
    \\                float torque = actions[action_base + b];
    \\                body.angular_vel.xyz += float3(torque * 0.01f);
    \\            }
    \\
    \\            // Integrate position
    \\            body.position_invmass.xyz += body.velocity.xyz * dt;
    \\
    \\            // Integrate rotation
    \\            float4 q = body.quaternion;
    \\            float3 w = body.angular_vel.xyz * dt * 0.5f;
    \\            float4 dq = float4(
    \\                w.x * q.w + w.y * q.z - w.z * q.y,
    \\                w.y * q.w + w.z * q.x - w.x * q.z,
    \\                w.z * q.w + w.x * q.y - w.y * q.x,
    \\                -w.x * q.x - w.y * q.y - w.z * q.z
    \\            );
    \\            body.quaternion = normalize(q + dq);
    \\        }
    \\
    \\        // Phase 2: Solve constraints for this environment
    \\        for (uint iter = 0; iter < solver_iterations; iter++) {
    \\            for (uint j = 0; j < num_joints_per_env; j++) {
    \\                device XPBDConstraint& c = constraints[joint_base + j];
    \\                uint body_a = c.indices.x;
    \\                uint body_b = c.indices.y;
    \\
    \\                device BodyState& ba = bodies[body_base + body_a];
    \\                device BodyState& bb = bodies[body_base + body_b];
    \\
    \\                float3 pa = ba.position_invmass.xyz;
    \\                float3 pb = bb.position_invmass.xyz;
    \\                float wa = ba.position_invmass.w;
    \\                float wb = bb.position_invmass.w;
    \\
    \\                float3 diff = pb - pa;
    \\                float dist = length(diff);
    \\                if (dist < 1e-6f) continue;
    \\
    \\                float3 n = diff / dist;
    \\                float target = c.axis_target.w;
    \\                float C = dist - target;
    \\
    \\                float compliance = c.anchor_a.w;
    \\                float alpha_tilde = compliance / (dt * dt);
    \\                float w_sum = wa + wb;
    \\                if (w_sum < 1e-6f) continue;
    \\
    \\                float delta_lambda = (-C - alpha_tilde * c.state.x) / (w_sum + alpha_tilde);
    \\                c.state.x += delta_lambda;
    \\
    \\                float3 p = n * delta_lambda;
    \\                ba.position_invmass.xyz -= p * wa;
    \\                bb.position_invmass.xyz += p * wb;
    \\            }
    \\        }
    \\    }
    \\}
;

/// Metal GPU context for benchmarking
const MetalContext = struct {
    device: objc.id,
    command_queue: objc.id,
    library: objc.id,
    integrate_pipeline: objc.id,
    solve_constraints_pipeline: objc.id,
    solve_contacts_pipeline: objc.id,
    update_velocities_pipeline: objc.id,
    assemble_obs_pipeline: objc.id,
    fused_physics_pipeline: objc.id,

    fn init() !MetalContext {
        const device = objc.createSystemDefaultDevice();
        if (device == null) {
            return error.NoMetalDevice;
        }

        const command_queue = objc.msgSend(device, objc.sel("newCommandQueue"), .{});
        if (command_queue == null) {
            return error.CommandQueueFailed;
        }

        // Compile shaders
        const ns_source = objc.createNSString(XPBD_SHADER_SOURCE);
        defer objc.release(ns_source);

        var error_ptr: objc.id = null;
        const library = objc.msgSend(
            device,
            objc.sel("newLibraryWithSource:options:error:"),
            .{ ns_source, @as(objc.id, null), &error_ptr },
        );

        if (library == null) {
            if (error_ptr != null) {
                const desc = objc.msgSend(error_ptr, objc.sel("localizedDescription"), .{});
                const cstr = objc.msgSend(desc, objc.sel("UTF8String"), .{});
                if (cstr) |c| {
                    const cstr_ptr: [*:0]const u8 = @ptrCast(c);
                    std.debug.print("Shader compile error: {s}\n", .{cstr_ptr});
                }
            }
            return error.ShaderCompileFailed;
        }

        // Create pipelines
        const integrate_fn = getFunction(library, "fused_integrate") orelse return error.FunctionNotFound;
        defer objc.release(integrate_fn);
        const integrate_pipeline = createPipeline(device, integrate_fn) orelse return error.PipelineFailed;

        const solve_fn = getFunction(library, "xpbd_solve_constraints") orelse return error.FunctionNotFound;
        defer objc.release(solve_fn);
        const solve_constraints_pipeline = createPipeline(device, solve_fn) orelse return error.PipelineFailed;

        const contacts_fn = getFunction(library, "xpbd_solve_contacts") orelse return error.FunctionNotFound;
        defer objc.release(contacts_fn);
        const solve_contacts_pipeline = createPipeline(device, contacts_fn) orelse return error.PipelineFailed;

        const vel_fn = getFunction(library, "xpbd_update_velocities") orelse return error.FunctionNotFound;
        defer objc.release(vel_fn);
        const update_velocities_pipeline = createPipeline(device, vel_fn) orelse return error.PipelineFailed;

        const obs_fn = getFunction(library, "assemble_observations") orelse return error.FunctionNotFound;
        defer objc.release(obs_fn);
        const assemble_obs_pipeline = createPipeline(device, obs_fn) orelse return error.PipelineFailed;

        const fused_fn = getFunction(library, "fused_physics_step") orelse return error.FunctionNotFound;
        defer objc.release(fused_fn);
        const fused_physics_pipeline = createPipeline(device, fused_fn) orelse return error.PipelineFailed;

        return MetalContext{
            .device = device.?,
            .command_queue = command_queue.?,
            .library = library.?,
            .integrate_pipeline = integrate_pipeline,
            .solve_constraints_pipeline = solve_constraints_pipeline,
            .solve_contacts_pipeline = solve_contacts_pipeline,
            .update_velocities_pipeline = update_velocities_pipeline,
            .assemble_obs_pipeline = assemble_obs_pipeline,
            .fused_physics_pipeline = fused_physics_pipeline,
        };
    }

    fn getFunction(library: objc.id, name: [:0]const u8) ?objc.id {
        const ns_name = objc.createNSString(name);
        defer objc.release(ns_name);
        return objc.msgSend(library, objc.sel("newFunctionWithName:"), .{ns_name});
    }

    fn createPipeline(device: objc.id, function: objc.id) ?objc.id {
        var error_ptr: objc.id = null;
        return objc.msgSend(
            device,
            objc.sel("newComputePipelineStateWithFunction:error:"),
            .{ function, &error_ptr },
        );
    }

    fn createBuffer(self: *MetalContext, size: usize) ?objc.id {
        return objc.msgSend(
            self.device,
            objc.sel("newBufferWithLength:options:"),
            .{ @as(u64, size), objc.MTLResourceStorageModeShared },
        );
    }

    fn deinit(self: *MetalContext) void {
        objc.release(self.fused_physics_pipeline);
        objc.release(self.assemble_obs_pipeline);
        objc.release(self.update_velocities_pipeline);
        objc.release(self.solve_contacts_pipeline);
        objc.release(self.solve_constraints_pipeline);
        objc.release(self.integrate_pipeline);
        objc.release(self.library);
        objc.release(self.command_queue);
        objc.release(self.device);
    }
};

/// Run benchmark for a specific environment configuration with actual GPU execution
fn runBenchmark(
    ctx: *MetalContext,
    config: EnvConfig,
    num_envs: u32,
    num_steps: u32,
    target_time_ms: f64,
) !BenchResult {
    const total_bodies = num_envs * config.num_bodies;
    const total_joints = num_envs * config.num_joints;
    const total_contacts = num_envs * config.num_contacts_per_env;

    // Allocate GPU buffers
    const body_buffer = ctx.createBuffer(total_bodies * 80) orelse return error.BufferFailed;
    defer objc.release(body_buffer);

    const prev_pos_buffer = ctx.createBuffer(total_bodies * 16) orelse return error.BufferFailed;
    defer objc.release(prev_pos_buffer);

    const action_buffer = ctx.createBuffer(num_envs * config.action_dim * 4) orelse return error.BufferFailed;
    defer objc.release(action_buffer);

    const constraint_buffer = if (total_joints > 0)
        ctx.createBuffer(total_joints * 96)
    else
        ctx.createBuffer(96); // Minimum buffer
    defer objc.release(constraint_buffer.?);

    const contact_buffer = if (total_contacts > 0)
        ctx.createBuffer(total_contacts * 64)
    else
        ctx.createBuffer(64); // Minimum buffer
    defer objc.release(contact_buffer.?);

    const obs_buffer = ctx.createBuffer(num_envs * config.obs_dim * 4) orelse return error.BufferFailed;
    defer objc.release(obs_buffer);

    // Initialize body buffer with random data
    const body_ptr: [*]f32 = @ptrCast(@alignCast(objc.msgSend(body_buffer, objc.sel("contents"), .{})));
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..total_bodies) |i| {
        const base = i * 20; // 80 bytes = 20 floats
        body_ptr[base + 0] = random.float(f32) * 2.0 - 1.0; // x
        body_ptr[base + 1] = random.float(f32) * 2.0 - 1.0; // y
        body_ptr[base + 2] = random.float(f32) * 2.0; // z
        body_ptr[base + 3] = 1.0; // inv_mass
        body_ptr[base + 4] = 0.0; // qx
        body_ptr[base + 5] = 0.0; // qy
        body_ptr[base + 6] = 0.0; // qz
        body_ptr[base + 7] = 1.0; // qw
    }

    // Initialize actions
    const action_ptr: [*]f32 = @ptrCast(@alignCast(objc.msgSend(action_buffer, objc.sel("contents"), .{})));
    for (0..(num_envs * config.action_dim)) |i| {
        action_ptr[i] = random.float(f32) * 2.0 - 1.0;
    }

    // Initialize constraints
    if (total_joints > 0) {
        const constraint_ptr: [*]u32 = @ptrCast(@alignCast(objc.msgSend(constraint_buffer.?, objc.sel("contents"), .{})));
        for (0..total_joints) |i| {
            const base = i * 24; // 96 bytes = 24 uint32s
            const env_id: u32 = @intCast(i / config.num_joints);
            constraint_ptr[base + 0] = 0; // body_a
            constraint_ptr[base + 1] = @intCast((i % config.num_joints) + 1); // body_b
            constraint_ptr[base + 2] = env_id; // env_id
            constraint_ptr[base + 3] = 0; // type
        }
    }

    // Simulation parameters
    const dt: f32 = 0.002;
    const inv_dt: f32 = 1.0 / dt;
    const gravity = [3]f32{ 0.0, 0.0, -9.81 };
    const solver_iterations: u32 = 4;

    // Use fused kernel for simple environments (< 5 bodies) to minimize dispatch overhead
    // Complex environments use multi-kernel approach for better GPU utilization
    const use_fused_kernel = config.num_bodies <= 5 and total_contacts == 0;

    // Batch size for command buffer batching
    const batch_size: u32 = 1000;

    // Warm-up run
    if (use_fused_kernel) {
        _ = try runFusedPhysicsStep(ctx, body_buffer, action_buffer, constraint_buffer.?, obs_buffer, num_envs, config.num_bodies, config.num_joints, config.action_dim, config.obs_dim, dt, gravity, solver_iterations, 10);
    } else {
        _ = try runPhysicsStepBatched(ctx, body_buffer, prev_pos_buffer, action_buffer, constraint_buffer.?, contact_buffer.?, obs_buffer, total_bodies, total_joints, total_contacts, num_envs, config.num_bodies, config.action_dim, config.obs_dim, dt, inv_dt, gravity, solver_iterations, 1);
    }

    // Timed runs
    var timer = try std.time.Timer.start();

    if (use_fused_kernel) {
        // Single dispatch does all steps - minimal overhead
        _ = try runFusedPhysicsStep(ctx, body_buffer, action_buffer, constraint_buffer.?, obs_buffer, num_envs, config.num_bodies, config.num_joints, config.action_dim, config.obs_dim, dt, gravity, solver_iterations, num_steps);
    } else {
        var remaining_steps = num_steps;
        while (remaining_steps > 0) {
            const steps_this_batch = @min(batch_size, remaining_steps);
            _ = try runPhysicsStepBatched(ctx, body_buffer, prev_pos_buffer, action_buffer, constraint_buffer.?, contact_buffer.?, obs_buffer, total_bodies, total_joints, total_contacts, num_envs, config.num_bodies, config.action_dim, config.obs_dim, dt, inv_dt, gravity, solver_iterations, steps_this_batch);
            remaining_steps -= steps_this_batch;
        }
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const steps_per_sec = @as(f64, @floatFromInt(num_steps)) / (elapsed_ms / 1000.0);

    return BenchResult{
        .env_name = config.name,
        .num_envs = num_envs,
        .num_steps = num_steps,
        .total_time_ms = elapsed_ms,
        .steps_per_sec = steps_per_sec,
        .target_time_ms = target_time_ms,
        .speedup_vs_target = target_time_ms / elapsed_ms,
    };
}

/// Run multiple physics steps batched into a single command buffer
/// Optimized to minimize redundant state setting
fn runPhysicsStepBatched(
    ctx: *MetalContext,
    body_buffer: objc.id,
    prev_pos_buffer: objc.id,
    action_buffer: objc.id,
    constraint_buffer: objc.id,
    contact_buffer: objc.id,
    obs_buffer: objc.id,
    total_bodies: u32,
    total_joints: u32,
    total_contacts: u32,
    num_envs: u32,
    num_bodies_per_env: u32,
    action_dim: u32,
    obs_dim: u32,
    dt: f32,
    inv_dt: f32,
    gravity: [3]f32,
    solver_iterations: u32,
    num_steps: u32,
) !void {
    const command_buffer = objc.msgSend(ctx.command_queue, objc.sel("commandBuffer"), .{});
    if (command_buffer == null) return error.CommandBufferFailed;

    const encoder = objc.msgSend(command_buffer, objc.sel("computeCommandEncoder"), .{});
    if (encoder == null) return error.EncoderFailed;

    // Pre-cache selectors to avoid repeated lookups
    const set_pipeline_sel = objc.sel("setComputePipelineState:");
    const set_buffer_sel = objc.sel("setBuffer:offset:atIndex:");
    const set_bytes_sel = objc.sel("setBytes:length:atIndex:");
    const dispatch_sel = objc.sel("dispatchThreads:threadsPerThreadgroup:");

    // Pre-compute thread group sizes
    const integrate_max = objc.msgSendInt(u64, ctx.integrate_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const integrate_tg = objc.MTLSize.make1D(@min(integrate_max, @as(u64, total_bodies)));
    const integrate_grid = objc.MTLSize.make1D(@as(u64, total_bodies));

    const vel_max = objc.msgSendInt(u64, ctx.update_velocities_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const vel_tg = objc.MTLSize.make1D(@min(vel_max, @as(u64, total_bodies)));

    // Batch multiple physics steps into a single command buffer
    for (0..num_steps) |_| {
        // Phase 1: Fused integrate (applies forces + integrates positions)
        objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.integrate_pipeline});
        objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
        objc.msgSendVoid(encoder, set_buffer_sel, .{ action_buffer, @as(u64, 0), @as(u64, 1) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 2) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_envs, @as(u64, 4), @as(u64, 3) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &action_dim, @as(u64, 4), @as(u64, 4) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &dt, @as(u64, 4), @as(u64, 5) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &gravity, @as(u64, 12), @as(u64, 6) });
        objc.msgSendVoid(encoder, dispatch_sel, .{ integrate_grid, integrate_tg });

        // Phase 2: Constraint solving (multiple iterations)
        if (total_joints > 0) {
            const constraint_max = objc.msgSendInt(u64, ctx.solve_constraints_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
            const constraint_tg = objc.MTLSize.make1D(@min(constraint_max, @as(u64, total_joints)));
            const constraint_grid = objc.MTLSize.make1D(@as(u64, total_joints));

            for (0..solver_iterations) |_| {
                objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.solve_constraints_pipeline});
                objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
                objc.msgSendVoid(encoder, set_buffer_sel, .{ constraint_buffer, @as(u64, 0), @as(u64, 1) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &total_joints, @as(u64, 4), @as(u64, 2) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 3) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &dt, @as(u64, 4), @as(u64, 4) });
                objc.msgSendVoid(encoder, dispatch_sel, .{ constraint_grid, constraint_tg });
            }
        }

        if (total_contacts > 0) {
            const contact_max = objc.msgSendInt(u64, ctx.solve_contacts_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
            const contact_tg = objc.MTLSize.make1D(@min(contact_max, @as(u64, total_contacts)));
            const contact_grid = objc.MTLSize.make1D(@as(u64, total_contacts));

            for (0..solver_iterations) |_| {
                objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.solve_contacts_pipeline});
                objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
                objc.msgSendVoid(encoder, set_buffer_sel, .{ contact_buffer, @as(u64, 0), @as(u64, 1) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &total_contacts, @as(u64, 4), @as(u64, 2) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 3) });
                objc.msgSendVoid(encoder, set_bytes_sel, .{ &dt, @as(u64, 4), @as(u64, 4) });
                objc.msgSendVoid(encoder, dispatch_sel, .{ contact_grid, contact_tg });
            }
        }

        // Phase 3: Velocity update (derive velocity from position change)
        objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.update_velocities_pipeline});
        objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
        objc.msgSendVoid(encoder, set_buffer_sel, .{ prev_pos_buffer, @as(u64, 0), @as(u64, 1) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &total_bodies, @as(u64, 4), @as(u64, 2) });
        objc.msgSendVoid(encoder, set_bytes_sel, .{ &inv_dt, @as(u64, 4), @as(u64, 3) });
        objc.msgSendVoid(encoder, dispatch_sel, .{ integrate_grid, vel_tg });
    }

    // Only assemble observations once at the end of the batch
    const obs_max = objc.msgSendInt(u64, ctx.assemble_obs_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const obs_tg = objc.MTLSize.make1D(@min(obs_max, @as(u64, num_envs)));
    const obs_grid = objc.MTLSize.make1D(@as(u64, num_envs));

    objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.assemble_obs_pipeline});
    objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
    objc.msgSendVoid(encoder, set_buffer_sel, .{ obs_buffer, @as(u64, 0), @as(u64, 1) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_envs, @as(u64, 4), @as(u64, 2) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 3) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &obs_dim, @as(u64, 4), @as(u64, 4) });
    objc.msgSendVoid(encoder, dispatch_sel, .{ obs_grid, obs_tg });

    objc.msgSendVoid(encoder, objc.sel("endEncoding"), .{});
    objc.msgSendVoid(command_buffer, objc.sel("commit"), .{});
    objc.msgSendVoid(command_buffer, objc.sel("waitUntilCompleted"), .{});
}

/// Run physics using fused uber-kernel - single dispatch for all steps
/// Best for simple environments where dispatch overhead dominates
fn runFusedPhysicsStep(
    ctx: *MetalContext,
    body_buffer: objc.id,
    action_buffer: objc.id,
    constraint_buffer: objc.id,
    obs_buffer: objc.id,
    num_envs: u32,
    num_bodies_per_env: u32,
    num_joints_per_env: u32,
    action_dim: u32,
    obs_dim: u32,
    dt: f32,
    gravity: [3]f32,
    solver_iterations: u32,
    num_steps: u32,
) !void {
    const command_buffer = objc.msgSend(ctx.command_queue, objc.sel("commandBuffer"), .{});
    if (command_buffer == null) return error.CommandBufferFailed;

    const encoder = objc.msgSend(command_buffer, objc.sel("computeCommandEncoder"), .{});
    if (encoder == null) return error.EncoderFailed;

    const set_pipeline_sel = objc.sel("setComputePipelineState:");
    const set_buffer_sel = objc.sel("setBuffer:offset:atIndex:");
    const set_bytes_sel = objc.sel("setBytes:length:atIndex:");
    const dispatch_sel = objc.sel("dispatchThreads:threadsPerThreadgroup:");

    // Fused physics kernel - one thread per environment, does all physics work
    objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.fused_physics_pipeline});
    objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
    objc.msgSendVoid(encoder, set_buffer_sel, .{ action_buffer, @as(u64, 0), @as(u64, 1) });
    objc.msgSendVoid(encoder, set_buffer_sel, .{ constraint_buffer, @as(u64, 0), @as(u64, 2) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 3) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_joints_per_env, @as(u64, 4), @as(u64, 4) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &action_dim, @as(u64, 4), @as(u64, 5) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &dt, @as(u64, 4), @as(u64, 6) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &gravity, @as(u64, 12), @as(u64, 7) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &solver_iterations, @as(u64, 4), @as(u64, 8) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_steps, @as(u64, 4), @as(u64, 9) });

    const fused_max = objc.msgSendInt(u64, ctx.fused_physics_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const fused_tg = objc.MTLSize.make1D(@min(fused_max, @as(u64, num_envs)));
    const fused_grid = objc.MTLSize.make1D(@as(u64, num_envs));
    objc.msgSendVoid(encoder, dispatch_sel, .{ fused_grid, fused_tg });

    // Observation assembly
    objc.msgSendVoid(encoder, set_pipeline_sel, .{ctx.assemble_obs_pipeline});
    objc.msgSendVoid(encoder, set_buffer_sel, .{ body_buffer, @as(u64, 0), @as(u64, 0) });
    objc.msgSendVoid(encoder, set_buffer_sel, .{ obs_buffer, @as(u64, 0), @as(u64, 1) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_envs, @as(u64, 4), @as(u64, 2) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &num_bodies_per_env, @as(u64, 4), @as(u64, 3) });
    objc.msgSendVoid(encoder, set_bytes_sel, .{ &obs_dim, @as(u64, 4), @as(u64, 4) });

    const obs_max = objc.msgSendInt(u64, ctx.assemble_obs_pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const obs_tg = objc.MTLSize.make1D(@min(obs_max, @as(u64, num_envs)));
    const obs_grid = objc.MTLSize.make1D(@as(u64, num_envs));
    objc.msgSendVoid(encoder, dispatch_sel, .{ obs_grid, obs_tg });

    objc.msgSendVoid(encoder, objc.sel("endEncoding"), .{});
    objc.msgSendVoid(command_buffer, objc.sel("commit"), .{});
    objc.msgSendVoid(command_buffer, objc.sel("waitUntilCompleted"), .{});
}

fn dispatchThreads(encoder: objc.id, pipeline: objc.id, count: u32) void {
    // Get max threads per threadgroup
    const max_threads = objc.msgSendInt(u64, pipeline, objc.sel("maxTotalThreadsPerThreadgroup"), .{});
    const thread_width = @min(max_threads, @as(u64, count));

    const grid_size = objc.MTLSize.make1D(@as(u64, count));
    const threadgroup_size = objc.MTLSize.make1D(thread_width);

    objc.msgSendVoid(encoder, objc.sel("dispatchThreads:threadsPerThreadgroup:"), .{ grid_size, threadgroup_size });
}

fn printHeader() void {
    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║          Zeno Environment Benchmarks — Tech Spec 7.1 (Metal GPU)            ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("{s:<12} {s:>8} {s:>8} {s:>12} {s:>12} {s:>12} {s:>10}\n", .{
        "Environment",
        "Envs",
        "Steps",
        "Time (ms)",
        "Target (ms)",
        "Steps/sec",
        "vs Target",
    });
    std.debug.print("─" ** 78 ++ "\n", .{});
}

fn printResult(result: BenchResult) void {
    const status = if (result.speedup_vs_target >= 1.0) "✓" else "✗";
    std.debug.print("{s:<12} {d:>8} {d:>8} {d:>12.1} {d:>12.0} {d:>12.0} {d:>8.1}x {s}\n", .{
        result.env_name,
        result.num_envs,
        result.num_steps,
        result.total_time_ms,
        result.target_time_ms,
        result.steps_per_sec,
        result.speedup_vs_target,
        status,
    });
}

pub fn main() !void {
    // Initialize Metal context
    var ctx = MetalContext.init() catch |err| {
        std.debug.print("Failed to initialize Metal: {}\n", .{err});
        std.debug.print("Make sure you're running on Apple Silicon with Metal support.\n", .{});
        return;
    };
    defer ctx.deinit();

    // Get device name
    const device_name = objc.msgSend(ctx.device, objc.sel("name"), .{});
    const cstr = objc.msgSend(device_name, objc.sel("UTF8String"), .{});
    if (cstr) |c| {
        const cstr_ptr: [*:0]const u8 = @ptrCast(c);
        std.debug.print("\nMetal Device: {s}\n", .{cstr_ptr});
    }

    printHeader();

    // Benchmark configurations from tech spec 7.1
    const benchmarks = [_]struct { config: EnvConfig, num_envs: u32, target_ms: f64 }{
        .{ .config = PENDULUM, .num_envs = 1024, .target_ms = 50 },
        .{ .config = CARTPOLE, .num_envs = 1024, .target_ms = 80 },
        .{ .config = ANT, .num_envs = 1024, .target_ms = 800 },
        .{ .config = HUMANOID, .num_envs = 1024, .target_ms = 2000 },
        .{ .config = ANT, .num_envs = 4096, .target_ms = 3000 },
        .{ .config = ANT, .num_envs = 16384, .target_ms = 10000 },
    };

    const num_steps = 1000;

    for (benchmarks) |bench| {
        const result = runBenchmark(
            &ctx,
            bench.config,
            bench.num_envs,
            num_steps,
            bench.target_ms,
        ) catch |err| {
            std.debug.print("{s:<12} {d:>8} {d:>8} FAILED: {}\n", .{
                bench.config.name,
                bench.num_envs,
                num_steps,
                err,
            });
            continue;
        };
        printResult(result);
    }

    std.debug.print("─" ** 78 ++ "\n", .{});
    std.debug.print("\nNote: Benchmarks use actual Metal GPU compute shaders.\n", .{});
    std.debug.print("Target times are from tech spec 7.1 (MuJoCo baseline comparisons).\n", .{});
}
