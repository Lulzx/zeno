#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Bandwidth Microbenchmarks for M4 Pro
// Target: Measure achievable bandwidth with physics-shaped access patterns
// =============================================================================

// Pattern 1: Sequential read (theoretical max)
kernel void sequential_read(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    float4 sum = 0;
    uint base = tid * 256;  // Each thread reads 256 float4s = 4KB

    for (uint i = 0; i < 256; i++) {
        sum += input[base + i];
    }

    // Prevent optimization, single write per thread
    if (sum.x != 0 || sum.y != 0 || sum.z != 0 || sum.w != 0) {
        atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    }
    output[tid] = sum;
}

// Pattern 2: Strided read (SoA body state pattern)
// Simulates reading position[env_id * num_bodies + body_id] across envs
kernel void strided_read(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant uint& stride [[buffer(3)]],  // num_bodies, typically 9-14
    uint tid [[thread_position_in_grid]]
) {
    float4 sum = 0;
    uint base = tid;

    // Read 256 elements with stride (simulates iterating over envs for one body)
    for (uint i = 0; i < 256; i++) {
        sum += input[base + i * stride];
    }

    if (sum.x != 0 || sum.y != 0 || sum.z != 0 || sum.w != 0) {
        atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    }
    output[tid] = sum;
}

// Pattern 3: Gather (contact pair lookups)
// Simulates reading body states for contact pairs: body_a[contact.idx_a], body_b[contact.idx_b]
kernel void gather_read(
    device const float4* bodies [[buffer(0)]],
    device const uint2* pairs [[buffer(1)]],  // (idx_a, idx_b) pairs
    device float4* output [[buffer(2)]],
    device atomic_uint* counter [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float4 sum = 0;

    // Each thread processes 128 contact pairs
    for (uint i = 0; i < 128; i++) {
        uint2 pair = pairs[tid * 128 + i];
        sum += bodies[pair.x];
        sum += bodies[pair.y];
    }

    if (sum.x != 0 || sum.y != 0 || sum.z != 0 || sum.w != 0) {
        atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    }
    output[tid] = sum;
}

// Pattern 4: Read-modify-write (PBD solver pattern)
// Simulates: read two bodies, compute impulse, write both back
kernel void solver_pattern(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device const uint2* pairs [[buffer(2)]],
    device const float4* impulses [[buffer(3)]],  // precomputed for benchmark
    device atomic_uint* counter [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one contact
    uint2 pair = pairs[tid];
    float4 impulse = impulses[tid];

    // Read
    float4 pos_a = positions[pair.x];
    float4 pos_b = positions[pair.y];
    float4 vel_a = velocities[pair.x];
    float4 vel_b = velocities[pair.y];

    // "Compute" (minimal to measure memory, not ALU)
    pos_a += impulse * 0.5f;
    pos_b -= impulse * 0.5f;
    vel_a += impulse;
    vel_b -= impulse;

    // Write (atomic to handle multiple contacts per body)
    // In real solver we'd use atomics or coloring - here we just measure
    positions[pair.x] = pos_a;
    positions[pair.y] = pos_b;
    velocities[pair.x] = vel_a;
    velocities[pair.y] = vel_b;

    atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
}

// Pattern 5: Coalesced write (observation assembly)
kernel void sequential_write(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Read one, write many (simulates assembling obs from scattered state)
    float4 val = input[tid];
    uint base = tid * 64;

    for (uint i = 0; i < 64; i++) {
        output[base + i] = val + float4(i);
    }
}

// Pattern 6: Threadgroup shared memory (for reduction patterns)
kernel void shared_memory_test(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    threadgroup float4* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Load to shared
    shared[lid] = input[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Single write per threadgroup
    if (lid == 0) {
        output[gid] = shared[0];
    }
}
