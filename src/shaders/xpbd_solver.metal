#include <metal_stdlib>
using namespace metal;

// =============================================================================
// XPBD Constraint Solver - Optimized for M4 Pro
// =============================================================================
// Key optimizations:
// - Unified constraint format for joints and contacts
// - Fused kernels to reduce memory traffic
// - Coalesced memory access patterns
// - Minimal divergence in constraint solve
// =============================================================================

// Constraint types
constant uint CONSTRAINT_CONTACT_NORMAL = 0;
constant uint CONSTRAINT_CONTACT_FRICTION = 1;
constant uint CONSTRAINT_POSITIONAL = 2;
constant uint CONSTRAINT_ANGULAR = 3;
constant uint CONSTRAINT_ANGULAR_LIMIT = 4;
constant uint CONSTRAINT_LINEAR_LIMIT = 5;

// GPU constraint structure (96 bytes, aligned)
struct XPBDConstraint {
    uint4 indices;          // body_a, body_b, env_id, type
    float4 anchor_a;        // local_a.xyz, compliance
    float4 anchor_b;        // local_b.xyz, damping
    float4 axis_target;     // axis.xyz, target
    float4 limits;          // lower, upper, friction, restitution
    float4 state;           // lambda, lambda_prev, violation, effective_mass
};

// Body state (position + quaternion + velocity in SoA)
struct BodyState {
    float4 position;        // xyz + inv_mass
    float4 quaternion;      // xyzw
    float4 velocity;        // xyz + unused
    float4 angular_vel;     // xyz + unused
    float4 inv_inertia;     // xyz (diagonal) + unused
};

// Solver parameters
struct SolverParams {
    uint num_envs;
    uint max_constraints;
    uint num_bodies;
    uint iteration;

    float dt;
    float inv_dt;
    float inv_dt_sq;
    float relaxation;

    float4 gravity;
};

// =============================================================================
// Math utilities
// =============================================================================

inline float3 quat_rotate(float4 q, float3 v) {
    float3 u = q.xyz;
    float s = q.w;
    return 2.0f * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0f * s * cross(u, v);
}

inline float4 quat_mul(float4 a, float4 b) {
    return float4(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz)
    );
}

inline float4 quat_conjugate(float4 q) {
    return float4(-q.xyz, q.w);
}

inline float3 quat_to_angular(float4 q0, float4 q1, float dt) {
    // Compute angular velocity from quaternion difference
    float4 dq = quat_mul(q1, quat_conjugate(q0));
    if (dq.w < 0) dq = -dq;
    return 2.0f * dq.xyz / dt;
}

// =============================================================================
// Fused Integration Kernel
// Combines: apply_forces + integrate_positions + integrate_velocities
// =============================================================================

kernel void fused_integrate(
    device BodyState* bodies [[buffer(0)]],
    device const float* joint_torques [[buffer(1)]],
    device const uint* joint_to_body [[buffer(2)]],
    constant SolverParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint env_id = tid / params.num_bodies;
    uint body_id = tid % params.num_bodies;

    if (env_id >= params.num_envs) return;

    uint idx = env_id * params.num_bodies + body_id;
    BodyState body = bodies[idx];

    float inv_mass = body.position.w;
    float3 pos = body.position.xyz;
    float4 quat = body.quaternion;
    float3 vel = body.velocity.xyz;
    float3 omega = body.angular_vel.xyz;
    float3 inv_I = body.inv_inertia.xyz;

    // Skip static bodies
    if (inv_mass < 1e-8f) return;

    // Apply gravity
    float3 force = params.gravity.xyz / inv_mass;  // F = m * g

    // Apply joint torques (accumulated from actuators)
    // This is done in a separate pass for efficiency

    // Semi-implicit Euler integration
    // v(t+dt) = v(t) + a * dt
    vel += force * inv_mass * params.dt;

    // x(t+dt) = x(t) + v(t+dt) * dt
    pos += vel * params.dt;

    // Angular integration using quaternion exponential
    // q(t+dt) = q(t) + 0.5 * ω_quat * q(t) * dt
    float4 omega_quat = float4(omega, 0.0f);
    quat += 0.5f * quat_mul(omega_quat, quat) * params.dt;
    quat = normalize(quat);

    // Write back
    bodies[idx].position = float4(pos, inv_mass);
    bodies[idx].quaternion = quat;
    bodies[idx].velocity = float4(vel, 0);
    bodies[idx].angular_vel = float4(omega, 0);
}

// =============================================================================
// XPBD Constraint Solve Kernel
// Unified solver for all constraint types
// =============================================================================

kernel void xpbd_solve_constraints(
    device BodyState* bodies [[buffer(0)]],
    device XPBDConstraint* constraints [[buffer(1)]],
    device const uint* constraint_counts [[buffer(2)]],
    constant SolverParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint env_id = tid / params.max_constraints;
    uint constraint_id = tid % params.max_constraints;

    if (env_id >= params.num_envs) return;

    // Check if this constraint is active
    uint count = constraint_counts[env_id];
    if (constraint_id >= count) return;

    uint cidx = env_id * params.max_constraints + constraint_id;
    XPBDConstraint c = constraints[cidx];

    uint body_a = c.indices.x;
    uint body_b = c.indices.y;
    uint constraint_type = c.indices.w;

    // Get body state
    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;

    BodyState state_a = bodies[idx_a];
    BodyState state_b = bodies[idx_b];

    float inv_mass_a = state_a.position.w;
    float inv_mass_b = state_b.position.w;

    float3 pos_a = state_a.position.xyz;
    float3 pos_b = state_b.position.xyz;
    float4 quat_a = state_a.quaternion;
    float4 quat_b = state_b.quaternion;

    // Compute constraint-specific values
    float violation = 0.0f;
    float3 gradient = float3(0);
    float3 r_a = float3(0);
    float3 r_b = float3(0);

    switch (constraint_type) {
        case CONSTRAINT_CONTACT_NORMAL: {
            // Contact non-penetration
            // C = -penetration (we want C >= 0)
            violation = c.state.z;  // penetration stored here
            gradient = c.axis_target.xyz;  // normal

            // World-space contact points
            r_a = quat_rotate(quat_a, c.anchor_a.xyz);
            r_b = quat_rotate(quat_b, c.anchor_b.xyz);
            break;
        }

        case CONSTRAINT_POSITIONAL: {
            // Point-to-point constraint
            // C = |world_a - world_b|
            float3 world_a = pos_a + quat_rotate(quat_a, c.anchor_a.xyz);
            float3 world_b = pos_b + quat_rotate(quat_b, c.anchor_b.xyz);

            float3 diff = world_a - world_b;
            float dist = length(diff);

            if (dist > 1e-6f) {
                violation = dist;
                gradient = diff / dist;
                r_a = quat_rotate(quat_a, c.anchor_a.xyz);
                r_b = quat_rotate(quat_b, c.anchor_b.xyz);
            }
            break;
        }

        case CONSTRAINT_ANGULAR: {
            // Angular alignment constraint
            // C = |axis_a_world × axis_b_world|
            float3 axis_a_world = quat_rotate(quat_a, c.anchor_a.xyz);
            float3 axis_b_world = quat_rotate(quat_b, c.anchor_b.xyz);

            float3 cross_axis = cross(axis_a_world, axis_b_world);
            violation = length(cross_axis);

            if (violation > 1e-6f) {
                gradient = cross_axis / violation;
            }
            break;
        }

        case CONSTRAINT_ANGULAR_LIMIT: {
            // Angular limit constraint
            float3 axis_world = quat_rotate(quat_a, c.anchor_a.xyz);

            // Compute relative rotation around axis
            float4 rel_quat = quat_mul(quat_conjugate(quat_a), quat_b);
            float angle = 2.0f * atan2(dot(rel_quat.xyz, axis_world), rel_quat.w);

            float lower = c.limits.x;
            float upper = c.limits.y;

            if (angle < lower) {
                violation = lower - angle;
                gradient = axis_world;
            } else if (angle > upper) {
                violation = angle - upper;
                gradient = -axis_world;
            }
            break;
        }

        default:
            return;
    }

    // Skip if no violation
    if (abs(violation) < 1e-8f) return;

    // Compute effective mass
    // w = m_a^-1 + m_b^-1 + (r_a × n)ᵀ I_a^-1 (r_a × n) + ...
    float3 rn_a = cross(r_a, gradient);
    float3 rn_b = cross(r_b, gradient);

    float angular_a = dot(rn_a * rn_a, state_a.inv_inertia.xyz);
    float angular_b = dot(rn_b * rn_b, state_b.inv_inertia.xyz);

    float w = inv_mass_a + inv_mass_b + angular_a + angular_b;

    if (w < 1e-8f) return;

    // XPBD solve: λ = (-C - α̃ * λ_prev) / (w + α̃)
    float compliance = c.anchor_a.w;
    float alpha_tilde = compliance * params.inv_dt_sq;

    float lambda_prev = (params.iteration == 0) ? 0.0f : c.state.x;
    float delta_lambda = (-violation - alpha_tilde * lambda_prev) / (w + alpha_tilde);

    // For contacts, clamp lambda >= 0 (no pulling)
    if (constraint_type == CONSTRAINT_CONTACT_NORMAL) {
        float new_lambda = lambda_prev + delta_lambda;
        if (new_lambda < 0.0f) {
            delta_lambda = -lambda_prev;
        }
    }

    // Apply relaxation
    delta_lambda *= params.relaxation;

    // Store accumulated lambda
    constraints[cidx].state.x = lambda_prev + delta_lambda;

    // Apply position correction
    float3 impulse = gradient * delta_lambda;

    float3 delta_pos_a = impulse * inv_mass_a;
    float3 delta_pos_b = -impulse * inv_mass_b;

    // Angular correction
    float3 delta_omega_a = rn_a * delta_lambda * state_a.inv_inertia.xyz;
    float3 delta_omega_b = -rn_b * delta_lambda * state_b.inv_inertia.xyz;

    // Atomic add to positions (handles multiple constraints per body)
    // Note: In practice, use graph coloring to avoid atomics
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.x, delta_pos_a.x, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.y, delta_pos_a.y, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.z, delta_pos_a.z, memory_order_relaxed);

    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.x, delta_pos_b.x, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.y, delta_pos_b.y, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.z, delta_pos_b.z, memory_order_relaxed);
}

// =============================================================================
// Velocity Update Kernel
// Updates velocities from position changes (after all XPBD iterations)
// =============================================================================

kernel void xpbd_update_velocities(
    device BodyState* bodies [[buffer(0)]],
    device const float4* prev_positions [[buffer(1)]],
    device const float4* prev_quaternions [[buffer(2)]],
    constant SolverParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_envs * params.num_bodies) return;

    BodyState body = bodies[tid];
    float4 prev_pos = prev_positions[tid];
    float4 prev_quat = prev_quaternions[tid];

    float inv_mass = body.position.w;
    if (inv_mass < 1e-8f) return;

    // Update linear velocity from position change
    float3 delta_pos = body.position.xyz - prev_pos.xyz;
    body.velocity.xyz = delta_pos * params.inv_dt;

    // Update angular velocity from quaternion change
    float4 dq = quat_mul(body.quaternion, quat_conjugate(prev_quat));
    if (dq.w < 0) dq = -dq;
    body.angular_vel.xyz = 2.0f * dq.xyz * params.inv_dt;

    bodies[tid] = body;
}

// =============================================================================
// Contact Friction Kernel (after normal solve)
// =============================================================================

kernel void xpbd_solve_friction(
    device BodyState* bodies [[buffer(0)]],
    device XPBDConstraint* constraints [[buffer(1)]],
    device const uint* constraint_counts [[buffer(2)]],
    constant SolverParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint env_id = tid / params.max_constraints;
    uint constraint_id = tid % params.max_constraints;

    if (env_id >= params.num_envs) return;

    uint count = constraint_counts[env_id];
    if (constraint_id >= count) return;

    uint cidx = env_id * params.max_constraints + constraint_id;
    XPBDConstraint c = constraints[cidx];

    // Only process contact constraints
    if (c.indices.w != CONSTRAINT_CONTACT_NORMAL) return;

    float lambda_n = c.state.x;  // Normal impulse
    if (lambda_n <= 0.0f) return;  // No contact force, no friction

    uint body_a = c.indices.x;
    uint body_b = c.indices.y;

    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;

    BodyState state_a = bodies[idx_a];
    BodyState state_b = bodies[idx_b];

    float3 normal = c.axis_target.xyz;
    float friction = c.limits.z;

    // Compute relative velocity at contact
    float3 r_a = quat_rotate(state_a.quaternion, c.anchor_a.xyz);
    float3 r_b = quat_rotate(state_b.quaternion, c.anchor_b.xyz);

    float3 vel_a = state_a.velocity.xyz + cross(state_a.angular_vel.xyz, r_a);
    float3 vel_b = state_b.velocity.xyz + cross(state_b.angular_vel.xyz, r_b);
    float3 rel_vel = vel_a - vel_b;

    // Tangent velocity (remove normal component)
    float vn = dot(rel_vel, normal);
    float3 vt = rel_vel - vn * normal;
    float vt_len = length(vt);

    if (vt_len < 1e-6f) return;

    float3 tangent = vt / vt_len;

    // Effective mass for tangent direction
    float3 rn_a = cross(r_a, tangent);
    float3 rn_b = cross(r_b, tangent);

    float w = state_a.position.w + state_b.position.w +
              dot(rn_a * rn_a, state_a.inv_inertia.xyz) +
              dot(rn_b * rn_b, state_b.inv_inertia.xyz);

    if (w < 1e-8f) return;

    // Friction impulse (Coulomb model with clamping)
    float max_friction = friction * lambda_n;
    float delta_lambda_t = -vt_len / w;
    delta_lambda_t = clamp(delta_lambda_t, -max_friction, max_friction);

    // Apply friction correction
    float3 impulse = tangent * delta_lambda_t * params.inv_dt;

    // Position correction for friction
    float3 delta_a = impulse * state_a.position.w * params.dt;
    float3 delta_b = -impulse * state_b.position.w * params.dt;

    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.x, delta_a.x, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.y, delta_a.y, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_a].position.z, delta_a.z, memory_order_relaxed);

    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.x, delta_b.x, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.y, delta_b.y, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_float*)&bodies[idx_b].position.z, delta_b.z, memory_order_relaxed);
}

// =============================================================================
// Fused Observation Assembly
// Reads body state + joint state into observation buffer
// =============================================================================

kernel void assemble_observations(
    device const BodyState* bodies [[buffer(0)]],
    device const float* joint_positions [[buffer(1)]],
    device const float* joint_velocities [[buffer(2)]],
    device float* observations [[buffer(3)]],
    constant uint& num_bodies [[buffer(4)]],
    constant uint& num_joints [[buffer(5)]],
    constant uint& obs_dim [[buffer(6)]],
    uint env_id [[thread_position_in_grid]]
) {
    uint obs_offset = env_id * obs_dim;
    uint body_offset = env_id * num_bodies;
    uint joint_offset = env_id * num_joints;

    uint idx = 0;

    // Root body state (position, quaternion, velocity, angular velocity)
    BodyState root = bodies[body_offset];
    observations[obs_offset + idx++] = root.position.x;
    observations[obs_offset + idx++] = root.position.y;
    observations[obs_offset + idx++] = root.position.z;
    observations[obs_offset + idx++] = root.quaternion.x;
    observations[obs_offset + idx++] = root.quaternion.y;
    observations[obs_offset + idx++] = root.quaternion.z;
    observations[obs_offset + idx++] = root.quaternion.w;
    observations[obs_offset + idx++] = root.velocity.x;
    observations[obs_offset + idx++] = root.velocity.y;
    observations[obs_offset + idx++] = root.velocity.z;
    observations[obs_offset + idx++] = root.angular_vel.x;
    observations[obs_offset + idx++] = root.angular_vel.y;
    observations[obs_offset + idx++] = root.angular_vel.z;

    // Joint positions and velocities
    for (uint j = 0; j < num_joints; j++) {
        observations[obs_offset + idx++] = joint_positions[joint_offset + j];
    }
    for (uint j = 0; j < num_joints; j++) {
        observations[obs_offset + idx++] = joint_velocities[joint_offset + j];
    }
}
