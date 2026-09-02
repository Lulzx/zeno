//
// Zeno Physics Engine - Metal Compute Shaders
// GPU-accelerated batched rigid body simulation
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Common Types and Utilities
// ============================================================================

struct SimParams {
    uint num_envs;
    uint num_bodies;
    uint num_joints;
    uint num_actuators;
    uint num_geoms;
    uint num_sensors;
    uint max_contacts;
    uint contact_iterations;
    float dt;
    float gravity_x;
    float gravity_y;
    float gravity_z;
    float friction;
    float restitution;
    float baumgarte;
    float slop;
    uint target_color;      // For constraint graph coloring (solve_joints)
    uint num_constraints;   // Total constraints to process
    uint constraint_offset; // Offset into constraint buffer for current color
    uint obs_dim;           // Observation dimension per environment
};

struct EnvDispatchParams {
    uint num_envs;
    uint dispatch_envs;
    uint use_active_env_ids;
    uint _pad;
};

struct TaskParams {
    uint enabled;
    uint root_body;
    uint forward_axis;
    uint max_episode_steps;
    float forward_reward_weight;
    float control_cost_weight;
    float healthy_bonus;
    float healthy_z_min;
    float healthy_z_max;
    uint terminate_when_unhealthy;
    uint _pad0;
    uint _pad1;
};

inline uint physical_env_id(
    uint dispatch_env_id,
    constant EnvDispatchParams& dispatch,
    device const uint* active_env_ids
) {
    return dispatch.use_active_env_ids != 0
        ? active_env_ids[dispatch_env_id]
        : dispatch_env_id;
}

struct BodyData {
    float4 position;
    float4 quaternion;
    float4 inv_mass_inertia;
    float4 params;
    float4 com_offset;   // center of mass offset from body frame origin (local coords)
};

struct JointData {
    float4 anchor_parent;
    float4 anchor_child;
    float4 axis;
    float4 params;
    float4 params2;
};

struct GeomData {
    uint4 type_body;
    float4 pos_size0;
    float4 quat;
    float4 params;
};

struct ActuatorData {
    float4 params;
    float4 params2;
};

struct SensorData {
    uint4 type_object;
    float4 params;
};

struct Contact {
    float4 position_pen;
    float4 normal_friction;
    uint4 indices;
    float4 impulses;
};

// Quaternion operations
float4 quat_multiply(float4 a, float4 b) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

float4 quat_conjugate(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

float3 rotate_by_quat(float3 v, float4 q) {
    float3 qv = float3(q.x, q.y, q.z);
    float3 uv = cross(qv, v);
    float3 uuv = cross(qv, uv);
    return v + 2.0 * (q.w * uv + uuv);
}

float4 quat_normalize(float4 q) {
    float len = length(q);
    return len > 1e-8 ? q / len : float4(0, 0, 0, 1);
}

float3 geom_world_position(float4 body_position, float4 body_quat, GeomData geom) {
    return body_position.xyz + rotate_by_quat(geom.pos_size0.xyz, body_quat);
}

float4 geom_world_quat(float4 body_quat, GeomData geom) {
    return quat_normalize(quat_multiply(body_quat, geom.quat));
}

float geom_bounding_radius(GeomData geom) {
    uint type = geom.type_body.x;
    float x = geom.pos_size0.w;
    if (type == 0) return x;                         // sphere
    if (type == 1) return x + geom.params.x;         // capsule
    if (type == 2) return length(float3(x, geom.params.x, geom.params.y)); // box
    if (type == 4) return length(float2(x, geom.params.x)); // cylinder
    return x; // mesh/heightfield placeholders; planes bypass finite tests
}

float3 contact_tangent(float3 normal) {
    float3 reference = abs(normal.z) < 0.9f
        ? float3(0, 0, 1)
        : float3(0, 1, 0);
    return normalize(cross(reference, normal));
}

float3 closest_point_on_segment(float3 point, float3 a, float3 b) {
    float3 ab = b - a;
    float denom = dot(ab, ab);
    if (denom < 1e-12f) return a;
    float t = clamp(dot(point - a, ab) / denom, 0.0f, 1.0f);
    return a + t * ab;
}

struct SegmentClosestPoints {
    float3 a;
    float3 b;
};

SegmentClosestPoints closest_points_on_segments(
    float3 p1, float3 q1, float3 p2, float3 q2
) {
    float3 d1 = q1 - p1;
    float3 d2 = q2 - p2;
    float3 r = p1 - p2;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= 1e-12f && e <= 1e-12f) {
        return {p1, p2};
    }
    if (a <= 1e-12f) {
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= 1e-12f) {
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (abs(denom) > 1e-12f) s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    return {p1 + d1 * s, p2 + d2 * t};
}

struct SegmentBoxClosestPoints {
    float3 segment;
    float3 box;
    float distance_sq;
};

float3 nearest_aabb_face_normal(float3 point, float3 extents) {
    float3 clearance = extents - abs(point);
    uint axis = clearance.y < clearance.x ? 1u : 0u;
    if (clearance.z < clearance[axis]) axis = 2u;
    float3 normal = float3(0.0f);
    normal[axis] = point[axis] >= 0.0f ? 1.0f : -1.0f;
    return normal;
}

// Exact closest points between a segment and an axis-aligned box. Squared
// distance is piecewise quadratic in segment t; slab crossings partition the
// pieces, then each interval's analytic minimum is evaluated.
SegmentBoxClosestPoints closest_points_segment_aabb(
    float3 start, float3 end, float3 extents
) {
    float3 direction = end - start;
    float breaks[8];
    uint break_count = 2;
    breaks[0] = 0.0f;
    breaks[1] = 1.0f;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) <= 1e-12f) continue;
        float low_t = (-extents[axis] - start[axis]) / direction[axis];
        float high_t = (extents[axis] - start[axis]) / direction[axis];
        if (low_t > 0.0f && low_t < 1.0f) breaks[break_count++] = low_t;
        if (high_t > 0.0f && high_t < 1.0f) breaks[break_count++] = high_t;
    }
    for (uint i = 1; i < break_count; ++i) {
        float key = breaks[i];
        int j = int(i) - 1;
        while (j >= 0 && breaks[uint(j)] > key) {
            breaks[uint(j) + 1] = breaks[uint(j)];
            --j;
        }
        breaks[uint(j + 1)] = key;
    }

    SegmentBoxClosestPoints best;
    best.segment = start;
    best.box = clamp(start, -extents, extents);
    best.distance_sq = dot(best.segment - best.box, best.segment - best.box);

    for (uint i = 0; i < break_count; ++i) {
        float t = breaks[i];
        float3 segment_point = start + direction * t;
        float3 box_point = clamp(segment_point, -extents, extents);
        float distance_sq = dot(segment_point - box_point, segment_point - box_point);
        if (distance_sq < best.distance_sq) {
            best = {segment_point, box_point, distance_sq};
        }
    }

    for (uint i = 0; i + 1 < break_count; ++i) {
        float lo = breaks[i];
        float hi = breaks[i + 1];
        if (hi - lo <= 1e-12f) continue;
        float3 midpoint = start + direction * ((lo + hi) * 0.5f);
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (uint axis = 0; axis < 3; ++axis) {
            float bound;
            if (midpoint[axis] < -extents[axis]) {
                bound = -extents[axis];
            } else if (midpoint[axis] > extents[axis]) {
                bound = extents[axis];
            } else {
                continue;
            }
            numerator += direction[axis] * (start[axis] - bound);
            denominator += direction[axis] * direction[axis];
        }
        if (denominator <= 1e-12f) continue;
        float t = clamp(-numerator / denominator, lo, hi);
        float3 segment_point = start + direction * t;
        float3 box_point = clamp(segment_point, -extents, extents);
        float distance_sq = dot(segment_point - box_point, segment_point - box_point);
        if (distance_sq < best.distance_sq) {
            best = {segment_point, box_point, distance_sq};
        }
    }
    return best;
}

// Atomic float add helper
void atomic_add_float(device float* address, float val) {
    device atomic_uint* atom = (device atomic_uint*)address;
    uint old = atomic_load_explicit(atom, memory_order_relaxed);
    uint expected = old;
    while (true) {
        float f_old = as_type<float>(old);
        float f_new = f_old + val;
        uint u_new = as_type<uint>(f_new);
        if (atomic_compare_exchange_weak_explicit(atom, &expected, u_new, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        old = expected;
    }
}

// Atomic float3 add into the xyz components of a float4 buffer entry.
// Used by kernels where multiple threads may update the same body (e.g. one
// thread per contact, several contacts sharing a body).
void atomic_add_float3(device float4* target, float3 val) {
    device float* p = (device float*)target;
    atomic_add_float(p + 0, val.x);
    atomic_add_float(p + 1, val.y);
    atomic_add_float(p + 2, val.z);
}

// ============================================================================
// Apply Joint Forces Kernel
// ============================================================================

kernel void apply_joint_forces(
    device float4* torques [[buffer(0)]],
    device const float* joint_torques [[buffer(1)]],
    device const JointData* joints [[buffer(2)]],
    device const float4* quaternions [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    device float4* forces [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_joints;
    uint joint_id = gid % params.num_joints;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    uint joint_idx = env_id * params.num_joints + joint_id;
    JointData joint = joints[joint_id];

    float applied_force = joint_torques[joint_idx];

    if (abs(applied_force) < 1e-6) return;

    uint joint_type = uint(joint.params.x);
    uint body_a = uint(joint.params.y);
    uint body_b = uint(joint.params.z);

    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;

    float4 q_a = quaternions[idx_a];

    float3 axis_local = joint.axis.xyz;
    float3 axis_world = rotate_by_quat(axis_local, q_a);

    if (joint_type == 2) {
        // Prismatic/slide joint: apply linear force along axis
        float3 force_world = axis_world * applied_force;

        if (body_a > 0) {
            device float* ptr = (device float*)&forces[idx_a];
            atomic_add_float(ptr + 0, -force_world.x);
            atomic_add_float(ptr + 1, -force_world.y);
            atomic_add_float(ptr + 2, -force_world.z);
        }

        if (body_b > 0) {
            device float* ptr = (device float*)&forces[idx_b];
            atomic_add_float(ptr + 0, force_world.x);
            atomic_add_float(ptr + 1, force_world.y);
            atomic_add_float(ptr + 2, force_world.z);
        }
    } else {
        // Revolute/hinge and other joints: apply torque
        float3 torque_world = axis_world * applied_force;

        if (body_a > 0) {
            device float* ptr = (device float*)&torques[idx_a];
            atomic_add_float(ptr + 0, -torque_world.x);
            atomic_add_float(ptr + 1, -torque_world.y);
            atomic_add_float(ptr + 2, -torque_world.z);
        }

        if (body_b > 0) {
            device float* ptr = (device float*)&torques[idx_b];
            atomic_add_float(ptr + 0, torque_world.x);
            atomic_add_float(ptr + 1, torque_world.y);
            atomic_add_float(ptr + 2, torque_world.z);
        }
    }
}

// ============================================================================
// Apply Actions Kernel
// ============================================================================

kernel void apply_actions(
    device const float* actions [[buffer(0)]],
    device float* joint_torques [[buffer(1)]],
    device const ActuatorData* actuators [[buffer(2)]],
    constant SimParams& params [[buffer(3)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_actuators = params.num_actuators;
    if (num_actuators == 0) return;

    uint dispatch_env_id = gid / num_actuators;
    uint act_id = gid % num_actuators;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    ActuatorData act = actuators[act_id];
    uint joint_id = uint(act.params.x);
    float ctrl_min = act.params.y;
    float ctrl_max = act.params.z;
    float gear = act.params.w;

    float ctrl = actions[env_id * num_actuators + act_id];
    ctrl = clamp(ctrl, ctrl_min, ctrl_max);

    float torque = ctrl * gear;

    // Apply force limits
    float force_min = act.params2.x;
    float force_max = act.params2.y;
    torque = clamp(torque, force_min, force_max);

    uint joint_idx = env_id * params.num_joints + joint_id;
    joint_torques[joint_idx] = torque;
}

// ============================================================================
// Forward Kinematics Kernel
// ============================================================================

kernel void forward_kinematics(
    device float4* positions [[buffer(0)]],
    device float4* quaternions [[buffer(1)]],
    device const float* joint_positions [[buffer(2)]],
    device const JointData* joints [[buffer(3)]],
    device const BodyData* bodies [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    BodyData body = bodies[body_id];
    int parent_id = int(body.params.x);

    if (parent_id < 0) {
        // Root body - position is absolute
        return;
    }

    // Find joint connecting this body to parent
    for (uint j = 0; j < params.num_joints; j++) {
        JointData joint = joints[j];
        uint child_body = uint(joint.params.z);

        if (child_body == body_id) {
            uint parent_body = uint(joint.params.y);
            uint joint_type = uint(joint.params.x);

            uint parent_idx = env_id * params.num_bodies + parent_body;
            uint joint_idx = env_id * params.num_joints + j;

            float4 parent_pos = positions[parent_idx];
            float4 parent_quat = quaternions[parent_idx];
            float3 anchor = joint.anchor_parent.xyz;
            float3 axis = joint.axis.xyz;

            // Transform anchor to world space
            float3 world_anchor = rotate_by_quat(anchor, parent_quat) + parent_pos.xyz;

            // Compute child orientation based on joint type
            float4 child_quat = parent_quat;

            if (joint_type == 1) { // Revolute/hinge
                float angle = joint_positions[joint_idx];
                float half_angle = angle * 0.5;
                float s = sin(half_angle);
                float c = cos(half_angle);
                float4 rot_quat = float4(axis * s, c);
                child_quat = quat_multiply(parent_quat, rot_quat);
            }

            // Set child position and orientation
            positions[gid] = float4(world_anchor, 0);
            quaternions[gid] = quat_normalize(child_quat);
            break;
        }
    }
}

// ============================================================================
// Update Kinematic Bodies Kernel
// ============================================================================
// Kinematic bodies follow their specified velocities but aren't affected by forces.
// They have infinite mass (inv_mass = 0) so the integrate kernel skips them.
// This kernel updates their positions from their velocities.

kernel void update_kinematic(
    device float4* positions [[buffer(0)]],
    device const float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device const float4* angular_velocities [[buffer(3)]],
    device const BodyData* bodies [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    BodyData body = bodies[body_id];
    uint body_type = uint(body.params.y);

    // Only process kinematic bodies (type 1)
    if (body_type != 1) return;

    float dt = params.dt;

    // Linear position update: p = p + v * dt
    float3 vel = velocities[gid].xyz;
    float3 pos = positions[gid].xyz;
    pos += vel * dt;
    positions[gid] = float4(pos, 0);

    // Angular position update: q = q + 0.5 * ω_quat * q * dt
    float3 omega = angular_velocities[gid].xyz;
    float4 quat = quaternions[gid];

    float4 omega_quat = float4(omega * dt * 0.5, 0);
    float4 dq = quat_multiply(omega_quat, quat);
    quaternions[gid] = quat_normalize(quat + dq);
}

// ============================================================================
// Compute Forces Kernel
// ============================================================================

kernel void compute_forces(
    device const float4* positions [[buffer(0)]],
    device const float4* velocities [[buffer(1)]],
    device float4* forces [[buffer(2)]],
    device float4* torques [[buffer(3)]],
    device const float* joint_torques [[buffer(4)]],
    device const float4* inv_mass_inertia [[buffer(5)]],
    constant SimParams& params [[buffer(6)]],
    device const float4* quaternions [[buffer(7)]],
    device const BodyData* body_data [[buffer(8)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    float4 inv_mi = inv_mass_inertia[gid];
    float inv_mass = inv_mi.x;

    // Static bodies don't move
    if (inv_mass < 1e-8) {
        forces[gid] = float4(0);
        torques[gid] = float4(0);
        return;
    }

    float mass = 1.0 / inv_mass;

    // Gravity (add to existing forces from apply_joint_forces for prismatic joints)
    float gravity_scale = body_data[body_id].params.z;
    float3 gravity = float3(params.gravity_x, params.gravity_y, params.gravity_z) * gravity_scale;
    float3 force = gravity * mass;

    forces[gid] = float4(force, 0);

    // Gravitational torque from COM offset.
    // When the body frame origin differs from the center of mass,
    // gravity acting at the COM creates a torque about the body origin:
    // τ = r_com_world × (m * g)
    float3 com_local = body_data[body_id].com_offset.xyz;
    float3 com_world = rotate_by_quat(com_local, quaternions[gid]);
    float3 grav_torque = cross(com_world, force);
    torques[gid] = float4(grav_torque, 0);
}

// ============================================================================
// Integration Kernel (Semi-implicit Euler)
// ============================================================================

kernel void integrate(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device const float4* forces [[buffer(4)]],
    device const float4* torques [[buffer(5)]],
    device const float4* inv_mass_inertia [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    device const BodyData* body_data [[buffer(8)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    float4 inv_mi = inv_mass_inertia[gid];
    float inv_mass = inv_mi.x;

    // Static bodies don't move
    if (inv_mass < 1e-8) return;

    float dt = params.dt;

    // Linear integration
    float3 force = forces[gid].xyz;
    float3 vel = velocities[gid].xyz;
    float3 pos = positions[gid].xyz;

    // v(t+dt) = v(t) + a(t) * dt
    float3 accel = force * inv_mass;
    vel += accel * dt;

    // Per-body damping is expressed as a rate, so its effect is stable across
    // timestep/substep changes. Clamp malformed negative values to zero.
    float linear_damping = max(body_data[body_id].params.w, 0.0f);
    vel *= max(0.0f, 1.0f - linear_damping * dt);

    // x(t+dt) = x(t) + v(t+dt) * dt
    pos += vel * dt;

    velocities[gid] = float4(vel, 0);
    positions[gid] = float4(pos, 0);

    // Angular integration
    float3 inv_inertia = inv_mi.yzw;
    float3 torque = torques[gid].xyz;
    float3 omega = angular_velocities[gid].xyz;
    float4 quat = quaternions[gid];

    // ω(t+dt) = ω(t) + I⁻¹ * τ * dt
    omega += inv_inertia * torque * dt;
    float angular_damping = max(body_data[body_id].com_offset.w, 0.0f);
    omega *= max(0.0f, 1.0f - angular_damping * dt);

    // Quaternion integration: q(t+dt) = q(t) + 0.5 * ω_quat * q(t) * dt
    float4 omega_quat = float4(omega * dt * 0.5, 0);
    float4 dq = quat_multiply(omega_quat, quat);
    quat = quat + dq;

    // Explicit quaternion renormalization to prevent drift accumulation.
    // This is critical for long simulations where floating-point errors
    // in the quaternion can compound, causing non-unit quaternions that
    // distort rotations and destabilize the simulation.
    float qlen = length(quat);
    if (qlen > 1e-8) {
        quat = quat / qlen;
    } else {
        quat = float4(0, 0, 0, 1); // Reset to identity on degenerate quaternion
    }

    angular_velocities[gid] = float4(omega, 0);
    quaternions[gid] = quat;
}

// ============================================================================
// Broad Phase Collision Detection
// ============================================================================

// Clear a uint buffer on the GPU. Needed because CPU-side memsets execute at
// encode time, before any substep runs — counts must be reset between substeps
// inside the command buffer.
kernel void clear_uint_buffer(
    device uint* buf [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) buf[gid] = 0;
}

kernel void clear_active_uint_slices(
    device uint* buffer [[buffer(0)]],
    constant uint& elements_per_env [[buffer(1)]],
    constant EnvDispatchParams& dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / elements_per_env;
    if (dispatch_env_id >= dispatch.dispatch_envs) return;
    uint element_id = gid % elements_per_env;
    uint env_id = physical_env_id(dispatch_env_id, dispatch, active_env_ids);
    buffer[env_id * elements_per_env + element_id] = 0;
}

kernel void broad_phase(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device const GeomData* geoms [[buffer(2)]],
    device Contact* contacts [[buffer(3)]],
    device atomic_uint* contact_counts [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_geoms;
    uint geom_id = gid % params.num_geoms;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    GeomData geom_a = geoms[geom_id];
    uint body_a = geom_a.type_body.y;
    uint body_idx_a = env_id * params.num_bodies + body_a;

    float3 pos_a = geom_world_position(positions[body_idx_a], quaternions[body_idx_a], geom_a);
    float radius_a = geom_bounding_radius(geom_a);

    uint type_a = geom_a.type_body.x;
    uint group_a = geom_a.type_body.w;  // contype
    uint mask_a = geom_a.type_body.z;   // conaffinity

    // Check against all other geoms (simple O(n²) for now)
    for (uint other = geom_id + 1; other < params.num_geoms; other++) {
        GeomData geom_b = geoms[other];
        uint body_b = geom_b.type_body.y;

        // Skip self-collision
        if (body_a == body_b) continue;

        uint type_b = geom_b.type_body.x;
        uint group_b = geom_b.type_body.w;  // contype
        uint mask_b = geom_b.type_body.z;   // conaffinity

        // contype/conaffinity filter (MuJoCo semantics)
        if ((group_a & mask_b) == 0 && (group_b & mask_a) == 0) continue;

        uint body_idx_b = env_id * params.num_bodies + body_b;
        float3 pos_b = geom_world_position(positions[body_idx_b], quaternions[body_idx_b], geom_b);
        float radius_b = geom_bounding_radius(geom_b);

        // Plane bypass — planes are infinite, always emit candidate
        bool is_plane_pair = (type_a == 3 || type_b == 3);

        if (!is_plane_pair) {
            // AABB test (simplified as sphere-sphere)
            float3 diff = pos_b - pos_a;
            float dist_sq = dot(diff, diff);
            float min_dist = radius_a + radius_b + 0.1; // Margin

            if (dist_sq >= min_dist * min_dist) continue;
        }

        // Potential collision - add to narrow phase
        uint count = atomic_fetch_add_explicit(
            &contact_counts[env_id], 1, memory_order_relaxed);

        if (count < params.max_contacts) {
            uint contact_idx = env_id * params.max_contacts + count;
            contacts[contact_idx].indices = uint4(body_a, body_b, geom_id, other);
        }
    }
}

// ============================================================================
// GPU Spatial Hash Broad Phase (3-pass)
// ============================================================================

// Grid parameters embedded in SimParams are insufficient for spatial hash,
// so we use a separate constant buffer for grid config.
struct SpatialHashParams {
    uint grid_dim_x;
    uint grid_dim_y;
    uint grid_dim_z;
    uint total_cells;
    float cell_size;
    float inv_cell_size;
    uint num_geoms;
    uint num_envs;
    uint max_contacts;
    float origin_x;
    float origin_y;
    float origin_z;
};

// Pass 1: Compute cell ID for each geom and atomically increment cell counts.
kernel void broad_phase_count_cells(
    device const float4* positions [[buffer(0)]],
    device const GeomData* geoms [[buffer(1)]],
    device atomic_uint* cell_counts [[buffer(2)]],
    device uint* cell_ids [[buffer(3)]],
    constant SpatialHashParams& grid [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    device const float4* quaternions [[buffer(6)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / grid.num_geoms;
    uint geom_id = gid % grid.num_geoms;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    GeomData geom = geoms[geom_id];
    uint body_id = geom.type_body.y;
    uint body_idx = env_id * params.num_bodies + body_id;
    float3 pos = geom_world_position(positions[body_idx], quaternions[body_idx], geom);

    // Compute cell coordinates relative to grid origin (clamped to grid bounds)
    float3 grid_origin = float3(grid.origin_x, grid.origin_y, grid.origin_z);
    int3 cell_coord = int3(floor((pos - grid_origin) * grid.inv_cell_size));
    cell_coord = clamp(cell_coord, int3(0), int3(grid.grid_dim_x - 1, grid.grid_dim_y - 1, grid.grid_dim_z - 1));

    uint cell_id = uint(cell_coord.x) + uint(cell_coord.y) * grid.grid_dim_x +
                   uint(cell_coord.z) * grid.grid_dim_x * grid.grid_dim_y;

    // Store per-env cell offset
    uint env_cell_offset = env_id * grid.total_cells;
    uint global_geom_idx = env_id * grid.num_geoms + geom_id;

    cell_ids[global_geom_idx] = cell_id;
    atomic_fetch_add_explicit(&cell_counts[env_cell_offset + cell_id], 1, memory_order_relaxed);
}

// Pass 2: Prefix sum over cell_counts to produce cell_offsets.
// One threadgroup per env, serial scan (sufficient for grids up to ~262k cells).
kernel void broad_phase_prefix_sum(
    device uint* cell_counts [[buffer(0)]],
    device uint* cell_offsets [[buffer(1)]],
    constant SpatialHashParams& grid [[buffer(2)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint dispatch_env_id [[thread_position_in_grid]]
) {
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint base = env_id * grid.total_cells;
    uint sum = 0;
    for (uint i = 0; i < grid.total_cells; i++) {
        cell_offsets[base + i] = sum;
        sum += cell_counts[base + i];
        cell_counts[base + i] = 0; // Reset for scatter pass
    }
}

// Pass 3a: Scatter geoms into sorted order.
// Split from the detect pass: a threadgroup_barrier cannot synchronize across
// threadgroups, so scatter and query must be separate dispatches with a
// device-memory barrier between them.
kernel void broad_phase_scatter(
    device const uint* cell_ids [[buffer(0)]],
    device const uint* cell_offsets [[buffer(1)]],
    device atomic_uint* cell_counts [[buffer(2)]],
    device uint* sorted_geoms [[buffer(3)]],
    constant SpatialHashParams& grid [[buffer(4)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / grid.num_geoms;
    uint geom_id = gid % grid.num_geoms;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint env_cell_base = env_id * grid.total_cells;
    uint env_geom_base = env_id * grid.num_geoms;
    uint global_geom_idx = env_geom_base + geom_id;

    uint cell_id = cell_ids[global_geom_idx];

    uint slot = atomic_fetch_add_explicit(&cell_counts[env_cell_base + cell_id], 1, memory_order_relaxed);
    sorted_geoms[env_geom_base + cell_offsets[env_cell_base + cell_id] + slot] = geom_id;
}

// Pass 3b: Detect collisions via cell neighbors (runs after scatter completes).
kernel void broad_phase_detect(
    device const float4* positions [[buffer(0)]],
    device const GeomData* geoms [[buffer(1)]],
    device const uint* cell_ids [[buffer(2)]],
    device uint* cell_offsets [[buffer(3)]],
    device atomic_uint* cell_counts [[buffer(4)]],
    device uint* sorted_geoms [[buffer(5)]],
    device Contact* contacts [[buffer(6)]],
    device atomic_uint* contact_counts [[buffer(7)]],
    constant SpatialHashParams& grid [[buffer(8)]],
    constant SimParams& params [[buffer(9)]],
    device const float4* quaternions [[buffer(10)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / grid.num_geoms;
    uint geom_id = gid % grid.num_geoms;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint env_cell_base = env_id * grid.total_cells;
    uint env_geom_base = env_id * grid.num_geoms;

    // Query own cell + 26 neighbors for potential collisions
    GeomData geom_a = geoms[geom_id];
    uint body_a = geom_a.type_body.y;
    uint body_idx_a = env_id * params.num_bodies + body_a;
    float3 pos_a = geom_world_position(positions[body_idx_a], quaternions[body_idx_a], geom_a);
    float radius_a = geom_bounding_radius(geom_a);
    uint type_a = geom_a.type_body.x;
    uint group_a = geom_a.type_body.w;
    uint mask_a = geom_a.type_body.z;

    // Infinite planes are paired explicitly by each finite geom below. A
    // plane cannot be represented by one spatial-hash cell.
    if (type_a == 3) return;

    float3 grid_origin = float3(grid.origin_x, grid.origin_y, grid.origin_z);
    int3 center = int3(floor((pos_a - grid_origin) * grid.inv_cell_size));
    center = clamp(center, int3(0), int3(grid.grid_dim_x - 1, grid.grid_dim_y - 1, grid.grid_dim_z - 1));

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 nc = center + int3(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) continue;
                if (uint(nc.x) >= grid.grid_dim_x || uint(nc.y) >= grid.grid_dim_y || uint(nc.z) >= grid.grid_dim_z) continue;

                uint ncid = uint(nc.x) + uint(nc.y) * grid.grid_dim_x + uint(nc.z) * grid.grid_dim_x * grid.grid_dim_y;
                uint start = cell_offsets[env_cell_base + ncid];
                uint end = start + atomic_load_explicit(&cell_counts[env_cell_base + ncid], memory_order_relaxed);

                for (uint s = start; s < end; s++) {
                    uint other = sorted_geoms[env_geom_base + s];
                    if (other <= geom_id) continue; // Avoid duplicates

                    GeomData geom_b = geoms[other];
                    uint body_b = geom_b.type_body.y;
                    if (body_a == body_b) continue;

                    uint group_b = geom_b.type_body.w;
                    uint mask_b = geom_b.type_body.z;
                    if ((group_a & mask_b) == 0 && (group_b & mask_a) == 0) continue;

                    uint type_b = geom_b.type_body.x;
                    if (type_b == 3) continue;
                    uint body_idx_b = env_id * params.num_bodies + body_b;
                    float3 pos_b = geom_world_position(positions[body_idx_b], quaternions[body_idx_b], geom_b);
                    float radius_b = geom_bounding_radius(geom_b);

                    float3 diff = pos_b - pos_a;
                    float dist_sq = dot(diff, diff);
                    float min_dist = radius_a + radius_b + 0.1;
                    if (dist_sq >= min_dist * min_dist) continue;

                    uint count = atomic_fetch_add_explicit(
                        &contact_counts[env_id], 1, memory_order_relaxed);

                    if (count < params.max_contacts) {
                        uint contact_idx = env_id * params.max_contacts + count;
                        contacts[contact_idx].indices = uint4(body_a, body_b, geom_id, other);
                    }
                }
            }
        }
    }

    // Emit each compatible infinite-plane pair exactly once from the finite
    // geom's thread, regardless of cell distance.
    for (uint plane_id = 0; plane_id < grid.num_geoms; ++plane_id) {
        GeomData plane = geoms[plane_id];
        if (plane.type_body.x != 3) continue;
        uint plane_body = plane.type_body.y;
        if (plane_body == body_a) continue;
        uint plane_group = plane.type_body.w;
        uint plane_mask = plane.type_body.z;
        if ((group_a & plane_mask) == 0 && (plane_group & mask_a) == 0) continue;

        uint count = atomic_fetch_add_explicit(
            &contact_counts[env_id], 1, memory_order_relaxed);
        if (count < params.max_contacts) {
            uint contact_idx = env_id * params.max_contacts + count;
            if (plane_id < geom_id) {
                contacts[contact_idx].indices = uint4(
                    plane_body, body_a, plane_id, geom_id
                );
            } else {
                contacts[contact_idx].indices = uint4(
                    body_a, plane_body, geom_id, plane_id
                );
            }
        }
    }
}

// ============================================================================
// Deterministic Contact Sorting
// ============================================================================

// Sort contacts within each env by (min_body, max_body) for deterministic results.
// One thread per env, serial insertion sort on ≤max_contacts items.
kernel void sort_contacts(
    device Contact* contacts [[buffer(0)]],
    device const uint* contact_counts [[buffer(1)]],
    constant SimParams& params [[buffer(2)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint dispatch_env_id [[thread_position_in_grid]]
) {
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint count = min(contact_counts[env_id], params.max_contacts);
    if (count <= 1) return;

    uint base = env_id * params.max_contacts;

    // Insertion sort by composite key: (min_body, max_body, min_geom, max_geom).
    // Geom ids break ties between multiple contacts of the same body pair —
    // without them, equal keys stay in nondeterministic atomic-append order.
    for (uint i = 1; i < count; i++) {
        Contact key = contacts[base + i];
        ulong ka = min(key.indices.x, key.indices.y);
        ulong kb = max(key.indices.x, key.indices.y);
        ulong kga = min(key.indices.z, key.indices.w);
        ulong kgb = max(key.indices.z, key.indices.w);
        ulong key_val = (ka << 48) | (kb << 32) | (kga << 16) | kgb;

        int j = int(i) - 1;
        while (j >= 0) {
            Contact cj = contacts[base + uint(j)];
            ulong ja = min(cj.indices.x, cj.indices.y);
            ulong jb = max(cj.indices.x, cj.indices.y);
            ulong jga = min(cj.indices.z, cj.indices.w);
            ulong jgb = max(cj.indices.z, cj.indices.w);
            ulong j_val = (ja << 48) | (jb << 32) | (jga << 16) | jgb;

            if (j_val <= key_val) break;

            contacts[base + uint(j) + 1] = contacts[base + uint(j)];
            j--;
        }
        contacts[base + uint(j) + 1] = key;
    }
}

// ============================================================================
// Narrow Phase Collision Detection
// ============================================================================

kernel void narrow_phase(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device const GeomData* geoms [[buffer(2)]],
    device Contact* contacts [[buffer(3)]],
    device const uint* contact_counts [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint count = min(contact_counts[env_id], params.max_contacts);
    if (contact_id >= count) return;

    uint contact_idx = env_id * params.max_contacts + contact_id;
    Contact c = contacts[contact_idx];

    uint body_a = c.indices.x;
    uint body_b = c.indices.y;
    uint geom_a_id = c.indices.z;
    uint geom_b_id = c.indices.w;

    GeomData geom_a = geoms[geom_a_id];
    GeomData geom_b = geoms[geom_b_id];

    uint body_idx_a = env_id * params.num_bodies + body_a;
    uint body_idx_b = env_id * params.num_bodies + body_b;

    float3 pos_a = geom_world_position(positions[body_idx_a], quaternions[body_idx_a], geom_a);
    float3 pos_b = geom_world_position(positions[body_idx_b], quaternions[body_idx_b], geom_b);
    float4 quat_a = geom_world_quat(quaternions[body_idx_a], geom_a);
    float4 quat_b = geom_world_quat(quaternions[body_idx_b], geom_b);

    uint type_a = geom_a.type_body.x;
    uint type_b = geom_b.type_body.x;

    float3 normal;
    float penetration;
    float3 contact_point;

    bool has_contact = false;

    // Sphere-sphere (type 0)
    if (type_a == 0 && type_b == 0) {
        float radius_a = geom_a.pos_size0.w;
        float radius_b = geom_b.pos_size0.w;

        // Contact normals point from body B toward body A because the solver
        // applies +normal to A and -normal to B.
        float3 diff = pos_a - pos_b;
        float dist = length(diff);
        float min_dist = radius_a + radius_b;

        if (dist < min_dist) {
            normal = dist > 1e-6 ? diff / dist : float3(1, 0, 0);
            penetration = min_dist - dist;
            contact_point = pos_a - normal * (radius_a - penetration * 0.5);
            has_contact = true;
        }
    }
    // Sphere-plane (type 0-3)
    else if ((type_a == 0 && type_b == 3) || (type_a == 3 && type_b == 0)) {
        float3 sphere_pos;
        float sphere_radius;
        float3 plane_pos;
        float3 plane_normal;

        if (type_a == 0) {
            sphere_pos = pos_a;
            sphere_radius = geom_a.pos_size0.w;
            plane_pos = pos_b;
            plane_normal = rotate_by_quat(float3(0, 0, 1), quat_b);
        } else {
            sphere_pos = pos_b;
            sphere_radius = geom_b.pos_size0.w;
            plane_pos = pos_a;
            plane_normal = rotate_by_quat(float3(0, 0, 1), quat_a);
        }

        float signed_dist = dot(sphere_pos - plane_pos, plane_normal);

        if (signed_dist < sphere_radius) {
            normal = plane_normal;
            penetration = sphere_radius - signed_dist;
            contact_point = sphere_pos - plane_normal * (signed_dist + penetration * 0.5);
            has_contact = true;

            // Flip normal if needed
            if (type_a == 3) {
                normal = -normal;
            }
        }
    }
    // Capsule-plane (type 1-3)
    else if ((type_a == 1 && type_b == 3) || (type_a == 3 && type_b == 1)) {
        float3 capsule_pos;
        float capsule_radius;
        float capsule_half_len;
        float4 capsule_quat;
        float3 plane_normal;

        if (type_a == 1) {
            capsule_pos = pos_a;
            capsule_radius = geom_a.pos_size0.w;
            capsule_half_len = geom_a.params.x;
            capsule_quat = quat_a;
            plane_normal = rotate_by_quat(float3(0, 0, 1), quat_b);
        } else {
            capsule_pos = pos_b;
            capsule_radius = geom_b.pos_size0.w;
            capsule_half_len = geom_b.params.x;
            capsule_quat = quat_b;
            plane_normal = rotate_by_quat(float3(0, 0, 1), quat_a);
        }

        // Capsule axis
        float3 axis = rotate_by_quat(float3(0, 0, 1), capsule_quat);
        float3 p1 = capsule_pos - axis * capsule_half_len;
        float3 p2 = capsule_pos + axis * capsule_half_len;

        // Check both endpoints
        float3 plane_pos = type_a == 3 ? pos_a : pos_b;
        float d1 = dot(p1 - plane_pos, plane_normal) - capsule_radius;
        float d2 = dot(p2 - plane_pos, plane_normal) - capsule_radius;

        if (d1 < 0 || d2 < 0) {
            float3 deepest = d1 < d2 ? p1 : p2;
            float dist = min(d1, d2);

            normal = type_a == 1 ? plane_normal : -plane_normal;
            penetration = -dist;
            contact_point = deepest - plane_normal * dot(deepest - plane_pos, plane_normal);
            has_contact = true;
        }
    }
    // Sphere-capsule (type 0-1), with fully composed capsule orientation.
    else if ((type_a == 0 && type_b == 1) || (type_a == 1 && type_b == 0)) {
        float3 sphere_pos = type_a == 0 ? pos_a : pos_b;
        float sphere_radius = type_a == 0 ? geom_a.pos_size0.w : geom_b.pos_size0.w;
        GeomData capsule = type_a == 1 ? geom_a : geom_b;
        float3 capsule_pos = type_a == 1 ? pos_a : pos_b;
        float4 capsule_quat = type_a == 1 ? quat_a : quat_b;
        float capsule_radius = capsule.pos_size0.w;
        float capsule_half_len = capsule.params.x;
        float3 axis = rotate_by_quat(float3(0, 0, 1), capsule_quat);
        float3 closest = closest_point_on_segment(
            sphere_pos,
            capsule_pos - axis * capsule_half_len,
            capsule_pos + axis * capsule_half_len
        );
        float3 capsule_to_sphere = sphere_pos - closest;
        float distance = length(capsule_to_sphere);
        float radius_sum = sphere_radius + capsule_radius;
        if (distance < radius_sum) {
            float3 outward = distance > 1e-6
                ? capsule_to_sphere / distance
                : float3(1, 0, 0);
            normal = type_a == 0 ? outward : -outward;
            penetration = radius_sum - distance;
            contact_point = sphere_pos - outward * (sphere_radius - penetration * 0.5f);
            has_contact = true;
        }
    }
    // Sphere-box (type 0-2), evaluated in the oriented box's local frame.
    else if ((type_a == 0 && type_b == 2) || (type_a == 2 && type_b == 0)) {
        float3 sphere_pos = type_a == 0 ? pos_a : pos_b;
        float sphere_radius = type_a == 0 ? geom_a.pos_size0.w : geom_b.pos_size0.w;
        GeomData box = type_a == 2 ? geom_a : geom_b;
        float3 box_pos = type_a == 2 ? pos_a : pos_b;
        float4 box_quat = type_a == 2 ? quat_a : quat_b;
        float3 extents = float3(box.pos_size0.w, box.params.x, box.params.y);
        float3 sphere_local = rotate_by_quat(
            sphere_pos - box_pos,
            quat_conjugate(box_quat)
        );
        float3 box_local = clamp(sphere_local, -extents, extents);
        float3 delta_local = sphere_local - box_local;
        float distance = length(delta_local);

        if (distance < sphere_radius) {
            float3 outward_local;
            float3 sphere_surface_local;
            if (distance > 1e-6f) {
                outward_local = delta_local / distance;
                penetration = sphere_radius - distance;
                sphere_surface_local = sphere_local - outward_local * sphere_radius;
            } else {
                outward_local = nearest_aabb_face_normal(sphere_local, extents);
                float face_clearance = dot(extents - abs(sphere_local), abs(outward_local));
                penetration = sphere_radius + face_clearance;
                box_local = sphere_local + outward_local * face_clearance;
                sphere_surface_local = sphere_local + outward_local * sphere_radius;
            }
            float3 outward = rotate_by_quat(outward_local, box_quat);
            normal = type_a == 0 ? outward : -outward;
            float3 box_surface = box_pos + rotate_by_quat(box_local, box_quat);
            float3 sphere_surface = box_pos + rotate_by_quat(sphere_surface_local, box_quat);
            contact_point = (box_surface + sphere_surface) * 0.5f;
            has_contact = true;
        }
    }
    // Sphere-cylinder (type 0-4), evaluated against the exact finite cylinder
    // in its oriented local frame.
    else if ((type_a == 0 && type_b == 4) || (type_a == 4 && type_b == 0)) {
        float3 sphere_pos = type_a == 0 ? pos_a : pos_b;
        float sphere_radius = type_a == 0 ? geom_a.pos_size0.w : geom_b.pos_size0.w;
        GeomData cylinder = type_a == 4 ? geom_a : geom_b;
        float3 cylinder_pos = type_a == 4 ? pos_a : pos_b;
        float4 cylinder_quat = type_a == 4 ? quat_a : quat_b;
        float cylinder_radius = cylinder.pos_size0.w;
        float cylinder_half_height = cylinder.params.x;
        float3 sphere_local = rotate_by_quat(
            sphere_pos - cylinder_pos,
            quat_conjugate(cylinder_quat)
        );
        float radial_length = length(sphere_local.xy);
        float2 radial_direction = radial_length > 1e-6f
            ? sphere_local.xy / radial_length
            : float2(1, 0);
        float3 cylinder_surface_local = float3(
            radial_direction * min(radial_length, cylinder_radius),
            clamp(sphere_local.z, -cylinder_half_height, cylinder_half_height)
        );
        float3 delta_local = sphere_local - cylinder_surface_local;
        float distance = length(delta_local);

        if (distance < sphere_radius) {
            float3 outward_local;
            float3 sphere_surface_local;
            if (distance > 1e-6f) {
                outward_local = delta_local / distance;
                penetration = sphere_radius - distance;
                sphere_surface_local = sphere_local - outward_local * sphere_radius;
            } else {
                float side_clearance = cylinder_radius - radial_length;
                float cap_clearance = cylinder_half_height - abs(sphere_local.z);
                if (side_clearance < cap_clearance) {
                    outward_local = float3(radial_direction, 0);
                    penetration = sphere_radius + side_clearance;
                    cylinder_surface_local = sphere_local + outward_local * side_clearance;
                } else {
                    outward_local = float3(0, 0, sphere_local.z >= 0.0f ? 1.0f : -1.0f);
                    penetration = sphere_radius + cap_clearance;
                    cylinder_surface_local = sphere_local + outward_local * cap_clearance;
                }
                sphere_surface_local = sphere_local + outward_local * sphere_radius;
            }
            float3 outward = rotate_by_quat(outward_local, cylinder_quat);
            normal = type_a == 0 ? outward : -outward;
            float3 cylinder_surface = cylinder_pos + rotate_by_quat(
                cylinder_surface_local, cylinder_quat
            );
            float3 sphere_surface = cylinder_pos + rotate_by_quat(
                sphere_surface_local, cylinder_quat
            );
            contact_point = (cylinder_surface + sphere_surface) * 0.5f;
            has_contact = true;
        }
    }
    // Capsule-capsule (type 1-1), using closest points on oriented segments.
    else if (type_a == 1 && type_b == 1) {
        float3 axis_a = rotate_by_quat(float3(0, 0, 1), quat_a);
        float3 axis_b = rotate_by_quat(float3(0, 0, 1), quat_b);
        float half_a = geom_a.params.x;
        float half_b = geom_b.params.x;
        SegmentClosestPoints closest = closest_points_on_segments(
            pos_a - axis_a * half_a,
            pos_a + axis_a * half_a,
            pos_b - axis_b * half_b,
            pos_b + axis_b * half_b
        );
        float3 b_to_a = closest.a - closest.b;
        float distance = length(b_to_a);
        float radius_a = geom_a.pos_size0.w;
        float radius_b = geom_b.pos_size0.w;
        float radius_sum = radius_a + radius_b;
        if (distance < radius_sum) {
            if (distance > 1e-6f) {
                normal = b_to_a / distance;
            } else {
                float3 center_delta = pos_a - pos_b;
                normal = length(center_delta) > 1e-6f
                    ? normalize(center_delta)
                    : float3(1, 0, 0);
            }
            penetration = radius_sum - distance;
            float3 surface_a = closest.a - normal * radius_a;
            float3 surface_b = closest.b + normal * radius_b;
            contact_point = (surface_a + surface_b) * 0.5f;
            has_contact = true;
        }
    }
    // Capsule-box (type 1-2). Transform the capsule segment into box-local
    // space and solve the exact piecewise-quadratic segment/AABB distance.
    else if ((type_a == 1 && type_b == 2) || (type_a == 2 && type_b == 1)) {
        GeomData capsule = type_a == 1 ? geom_a : geom_b;
        float3 capsule_pos = type_a == 1 ? pos_a : pos_b;
        float4 capsule_quat = type_a == 1 ? quat_a : quat_b;
        GeomData box = type_a == 2 ? geom_a : geom_b;
        float3 box_pos = type_a == 2 ? pos_a : pos_b;
        float4 box_quat = type_a == 2 ? quat_a : quat_b;
        float3 extents = float3(box.pos_size0.w, box.params.x, box.params.y);
        float capsule_radius = capsule.pos_size0.w;
        float3 capsule_axis = rotate_by_quat(float3(0, 0, 1), capsule_quat);
        float3 segment_start = capsule_pos - capsule_axis * capsule.params.x;
        float3 segment_end = capsule_pos + capsule_axis * capsule.params.x;
        float4 box_inverse = quat_conjugate(box_quat);
        float3 start_local = rotate_by_quat(segment_start - box_pos, box_inverse);
        float3 end_local = rotate_by_quat(segment_end - box_pos, box_inverse);
        SegmentBoxClosestPoints closest = closest_points_segment_aabb(
            start_local, end_local, extents
        );
        float distance = sqrt(max(closest.distance_sq, 0.0f));

        if (distance < capsule_radius) {
            float3 outward_local;
            float3 capsule_surface_local;
            if (distance > 1e-6f) {
                outward_local = (closest.segment - closest.box) / distance;
                penetration = capsule_radius - distance;
                capsule_surface_local = closest.segment - outward_local * capsule_radius;
            } else {
                outward_local = nearest_aabb_face_normal(closest.segment, extents);
                float face_clearance = dot(extents - abs(closest.segment), abs(outward_local));
                penetration = capsule_radius + face_clearance;
                closest.box = closest.segment + outward_local * face_clearance;
                capsule_surface_local = closest.segment + outward_local * capsule_radius;
            }
            float3 outward = rotate_by_quat(outward_local, box_quat);
            normal = type_a == 1 ? outward : -outward;
            float3 box_surface = box_pos + rotate_by_quat(closest.box, box_quat);
            float3 capsule_surface = box_pos + rotate_by_quat(capsule_surface_local, box_quat);
            contact_point = (box_surface + capsule_surface) * 0.5f;
            has_contact = true;
        }
    }
    // Oriented box-box (type 2-2), using all 15 separating axes. The current
    // contact representation stores one support-point contact rather than a
    // clipped face manifold.
    else if (type_a == 2 && type_b == 2) {
        float3 extents_a = float3(geom_a.pos_size0.w, geom_a.params.x, geom_a.params.y);
        float3 extents_b = float3(geom_b.pos_size0.w, geom_b.params.x, geom_b.params.y);
        float3 axes_a[3] = {
            rotate_by_quat(float3(1, 0, 0), quat_a),
            rotate_by_quat(float3(0, 1, 0), quat_a),
            rotate_by_quat(float3(0, 0, 1), quat_a)
        };
        float3 axes_b[3] = {
            rotate_by_quat(float3(1, 0, 0), quat_b),
            rotate_by_quat(float3(0, 1, 0), quat_b),
            rotate_by_quat(float3(0, 0, 1), quat_b)
        };
        float3 center_delta = pos_b - pos_a;
        float minimum_overlap = INFINITY;
        float3 minimum_normal = float3(1, 0, 0);
        bool separated = false;

        for (uint i = 0; i < 3 && !separated; ++i) {
            float3 axis = axes_a[i];
            float radius_a = extents_a[i];
            float radius_b = 0.0f;
            for (uint j = 0; j < 3; ++j) {
                radius_b += extents_b[j] * abs(dot(axis, axes_b[j]));
            }
            float signed_distance = dot(center_delta, axis);
            float overlap = radius_a + radius_b - abs(signed_distance);
            if (overlap <= 0.0f) {
                separated = true;
            } else if (overlap < minimum_overlap) {
                minimum_overlap = overlap;
                minimum_normal = signed_distance >= 0.0f ? -axis : axis;
            }
        }
        for (uint i = 0; i < 3 && !separated; ++i) {
            float3 axis = axes_b[i];
            float radius_a = 0.0f;
            for (uint j = 0; j < 3; ++j) {
                radius_a += extents_a[j] * abs(dot(axis, axes_a[j]));
            }
            float radius_b = extents_b[i];
            float signed_distance = dot(center_delta, axis);
            float overlap = radius_a + radius_b - abs(signed_distance);
            if (overlap <= 0.0f) {
                separated = true;
            } else if (overlap < minimum_overlap) {
                minimum_overlap = overlap;
                minimum_normal = signed_distance >= 0.0f ? -axis : axis;
            }
        }
        for (uint i = 0; i < 3 && !separated; ++i) {
            for (uint j = 0; j < 3 && !separated; ++j) {
                float3 raw_axis = cross(axes_a[i], axes_b[j]);
                float axis_length = length(raw_axis);
                if (axis_length <= 1e-6f) continue;
                float3 axis = raw_axis / axis_length;
                float radius_a = 0.0f;
                float radius_b = 0.0f;
                for (uint k = 0; k < 3; ++k) {
                    radius_a += extents_a[k] * abs(dot(axis, axes_a[k]));
                    radius_b += extents_b[k] * abs(dot(axis, axes_b[k]));
                }
                float signed_distance = dot(center_delta, axis);
                float overlap = radius_a + radius_b - abs(signed_distance);
                if (overlap <= 0.0f) {
                    separated = true;
                } else if (overlap < minimum_overlap) {
                    minimum_overlap = overlap;
                    minimum_normal = signed_distance >= 0.0f ? -axis : axis;
                }
            }
        }

        if (!separated) {
            normal = minimum_normal;
            penetration = minimum_overlap;
            float3 support_a = pos_a;
            float3 support_b = pos_b;
            for (uint i = 0; i < 3; ++i) {
                float sign_a = dot(-normal, axes_a[i]) >= 0.0f ? 1.0f : -1.0f;
                float sign_b = dot(normal, axes_b[i]) >= 0.0f ? 1.0f : -1.0f;
                support_a += axes_a[i] * extents_a[i] * sign_a;
                support_b += axes_b[i] * extents_b[i] * sign_b;
            }
            contact_point = (support_a + support_b) * 0.5f;
            has_contact = true;
        }
    }
    // Oriented box-plane (type 2-3). Project the box half extents onto the
    // plane normal, then use the deepest support point as the contact anchor.
    else if ((type_a == 2 && type_b == 3) || (type_a == 3 && type_b == 2)) {
        GeomData box = type_a == 2 ? geom_a : geom_b;
        float3 box_pos = type_a == 2 ? pos_a : pos_b;
        float4 box_quat = type_a == 2 ? quat_a : quat_b;
        float3 plane_pos = type_a == 3 ? pos_a : pos_b;
        float4 plane_quat = type_a == 3 ? quat_a : quat_b;
        float3 plane_normal = rotate_by_quat(float3(0, 0, 1), plane_quat);
        float3 normal_local = rotate_by_quat(plane_normal, quat_conjugate(box_quat));
        float3 half_extents = float3(box.pos_size0.w, box.params.x, box.params.y);
        float projected_extent = dot(abs(normal_local), half_extents);
        float signed_center_distance = dot(box_pos - plane_pos, plane_normal);

        if (signed_center_distance < projected_extent) {
            penetration = projected_extent - signed_center_distance;
            contact_point = box_pos - plane_normal * projected_extent;
            normal = type_a == 2 ? plane_normal : -plane_normal;
            has_contact = true;
        }
    }
    // Oriented finite cylinder-plane (type 4-3). The support radius along the
    // plane normal is r*|n_xy| + h*|n_z| in cylinder-local coordinates.
    else if ((type_a == 4 && type_b == 3) || (type_a == 3 && type_b == 4)) {
        GeomData cylinder = type_a == 4 ? geom_a : geom_b;
        float3 cylinder_pos = type_a == 4 ? pos_a : pos_b;
        float4 cylinder_quat = type_a == 4 ? quat_a : quat_b;
        float3 plane_pos = type_a == 3 ? pos_a : pos_b;
        float4 plane_quat = type_a == 3 ? quat_a : quat_b;
        float3 plane_normal = rotate_by_quat(float3(0, 0, 1), plane_quat);
        float3 normal_local = rotate_by_quat(
            plane_normal, quat_conjugate(cylinder_quat)
        );
        float cylinder_radius = cylinder.pos_size0.w;
        float cylinder_half_height = cylinder.params.x;
        float radial_normal_length = length(normal_local.xy);
        float projected_extent = cylinder_radius * radial_normal_length
            + cylinder_half_height * abs(normal_local.z);
        float signed_center_distance = dot(cylinder_pos - plane_pos, plane_normal);

        if (signed_center_distance < projected_extent) {
            float2 radial_support = radial_normal_length > 1e-6f
                ? -normal_local.xy * (cylinder_radius / radial_normal_length)
                : float2(0);
            float z_support = normal_local.z >= 0.0f
                ? -cylinder_half_height
                : cylinder_half_height;
            float3 support_local = float3(radial_support, z_support);
            penetration = projected_extent - signed_center_distance;
            contact_point = cylinder_pos + rotate_by_quat(support_local, cylinder_quat);
            normal = type_a == 4 ? plane_normal : -plane_normal;
            has_contact = true;
        }
    }

    if (has_contact) {
        float combined_friction = sqrt(geom_a.params.z * geom_b.params.z);

        contacts[contact_idx].position_pen = float4(contact_point, penetration);
        contacts[contact_idx].normal_friction = float4(normal, combined_friction);
        contacts[contact_idx].impulses = float4(0, 0, 0, params.restitution);
    } else {
        // Invalidate contact
        contacts[contact_idx].position_pen.w = -1;
    }
}

// ============================================================================
// Joint Solver (XPBD)
// ============================================================================

struct XPBDConstraint {
    uint4 indices;       // body_a, body_b, env_id, type
    float4 anchor_a;     // local_a, compliance
    float4 anchor_b;     // local_b, damping
    float4 axis_target;  // axis, target
    float4 limits;       // lower, upper, friction, restitution
    float4 state;        // lambda, lambda_prev, violation, effective_mass
};

kernel void solve_joints(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device XPBDConstraint* constraints [[buffer(4)]],
    device const float4* inv_mass_inertia [[buffer(5)]],
    constant SimParams& params [[buffer(6)]],
    device const BodyData* body_data [[buffer(7)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    // Graph coloring dispatch: each color group is dispatched separately
    // to avoid race conditions when constraints share bodies.
    //
    // Buffer layout: [env0_c0, env0_c1, ..., env1_c0, env1_c1, ...]
    // Within each env, constraints are sorted by color.
    // params.constraint_offset = start index within each env's constraints
    // params.num_constraints = count for current color (per env)
    //
    // gid = env_id * count + local_constraint_idx
    // constraint_idx = env_id * total_constraints_per_env + offset + local_idx

    uint count = params.num_constraints;
    if (count == 0) return;

    uint dispatch_env_id = gid / count;
    uint local_idx = gid % count;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    // Total constraints per env (needed to compute actual buffer index)
    // We pass this as constraint_offset's high bits or compute from context
    // For now, use: constraint_idx = env_id * (total per env) + offset + local_idx
    // We need total_constraints_per_env, which we can derive from dispatch size
    // Actually, let's pass it explicitly via target_color (repurposed as constraints_per_env)
    uint constraints_per_env = params.target_color;  // Repurposed field
    uint constraint_idx = env_id * constraints_per_env + params.constraint_offset + local_idx;

    XPBDConstraint c = constraints[constraint_idx];
    uint type_and_color = c.indices.w;
    uint type = type_and_color & 0xFF;  // Lower 8 bits = type

    // Skip if invalid/padding
    if (type > 10) return;

    uint body_a = c.indices.x;
    uint body_b = c.indices.y;
    // env_id already computed from gid above (matches c.indices.z)

    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;
    
    float4 inv_mi_a = inv_mass_inertia[idx_a];
    float4 inv_mi_b = inv_mass_inertia[idx_b];
    
    float inv_mass_a = inv_mi_a.x;
    float inv_mass_b = inv_mi_b.x;
    float inv_mass_sum = inv_mass_a + inv_mass_b;
    
    if (inv_mass_sum < 1e-8) return;
    
    // Fetch state
    float3 pos_a = positions[idx_a].xyz;
    float3 pos_b = positions[idx_b].xyz;
    float4 quat_a = quaternions[idx_a];
    float4 quat_b = quaternions[idx_b];
    
    float compliance = c.anchor_a.w;
    float dt = params.dt;
    float alpha_tilde = compliance / (dt * dt);
    
    float C = 0.0;
    
    // --- Positional Constraint (Point-to-Point) ---
    if (type == 2) { // positional
        float3 r_a = rotate_by_quat(c.anchor_a.xyz, quat_a);
        float3 r_b = rotate_by_quat(c.anchor_b.xyz, quat_b);

        float3 diff = (pos_a + r_a) - (pos_b + r_b);
        C = length(diff);

        if (C > 1e-6) {
             float3 n = diff / C;

             // Generalized inverse mass uses body-origin offsets (r_a, r_b)
             // because these determine how rotation affects the constraint point.
             float3 rn_a = cross(r_a, n);
             float3 rn_b = cross(r_b, n);

             float w_a = inv_mass_a + dot(rn_a * inv_mi_a.yzw, rn_a);
             float w_b = inv_mass_b + dot(rn_b * inv_mi_b.yzw, rn_b);
             float w = w_a + w_b;

             // XPBD Update
             float lambda_prev = c.state.x;
             float d_lambda = (-C - alpha_tilde * lambda_prev) / (w + alpha_tilde);

             c.state.x = lambda_prev + d_lambda; // Update accumulated lambda

             float3 impulse = d_lambda * n;

             // Angular corrections use COM-relative offsets (rc_a, rc_b)
             // to create the correct inertial torque about the COM.
             // This is needed when body origin != COM.
             float3 com_a = body_data[body_a].com_offset.xyz;
             float3 com_b = body_data[body_b].com_offset.xyz;
             float3 rc_a = r_a - rotate_by_quat(com_a, quat_a);
             float3 rc_b = r_b - rotate_by_quat(com_b, quat_b);

             if (inv_mass_a > 0) {
                 positions[idx_a].xyz += impulse * inv_mass_a;
                 float3 ang_impulse = cross(rc_a, impulse);
                 float3 d_omega = ang_impulse * inv_mi_a.yzw;
                 float4 dq = quat_multiply(float4(d_omega, 0), quat_a) * 0.5;
                 quaternions[idx_a] = quat_normalize(quat_a + dq);
             }

             if (inv_mass_b > 0) {
                 positions[idx_b].xyz -= impulse * inv_mass_b;
                 float3 ang_impulse = cross(rc_b, impulse);
                 float3 d_omega = ang_impulse * inv_mi_b.yzw;
                 float4 dq = quat_multiply(float4(d_omega, 0), quat_b) * 0.5;
                 quaternions[idx_b] = quat_normalize(quat_b - dq);
             }
        }
    }
    // --- Weld Constraint (Positional + Angular) ---
    else if (type == 7) { // weld
        // Weld corrections are applied only to body_b (child).
        // Correcting body_a (parent) causes instability when body_a has other
        // constraints (e.g., hinge): the weld and hinge corrections fight each
        // other, injecting energy through the XPBD velocity update.
        // body_b passively follows body_a's motion.

        // 1. Positional part: keep body_b's anchor at body_a's anchor
        float3 r_a = rotate_by_quat(c.anchor_a.xyz, quat_a);
        float3 r_b = rotate_by_quat(c.anchor_b.xyz, quat_b);
        float3 diff = (pos_a + r_a) - (pos_b + r_b);
        C = length(diff);

        if (C > 1e-6) {
             float3 n = diff / C;
             float3 rn_b = cross(r_b, n);
             float w_b = inv_mass_b + dot(rn_b * inv_mi_b.yzw, rn_b);

             if (w_b > 1e-8) {
                 float d_lambda = (-C) / (w_b + alpha_tilde);
                 float3 impulse = d_lambda * n;

                 positions[idx_b].xyz -= impulse * inv_mass_b;
                 float3 ang_impulse = cross(r_b, impulse);
                 float3 d_omega = ang_impulse * inv_mi_b.yzw;
                 float4 dq = quat_multiply(float4(d_omega, 0), quat_b) * 0.5;
                 quaternions[idx_b] = quat_normalize(quat_b - dq);
             }
        }
        
        // 2. Angular part (lock relative orientation)
        // Target rel_quat stored in axis_target (q_a^-1 * q_b at rest)
        float4 q_rel_target = c.axis_target;

        // Get updated quaternions after positional correction
        float4 quat_a_upd = quaternions[idx_a];
        float4 quat_b_upd = quaternions[idx_b];

        // Current relative quaternion: q_rel = q_a^-1 * q_b
        float4 q_a_inv = quat_conjugate(quat_a_upd);
        float4 q_rel = quat_multiply(q_a_inv, quat_b_upd);

        // Error quaternion: q_err = q_rel * q_target^-1
        // If q_rel == q_target, q_err = identity (0,0,0,1)
        float4 q_target_inv = quat_conjugate(q_rel_target);
        float4 q_err = quat_multiply(q_rel, q_target_inv);

        // Ensure quaternion is in positive hemisphere for consistent error direction
        if (q_err.w < 0) {
            q_err = -q_err;
        }

        // Compute rotation angle and axis from error quaternion
        // q_err = (sin(θ/2) * axis, cos(θ/2))
        float3 q_err_xyz = float3(q_err.x, q_err.y, q_err.z);
        float sin_half_angle = length(q_err_xyz);

        if (sin_half_angle > 1e-6) {
            // Rotation axis (normalized)
            float3 axis_rel = q_err_xyz / sin_half_angle;

            // Full rotation angle using atan2 for numerical stability
            // This works correctly for angles up to 2π
            float angle = 2.0 * atan2(sin_half_angle, q_err.w);

            // Transform axis from body A frame to world frame
            float3 axis_world = rotate_by_quat(axis_rel, quat_a_upd);

            // Generalized inverse mass for angular constraint
            // Only body_b contributes — body_a's quaternion is left untouched
            // to prevent angular corrections from contaminating the parent body's
            // free DOFs (e.g., hinge axis rotation).
            float w_b = dot(axis_world * inv_mi_b.yzw, axis_world);

            if (w_b > 1e-8) {
                // XPBD angular correction (applied only to body_b)
                // Sign: +angle (not -angle) because q_err axis points in the
                // direction of body_b's excess rotation. The body_b update uses
                // (q_b - dq), so a positive d_lambda produces a positive d_omega
                // along the axis, and subtracting the resulting dq rotates body_b
                // in the -axis direction (i.e., back toward the target).
                float d_lambda_ang = (angle) / (w_b + alpha_tilde);
                float3 ang_impulse = d_lambda_ang * axis_world;

                if (inv_mass_b > 0) {
                    float3 d_omega = ang_impulse * inv_mi_b.yzw;
                    float4 dq = quat_multiply(float4(d_omega, 0), quat_b_upd) * 0.5;
                    quaternions[idx_b] = quat_normalize(quat_b_upd - dq);
                }
            }
        }
    }

    // --- Angular Constraint (Hinge alignment) ---
    else if (type == 3) { // angular
        // Ensure two axes (one in each body) remain aligned
        // axis_a is in anchor_a.xyz (in body A local frame)
        // axis_b is in anchor_b.xyz (in body B local frame)
        float3 axis_a_local = c.anchor_a.xyz;
        float3 axis_b_local = c.anchor_b.xyz;

        // Transform axes to world space
        float4 quat_a_curr = quaternions[idx_a];
        float4 quat_b_curr = quaternions[idx_b];

        float3 axis_a_world = rotate_by_quat(axis_a_local, quat_a_curr);
        float3 axis_b_world = rotate_by_quat(axis_b_local, quat_b_curr);

        // Constraint: axis_a_world × axis_b_world should be zero (parallel)
        // The cross product gives us the rotation axis needed to align them
        float3 cross_ab = cross(axis_a_world, axis_b_world);
        float sin_angle = length(cross_ab);

        if (sin_angle > 1e-6) {
            // Rotation error direction (normalized)
            float3 n = cross_ab / sin_angle;

            // For small angles, sin_angle ≈ angle, so C = sin_angle
            // For larger angles, use asin but clamp for safety
            float ang_C = asin(clamp(sin_angle, -1.0f, 1.0f));

            // Generalized inverse mass for angular constraint
            // w = n^T I_a^-1 n + n^T I_b^-1 n
            float w_a = dot(n * inv_mi_a.yzw, n);
            float w_b = dot(n * inv_mi_b.yzw, n);
            float w = w_a + w_b;

            if (w > 1e-8) {
                // Standard XPBD angular correction:
                //   λ = -C / (w + α̃)
                //   Δθ_a = +λ * I_a⁻¹ * n  (rotate A toward alignment)
                //   Δθ_b = -λ * I_b⁻¹ * n  (rotate B toward alignment)
                // The generalized inverse mass w already distributes corrections
                // proportionally to each body's inverse inertia.
                float d_lambda = -ang_C / (w + alpha_tilde);

                if (inv_mass_a > 0) {
                    float3 d_omega_a = d_lambda * n * inv_mi_a.yzw;
                    float4 dq = quat_multiply(float4(d_omega_a, 0), quat_a_curr) * 0.5;
                    quaternions[idx_a] = quat_normalize(quat_a_curr + dq);
                }

                if (inv_mass_b > 0) {
                    float3 d_omega_b = -d_lambda * n * inv_mi_b.yzw;
                    float4 dq = quat_multiply(float4(d_omega_b, 0), quat_b_curr) * 0.5;
                    quaternions[idx_b] = quat_normalize(quat_b_curr + dq);
                }
            }
        }
    }
    // --- Angular Limit Constraint ---
    else if (type == 4) { // angular_limit
        // Limit rotation around an axis between lower and upper bounds
        float3 axis_local = c.anchor_a.xyz;
        float lower = c.limits.x;
        float upper = c.limits.y;

        // Get current rotation angle around the axis
        float4 quat_a_curr = quaternions[idx_a];
        float4 quat_b_curr = quaternions[idx_b];

        // Relative rotation: q_rel = q_a^-1 * q_b
        float4 q_a_inv = quat_conjugate(quat_a_curr);
        float4 q_rel = quat_multiply(q_a_inv, quat_b_curr);

        // Project onto axis to get rotation angle
        float3 q_vec = float3(q_rel.x, q_rel.y, q_rel.z);
        float sin_half = dot(q_vec, axis_local);
        float cos_half = q_rel.w;
        float angle = 2.0 * atan2(sin_half, cos_half);

        // Check if limit is violated
        float violation = 0.0;
        if (angle < lower) {
            violation = lower - angle;
        } else if (angle > upper) {
            violation = upper - angle;
        }

        if (abs(violation) > 1e-6) {
            // World-space axis
            float3 axis_world = rotate_by_quat(axis_local, quat_a_curr);

            // Generalized inverse mass
            float w_a = dot(axis_world * inv_mi_a.yzw, axis_world);
            float w_b = dot(axis_world * inv_mi_b.yzw, axis_world);
            float w = w_a + w_b;

            if (w > 1e-8) {
                float d_lambda = violation / (w + alpha_tilde);
                float3 ang_impulse = d_lambda * axis_world;

                if (inv_mass_a > 0) {
                    float3 d_omega = ang_impulse * inv_mi_a.yzw;
                    float4 dq = quat_multiply(float4(d_omega, 0), quat_a_curr) * 0.5;
                    quaternions[idx_a] = quat_normalize(quat_a_curr + dq);
                }

                if (inv_mass_b > 0) {
                    float3 d_omega = ang_impulse * inv_mi_b.yzw;
                    float4 dq = quat_multiply(float4(d_omega, 0), quat_b_curr) * 0.5;
                    quaternions[idx_b] = quat_normalize(quat_b_curr - dq);
                }
            }
        }
    }
    // --- Linear Limit Constraint (Prismatic) ---
    else if (type == 5) { // linear_limit
        // Limit translation along an axis
        float3 axis_local = c.anchor_a.xyz;
        float lower = c.limits.x;
        float upper = c.limits.y;

        float4 quat_a_curr = quaternions[idx_a];
        float4 quat_b_curr = quaternions[idx_b];

        // Transform anchors and axis to world space
        float3 r_a = rotate_by_quat(float3(0, 0, 0), quat_a_curr); // Anchor at origin for simplicity
        float3 r_b = rotate_by_quat(c.anchor_b.xyz, quat_b_curr);
        float3 axis_world = rotate_by_quat(axis_local, quat_a_curr);

        // Current distance along axis
        float3 diff = (pos_b + r_b) - (pos_a + r_a);
        float dist = dot(diff, axis_world);

        // Check limit violation
        float violation = 0.0;
        if (dist < lower) {
            violation = lower - dist;
        } else if (dist > upper) {
            violation = upper - dist;
        }

        if (abs(violation) > 1e-6) {
            // Use axis as gradient direction
            float3 n = violation > 0 ? axis_world : -axis_world;
            float abs_violation = abs(violation);

            float3 rn_a = cross(r_a, n);
            float3 rn_b = cross(r_b, n);

            float w_a = inv_mass_a + dot(rn_a * inv_mi_a.yzw, rn_a);
            float w_b = inv_mass_b + dot(rn_b * inv_mi_b.yzw, rn_b);
            float w = w_a + w_b;

            if (w > 1e-8) {
                float d_lambda = abs_violation / (w + alpha_tilde);
                float3 impulse = d_lambda * n;

                if (inv_mass_a > 0) {
                    positions[idx_a].xyz -= impulse * inv_mass_a;
                }

                if (inv_mass_b > 0) {
                    positions[idx_b].xyz += impulse * inv_mass_b;
                }
            }
        }
    }

    // --- Slider Constraint (Prismatic: perpendicular positional + angular weld) ---
    else if (type == 10) { // slider
        // 1. Perpendicular positional constraint: constrain displacement perpendicular to slide axis
        float3 r_a = rotate_by_quat(c.anchor_a.xyz, quat_a);
        float3 r_b = rotate_by_quat(c.anchor_b.xyz, quat_b);

        // diff convention matches positional constraint: A - B
        float3 diff = (pos_a + r_a) - (pos_b + r_b);

        // Slide axis in world frame (stored in axis_target)
        float3 axis_local = c.axis_target.xyz;
        float3 axis_world = rotate_by_quat(axis_local, quat_a);

        // Remove the component along the slide axis (allow free translation along it)
        float along = dot(diff, axis_world);
        float3 perp = diff - along * axis_world;
        float perp_len = length(perp);

        if (perp_len > 1e-6) {
            float3 n = perp / perp_len;

            float3 rn_a = cross(r_a, n);
            float3 rn_b = cross(r_b, n);

            float w_a = inv_mass_a + dot(rn_a * inv_mi_a.yzw, rn_a);
            float w_b = inv_mass_b + dot(rn_b * inv_mi_b.yzw, rn_b);
            float w = w_a + w_b;

            float d_lambda = (-perp_len) / (w + alpha_tilde);
            float3 impulse = d_lambda * n;

            if (inv_mass_a > 0) {
                positions[idx_a].xyz += impulse * inv_mass_a;
                float3 ang_impulse = cross(r_a, impulse);
                float3 d_omega = ang_impulse * inv_mi_a.yzw;
                float4 dq = quat_multiply(float4(d_omega, 0), quat_a) * 0.5;
                quaternions[idx_a] = quat_normalize(quat_a + dq);
            }

            if (inv_mass_b > 0) {
                positions[idx_b].xyz -= impulse * inv_mass_b;
                float3 ang_impulse = cross(r_b, impulse);
                float3 d_omega = ang_impulse * inv_mi_b.yzw;
                float4 dq = quat_multiply(float4(d_omega, 0), quat_b) * 0.5;
                quaternions[idx_b] = quat_normalize(quat_b - dq);
            }
        }

        // 2. Angular weld: lock relative orientation
        // Reference relative quaternion stored in limits field
        float4 q_rel_target = float4(c.limits[0], c.limits[1], c.limits[2], c.limits[3]);

        float4 quat_a_upd = quaternions[idx_a];
        float4 quat_b_upd = quaternions[idx_b];

        float4 q_a_inv = quat_conjugate(quat_a_upd);
        float4 q_rel = quat_multiply(q_a_inv, quat_b_upd);

        float4 q_target_inv = quat_conjugate(q_rel_target);
        float4 q_err = quat_multiply(q_rel, q_target_inv);

        if (q_err.w < 0) {
            q_err = -q_err;
        }

        float3 q_err_xyz = float3(q_err.x, q_err.y, q_err.z);
        float sin_half_angle = length(q_err_xyz);

        if (sin_half_angle > 1e-6) {
            float3 axis_err = q_err_xyz / sin_half_angle;
            float angle_err = 2.0 * atan2(sin_half_angle, q_err.w);
            float3 axis_err_world = rotate_by_quat(axis_err, quat_a_upd);

            float w_a = dot(axis_err_world * inv_mi_a.yzw, axis_err_world);
            float w_b = dot(axis_err_world * inv_mi_b.yzw, axis_err_world);
            float w = w_a + w_b;

            if (w > 1e-8) {
                // Sign: +angle_err (not -angle_err) because q_err axis points
                // in the direction of body_b's excess rotation. With +angle_err,
                // body_a rotates in +axis direction (catching up) via += dq, and
                // body_b rotates in -axis direction (going back) via -= dq.
                float d_lambda_ang = (angle_err) / (w + alpha_tilde);
                float3 ang_impulse = d_lambda_ang * axis_err_world;

                if (inv_mass_a > 0) {
                    float3 d_omega = ang_impulse * inv_mi_a.yzw;
                    float4 dq = quat_multiply(float4(d_omega, 0), quat_a_upd) * 0.5;
                    quaternions[idx_a] = quat_normalize(quat_a_upd + dq);
                }
                if (inv_mass_b > 0) {
                    float3 d_omega = ang_impulse * inv_mi_b.yzw;
                    float4 dq = quat_multiply(float4(d_omega, 0), quat_b_upd) * 0.5;
                    quaternions[idx_b] = quat_normalize(quat_b_upd - dq);
                }
            }
        }
    }

    // Store updated constraint state
    constraints[constraint_idx] = c;
}

// ============================================================================
// Warm Start Constraints Kernel
// ============================================================================
// At the start of each timestep, initialize lambda from lambda_prev
// (scaled by a warm start factor) for faster convergence.
// At the end of each timestep, copy lambda to lambda_prev.

kernel void warm_start_constraints(
    device XPBDConstraint* constraints [[buffer(0)]],
    constant SimParams& params [[buffer(1)]],
    constant float& warm_start_factor [[buffer(2)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint constraints_per_env = params.target_color; // Repurposed field
    if (constraints_per_env == 0) return;

    uint dispatch_env_id = gid / constraints_per_env;
    uint local_idx = gid % constraints_per_env;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint idx = env_id * constraints_per_env + local_idx;
    // state.x = lambda, state.y = lambda_prev
    // Initialize lambda from previous frame's lambda (warm start)
    constraints[idx].state.x = warm_start_factor * constraints[idx].state.y;
}

kernel void store_lambda_prev(
    device XPBDConstraint* constraints [[buffer(0)]],
    constant SimParams& params [[buffer(1)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint constraints_per_env = params.target_color; // Repurposed field
    if (constraints_per_env == 0) return;

    uint dispatch_env_id = gid / constraints_per_env;
    uint local_idx = gid % constraints_per_env;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint idx = env_id * constraints_per_env + local_idx;
    // Copy current lambda to lambda_prev for next frame's warm starting
    constraints[idx].state.y = constraints[idx].state.x;
}

// ============================================================================
// Contact Caching - Persist contacts across frames for temporal coherence
// ============================================================================

// Cache current contacts to previous-frame buffer for reuse next frame.
// Contacts that persist between frames retain their accumulated impulses,
// improving convergence and reducing jitter.

kernel void cache_contacts(
    device const Contact* contacts [[buffer(0)]],
    device Contact* prev_contacts [[buffer(1)]],
    device const uint* contact_counts [[buffer(2)]],
    device uint* prev_contact_counts [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint count = min(contact_counts[env_id], params.max_contacts);

    // Copy count for first thread of each env
    if (contact_id == 0) {
        prev_contact_counts[env_id] = count;
    }

    if (contact_id >= count) return;

    uint idx = env_id * params.max_contacts + contact_id;
    prev_contacts[idx] = contacts[idx];
}

// Match new contacts against cached contacts from the previous frame.
// If a matching contact is found (same body pair, nearby position),
// copy the accumulated impulse for warm starting the contact solver.
kernel void match_cached_contacts(
    device Contact* contacts [[buffer(0)]],
    device const Contact* prev_contacts [[buffer(1)]],
    device const uint* contact_counts [[buffer(2)]],
    device const uint* prev_contact_counts [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint count = min(contact_counts[env_id], params.max_contacts);
    if (contact_id >= count) return;

    uint idx = env_id * params.max_contacts + contact_id;
    Contact c = contacts[idx];

    // Skip invalid contacts
    if (c.position_pen.w < 0) return;

    uint prev_count = prev_contact_counts[env_id];
    uint body_a = c.indices.x;
    uint body_b = c.indices.y;
    float3 pos = c.position_pen.xyz;

    // Search for matching contact in previous frame
    float best_dist_sq = 0.04; // 0.2^2 position match threshold
    int best_match = -1;

    for (uint i = 0; i < prev_count && i < params.max_contacts; i++) {
        uint prev_idx = env_id * params.max_contacts + i;
        Contact prev = prev_contacts[prev_idx];

        // Skip invalid previous contacts
        if (prev.position_pen.w < 0) continue;

        // Match by body pair (order-independent)
        bool same_pair = (prev.indices.x == body_a && prev.indices.y == body_b) ||
                         (prev.indices.x == body_b && prev.indices.y == body_a);

        if (!same_pair) continue;

        // Check position proximity
        float3 diff = prev.position_pen.xyz - pos;
        float dist_sq = dot(diff, diff);

        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_match = int(i);
        }
    }

    // If a match was found, transfer accumulated impulse for warm starting
    if (best_match >= 0) {
        uint prev_idx = env_id * params.max_contacts + uint(best_match);
        contacts[idx].impulses.x = prev_contacts[prev_idx].impulses.x * 0.8;
        contacts[idx].impulses.y = prev_contacts[prev_idx].impulses.y * 0.8;
        contacts[idx].impulses.z = prev_contacts[prev_idx].impulses.z * 0.8;
    }
}

// ============================================================================
// Update Joint States Kernel (Inverse Kinematics / Feedback)
// ============================================================================

kernel void update_joint_states(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device const float4* velocities [[buffer(2)]],
    device const float4* angular_velocities [[buffer(3)]],
    device const JointData* joints [[buffer(4)]],
    device float* joint_positions [[buffer(5)]],
    device float* joint_velocities [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_joints;
    uint joint_id = gid % params.num_joints;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    uint joint_idx = env_id * params.num_joints + joint_id;
    JointData joint = joints[joint_id];
    uint type = uint(joint.params.x);
    
    // Default to 0
    joint_positions[joint_idx] = 0;
    joint_velocities[joint_idx] = 0;
    
    uint body_a = uint(joint.params.y);
    uint body_b = uint(joint.params.z);
    
    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;
    
    float4 q_a = quaternions[idx_a];
    float4 q_b = quaternions[idx_b];
    float3 p_a = positions[idx_a].xyz;
    float3 p_b = positions[idx_b].xyz;
    
    float3 v_a = velocities[idx_a].xyz;
    float3 v_b = velocities[idx_b].xyz;
    float3 w_a = angular_velocities[idx_a].xyz;
    float3 w_b = angular_velocities[idx_b].xyz;
    
    // Revolute/Hinge
    if (type == 1) {
        // Axis in A frame
        float3 axis_local = joint.axis.xyz;
        float3 axis_world_a = rotate_by_quat(axis_local, q_a);
        
        // Relative rotation q_rel = q_a^-1 * q_b
        float4 q_a_inv = quat_conjugate(q_a);
        float4 q_rel = quat_multiply(q_a_inv, q_b);
        
        // Extract angle around axis
        // q_rel = [sin(theta/2)*axis, cos(theta/2)]
        // We project imaginary part onto axis
        float3 q_vec = float3(q_rel.x, q_rel.y, q_rel.z);
        float sin_half = dot(q_vec, axis_local);
        float cos_half = q_rel.w;
        float angle = 2.0 * atan2(sin_half, cos_half);
        
        joint_positions[joint_idx] = angle;
        
        // Velocity: (w_b - w_a) . axis_world
        float3 rel_omega = w_b - w_a;
        joint_velocities[joint_idx] = dot(rel_omega, axis_world_a);
    }
    // Prismatic/Slide
    else if (type == 2) {
        float3 axis_local = joint.axis.xyz;
        float3 axis_world_a = rotate_by_quat(axis_local, q_a);
        
        float3 r_a = rotate_by_quat(joint.anchor_parent.xyz, q_a);
        float3 r_b = rotate_by_quat(joint.anchor_child.xyz, q_b);
        
        float3 anchor_a_world = p_a + r_a;
        float3 anchor_b_world = p_b + r_b;
        
        float3 diff = anchor_b_world - anchor_a_world;
        float dist = dot(diff, axis_world_a);
        
        joint_positions[joint_idx] = dist;
        
        // Velocity
        // v_point_b - v_point_a
        float3 v_pt_a = v_a + cross(w_a, r_a);
        float3 v_pt_b = v_b + cross(w_b, r_b);
        float3 rel_vel = v_pt_b - v_pt_a;
        
        joint_velocities[joint_idx] = dot(rel_vel, axis_world_a);
    }
}

// ============================================================================
// Contact Solver (Position-Based Dynamics)
// ============================================================================

kernel void solve_contacts(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device Contact* contacts [[buffer(4)]],
    device const uint* contact_counts [[buffer(5)]],
    device const float4* inv_mass_inertia [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint count = min(contact_counts[env_id], params.max_contacts);
    if (contact_id >= count) return;

    uint contact_idx = env_id * params.max_contacts + contact_id;
    Contact c = contacts[contact_idx];

    // Skip invalid contacts
    if (c.position_pen.w < 0) return;

    uint body_a = c.indices.x;
    uint body_b = c.indices.y;

    uint idx_a = env_id * params.num_bodies + body_a;
    uint idx_b = env_id * params.num_bodies + body_b;

    float4 inv_mi_a = inv_mass_inertia[idx_a];
    float4 inv_mi_b = inv_mass_inertia[idx_b];

    float inv_mass_a = inv_mi_a.x;
    float inv_mass_b = inv_mi_b.x;

    float inv_mass_sum = inv_mass_a + inv_mass_b;
    if (inv_mass_sum < 1e-8) return;

    float3 normal = c.normal_friction.xyz;
    float penetration = c.position_pen.w;

    // Narrow phase computes penetration once per substep. Position correction
    // is dispatched once; repeating this stale correction per solver iteration
    // injects artificial velocity when the XPBD velocity update runs.
    float pos_correction = params.baumgarte
        * max(penetration - params.slop, 0.0f);

    // Atomic adds: several contacts in one env can share a body, and each
    // contact runs on its own thread — plain += loses impulses.
    if (inv_mass_a > 1e-8) {
        float mass_ratio_a = inv_mass_a / inv_mass_sum;
        atomic_add_float3(&positions[idx_a], pos_correction * mass_ratio_a * normal);
    }

    if (inv_mass_b > 1e-8) {
        float mass_ratio_b = inv_mass_b / inv_mass_sum;
        atomic_add_float3(&positions[idx_b], -(pos_correction * mass_ratio_b * normal));
    }
}

// Apply cached impulses only after XPBD has reconstructed velocities from the
// corrected poses; doing it before that update discards the warm start.
kernel void warm_start_contacts(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device Contact* contacts [[buffer(4)]],
    device const uint* contact_counts [[buffer(5)]],
    device const float4* inv_mass_inertia [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    if (contact_id >= min(contact_counts[env_id], params.max_contacts)) return;
    uint contact_idx = env_id * params.max_contacts + contact_id;
    Contact c = contacts[contact_idx];
    if (c.position_pen.w < 0) return;

    uint idx_a = env_id * params.num_bodies + c.indices.x;
    uint idx_b = env_id * params.num_bodies + c.indices.y;
    float4 inv_mi_a = inv_mass_inertia[idx_a];
    float4 inv_mi_b = inv_mass_inertia[idx_b];
    float3 normal = c.normal_friction.xyz;
    float3 tangent_a = contact_tangent(normal);
    float3 tangent_b = cross(normal, tangent_a);
    float3 impulse = normal * c.impulses.x
        + tangent_a * c.impulses.y
        + tangent_b * c.impulses.z;
    float3 r_a = c.position_pen.xyz - positions[idx_a].xyz;
    float3 r_b = c.position_pen.xyz - positions[idx_b].xyz;
    if (inv_mi_a.x > 1e-8f) {
        atomic_add_float3(&velocities[idx_a], impulse * inv_mi_a.x);
        atomic_add_float3(&angular_velocities[idx_a], cross(r_a, impulse) * inv_mi_a.yzw);
    }
    if (inv_mi_b.x > 1e-8f) {
        atomic_add_float3(&velocities[idx_b], -impulse * inv_mi_b.x);
        atomic_add_float3(&angular_velocities[idx_b], -cross(r_b, impulse) * inv_mi_b.yzw);
    }
}

// Projected Gauss-Seidel velocity solve with accumulated normal and two-axis
// Coulomb impulses. Position correction is deliberately separate so this work
// survives XPBD velocity reconstruction.
kernel void solve_contact_velocities(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device Contact* contacts [[buffer(4)]],
    device const uint* contact_counts [[buffer(5)]],
    device const float4* inv_mass_inertia [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    constant uint& iteration [[buffer(8)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    if (contact_id >= min(contact_counts[env_id], params.max_contacts)) return;
    uint contact_idx = env_id * params.max_contacts + contact_id;
    Contact c = contacts[contact_idx];
    if (c.position_pen.w < 0) return;

    uint idx_a = env_id * params.num_bodies + c.indices.x;
    uint idx_b = env_id * params.num_bodies + c.indices.y;
    float4 inv_mi_a = inv_mass_inertia[idx_a];
    float4 inv_mi_b = inv_mass_inertia[idx_b];
    if (inv_mi_a.x + inv_mi_b.x < 1e-8f) return;
    float3 r_a = c.position_pen.xyz - positions[idx_a].xyz;
    float3 r_b = c.position_pen.xyz - positions[idx_b].xyz;
    float3 vel_a = velocities[idx_a].xyz + cross(angular_velocities[idx_a].xyz, r_a);
    float3 vel_b = velocities[idx_b].xyz + cross(angular_velocities[idx_b].xyz, r_b);
    float3 relative_velocity = vel_a - vel_b;
    float3 normal = c.normal_friction.xyz;
    float normal_velocity = dot(relative_velocity, normal);
    float3 rn_a = cross(r_a, normal);
    float3 rn_b = cross(r_b, normal);
    float normal_mass = inv_mi_a.x + dot(rn_a * inv_mi_a.yzw, rn_a)
        + inv_mi_b.x + dot(rn_b * inv_mi_b.yzw, rn_b);
    if (normal_mass < 1e-8f) return;

    float target_normal_velocity;
    if (iteration == 0) {
        target_normal_velocity = normal_velocity < -0.5f
            ? -c.impulses.w * normal_velocity
            : 0.0f;
        contacts[contact_idx].impulses.w = target_normal_velocity;
    } else {
        target_normal_velocity = c.impulses.w;
    }
    float old_normal_impulse = c.impulses.x;
    float new_normal_impulse = max(
        old_normal_impulse + (target_normal_velocity - normal_velocity) / normal_mass,
        0.0f
    );
    float3 normal_impulse = normal * (new_normal_impulse - old_normal_impulse);

    float3 tangent_a = contact_tangent(normal);
    float3 tangent_b = cross(normal, tangent_a);
    float tangent_velocity_a = dot(relative_velocity, tangent_a);
    float tangent_velocity_b = dot(relative_velocity, tangent_b);
    float3 rt1_a = cross(r_a, tangent_a);
    float3 rt1_b = cross(r_b, tangent_a);
    float3 rt2_a = cross(r_a, tangent_b);
    float3 rt2_b = cross(r_b, tangent_b);
    float tangent_mass_a = inv_mi_a.x + dot(rt1_a * inv_mi_a.yzw, rt1_a)
        + inv_mi_b.x + dot(rt1_b * inv_mi_b.yzw, rt1_b);
    float tangent_mass_b = inv_mi_a.x + dot(rt2_a * inv_mi_a.yzw, rt2_a)
        + inv_mi_b.x + dot(rt2_b * inv_mi_b.yzw, rt2_b);
    float2 old_tangent_impulse = c.impulses.yz;
    float2 new_tangent_impulse = old_tangent_impulse;
    if (tangent_mass_a > 1e-8f) new_tangent_impulse.x -= tangent_velocity_a / tangent_mass_a;
    if (tangent_mass_b > 1e-8f) new_tangent_impulse.y -= tangent_velocity_b / tangent_mass_b;
    float friction_limit = c.normal_friction.w * new_normal_impulse;
    float tangent_magnitude = length(new_tangent_impulse);
    if (tangent_magnitude > friction_limit && tangent_magnitude > 1e-8f) {
        new_tangent_impulse *= friction_limit / tangent_magnitude;
    }
    float2 tangent_delta = new_tangent_impulse - old_tangent_impulse;
    float3 impulse = normal_impulse
        + tangent_a * tangent_delta.x
        + tangent_b * tangent_delta.y;
    if (inv_mi_a.x > 1e-8f) {
        atomic_add_float3(&velocities[idx_a], impulse * inv_mi_a.x);
        atomic_add_float3(&angular_velocities[idx_a], cross(r_a, impulse) * inv_mi_a.yzw);
    }
    if (inv_mi_b.x > 1e-8f) {
        atomic_add_float3(&velocities[idx_b], -impulse * inv_mi_b.x);
        atomic_add_float3(&angular_velocities[idx_b], -cross(r_b, impulse) * inv_mi_b.yzw);
    }
    contacts[contact_idx].impulses.x = new_normal_impulse;
    contacts[contact_idx].impulses.yz = new_tangent_impulse;
}

// ============================================================================
// Sensor Reading Kernel
// ============================================================================

kernel void read_sensors(
    device const float4* positions [[buffer(0)]],
    device const float4* velocities [[buffer(1)]],
    device const float4* quaternions [[buffer(2)]],
    device const float4* angular_velocities [[buffer(3)]],
    device const float* joint_positions [[buffer(4)]],
    device const float* joint_velocities [[buffer(5)]],
    device const SensorData* sensors [[buffer(6)]],
    device float* observations [[buffer(7)]],
    constant SimParams& params [[buffer(8)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_sensors = params.num_sensors;
    if (num_sensors == 0) return;

    uint dispatch_env_id = gid / num_sensors;
    uint sensor_id = gid % num_sensors;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    SensorData sensor = sensors[sensor_id];
    uint sensor_type = sensor.type_object.x;
    uint object_id = sensor.type_object.y;
    uint output_offset = uint(sensor.params.w);

    uint obs_base = env_id * params.obs_dim;

    switch (sensor_type) {
        case 0: { // joint_pos
            uint joint_idx = env_id * params.num_joints + object_id;
            observations[obs_base + output_offset] = joint_positions[joint_idx];
            break;
        }
        case 1: { // joint_vel
            uint joint_idx = env_id * params.num_joints + object_id;
            observations[obs_base + output_offset] = joint_velocities[joint_idx];
            break;
        }
        case 2: { // accelerometer
            // Simplified - just return gravity direction in body frame
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 quat = quaternions[body_idx];
            float3 gravity = float3(params.gravity_x, params.gravity_y, params.gravity_z);
            float3 local_gravity = rotate_by_quat(gravity, quat_conjugate(quat));
            observations[obs_base + output_offset + 0] = local_gravity.x;
            observations[obs_base + output_offset + 1] = local_gravity.y;
            observations[obs_base + output_offset + 2] = local_gravity.z;
            break;
        }
        case 3: { // gyro
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 omega = angular_velocities[body_idx];
            observations[obs_base + output_offset + 0] = omega.x;
            observations[obs_base + output_offset + 1] = omega.y;
            observations[obs_base + output_offset + 2] = omega.z;
            break;
        }
        case 7: { // framepos
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 pos = positions[body_idx];
            observations[obs_base + output_offset + 0] = pos.x;
            observations[obs_base + output_offset + 1] = pos.y;
            observations[obs_base + output_offset + 2] = pos.z;
            break;
        }
        case 8: { // framequat
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 quat = quaternions[body_idx];
            observations[obs_base + output_offset + 0] = quat.x;
            observations[obs_base + output_offset + 1] = quat.y;
            observations[obs_base + output_offset + 2] = quat.z;
            observations[obs_base + output_offset + 3] = quat.w;
            break;
        }
        case 9: { // framelinvel
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 vel = velocities[body_idx];
            observations[obs_base + output_offset + 0] = vel.x;
            observations[obs_base + output_offset + 1] = vel.y;
            observations[obs_base + output_offset + 2] = vel.z;
            break;
        }
        case 10: { // frameangvel
            uint body_idx = env_id * params.num_bodies + object_id;
            float4 omega = angular_velocities[body_idx];
            observations[obs_base + output_offset + 0] = omega.x;
            observations[obs_base + output_offset + 1] = omega.y;
            observations[obs_base + output_offset + 2] = omega.z;
            break;
        }
        default:
            break;
    }
}

// ============================================================================
// XPBD Save Previous State
// ============================================================================
// Save positions and quaternions before integration for XPBD velocity update.
// On the first substep, also snapshot the full-step starting velocity without
// adding another body-wide dispatch.

kernel void save_prev_state(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device float4* prev_positions [[buffer(2)]],
    device float4* prev_quaternions [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    device const float4* velocities [[buffer(5)]],
    device float4* step_start_velocities [[buffer(6)]],
    constant uint& snapshot_full_step [[buffer(7)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    prev_positions[gid] = positions[gid];
    prev_quaternions[gid] = quaternions[gid];
    if (snapshot_full_step != 0) {
        step_start_velocities[gid] = velocities[gid];
    }
}

// ============================================================================
// XPBD Velocity Update
// ============================================================================
// After constraint solving, derive velocities from position/quaternion changes.
// v = (x - x_prev) / dt
// omega = 2 * dq.xyz / dt (from quaternion difference)

kernel void xpbd_update_velocities(
    device const float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device const float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device const float4* prev_positions [[buffer(4)]],
    device const float4* prev_quaternions [[buffer(5)]],
    device const float4* inv_mass_inertia [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    device const float4* step_start_velocities [[buffer(8)]],
    device float4* accelerations [[buffer(9)]],
    constant float& inv_full_step [[buffer(10)]],
    constant uint& derive_acceleration [[buffer(11)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    gid = env_id * params.num_bodies + body_id;

    float4 inv_mi = inv_mass_inertia[gid];
    float inv_mass = inv_mi.x;

    // Skip static/kinematic bodies
    if (inv_mass < 1e-8) {
        if (derive_acceleration != 0) accelerations[gid] = float4(0.0f);
        return;
    }

    float dt = params.dt;
    float inv_dt = 1.0 / dt;

    // Linear velocity from position change
    float3 pos = positions[gid].xyz;
    float3 prev_pos = prev_positions[gid].xyz;
    velocities[gid] = float4((pos - prev_pos) * inv_dt, 0);

    if (derive_acceleration != 0) {
        accelerations[gid] = float4(
            (velocities[gid].xyz - step_start_velocities[gid].xyz) * inv_full_step,
            0.0f
        );
    }

    // Angular velocity from quaternion change
    // dq = q * q_prev^-1
    float4 q = quaternions[gid];
    float4 q_prev = prev_quaternions[gid];
    float4 dq = quat_multiply(q, quat_conjugate(q_prev));

    // Ensure positive hemisphere
    if (dq.w < 0) dq = -dq;

    // omega = 2 * dq.xyz / dt (small angle approximation)
    angular_velocities[gid] = float4(2.0 * dq.xyz * inv_dt, 0);
}

// Derive the public full-step acceleration only after all contact impulses have
// updated velocity. Running this inside xpbd_update_velocities omits the
// post-XPBD normal and friction response.
kernel void derive_accelerations(
    device const float4* velocities [[buffer(0)]],
    device const float4* step_start_velocities [[buffer(1)]],
    device float4* accelerations [[buffer(2)]],
    device const float4* inv_mass_inertia [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    constant float& inv_full_step [[buffer(5)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    uint body_index = env_id * params.num_bodies + body_id;
    if (inv_mass_inertia[body_index].x < 1e-8f) {
        accelerations[body_index] = float4(0.0f);
        return;
    }
    accelerations[body_index] = float4(
        (velocities[body_index].xyz - step_start_velocities[body_index].xyz)
            * inv_full_step,
        0.0f
    );
}

// ============================================================================
// GPU Task Evaluation
// ============================================================================

kernel void compute_task_outputs(
    device const float4* positions [[buffer(0)]],
    device const float4* velocities [[buffer(1)]],
    device const float* actions [[buffer(2)]],
    device float* rewards [[buffer(3)]],
    device uchar* dones [[buffer(4)]],
    device uint* episode_steps [[buffer(5)]],
    constant TaskParams& task [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    constant EnvDispatchParams& dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint dispatch_env_id [[thread_position_in_grid]]
) {
    if (dispatch_env_id >= dispatch.dispatch_envs || task.enabled == 0) return;

    uint env_id = physical_env_id(dispatch_env_id, dispatch, active_env_ids);
    uint root_index = env_id * params.num_bodies + task.root_body;
    float3 root_position = positions[root_index].xyz;
    float3 root_velocity = velocities[root_index].xyz;
    bool valid = all(isfinite(root_position)) && all(isfinite(root_velocity));

    float control_cost = 0.0f;
    uint action_start = env_id * params.num_actuators;
    for (uint actuator = 0; actuator < params.num_actuators; ++actuator) {
        float action = actions[action_start + actuator];
        if (!isfinite(action)) {
            valid = false;
        } else {
            control_cost += action * action;
        }
    }

    bool healthy = valid &&
        root_position.z >= task.healthy_z_min &&
        root_position.z <= task.healthy_z_max;
    float reward = task.forward_reward_weight * root_velocity[task.forward_axis]
        - task.control_cost_weight * control_cost
        + (healthy ? task.healthy_bonus : 0.0f);
    rewards[env_id] = valid && isfinite(reward) ? reward : 0.0f;

    uint next_step = episode_steps[env_id];
    if (next_step != 0xffffffffu) next_step += 1;
    episode_steps[env_id] = next_step;
    bool horizon_reached = task.max_episode_steps != 0 &&
        next_step >= task.max_episode_steps;
    dones[env_id] = uchar(horizon_reached ||
        (task.terminate_when_unhealthy != 0 && !healthy));
}

// ============================================================================
// Environment Reset Kernel
// ============================================================================

kernel void reset_bodies(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* accelerations [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device float4* quaternions [[buffer(4)]],
    device float4* forces [[buffer(5)]],
    device float4* torques [[buffer(6)]],
    device float4* prev_positions [[buffer(7)]],
    device float4* prev_quaternions [[buffer(8)]],
    device float4* prev_velocities [[buffer(9)]],
    device const float4* initial_positions [[buffer(10)]],
    device const float4* initial_velocities [[buffer(11)]],
    device const float4* initial_quaternions [[buffer(12)]],
    device const float4* initial_angular_velocities [[buffer(13)]],
    constant SimParams& params [[buffer(14)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    uint dispatch_env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);
    uint body_index = env_id * params.num_bodies + body_id;

    // Copy initial state
    positions[body_index] = initial_positions[body_id];
    velocities[body_index] = initial_velocities[body_id];
    accelerations[body_index] = float4(0);
    angular_velocities[body_index] = initial_angular_velocities[body_id];
    quaternions[body_index] = initial_quaternions[body_id];
    forces[body_index] = float4(0);
    torques[body_index] = float4(0);
    prev_positions[body_index] = initial_positions[body_id];
    prev_quaternions[body_index] = initial_quaternions[body_id];
    prev_velocities[body_index] = initial_velocities[body_id];
}

kernel void reset_env_aux(
    device float* joint_positions [[buffer(0)]],
    device float* joint_velocities [[buffer(1)]],
    device float* joint_torques [[buffer(2)]],
    device float* actions [[buffer(3)]],
    device float* observations [[buffer(4)]],
    device float* rewards [[buffer(5)]],
    device uchar* dones [[buffer(6)]],
    device Contact* contacts [[buffer(7)]],
    device uint* contact_counts [[buffer(8)]],
    device Contact* prev_contacts [[buffer(9)]],
    device uint* prev_contact_counts [[buffer(10)]],
    device XPBDConstraint* constraints [[buffer(11)]],
    device uint* episode_steps [[buffer(12)]],
    device const float* initial_joint_positions [[buffer(13)]],
    device const float* initial_joint_velocities [[buffer(14)]],
    constant SimParams& params [[buffer(15)]],
    constant EnvDispatchParams& env_dispatch [[buffer(29)]],
    device const uint* active_env_ids [[buffer(30)]],
    uint dispatch_env_id [[thread_position_in_grid]]
) {
    if (dispatch_env_id >= env_dispatch.dispatch_envs) return;
    uint env_id = physical_env_id(dispatch_env_id, env_dispatch, active_env_ids);

    uint joint_start = env_id * params.num_joints;
    for (uint joint = 0; joint < params.num_joints; ++joint) {
        joint_positions[joint_start + joint] = initial_joint_positions[joint];
        joint_velocities[joint_start + joint] = initial_joint_velocities[joint];
        joint_torques[joint_start + joint] = 0.0f;
    }
    // constraint_offset is a private reset-dispatch flag here. Fused autoreset
    // runs after the host has copied the next action batch, so preserve it.
    if (params.constraint_offset == 0) {
        uint action_start = env_id * params.num_actuators;
        for (uint actuator = 0; actuator < params.num_actuators; ++actuator) {
            actions[action_start + actuator] = 0.0f;
        }
    }
    uint obs_start = env_id * params.obs_dim;
    for (uint obs = 0; obs < params.obs_dim; ++obs) {
        observations[obs_start + obs] = 0.0f;
    }

    uint contact_start = env_id * params.max_contacts;
    for (uint contact_id = 0; contact_id < params.max_contacts; ++contact_id) {
        uint index = contact_start + contact_id;
        contacts[index].position_pen = float4(0, 0, 0, -1);
        contacts[index].normal_friction = float4(0);
        contacts[index].indices = uint4(0);
        contacts[index].impulses = float4(0);
        prev_contacts[index].position_pen = float4(0, 0, 0, -1);
        prev_contacts[index].normal_friction = float4(0);
        prev_contacts[index].indices = uint4(0);
        prev_contacts[index].impulses = float4(0);
    }

    uint constraints_per_env = params.target_color;
    uint constraint_start = env_id * constraints_per_env;
    for (uint constraint_id = 0; constraint_id < constraints_per_env; ++constraint_id) {
        constraints[constraint_start + constraint_id].state = float4(0);
    }

    rewards[env_id] = 0.0f;
    dones[env_id] = 0;
    episode_steps[env_id] = 0;
    contact_counts[env_id] = 0;
    prev_contact_counts[env_id] = 0;
}
