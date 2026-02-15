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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_joints;
    uint joint_id = gid % params.num_joints;

    if (env_id >= params.num_envs) return;

    uint joint_idx = gid;
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
    uint gid [[thread_position_in_grid]]
) {
    uint num_actuators = params.num_actuators;
    if (num_actuators == 0) return;

    uint env_id = gid / num_actuators;
    uint act_id = gid % num_actuators;

    if (env_id >= params.num_envs) return;

    ActuatorData act = actuators[act_id];
    uint joint_id = uint(act.params.x);
    float ctrl_min = act.params.y;
    float ctrl_max = act.params.z;
    float gear = act.params.w;

    float ctrl = actions[gid];
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;

    if (env_id >= params.num_envs) return;

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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;

    if (env_id >= params.num_envs) return;

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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;

    if (env_id >= params.num_envs) return;

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
    float3 gravity = float3(params.gravity_x, params.gravity_y, params.gravity_z);
    float3 force = gravity * mass;

    forces[gid] = float4(forces[gid].xyz + force, 0);

    // Gravitational torque from COM offset.
    // When the body frame origin differs from the center of mass,
    // gravity acting at the COM creates a torque about the body origin:
    // τ = r_com_world × (m * g)
    float3 com_local = body_data[body_id].com_offset.xyz;
    float3 com_world = rotate_by_quat(com_local, quaternions[gid]);
    float3 grav_torque = cross(com_world, force);
    torques[gid] = float4(torques[gid].xyz + grav_torque, 0);
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    if (env_id >= params.num_envs) return;

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

    // Velocity damping
    vel *= 0.999;

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
    omega *= 0.999; // Damping

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

kernel void broad_phase(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device const GeomData* geoms [[buffer(2)]],
    device Contact* contacts [[buffer(3)]],
    device atomic_uint* contact_counts [[buffer(4)]],
    constant SimParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_geoms;
    uint geom_id = gid % params.num_geoms;

    if (env_id >= params.num_envs) return;

    GeomData geom_a = geoms[geom_id];
    uint body_a = geom_a.type_body.y;
    uint body_idx_a = env_id * params.num_bodies + body_a;

    float3 pos_a = positions[body_idx_a].xyz + geom_a.pos_size0.xyz;
    float radius_a = geom_a.pos_size0.w;

    // Check against all other geoms (simple O(n²) for now)
    for (uint other = geom_id + 1; other < params.num_geoms; other++) {
        GeomData geom_b = geoms[other];
        uint body_b = geom_b.type_body.y;

        // Skip self-collision
        if (body_a == body_b) continue;

        uint body_idx_b = env_id * params.num_bodies + body_b;
        float3 pos_b = positions[body_idx_b].xyz + geom_b.pos_size0.xyz;
        float radius_b = geom_b.pos_size0.w;

        // AABB test (simplified as sphere-sphere)
        float3 diff = pos_b - pos_a;
        float dist_sq = dot(diff, diff);
        float min_dist = radius_a + radius_b + 0.1; // Margin

        if (dist_sq < min_dist * min_dist) {
            // Potential collision - add to narrow phase
            uint count = atomic_fetch_add_explicit(
                &contact_counts[env_id], 1, memory_order_relaxed);

            if (count < params.max_contacts) {
                uint contact_idx = env_id * params.max_contacts + count;
                contacts[contact_idx].indices = uint4(body_a, body_b, geom_id, other);
            }
        }
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;

    if (env_id >= params.num_envs) return;

    uint count = contact_counts[env_id];
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

    float3 pos_a = positions[body_idx_a].xyz + geom_a.pos_size0.xyz;
    float3 pos_b = positions[body_idx_b].xyz + geom_b.pos_size0.xyz;

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

        float3 diff = pos_b - pos_a;
        float dist = length(diff);
        float min_dist = radius_a + radius_b;

        if (dist < min_dist && dist > 1e-6) {
            normal = diff / dist;
            penetration = min_dist - dist;
            contact_point = pos_a + normal * (radius_a - penetration * 0.5);
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
            plane_normal = float3(0, 0, 1); // Assume Z-up plane
        } else {
            sphere_pos = pos_b;
            sphere_radius = geom_b.pos_size0.w;
            plane_pos = pos_a;
            plane_normal = float3(0, 0, 1);
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
        float3 plane_normal = float3(0, 0, 1);

        if (type_a == 1) {
            capsule_pos = pos_a;
            capsule_radius = geom_a.pos_size0.w;
            capsule_half_len = geom_a.params.x;
            capsule_quat = geom_a.quat;
        } else {
            capsule_pos = pos_b;
            capsule_radius = geom_b.pos_size0.w;
            capsule_half_len = geom_b.params.x;
            capsule_quat = geom_b.quat;
        }

        // Capsule axis
        float3 axis = rotate_by_quat(float3(0, 0, 1), capsule_quat);
        float3 p1 = capsule_pos - axis * capsule_half_len;
        float3 p2 = capsule_pos + axis * capsule_half_len;

        // Check both endpoints
        float d1 = p1.z - capsule_radius;
        float d2 = p2.z - capsule_radius;

        if (d1 < 0 || d2 < 0) {
            float3 deepest = d1 < d2 ? p1 : p2;
            float dist = min(d1, d2);

            normal = type_a == 1 ? plane_normal : -plane_normal;
            penetration = -dist;
            contact_point = deepest;
            contact_point.z = 0;
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

    uint env_id = gid / count;
    uint local_idx = gid % count;

    if (env_id >= params.num_envs) return;

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
    float3 grad_a = float3(0);
    float3 grad_b = float3(0);
    
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
    uint gid [[thread_position_in_grid]]
) {
    uint constraints_per_env = params.target_color; // Repurposed field
    if (constraints_per_env == 0) return;

    uint env_id = gid / constraints_per_env;
    uint local_idx = gid % constraints_per_env;

    if (env_id >= params.num_envs) return;

    uint idx = env_id * constraints_per_env + local_idx;
    // state.x = lambda, state.y = lambda_prev
    // Initialize lambda from previous frame's lambda (warm start)
    constraints[idx].state.x = warm_start_factor * constraints[idx].state.y;
}

kernel void store_lambda_prev(
    device XPBDConstraint* constraints [[buffer(0)]],
    constant SimParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint constraints_per_env = params.target_color; // Repurposed field
    if (constraints_per_env == 0) return;

    uint env_id = gid / constraints_per_env;
    uint local_idx = gid % constraints_per_env;

    if (env_id >= params.num_envs) return;

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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;

    if (env_id >= params.num_envs) return;

    uint count = contact_counts[env_id];

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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;

    if (env_id >= params.num_envs) return;

    uint count = contact_counts[env_id];
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_joints;
    uint joint_id = gid % params.num_joints;
    
    if (env_id >= params.num_envs) return;
    
    uint joint_idx = gid;
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.max_contacts;
    uint contact_id = gid % params.max_contacts;

    if (env_id >= params.num_envs) return;

    uint count = contact_counts[env_id];
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
    float friction = c.normal_friction.w;
    float3 contact_point = c.position_pen.xyz;

    // Get current state
    float3 pos_a = positions[idx_a].xyz;
    float3 pos_b = positions[idx_b].xyz;
    float4 quat_a = quaternions[idx_a];
    float4 quat_b = quaternions[idx_b];

    // Compute contact point relative to body centers
    float3 r_a = contact_point - pos_a;
    float3 r_b = contact_point - pos_b;

    // Get angular velocities
    float3 omega_a = angular_velocities[idx_a].xyz;
    float3 omega_b = angular_velocities[idx_b].xyz;

    // Compute full contact velocity including rotation: v = v_cm + ω × r
    float3 vel_a = velocities[idx_a].xyz + cross(omega_a, r_a);
    float3 vel_b = velocities[idx_b].xyz + cross(omega_b, r_b);
    float3 rel_vel = vel_a - vel_b;

    // Normal and tangential components
    float vel_normal = dot(rel_vel, normal);
    float3 vel_tangent = rel_vel - vel_normal * normal;
    float tangent_speed = length(vel_tangent);

    // Compute generalized inverse mass for normal direction
    float3 rn_a = cross(r_a, normal);
    float3 rn_b = cross(r_b, normal);
    float w_n_a = inv_mass_a + dot(rn_a * inv_mi_a.yzw, rn_a);
    float w_n_b = inv_mass_b + dot(rn_b * inv_mi_b.yzw, rn_b);
    float w_normal = w_n_a + w_n_b;

    if (w_normal < 1e-8) return;

    // Baumgarte stabilization
    float bias = params.baumgarte * max(penetration - params.slop, 0.0f) / params.dt;

    // Restitution (only applied for separating velocity)
    float restitution = c.impulses.w;
    float vel_restitution = vel_normal < -0.5 ? restitution * vel_normal : 0.0;

    // Normal impulse magnitude
    float j_n = (-(vel_normal + vel_restitution) + bias) / w_normal;
    j_n = max(j_n, 0.0f); // Only push apart (unilateral constraint)

    // Apply normal impulse
    float3 impulse_n = j_n * normal;

    if (inv_mass_a > 1e-8) {
        velocities[idx_a].xyz += impulse_n * inv_mass_a;
        angular_velocities[idx_a].xyz += cross(r_a, impulse_n) * inv_mi_a.yzw;

        // Position correction
        float mass_ratio_a = inv_mass_a / inv_mass_sum;
        positions[idx_a].xyz += penetration * mass_ratio_a * normal;
    }

    if (inv_mass_b > 1e-8) {
        velocities[idx_b].xyz -= impulse_n * inv_mass_b;
        angular_velocities[idx_b].xyz -= cross(r_b, impulse_n) * inv_mi_b.yzw;

        // Position correction
        float mass_ratio_b = inv_mass_b / inv_mass_sum;
        positions[idx_b].xyz -= penetration * mass_ratio_b * normal;
    }

    // Friction (Coulomb friction cone)
    // Use accumulated normal impulse for friction cone, not just this iteration's delta
    float j_n_accumulated = contacts[contact_idx].impulses.x + j_n;

    if (tangent_speed > 1e-6 && j_n_accumulated > 1e-6) {
        float3 tangent = vel_tangent / tangent_speed;

        // Compute generalized inverse mass for tangent direction
        float3 rt_a = cross(r_a, tangent);
        float3 rt_b = cross(r_b, tangent);
        float w_t_a = inv_mass_a + dot(rt_a * inv_mi_a.yzw, rt_a);
        float w_t_b = inv_mass_b + dot(rt_b * inv_mi_b.yzw, rt_b);
        float w_tangent = w_t_a + w_t_b;

        if (w_tangent > 1e-8) {
            // Desired friction impulse to stop tangential motion
            float j_t_desired = tangent_speed / w_tangent;

            // Friction cone limit: |j_t| ≤ μ * j_n_accumulated
            float j_t_max = friction * j_n_accumulated;
            float j_t = min(j_t_desired, j_t_max);

            // Apply friction impulse
            float3 impulse_t = j_t * tangent;

            if (inv_mass_a > 1e-8) {
                velocities[idx_a].xyz -= impulse_t * inv_mass_a;
                angular_velocities[idx_a].xyz -= cross(r_a, impulse_t) * inv_mi_a.yzw;
            }
            if (inv_mass_b > 1e-8) {
                velocities[idx_b].xyz += impulse_t * inv_mass_b;
                angular_velocities[idx_b].xyz += cross(r_b, impulse_t) * inv_mi_b.yzw;
            }
        }
    }

    // Store accumulated normal impulse for warm starting
    contacts[contact_idx].impulses.x = j_n;
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
    uint gid [[thread_position_in_grid]]
) {
    uint num_sensors = params.num_sensors;
    if (num_sensors == 0) return;

    uint env_id = gid / num_sensors;
    uint sensor_id = gid % num_sensors;

    if (env_id >= params.num_envs) return;

    SensorData sensor = sensors[sensor_id];
    uint sensor_type = sensor.type_object.x;
    uint object_id = sensor.type_object.y;
    uint dim = sensor.type_object.z;
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

kernel void save_prev_state(
    device const float4* positions [[buffer(0)]],
    device const float4* quaternions [[buffer(1)]],
    device float4* prev_positions [[buffer(2)]],
    device float4* prev_quaternions [[buffer(3)]],
    constant SimParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    if (env_id >= params.num_envs) return;

    prev_positions[gid] = positions[gid];
    prev_quaternions[gid] = quaternions[gid];
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
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    if (env_id >= params.num_envs) return;

    float4 inv_mi = inv_mass_inertia[gid];
    float inv_mass = inv_mi.x;

    // Skip static/kinematic bodies
    if (inv_mass < 1e-8) return;

    float dt = params.dt;
    float inv_dt = 1.0 / dt;

    // Linear velocity from position change
    float3 pos = positions[gid].xyz;
    float3 prev_pos = prev_positions[gid].xyz;
    velocities[gid] = float4((pos - prev_pos) * inv_dt, 0);

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

// ============================================================================
// Environment Reset Kernel
// ============================================================================

kernel void reset_env(
    device float4* positions [[buffer(0)]],
    device float4* velocities [[buffer(1)]],
    device float4* quaternions [[buffer(2)]],
    device float4* angular_velocities [[buffer(3)]],
    device const float4* initial_positions [[buffer(4)]],
    device const float4* initial_quaternions [[buffer(5)]],
    device const uint* reset_mask [[buffer(6)]],
    constant SimParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint env_id = gid / params.num_bodies;
    uint body_id = gid % params.num_bodies;

    if (env_id >= params.num_envs) return;

    // Check if this env should be reset
    if (reset_mask[env_id] == 0) return;

    // Copy initial state
    positions[gid] = initial_positions[body_id];
    velocities[gid] = float4(0);
    quaternions[gid] = initial_quaternions[body_id];
    angular_velocities[gid] = float4(0);
}
