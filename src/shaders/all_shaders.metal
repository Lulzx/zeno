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
};

struct BodyData {
    float4 position;
    float4 quaternion;
    float4 inv_mass_inertia;
    float4 params;
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

    // Gravity
    float3 gravity = float3(params.gravity_x, params.gravity_y, params.gravity_z);
    float3 force = gravity * mass;

    forces[gid] = float4(force, 0);
    torques[gid] = float4(0);
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
    quat = quat_normalize(quat + dq);

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

    // Position correction (Baumgarte stabilization)
    float bias = params.baumgarte * max(penetration - params.slop, 0.0f) / params.dt;

    // Relative velocity
    float3 vel_a = velocities[idx_a].xyz;
    float3 vel_b = velocities[idx_b].xyz;
    float3 rel_vel = vel_a - vel_b;
    float vel_normal = dot(rel_vel, normal);

    // Normal impulse
    float restitution = c.impulses.w;
    float j = -(1.0 + restitution) * vel_normal + bias;
    j /= inv_mass_sum;
    j = max(j, 0.0f); // Only push apart

    // Apply impulse
    if (inv_mass_a > 1e-8) {
        velocities[idx_a] = float4(vel_a + j * inv_mass_a * normal, 0);

        // Position correction
        float3 pos_a = positions[idx_a].xyz;
        pos_a += penetration * inv_mass_a / inv_mass_sum * normal;
        positions[idx_a] = float4(pos_a, 0);
    }

    if (inv_mass_b > 1e-8) {
        velocities[idx_b] = float4(vel_b - j * inv_mass_b * normal, 0);

        // Position correction
        float3 pos_b = positions[idx_b].xyz;
        pos_b -= penetration * inv_mass_b / inv_mass_sum * normal;
        positions[idx_b] = float4(pos_b, 0);
    }

    // Friction
    float3 tangent_vel = rel_vel - vel_normal * normal;
    float tangent_speed = length(tangent_vel);

    if (tangent_speed > 1e-6) {
        float3 tangent = tangent_vel / tangent_speed;
        float friction_impulse = min(friction * abs(j), tangent_speed / inv_mass_sum);

        if (inv_mass_a > 1e-8) {
            float3 v = velocities[idx_a].xyz;
            velocities[idx_a] = float4(v - friction_impulse * inv_mass_a * tangent, 0);
        }
        if (inv_mass_b > 1e-8) {
            float3 v = velocities[idx_b].xyz;
            velocities[idx_b] = float4(v + friction_impulse * inv_mass_b * tangent, 0);
        }
    }
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

    uint obs_base = env_id * params.num_sensors; // Simplified - should use total obs dim

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
