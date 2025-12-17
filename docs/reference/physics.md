# Zeno Physics Model

## Rigid Body Dynamics

### State Representation

Zeno uses maximal coordinates where each body has independent state:

- **Position**: 3D vector (x, y, z)
- **Orientation**: Unit quaternion (x, y, z, w)
- **Linear velocity**: 3D vector
- **Angular velocity**: 3D vector in world frame

### Integration Method

Semi-implicit (symplectic) Euler integration:

```
// Linear
v(t+dt) = v(t) + a(t) * dt
x(t+dt) = x(t) + v(t+dt) * dt

// Angular
ω(t+dt) = ω(t) + I⁻¹ * τ * dt
q(t+dt) = normalize(q(t) + 0.5 * ω_quat * q(t) * dt)
```

Where `ω_quat = (ωx, ωy, ωz, 0)` is the angular velocity as a quaternion.

This method is:
- First-order accurate
- Symplectic (conserves energy approximately)
- Unconditionally stable for damped systems
- Very fast (single force evaluation per step)

## Joint Constraints

### Supported Joint Types

| Type | DOF | Description |
|------|-----|-------------|
| `fixed` | 0 | Bodies rigidly attached |
| `revolute` | 1 | Rotation around single axis (hinge) |
| `prismatic` | 1 | Translation along single axis (slider) |
| `ball` | 3 | Free rotation (spherical joint) |
| `free` | 6 | No constraint (floating base) |

### Constraint Formulation

Joints are enforced using Position-Based Dynamics (PBD):

1. Compute constraint violation: `C(x)`
2. Compute gradient: `∇C`
3. Project positions: `Δx = -λ * ∇C`
4. Update velocities: `v = (x_new - x_old) / dt`

The Lagrange multiplier `λ` is computed as:

```
λ = C(x) / (∇C · W · ∇C^T + α/dt²)
```

Where:
- `W` is the inverse mass matrix
- `α` is compliance (inverse stiffness)

### Joint Limits

Angular and linear limits are enforced as inequality constraints:

```
if angle < limit_lower:
    apply_impulse(limit_lower - angle)
if angle > limit_upper:
    apply_impulse(limit_upper - angle)
```

## Collision Detection

### Broad Phase: Spatial Hashing

World space is divided into a uniform grid. Each geometry is assigned to cells it overlaps:

```
cell_id = hash(floor(position / cell_size))
```

Potential collision pairs are found by checking geometries in the same or adjacent cells.

Properties:
- O(n) update time
- O(1) query per cell
- Memory: O(grid_size³ + n)

### Narrow Phase: Primitive Tests

| Pair | Algorithm | Complexity |
|------|-----------|------------|
| Sphere-Sphere | Distance check | O(1) |
| Sphere-Capsule | Point-segment distance | O(1) |
| Sphere-Plane | Point-plane distance | O(1) |
| Capsule-Capsule | Segment-segment distance | O(1) |
| Box-Plane | 8 vertex tests | O(1) |
| Box-Box | SAT (15 axes) | O(1) |

### Contact Data

Each contact stores:
- Position (world space)
- Normal (from A to B)
- Penetration depth
- Body/geom indices
- Material properties (friction, restitution)

## Contact Resolution

### Position Correction

PBD projection removes interpenetration:

```
Δx_a = -d * n * (m_inv_a / (m_inv_a + m_inv_b))
Δx_b = +d * n * (m_inv_b / (m_inv_a + m_inv_b))
```

Where:
- `d` is penetration depth
- `n` is contact normal
- `m_inv` is inverse mass

### Velocity Correction

Impact impulse with restitution:

```
v_rel = v_a - v_b
v_n = dot(v_rel, n)

if v_n < 0:  // Approaching
    j = -(1 + e) * v_n / (m_inv_a + m_inv_b)
    v_a += j * m_inv_a * n
    v_b -= j * m_inv_b * n
```

### Friction

Coulomb friction with clamping:

```
v_t = v_rel - v_n * n  // Tangent velocity
if length(v_t) > ε:
    j_friction = min(μ * |j_normal|, length(v_t) / (m_inv_a + m_inv_b))
    v_a -= j_friction * normalize(v_t) * m_inv_a
    v_b += j_friction * normalize(v_t) * m_inv_b
```

## Solver Configuration

### Recommended Settings

| Parameter | Default | Fast | Accurate |
|-----------|---------|------|----------|
| `timestep` | 0.002 | 0.01 | 0.001 |
| `contact_iterations` | 4 | 2 | 8 |
| `substeps` | 1 | 1 | 4 |
| `baumgarte` | 0.2 | 0.3 | 0.1 |

### Stability Tips

1. **Use smaller timesteps** for stiff systems
2. **Increase iterations** for complex contact scenes
3. **Add damping** to reduce high-frequency oscillations
4. **Use substeps** for highly dynamic scenarios

## Actuators

### Motor Actuator

Direct torque control:
```
torque = gear * clamp(ctrl, ctrl_min, ctrl_max)
```

### Position Actuator

PD controller:
```
torque = kp * (target - q) - kv * qdot
```

### Velocity Actuator

Velocity servo:
```
torque = kv * (target_vel - qdot)
```

## Sensors

### Available Sensor Types

| Type | Output Dim | Description |
|------|------------|-------------|
| `jointpos` | 1 | Joint angle/position |
| `jointvel` | 1 | Joint velocity |
| `accelerometer` | 3 | Linear acceleration |
| `gyro` | 3 | Angular velocity |
| `framepos` | 3 | Body position |
| `framequat` | 4 | Body orientation |
| `framelinvel` | 3 | Body linear velocity |
| `frameangvel` | 3 | Body angular velocity |

### Observation Computation

Sensors are read after physics stepping, providing consistent snapshots of the post-step state.
