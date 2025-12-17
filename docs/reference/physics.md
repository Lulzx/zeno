# Zeno Physics Model

## Rigid Body Dynamics

### State Representation

Zeno uses maximal coordinates where each body has independent state:

- **Position**: 3D vector (x, y, z)
- **Orientation**: Unit quaternion (x, y, z, w)
- **Linear velocity**: 3D vector
- **Angular velocity**: 3D vector in world frame

### Simulation Pipeline

Each physics step executes the following stages:

| Stage | Description |
|-------|-------------|
| 1. Apply Actions | Convert control inputs to joint torques via actuators |
| 1.5. Apply Joint Forces | Map joint torques to body forces using action/reaction |
| 2. Update Kinematic | Update kinematic body positions from their velocities |
| 3. Compute Forces | Compute gravity, damping, and external forces |
| 4. Integrate | Semi-implicit Euler integration |
| 5. Broad Phase | Spatial hashing for collision candidate pairs |
| 6. Narrow Phase | Precise contact generation |
| 7. Solve Joints | XPBD constraint iterations for joint constraints |
| 8. Solve Contacts | XPBD iterations for contact constraints |
| 9. Update Joint States | Compute joint angles/velocities from body state |
| 10. Read Sensors | Generate observations from simulation state |

### Body Types

| Type | Description | Behavior |
|------|-------------|----------|
| `dynamic` | Fully simulated | Affected by forces, gravity, collisions |
| `kinematic` | Animation-driven | Follows specified velocity, infinite mass |
| `static` | Immovable | Zero velocity, infinite mass |

Kinematic bodies are useful for:
- Moving platforms
- Animated obstacles
- User-controlled objects

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

### XPBD Constraint Solver

Zeno uses **Extended Position-Based Dynamics (XPBD)** for constraint solving, which provides:
- Unified handling of joints and contacts
- Proper physics with compliance and damping
- Stable simulation at large timesteps
- Better energy conservation than standard PBD

#### XPBD Update Equation

The core XPBD update computes position corrections via Lagrange multipliers:

```
λ = (-C - α̃ * λ_prev) / (w + α̃)
Δx = λ * ∇C
```

Where:
- `C` is the constraint violation
- `α̃ = compliance / dt²` is the time-scaled compliance
- `w = Σ(m_inv * |∇C|²)` is the generalized inverse mass
- `λ` is the accumulated Lagrange multiplier

#### Constraint Types

| Type | Code | Description |
|------|------|-------------|
| Contact Normal | 0 | Non-penetration constraint |
| Contact Friction | 1 | Tangential friction |
| Positional | 2 | Point-to-point (ball joint anchor) |
| Angular | 3 | Axis alignment (hinge constraint) |
| Angular Limit | 4 | Joint rotation limits |
| Linear Limit | 5 | Prismatic joint limits |
| Tendon | 6 | Cable/muscle length constraint |
| Weld | 7 | Fixed relative pose |
| Connect | 8 | Distance between anchor points |
| Joint Equality | 9 | Coupled joint positions |

#### Joint Decomposition

High-level joint types are decomposed into primitive constraints:

| Joint Type | Primitive Constraints |
|-----------|----------------------|
| Fixed | Weld (position + orientation) |
| Revolute | Point + Angular + Angular Limit (optional) |
| Prismatic | Slider + Linear Limit (optional) |
| Ball | Point + Cone Limit (optional) |
| Universal | Point + 2× Angular |
| Free | None |

#### Warm Starting

XPBD uses warm starting for faster convergence:
- Accumulated impulses (λ) are cached between frames
- Initial guess uses `λ_prev` from the previous timestep
- Significantly improves stability for stiff constraints

#### Compliance and Damping

Soft constraints are achieved via compliance:
- `compliance = 0`: Rigid constraint
- `compliance > 0`: Soft constraint (spring-like behavior)
- `damping`: Velocity-dependent resistance

Example configurations:
```
// Rigid joint
compliance = 0.0, damping = 0.0

// Soft spring joint
compliance = 1e-4, damping = 0.01

// Very soft (rubber-like)
compliance = 1e-2, damping = 0.1
```

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
| Sphere-Box | Closest point on box | O(1) |
| Sphere-Heightfield | Bilinear height sampling | O(1) |
| Capsule-Capsule | Segment-segment distance | O(1) |
| Capsule-Plane | Endpoint tests | O(1) |
| Capsule-Heightfield | Endpoint height sampling | O(1) |
| Box-Plane | 8 vertex tests | O(1) |
| Box-Box | SAT (15 axes) | O(1) |
| Box-Heightfield | 8 vertex height tests | O(1) |

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

## Heightfield Terrain

Heightfield collision uses a regular grid of height samples for efficient terrain representation.

### Data Structure

```zig
pub const Heightfield = struct {
    data: []const f32,    // Height samples (row-major)
    rows: u32,            // Grid rows (Y direction)
    cols: u32,            // Grid columns (X direction)
    spacing_x: f32,       // Grid cell size X
    spacing_y: f32,       // Grid cell size Y
    base_height: f32,     // Height offset
    height_scale: f32,    // Height multiplier
};
```

### Height Sampling

Height at any XY position is computed via bilinear interpolation:

```
grid_x = (world_x + half_width) / spacing_x
grid_y = (world_y + half_height) / spacing_y

h00, h10, h01, h11 = grid_heights[ix:ix+1, iy:iy+1]
height = bilerp(h00, h10, h01, h11, fx, fy)
```

### Normal Computation

Surface normals are computed using central differences:

```
dx = (h[x+1,y] - h[x-1,y]) / (2 * spacing_x)
dy = (h[x,y+1] - h[x,y-1]) / (2 * spacing_y)
normal = normalize(-dx, -dy, 1)
```

## Tendons

Tendons connect multiple joints or sites with spring-like behavior.

### Fixed Tendons

A fixed tendon computes its length as a linear combination of joint positions:

```
length = Σ (coef_i * joint_pos_i)
```

The tendon applies forces to maintain rest length:

```
force = stiffness * (length - rest_length) + damping * length_velocity
```

### Spatial Tendons

Spatial tendons route through a series of via-points (sites):

```
length = Σ |site[i+1] - site[i]|
```

They support wrapping around geometric objects (spheres, cylinders).

## Equality Constraints

Equality constraints enforce relationships between bodies, joints, or tendons.

### Weld Constraint

Rigidly attaches two bodies:

```
C_pos = pos_b - pos_a - R_a * anchor
C_rot = quat_error(quat_b, quat_a * rel_quat)
```

### Connect Constraint

Connects bodies at anchor points (allows relative rotation):

```
C = pos_b + R_b * anchor_b - pos_a - R_a * anchor_a
```

### Joint Constraint

Couples joint positions via polynomial:

```
C = joint2 - Σ (polycoef_i * joint1^i)
```

### Solver Integration

All equality constraints are solved using XPBD (Extended Position-Based Dynamics) with:
- Warm starting for faster convergence
- Compliance for soft constraints
- Configurable iteration count

## Soft Bodies

Zeno supports deformable soft bodies using Position-Based Dynamics (PBD).

### Particle Representation

Soft bodies are composed of particles connected by constraints:

```zig
pub const Particle = struct {
    position: [3]f32,        // Current position
    prev_position: [3]f32,   // Previous position (for velocity)
    velocity: [3]f32,        // Current velocity
    inv_mass: f32,           // Inverse mass (0 = pinned)
};
```

### Distance Constraints

Distance constraints maintain rest lengths between particle pairs:

```
C = |p1 - p2| - rest_length

// Position correction
Δp = w * C * (p1 - p2) / |p1 - p2|
p1 -= Δp * w1 / (w1 + w2)
p2 += Δp * w2 / (w1 + w2)
```

Where `w` is the inverse mass weighting.

### Volume Constraints

For volumetric soft bodies, tetrahedral volume constraints preserve shape:

```
V = (1/6) * dot(p2-p1, cross(p3-p1, p4-p1))
C = V - rest_volume

// Gradient-based correction applied to all 4 particles
```

### Soft Body Types

| Type | Description | Constraints |
|------|-------------|-------------|
| Cloth | 2D sheet | Distance (structural, shear, bend) |
| Volumetric | 3D tetrahedral mesh | Distance + Volume |

### Factory Methods

```zig
// Create a cube of soft particles
SoftBody.createCube(center, size, resolution, mass, stiffness)

// Create a cloth sheet
SoftBody.createCloth(corner, width, height, res_x, res_y, mass, stiffness)
```

### Simulation Loop

```
for substep in 0..substeps:
    // Apply external forces (gravity)
    for particle in particles:
        particle.velocity += gravity * dt
        particle.prev_position = particle.position
        particle.position += particle.velocity * dt

    // Constraint projection
    for iteration in 0..solver_iterations:
        for constraint in distance_constraints:
            solve_distance(constraint)
        for constraint in volume_constraints:
            solve_volume(constraint)

    // Velocity update
    for particle in particles:
        particle.velocity = (particle.position - particle.prev_position) / dt
```

## Fluid Simulation

Zeno implements Smoothed Particle Hydrodynamics (SPH) for fluid simulation.

### Particle Properties

```zig
pub const FluidParticle = struct {
    position: [3]f32,     // World position
    velocity: [3]f32,     // Velocity
    density: f32,         // Computed density
    pressure: f32,        // Computed pressure
};
```

### SPH Kernels

The simulation uses cubic spline kernels:

**Poly6 Kernel (Density):**
```
W(r, h) = (315 / 64πh⁹) * (h² - r²)³  for r ≤ h
```

**Spiky Kernel Gradient (Pressure):**
```
∇W(r, h) = -(45 / πh⁶) * (h - r)² * (r / |r|)
```

**Viscosity Kernel Laplacian:**
```
∇²W(r, h) = (45 / πh⁶) * (h - r)
```

### Fluid Parameters

```zig
pub const FluidParams = struct {
    rest_density: f32 = 1000.0,     // kg/m³ (water)
    gas_constant: f32 = 2000.0,     // Pressure stiffness
    viscosity: f32 = 0.001,         // Dynamic viscosity
    smoothing_radius: f32 = 0.1,    // SPH kernel radius
    particle_mass: f32 = 0.02,      // Mass per particle
    gravity: [3]f32 = .{0, 0, -9.81},
};
```

### Neighbor Search

Spatial hashing accelerates neighbor queries:

```
cell_id = hash(floor(position / cell_size))

// Query 27 neighboring cells
for dx, dy, dz in -1..1:
    neighbor_cell = hash(cell + (dx, dy, dz))
    for particle in cell_particles[neighbor_cell]:
        if distance < smoothing_radius:
            add_neighbor(particle)
```

### Simulation Steps

1. **Neighbor search**: Build spatial hash, find neighbors within smoothing radius
2. **Density computation**: Sum contributions from neighbors using poly6 kernel
3. **Pressure computation**: `P = k * (ρ - ρ₀)` (Tait equation)
4. **Force computation**:
   - Pressure force: `F_p = -Σ m_j * (P_i + P_j) / (2ρ_j) * ∇W`
   - Viscosity force: `F_v = μ * Σ m_j * (v_j - v_i) / ρ_j * ∇²W`
   - External forces: gravity, boundaries
5. **Integration**: Semi-implicit Euler

### Boundary Handling

Boundary conditions use penalty forces:

```
if particle.position.z < boundary_min:
    force += boundary_stiffness * (boundary_min - particle.position.z) * normal
    velocity *= boundary_damping
```

## Materials and Rendering

Zeno supports PBR (Physically Based Rendering) materials with textures.

### Material Properties

```zig
pub const Material = struct {
    name: []const u8,
    base_color: [4]f32 = .{1, 1, 1, 1},    // RGBA
    metallic: f32 = 0.0,                     // 0 = dielectric, 1 = metal
    roughness: f32 = 0.5,                    // Surface roughness
    emissive: [3]f32 = .{0, 0, 0},          // Emission color
    base_color_texture: ?*Texture = null,
    normal_texture: ?*Texture = null,
    metallic_roughness_texture: ?*Texture = null,
};
```

### Texture System

```zig
pub const Texture = struct {
    width: u32,
    height: u32,
    channels: u32,
    data: []u8,              // CPU-side pixel data
    gpu_texture: ?*anyopaque, // Metal texture handle
};
```

### Procedural Textures

Built-in procedural texture generators:

| Generator | Description |
|-----------|-------------|
| `checkerboard` | Two-color checkerboard pattern |
| `gradient` | Linear gradient between two colors |
| `noise` | Perlin-like noise texture |

### GPU Material Format

Materials are packed for efficient GPU access:

```zig
pub const MaterialGPU = extern struct {
    base_color: [4]f32,
    metallic: f32,
    roughness: f32,
    emissive: [3]f32,
    texture_indices: [4]i32,  // -1 if no texture
};
```