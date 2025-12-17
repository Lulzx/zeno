# MJCF Models

Zeno supports a subset of the MuJoCo XML format (MJCF), allowing you to load existing models or create new ones.

## Basic Structure

```xml
<mujoco model="my_robot">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="torso" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>

      <body name="leg" pos="0 0 -0.2">
        <joint name="hip" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="hip" gear="100" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

## Supported Elements

### `<option>`

Global simulation options.

```xml
<option timestep="0.002" gravity="0 0 -9.81"/>
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `timestep` | float | 0.002 | Physics timestep in seconds |
| `gravity` | vec3 | "0 0 -9.81" | Gravity vector |

### `<body>`

Rigid body definition.

```xml
<body name="torso" pos="0 0 1" quat="1 0 0 0">
  <!-- geoms and child bodies -->
</body>
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "" | Body identifier |
| `pos` | vec3 | "0 0 0" | Position relative to parent |
| `quat` | vec4 | "1 0 0 0" | Orientation (w x y z) |

### `<joint>`

Joint connecting bodies.

```xml
<joint name="hip" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.1"/>
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "" | Joint identifier |
| `type` | string | "hinge" | Joint type (see below) |
| `axis` | vec3 | "1 0 0" | Rotation/translation axis |
| `range` | vec2 | "-inf inf" | Joint limits |
| `damping` | float | 0 | Velocity damping |
| `stiffness` | float | 0 | Position spring stiffness |

**Joint Types:**

| Type | DOF | Description |
|------|-----|-------------|
| `free` | 6 | Floating base (no constraint) |
| `ball` | 3 | Spherical joint |
| `hinge` | 1 | Revolute joint (rotation) |
| `slide` | 1 | Prismatic joint (translation) |

### `<geom>`

Collision geometry and visual shape.

```xml
<geom name="torso_geom" type="box" size="0.1 0.2 0.1" mass="1.0" friction="0.8"/>
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "" | Geometry identifier |
| `type` | string | "sphere" | Geometry type |
| `size` | vec | required | Size parameters (type-dependent) |
| `mass` | float | 1.0 | Mass in kg |
| `friction` | float | 0.5 | Friction coefficient |
| `pos` | vec3 | "0 0 0" | Offset from body frame |

**Geometry Types:**

| Type | Size Format | Description |
|------|-------------|-------------|
| `sphere` | radius | Sphere |
| `capsule` | radius, half-length | Cylinder with hemisphere caps |
| `box` | half-x, half-y, half-z | Rectangular box |
| `plane` | - | Infinite ground plane |

**Capsule with `fromto`:**

```xml
<geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3"/>
```

### `<actuator>`

Motor or servo driving a joint.

```xml
<motor joint="hip" gear="100" ctrlrange="-1 1"/>
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `joint` | string | required | Target joint name |
| `gear` | float | 1.0 | Torque multiplier |
| `ctrlrange` | vec2 | "-1 1" | Control signal limits |

**Actuator Types:**

- `<motor>` - Direct torque control
- `<position>` - Position servo (PD controller)
- `<velocity>` - Velocity servo

### `<sensor>`

State observation sensors.

```xml
<jointpos joint="hip"/>
<accelerometer site="torso_imu"/>
```

| Type | Output | Description |
|------|--------|-------------|
| `jointpos` | 1 | Joint angle/position |
| `jointvel` | 1 | Joint velocity |
| `accelerometer` | 3 | Linear acceleration |
| `gyro` | 3 | Angular velocity |
| `framepos` | 3 | Body position |
| `framequat` | 4 | Body orientation |

## Example: Simple Pendulum

```xml
<mujoco model="pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Fixed mount point -->
    <body name="mount" pos="0 0 2">
      <geom type="sphere" size="0.05" mass="0"/>

      <!-- Swinging arm -->
      <body name="arm" pos="0 0 0">
        <joint name="pivot" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.5" mass="1.0"/>

        <!-- Weight at end -->
        <body name="weight" pos="0 0 -0.5">
          <geom type="sphere" size="0.1" mass="5.0"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="pivot" gear="10" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointpos joint="pivot"/>
    <jointvel joint="pivot"/>
  </sensor>
</mujoco>
```

## Example: Quadruped (Ant)

```xml
<mujoco model="ant">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <geom type="plane" size="10 10 0.1"/>

    <body name="torso" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="sphere" size="0.25" mass="1.0"/>

      <!-- Front-right leg -->
      <body name="leg_fr" pos="0.2 -0.2 0">
        <joint name="hip_fr" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.2 -0.2 0"/>

        <body name="ankle_fr" pos="0.2 -0.2 0">
          <joint name="ankle_fr" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3"/>
        </body>
      </body>

      <!-- Additional legs... -->
    </body>
  </worldbody>

  <actuator>
    <motor joint="hip_fr" gear="100" ctrlrange="-1 1"/>
    <motor joint="ankle_fr" gear="100" ctrlrange="-1 1"/>
    <!-- Additional actuators... -->
  </actuator>
</mujoco>
```

## Not Yet Supported

The following MJCF features are not currently supported:

- Tendons (`<tendon>`)
- Equality constraints (`<equality>`)
- Mesh geometry (`type="mesh"`)
- Heightfield terrain (`type="hfield"`)
- Soft bodies
- Sites (partially supported for sensors)
- Include files (`<include>`)
- Assets and textures
