# Python API

## Environment Creation

### `zeno.make()`

Create a Zeno environment from an MJCF model file.

```python
import zeno

env = zeno.make(
    model="assets/ant.xml",  # Path to MJCF file
    num_envs=1024,           # Number of parallel environments
    timestep=0.002,          # Physics timestep (seconds)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Path to MJCF XML file |
| `num_envs` | `int` | `1` | Number of parallel environments |
| `timestep` | `float` | `0.002` | Physics timestep in seconds |

**Returns:** `ZenoEnv` instance

## ZenoEnv Class

### Properties

```python
env.num_envs          # int: Number of parallel environments
env.observation_dim   # int: Observation vector dimension
env.action_dim        # int: Action vector dimension
env.timestep          # float: Physics timestep
```

### Methods

#### `reset(mask=None)`

Reset environments to initial state.

```python
# Reset all environments
obs = env.reset()

# Reset only specific environments
mask = np.array([True, False, True, ...])  # shape: (num_envs,)
obs = env.reset(mask=mask)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask` | `np.ndarray` | `None` | Boolean mask of environments to reset |

**Returns:** `np.ndarray` of shape `(num_envs, observation_dim)`

#### `step(actions)`

Advance simulation by one timestep.

```python
actions = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
obs, rewards, dones, info = env.step(actions)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `actions` | `np.ndarray` | Actions of shape `(num_envs, action_dim)` |

**Returns:** Tuple of:

- `obs`: `np.ndarray` of shape `(num_envs, observation_dim)`
- `rewards`: `np.ndarray` of shape `(num_envs,)`
- `dones`: `np.ndarray` of shape `(num_envs,)`, dtype `bool`
- `info`: `dict` with additional information

#### `get_body_positions()`

Get world-space positions of all bodies.

```python
positions = env.get_body_positions()
# Shape: (num_envs, num_bodies, 3)
```

#### `get_body_quaternions()`

Get world-space orientations of all bodies as quaternions.

```python
quats = env.get_body_quaternions()
# Shape: (num_envs, num_bodies, 4)
# Format: (x, y, z, w)
```

#### `get_contacts()`

Get contact information for all environments.

```python
contacts = env.get_contacts()
# Returns dict with contact data
```

**Returns:** Dictionary with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `position` | `(num_envs, max_contacts, 3)` | World-space contact position |
| `normal` | `(num_envs, max_contacts, 3)` | Contact normal (from body A to B) |
| `penetration` | `(num_envs, max_contacts)` | Penetration depth |
| `body_a` | `(num_envs, max_contacts)` | Index of first body |
| `body_b` | `(num_envs, max_contacts)` | Index of second body |
| `friction` | `(num_envs, max_contacts)` | Friction coefficient |
| `impulse` | `(num_envs, max_contacts)` | Normal impulse magnitude |
| `count` | `(num_envs,)` | Number of active contacts per environment |

**Example:**

```python
contacts = env.get_contacts()

# Get number of contacts in first environment
num_contacts = contacts['count'][0]

# Get positions of active contacts
active_positions = contacts['position'][0, :num_contacts]

# Find contacts involving specific body
body_id = 3
body_contacts = np.where(
    (contacts['body_a'][0] == body_id) |
    (contacts['body_b'][0] == body_id)
)[0]

# Calculate total contact force on a body
total_force = np.sum(
    contacts['impulse'][0, body_contacts, np.newaxis] *
    contacts['normal'][0, body_contacts]
)
```

#### `get_contact_counts()`

Get the number of active contacts per environment.

```python
counts = env.get_contact_counts()
# Shape: (num_envs,)
```

#### `close()`

Release resources and clean up.

```python
env.close()
```

## Contact Data Format

Contacts are stored in GPU buffers with the following structure:

### GPU Memory Layout (CompactContact)

Each contact is 64 bytes:

```
Offset  Size  Description
0       16    position_pen: (x, y, z, penetration)
16      16    normal_friction: (nx, ny, nz, friction)
32      16    indices: (body_a, body_b, geom_a, geom_b)
48      16    impulses: (normal, tangent1, tangent2, restitution)
```

### Contact Processing Example

```python
import numpy as np

def process_contacts(env):
    """Analyze contact forces for reward shaping."""
    contacts = env.get_contacts()

    # Per-environment contact analysis
    for env_id in range(env.num_envs):
        n = contacts['count'][env_id]
        if n == 0:
            continue

        # Extract active contacts
        pos = contacts['position'][env_id, :n]
        normal = contacts['normal'][env_id, :n]
        impulse = contacts['impulse'][env_id, :n]

        # Calculate total normal force
        total_force = np.abs(impulse).sum()

        # Calculate center of pressure
        if total_force > 0:
            cop = np.average(pos, weights=np.abs(impulse), axis=0)

        # Check for ground contacts (body_b == 0 typically)
        ground_contacts = contacts['body_b'][env_id, :n] == 0
        ground_force = np.abs(impulse[ground_contacts]).sum()

        print(f"Env {env_id}: {n} contacts, ground force: {ground_force:.2f}")
```

### Contact Filtering

```python
def get_foot_contacts(env, foot_body_ids):
    """Get contacts for specific foot bodies."""
    contacts = env.get_contacts()
    foot_contacts = {}

    for env_id in range(env.num_envs):
        n = contacts['count'][env_id]
        foot_contacts[env_id] = []

        for foot_id in foot_body_ids:
            # Find contacts involving this foot
            is_foot_contact = (
                (contacts['body_a'][env_id, :n] == foot_id) |
                (contacts['body_b'][env_id, :n] == foot_id)
            )
            if np.any(is_foot_contact):
                foot_contacts[env_id].append({
                    'body_id': foot_id,
                    'positions': contacts['position'][env_id, :n][is_foot_contact],
                    'normals': contacts['normal'][env_id, :n][is_foot_contact],
                    'forces': contacts['impulse'][env_id, :n][is_foot_contact]
                })

    return foot_contacts
```

## Zero-Copy Memory

Zeno uses unified memory shared between CPU and GPU. The arrays returned by `step()` and other methods are **views** into GPU memory, not copies.

```python
obs, rewards, dones, info = env.step(actions)

# obs is a view into GPU memory - no copy!
# Modifications affect the underlying buffer
```

!!! warning "Array Lifetime"
    Arrays are valid until the next `step()` or `reset()` call. Copy if you need to keep data:
    ```python
    obs_copy = obs.copy()
    ```

## Data Types

All arrays use `float32` for numerical data:

```python
actions = actions.astype(np.float32)  # Ensure correct dtype
```

## Example: Full Training Loop

```python
import zeno
import numpy as np

def train_random_policy(model_path, num_envs=1024, num_steps=10000):
    env = zeno.make(model_path, num_envs=num_envs)

    obs = env.reset()
    total_reward = 0

    for step in range(num_steps):
        # Sample random actions
        actions = np.random.uniform(
            -1, 1,
            (num_envs, env.action_dim)
        ).astype(np.float32)

        # Step simulation
        obs, rewards, dones, info = env.step(actions)
        total_reward += rewards.sum()

        # Reset done environments
        if np.any(dones):
            env.reset(mask=dones)

    env.close()

    print(f"Average reward: {total_reward / (num_steps * num_envs):.3f}")

train_random_policy("assets/ant.xml")
```
