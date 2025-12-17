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

#### `close()`

Release resources and clean up.

```python
env.close()
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
