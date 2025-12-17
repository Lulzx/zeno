# Python API

Zeno provides a high-performance Python API with zero-copy GPU memory access and full Gymnasium compatibility.

## Installation

```bash
cd python
pip install -e .
```

## Quick Start

```python
import zeno
import numpy as np

# Create environment with 1024 parallel instances
env = zeno.make("ant.xml", num_envs=1024)

# Reset all environments
obs = env.reset()

# Run simulation
for _ in range(1000):
    actions = np.random.uniform(-1, 1, (1024, env.action_dim))
    obs, rewards, dones, info = env.step(actions)

env.close()
```

## Zero-Copy Memory Access

Zeno leverages Apple Silicon's unified memory architecture to provide true zero-copy access to simulation state. All state arrays are backed by Metal buffers with `storageModeShared`, meaning both CPU and GPU access the same physical memory.

### Direct State Access

```python
# Get zero-copy view into GPU memory
positions = env._world.get_body_positions(zero_copy=True)
velocities = env._world.get_body_velocities(zero_copy=True)
quaternions = env._world.get_body_quaternions(zero_copy=True)

# Modify state directly (changes are reflected on GPU)
positions[0, 0, 2] += 0.1  # Lift first body

# Safe copy for storage
positions_copy = env._world.get_body_positions(zero_copy=False)
```

### Available State Accessors

| Method | Shape | Description |
|--------|-------|-------------|
| `get_observations()` | `(num_envs, obs_dim)` | Sensor observations |
| `get_rewards()` | `(num_envs,)` | Step rewards |
| `get_dones()` | `(num_envs,)` | Episode done flags |
| `get_body_positions()` | `(num_envs, num_bodies, 4)` | Body positions (x,y,z,pad) |
| `get_body_quaternions()` | `(num_envs, num_bodies, 4)` | Body orientations (x,y,z,w) |
| `get_body_velocities()` | `(num_envs, num_bodies, 4)` | Linear velocities |
| `get_body_angular_velocities()` | `(num_envs, num_bodies, 4)` | Angular velocities |
| `get_joint_positions()` | `(num_envs, num_joints)` | Joint angles/positions |
| `get_joint_velocities()` | `(num_envs, num_joints)` | Joint velocities |
| `get_contact_forces()` | `(num_envs, max_contacts, 4)` | Contact forces |
| `get_sensor_data()` | `(num_envs, num_sensors)` | Sensor readings |

### State Checkpointing

```python
# Save complete state
state = env._world.get_state()

# Run simulation
for _ in range(100):
    env.step(actions)

# Restore state
env._world.set_state(state)
```

## Gymnasium Integration

Zeno provides full Gymnasium API compatibility, enabling seamless integration with RL libraries.

### Single Environment

```python
import gymnasium as gym
import zeno.gym  # Register environments

env = gym.make("Zeno/Ant-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized Environment

```python
from zeno.gym import make_vec

# Create 1024 parallel environments (native GPU batching)
envs = make_vec("ant", num_envs=1024)
obs, info = envs.reset()

for _ in range(1000):
    actions = envs.action_space.sample()
    obs, rewards, terminated, truncated, info = envs.step(actions)

envs.close()
```

### Available Environments

| Environment | Description |
|-------------|-------------|
| `Zeno/Pendulum-v0` | Inverted pendulum |
| `Zeno/Cartpole-v0` | Cart-pole balancing |
| `Zeno/Ant-v0` | Quadruped locomotion |
| `Zeno/Humanoid-v0` | Bipedal humanoid |
| `Zeno/HalfCheetah-v0` | Cheetah running |
| `Zeno/Hopper-v0` | Single-leg hopping |
| `Zeno/Walker2d-v0` | Bipedal walking |
| `Zeno/Swimmer-v0` | 3-link swimmer |

## Stable-Baselines3 Integration

```python
from stable_baselines3 import PPO
from zeno.gym import make_sb3_env

# Create environment
env = make_sb3_env("ant", num_envs=8)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Save model
model.save("ppo_ant")
```

## Environment Wrappers

Zeno provides standard wrappers for RL training:

### Observation Normalization

```python
from zeno import ZenoEnv, NormalizeObservation

env = NormalizeObservation(
    ZenoEnv("ant.xml", num_envs=8),
    clip=10.0
)

# Observations are normalized using running statistics
obs = env.reset()
```

### Reward Normalization

```python
from zeno import ZenoEnv, NormalizeReward

env = NormalizeReward(
    ZenoEnv("ant.xml", num_envs=8),
    gamma=0.99,
    clip=10.0
)
```

### Episode Statistics

```python
from zeno import ZenoEnv, EpisodeStats

env = EpisodeStats(ZenoEnv("ant.xml", num_envs=8))

for _ in range(1000):
    obs, rewards, dones, info = env.step(actions)

    if "episode_returns" in info:
        print(f"Episode returns: {info['episode_returns']}")

# Get statistics
print(env.get_stats())
```

### Combined Wrappers

```python
from zeno import ZenoEnv, wrap_env

env = wrap_env(
    ZenoEnv("ant.xml", num_envs=8),
    normalize_obs=True,
    normalize_reward=True,
    clip_action=True,
    track_stats=True,
)
```

## Advanced Usage

### Domain Randomization

```python
# Modify gravity
env._world.set_gravity((0, 0, -9.81 * np.random.uniform(0.8, 1.2)))

# Modify timestep
env._world.set_timestep(0.002 * np.random.uniform(0.9, 1.1))

# Set body positions with noise
positions = env._world.get_body_positions(zero_copy=False)
positions += np.random.normal(0, 0.01, positions.shape)
env._world.set_body_positions(positions)
```

### Curriculum Learning

```python
# Reset specific environments to specific states
mask = np.zeros(num_envs, dtype=np.uint8)
mask[difficult_envs] = 1

env._world.reset_to_state(
    positions=saved_positions,
    quaternions=saved_quaternions,
    velocities=saved_velocities,
    mask=mask
)
```

### Profiling

```python
# Enable profiling
env = ZenoEnv("ant.xml", num_envs=1024, enable_profiling=True)

# Run simulation
for _ in range(100):
    env.step(actions)

# Get profiling data
prof = env._world.get_profiling_data()
print(f"Step time: {prof['total_step_ms']:.2f} ms")
print(f"Contacts: {prof['num_contacts']}")
```

## Memory Layout

All state arrays use Structure-of-Arrays (SoA) format with float4 alignment for optimal GPU performance:

```
positions[env_idx, body_idx, :] = [x, y, z, padding]
quaternions[env_idx, body_idx, :] = [x, y, z, w]
velocities[env_idx, body_idx, :] = [vx, vy, vz, padding]
```

The padding ensures 16-byte alignment for efficient GPU access.

## Thread Safety

- State accessors are thread-safe for reading
- Writing to state arrays should only be done when no GPU work is in flight
- The `step()` method handles synchronization automatically
