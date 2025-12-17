# Basic Usage Examples

## Simple Pendulum Control

A minimal example demonstrating environment creation and stepping.

```python
import zeno
import numpy as np

# Create a single pendulum environment
env = zeno.make("assets/pendulum.xml", num_envs=1)

print(f"Observation dim: {env.observation_dim}")
print(f"Action dim: {env.action_dim}")

# Reset and run
obs = env.reset()

for step in range(500):
    # Apply constant torque
    action = np.array([[0.5]], dtype=np.float32)
    obs, reward, done, info = env.step(action)

    if step % 100 == 0:
        print(f"Step {step}: obs = {obs[0][:3]}")

env.close()
```

## Batched Simulation

Running multiple environments in parallel.

```python
import zeno
import numpy as np

NUM_ENVS = 1024

# Create batched environment
env = zeno.make("assets/ant.xml", num_envs=NUM_ENVS)

# Reset all environments
obs = env.reset()

# Track statistics
total_rewards = np.zeros(NUM_ENVS)
episode_lengths = np.zeros(NUM_ENVS)

for step in range(1000):
    # Random actions for all environments
    actions = np.random.uniform(-1, 1, (NUM_ENVS, env.action_dim)).astype(np.float32)

    # Step all environments
    obs, rewards, dones, info = env.step(actions)

    # Update statistics
    total_rewards += rewards
    episode_lengths += 1

    # Reset done environments
    if np.any(dones):
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            print(f"Env {idx}: episode length = {episode_lengths[idx]}, reward = {total_rewards[idx]:.2f}")
            total_rewards[idx] = 0
            episode_lengths[idx] = 0
        env.reset(mask=dones)

print(f"\nMean reward: {total_rewards.mean():.2f}")
env.close()
```

## Accessing Body State

Reading positions and orientations of rigid bodies.

```python
import zeno
import numpy as np

env = zeno.make("assets/ant.xml", num_envs=16)
obs = env.reset()

# Run a few steps
for _ in range(100):
    actions = np.zeros((16, env.action_dim), dtype=np.float32)
    env.step(actions)

# Get body positions
positions = env.get_body_positions()
print(f"Positions shape: {positions.shape}")  # (16, num_bodies, 3)

# Get body orientations
quaternions = env.get_body_quaternions()
print(f"Quaternions shape: {quaternions.shape}")  # (16, num_bodies, 4)

# Print torso position for first environment
torso_pos = positions[0, 0]  # First env, first body (torso)
print(f"Torso position: x={torso_pos[0]:.3f}, y={torso_pos[1]:.3f}, z={torso_pos[2]:.3f}")

env.close()
```

## Performance Measurement

Benchmarking simulation speed.

```python
import zeno
import numpy as np
import time

def benchmark(model, num_envs, num_steps):
    env = zeno.make(model, num_envs=num_envs)

    # Warmup
    obs = env.reset()
    actions = np.random.uniform(-1, 1, (num_envs, env.action_dim)).astype(np.float32)
    for _ in range(10):
        env.step(actions)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        env.step(actions)
    elapsed = time.perf_counter() - start

    env.close()

    total_steps = num_envs * num_steps
    print(f"{model}:")
    print(f"  {num_envs} envs × {num_steps} steps = {total_steps:,} total steps")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Steps/sec: {total_steps/elapsed:,.0f}")
    print(f"  μs/step/env: {elapsed/num_steps*1e6/num_envs:.2f}")
    print()

benchmark("assets/pendulum.xml", 1024, 1000)
benchmark("assets/cartpole.xml", 1024, 1000)
benchmark("assets/ant.xml", 1024, 1000)
benchmark("assets/humanoid.xml", 1024, 1000)
```

## Custom Reward Function

Implementing a custom reward outside the environment.

```python
import zeno
import numpy as np

env = zeno.make("assets/ant.xml", num_envs=256)
obs = env.reset()

def custom_reward(obs, actions, positions):
    """Reward forward velocity while penalizing energy."""
    # Assume obs contains velocity in first few dimensions
    forward_vel = obs[:, 0]  # x-velocity

    # Energy penalty
    energy = np.sum(actions ** 2, axis=1)

    # Height bonus (stay upright)
    torso_height = positions[:, 0, 2]  # First body z-position
    height_bonus = np.clip(torso_height - 0.3, 0, 0.2)

    return forward_vel - 0.1 * energy + height_bonus

total_custom_reward = 0

for step in range(1000):
    actions = np.random.uniform(-1, 1, (256, env.action_dim)).astype(np.float32)
    obs, _, dones, info = env.step(actions)

    positions = env.get_body_positions()
    rewards = custom_reward(obs, actions, positions)
    total_custom_reward += rewards.sum()

    if np.any(dones):
        env.reset(mask=dones)

print(f"Total custom reward: {total_custom_reward:.2f}")
env.close()
```

## Deterministic Simulation

Running reproducible simulations.

```python
import zeno
import numpy as np

def run_episode(seed):
    np.random.seed(seed)
    env = zeno.make("assets/ant.xml", num_envs=1)

    obs = env.reset()
    total_reward = 0

    for _ in range(100):
        action = np.random.uniform(-1, 1, (1, env.action_dim)).astype(np.float32)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

    env.close()
    return total_reward

# Same seed should give same result
reward1 = run_episode(42)
reward2 = run_episode(42)
reward3 = run_episode(123)

print(f"Seed 42 (run 1): {reward1:.4f}")
print(f"Seed 42 (run 2): {reward2:.4f}")
print(f"Seed 123: {reward3:.4f}")

assert abs(reward1 - reward2) < 1e-6, "Results should be identical!"
```

## Multiple Model Types

Working with different robot models.

```python
import zeno
import numpy as np

models = [
    ("assets/pendulum.xml", "Pendulum"),
    ("assets/cartpole.xml", "Cartpole"),
    ("assets/ant.xml", "Ant"),
    ("assets/humanoid.xml", "Humanoid"),
]

for model_path, name in models:
    env = zeno.make(model_path, num_envs=64)

    print(f"{name}:")
    print(f"  Observation dim: {env.observation_dim}")
    print(f"  Action dim: {env.action_dim}")

    obs = env.reset()
    actions = np.zeros((64, env.action_dim), dtype=np.float32)

    for _ in range(100):
        obs, rewards, dones, info = env.step(actions)

    print(f"  Mean final obs[0]: {obs[:, 0].mean():.3f}")
    print()

    env.close()
```
