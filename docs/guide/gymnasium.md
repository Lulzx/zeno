# Gymnasium Integration

Zeno provides native integration with [Gymnasium](https://gymnasium.farama.org/), the standard API for reinforcement learning environments.

## Registration

Import `zeno.gym` to register Zeno environments:

```python
import gymnasium as gym
import zeno.gym  # Registers Zeno environments

# Now you can use gym.make()
env = gym.make("Zeno/Ant-v0")
```

## Available Environments

| Environment ID | Model | Actions | Description |
|----------------|-------|---------|-------------|
| `Zeno/Pendulum-v0` | pendulum.xml | 1 | Inverted pendulum |
| `Zeno/Cartpole-v0` | cartpole.xml | 1 | Cart-pole balancing |
| `Zeno/Ant-v0` | ant.xml | 8 | Quadruped locomotion |
| `Zeno/Humanoid-v0` | humanoid.xml | 13 | Bipedal humanoid |

## Single Environment

### Basic Usage

```python
import gymnasium as gym
import zeno.gym

env = gym.make("Zeno/Ant-v0")

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Custom Configuration

```python
env = gym.make(
    "Zeno/Ant-v0",
    timestep=0.002,
    max_episode_steps=1000,
)
```

## Vectorized Environments

For training, use Zeno's native vectorization for maximum performance.

### Using `make_vec()`

```python
from zeno.gym import make_vec

# Create 1024 parallel environments
envs = make_vec("ant", num_envs=1024)

obs, info = envs.reset()
print(f"Observations shape: {obs.shape}")  # (1024, obs_dim)

for _ in range(1000):
    actions = envs.action_space.sample()  # (1024, action_dim)
    obs, rewards, terminateds, truncateds, infos = envs.step(actions)

envs.close()
```

### Automatic Reset

The vector environment automatically resets terminated environments:

```python
envs = make_vec("ant", num_envs=1024, auto_reset=True)

obs, info = envs.reset()
for _ in range(10000):
    actions = envs.action_space.sample()
    obs, rewards, terminateds, truncateds, infos = envs.step(actions)
    # No need to manually reset - done automatically
```

## Integration with RL Libraries

### Stable-Baselines3

```python
from stable_baselines3 import PPO
from zeno.gym import make_vec

# Create vectorized environment
envs = make_vec("ant", num_envs=16)

# Train with SB3
model = PPO("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=1_000_000)

# Save and load
model.save("ppo_ant")
model = PPO.load("ppo_ant")

# Evaluate
obs, info = envs.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, terminateds, truncateds, infos = envs.step(action)
```

### CleanRL

```python
import numpy as np
from zeno.gym import make_vec

def make_env(env_id, num_envs):
    return make_vec(env_id, num_envs=num_envs)

envs = make_env("ant", num_envs=64)

# CleanRL-style training loop
obs, _ = envs.reset()
for global_step in range(100000):
    # Get actions from your policy
    actions = policy(obs)

    # Step environments
    next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

    # Store transition in replay buffer
    buffer.add(obs, actions, rewards, next_obs, terminateds)

    obs = next_obs
```

### RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
import zeno.gym

config = (
    PPOConfig()
    .environment("Zeno/Ant-v0")
    .training(train_batch_size=4000)
    .resources(num_gpus=0)
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
```

## Custom Environments

Create custom Gymnasium environments from MJCF files:

```python
from zeno.gym import ZenoGymnasiumEnv

class MyRobotEnv(ZenoGymnasiumEnv):
    def __init__(self, **kwargs):
        super().__init__(
            model_path="path/to/my_robot.xml",
            **kwargs
        )

    def compute_reward(self, obs, action, info):
        # Custom reward function
        velocity = info.get("velocity", 0)
        energy = np.sum(action ** 2)
        return velocity - 0.1 * energy

    def compute_terminated(self, obs, info):
        # Custom termination condition
        height = obs[2]  # Assuming z-position is in obs
        return height < 0.3

# Register the custom environment
from gymnasium.envs.registration import register

register(
    id="MyRobot-v0",
    entry_point=MyRobotEnv,
)

# Use it
env = gym.make("MyRobot-v0")
```

## Performance Tips

1. **Use native vectorization**: `make_vec()` is much faster than `gym.vector.SyncVectorEnv`

2. **Batch size**: Larger batch sizes (512-2048) better utilize the GPU

3. **Avoid Python loops**: Let Zeno handle the batched computation

4. **Minimize data copies**: The returned arrays are views into GPU memory

```python
# Good - native vectorization
envs = make_vec("ant", num_envs=1024)

# Slower - Python loop overhead
envs = gym.vector.SyncVectorEnv([
    lambda: gym.make("Zeno/Ant-v0") for _ in range(1024)
])
```
