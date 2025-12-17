# Advanced Patterns

This guide covers advanced usage patterns for Zeno, including batched operations, curriculum learning, state checkpointing, and domain randomization.

## Batched Operations

### Selective Environment Resets

Reset only specific environments while others continue:

```python
import numpy as np

# Reset environments where episode terminated
obs, rewards, dones, info = env.step(actions)

if np.any(dones):
    # Reset only done environments
    env.reset(mask=dones)

    # Alternatively, reset specific indices
    done_indices = np.where(dones)[0]
    for idx in done_indices:
        # Custom per-environment initialization
        pass
```

### Masked Stepping

Step only a subset of environments (useful for variable-length episodes):

```python
# Step all environments
env.step(actions)

# Step specific environments with masked actions
# Inactive environments receive zero actions
active_mask = np.array([True, True, False, True, ...])
masked_actions = actions.copy()
masked_actions[~active_mask] = 0
env.step(masked_actions)
```

### Environment Groups

Organize environments into groups for different tasks:

```python
class EnvironmentGroups:
    def __init__(self, env, group_sizes):
        self.env = env
        self.groups = {}
        idx = 0
        for name, size in group_sizes.items():
            self.groups[name] = slice(idx, idx + size)
            idx += size

    def reset_group(self, group_name):
        mask = np.zeros(self.env.num_envs, dtype=bool)
        mask[self.groups[group_name]] = True
        self.env.reset(mask=mask)

    def get_group_obs(self, obs, group_name):
        return obs[self.groups[group_name]]

# Usage
groups = EnvironmentGroups(env, {'train': 900, 'eval': 100})
groups.reset_group('eval')
```

## Curriculum Learning

### Difficulty-Based Curriculum

Gradually increase task difficulty based on performance:

```python
class DifficultyScheduler:
    def __init__(self, env, initial_difficulty=0.0, max_difficulty=1.0):
        self.env = env
        self.difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_buffer = []
        self.window_size = 100

    def update(self, success_rate):
        self.success_buffer.append(success_rate)
        if len(self.success_buffer) > self.window_size:
            self.success_buffer.pop(0)

        avg_success = np.mean(self.success_buffer)

        # Increase difficulty if success rate is high
        if avg_success > 0.8 and self.difficulty < self.max_difficulty:
            self.difficulty = min(self.difficulty + 0.1, self.max_difficulty)
            self.apply_difficulty()

    def apply_difficulty(self):
        # Adjust simulation parameters based on difficulty
        # Example: increase obstacle speed, terrain roughness, etc.
        self.env.set_param('obstacle_speed', 1.0 + self.difficulty * 2.0)
        self.env.set_param('terrain_noise', self.difficulty * 0.1)
```

### Stage-Based Training

Train different behaviors in stages:

```python
class StageCurriculum:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage = 0

    def get_reward_weights(self):
        return self.stages[self.current_stage]['reward_weights']

    def get_termination_conditions(self):
        return self.stages[self.current_stage]['termination']

    def check_advancement(self, metrics):
        stage = self.stages[self.current_stage]
        if metrics['success_rate'] > stage['threshold']:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                return True
        return False

# Define curriculum stages
stages = [
    {
        'name': 'stand',
        'reward_weights': {'stand': 1.0, 'walk': 0.0},
        'termination': {'fall_height': 0.3},
        'threshold': 0.9
    },
    {
        'name': 'walk',
        'reward_weights': {'stand': 0.5, 'walk': 0.5},
        'termination': {'fall_height': 0.2},
        'threshold': 0.8
    },
    {
        'name': 'run',
        'reward_weights': {'stand': 0.2, 'walk': 0.8},
        'termination': {'fall_height': 0.1},
        'threshold': 0.7
    }
]
```

## State Checkpointing

### Save and Restore Simulation State

```python
class SimulationCheckpoint:
    def __init__(self, env):
        self.env = env

    def save(self):
        """Save complete simulation state."""
        return {
            'positions': self.env.get_body_positions().copy(),
            'quaternions': self.env.get_body_quaternions().copy(),
            'velocities': self.env.get_body_velocities().copy(),
            'angular_velocities': self.env.get_body_angular_velocities().copy(),
            'joint_positions': self.env.get_joint_positions().copy(),
            'joint_velocities': self.env.get_joint_velocities().copy(),
        }

    def restore(self, state):
        """Restore simulation state."""
        self.env.set_body_positions(state['positions'])
        self.env.set_body_quaternions(state['quaternions'])
        self.env.set_body_velocities(state['velocities'])
        self.env.set_body_angular_velocities(state['angular_velocities'])
        self.env.set_joint_positions(state['joint_positions'])
        self.env.set_joint_velocities(state['joint_velocities'])

# Usage
checkpoint = SimulationCheckpoint(env)

# Save before risky action
state = checkpoint.save()

# Try action
env.step(risky_action)

# Restore if failed
if failed:
    checkpoint.restore(state)
```

### Trajectory Recording

```python
class TrajectoryRecorder:
    def __init__(self, env, record_contacts=False):
        self.env = env
        self.record_contacts = record_contacts
        self.trajectories = []

    def start_episode(self):
        self.current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'positions': [],
            'contacts': [] if self.record_contacts else None
        }

    def record_step(self, obs, action, reward):
        self.current_trajectory['observations'].append(obs.copy())
        self.current_trajectory['actions'].append(action.copy())
        self.current_trajectory['rewards'].append(reward.copy())
        self.current_trajectory['positions'].append(
            self.env.get_body_positions().copy()
        )

        if self.record_contacts:
            contacts = self.env.get_contacts()
            self.current_trajectory['contacts'].append({
                k: v.copy() for k, v in contacts.items()
            })

    def end_episode(self):
        self.trajectories.append(self.current_trajectory)

    def save(self, path):
        np.savez(path, trajectories=self.trajectories)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.trajectories = data['trajectories'].tolist()
```

## Domain Randomization

### Physics Parameter Randomization

```python
class PhysicsRandomizer:
    def __init__(self, env):
        self.env = env
        self.default_params = {
            'gravity': -9.81,
            'friction': 1.0,
            'joint_damping': 0.5,
            'contact_stiffness': 1e4,
        }

    def randomize(self, ranges=None):
        """Randomize physics parameters within specified ranges."""
        ranges = ranges or {
            'gravity': (-10.5, -8.5),
            'friction': (0.5, 1.5),
            'joint_damping': (0.2, 1.0),
            'contact_stiffness': (5e3, 2e4),
        }

        for param, (low, high) in ranges.items():
            value = np.random.uniform(low, high)
            self.env.set_param(param, value)

    def reset_to_default(self):
        """Reset all parameters to defaults."""
        for param, value in self.default_params.items():
            self.env.set_param(param, value)
```

### Initial State Randomization

```python
class StateRandomizer:
    def __init__(self, env):
        self.env = env

    def randomize_positions(self, noise_scale=0.05):
        """Add noise to body positions."""
        positions = self.env.get_body_positions()
        noise = np.random.uniform(-noise_scale, noise_scale, positions.shape)
        self.env.set_body_positions(positions + noise)

    def randomize_velocities(self, max_linear=0.5, max_angular=0.3):
        """Randomize body velocities."""
        shape = self.env.get_body_velocities().shape
        linear_vel = np.random.uniform(-max_linear, max_linear, shape)
        self.env.set_body_velocities(linear_vel)

        shape = self.env.get_body_angular_velocities().shape
        angular_vel = np.random.uniform(-max_angular, max_angular, shape)
        self.env.set_body_angular_velocities(angular_vel)

    def randomize_joint_positions(self, fraction=0.2):
        """Randomize joint positions within limits."""
        joint_pos = self.env.get_joint_positions()
        limits = self.env.get_joint_limits()

        ranges = limits[:, 1] - limits[:, 0]
        noise = np.random.uniform(-fraction, fraction, joint_pos.shape) * ranges
        new_pos = np.clip(joint_pos + noise, limits[:, 0], limits[:, 1])
        self.env.set_joint_positions(new_pos)
```

### Visual Randomization

For sim-to-real transfer, randomize visual properties:

```python
class VisualRandomizer:
    def __init__(self, env):
        self.env = env

    def randomize_colors(self):
        """Randomize body colors."""
        for body_id in range(self.env.num_bodies):
            color = np.random.uniform(0, 1, 4)
            color[3] = 1.0  # Keep alpha = 1
            self.env.set_body_color(body_id, color)

    def randomize_lighting(self):
        """Randomize lighting conditions."""
        ambient = np.random.uniform(0.2, 0.6)
        diffuse = np.random.uniform(0.4, 0.8)
        direction = np.random.uniform(-1, 1, 3)
        direction = direction / np.linalg.norm(direction)

        self.env.set_lighting(
            ambient=ambient,
            diffuse=diffuse,
            direction=direction
        )
```

## Asynchronous Evaluation

### Parallel Policy Evaluation

```python
import multiprocessing as mp

def evaluate_policy(policy_weights, env_config, num_episodes):
    """Evaluate policy in separate process."""
    env = zeno.make(**env_config)
    policy = create_policy(policy_weights)

    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.sum()

    env.close()
    return total_reward / num_episodes

def parallel_evaluation(policies, env_config, num_workers=4):
    """Evaluate multiple policies in parallel."""
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(evaluate_policy, [
            (p, env_config, 10) for p in policies
        ])
    return results
```

### Background Data Collection

```python
import threading
import queue

class AsyncDataCollector:
    def __init__(self, env, policy, buffer_size=10000):
        self.env = env
        self.policy = policy
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _collect(self):
        obs = self.env.reset()
        while self.running:
            action = self.policy(obs)
            next_obs, reward, done, _ = self.env.step(action)

            try:
                self.buffer.put_nowait({
                    'obs': obs.copy(),
                    'action': action.copy(),
                    'reward': reward.copy(),
                    'next_obs': next_obs.copy(),
                    'done': done.copy()
                })
            except queue.Full:
                pass  # Drop sample if buffer full

            obs = next_obs
            if np.any(done):
                self.env.reset(mask=done)

    def get_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return batch
```

## Multi-Task Learning

### Task Embeddings

```python
class MultiTaskEnv:
    def __init__(self, env, num_tasks):
        self.env = env
        self.num_tasks = num_tasks
        self.task_ids = np.zeros(env.num_envs, dtype=np.int32)

    def reset(self, mask=None):
        obs = self.env.reset(mask=mask)

        # Assign random tasks to reset environments
        if mask is None:
            mask = np.ones(self.env.num_envs, dtype=bool)
        self.task_ids[mask] = np.random.randint(0, self.num_tasks, mask.sum())

        # Append task embedding to observations
        task_embeddings = np.eye(self.num_tasks)[self.task_ids]
        return np.concatenate([obs, task_embeddings], axis=1)

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)

        # Task-specific reward shaping
        for task_id in range(self.num_tasks):
            task_mask = self.task_ids == task_id
            rewards[task_mask] = self.compute_task_reward(
                task_id, obs[task_mask], rewards[task_mask]
            )

        # Append task embedding
        task_embeddings = np.eye(self.num_tasks)[self.task_ids]
        obs = np.concatenate([obs, task_embeddings], axis=1)

        return obs, rewards, dones, info

    def compute_task_reward(self, task_id, obs, base_reward):
        # Override for task-specific rewards
        return base_reward
```

## Error Recovery

### Graceful Degradation

```python
class RobustSimulation:
    def __init__(self, env):
        self.env = env
        self.checkpoint = None
        self.error_count = 0
        self.max_errors = 3

    def step(self, actions):
        try:
            result = self.env.step(actions)

            # Check for physics instability
            positions = self.env.get_body_positions()
            if np.any(np.isnan(positions)) or np.any(np.abs(positions) > 100):
                raise RuntimeError("Physics instability detected")

            # Save checkpoint periodically
            if np.random.random() < 0.01:
                self.checkpoint = self._save_state()

            self.error_count = 0
            return result

        except Exception as e:
            self.error_count += 1
            print(f"Error in simulation: {e}")

            if self.error_count >= self.max_errors:
                raise RuntimeError("Too many consecutive errors")

            # Try to recover
            if self.checkpoint is not None:
                self._restore_state(self.checkpoint)
            else:
                self.env.reset()

            return self.env.step(np.zeros_like(actions))

    def _save_state(self):
        return {
            'positions': self.env.get_body_positions().copy(),
            'velocities': self.env.get_body_velocities().copy(),
        }

    def _restore_state(self, state):
        self.env.set_body_positions(state['positions'])
        self.env.set_body_velocities(state['velocities'])
```
