# Profiling Guide

Zeno provides built-in GPU profiling to help identify performance bottlenecks and optimize your simulations.

## Using the Profiling API

```python
import zeno

# Create environment
env = zeno.make("ant.xml", num_envs=1024)

# Run simulation
for _ in range(100):
    env.step(actions)

# Get profiling data
profile = env.get_profiling_data()
print(profile)
```

## Profiling Data Structure

The profiling data returns timing information for each GPU kernel:

```python
{
    'apply_actions': 0.12,       # ms - Convert control inputs to torques
    'apply_joint_forces': 0.08,  # ms - Map torques to body forces
    'update_kinematic': 0.05,    # ms - Update kinematic body positions
    'compute_forces': 0.15,      # ms - Gravity, damping forces
    'integrate': 0.18,           # ms - Semi-implicit Euler
    'broad_phase': 0.22,         # ms - Spatial hashing collision detection
    'narrow_phase': 0.35,        # ms - Precise contact generation
    'solve_joints': 0.45,        # ms - XPBD joint constraint solver
    'solve_contacts': 0.40,      # ms - Contact constraint solver
    'update_joint_states': 0.10, # ms - Inverse kinematics for sensors
    'read_sensors': 0.08,        # ms - Generate observations
    'total': 2.18,               # ms - Total step time
}
```

## Identifying Bottlenecks

### Typical Performance Profile

For most simulations, the following stages dominate:

1. **solve_joints** (20-30%): XPBD constraint iterations
2. **solve_contacts** (15-25%): Contact resolution
3. **narrow_phase** (10-20%): Contact detection
4. **integrate** (5-10%): Physics integration

### Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High `solve_joints` | Too many constraints or iterations | Reduce `contact_iterations`, simplify model |
| High `solve_contacts` | Many active contacts | Reduce `max_contacts_per_env` |
| High `narrow_phase` | Complex collision geometry | Use simpler collision shapes |
| High `broad_phase` | Too many geometries | Reduce geom count, use collision groups |

## Optimization Strategies

### 1. Reduce Constraint Iterations

```python
# Default: 4 iterations - balanced
env = zeno.make("ant.xml", contact_iterations=4)

# Fast: 2 iterations - less accurate but faster
env = zeno.make("ant.xml", contact_iterations=2)

# Accurate: 8 iterations - more stable, slower
env = zeno.make("ant.xml", contact_iterations=8)
```

### 2. Use Substeps Wisely

```python
# Single step (default)
env = zeno.make("ant.xml", substeps=1)

# Multiple substeps for stability
# Note: total_time = substeps * compute_time
env = zeno.make("ant.xml", substeps=4)
```

### 3. Limit Maximum Contacts

```python
# Default: 64 contacts per environment
env = zeno.make("ant.xml", max_contacts_per_env=64)

# Reduce for simple scenes
env = zeno.make("ant.xml", max_contacts_per_env=16)
```

### 4. Optimize Model Complexity

- Use spheres/capsules instead of boxes where possible
- Minimize body count
- Combine fixed bodies into single static body
- Use collision groups to skip unnecessary collision checks

## Scaling Analysis

### Environment Count Scaling

```python
import zeno
import time

for num_envs in [128, 256, 512, 1024, 2048, 4096]:
    env = zeno.make("ant.xml", num_envs=num_envs)

    # Warmup
    for _ in range(10):
        env.step(np.zeros((num_envs, env.action_dim)))

    # Benchmark
    start = time.time()
    for _ in range(100):
        env.step(np.zeros((num_envs, env.action_dim)))
    elapsed = time.time() - start

    steps_per_sec = 100 / elapsed
    env_steps_per_sec = steps_per_sec * num_envs

    print(f"{num_envs} envs: {steps_per_sec:.1f} steps/s, {env_steps_per_sec:.0f} env-steps/s")
    env.close()
```

### Expected Scaling

| Environment Count | Steps/sec | Env-Steps/sec |
|------------------|-----------|---------------|
| 128 | ~2000 | ~256,000 |
| 512 | ~800 | ~410,000 |
| 1024 | ~450 | ~460,000 |
| 2048 | ~250 | ~512,000 |
| 4096 | ~130 | ~532,000 |

*Results vary based on model complexity and hardware*

## Memory Profiling

```python
# Get memory usage information
info = env.get_info()
print(f"Memory usage: {info['memory_usage'] / 1024 / 1024:.1f} MB")

# Breakdown by buffer
# - State buffers: positions, velocities, quaternions
# - Constraint buffers: joint and contact constraints
# - Observation buffers: sensor readings
```

### Memory Estimation

```
memory_per_env ≈
    num_bodies * 160 bytes (state) +
    num_constraints * 96 bytes (XPBD) +
    max_contacts * 64 bytes (contacts) +
    obs_dim * 4 bytes (observations)
```

## Real-time Performance Tips

For real-time applications (visualization, teleoperation):

1. **Use smaller timesteps**: `timestep=0.002` for smooth motion
2. **Limit environment count**: 1-4 for real-time
3. **Reduce iterations**: `contact_iterations=2`
4. **Skip frames**: Step multiple times between renders

```python
# Real-time loop
import time

env = zeno.make("ant.xml", num_envs=1, timestep=0.002)
target_fps = 60
target_dt = 1.0 / target_fps

while running:
    start = time.time()

    # Step physics (multiple substeps for stability)
    for _ in range(4):
        env.step(action)

    # Render
    render()

    # Maintain framerate
    elapsed = time.time() - start
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)
```

## Profiling GPU Kernels

For detailed Metal GPU profiling, use Instruments:

1. Open Instruments (Xcode → Open Developer Tool → Instruments)
2. Select "Metal System Trace" template
3. Attach to your Python process
4. Run simulation
5. Analyze GPU timeline

This shows:
- Kernel execution times
- Memory transfers
- GPU utilization
- Buffer access patterns
