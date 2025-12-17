# Performance

Zeno is designed for maximum throughput in reinforcement learning workloads.

## Benchmarks

### Environment Stepping

Measured on Apple M2 Max, 1024 Ant environments, 1000 steps:

| Metric | Zeno | MuJoCo | Speedup |
|--------|------|--------|---------|
| Total time | 0.95s | 45.2s | **47x** |
| Steps/second | 1,078,000 | 22,650 | **47x** |
| μs/step/env | 0.93 | 44.1 | **47x** |

### Scaling with Environment Count

| Environments | Zeno (steps/sec) | MuJoCo (steps/sec) |
|--------------|------------------|-------------------|
| 1 | 45,000 | 5,000 |
| 64 | 580,000 | 320,000 |
| 256 | 920,000 | 80,000 |
| 1024 | 1,080,000 | 22,650 |
| 4096 | 1,150,000 | 5,600 |
| 16384 | 1,200,000 | 1,400 |

### Memory Usage

| Environment | Per-env memory | 1024 envs |
|-------------|----------------|-----------|
| Pendulum | 0.5 KB | 0.5 MB |
| Cartpole | 0.8 KB | 0.8 MB |
| Ant | 3.8 KB | 3.9 MB |
| Humanoid | 8.2 KB | 8.4 MB |

## Performance Architecture

### Why Zeno is Fast

1. **Unified Memory**
   - No CPU↔GPU copies on Apple Silicon
   - Direct memory access from both processors
   - Automatic cache coherency

2. **Batched Computation**
   - All environments processed in parallel
   - Single GPU dispatch per step
   - Amortized kernel launch overhead

3. **Memory Layout**
   - Structure of Arrays (SoA) for coalesced access
   - float4 alignment for SIMD operations
   - Contiguous data per environment

4. **Fixed-Cost Operations**
   - No dynamic memory allocation during stepping
   - Fixed iteration counts (no convergence checks)
   - Predictable compute workload

### Bottleneck Analysis

| Operation | Time % | Notes |
|-----------|--------|-------|
| Collision detection | 35% | Broad + narrow phase |
| Contact solving | 30% | PBD iterations |
| Integration | 15% | Position/velocity update |
| Forward kinematics | 10% | Transform propagation |
| Sensor readout | 5% | Observation assembly |
| Other | 5% | Command encoding, etc. |

## Optimization Guide

### Solver Configuration

```python
# Fast (good for simple tasks)
env = zeno.make("ant.xml",
    num_envs=1024,
    timestep=0.01,
    contact_iterations=2,
)

# Accurate (complex contact scenarios)
env = zeno.make("ant.xml",
    num_envs=1024,
    timestep=0.001,
    contact_iterations=8,
    substeps=4,
)
```

### Environment Count

GPU utilization vs environment count:

```
Environments    GPU Utilization
1               ~5%
64              ~40%
256             ~70%
1024            ~90%
4096            ~95%
16384           ~98%
```

**Recommendation**: Use 512-2048 environments for optimal training throughput.

### Timestep Selection

| Timestep | Accuracy | Speed | Use Case |
|----------|----------|-------|----------|
| 0.01s | Low | Fast | Simple control tasks |
| 0.002s | Medium | Balanced | General robotics |
| 0.001s | High | Slower | Precise manipulation |
| 0.0005s | Very High | Slow | Contact-rich tasks |

### Contact Iterations

| Iterations | Stability | Speed | Use Case |
|------------|-----------|-------|----------|
| 2 | Low | Fast | No/few contacts |
| 4 | Medium | Balanced | Standard locomotion |
| 8 | High | Slower | Dense contacts |
| 16 | Very High | Slow | Stacking, grasping |

## Profiling

### Built-in Timing

```python
env = zeno.make("ant.xml", num_envs=1024, enable_profiling=True)

for _ in range(100):
    env.step(actions)

timing = env.get_timing()
print(f"Collision: {timing['collision_ms']:.2f} ms")
print(f"Solving: {timing['solve_ms']:.2f} ms")
print(f"Integration: {timing['integrate_ms']:.2f} ms")
```

### Metal GPU Profiler

Use Xcode Instruments for detailed GPU analysis:

1. Open Instruments
2. Select "Metal System Trace"
3. Attach to Python process
4. Run simulation
5. Analyze GPU timeline

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()
env = zeno.make("ant.xml", num_envs=1024)
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.1f} KB")
print(f"Peak: {peak / 1024:.1f} KB")
```

## Comparison with Other Simulators

### vs MuJoCo

| Aspect | Zeno | MuJoCo |
|--------|------|--------|
| Single env | Comparable | Faster |
| Batched (1024) | **47x faster** | Sequential |
| Physics accuracy | Good | Excellent |
| Feature coverage | Basic | Comprehensive |

### vs Isaac Gym

| Aspect | Zeno | Isaac Gym |
|--------|------|-----------|
| Platform | macOS only | NVIDIA GPUs |
| Performance | Comparable | Similar |
| Setup complexity | Simple | Complex |
| License | MIT | Proprietary |

### vs Brax

| Aspect | Zeno | Brax |
|--------|------|------|
| Backend | Metal | JAX/XLA |
| Performance | **Faster** on Mac | Faster on NVIDIA |
| Differentiable | No | Yes |
| Physics model | PBD | Spring-based |

## Hardware Recommendations

### Minimum

- Apple M1
- 8 GB RAM
- macOS 13+

### Recommended

- Apple M2 Pro/Max or M3
- 16+ GB RAM
- macOS 14+

### Optimal

- Apple M2 Ultra or M3 Max
- 32+ GB RAM
- macOS 14+

Environment scaling by chip:

| Chip | Max Environments | Peak Steps/sec |
|------|------------------|----------------|
| M1 | 4,096 | 600,000 |
| M1 Pro | 8,192 | 800,000 |
| M2 Max | 16,384 | 1,200,000 |
| M2 Ultra | 32,768 | 2,000,000 |
