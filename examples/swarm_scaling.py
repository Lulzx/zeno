"""Swarm scaling benchmark.

Measures step times at various agent counts to characterize scaling behavior.
"""

import time
import zeno
from zeno.swarm import create_swarm_world

agent_counts = [100, 500, 1000, 5000, 10000]

print("Zeno Swarm Scaling Benchmark")
print("=" * 50)
print(f"{'Agents':>8}  {'Step (ms)':>10}  {'Steps/sec':>10}  {'Edges':>8}")
print("-" * 50)

for n in agent_counts:
    world, swarm = create_swarm_world(
        num_agents=n,
        layout="grid",
        spacing=0.3,
        communication_range=2.0,
    )
    world.reset()

    # Warmup
    for _ in range(5):
        swarm.step()

    # Benchmark
    num_steps = 50
    start = time.perf_counter()
    for _ in range(num_steps):
        swarm.step()
    elapsed = time.perf_counter() - start

    step_ms = (elapsed / num_steps) * 1000
    steps_per_sec = num_steps / elapsed
    metrics = swarm.get_metrics()

    print(f"{n:>8}  {step_ms:>10.2f}  {steps_per_sec:>10.0f}  {metrics.total_edges:>8}")

    del swarm
    del world

print("\nDone.")
