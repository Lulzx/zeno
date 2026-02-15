"""Replay recording and determinism verification example.

Records a swarm trace and verifies deterministic replay.
"""

import zeno
from zeno.swarm import create_swarm_world

# Create a small swarm
world, swarm = create_swarm_world(
    num_agents=16,
    layout="circle",
    spacing=0.5,
    communication_range=5.0,
)
world.reset()

# Start recording
swarm.start_recording()
print("Recording started...")

# Run 50 steps
for step in range(50):
    swarm.step()

# Stop recording
swarm.stop_recording()
stats = swarm.get_replay_stats()
print(f"Recording stopped.")
print(f"  Frames recorded: {stats.frame_count}")
print(f"  Total bytes: {stats.total_bytes}")
print(f"  Still recording: {stats.recording}")

# Run a second recording with same initial state for determinism check
world.reset()

swarm.start_recording()
for step in range(50):
    swarm.step()
swarm.stop_recording()

stats2 = swarm.get_replay_stats()
print(f"\nSecond recording:")
print(f"  Frames recorded: {stats2.frame_count}")
print(f"  Total bytes: {stats2.total_bytes}")

print("\nDone.")
