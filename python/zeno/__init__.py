"""
Zeno Physics Engine - High-Performance Batched Robotics Simulation

A GPU-accelerated rigid body physics simulation engine optimized for
reinforcement learning and robot policy training on Apple Silicon.

Features
--------
- Native Metal GPU acceleration with unified memory
- Zero-copy CPU/GPU data access via CFFI
- 1,024 to 16,384+ parallel environments
- Full Gymnasium API compatibility
- MJCF (MuJoCo XML) model loading

Quick Start
-----------
>>> import zeno
>>> import numpy as np
>>>
>>> # Create 1024 parallel environments
>>> env = zeno.make("ant.xml", num_envs=1024)
>>>
>>> # Reset all environments
>>> obs = env.reset()
>>>
>>> # Run simulation
>>> for _ in range(1000):
...     actions = np.random.uniform(-1, 1, (1024, env.action_dim))
...     obs, rewards, dones, info = env.step(actions)
>>>
>>> env.close()

Gymnasium Integration
--------------------
>>> import gymnasium as gym
>>> import zeno.gym  # Register environments
>>>
>>> # Single environment
>>> env = gym.make("Zeno/Ant-v0")
>>> obs, info = env.reset()
>>>
>>> # Vectorized (native GPU batching)
>>> from zeno.gym import make_vec
>>> envs = make_vec("ant", num_envs=1024)

Zero-Copy State Access
---------------------
>>> # Direct access to GPU memory (no copy)
>>> positions = env._world.get_body_positions(zero_copy=True)
>>> velocities = env._world.get_body_velocities(zero_copy=True)
>>>
>>> # Modify state directly
>>> positions[0, 0, 2] += 0.1  # Changes GPU memory
"""

__version__ = "0.1.0"
__author__ = "Lulzx"

from .env import ZenoEnv, make
from ._ffi import (
    is_metal_available,
    version,
    ZenoWorld,
    ZeroCopyArray,
)

# Wrappers for RL training
from .wrappers import (
    NormalizeObservation,
    NormalizeReward,
    ActionRepeat,
    EpisodeStats,
    ClipAction,
    TimeLimit,
    wrap_env,
    RunningMeanStd,
)

__all__ = [
    # Core
    "ZenoEnv",
    "ZenoWorld",
    "make",
    # Zero-copy
    "ZeroCopyArray",
    # Utilities
    "is_metal_available",
    "version",
    # Wrappers
    "NormalizeObservation",
    "NormalizeReward",
    "ActionRepeat",
    "EpisodeStats",
    "ClipAction",
    "TimeLimit",
    "wrap_env",
    "RunningMeanStd",
]


def info() -> dict:
    """
    Get information about the Zeno installation.

    Returns
    -------
    dict
        Dictionary containing version and capability information.
    """
    return {
        "version": __version__,
        "metal_available": is_metal_available(),
        "library_version": version(),
    }
