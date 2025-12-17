"""
Zeno Physics Engine - High-Performance Batched Robotics Simulation

A GPU-accelerated rigid body physics simulation engine optimized for
reinforcement learning and robot policy training on Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "Lulzx"

from .env import ZenoEnv, make
from ._ffi import is_metal_available, version

__all__ = [
    "ZenoEnv",
    "make",
    "is_metal_available",
    "version",
]
