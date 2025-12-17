"""
Gymnasium integration for Zeno physics engine.

Provides Gymnasium-compatible wrappers for Zeno environments.
"""

from .registration import register_envs, ZenoGymnasiumEnv, ZenoVectorEnv

__all__ = [
    "register_envs",
    "ZenoGymnasiumEnv",
    "ZenoVectorEnv",
]

# Auto-register environments on import
try:
    register_envs()
except ImportError:
    # Gymnasium not installed
    pass
