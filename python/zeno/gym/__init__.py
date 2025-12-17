"""
Gymnasium integration for Zeno physics engine.

Provides Gymnasium-compatible wrappers for Zeno environments, enabling
seamless integration with RL libraries like Stable-Baselines3, CleanRL,
and others.

Quick Start
-----------
>>> import gymnasium as gym
>>> import zeno.gym  # Register environments
>>>
>>> # Single environment
>>> env = gym.make("Zeno/Ant-v0")
>>> obs, info = env.reset()
>>>
>>> # Vectorized environment (native GPU batching)
>>> from zeno.gym import make_vec
>>> envs = make_vec("ant", num_envs=1024)
>>> obs, info = envs.reset()

Available Environments
---------------------
- Zeno/Pendulum-v0
- Zeno/Cartpole-v0
- Zeno/Ant-v0
- Zeno/Humanoid-v0
- Zeno/HalfCheetah-v0
- Zeno/Hopper-v0
- Zeno/Walker2d-v0
- Zeno/Swimmer-v0
- Zeno/Reacher-v0
- Zeno/Pusher-v0
"""

from .registration import (
    register_envs,
    ZenoGymnasiumEnv,
    ZenoVectorEnv,
    make_vec,
    make_sb3_env,
    check_env,
    ENV_CONFIGS,
)

__all__ = [
    "register_envs",
    "ZenoGymnasiumEnv",
    "ZenoVectorEnv",
    "make_vec",
    "make_sb3_env",
    "check_env",
    "ENV_CONFIGS",
]

# Auto-register environments on import
try:
    register_envs()
except ImportError:
    # Gymnasium not installed
    pass
