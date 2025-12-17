"""
Gymnasium environment registration and wrappers.

This module provides full Gymnasium API compatibility for Zeno environments,
including single-environment and vectorized wrappers.

Features
--------
- Full Gymnasium API compliance (v0.29+)
- Native vectorization without subprocess overhead
- Zero-copy observation access for maximum performance
- Automatic environment registration
- Stable-Baselines3 compatibility
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import VectorEnv
    from gymnasium.envs.registration import EnvSpec
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    VectorEnv = object
    EnvSpec = None

from ..env import ZenoEnv


# Environment configurations
ENV_CONFIGS = {
    "Pendulum": {
        "mjcf_file": "pendulum.xml",
        "max_episode_steps": 200,
        "reward_threshold": -200.0,
    },
    "Cartpole": {
        "mjcf_file": "cartpole.xml",
        "max_episode_steps": 500,
        "reward_threshold": 475.0,
    },
    "Ant": {
        "mjcf_file": "ant.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 6000.0,
    },
    "Humanoid": {
        "mjcf_file": "humanoid.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 6000.0,
    },
    "HalfCheetah": {
        "mjcf_file": "cheetah.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 4800.0,
    },
    "Hopper": {
        "mjcf_file": "hopper.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 3800.0,
    },
    "Walker2d": {
        "mjcf_file": "walker.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 5000.0,
    },
    "Swimmer": {
        "mjcf_file": "swimmer.xml",
        "max_episode_steps": 1000,
        "reward_threshold": 360.0,
    },
}


class ZenoGymnasiumEnv(gym.Env if HAS_GYMNASIUM else object):
    """
    Gymnasium-compatible wrapper for a single Zeno environment.

    This wraps a batched Zeno environment with num_envs=1 to provide
    the standard Gymnasium API with full compatibility.

    Parameters
    ----------
    mjcf_path : str
        Path to MJCF file or model name.
    render_mode : str, optional
        Rendering mode ("rgb_array" or "human").
    max_episode_steps : int, optional
        Maximum steps per episode (default: 1000).
    **kwargs
        Additional arguments passed to ZenoEnv.

    Examples
    --------
    >>> env = ZenoGymnasiumEnv("ant.xml")
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         obs, info = env.reset()
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 60,
    }

    def __init__(
        self,
        mjcf_path: str,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        **kwargs
    ):
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium is required for ZenoGymnasiumEnv. "
                "Install with: pip install gymnasium"
            )

        super().__init__()

        # Force single environment
        kwargs["num_envs"] = 1
        self._env = ZenoEnv(mjcf_path=mjcf_path, **kwargs)
        self._mjcf_path = mjcf_path
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Define observation space
        obs_dim = self._env.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Define action space
        act_dim = self._env.action_dim
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        # For rendering
        self._viewer = None
        self._render_buffer = None

    @property
    def unwrapped(self) -> 'ZenoGymnasiumEnv':
        """Return the unwrapped environment."""
        return self

    @property
    def spec(self) -> Optional['EnvSpec']:
        """Return the environment spec."""
        return getattr(self, '_spec', None)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        observation : np.ndarray
            Initial observation.
        info : dict
            Additional information.
        """
        super().reset(seed=seed)

        # Handle seeding
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Reset environment
        obs = self._env.reset()
        self._elapsed_steps = 0

        info = {
            "TimeLimit.truncated": False,
        }

        return obs[0].astype(np.float32), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : np.ndarray
            Action to take.

        Returns
        -------
        observation : np.ndarray
            Observation after the step.
        reward : float
            Reward from the step.
        terminated : bool
            Whether the episode ended due to terminal state.
        truncated : bool
            Whether the episode was truncated (e.g., time limit).
        info : dict
            Additional information.
        """
        # Ensure action is correct format
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self._env.action_dim,):
            action = action.reshape(self._env.action_dim)

        # Reshape for batched env
        action = action.reshape(1, -1)

        # Step the environment
        obs, rewards, dones, env_info = self._env.step(action)

        self._elapsed_steps += 1

        # Determine termination/truncation
        terminated = bool(dones[0])
        truncated = self._elapsed_steps >= self._max_episode_steps

        # Build info dict
        info = {
            "TimeLimit.truncated": truncated and not terminated,
            "episode_step": self._elapsed_steps,
        }

        # Add env info
        if "truncated" in env_info:
            info["env_truncated"] = env_info["truncated"][0]

        return (
            obs[0].astype(np.float32),
            float(rewards[0]),
            terminated,
            truncated,
            info
        )

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns
        -------
        frame : np.ndarray or None
            RGB frame if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode == "rgb_array":
            # Get body positions and quaternions for rendering
            positions = self._env.get_body_positions()[0]  # First env
            quaternions = self._env.get_body_quaternions()[0]

            # Create simple visualization (placeholder)
            # In production, this would use Metal rendering
            width, height = 640, 480
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = [135, 206, 235]  # Sky blue background

            # Simple projection of body positions
            for i, pos in enumerate(positions):
                x = int(width / 2 + pos[0] * 100)
                y = int(height / 2 - pos[2] * 100)
                if 0 <= x < width and 0 <= y < height:
                    # Draw a circle for each body
                    radius = 5
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            if dx * dx + dy * dy <= radius * radius:
                                px, py = x + dx, y + dy
                                if 0 <= px < width and 0 <= py < height:
                                    frame[py, px] = [255, 100, 100]

            return frame

        elif self.render_mode == "human":
            # Would display using a window
            pass

        return None

    def close(self) -> None:
        """Close the environment and release resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._env.close()

    def get_wrapper_attr(self, name: str) -> Any:
        """Get attribute from wrapper chain."""
        return getattr(self, name)


class ZenoVectorEnv(VectorEnv if HAS_GYMNASIUM else object):
    """
    Gymnasium VectorEnv wrapper for batched Zeno environments.

    This provides the standard Gymnasium vector environment API while
    leveraging Zeno's native batched GPU simulation for maximum performance.

    Unlike standard Gymnasium vectorization (which uses subprocesses),
    ZenoVectorEnv runs all environments in a single process on the GPU,
    achieving orders of magnitude better performance.

    Parameters
    ----------
    mjcf_path : str
        Path to MJCF file or model name.
    num_envs : int
        Number of parallel environments.
    max_episode_steps : int, optional
        Maximum steps per episode (default: 1000).
    **kwargs
        Additional arguments passed to ZenoEnv.

    Examples
    --------
    >>> envs = ZenoVectorEnv("ant.xml", num_envs=1024)
    >>> obs, info = envs.reset()
    >>> for _ in range(1000):
    ...     actions = envs.action_space.sample()
    ...     obs, rewards, terminated, truncated, info = envs.step(actions)
    >>> envs.close()
    """

    def __init__(
        self,
        mjcf_path: str,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        **kwargs
    ):
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium is required for ZenoVectorEnv. "
                "Install with: pip install gymnasium"
            )

        self._env = ZenoEnv(mjcf_path=mjcf_path, num_envs=num_envs, **kwargs)
        self._mjcf_path = mjcf_path
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)

        # Define spaces
        obs_dim = self._env.observation_dim
        act_dim = self._env.action_dim

        single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        super().__init__(
            num_envs=num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )

        # Auto-reset tracking
        self._autoreset_envs = np.zeros(num_envs, dtype=bool)
        self._final_observations = [None] * num_envs
        self._final_infos = [{} for _ in range(num_envs)]

        # For async API
        self._pending_actions = None
        self._pending_reset = False

    @property
    def unwrapped(self) -> 'ZenoVectorEnv':
        """Return the unwrapped environment."""
        return self

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments.

        Parameters
        ----------
        seed : int or list of int, optional
            Random seed(s) for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        observations : np.ndarray
            Initial observations of shape (num_envs, obs_dim).
        infos : dict
            Dictionary of info arrays.
        """
        # Handle seeding
        if seed is not None:
            if isinstance(seed, int):
                seeds = [seed + i for i in range(self.num_envs)]
            else:
                seeds = seed
            self._np_random = np.random.default_rng(seeds[0])

        # Reset environment
        obs = self._env.reset()
        self._elapsed_steps[:] = 0
        self._autoreset_envs[:] = False

        infos = {
            "TimeLimit.truncated": np.zeros(self.num_envs, dtype=bool),
            "_elapsed_steps": self._elapsed_steps.copy(),
        }

        return obs.astype(np.float32), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step in all environments.

        Parameters
        ----------
        actions : np.ndarray
            Actions of shape (num_envs, action_dim).

        Returns
        -------
        observations : np.ndarray
            Observations of shape (num_envs, obs_dim).
        rewards : np.ndarray
            Rewards of shape (num_envs,).
        terminated : np.ndarray
            Termination flags of shape (num_envs,).
        truncated : np.ndarray
            Truncation flags of shape (num_envs,).
        infos : dict
            Dictionary of info arrays.
        """
        actions = np.asarray(actions, dtype=np.float32)

        # Auto-reset environments that were done last step
        if np.any(self._autoreset_envs):
            # Store final observations before reset
            self._env.reset(self._autoreset_envs.astype(np.uint8))
            self._elapsed_steps[self._autoreset_envs] = 0
            self._autoreset_envs[:] = False

        # Step physics
        obs, rewards, dones, env_info = self._env.step(actions)
        self._elapsed_steps += 1

        # Determine termination/truncation
        terminated = dones.astype(bool)
        truncated = self._elapsed_steps >= self._max_episode_steps

        # Mark for auto-reset
        self._autoreset_envs = terminated | truncated

        # Build info dict
        infos = {
            "TimeLimit.truncated": truncated & ~terminated,
            "_elapsed_steps": self._elapsed_steps.copy(),
            "final_observation": np.array([None] * self.num_envs, dtype=object),
            "final_info": np.array([{}] * self.num_envs, dtype=object),
        }

        # Store final observations for environments that will reset
        if np.any(self._autoreset_envs):
            for i in np.where(self._autoreset_envs)[0]:
                infos["final_observation"][i] = obs[i].copy()
                infos["final_info"][i] = {"episode_step": int(self._elapsed_steps[i])}

        return (
            obs.astype(np.float32),
            rewards.astype(np.float32),
            terminated,
            truncated,
            infos
        )

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Async reset (immediate for Zeno as it's GPU-native)."""
        self._pending_reset = True
        self._reset_seed = seed
        self._reset_options = options

    def reset_wait(
        self,
        timeout: Optional[float] = None,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wait for async reset to complete."""
        seed = seed or getattr(self, '_reset_seed', None)
        options = options or getattr(self, '_reset_options', None)
        self._pending_reset = False
        return self.reset(seed=seed, options=options)

    def step_async(self, actions: np.ndarray) -> None:
        """Async step (stores actions for step_wait)."""
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(
        self, timeout: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Wait for async step to complete."""
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        result = self.step(self._pending_actions)
        self._pending_actions = None
        return result

    def call(self, name: str, *args, **kwargs) -> Tuple[Any, ...]:
        """Call a method on all environments."""
        method = getattr(self._env, name, None)
        if method is None:
            raise AttributeError(f"Environment has no method '{name}'")
        if callable(method):
            result = method(*args, **kwargs)
            # Return as tuple for compatibility
            return (result,) * self.num_envs if not isinstance(result, np.ndarray) else tuple(result)
        raise TypeError(f"'{name}' is not callable")

    def get_attr(self, name: str) -> Tuple[Any, ...]:
        """Get an attribute from all environments."""
        attr = getattr(self._env, name)
        return (attr,) * self.num_envs

    def set_attr(self, name: str, values: Union[Any, Sequence[Any]]) -> None:
        """Set an attribute on all environments."""
        if isinstance(values, (list, tuple)):
            value = values[0]
        else:
            value = values
        setattr(self._env, name, value)

    def close_extras(self, **kwargs) -> None:
        """Close extra resources."""
        self._env.close()

    def close(self) -> None:
        """Close the environment."""
        self.close_extras()

    # Additional methods for RL compatibility

    def get_body_positions(self, zero_copy: bool = True) -> np.ndarray:
        """Get body positions for all environments."""
        return self._env._world.get_body_positions(zero_copy=zero_copy)

    def get_body_quaternions(self, zero_copy: bool = True) -> np.ndarray:
        """Get body quaternions for all environments."""
        return self._env._world.get_body_quaternions(zero_copy=zero_copy)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get complete state for checkpointing."""
        return self._env._world.get_state()

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore state from checkpoint."""
        self._env._world.set_state(state)


def register_envs() -> None:
    """
    Register all Zeno environments with Gymnasium.

    This is called automatically when importing zeno.gym.
    """
    if not HAS_GYMNASIUM:
        return

    from pathlib import Path

    for env_name, config in ENV_CONFIGS.items():
        env_id = f"Zeno/{env_name}-v0"

        try:
            gym.register(
                id=env_id,
                entry_point="zeno.gym.registration:ZenoGymnasiumEnv",
                kwargs={"mjcf_path": config["mjcf_file"]},
                max_episode_steps=config["max_episode_steps"],
                reward_threshold=config.get("reward_threshold"),
            )
        except gym.error.Error:
            # Already registered
            pass


def make_vec(
    model: str,
    num_envs: int = 1,
    **kwargs
) -> ZenoVectorEnv:
    """
    Create a vectorized Zeno environment.

    This is the recommended way to create batched environments for
    reinforcement learning, as it provides native GPU vectorization
    without subprocess overhead.

    Parameters
    ----------
    model : str
        Model name (e.g., "ant", "humanoid") or path to MJCF file.
    num_envs : int
        Number of parallel environments.
    **kwargs
        Additional arguments passed to ZenoVectorEnv.

    Returns
    -------
    envs : ZenoVectorEnv
        Vectorized environment.

    Examples
    --------
    >>> envs = make_vec("ant", num_envs=1024)
    >>> obs, info = envs.reset()
    >>> print(obs.shape)  # (1024, obs_dim)
    """
    from pathlib import Path

    # Find MJCF file
    path = Path(model)
    if not path.exists():
        asset_dirs = [
            Path(__file__).parent.parent.parent.parent / "assets",
            Path.cwd() / "assets",
            Path.home() / ".zeno" / "assets",
        ]

        if not model.endswith(".xml"):
            model = f"{model}.xml"

        for asset_dir in asset_dirs:
            asset_path = asset_dir / model
            if asset_path.exists():
                path = asset_path
                break

    if not path.exists():
        raise FileNotFoundError(f"Could not find model: {model}")

    return ZenoVectorEnv(mjcf_path=str(path), num_envs=num_envs, **kwargs)


# Stable-Baselines3 compatibility helpers

def make_sb3_env(
    model: str,
    num_envs: int = 1,
    **kwargs
) -> ZenoVectorEnv:
    """
    Create a Zeno environment compatible with Stable-Baselines3.

    This is equivalent to make_vec but emphasizes SB3 compatibility.
    The returned environment can be directly used with SB3 algorithms.

    Parameters
    ----------
    model : str
        Model name or MJCF path.
    num_envs : int
        Number of parallel environments.
    **kwargs
        Additional arguments.

    Returns
    -------
    env : ZenoVectorEnv
        SB3-compatible vectorized environment.

    Examples
    --------
    >>> from stable_baselines3 import PPO
    >>> from zeno.gym import make_sb3_env
    >>>
    >>> env = make_sb3_env("ant", num_envs=8)
    >>> model = PPO("MlpPolicy", env, verbose=1)
    >>> model.learn(total_timesteps=100000)
    """
    return make_vec(model, num_envs=num_envs, **kwargs)


def check_env(env: Union[ZenoGymnasiumEnv, ZenoVectorEnv]) -> bool:
    """
    Check if an environment is properly configured.

    Runs basic validation to ensure the environment works correctly.

    Parameters
    ----------
    env : ZenoGymnasiumEnv or ZenoVectorEnv
        Environment to check.

    Returns
    -------
    valid : bool
        True if all checks pass.
    """
    try:
        # Check reset
        obs, info = env.reset()
        assert obs is not None, "Reset returned None observation"

        if isinstance(env, ZenoVectorEnv):
            assert obs.shape[0] == env.num_envs, "Wrong batch size"

        # Check step
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "Step should return 5 values"

        obs, reward, terminated, truncated, info = result
        assert obs is not None, "Step returned None observation"

        # Check spaces
        assert env.observation_space.contains(
            obs if obs.ndim == 1 else obs[0]
        ), "Observation not in space"

        return True
    except Exception as e:
        print(f"Environment check failed: {e}")
        return False
