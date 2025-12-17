"""
Gymnasium environment registration and wrappers.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import VectorEnv
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    VectorEnv = object

from ..env import ZenoEnv


class ZenoGymnasiumEnv(gym.Env if HAS_GYMNASIUM else object):
    """
    Gymnasium-compatible wrapper for a single Zeno environment.

    This wraps a batched Zeno environment with num_envs=1 to provide
    the standard Gymnasium API.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        mjcf_path: str,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required for ZenoGymnasiumEnv")

        super().__init__()

        # Force single environment
        kwargs["num_envs"] = 1
        self._env = ZenoEnv(mjcf_path=mjcf_path, **kwargs)
        self.render_mode = render_mode

        # Define spaces
        obs_dim = self._env.observation_dim
        act_dim = self._env.action_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        obs = self._env.reset()
        return obs[0], {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)

        obs, rewards, dones, info = self._env.step(action)

        terminated = bool(dones[0])
        truncated = bool(info.get("truncated", [False])[0])

        return obs[0], float(rewards[0]), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return placeholder - actual rendering would require additional setup
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class ZenoVectorEnv(VectorEnv if HAS_GYMNASIUM else object):
    """
    Gymnasium VectorEnv wrapper for batched Zeno environments.

    This provides the standard Gymnasium vector environment API while
    leveraging Zeno's native batched simulation.
    """

    def __init__(
        self,
        mjcf_path: str,
        num_envs: int = 1,
        **kwargs
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required for ZenoVectorEnv")

        self._env = ZenoEnv(mjcf_path=mjcf_path, num_envs=num_envs, **kwargs)

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

        self._autoreset_envs = np.zeros(num_envs, dtype=bool)

    def reset(
        self,
        seed: Optional[Union[int, list]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        obs = self._env.reset()
        self._autoreset_envs[:] = False
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one step in all environments."""
        actions = np.asarray(actions, dtype=np.float32)

        # Auto-reset environments that were done
        if np.any(self._autoreset_envs):
            mask = self._autoreset_envs.astype(np.uint8)
            self._env.reset(mask)
            self._autoreset_envs[:] = False

        obs, rewards, dones, info = self._env.step(actions)

        terminated = dones.astype(bool)
        truncated = info.get("truncated", np.zeros(self.num_envs, dtype=bool))

        # Mark environments for auto-reset
        self._autoreset_envs = terminated | truncated

        return obs, rewards, terminated, truncated, info

    def reset_async(
        self,
        seed: Optional[Union[int, list]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Async reset (immediate for Zeno)."""
        self._pending_reset = True

    def reset_wait(
        self,
        seed: Optional[Union[int, list]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Wait for async reset."""
        return self.reset(seed=seed, options=options)

    def step_async(self, actions: np.ndarray) -> None:
        """Async step (immediate for Zeno)."""
        self._pending_actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Wait for async step."""
        return self.step(self._pending_actions)

    def call(self, name: str, *args, **kwargs) -> list:
        """Call a method on all environments."""
        method = getattr(self._env, name, None)
        if method is not None and callable(method):
            return [method(*args, **kwargs)]
        raise AttributeError(f"Environment has no method '{name}'")

    def get_attr(self, name: str) -> list:
        """Get an attribute from all environments."""
        return [getattr(self._env, name)]

    def set_attr(self, name: str, values: list) -> None:
        """Set an attribute on all environments."""
        setattr(self._env, name, values[0])

    def close_extras(self, **kwargs) -> None:
        """Close extra resources."""
        self._env.close()


def register_envs():
    """Register Zeno environments with Gymnasium."""
    if not HAS_GYMNASIUM:
        return

    # Standard environments
    envs = [
        ("Zeno/Pendulum-v0", "pendulum.xml"),
        ("Zeno/Cartpole-v0", "cartpole.xml"),
        ("Zeno/Ant-v0", "ant.xml"),
        ("Zeno/Humanoid-v0", "humanoid.xml"),
    ]

    for env_id, mjcf_file in envs:
        try:
            gym.register(
                id=env_id,
                entry_point="zeno.gym.registration:ZenoGymnasiumEnv",
                kwargs={"mjcf_path": mjcf_file},
                max_episode_steps=1000,
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

    Parameters
    ----------
    model : str
        Model name or path to MJCF file.
    num_envs : int
        Number of parallel environments.
    **kwargs
        Additional arguments passed to ZenoVectorEnv.

    Returns
    -------
    env : ZenoVectorEnv
        Vectorized environment.
    """
    from pathlib import Path

    # Find MJCF file
    path = Path(model)
    if not path.exists():
        asset_dirs = [
            Path(__file__).parent.parent.parent.parent / "assets",
            Path.cwd() / "assets",
        ]

        if not model.endswith(".xml"):
            model = f"{model}.xml"

        for asset_dir in asset_dirs:
            asset_path = asset_dir / model
            if asset_path.exists():
                path = asset_path
                break

    return ZenoVectorEnv(mjcf_path=str(path), num_envs=num_envs, **kwargs)
