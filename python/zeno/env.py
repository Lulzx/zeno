"""
Main Zeno environment class with Gymnasium-compatible API.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ._ffi import ZenoWorld


class ZenoEnv:
    """
    High-performance batched physics environment.

    This environment simulates multiple parallel instances of a physics
    world, optimized for GPU acceleration on Apple Silicon.

    Parameters
    ----------
    mjcf_path : str, optional
        Path to an MJCF (MuJoCo XML) file describing the physics model.
    mjcf_string : str, optional
        MJCF XML string (alternative to mjcf_path).
    num_envs : int
        Number of parallel environments to simulate.
    timestep : float
        Physics timestep in seconds.
    contact_iterations : int
        Number of contact solver iterations.
    max_contacts_per_env : int
        Maximum contacts per environment.
    seed : int
        Random seed for reproducibility.
    substeps : int
        Number of physics substeps per step() call.

    Examples
    --------
    >>> env = ZenoEnv("ant.xml", num_envs=1024)
    >>> obs = env.reset()
    >>> for _ in range(1000):
    ...     actions = np.random.uniform(-1, 1, (1024, env.action_dim))
    ...     obs, rewards, dones, info = env.step(actions)
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        mjcf_string: Optional[str] = None,
        num_envs: int = 1,
        timestep: float = 0,  # 0 = use MJCF timestep
        contact_iterations: int = 4,
        max_contacts_per_env: int = 64,
        seed: int = 42,
        substeps: int = 1,
    ):
        self._world = ZenoWorld(
            mjcf_path=mjcf_path,
            mjcf_string=mjcf_string,
            num_envs=num_envs,
            timestep=timestep,
            contact_iterations=contact_iterations,
            max_contacts_per_env=max_contacts_per_env,
            seed=seed,
            substeps=substeps,
        )

        self._num_envs = num_envs
        self._step_count = 0
        self._max_episode_steps = 1000
        self._mjcf_path = mjcf_path  # Store for viewer

        # Cache observation and action shapes
        self._obs_shape = (num_envs, self._world.obs_dim)
        self._action_shape = (num_envs, self._world.action_dim)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def observation_dim(self) -> int:
        """Observation dimension per environment."""
        return self._world.obs_dim

    @property
    def action_dim(self) -> int:
        """Action dimension per environment."""
        return self._world.action_dim

    @property
    def timestep(self) -> float:
        """Physics timestep."""
        return self._world.timestep

    @property
    def observation_shape(self) -> Tuple[int, int]:
        """Shape of observation array: (num_envs, obs_dim)."""
        return self._obs_shape

    @property
    def action_shape(self) -> Tuple[int, int]:
        """Shape of action array: (num_envs, action_dim)."""
        return self._action_shape

    def reset(
        self, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reset environments.

        Parameters
        ----------
        mask : np.ndarray, optional
            Boolean array of shape (num_envs,) indicating which environments
            to reset. If None, all environments are reset.

        Returns
        -------
        observations : np.ndarray
            Initial observations of shape (num_envs, obs_dim).
        """
        self._world.reset(mask)
        self._step_count = 0
        return self._world.get_observations().copy()

    def step(
        self, actions: np.ndarray, substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        actions : np.ndarray
            Actions of shape (num_envs, action_dim) or (num_envs * action_dim,).
        substeps : int, optional
            Override number of physics substeps. 0 uses default.

        Returns
        -------
        observations : np.ndarray
            Observations of shape (num_envs, obs_dim).
        rewards : np.ndarray
            Rewards of shape (num_envs,).
        dones : np.ndarray
            Done flags of shape (num_envs,).
        info : dict
            Additional information.
        """
        # Ensure actions are float32 and correct shape
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self._action_shape:
            actions = actions.reshape(self._action_shape)

        # Step physics
        self._world.step(actions, substeps)
        self._step_count += 1

        # Get results
        observations = self._world.get_observations().copy()
        rewards = self._world.get_rewards()
        dones = self._world.get_dones()

        # Check for truncation (max episode steps)
        truncated = np.zeros(self._num_envs, dtype=bool)
        if self._step_count >= self._max_episode_steps:
            truncated[:] = True

        info = {
            "step_count": self._step_count,
            "truncated": truncated,
        }

        return observations, rewards, dones.astype(bool), info

    def get_body_positions(self) -> np.ndarray:
        """
        Get body positions for visualization.

        Returns
        -------
        positions : np.ndarray
            Body positions of shape (num_envs, num_bodies, 4).
            The 4th component is padding.
        """
        return self._world.get_body_positions()

    def get_body_quaternions(self) -> np.ndarray:
        """
        Get body orientations for visualization.

        Returns
        -------
        quaternions : np.ndarray
            Body quaternions of shape (num_envs, num_bodies, 4).
            Format is (x, y, z, w).
        """
        return self._world.get_body_quaternions()

    def close(self) -> None:
        """Release resources."""
        # World cleanup is handled by __del__
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make(
    model: str,
    num_envs: int = 1,
    **kwargs
) -> ZenoEnv:
    """
    Create a Zeno environment from a model name or path.

    Parameters
    ----------
    model : str
        Model name (e.g., "ant", "humanoid") or path to MJCF file.
    num_envs : int
        Number of parallel environments.
    **kwargs
        Additional arguments passed to ZenoEnv.

    Returns
    -------
    env : ZenoEnv
        The created environment.

    Examples
    --------
    >>> env = zeno.make("ant.xml", num_envs=1024)
    >>> env = zeno.make("assets/humanoid.xml", num_envs=256)
    """
    # Check if it's a path
    path = Path(model)
    if path.exists():
        return ZenoEnv(mjcf_path=str(path), num_envs=num_envs, **kwargs)

    # Check standard asset locations
    asset_dirs = [
        Path(__file__).parent.parent.parent / "assets",  # ../../../assets
        Path.cwd() / "assets",
        Path.home() / ".zeno" / "assets",
    ]

    # Add .xml extension if not present
    if not model.endswith(".xml"):
        model = f"{model}.xml"

    for asset_dir in asset_dirs:
        asset_path = asset_dir / model
        if asset_path.exists():
            return ZenoEnv(mjcf_path=str(asset_path), num_envs=num_envs, **kwargs)

    raise FileNotFoundError(
        f"Could not find model '{model}'. "
        f"Searched: {[str(d) for d in asset_dirs]}"
    )
