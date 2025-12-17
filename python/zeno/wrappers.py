"""
Environment wrappers for common RL training patterns.

This module provides wrappers that implement standard preprocessing
and utility transformations commonly used in reinforcement learning.

Features
--------
- Observation normalization (running mean/std)
- Reward scaling and clipping
- Action repeat (frame skip)
- Episode statistics tracking
- Checkpointing support
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from .env import ZenoEnv


class RunningMeanStd:
    """
    Tracks running mean and standard deviation using Welford's algorithm.

    This is numerically stable and suitable for online computation.
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Small initial count for stability

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch of observations."""
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update mean
        self.mean = self.mean + delta * batch_count / total_count

        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count

        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize observations using running statistics."""
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -clip,
            clip
        ).astype(np.float32)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get state for serialization."""
        return {
            "mean": self.mean,
            "var": self.var,
            "count": np.array([self.count]),
        }

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore state from serialization."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = float(state["count"][0])


class NormalizeObservation:
    """
    Wrapper that normalizes observations using running statistics.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    clip : float
        Clipping range for normalized observations.
    epsilon : float
        Small constant for numerical stability.

    Examples
    --------
    >>> env = NormalizeObservation(ZenoEnv("ant.xml", num_envs=8))
    >>> obs = env.reset()
    >>> # Observations are now normalized
    """

    def __init__(
        self,
        env: ZenoEnv,
        clip: float = 10.0,
        epsilon: float = 1e-8
    ):
        self.env = env
        self.clip = clip
        self.obs_rms = RunningMeanStd(
            shape=(env.observation_dim,),
            epsilon=epsilon
        )
        self._training = True

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def train(self) -> None:
        """Enable training mode (update statistics)."""
        self._training = True

    def eval(self) -> None:
        """Enable evaluation mode (freeze statistics)."""
        self._training = False

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset and normalize observations."""
        obs = self.env.reset(mask)
        if self._training:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs, self.clip)

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step and normalize observations."""
        obs, rewards, dones, info = self.env.step(actions, substeps)
        if self._training:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs, self.clip), rewards, dones, info

    def get_wrapper_state(self) -> Dict[str, Any]:
        """Get wrapper state for checkpointing."""
        return {"obs_rms": self.obs_rms.get_state()}

    def set_wrapper_state(self, state: Dict[str, Any]) -> None:
        """Restore wrapper state from checkpoint."""
        self.obs_rms.set_state(state["obs_rms"])

    def close(self) -> None:
        self.env.close()


class NormalizeReward:
    """
    Wrapper that normalizes rewards using running statistics.

    Normalizes rewards by dividing by the running standard deviation
    of discounted returns, without shifting the mean.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    gamma : float
        Discount factor for return estimation.
    clip : float
        Clipping range for normalized rewards.
    epsilon : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        env: ZenoEnv,
        gamma: float = 0.99,
        clip: float = 10.0,
        epsilon: float = 1e-8
    ):
        self.env = env
        self.gamma = gamma
        self.clip = clip
        self.return_rms = RunningMeanStd(shape=(), epsilon=epsilon)
        self._returns = np.zeros(env.num_envs, dtype=np.float32)
        self._training = True

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def train(self) -> None:
        """Enable training mode."""
        self._training = True

    def eval(self) -> None:
        """Enable evaluation mode."""
        self._training = False

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environments."""
        obs = self.env.reset(mask)
        if mask is None:
            self._returns[:] = 0
        else:
            self._returns[mask] = 0
        return obs

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step and normalize rewards."""
        obs, rewards, dones, info = self.env.step(actions, substeps)

        # Update return estimates
        self._returns = self._returns * self.gamma + rewards

        if self._training:
            self.return_rms.update(self._returns.reshape(-1, 1))

        # Normalize rewards
        norm_rewards = np.clip(
            rewards / np.sqrt(self.return_rms.var + 1e-8),
            -self.clip,
            self.clip
        ).astype(np.float32)

        # Reset returns for done environments
        self._returns[dones.astype(bool)] = 0

        return obs, norm_rewards, dones, info

    def get_wrapper_state(self) -> Dict[str, Any]:
        """Get wrapper state for checkpointing."""
        return {
            "return_rms": self.return_rms.get_state(),
            "returns": self._returns.copy(),
        }

    def set_wrapper_state(self, state: Dict[str, Any]) -> None:
        """Restore wrapper state."""
        self.return_rms.set_state(state["return_rms"])
        self._returns = state["returns"].copy()

    def close(self) -> None:
        self.env.close()


class ActionRepeat:
    """
    Wrapper that repeats actions for multiple steps.

    Useful for frame skipping in high-frequency simulations.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    repeat : int
        Number of times to repeat each action.
    """

    def __init__(self, env: ZenoEnv, repeat: int = 4):
        self.env = env
        self.repeat = repeat

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.env.reset(mask)

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute action multiple times, accumulating rewards."""
        total_rewards = np.zeros(self.env.num_envs, dtype=np.float32)

        for _ in range(self.repeat):
            obs, rewards, dones, info = self.env.step(actions, substeps)
            total_rewards += rewards

            # Break if any environment is done
            if np.any(dones):
                break

        return obs, total_rewards, dones, info

    def close(self) -> None:
        self.env.close()


class EpisodeStats:
    """
    Wrapper that tracks episode statistics.

    Tracks episode returns and lengths for logging.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    """

    def __init__(self, env: ZenoEnv):
        self.env = env

        # Per-environment tracking
        self._episode_rewards = np.zeros(env.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(env.num_envs, dtype=np.int32)

        # Completed episode stats
        self.completed_episodes = 0
        self.total_rewards = 0.0
        self.total_lengths = 0

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        obs = self.env.reset(mask)
        if mask is None:
            self._episode_rewards[:] = 0
            self._episode_lengths[:] = 0
        else:
            self._episode_rewards[mask] = 0
            self._episode_lengths[mask] = 0
        return obs

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rewards, dones, info = self.env.step(actions, substeps)

        # Update running stats
        self._episode_rewards += rewards
        self._episode_lengths += 1

        # Track completed episodes
        done_mask = dones.astype(bool)
        if np.any(done_mask):
            num_done = np.sum(done_mask)
            self.completed_episodes += num_done
            self.total_rewards += np.sum(self._episode_rewards[done_mask])
            self.total_lengths += np.sum(self._episode_lengths[done_mask])

            # Add to info
            info["episode_returns"] = self._episode_rewards[done_mask].copy()
            info["episode_lengths"] = self._episode_lengths[done_mask].copy()

            # Reset stats for done environments
            self._episode_rewards[done_mask] = 0
            self._episode_lengths[done_mask] = 0

        return obs, rewards, dones, info

    @property
    def mean_episode_return(self) -> float:
        """Mean episode return over all completed episodes."""
        if self.completed_episodes == 0:
            return 0.0
        return self.total_rewards / self.completed_episodes

    @property
    def mean_episode_length(self) -> float:
        """Mean episode length over all completed episodes."""
        if self.completed_episodes == 0:
            return 0.0
        return self.total_lengths / self.completed_episodes

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            "completed_episodes": self.completed_episodes,
            "mean_return": self.mean_episode_return,
            "mean_length": self.mean_episode_length,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.completed_episodes = 0
        self.total_rewards = 0.0
        self.total_lengths = 0

    def close(self) -> None:
        self.env.close()


class ClipAction:
    """
    Wrapper that clips actions to a valid range.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    low : float
        Lower bound for actions.
    high : float
        Upper bound for actions.
    """

    def __init__(
        self,
        env: ZenoEnv,
        low: float = -1.0,
        high: float = 1.0
    ):
        self.env = env
        self.low = low
        self.high = high

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.env.reset(mask)

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        clipped_actions = np.clip(actions, self.low, self.high)
        return self.env.step(clipped_actions, substeps)

    def close(self) -> None:
        self.env.close()


class TimeLimit:
    """
    Wrapper that enforces a maximum episode length.

    Parameters
    ----------
    env : ZenoEnv
        The environment to wrap.
    max_steps : int
        Maximum steps per episode.
    """

    def __init__(self, env: ZenoEnv, max_steps: int = 1000):
        self.env = env
        self.max_steps = max_steps
        self._step_counts = np.zeros(env.num_envs, dtype=np.int32)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        obs = self.env.reset(mask)
        if mask is None:
            self._step_counts[:] = 0
        else:
            self._step_counts[mask] = 0
        return obs

    def step(
        self,
        actions: np.ndarray,
        substeps: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rewards, dones, info = self.env.step(actions, substeps)
        self._step_counts += 1

        # Check for time limit
        truncated = self._step_counts >= self.max_steps
        dones = dones | truncated.astype(np.uint8)
        info["TimeLimit.truncated"] = truncated

        # Reset step counts for done environments
        self._step_counts[dones.astype(bool)] = 0

        return obs, rewards, dones, info

    def close(self) -> None:
        self.env.close()


def wrap_env(
    env: ZenoEnv,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_action: bool = True,
    track_stats: bool = True,
    gamma: float = 0.99,
    clip: float = 10.0,
) -> ZenoEnv:
    """
    Apply standard wrappers to a Zeno environment.

    This is a convenience function that applies commonly used wrappers
    for RL training.

    Parameters
    ----------
    env : ZenoEnv
        Base environment.
    normalize_obs : bool
        Whether to normalize observations.
    normalize_reward : bool
        Whether to normalize rewards.
    clip_action : bool
        Whether to clip actions.
    track_stats : bool
        Whether to track episode statistics.
    gamma : float
        Discount factor for reward normalization.
    clip : float
        Clipping range for normalization.

    Returns
    -------
    env : ZenoEnv
        Wrapped environment.

    Examples
    --------
    >>> from zeno import ZenoEnv
    >>> from zeno.wrappers import wrap_env
    >>>
    >>> env = wrap_env(ZenoEnv("ant.xml", num_envs=8))
    >>> obs = env.reset()  # Normalized observations
    """
    if clip_action:
        env = ClipAction(env)

    if normalize_obs:
        env = NormalizeObservation(env, clip=clip)

    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma, clip=clip)

    if track_stats:
        env = EpisodeStats(env)

    return env
