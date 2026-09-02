"""Stable-Baselines3 adapter for Zeno's native batched environment."""

from typing import Any, Iterable, Optional, Type

import numpy as np

from .registration import ZenoVectorEnv

try:
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - exercised by the public factory
    raise ImportError(
        "Stable-Baselines3 support requires `pip install zeno-physics[sb3]`"
    ) from exc


class ZenoSB3VecEnv(VecEnv):
    """Translate Zeno's Gymnasium vector API to SB3's ``VecEnv`` contract."""

    def __init__(self, env: ZenoVectorEnv):
        self.env = env
        self._pending_actions: Optional[np.ndarray] = None
        self.render_mode = None
        super().__init__(
            env.num_envs,
            env.single_observation_space,
            env.single_action_space,
        )

    def reset(self) -> np.ndarray:
        seed = next((value for value in self._seeds if value is not None), None)
        observations, infos = self.env.reset(seed=seed)
        self.reset_infos = [
            {key: value[i] for key, value in infos.items() if not key.startswith("_")}
            for i in range(self.num_envs)
        ]
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        if self._pending_actions is not None:
            raise RuntimeError("step_async called while another step is pending")
        actions = np.asarray(actions, dtype=np.float32)
        self.env.step_async(actions)
        self._pending_actions = actions

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        try:
            observations, rewards, terminated, truncated, vector_info = self.env.step_wait()
        finally:
            self._pending_actions = None
        dones = terminated | truncated
        infos = []
        for i in range(self.num_envs):
            info = {
                "TimeLimit.truncated": bool(truncated[i] and not terminated[i]),
                "episode_step": int(self.env._elapsed_steps[i]),
            }
            for key, value in vector_info.items():
                if key.startswith("_") or key in {"final_observation", "final_info"}:
                    continue
                if isinstance(value, np.ndarray) and value.shape[:1] == (self.num_envs,):
                    info[key] = value[i]
            infos.append(info)

        if np.any(dones):
            terminal_observations = observations.copy()
            reset_observations = self.env._env.reset(dones.astype(np.uint8))
            observations[dones] = reset_observations[dones]
            self.env._elapsed_steps[dones] = 0
            self.env._autoreset_envs[dones] = False
            for i in np.flatnonzero(dones):
                infos[i]["terminal_observation"] = terminal_observations[i]

        return observations, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def _indices(self, indices) -> list[int]:
        return list(self._get_indices(indices))

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        if hasattr(self.env, attr_name):
            value = getattr(self.env, attr_name)
        elif hasattr(self.env._env, attr_name):
            value = getattr(self.env._env, attr_name)
        else:
            raise AttributeError(attr_name)
        return [value for _ in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        selected = self._indices(indices)
        if selected != list(range(self.num_envs)):
            raise ValueError(f"{attr_name!r} is shared by the native batch and cannot be set per environment")
        target = self.env if hasattr(self.env, attr_name) else self.env._env
        setattr(target, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        method = getattr(self.env, method_name, None) or getattr(self.env._env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in self._indices(indices)]

    def env_is_wrapped(self, wrapper_class: Type, indices=None) -> list[bool]:
        return [False for _ in self._indices(indices)]
