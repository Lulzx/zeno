"""
Swarm simulation API for Zeno.

Provides high-level Python bindings for the swarm platform,
enabling multi-agent simulations with communication and neighbor detection.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np

from ._ffi import _check_lib, _lib, ffi, ZenoWorld


@dataclass
class SwarmConfig:
    """Configuration for a swarm simulation."""
    num_agents: int = 0
    communication_range: float = 10.0
    max_neighbors: int = 32
    max_message_bytes: int = 48
    max_messages_per_step: int = 4
    grid_cell_size: float = 10.0
    seed: int = 42
    enable_physics: bool = True
    latency_ticks: int = 0
    drop_prob: float = 0.0
    max_broadcast_recipients: int = 0xFFFFFFFF
    max_inbox_per_agent: int = 0
    strict_determinism: bool = True


@dataclass
class SwarmMetrics:
    """Metrics from a swarm step."""
    connectivity_ratio: float = 0.0
    fragmentation_score: float = 0.0
    collision_count: int = 0
    message_count: int = 0
    bytes_sent: int = 0
    total_edges: int = 0
    avg_neighbors: float = 0.0
    messages_dropped: int = 0
    convergence_time_ms: float = 0.0
    near_miss_count: int = 0
    task_success: float = 0.0


@dataclass
class TaskResult:
    """Result from a task evaluation."""
    score: float = 0.0
    complete: bool = False
    detail: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class ReplayStats:
    """Statistics from the replay recorder."""
    frame_count: int = 0
    total_bytes: int = 0
    recording: bool = False


TASK_TYPES = {
    "formation": 0,
    "coverage": 1,
    "pursuit": 2,
    "tracking": 3,
}

ATTACK_TYPES = {
    "none": 0,
    "jamming": 1,
    "dropout": 2,
    "byzantine": 3,
    "partition": 4,
}


class ZenoSwarm:
    """
    High-level swarm simulation wrapper.

    Manages agent neighbor detection, communication, and physics integration.

    Parameters
    ----------
    world : ZenoWorld
        The physics world containing agent bodies.
    config : SwarmConfig
        Swarm configuration.
    body_offset : int
        Index of the first agent body in the world's body array.
    """

    def __init__(self, world: ZenoWorld, config: SwarmConfig, body_offset: int = 0):
        _check_lib()

        self._world = world
        self._config = config
        self._num_agents = config.num_agents

        # Create native config
        c_config = ffi.new("ZenoSwarmConfig*")
        c_config.num_agents = config.num_agents
        c_config.communication_range = config.communication_range
        c_config.max_neighbors = config.max_neighbors
        c_config.max_message_bytes = config.max_message_bytes
        c_config.max_messages_per_step = config.max_messages_per_step
        c_config.grid_cell_size = config.grid_cell_size
        c_config.seed = config.seed
        c_config.enable_physics = config.enable_physics
        c_config.latency_ticks = config.latency_ticks
        c_config.drop_prob = config.drop_prob
        c_config.max_broadcast_recipients = config.max_broadcast_recipients
        c_config.max_inbox_per_agent = config.max_inbox_per_agent
        c_config.strict_determinism = config.strict_determinism

        self._handle = _lib.zeno_swarm_create(world._handle, c_config)
        if self._handle == ffi.NULL:
            raise RuntimeError("Failed to create swarm")

        _lib.zeno_swarm_set_body_offset(self._handle, body_offset)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle != ffi.NULL:
            _lib.zeno_swarm_destroy(self._handle)

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def step(self, actions: Optional[np.ndarray] = None) -> None:
        """
        Execute one swarm step (neighbor detection, communication, metrics).

        Parameters
        ----------
        actions : np.ndarray, optional
            Actions for all agents. If None, no external actions applied.
        """
        if actions is not None:
            actions = np.ascontiguousarray(actions, dtype=np.float32).flatten()
            actions_ptr = ffi.cast("float*", actions.ctypes.data)
        else:
            actions_ptr = ffi.NULL

        result = _lib.zeno_swarm_step(self._handle, self._world._handle, actions_ptr)
        if result != 0:
            raise RuntimeError(f"Swarm step failed with error code {result}")

    def get_metrics(self) -> SwarmMetrics:
        """Get metrics from the most recent step."""
        c_metrics = ffi.new("ZenoSwarmMetrics*")
        _lib.zeno_swarm_get_metrics(self._handle, c_metrics)
        return SwarmMetrics(
            connectivity_ratio=c_metrics.connectivity_ratio,
            fragmentation_score=c_metrics.fragmentation_score,
            collision_count=c_metrics.collision_count,
            message_count=c_metrics.message_count,
            bytes_sent=c_metrics.bytes_sent,
            total_edges=c_metrics.total_edges,
            avg_neighbors=c_metrics.avg_neighbors,
            messages_dropped=c_metrics.messages_dropped,
            convergence_time_ms=c_metrics.convergence_time_ms,
            near_miss_count=c_metrics.near_miss_count,
            task_success=c_metrics.task_success,
        )

    def get_neighbor_counts(self) -> np.ndarray:
        """Get neighbor count for each agent as numpy array."""
        out = np.zeros(self._num_agents, dtype=np.uint32)
        out_ptr = ffi.cast("uint32_t*", out.ctypes.data)
        _lib.zeno_swarm_get_neighbor_counts(self._handle, out_ptr, self._num_agents)
        return out

    def set_body_offset(self, offset: int) -> None:
        """Set the body offset for agent bodies."""
        _lib.zeno_swarm_set_body_offset(self._handle, offset)

    def evaluate_task(self, task_type: str, **params) -> TaskResult:
        """
        Evaluate a cooperative task objective.

        Parameters
        ----------
        task_type : str
            One of: "formation", "coverage", "pursuit", "tracking".
        **params : float
            Task-specific parameters. See docs for each task type.

        Returns
        -------
        TaskResult
            Score and completion status.
        """
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task type: {task_type}. Must be one of {list(TASK_TYPES)}")

        # Build params array from kwargs
        param_names = {
            "formation": ["center_x", "center_y", "target_radius", "formation_type"],
            "coverage": ["x_min", "y_min", "x_max", "y_max", "cell_size"],
            "pursuit": ["num_pursuers", "capture_radius"],
            "tracking": ["target_x", "target_y", "target_z", "track_radius"],
        }

        param_arr = [0.0] * 8
        for i, name in enumerate(param_names.get(task_type, [])):
            if name in params:
                param_arr[i] = float(params[name])

        c_params = ffi.new("float[8]", param_arr)
        c_result = ffi.new("ZenoTaskResult*")

        result = _lib.zeno_swarm_evaluate_task(
            self._handle, self._world._handle,
            TASK_TYPES[task_type], c_params, c_result
        )
        if result != 0:
            raise RuntimeError(f"Task evaluation failed with error code {result}")

        return TaskResult(
            score=c_result.score,
            complete=bool(c_result.complete),
            detail=(c_result.detail[0], c_result.detail[1],
                    c_result.detail[2], c_result.detail[3]),
        )

    def apply_attack(
        self,
        attack_type: str,
        intensity: float = 0.0,
        target_agents: Optional[list] = None,
        seed: int = 0,
    ) -> None:
        """
        Apply an adversarial attack to the swarm.

        Parameters
        ----------
        attack_type : str
            One of: "none", "jamming", "dropout", "byzantine", "partition".
        intensity : float
            Attack severity [0, 1].
        target_agents : list[int], optional
            Agent IDs to target (max 16).
        seed : int
            Random seed for deterministic attacks.
        """
        if attack_type not in ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}. Must be one of {list(ATTACK_TYPES)}")

        c_config = ffi.new("ZenoAttackConfig*")
        c_config.attack_type = ATTACK_TYPES[attack_type]
        c_config.intensity = intensity
        c_config.seed = seed

        if target_agents:
            c_config.num_targets = min(len(target_agents), 16)
            for i in range(c_config.num_targets):
                c_config.target_agents[i] = target_agents[i]

        result = _lib.zeno_swarm_apply_attack(self._handle, c_config)
        if result != 0:
            raise RuntimeError(f"Attack application failed with error code {result}")

    def start_recording(self) -> None:
        """Start recording replay frames."""
        result = _lib.zeno_swarm_start_recording(self._handle)
        if result != 0:
            raise RuntimeError(f"Start recording failed with error code {result}")

    def stop_recording(self) -> None:
        """Stop recording replay frames."""
        result = _lib.zeno_swarm_stop_recording(self._handle)
        if result != 0:
            raise RuntimeError(f"Stop recording failed with error code {result}")

    def get_replay_stats(self) -> ReplayStats:
        """Get replay recorder statistics."""
        c_stats = ffi.new("ZenoReplayStats*")
        result = _lib.zeno_swarm_get_replay_stats(self._handle, c_stats)
        if result != 0:
            raise RuntimeError(f"Get replay stats failed with error code {result}")
        return ReplayStats(
            frame_count=c_stats.frame_count,
            total_bytes=c_stats.total_bytes,
            recording=bool(c_stats.recording),
        )


def create_swarm_world(
    num_agents: int,
    agent_radius: float = 0.1,
    num_envs: int = 1,
    layout: str = "grid",
    spacing: float = 0.5,
    communication_range: float = 10.0,
    max_contacts_per_env: int = 256,
) -> Tuple[ZenoWorld, ZenoSwarm]:
    """
    Create a world with N agent bodies + ground plane and a swarm instance.

    Parameters
    ----------
    num_agents : int
        Number of agents per environment.
    agent_radius : float
        Radius of each agent sphere.
    num_envs : int
        Number of parallel environments.
    layout : str
        Initial layout: "grid", "random", or "circle".
    spacing : float
        Spacing between agents in grid layout.
    communication_range : float
        Communication range for neighbor detection.
    max_contacts_per_env : int
        Maximum contacts per environment.

    Returns
    -------
    tuple
        (ZenoWorld, ZenoSwarm) ready for simulation.
    """
    _check_lib()

    # Generate MJCF XML with ground + agent bodies
    bodies_xml = ""
    side = max(1, int(math.ceil(math.sqrt(num_agents))))

    for i in range(num_agents):
        if layout == "grid":
            row = i // side
            col = i % side
            x = col * spacing
            y = row * spacing
            z = agent_radius + 0.01
        elif layout == "circle":
            angle = 2 * math.pi * i / num_agents
            radius = spacing * num_agents / (2 * math.pi)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = agent_radius + 0.01
        else:  # random
            x = (i % side) * spacing
            y = (i // side) * spacing
            z = agent_radius + 0.01

        bodies_xml += f"""
        <body pos="{x:.4f} {y:.4f} {z:.4f}">
            <freejoint/>
            <geom type="sphere" size="{agent_radius}" mass="1.0"
                   contype="0" conaffinity="0"/>
        </body>"""

    mjcf_xml = f"""<mujoco>
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <worldbody>
        <geom type="plane" size="1000 1000 0.01" pos="0 0 0"/>
        {bodies_xml}
    </worldbody>
</mujoco>"""

    world = ZenoWorld(
        mjcf_string=mjcf_xml,
        num_envs=num_envs,
        max_contacts_per_env=max_contacts_per_env,
        max_bodies_per_env=num_agents + 1,
        max_geoms_per_env=num_agents + 1,
    )

    swarm_config = SwarmConfig(
        num_agents=num_agents,
        communication_range=communication_range,
        grid_cell_size=max(communication_range, spacing * side),
    )

    swarm = ZenoSwarm(world, swarm_config, body_offset=1)  # skip ground plane

    return world, swarm
