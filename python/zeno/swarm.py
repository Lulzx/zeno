"""
Swarm simulation API for Zeno.

Provides high-level Python bindings for the swarm platform,
enabling multi-agent simulations with communication and neighbor detection.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
