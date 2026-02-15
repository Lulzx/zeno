"""
CFFI bindings for the Zeno physics engine.
Provides zero-copy access to GPU buffers via unified memory.

Zero-Copy Architecture
----------------------
Zeno leverages Apple Silicon's unified memory to provide true zero-copy
access to simulation state. All state arrays (positions, velocities, etc.)
are stored in MTLBuffer objects with storageModeShared, meaning both CPU
and GPU can access the same physical memory without copies.

The numpy arrays returned by get_* methods are views into this shared memory,
providing O(1) state access regardless of the number of environments.

Thread Safety
-------------
State accessors are thread-safe for reading. Writing to state arrays
should only be done when no GPU work is in flight (after step() returns).

Memory Layout
-------------
All state is stored in Structure-of-Arrays (SoA) format with float4 alignment:
- Positions: [num_envs, num_bodies, 4] (x, y, z, padding)
- Quaternions: [num_envs, num_bodies, 4] (x, y, z, w)
- Velocities: [num_envs, num_bodies, 4] (vx, vy, vz, padding)
"""

import os
import sys
import weakref
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cffi
import numpy as np

# Initialize FFI
ffi = cffi.FFI()

# C declarations - Extended API
ffi.cdef("""
    // Types
    typedef void* ZenoWorldHandle;

    typedef struct {
        uint32_t num_envs;
        float timestep;
        uint32_t contact_iterations;
        uint32_t max_contacts_per_env;
        uint64_t seed;
        uint32_t substeps;
        bool enable_profiling;
        uint32_t max_bodies_per_env;
        uint32_t max_joints_per_env;
        uint32_t max_geoms_per_env;
    } ZenoConfig;

    typedef struct {
        uint32_t num_envs;
        uint32_t num_bodies;
        uint32_t num_joints;
        uint32_t num_actuators;
        uint32_t num_sensors;
        uint32_t num_geoms;
        uint32_t obs_dim;
        uint32_t action_dim;
        float timestep;
        uint64_t memory_usage;
        uint64_t gpu_memory_usage;
        bool metal_available;
    } ZenoInfo;

    typedef struct {
        float integrate_ns;
        float collision_broad_ns;
        float collision_narrow_ns;
        float constraint_solve_ns;
        float total_step_ns;
        uint32_t num_contacts;
        uint32_t num_active_constraints;
    } ZenoProfilingData;

    typedef enum {
        ZENO_SUCCESS = 0,
        ZENO_INVALID_HANDLE = -1,
        ZENO_FILE_NOT_FOUND = -2,
        ZENO_PARSE_ERROR = -3,
        ZENO_METAL_ERROR = -4,
        ZENO_OUT_OF_MEMORY = -5,
        ZENO_INVALID_ARGUMENT = -6,
        ZENO_NOT_IMPLEMENTED = -7,
    } ZenoError;

    // World lifecycle
    ZenoWorldHandle zeno_world_create(
        const char* mjcf_path,
        const ZenoConfig* config
    );

    ZenoWorldHandle zeno_world_create_from_string(
        const char* mjcf_string,
        const ZenoConfig* config
    );

    void zeno_world_destroy(ZenoWorldHandle world);

    // Simulation
    int zeno_world_step(
        ZenoWorldHandle world,
        const float* actions,
        uint32_t substeps
    );

    int zeno_world_reset(
        ZenoWorldHandle world,
        const uint8_t* env_mask
    );

    int zeno_world_reset_to_state(
        ZenoWorldHandle world,
        const float* positions,
        const float* quaternions,
        const float* velocities,
        const float* angular_velocities,
        const uint8_t* env_mask
    );

    // State access (zero-copy) - Core
    float* zeno_world_get_observations(ZenoWorldHandle world);
    float* zeno_world_get_rewards(ZenoWorldHandle world);
    uint8_t* zeno_world_get_dones(ZenoWorldHandle world);

    // State access (zero-copy) - Body state
    float* zeno_world_get_body_positions(ZenoWorldHandle world);
    float* zeno_world_get_body_quaternions(ZenoWorldHandle world);
    float* zeno_world_get_body_velocities(ZenoWorldHandle world);
    float* zeno_world_get_body_angular_velocities(ZenoWorldHandle world);
    float* zeno_world_get_body_accelerations(ZenoWorldHandle world);

    // State access (zero-copy) - Joint state
    float* zeno_world_get_joint_positions(ZenoWorldHandle world);
    float* zeno_world_get_joint_velocities(ZenoWorldHandle world);
    float* zeno_world_get_joint_forces(ZenoWorldHandle world);

    // State access (zero-copy) - Contact state
    void* zeno_world_get_contacts(ZenoWorldHandle world);
    uint32_t* zeno_world_get_contact_counts(ZenoWorldHandle world);

    // State access (zero-copy) - Sensor data
    float* zeno_world_get_sensor_data(ZenoWorldHandle world);

    // State mutation (for curriculum learning, domain randomization)
    int zeno_world_set_body_positions(
        ZenoWorldHandle world,
        const float* positions,
        const uint8_t* env_mask
    );

    int zeno_world_set_body_velocities(
        ZenoWorldHandle world,
        const float* velocities,
        const uint8_t* env_mask
    );

    int zeno_world_set_gravity(
        ZenoWorldHandle world,
        float gx, float gy, float gz
    );

    int zeno_world_set_timestep(
        ZenoWorldHandle world,
        float timestep
    );

    // Metadata
    uint32_t zeno_world_num_envs(ZenoWorldHandle world);
    uint32_t zeno_world_num_bodies(ZenoWorldHandle world);
    uint32_t zeno_world_num_joints(ZenoWorldHandle world);
    uint32_t zeno_world_num_sensors(ZenoWorldHandle world);
    uint32_t zeno_world_obs_dim(ZenoWorldHandle world);
    uint32_t zeno_world_action_dim(ZenoWorldHandle world);
    int zeno_world_get_info(ZenoWorldHandle world, ZenoInfo* info);

    // Profiling
    int zeno_world_get_profiling(ZenoWorldHandle world, ZenoProfilingData* data);
    void zeno_world_reset_profiling(ZenoWorldHandle world);

    // Batched operations
    int zeno_world_step_subset(
        ZenoWorldHandle world,
        const float* actions,
        const uint8_t* env_mask,
        uint32_t substeps
    );

    // Utility
    const char* zeno_version();
    bool zeno_metal_available();
    const char* zeno_error_string(int error_code);

    // Memory management
    uint64_t zeno_world_memory_usage(ZenoWorldHandle world);
    void zeno_world_compact_memory(ZenoWorldHandle world);

    // Query limits
    uint32_t zeno_world_max_contacts_per_env(ZenoWorldHandle world);

    // Swarm types
    typedef void* ZenoSwarmHandle;

    typedef struct {
        uint32_t num_agents;
        float communication_range;
        uint32_t max_neighbors;
        uint32_t max_message_bytes;
        uint32_t max_messages_per_step;
        float grid_cell_size;
        uint64_t seed;
        bool enable_physics;
        uint32_t latency_ticks;
        float drop_prob;
        uint32_t max_broadcast_recipients;
        uint32_t max_inbox_per_agent;
        bool strict_determinism;
        uint8_t _pad[2];
    } ZenoSwarmConfig;

    typedef struct {
        float connectivity_ratio;
        float fragmentation_score;
        uint32_t collision_count;
        uint32_t message_count;
        uint32_t bytes_sent;
        uint32_t total_edges;
        float avg_neighbors;
        uint32_t messages_dropped;
        float convergence_time_ms;
        uint32_t near_miss_count;
        float task_success;
        uint32_t _pad;
    } ZenoSwarmMetrics;

    typedef struct {
        uint32_t agent_id;
        uint32_t team_id;
        uint32_t status;
        uint32_t flags;
        float local_state[4];
    } ZenoAgentState;

    typedef struct {
        float score;
        bool complete;
        uint8_t _pad1[3];
        float detail[4];
    } ZenoTaskResult;

    typedef struct {
        uint32_t attack_type;
        float intensity;
        uint32_t target_agents[16];
        uint32_t num_targets;
        uint32_t seed;
        uint32_t _pad[2];
    } ZenoAttackConfig;

    typedef struct {
        uint64_t frame_count;
        uint64_t total_bytes;
        bool recording;
        uint8_t _pad[7];
    } ZenoReplayStats;

    // Swarm lifecycle
    ZenoSwarmHandle zeno_swarm_create(ZenoWorldHandle world, const ZenoSwarmConfig* config);
    void zeno_swarm_destroy(ZenoSwarmHandle swarm);

    // Swarm simulation
    int zeno_swarm_step(ZenoSwarmHandle swarm, ZenoWorldHandle world, float* actions);
    int zeno_swarm_get_metrics(ZenoSwarmHandle swarm, ZenoSwarmMetrics* metrics);
    ZenoAgentState* zeno_swarm_get_agent_states(ZenoSwarmHandle swarm);
    int zeno_swarm_get_neighbor_counts(ZenoSwarmHandle swarm, uint32_t* out, uint32_t count);
    void zeno_swarm_set_body_offset(ZenoSwarmHandle swarm, uint32_t offset);

    // Task evaluation
    int zeno_swarm_evaluate_task(ZenoSwarmHandle swarm, void* world,
                                 uint32_t task_type, const float params[8],
                                 ZenoTaskResult* result);

    // Attack simulation
    int zeno_swarm_apply_attack(ZenoSwarmHandle swarm, const ZenoAttackConfig* config);

    // Graph access (zero-copy)
    uint32_t* zeno_swarm_get_neighbor_index_ptr(ZenoSwarmHandle swarm);
    uint32_t* zeno_swarm_get_neighbor_row_ptr(ZenoSwarmHandle swarm);

    // Replay recording
    int zeno_swarm_start_recording(ZenoSwarmHandle swarm);
    int zeno_swarm_stop_recording(ZenoSwarmHandle swarm);
    int zeno_swarm_get_replay_stats(ZenoSwarmHandle swarm, ZenoReplayStats* stats);
""")


def _find_library() -> str:
    """Find the Zeno shared library."""
    # Possible library names
    if sys.platform == "darwin":
        lib_names = ["libzeno.dylib", "zeno.dylib"]
    else:
        lib_names = ["libzeno.so", "zeno.so"]

    # Search paths
    search_paths = [
        Path(__file__).parent,  # Same directory as this file
        Path(__file__).parent.parent.parent / "zig-out" / "lib",  # Build output
        Path.cwd() / "zig-out" / "lib",  # Current directory build
        Path.home() / ".local" / "lib",  # User install
        Path("/usr/local/lib"),  # System install
    ]

    # Add LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
    lib_path = os.environ.get(
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH", ""
    )
    for p in lib_path.split(":"):
        if p:
            search_paths.append(Path(p))

    # Search for library
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

    raise RuntimeError(
        f"Could not find Zeno library. Searched: {[str(p) for p in search_paths]}\n"
        "Please build the library with: zig build -Doptimize=ReleaseFast"
    )


# Load the library
try:
    _lib_path = _find_library()
    _lib = ffi.dlopen(_lib_path)
except Exception as e:
    _lib = None
    _lib_error = str(e)


def _check_lib():
    """Check if library is loaded."""
    if _lib is None:
        raise RuntimeError(f"Zeno library not loaded: {_lib_error}")


def version() -> str:
    """Get the Zeno library version."""
    _check_lib()
    return ffi.string(_lib.zeno_version()).decode("utf-8")


def is_metal_available() -> bool:
    """Check if Metal GPU acceleration is available."""
    _check_lib()
    return _lib.zeno_metal_available()


class ZeroCopyArray:
    """
    A numpy array view into GPU-shared memory with lifecycle management.

    This class wraps a numpy array that points directly to unified memory
    shared between CPU and GPU. It ensures the underlying world remains
    alive while the array is in use.

    Attributes
    ----------
    array : np.ndarray
        The underlying numpy array (view into shared memory).
    shape : tuple
        Shape of the array.
    dtype : np.dtype
        Data type of the array.

    Notes
    -----
    Modifying this array directly modifies GPU memory. Changes are visible
    to the GPU on the next step() call without any explicit synchronization.
    """

    __slots__ = ('_array', '_world_ref', '_ptr')

    def __init__(self, array: np.ndarray, world: 'ZenoWorld', ptr):
        self._array = array
        self._world_ref = weakref.ref(world)
        self._ptr = ptr

    @property
    def array(self) -> np.ndarray:
        """Get the underlying numpy array."""
        if self._world_ref() is None:
            raise RuntimeError("World has been destroyed")
        return self._array

    def __array__(self, dtype=None):
        """Support numpy array conversion."""
        arr = self.array
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __getattr__(self, name):
        """Delegate attribute access to underlying array."""
        return getattr(self._array, name)

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return f"ZeroCopyArray({self._array.shape}, dtype={self._array.dtype})"

    def copy(self) -> np.ndarray:
        """Create a regular numpy copy (for safe storage)."""
        return self._array.copy()

    def is_valid(self) -> bool:
        """Check if the underlying world is still alive."""
        return self._world_ref() is not None


class ZenoWorld:
    """
    Low-level wrapper around a Zeno world handle.

    This class provides direct access to the simulation state via zero-copy
    numpy arrays backed by unified GPU memory. All state accessors return
    views into shared memory, enabling O(1) access regardless of batch size.

    Parameters
    ----------
    mjcf_path : str, optional
        Path to MJCF file.
    mjcf_string : str, optional
        MJCF XML string.
    num_envs : int
        Number of parallel environments (default: 1).
    timestep : float
        Physics timestep in seconds (default: 0.002).
    contact_iterations : int
        PBD solver iterations for contacts (default: 4).
    max_contacts_per_env : int
        Maximum contacts per environment (default: 64).
    seed : int
        Random seed (default: 42).
    substeps : int
        Physics substeps per step() call (default: 1).
    enable_profiling : bool
        Enable GPU profiling (default: False).

    Examples
    --------
    >>> world = ZenoWorld("ant.xml", num_envs=1024)
    >>> world.reset()
    >>>
    >>> # Zero-copy state access
    >>> positions = world.get_body_positions()  # View into GPU memory
    >>> print(positions.shape)  # (1024, 9, 4)
    >>>
    >>> # Modify state directly (reflected on GPU)
    >>> positions[0, 0, 2] += 0.1  # Lift first body in first env
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
        enable_profiling: bool = False,
        max_bodies_per_env: int = 64,
        max_joints_per_env: int = 64,
        max_geoms_per_env: int = 128,
    ):
        _check_lib()

        if mjcf_path is None and mjcf_string is None:
            raise ValueError("Either mjcf_path or mjcf_string must be provided")

        # Create config
        config = ffi.new("ZenoConfig*")
        config.num_envs = num_envs
        config.timestep = timestep
        config.contact_iterations = contact_iterations
        config.max_contacts_per_env = max_contacts_per_env
        config.seed = seed
        config.substeps = substeps
        config.enable_profiling = enable_profiling
        config.max_bodies_per_env = max_bodies_per_env
        config.max_joints_per_env = max_joints_per_env
        config.max_geoms_per_env = max_geoms_per_env

        # Create world
        if mjcf_path is not None:
            path_bytes = mjcf_path.encode("utf-8")
            self._handle = _lib.zeno_world_create(path_bytes, config)
        else:
            xml_bytes = mjcf_string.encode("utf-8")
            self._handle = _lib.zeno_world_create_from_string(xml_bytes, config)

        if self._handle == ffi.NULL:
            raise RuntimeError("Failed to create Zeno world")

        # Cache dimensions
        self._num_envs = _lib.zeno_world_num_envs(self._handle)
        self._obs_dim = _lib.zeno_world_obs_dim(self._handle)
        self._action_dim = _lib.zeno_world_action_dim(self._handle)

        # Get full info
        info = ffi.new("ZenoInfo*")
        _lib.zeno_world_get_info(self._handle, info)
        self._num_bodies = info.num_bodies
        self._num_joints = info.num_joints
        self._num_sensors = info.num_sensors
        self._num_geoms = info.num_geoms
        self._timestep = info.timestep
        self._memory_usage = info.memory_usage
        self._gpu_memory_usage = info.gpu_memory_usage

        # Cache max contacts per env from the engine
        self._max_contacts_per_env = _lib.zeno_world_max_contacts_per_env(self._handle)

        # Cache for zero-copy arrays (to prevent garbage collection issues)
        self._cached_arrays: Dict[str, ZeroCopyArray] = {}

    def __del__(self):
        if hasattr(self, "_handle") and self._handle != ffi.NULL:
            _lib.zeno_world_destroy(self._handle)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """Action dimension."""
        return self._action_dim

    @property
    def num_bodies(self) -> int:
        """Number of bodies per environment."""
        return self._num_bodies

    @property
    def num_joints(self) -> int:
        """Number of joints per environment."""
        return self._num_joints

    @property
    def num_sensors(self) -> int:
        """Number of sensors per environment."""
        return self._num_sensors

    @property
    def num_geoms(self) -> int:
        """Number of geometries per environment."""
        return self._num_geoms

    @property
    def timestep(self) -> float:
        """Physics timestep in seconds."""
        return self._timestep

    @property
    def memory_usage(self) -> int:
        """Total memory usage in bytes."""
        return self._memory_usage

    @property
    def gpu_memory_usage(self) -> int:
        """GPU memory usage in bytes."""
        return self._gpu_memory_usage

    def _make_zero_copy_array(
        self,
        ptr,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """Create a zero-copy numpy array from a pointer."""
        if ptr == ffi.NULL:
            return np.zeros(shape, dtype=dtype)

        # Calculate buffer size
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buffer = ffi.buffer(ptr, size)
        arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)

        # Optionally cache for lifecycle management
        if cache_key is not None:
            self._cached_arrays[cache_key] = ZeroCopyArray(arr, self, ptr)

        return arr

    def step(self, actions: np.ndarray, substeps: int = 0) -> None:
        """
        Execute physics step with given actions.

        Parameters
        ----------
        actions : np.ndarray
            Actions array of shape (num_envs, action_dim) or flattened.
        substeps : int, optional
            Override number of substeps (0 uses default).
        """
        actions = np.ascontiguousarray(actions, dtype=np.float32).flatten()
        actions_ptr = ffi.cast("float*", actions.ctypes.data)
        result = _lib.zeno_world_step(self._handle, actions_ptr, substeps)
        if result != 0:
            raise RuntimeError(f"Step failed with error code {result}")

    def step_subset(
        self,
        actions: np.ndarray,
        env_mask: np.ndarray,
        substeps: int = 0
    ) -> None:
        """
        Execute physics step only for selected environments.

        This is useful for asynchronous training where different
        environments may need different step counts.

        Parameters
        ----------
        actions : np.ndarray
            Actions for all environments.
        env_mask : np.ndarray
            Boolean mask indicating which environments to step.
        substeps : int, optional
            Override number of substeps.
        """
        actions = np.ascontiguousarray(actions, dtype=np.float32).flatten()
        mask = np.ascontiguousarray(env_mask, dtype=np.uint8)

        actions_ptr = ffi.cast("float*", actions.ctypes.data)
        mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)

        result = _lib.zeno_world_step_subset(
            self._handle, actions_ptr, mask_ptr, substeps
        )
        if result != 0:
            raise RuntimeError(f"Step subset failed with error code {result}")

    def reset(self, mask: Optional[np.ndarray] = None) -> None:
        """
        Reset environments to initial state.

        Parameters
        ----------
        mask : np.ndarray, optional
            Boolean mask indicating which environments to reset.
            If None, all environments are reset.
        """
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)
        else:
            mask_ptr = ffi.NULL
        result = _lib.zeno_world_reset(self._handle, mask_ptr)
        if result != 0:
            raise RuntimeError(f"Reset failed with error code {result}")

    def reset_to_state(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        angular_velocities: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset environments to a specific state.

        Useful for curriculum learning, domain randomization,
        or restoring checkpoints.

        Parameters
        ----------
        positions : np.ndarray
            Body positions of shape (num_envs, num_bodies, 4).
        quaternions : np.ndarray
            Body quaternions of shape (num_envs, num_bodies, 4).
        velocities : np.ndarray, optional
            Linear velocities of shape (num_envs, num_bodies, 4).
        angular_velocities : np.ndarray, optional
            Angular velocities of shape (num_envs, num_bodies, 4).
        mask : np.ndarray, optional
            Boolean mask for which environments to reset.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        quaternions = np.ascontiguousarray(quaternions, dtype=np.float32)

        pos_ptr = ffi.cast("float*", positions.ctypes.data)
        quat_ptr = ffi.cast("float*", quaternions.ctypes.data)

        vel_ptr = ffi.NULL
        angvel_ptr = ffi.NULL

        if velocities is not None:
            velocities = np.ascontiguousarray(velocities, dtype=np.float32)
            vel_ptr = ffi.cast("float*", velocities.ctypes.data)

        if angular_velocities is not None:
            angular_velocities = np.ascontiguousarray(angular_velocities, dtype=np.float32)
            angvel_ptr = ffi.cast("float*", angular_velocities.ctypes.data)

        mask_ptr = ffi.NULL
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)

        result = _lib.zeno_world_reset_to_state(
            self._handle, pos_ptr, quat_ptr, vel_ptr, angvel_ptr, mask_ptr
        )
        if result != 0:
            raise RuntimeError(f"Reset to state failed with error code {result}")

    # ===== Zero-Copy State Accessors =====

    def get_observations(self, zero_copy: bool = False) -> np.ndarray:
        """
        Get observations as numpy array.

        Parameters
        ----------
        zero_copy : bool
            If True, returns a view into GPU memory (faster but unsafe
            to store). If False, returns a copy (safe for storage).

        Returns
        -------
        np.ndarray
            Observations of shape (num_envs, obs_dim).
        """
        ptr = _lib.zeno_world_get_observations(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._obs_dim),
            cache_key="observations" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_rewards(self) -> np.ndarray:
        """Get rewards array of shape (num_envs,)."""
        ptr = _lib.zeno_world_get_rewards(self._handle)
        arr = self._make_zero_copy_array(ptr, (self._num_envs,))
        return arr.copy()

    def get_dones(self) -> np.ndarray:
        """Get done flags array of shape (num_envs,)."""
        ptr = _lib.zeno_world_get_dones(self._handle)
        if ptr == ffi.NULL:
            return np.zeros(self._num_envs, dtype=np.uint8)
        buffer = ffi.buffer(ptr, self._num_envs)
        return np.frombuffer(buffer, dtype=np.uint8).copy()

    def get_body_positions(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get body positions (zero-copy view into GPU memory).

        Returns
        -------
        np.ndarray
            Positions of shape (num_envs, num_bodies, 4).
            Format is (x, y, z, padding).
        """
        ptr = _lib.zeno_world_get_body_positions(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_bodies, 4),
            cache_key="body_positions" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_body_quaternions(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get body quaternions (zero-copy view into GPU memory).

        Returns
        -------
        np.ndarray
            Quaternions of shape (num_envs, num_bodies, 4).
            Format is (x, y, z, w).
        """
        ptr = _lib.zeno_world_get_body_quaternions(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_bodies, 4),
            cache_key="body_quaternions" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_body_velocities(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get body linear velocities (zero-copy view into GPU memory).

        Returns
        -------
        np.ndarray
            Velocities of shape (num_envs, num_bodies, 4).
            Format is (vx, vy, vz, padding).
        """
        ptr = _lib.zeno_world_get_body_velocities(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_bodies, 4),
            cache_key="body_velocities" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_body_angular_velocities(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get body angular velocities (zero-copy view into GPU memory).

        Returns
        -------
        np.ndarray
            Angular velocities of shape (num_envs, num_bodies, 4).
            Format is (wx, wy, wz, padding) in world frame.
        """
        ptr = _lib.zeno_world_get_body_angular_velocities(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_bodies, 4),
            cache_key="body_angular_velocities" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_body_accelerations(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get body linear accelerations.

        Returns
        -------
        np.ndarray
            Accelerations of shape (num_envs, num_bodies, 4).
            Format is (ax, ay, az, padding).
        """
        ptr = _lib.zeno_world_get_body_accelerations(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_bodies, 4),
            cache_key="body_accelerations" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_joint_positions(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get joint positions/angles.

        Returns
        -------
        np.ndarray
            Joint positions of shape (num_envs, num_joints).
        """
        ptr = _lib.zeno_world_get_joint_positions(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_joints),
            cache_key="joint_positions" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_joint_velocities(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get joint velocities.

        Returns
        -------
        np.ndarray
            Joint velocities of shape (num_envs, num_joints).
        """
        ptr = _lib.zeno_world_get_joint_velocities(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._num_joints),
            cache_key="joint_velocities" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    def get_contacts(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get contact data (zero-copy view into GPU memory).

        Returns
        -------
        np.ndarray
            Structured array of shape (num_envs, max_contacts) with fields:
            - position_pen: (4,) float32 (x, y, z, penetration)
            - normal_friction: (4,) float32 (nx, ny, nz, friction)
            - indices: (4,) uint32 (body_a, body_b, geom_a, geom_b)
            - impulses: (4,) float32 (impulse_n, impulse_t1, impulse_t2, restitution)
        """
        ptr = _lib.zeno_world_get_contacts(self._handle)
        max_contacts = self._max_contacts_per_env

        # Define structured dtype matching ContactGPU layout (64 bytes)
        contact_dtype = np.dtype([
            ('position_pen', '4f4'),
            ('normal_friction', '4f4'),
            ('indices', '4u4'),
            ('impulses', '4f4')
        ])

        if ptr == ffi.NULL:
            return np.zeros((self._num_envs, max_contacts), dtype=contact_dtype)

        size_bytes = self._num_envs * max_contacts * 64
        
        buffer = ffi.buffer(ptr, size_bytes)
        arr = np.frombuffer(buffer, dtype=contact_dtype).reshape(self._num_envs, max_contacts)

        if zero_copy:
             self._cached_arrays["contacts"] = ZeroCopyArray(arr, self, ptr)
             return arr
        return arr.copy()

    def get_contact_counts(self) -> np.ndarray:
        """
        Get number of active contacts per environment.

        Returns
        -------
        np.ndarray
            Counts of shape (num_envs,).
        """
        ptr = _lib.zeno_world_get_contact_counts(self._handle)
        if ptr == ffi.NULL:
            return np.zeros(self._num_envs, dtype=np.uint32)
        
        # Buffer size = num_envs * sizeof(uint32)
        size_bytes = self._num_envs * 4
        buffer = ffi.buffer(ptr, size_bytes)
        return np.frombuffer(buffer, dtype=np.uint32).copy()

    def get_sensor_data(self, zero_copy: bool = True) -> np.ndarray:
        """
        Get sensor readings.

        Returns
        -------
        np.ndarray
            Sensor data of shape (num_envs, obs_dim).
            This is the same per-env sensor vector returned by get_observations().
        """
        ptr = _lib.zeno_world_get_sensor_data(self._handle)
        arr = self._make_zero_copy_array(
            ptr,
            (self._num_envs, self._obs_dim),
            cache_key="sensor_data" if zero_copy else None
        )
        return arr if zero_copy else arr.copy()

    # ===== State Mutation =====

    def set_body_positions(
        self,
        positions: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Set body positions for specified environments.

        Parameters
        ----------
        positions : np.ndarray
            Positions of shape (num_envs, num_bodies, 4).
        mask : np.ndarray, optional
            Boolean mask for which environments to modify.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        pos_ptr = ffi.cast("float*", positions.ctypes.data)

        mask_ptr = ffi.NULL
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)

        result = _lib.zeno_world_set_body_positions(self._handle, pos_ptr, mask_ptr)
        if result != 0:
            raise RuntimeError(f"Set body positions failed: {result}")

    def set_body_velocities(
        self,
        velocities: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Set body velocities for specified environments.

        Parameters
        ----------
        velocities : np.ndarray
            Velocities of shape (num_envs, num_bodies, 4).
        mask : np.ndarray, optional
            Boolean mask for which environments to modify.
        """
        velocities = np.ascontiguousarray(velocities, dtype=np.float32)
        vel_ptr = ffi.cast("float*", velocities.ctypes.data)

        mask_ptr = ffi.NULL
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)

        result = _lib.zeno_world_set_body_velocities(self._handle, vel_ptr, mask_ptr)
        if result != 0:
            raise RuntimeError(f"Set body velocities failed: {result}")

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """
        Set gravity vector.

        Parameters
        ----------
        gravity : tuple
            Gravity vector (gx, gy, gz).
        """
        result = _lib.zeno_world_set_gravity(
            self._handle, gravity[0], gravity[1], gravity[2]
        )
        if result != 0:
            raise RuntimeError(f"Set gravity failed: {result}")

    def set_timestep(self, timestep: float) -> None:
        """
        Set physics timestep.

        Parameters
        ----------
        timestep : float
            New timestep in seconds.
        """
        result = _lib.zeno_world_set_timestep(self._handle, timestep)
        if result != 0:
            raise RuntimeError(f"Set timestep failed: {result}")
        self._timestep = timestep

    # ===== Profiling =====

    def get_profiling_data(self) -> Dict[str, Any]:
        """
        Get profiling data from the last step.

        Returns
        -------
        dict
            Dictionary with timing and count information.
        """
        data = ffi.new("ZenoProfilingData*")
        result = _lib.zeno_world_get_profiling(self._handle, data)
        if result != 0:
            return {}

        return {
            "integrate_ms": data.integrate_ns / 1e6,
            "collision_broad_ms": data.collision_broad_ns / 1e6,
            "collision_narrow_ms": data.collision_narrow_ns / 1e6,
            "constraint_solve_ms": data.constraint_solve_ns / 1e6,
            "total_step_ms": data.total_step_ns / 1e6,
            "num_contacts": data.num_contacts,
            "num_active_constraints": data.num_active_constraints,
        }

    def reset_profiling(self) -> None:
        """Reset profiling counters."""
        _lib.zeno_world_reset_profiling(self._handle)

    # ===== State Dictionary (for checkpointing) =====

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete simulation state as a dictionary.

        Useful for checkpointing and replay.

        Returns
        -------
        dict
            Dictionary containing all state arrays (copies).
        """
        return {
            "body_positions": self.get_body_positions(zero_copy=False),
            "body_quaternions": self.get_body_quaternions(zero_copy=False),
            "body_velocities": self.get_body_velocities(zero_copy=False),
            "body_angular_velocities": self.get_body_angular_velocities(zero_copy=False),
            "joint_positions": self.get_joint_positions(zero_copy=False),
            "joint_velocities": self.get_joint_velocities(zero_copy=False),
        }

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        """
        Restore simulation state from a dictionary.

        Parameters
        ----------
        state : dict
            State dictionary from get_state().
        """
        self.reset_to_state(
            positions=state["body_positions"],
            quaternions=state["body_quaternions"],
            velocities=state.get("body_velocities"),
            angular_velocities=state.get("body_angular_velocities"),
        )

    # ===== Info =====

    def get_info(self) -> Dict[str, Any]:
        """
        Get world information.

        Returns
        -------
        dict
            Dictionary with world metadata.
        """
        info = ffi.new("ZenoInfo*")
        _lib.zeno_world_get_info(self._handle, info)

        return {
            "num_envs": info.num_envs,
            "num_bodies": info.num_bodies,
            "num_joints": info.num_joints,
            "num_actuators": info.num_actuators,
            "num_sensors": info.num_sensors,
            "num_geoms": info.num_geoms,
            "obs_dim": info.obs_dim,
            "action_dim": info.action_dim,
            "timestep": info.timestep,
            "memory_usage_mb": info.memory_usage / (1024 * 1024),
            "gpu_memory_usage_mb": info.gpu_memory_usage / (1024 * 1024),
            "metal_available": info.metal_available,
        }
