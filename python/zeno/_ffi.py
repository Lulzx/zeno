"""
CFFI bindings for the Zeno physics engine.
Provides zero-copy access to GPU buffers via unified memory.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cffi
import numpy as np

# Initialize FFI
ffi = cffi.FFI()

# C declarations
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
    } ZenoConfig;

    typedef struct {
        uint32_t num_envs;
        uint32_t num_bodies;
        uint32_t num_joints;
        uint32_t num_actuators;
        uint32_t obs_dim;
        uint32_t action_dim;
        float timestep;
        uint64_t memory_usage;
    } ZenoInfo;

    typedef enum {
        ZENO_SUCCESS = 0,
        ZENO_INVALID_HANDLE = -1,
        ZENO_FILE_NOT_FOUND = -2,
        ZENO_PARSE_ERROR = -3,
        ZENO_METAL_ERROR = -4,
        ZENO_OUT_OF_MEMORY = -5,
        ZENO_INVALID_ARGUMENT = -6,
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

    // State access (zero-copy)
    float* zeno_world_get_observations(ZenoWorldHandle world);
    float* zeno_world_get_rewards(ZenoWorldHandle world);
    uint8_t* zeno_world_get_dones(ZenoWorldHandle world);
    float* zeno_world_get_body_positions(ZenoWorldHandle world);
    float* zeno_world_get_body_quaternions(ZenoWorldHandle world);

    // Metadata
    uint32_t zeno_world_num_envs(ZenoWorldHandle world);
    uint32_t zeno_world_obs_dim(ZenoWorldHandle world);
    uint32_t zeno_world_action_dim(ZenoWorldHandle world);
    int zeno_world_get_info(ZenoWorldHandle world, ZenoInfo* info);

    // Utility
    const char* zeno_version();
    bool zeno_metal_available();
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


class ZenoWorld:
    """Low-level wrapper around a Zeno world handle."""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        mjcf_string: Optional[str] = None,
        num_envs: int = 1,
        timestep: float = 0.002,
        contact_iterations: int = 4,
        max_contacts_per_env: int = 64,
        seed: int = 42,
        substeps: int = 1,
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
        self._timestep = info.timestep

    def __del__(self):
        if hasattr(self, "_handle") and self._handle != ffi.NULL:
            _lib.zeno_world_destroy(self._handle)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def timestep(self) -> float:
        return self._timestep

    def step(self, actions: np.ndarray, substeps: int = 0) -> None:
        """Execute physics step with given actions."""
        actions = np.ascontiguousarray(actions, dtype=np.float32).flatten()
        actions_ptr = ffi.cast("float*", actions.ctypes.data)
        result = _lib.zeno_world_step(self._handle, actions_ptr, substeps)
        if result != 0:
            raise RuntimeError(f"Step failed with error code {result}")

    def reset(self, mask: Optional[np.ndarray] = None) -> None:
        """Reset environments. If mask is None, reset all."""
        if mask is not None:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
            mask_ptr = ffi.cast("uint8_t*", mask.ctypes.data)
        else:
            mask_ptr = ffi.NULL
        result = _lib.zeno_world_reset(self._handle, mask_ptr)
        if result != 0:
            raise RuntimeError(f"Reset failed with error code {result}")

    def get_observations(self) -> np.ndarray:
        """Get observations as a zero-copy numpy array."""
        ptr = _lib.zeno_world_get_observations(self._handle)
        if ptr == ffi.NULL:
            return np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)

        # Create zero-copy view
        buffer = ffi.buffer(ptr, self._num_envs * self._obs_dim * 4)
        arr = np.frombuffer(buffer, dtype=np.float32)
        return arr.reshape(self._num_envs, self._obs_dim)

    def get_rewards(self) -> np.ndarray:
        """Get rewards as a zero-copy numpy array."""
        ptr = _lib.zeno_world_get_rewards(self._handle)
        if ptr == ffi.NULL:
            return np.zeros(self._num_envs, dtype=np.float32)

        buffer = ffi.buffer(ptr, self._num_envs * 4)
        return np.frombuffer(buffer, dtype=np.float32).copy()

    def get_dones(self) -> np.ndarray:
        """Get done flags as a numpy array."""
        ptr = _lib.zeno_world_get_dones(self._handle)
        if ptr == ffi.NULL:
            return np.zeros(self._num_envs, dtype=np.uint8)

        buffer = ffi.buffer(ptr, self._num_envs)
        return np.frombuffer(buffer, dtype=np.uint8).copy()

    def get_body_positions(self) -> np.ndarray:
        """Get body positions for visualization."""
        ptr = _lib.zeno_world_get_body_positions(self._handle)
        if ptr == ffi.NULL:
            return np.zeros((self._num_envs, self._num_bodies, 4), dtype=np.float32)

        size = self._num_envs * self._num_bodies * 4 * 4
        buffer = ffi.buffer(ptr, size)
        arr = np.frombuffer(buffer, dtype=np.float32)
        return arr.reshape(self._num_envs, self._num_bodies, 4)

    def get_body_quaternions(self) -> np.ndarray:
        """Get body quaternions for visualization."""
        ptr = _lib.zeno_world_get_body_quaternions(self._handle)
        if ptr == ffi.NULL:
            return np.zeros((self._num_envs, self._num_bodies, 4), dtype=np.float32)

        size = self._num_envs * self._num_bodies * 4 * 4
        buffer = ffi.buffer(ptr, size)
        arr = np.frombuffer(buffer, dtype=np.float32)
        return arr.reshape(self._num_envs, self._num_bodies, 4)
