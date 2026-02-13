"""
Python integration tests for the Zeno physics engine.

Tests cover:
1. FFI smoke test (world lifecycle)
2. Zero-copy memory (shared Metal buffers)
3. Gymnasium API compliance
4. Multi-env batching at scale
5. Reset masking (selective reset)
6. Action clamping (out-of-range actions)
7. All available MJCF environments

Requires: pytest, numpy, zeno (with compiled libzeno)
Optional: gymnasium (for gym API tests)
"""

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MJCF fixture: a minimal pendulum model with one hinge joint and one motor
# ---------------------------------------------------------------------------
PENDULUM_MJCF = """\
<mujoco model="test_pendulum">
    <option timestep="0.02" gravity="0 0 -9.81"/>
    <worldbody>
        <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 1.5">
            <geom type="sphere" size="0.05" rgba="0.3 0.3 0.3 1" mass="0"/>
            <body name="pole" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
                <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.02"
                      mass="1" rgba="0.8 0.2 0.2 1"/>
                <body name="bob" pos="0 0 -1">
                    <geom type="sphere" size="0.1" mass="1"
                          rgba="0.2 0.2 0.8 1"/>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor joint="hinge" ctrlrange="-5 5" gear="1"/>
    </actuator>
    <sensor>
        <jointpos joint="hinge"/>
        <jointvel joint="hinge"/>
    </sensor>
</mujoco>
"""

# Path to MJCF asset directory
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# All shipped environment XML files
ALL_MJCF_FILES = sorted(ASSETS_DIR.glob("*.xml")) if ASSETS_DIR.exists() else []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_lib():
    """Skip test if the native library is not available."""
    from zeno._ffi import _lib
    if _lib is None:
        pytest.skip("Zeno native library not available (build with zig build)")


def _has_gymnasium():
    try:
        import gymnasium  # noqa: F401
        return True
    except ImportError:
        return False


# ===================================================================
# 1. FFI Smoke Test
# ===================================================================

class TestFFISmokeTest:
    """Basic lifecycle: create from MJCF string, step, destroy."""

    def test_create_world_from_string(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        assert world.num_envs == 1
        assert world.obs_dim > 0
        assert world.action_dim > 0
        del world  # triggers __del__ -> zeno_world_destroy

    def test_create_requires_mjcf(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        with pytest.raises(ValueError, match="mjcf_path or mjcf_string"):
            ZenoWorld()

    def test_step_and_observations(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        world.step(actions)

        obs = world.get_observations()
        assert obs.shape == (1, world.obs_dim)
        assert obs.dtype == np.float32

    def test_multiple_steps(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        for _ in range(100):
            world.step(actions)

        obs = world.get_observations()
        assert obs.shape == (1, world.obs_dim)
        assert np.all(np.isfinite(obs)), "Observations contain NaN/Inf after 100 steps"

    def test_world_info(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=4)
        info = world.get_info()

        assert info["num_envs"] == 4
        assert info["num_bodies"] > 0
        assert info["num_joints"] > 0
        assert info["action_dim"] > 0
        assert info["obs_dim"] > 0
        assert info["timestep"] > 0

    def test_version_and_metal(self):
        _skip_if_no_lib()
        from zeno._ffi import version, is_metal_available

        v = version()
        assert isinstance(v, str)
        assert len(v) > 0

        metal = is_metal_available()
        assert isinstance(metal, bool)


# ===================================================================
# 2. Zero-Copy Memory
# ===================================================================

class TestZeroCopyMemory:
    """Verify numpy arrays point to shared Metal buffers (not copies)."""

    def test_body_positions_are_zero_copy(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=2)
        world.reset()

        pos1 = world.get_body_positions(zero_copy=True)
        pos2 = world.get_body_positions(zero_copy=True)

        # Both should reference the same underlying data
        assert pos1.ctypes.data == pos2.ctypes.data, (
            "zero_copy=True should return views into the same buffer"
        )

    def test_zero_copy_vs_copy(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=2)
        world.reset()

        pos_view = world.get_body_positions(zero_copy=True)
        pos_copy = world.get_body_positions(zero_copy=False)

        # Copy should have a different data pointer
        assert pos_view.ctypes.data != pos_copy.ctypes.data, (
            "zero_copy=False should return a copy with a different buffer"
        )
        # But the values should be equal
        np.testing.assert_array_equal(pos_view, pos_copy)

    def test_zero_copy_reflects_simulation_changes(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        pos = world.get_body_positions(zero_copy=True)
        initial_snapshot = pos.copy()

        # Apply a nonzero action to create movement
        actions = np.ones((1, world.action_dim), dtype=np.float32)
        for _ in range(10):
            world.step(actions)

        # The zero-copy view should reflect the updated positions automatically
        # (no need to call get_body_positions again)
        assert not np.array_equal(pos, initial_snapshot), (
            "Zero-copy array should reflect simulation state changes in-place"
        )

    def test_observation_copy_by_default(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        # Default get_observations returns a copy
        obs1 = world.get_observations(zero_copy=False)
        obs2 = world.get_observations(zero_copy=False)
        assert obs1.ctypes.data != obs2.ctypes.data, (
            "Default observations should be independent copies"
        )

    def test_body_state_shapes(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 4
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()
        nb = world.num_bodies

        pos = world.get_body_positions()
        assert pos.shape == (num_envs, nb, 4)
        assert pos.dtype == np.float32

        quat = world.get_body_quaternions()
        assert quat.shape == (num_envs, nb, 4)

        vel = world.get_body_velocities()
        assert vel.shape == (num_envs, nb, 4)

        angvel = world.get_body_angular_velocities()
        assert angvel.shape == (num_envs, nb, 4)

    def test_joint_state_shapes(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 4
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()
        nj = world.num_joints

        jpos = world.get_joint_positions()
        assert jpos.shape == (num_envs, nj)
        assert jpos.dtype == np.float32

        jvel = world.get_joint_velocities()
        assert jvel.shape == (num_envs, nj)


# ===================================================================
# 3. Gymnasium API Compliance
# ===================================================================

@pytest.mark.skipif(not _has_gymnasium(), reason="gymnasium not installed")
class TestGymnasiumAPI:
    """Reset/step cycle, observation shapes, done flags."""

    def test_single_env_reset_step(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv

        env = ZenoGymnasiumEnv(mjcf_path=str(ASSETS_DIR / "pendulum.xml"))

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_observation_in_space(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv

        env = ZenoGymnasiumEnv(mjcf_path=str(ASSETS_DIR / "pendulum.xml"))
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            f"Observation {obs} not in observation_space"
        )
        env.close()

    def test_action_space_sampling(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv

        env = ZenoGymnasiumEnv(mjcf_path=str(ASSETS_DIR / "pendulum.xml"))
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
        env.close()

    def test_vectorized_env_reset_step(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoVectorEnv

        num_envs = 8
        env = ZenoVectorEnv(
            mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
            num_envs=num_envs,
        )

        obs, info = env.reset()
        assert obs.shape[0] == num_envs
        assert obs.dtype == np.float32

        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs.shape[0] == num_envs
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

        env.close()

    def test_vectorized_env_5_step_return_types(self):
        """Gymnasium v0.29+ vector envs return (obs, rew, term, trunc, info)."""
        _skip_if_no_lib()
        from zeno.gym import ZenoVectorEnv

        env = ZenoVectorEnv(
            mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
            num_envs=4,
        )
        env.reset()
        result = env.step(env.action_space.sample())

        assert len(result) == 5, "VectorEnv.step must return 5 values"
        obs, rewards, terminated, truncated, info = result
        assert terminated.dtype == bool
        assert truncated.dtype == bool
        assert rewards.dtype == np.float32

        env.close()

    def test_check_env_utility(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv, check_env

        env = ZenoGymnasiumEnv(mjcf_path=str(ASSETS_DIR / "pendulum.xml"))
        assert check_env(env) is True
        env.close()

    def test_context_manager(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        with ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=1) as env:
            obs = env.reset()
            assert obs is not None


# ===================================================================
# 4. Multi-Env Batching
# ===================================================================

class TestMultiEnvBatching:
    """Create 1024 envs, verify shapes are (1024, ...)."""

    def test_1024_envs_shapes(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 1024
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()

        obs = world.get_observations()
        assert obs.shape[0] == num_envs
        assert obs.shape == (num_envs, world.obs_dim)

        rewards = world.get_rewards()
        assert rewards.shape == (num_envs,)

        dones = world.get_dones()
        assert dones.shape == (num_envs,)

        positions = world.get_body_positions()
        assert positions.shape[0] == num_envs
        assert positions.shape == (num_envs, world.num_bodies, 4)

    def test_1024_envs_step(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 1024
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()

        actions = np.random.uniform(
            -1, 1, (num_envs, world.action_dim)
        ).astype(np.float32)

        # Step should not raise
        world.step(actions)

        obs = world.get_observations()
        assert obs.shape == (num_envs, world.obs_dim)
        assert np.all(np.isfinite(obs))

    def test_batched_env_high_level(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        num_envs = 1024
        env = ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        obs = env.reset()
        assert obs.shape == (num_envs, env.observation_dim)

        actions = np.random.uniform(
            -1, 1, (num_envs, env.action_dim)
        ).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)
        assert obs.shape == (num_envs, env.observation_dim)
        assert rewards.shape == (num_envs,)
        assert dones.shape == (num_envs,)


# ===================================================================
# 5. Reset Masking
# ===================================================================

class TestResetMasking:
    """Selective reset with mask array."""

    def test_selective_reset(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 8
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()

        # Step all envs with a nonzero action to change state
        actions = np.ones((num_envs, world.action_dim), dtype=np.float32)
        for _ in range(20):
            world.step(actions)

        state_before = world.get_body_positions(zero_copy=False)

        # Reset only envs 0, 2, 4
        mask = np.zeros(num_envs, dtype=np.uint8)
        mask[[0, 2, 4]] = 1
        world.reset(mask=mask)

        state_after = world.get_body_positions(zero_copy=False)

        # Reset envs should be back to initial state (different from before)
        for i in [0, 2, 4]:
            assert not np.array_equal(state_before[i], state_after[i]), (
                f"Env {i} should have been reset"
            )

        # Non-reset envs should be unchanged
        for i in [1, 3, 5, 6, 7]:
            np.testing.assert_array_equal(
                state_before[i], state_after[i],
                err_msg=f"Env {i} should NOT have been reset",
            )

    def test_reset_all_with_none_mask(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 4
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()

        # Step to change state
        actions = np.ones((num_envs, world.action_dim), dtype=np.float32)
        for _ in range(10):
            world.step(actions)

        state_before = world.get_body_positions(zero_copy=False)

        # Reset all (mask=None)
        world.reset(mask=None)
        state_after = world.get_body_positions(zero_copy=False)

        # All envs should change back to initial
        for i in range(num_envs):
            assert not np.array_equal(state_before[i], state_after[i]), (
                f"Env {i} should have been reset"
            )

    def test_high_level_env_reset_with_mask(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        num_envs = 8
        env = ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        env.reset()

        # Step to change state
        actions = np.ones((num_envs, env.action_dim), dtype=np.float32)
        for _ in range(20):
            env.step(actions)

        # Only reset first half
        mask = np.zeros(num_envs, dtype=np.uint8)
        mask[:4] = 1
        obs = env.reset(mask=mask)
        assert obs.shape == (num_envs, env.observation_dim)


# ===================================================================
# 6. Action Clamping
# ===================================================================

class TestActionClamping:
    """Verify out-of-range actions are handled gracefully."""

    def test_extreme_actions_no_crash(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        # Very large actions -- should not crash or produce NaN
        actions = np.full(
            (1, world.action_dim), 1e6, dtype=np.float32
        )
        world.step(actions)

        obs = world.get_observations()
        assert np.all(np.isfinite(obs)), "Extreme actions produced NaN/Inf"

    def test_negative_extreme_actions(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        actions = np.full(
            (1, world.action_dim), -1e6, dtype=np.float32
        )
        world.step(actions)
        obs = world.get_observations()
        assert np.all(np.isfinite(obs)), "Negative extreme actions produced NaN/Inf"

    def test_zero_actions(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        for _ in range(50):
            world.step(actions)

        obs = world.get_observations()
        assert np.all(np.isfinite(obs)), "Zero actions produced NaN/Inf after 50 steps"

    def test_nan_actions_handled(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        # NaN actions -- the engine should either clamp or error, not segfault
        actions = np.full(
            (1, world.action_dim), np.nan, dtype=np.float32
        )
        try:
            world.step(actions)
            # If it doesn't raise, at least check we can still query state
            obs = world.get_observations()
            assert obs is not None
        except RuntimeError:
            # Acceptable: engine rejected NaN input
            pass

    def test_action_dtype_coercion(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        # Pass float64 actions -- should be coerced to float32 internally
        actions = np.zeros((1, world.action_dim), dtype=np.float64)
        world.step(actions)

        obs = world.get_observations()
        assert obs.dtype == np.float32


# ===================================================================
# 7. All Environments
# ===================================================================

class TestAllEnvironments:
    """Load each available MJCF model, run 10 steps without crash."""

    @pytest.mark.parametrize(
        "mjcf_path",
        ALL_MJCF_FILES,
        ids=[p.stem for p in ALL_MJCF_FILES],
    )
    def test_load_and_step(self, mjcf_path):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_path=str(mjcf_path), num_envs=2)
        world.reset()

        actions = np.zeros((2, world.action_dim), dtype=np.float32)
        for step_idx in range(10):
            world.step(actions)

        obs = world.get_observations()
        assert obs.shape == (2, world.obs_dim), (
            f"{mjcf_path.name}: unexpected obs shape {obs.shape}"
        )
        assert np.all(np.isfinite(obs)), (
            f"{mjcf_path.name}: NaN/Inf in observations after 10 steps"
        )

    @pytest.mark.parametrize(
        "mjcf_path",
        ALL_MJCF_FILES,
        ids=[p.stem for p in ALL_MJCF_FILES],
    )
    def test_env_metadata(self, mjcf_path):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_path=str(mjcf_path), num_envs=1)
        info = world.get_info()

        assert info["num_envs"] == 1
        assert info["num_bodies"] >= 1
        assert info["action_dim"] >= 1
        assert info["obs_dim"] >= 1
        assert info["timestep"] > 0


# ===================================================================
# 8. State Checkpointing
# ===================================================================

class TestStateCheckpointing:
    """get_state / set_state round-trip."""

    def test_state_roundtrip(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=2)
        world.reset()

        # Step a few times to get non-trivial state
        actions = np.ones((2, world.action_dim), dtype=np.float32) * 0.5
        for _ in range(20):
            world.step(actions)

        # Save state
        state = world.get_state()
        assert "body_positions" in state
        assert "body_quaternions" in state
        assert state["body_positions"].shape[0] == 2

        # Step further to diverge
        for _ in range(20):
            world.step(actions)
        diverged_pos = world.get_body_positions(zero_copy=False)

        # Restore state
        world.set_state(state)
        restored_pos = world.get_body_positions(zero_copy=False)

        np.testing.assert_array_almost_equal(
            restored_pos, state["body_positions"],
            err_msg="Restored state should match saved state",
        )


# ===================================================================
# 9. Gravity and Timestep Mutation
# ===================================================================

class TestWorldMutation:
    """Verify runtime parameter changes."""

    def test_set_gravity(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        # No gravity
        world.set_gravity((0.0, 0.0, 0.0))

        pos_before = world.get_body_positions(zero_copy=False)
        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        for _ in range(50):
            world.step(actions)
        pos_after = world.get_body_positions(zero_copy=False)

        # With zero gravity and zero actions, vertical motion should be minimal
        z_displacement = np.abs(pos_after[0, :, 2] - pos_before[0, :, 2])
        assert np.all(z_displacement < 0.5), (
            "With zero gravity, bodies should not fall significantly"
        )

    def test_set_timestep(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.reset()

        world.set_timestep(0.001)
        assert world.timestep == pytest.approx(0.001, rel=1e-5)

        # Should still be able to step
        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        world.step(actions)
        obs = world.get_observations()
        assert np.all(np.isfinite(obs))


# ===================================================================
# 10. Profiling API
# ===================================================================

class TestProfilingAPI:
    """Verify profiling data access."""

    def test_profiling_data_keys(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(
            mjcf_string=PENDULUM_MJCF,
            num_envs=1,
            enable_profiling=True,
        )
        world.reset()

        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        world.step(actions)

        data = world.get_profiling_data()
        # Profiling may or may not return data depending on build
        if data:
            assert "total_step_ms" in data
            assert data["total_step_ms"] >= 0

    def test_reset_profiling(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(
            mjcf_string=PENDULUM_MJCF,
            num_envs=1,
            enable_profiling=True,
        )
        world.reset()

        actions = np.zeros((1, world.action_dim), dtype=np.float32)
        world.step(actions)

        # Should not raise
        world.reset_profiling()
