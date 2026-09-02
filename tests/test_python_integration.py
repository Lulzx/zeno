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

    def test_world_close_is_idempotent(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        world.close()
        world.close()

    def test_free_body_matches_semi_implicit_ballistic_solution(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        dt = 0.002
        steps = 100
        xml = f"""<mujoco>
        <option timestep="{dt}" gravity="0 0 -9.81"/>
        <worldbody><body pos="0 0 10"><freejoint/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
        </body></worldbody></mujoco>"""
        world = ZenoWorld(mjcf_string=xml, timestep=dt)
        try:
            world.reset()
            actions = np.zeros((1, world.action_dim), dtype=np.float32)
            for _ in range(steps):
                world.step(actions)
            actual_z = float(world.get_body_positions()[0, 1, 2])
            expected_z = 10.0 - 9.81 * dt * dt * steps * (steps + 1) / 2
            assert actual_z == pytest.approx(expected_z, abs=1e-3)
        finally:
            world.close()

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

    def test_rgb_array_render_contract(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv

        env = ZenoGymnasiumEnv(
            mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
            render_mode="rgb_array",
        )
        try:
            env.reset()
            frame = env.render()
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8
        finally:
            env.close()

    def test_unsupported_human_render_rejected(self):
        _skip_if_no_lib()
        from zeno.gym import ZenoGymnasiumEnv

        with pytest.raises(ValueError, match="only 'rgb_array'"):
            ZenoGymnasiumEnv(
                mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
                render_mode="human",
            )

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
        assert env.single_observation_space.shape == obs.shape[1:]

        actions = env.action_space.sample()
        assert actions.shape == (num_envs, env.single_action_space.shape[0])
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs.shape[0] == num_envs
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

        env.close()
        assert getattr(env, "closed", True) is True

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

    def test_stable_baselines3_ppo_smoke(self):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import PPO
        from zeno.gym import make_sb3_env

        env = make_sb3_env("pendulum", num_envs=2)
        try:
            model = PPO("MlpPolicy", env, n_steps=2, batch_size=4, verbose=0)
            model.learn(total_timesteps=4)
        finally:
            env.close()

    def test_named_ant_vector_uses_explicit_zeno_metal_task_preset(self):
        _skip_if_no_lib()
        if not _has_gymnasium():
            pytest.skip("gymnasium is not installed")
        from zeno.gym import make_vec

        env = make_vec("ant", num_envs=4, max_episode_steps=2)
        try:
            env.reset()
            actions = np.zeros((4, env.single_action_space.shape[0]), dtype=np.float32)
            _, rewards, terminated, truncated, _ = env.step(actions)
            assert np.all(np.isfinite(rewards))
            assert np.all(rewards > 0.5)
            assert not terminated.any()
            assert not truncated.any()

            _, _, terminated, truncated, _ = env.step(actions)
            # The wrapper owns the horizon: it must be truncation, not native
            # task termination from the GPU done buffer.
            assert not terminated.any()
            assert truncated.all()
        finally:
            env.close()

    def test_registered_ant_resolves_asset_and_uses_task_preset(self):
        _skip_if_no_lib()
        gymnasium = pytest.importorskip("gymnasium")
        import zeno.gym  # noqa: F401 - triggers registration

        env = gymnasium.make("Zeno/Ant-v0")
        try:
            observation, _ = env.reset()
            observation, reward, terminated, truncated, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            assert observation.shape == env.observation_space.shape
            assert np.isfinite(reward)
            assert reward > 0.5
            assert not terminated
            assert not truncated
        finally:
            env.close()

    def test_model_path_remains_raw_physics_without_implicit_task(self):
        _skip_if_no_lib()
        if not _has_gymnasium():
            pytest.skip("gymnasium is not installed")
        from zeno.gym import make_vec

        env = make_vec(str(ASSETS_DIR / "ant.xml"), num_envs=2)
        try:
            env.reset()
            actions = np.zeros((2, env.single_action_space.shape[0]), dtype=np.float32)
            _, rewards, terminated, _, _ = env.step(actions)
            np.testing.assert_array_equal(rewards, np.zeros(2, dtype=np.float32))
            assert not terminated.any()
        finally:
            env.close()

    def test_explicit_task_config_overrides_named_preset(self):
        _skip_if_no_lib()
        if not _has_gymnasium():
            pytest.skip("gymnasium is not installed")
        from zeno.gym import make_vec

        env = make_vec(
            "ant",
            num_envs=2,
            task_config={
                "root_body": 1,
                "healthy_bonus": 3.0,
                "healthy_z_min": -100.0,
                "healthy_z_max": 100.0,
            },
        )
        try:
            env.reset()
            actions = np.zeros((2, env.single_action_space.shape[0]), dtype=np.float32)
            _, rewards, _, _, _ = env.step(actions)
            np.testing.assert_allclose(rewards, 3.0, atol=1e-5)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "model",
        ["ant", "humanoid", "cheetah", "hopper", "walker", "swimmer"],
    )
    def test_all_supported_named_task_presets_run_on_metal(self, model):
        _skip_if_no_lib()
        if not _has_gymnasium():
            pytest.skip("gymnasium is not installed")
        from zeno.gym import make_vec

        env = make_vec(model, num_envs=2)
        try:
            env.reset()
            actions = np.full(
                (2, env.single_action_space.shape[0]), 0.2, dtype=np.float32
            )
            _, rewards, _, _, _ = env.step(actions)
            assert np.all(np.isfinite(rewards))
            assert np.any(rewards != 0.0)
        finally:
            env.close()


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
# 5.5 Subset Stepping
# ===================================================================

class TestSubsetStepping:
    """Step only selected environments."""

    def test_step_subset_only_updates_masked_envs(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        num_envs = 4
        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=num_envs)
        world.reset()

        state_before = world.get_body_positions(zero_copy=False)
        actions = np.ones((num_envs, world.action_dim), dtype=np.float32)
        mask = np.array([1, 0, 1, 0], dtype=np.uint8)

        for _ in range(20):
            world.step_subset(actions, mask)

        state_after = world.get_body_positions(zero_copy=False)

        # Masked envs should evolve.
        assert not np.array_equal(state_before[0], state_after[0])
        assert not np.array_equal(state_before[2], state_after[2])

        # Unmasked envs should remain exactly unchanged.
        np.testing.assert_array_equal(state_before[1], state_after[1])
        np.testing.assert_array_equal(state_before[3], state_after[3])


# ===================================================================
# 6. Reset Masking
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
# 7. Action Clamping
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
# 8. All Environments
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
        # Passive scenes (no <actuator> block) legitimately have action_dim 0.
        has_actuators = "<actuator" in mjcf_path.read_text()
        if has_actuators:
            assert info["action_dim"] >= 1
        else:
            assert info["action_dim"] == 0
        assert info["obs_dim"] >= 1
        assert info["timestep"] > 0


# ===================================================================
# 9. State Checkpointing
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
# 10. Gravity and Timestep Mutation
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
# 11. Acceleration and Sensor API
# ===================================================================

class TestAdditionalStateAPI:
    """Validate acceleration and sensor data accessors."""

    def test_body_accelerations_shape(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=2)
        world.reset()

        actions = np.ones((2, world.action_dim), dtype=np.float32)
        world.step(actions)

        acc = world.get_body_accelerations()
        assert acc.shape == (2, world.num_bodies, 4)
        assert np.all(np.isfinite(acc))

    def test_low_level_vector_lengths_are_validated_before_ffi(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=2)
        try:
            short_actions = np.zeros((1, world.action_dim), dtype=np.float32)
            actions = np.zeros((2, world.action_dim), dtype=np.float32)
            with pytest.raises(ValueError, match="actions must contain"):
                world.step(short_actions)
            with pytest.raises(ValueError, match="env_mask must contain"):
                world.step_subset(actions, np.array([1], dtype=np.uint8))
            with pytest.raises(ValueError, match="mask must contain"):
                world.reset(np.array([1], dtype=np.uint8))
        finally:
            world.close()

    def test_sensor_data_shape_matches_observation_dim(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=3)
        world.reset()

        sensor_data = world.get_sensor_data()
        assert sensor_data.shape == (3, world.obs_dim)

    def test_step_outputs_can_remain_zero_copy(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        env = ZenoEnv(
            mjcf_string=PENDULUM_MJCF,
            num_envs=8,
            zero_copy_outputs=True,
        )
        try:
            reset_obs = env.reset()
            obs_view = env._world.get_observations(zero_copy=True)
            assert np.shares_memory(reset_obs, obs_view)

            actions = np.zeros(env.action_shape, dtype=np.float32)
            obs, rewards, dones, _ = env.step(actions)
            assert np.shares_memory(obs, env._world.get_observations(zero_copy=True))
            assert np.shares_memory(rewards, env._world.get_rewards(zero_copy=True))
            assert np.shares_memory(dones, env._world.get_dones(zero_copy=True))
            assert obs.dtype == np.float32
            assert rewards.dtype == np.float32
            assert dones.dtype == np.bool_
        finally:
            env.close()


# ===================================================================
# 5.6 GPU Task Outputs
# ===================================================================

class TestGPUTaskOutputs:
    """Reward, termination, and episode clocks are evaluated on Metal."""

    def test_low_level_reward_horizon_and_masked_reset(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=3)
        try:
            world.configure_task(
                root_body=0,
                max_episode_steps=2,
                control_cost_weight=0.25,
                healthy_bonus=1.5,
                healthy_z_min=-100.0,
                healthy_z_max=100.0,
                terminate_when_unhealthy=True,
            )
            actions = np.full((3, world.action_dim), 0.5, dtype=np.float32)

            world.step(actions)
            np.testing.assert_allclose(world.get_rewards(), 1.4375, atol=1e-6)
            assert not world.get_dones().any()

            world.step(actions)
            assert world.get_dones().all()

            world.reset(np.array([0, 1, 0], dtype=np.uint8))
            np.testing.assert_array_equal(world.get_dones(), [1, 0, 1])
            np.testing.assert_allclose(world.get_rewards(), [1.4375, 0.0, 1.4375])
        finally:
            world.close()

    def test_high_level_task_config_and_validation(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        env = ZenoEnv(
            mjcf_string=PENDULUM_MJCF,
            num_envs=2,
            zero_copy_outputs=True,
            task_config={
                "root_body": 0,
                "max_episode_steps": 1,
                "healthy_bonus": 2.0,
                "healthy_z_min": -100.0,
                "healthy_z_max": 100.0,
            },
        )
        try:
            _, rewards, dones, _ = env.step(
                np.zeros(env.action_shape, dtype=np.float32)
            )
            np.testing.assert_allclose(rewards, 2.0, atol=1e-6)
            assert dones.all()
            with pytest.raises(ValueError, match="Invalid task configuration"):
                env.configure_task(
                    root_body=env._world.num_bodies,
                    healthy_z_min=0.0,
                    healthy_z_max=1.0,
                )
        finally:
            env.close()


# ===================================================================
# 5.7 True Async Submission
# ===================================================================

class TestAsyncSubmission:
    """Async APIs commit native Metal work before wait."""

    def test_low_level_pending_lifecycle_and_read_guard(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=8)
        try:
            actions = np.zeros((8, world.action_dim), dtype=np.float32)
            world.step_async(actions)
            assert world.step_pending
            with pytest.raises(RuntimeError, match="not synchronized"):
                world.get_observations()
            with pytest.raises(RuntimeError, match="error code -8"):
                world.step_async(actions)
            world.step_wait()
            assert not world.step_pending
            assert np.all(np.isfinite(world.get_observations()))
            with pytest.raises(RuntimeError, match="error code -9"):
                world.step_wait()
        finally:
            world.close()

    def test_writable_shared_actions_skip_staging_copy(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        copied = ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=8)
        shared = ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=8)
        try:
            actions = np.linspace(
                0.04, 0.32, num=8 * copied.action_dim, dtype=np.float32
            ).reshape(copied.action_shape)
            shared_actions = shared.get_action_buffer()
            assert shared_actions.shape == shared.action_shape
            assert shared_actions.flags.writeable
            shared_actions[:] = actions

            copied_obs, _, _, _ = copied.step(actions)
            shared.step_current_actions_async()
            assert shared._world.step_pending
            with pytest.raises(RuntimeError, match="not synchronized"):
                shared.get_action_buffer()
            shared_obs, _, _, _ = shared.step_wait()
            np.testing.assert_allclose(shared_obs, copied_obs, atol=1e-5)
            np.testing.assert_allclose(shared_actions, actions, atol=0)
        finally:
            copied.close()
            shared.close()

    def test_high_level_and_vector_step_async_submit_immediately(self):
        _skip_if_no_lib()
        from zeno.env import ZenoEnv

        env = ZenoEnv(mjcf_string=PENDULUM_MJCF, num_envs=4)
        try:
            actions = np.zeros(env.action_shape, dtype=np.float32)
            env.step_async(actions)
            assert env._world.step_pending
            obs, rewards, dones, _ = env.step_wait()
            assert obs.shape == (4, env.observation_dim)
            assert rewards.shape == dones.shape == (4,)
            assert not env._world.step_pending
        finally:
            env.close()

        if _has_gymnasium():
            from zeno.gym.registration import ZenoVectorEnv

            vector_env = ZenoVectorEnv(
                mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
                num_envs=4,
            )
            try:
                actions = np.zeros((4, vector_env._env.action_dim), dtype=np.float32)
                vector_env.step_async(actions)
                assert vector_env._env._world.step_pending
                result = vector_env.step_wait()
                assert result[0].shape[0] == 4
                assert not vector_env._env._world.step_pending
            finally:
                vector_env.close()

    def test_gym_next_step_autoreset_is_fused_into_native_submission(self):
        _skip_if_no_lib()
        if not _has_gymnasium():
            pytest.skip("gymnasium is not installed")
        from unittest.mock import patch
        from zeno.gym.registration import ZenoVectorEnv

        env = ZenoVectorEnv(
            mjcf_path=str(ASSETS_DIR / "pendulum.xml"),
            num_envs=4,
        )
        try:
            env._autoreset_envs[:] = [False, True, False, True]
            env._elapsed_steps[:] = [3, 3, 3, 3]
            actions = np.full((4, env._env.action_dim), 0.25, dtype=np.float32)
            with patch.object(env._env, "reset", side_effect=AssertionError("separate reset")):
                env.step_async(actions)
                assert env._env._world.step_pending
                assert env._elapsed_steps.tolist() == [3, 0, 3, 0]
                observations, rewards, terminated, truncated, _ = env.step_wait()
            assert observations.shape[0] == 4
            assert rewards.shape == terminated.shape == truncated.shape == (4,)
            assert env._elapsed_steps.tolist() == [4, 1, 4, 1]
        finally:
            env.close()

    def test_sb3_step_async_delegates_to_native_submission(self):
        pytest.importorskip("stable_baselines3")
        from zeno.gym import make_sb3_env

        env = make_sb3_env("pendulum", num_envs=2)
        try:
            actions = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
            env.step_async(actions)
            assert env.env._env._world.step_pending
            observations, rewards, dones, infos = env.step_wait()
            assert observations.shape[0] == 2
            assert rewards.shape == dones.shape == (2,)
            assert len(infos) == 2
            assert not env.env._env._world.step_pending
        finally:
            env.close()


# ===================================================================
# 12. Profiling API
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


class TestSwarm:
    """Smoke tests for the swarm platform Python bindings."""

    def _make(self, num_agents=8):
        from zeno.swarm import ZenoSwarm, SwarmConfig, create_swarm_world

        world, swarm = create_swarm_world(num_agents=num_agents, communication_range=5.0)
        return world, swarm

    def test_create_step_destroy(self):
        _skip_if_no_lib()
        world, swarm = self._make()

        world.step(np.zeros((world.num_envs, max(world.action_dim, 1)), dtype=np.float32)[:, : world.action_dim])
        swarm.step()

        counts = swarm.get_neighbor_counts()
        assert counts.shape == (swarm.num_agents,)

    def test_metrics_populated(self):
        _skip_if_no_lib()
        world, swarm = self._make()

        for _ in range(3):
            swarm.step()

        metrics = swarm.get_metrics()
        assert metrics.total_edges >= 0
        assert 0.0 <= metrics.connectivity_ratio <= 1.0
        # A fully-clustered spawn grid within communication range is connected.
        assert metrics.fragmentation_score >= 1.0

    def test_oversized_swarm_rejected(self):
        _skip_if_no_lib()
        from zeno._ffi import ZenoWorld
        from zeno.swarm import ZenoSwarm, SwarmConfig

        world = ZenoWorld(mjcf_string=PENDULUM_MJCF, num_envs=1)
        with pytest.raises(RuntimeError):
            ZenoSwarm(world, SwarmConfig(num_agents=100000))
