#!/usr/bin/env python3
"""Live ballistic trajectory comparison between Zeno and MuJoCo."""

import argparse

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("MuJoCo is required: pip install 'mujoco>=3,<4'") from exc

try:
    from zeno import ZenoWorld
except ImportError as exc:
    raise SystemExit("Zeno is required; set PYTHONPATH=python and build the library") from exc


def ballistic_model(dt: float) -> str:
    return f"""<mujoco model="ballistic-validation">
    <option timestep="{dt:.9g}" gravity="0 0 -9.81" integrator="Euler"/>
    <worldbody>
        <body name="ball" pos="0 0 10">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1"
                  contype="0" conaffinity="0"/>
        </body>
    </worldbody>
</mujoco>"""


def run_comparison(steps: int, dt: float) -> dict:
    xml = ballistic_model(dt)

    zeno_world = ZenoWorld(mjcf_string=xml, timestep=dt)
    zeno_world.reset()
    zeno_positions = zeno_world.get_body_positions(zero_copy=True)
    # Zeno materializes an explicit world body at index 0.
    zeno_ball_index = 1

    mujoco_model = mujoco.MjModel.from_xml_string(xml)
    mujoco_data = mujoco.MjData(mujoco_model)
    mujoco.mj_forward(mujoco_model, mujoco_data)
    mujoco_ball_index = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )

    zeno_trajectory = []
    mujoco_trajectory = []
    actions = np.zeros((1, zeno_world.action_dim), dtype=np.float32)

    for step in range(steps + 1):
        zeno_trajectory.append(zeno_positions[0, zeno_ball_index, :3].copy())
        mujoco_trajectory.append(mujoco_data.xpos[mujoco_ball_index].copy())
        if step < steps:
            zeno_world.step(actions)
            mujoco.mj_step(mujoco_model, mujoco_data)

    zeno = np.asarray(zeno_trajectory)
    reference = np.asarray(mujoco_trajectory)
    errors = np.linalg.norm(zeno - reference, axis=1)
    zeno_world.close()
    return {
        "mean_error": float(errors.mean()),
        "max_error": float(errors.max()),
        "final_zeno": zeno[-1],
        "final_mujoco": reference[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--max-error", type=float, default=0.01)
    args = parser.parse_args()

    if args.steps <= 0 or args.dt <= 0 or args.max_error <= 0:
        parser.error("steps, dt, and max-error must be positive")

    result = run_comparison(args.steps, args.dt)
    print("Zeno vs MuJoCo ballistic validation")
    print(f"  steps: {args.steps}, dt: {args.dt:g} s")
    print(f"  final Zeno position:  {result['final_zeno']}")
    print(f"  final MuJoCo position: {result['final_mujoco']}")
    print(f"  mean position error: {result['mean_error']:.6f} m")
    print(f"  max position error:  {result['max_error']:.6f} m")
    print(f"  threshold:           {args.max_error:.6f} m")

    if not np.isfinite(result["max_error"]) or result["max_error"] > args.max_error:
        raise SystemExit("VALIDATION FAILED")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
