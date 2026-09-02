#!/usr/bin/env python3
"""Reproducible Zeno Metal versus sequential MuJoCo CPU throughput experiment.

The engines do not have matched solver, contact, observation, or task semantics.
This harness therefore reports an observed end-to-end API throughput ratio, not
a simulator speedup or physics-equivalence claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import zeno
except ImportError as exc:
    raise SystemExit(
        "Zeno is required; build it and run with PYTHONPATH=python"
    ) from exc

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("MuJoCo is required: pip install 'mujoco>=3,<4'") from exc


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance() -> dict:
    """Return non-sensitive hardware/software fields for result attribution."""
    result = {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "macos_build": _command_output(["sw_vers", "-buildVersion"]),
        "python": platform.python_version(),
        "zig": _command_output(["zig", "version"]),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
    }
    raw = _command_output(
        ["system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType"]
    )
    if raw:
        try:
            report = json.loads(raw)
            hardware = report.get("SPHardwareDataType", [{}])[0]
            display = report.get("SPDisplaysDataType", [{}])[0]
            result.update(
                {
                    "chip": hardware.get("chip_type"),
                    "cpu_configuration": hardware.get("number_processors"),
                    "memory": hardware.get("physical_memory"),
                    "gpu": display.get("sppci_model") or display.get("_name"),
                    "gpu_cores": display.get("sppci_cores"),
                    "metal_family": display.get("spdisplays_mtlgpufamilysupport"),
                }
            )
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    return {key: value for key, value in result.items() if value}


def _summary(samples: list[float], num_envs: int, num_steps: int) -> dict:
    median_seconds = statistics.median(samples)
    total_env_steps = num_envs * num_steps
    return {
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "median_env_steps_per_second": total_env_steps / median_seconds,
        "median_ms_per_batch_step": median_seconds * 1000.0 / num_steps,
    }


def benchmark_zeno(
    model_path: Path,
    num_envs: int,
    num_steps: int,
    warmup_steps: int,
    repeats: int,
    actions: np.ndarray,
) -> dict:
    samples: list[float] = []
    creation_samples: list[float] = []
    checksum = 0.0
    observation_dim = 0
    for _ in range(repeats):
        start = time.perf_counter()
        env = zeno.make(str(model_path), num_envs=num_envs)
        creation_samples.append(time.perf_counter() - start)
        try:
            env.reset()
            for _ in range(warmup_steps):
                env.step(actions)
            env.reset()

            start = time.perf_counter()
            for _ in range(num_steps):
                observations, rewards, dones, _ = env.step(actions)
            samples.append(time.perf_counter() - start)
            observation_dim = int(observations.shape[-1])
            checksum = float(
                np.sum(observations) + np.sum(rewards) + np.sum(dones)
            )
        finally:
            env.close()

    result = _summary(samples, num_envs, num_steps)
    result.update(
        {
            "creation_median_seconds": statistics.median(creation_samples),
            "observation_dim": observation_dim,
            "action_dim": int(actions.shape[-1]),
            "final_output_checksum": checksum,
        }
    )
    return result


def benchmark_mujoco(
    model_path: Path,
    num_envs: int,
    num_steps: int,
    warmup_steps: int,
    repeats: int,
    controls: np.ndarray,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    samples: list[float] = []
    creation_samples: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        datas = [mujoco.MjData(model) for _ in range(num_envs)]
        creation_samples.append(time.perf_counter() - start)

        for _ in range(warmup_steps):
            for env_id, data in enumerate(datas):
                data.ctrl[:] = controls[env_id]
                mujoco.mj_step(model, data)
        for data in datas:
            mujoco.mj_resetData(model, data)

        start = time.perf_counter()
        for _ in range(num_steps):
            for env_id, data in enumerate(datas):
                data.ctrl[:] = controls[env_id]
                mujoco.mj_step(model, data)
        samples.append(time.perf_counter() - start)
        checksum = float(
            sum(np.sum(data.qpos) + np.sum(data.qvel) for data in datas)
        )

    result = _summary(samples, num_envs, num_steps)
    result.update(
        {
            "creation_median_seconds": statistics.median(creation_samples),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "final_state_checksum": checksum,
        }
    )
    return result


def _print_result(label: str, result: dict) -> None:
    print(f"{label}:")
    print(
        "  median: "
        f"{result['median_seconds']:.3f}s "
        f"(range {result['min_seconds']:.3f}–{result['max_seconds']:.3f}s)"
    )
    print(
        "  throughput: "
        f"{result['median_env_steps_per_second']:,.0f} environment-steps/s"
    )
    print(f"  batch-step latency: {result['median_ms_per_batch_step']:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("assets/pendulum.xml"))
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.envs <= 0 or args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error(
            "envs, steps, and repeats must be positive; warmup cannot be negative"
        )
    model_path = args.model.resolve()
    if not model_path.is_file():
        parser.error(f"model does not exist: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    rng = np.random.default_rng(args.seed)
    controls = rng.uniform(-1.0, 1.0, (args.envs, model.nu)).astype(np.float32)

    # Fail explicitly instead of silently padding or dropping controls if a
    # model exposes a different actuator surface in the two parsers.
    probe = zeno.make(str(model_path), num_envs=1)
    try:
        zeno_action_dim = probe.action_dim
    finally:
        probe.close()
    if zeno_action_dim != model.nu:
        raise SystemExit(
            f"actuator mismatch: Zeno action_dim={zeno_action_dim}, "
            f"MuJoCo nu={model.nu}"
        )

    machine = provenance()
    print("Zeno Metal versus sequential MuJoCo CPU throughput experiment")
    print("Not a simulator speedup or physics-equivalence claim.")
    print(f"Model: {model_path}")
    print(
        f"Workload: {args.envs} envs × {args.steps} steps; "
        f"warm-up {args.warmup}; repeats {args.repeats}; seed {args.seed}"
    )
    print("Provenance: " + json.dumps(machine, sort_keys=True))

    zeno_result = benchmark_zeno(
        model_path, args.envs, args.steps, args.warmup, args.repeats, controls
    )
    mujoco_result = benchmark_mujoco(
        model_path, args.envs, args.steps, args.warmup, args.repeats, controls
    )
    ratio = (
        zeno_result["median_env_steps_per_second"]
        / mujoco_result["median_env_steps_per_second"]
    )
    _print_result("Zeno Metal batched API", zeno_result)
    _print_result("MuJoCo sequential CPU API", mujoco_result)
    print(f"Observed throughput ratio: {ratio:.3g}×")
    print("The ratio measures these execution strategies; semantics are not matched.")

    try:
        report_model = str(model_path.relative_to(Path.cwd()))
    except ValueError:
        report_model = str(model_path)
    report = {
        "schema_version": 1,
        "claim_boundary": "throughput-only; simulator semantics are not matched",
        "model": report_model,
        "num_envs": args.envs,
        "num_steps": args.steps,
        "warmup_steps": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "provenance": machine,
        "zeno_metal": zeno_result,
        "mujoco_sequential_cpu": mujoco_result,
        "observed_throughput_ratio": ratio,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON report: {args.json_output}")


if __name__ == "__main__":
    main()
