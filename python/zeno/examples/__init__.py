"""
Zeno Physics Engine Examples

Comprehensive examples demonstrating Zeno's capabilities for
high-performance batched robotics simulation and reinforcement learning.

Example Categories
------------------
basic/
    Core functionality: pendulum, cartpole, joints, shapes, viewer, forces

robot/
    Locomotion and manipulation: ant, humanoid, walker, hopper, swimmer,
    cheetah, reacher, pusher, policy

training/
    RL algorithms: ppo, evolution, random_search, curriculum, imitation,
    batched_training

utils/
    Advanced features: reset_patterns, state_access, reward_shaping,
    domain_randomization, parallel_evaluation, checkpointing

benchmark/
    Performance: throughput, scaling, latency, memory, comparison

Running Examples
----------------
Via command line::

    python -m zeno.examples <category>_<name>
    python -m zeno.examples basic_pendulum
    python -m zeno.examples robot_ant --num-envs 1024
    python -m zeno.examples training_ppo --env humanoid
    python -m zeno.examples benchmark_throughput --num-envs 4096

Or import directly::

    from zeno.examples.basic import pendulum
    pendulum.main()
"""

from pathlib import Path


def get_asset_directory() -> Path:
    """Get the path to the assets directory."""
    # Check relative to this file first
    pkg_assets = Path(__file__).parent.parent.parent.parent / "assets"
    if pkg_assets.exists():
        return pkg_assets

    # Check current working directory
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets.exists():
        return cwd_assets

    raise FileNotFoundError("Could not find assets directory")


def get_asset(filename: str) -> str:
    """Get the full path to an asset file."""
    asset_path = get_asset_directory() / filename
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset not found: {filename}")
    return str(asset_path)


def list_examples() -> dict:
    """List all available examples organized by category."""
    return {
        "basic": [
            "pendulum",
            "cartpole",
            "joints",
            "shapes",
            "viewer",
            "forces",
        ],
        "robot": [
            "ant",
            "humanoid",
            "walker",
            "hopper",
            "swimmer",
            "cheetah",
            "reacher",
            "pusher",
            "policy",
        ],
        "training": [
            "batched_training",
            "ppo",
            "evolution",
            "random_search",
            "curriculum",
            "imitation",
        ],
        "utils": [
            "reset_patterns",
            "state_access",
            "reward_shaping",
            "domain_randomization",
            "parallel_evaluation",
            "checkpointing",
        ],
        "benchmark": [
            "throughput",
            "scaling",
            "latency",
            "memory",
            "comparison",
        ],
    }


def count_examples() -> int:
    """Count total number of examples."""
    examples = list_examples()
    return sum(len(v) for v in examples.values())


__all__ = [
    "get_asset_directory",
    "get_asset",
    "list_examples",
    "count_examples",
]
