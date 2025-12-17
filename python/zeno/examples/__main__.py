"""
Entry point for running Zeno examples.

Usage:
    python -m zeno.examples                      # List all examples
    python -m zeno.examples <name>               # Run example
    python -m zeno.examples basic_pendulum       # Run pendulum example
    python -m zeno.examples robot_ant --num-envs 1024

Examples are named as <category>_<name>:
    basic_pendulum, basic_cartpole, basic_joints, ...
    robot_ant, robot_humanoid, robot_walker, ...
    training_ppo, training_evolution, ...
    utils_reset_patterns, utils_state_access, ...
    benchmark_throughput, benchmark_scaling, ...
"""

import sys


def print_help():
    """Print help message with all available examples."""
    from . import list_examples, count_examples

    examples = list_examples()

    print("Zeno Physics Engine Examples")
    print("=" * 60)
    print()
    print(f"Total examples: {count_examples()}")
    print()
    print("Usage: python -m zeno.examples <category>_<name> [options]")
    print()
    print("Available examples:")
    print("-" * 60)

    for category, names in examples.items():
        print(f"\n{category.upper()}:")
        for name in names:
            full_name = f"{category}_{name}"
            print(f"  {full_name}")

    print()
    print("Common options:")
    print("  --num-envs N    Number of parallel environments")
    print("  --steps N       Number of simulation steps")
    print("  --seed N        Random seed")
    print("  --help          Show example-specific help")
    print()
    print("Examples:")
    print("  python -m zeno.examples basic_pendulum")
    print("  python -m zeno.examples robot_ant --num-envs 1024")
    print("  python -m zeno.examples training_ppo --env humanoid")
    print("  python -m zeno.examples benchmark_throughput --num-envs 4096")


def main():
    """Main entry point."""
    from . import list_examples

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        return

    example_name = sys.argv[1]

    # Parse example name: category_name
    examples = list_examples()

    # Build lookup table
    example_map = {}
    for category, names in examples.items():
        for name in names:
            full_name = f"{category}_{name}"
            example_map[full_name] = (category, name)

    if example_name not in example_map:
        print(f"Error: Unknown example '{example_name}'")
        print()
        print("Available examples:")
        for full_name in sorted(example_map.keys()):
            print(f"  {full_name}")
        sys.exit(1)

    category, name = example_map[example_name]

    # Update argv for the example's argparse
    sys.argv = [f"zeno.examples.{category}.{name}"] + sys.argv[2:]

    # Import and run the example
    if category == "basic":
        if name == "pendulum":
            from .basic import pendulum
            pendulum.main()
        elif name == "cartpole":
            from .basic import cartpole
            cartpole.main()
        elif name == "joints":
            from .basic import joints
            joints.main()
        elif name == "shapes":
            from .basic import shapes
            shapes.main()
        elif name == "viewer":
            from .basic import viewer
            viewer.main()
        elif name == "forces":
            from .basic import forces
            forces.main()

    elif category == "robot":
        if name == "ant":
            from .robot import ant
            ant.main()
        elif name == "humanoid":
            from .robot import humanoid
            humanoid.main()
        elif name == "walker":
            from .robot import walker
            walker.main()
        elif name == "hopper":
            from .robot import hopper
            hopper.main()
        elif name == "swimmer":
            from .robot import swimmer
            swimmer.main()
        elif name == "cheetah":
            from .robot import cheetah
            cheetah.main()
        elif name == "reacher":
            from .robot import reacher
            reacher.main()
        elif name == "pusher":
            from .robot import pusher
            pusher.main()
        elif name == "policy":
            from .robot import policy
            policy.main()

    elif category == "training":
        if name == "batched_training":
            from .training import batched_training
            batched_training.main()
        elif name == "ppo":
            from .training import ppo
            ppo.main()
        elif name == "evolution":
            from .training import evolution
            evolution.main()
        elif name == "random_search":
            from .training import random_search
            random_search.main()
        elif name == "curriculum":
            from .training import curriculum
            curriculum.main()
        elif name == "imitation":
            from .training import imitation
            imitation.main()

    elif category == "utils":
        if name == "reset_patterns":
            from .utils import reset_patterns
            reset_patterns.main()
        elif name == "state_access":
            from .utils import state_access
            state_access.main()
        elif name == "reward_shaping":
            from .utils import reward_shaping
            reward_shaping.main()
        elif name == "domain_randomization":
            from .utils import domain_randomization
            domain_randomization.main()
        elif name == "parallel_evaluation":
            from .utils import parallel_evaluation
            parallel_evaluation.main()
        elif name == "checkpointing":
            from .utils import checkpointing
            checkpointing.main()

    elif category == "benchmark":
        if name == "throughput":
            from .benchmark import throughput
            throughput.main()
        elif name == "scaling":
            from .benchmark import scaling
            scaling.main()
        elif name == "latency":
            from .benchmark import latency
            latency.main()
        elif name == "memory":
            from .benchmark import memory
            memory.main()
        elif name == "comparison":
            from .benchmark import comparison
            comparison.main()


if __name__ == "__main__":
    main()
