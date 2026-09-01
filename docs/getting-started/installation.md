# Installation

## Requirements

- **macOS 13+** (Ventura or later)
- **Zig 0.16.0** ([download](https://ziglang.org/download/))
- **Apple Silicon** (M1/M2/M3/M4) recommended

!!! warning "Zig version"
    Zeno targets Zig 0.16, matching `.tool-versions` and CI. Zig 0.14 is no longer supported: besides the stdlib API differences, Zig 0.14 cannot link native binaries against the macOS 26 SDK at all (its libSystem stub parser predates that SDK format).

## Building from Source

### Clone the Repository

```bash
git clone https://github.com/lulzx/zeno.git
cd zeno
```

### Build the Library

```bash
# Debug build
zig build

# Optimized release build (recommended)
zig build -Doptimize=ReleaseFast

# Optional: validate shaders offline and install zig-out/lib/zeno.metallib
zig build metallib
```

### Run Tests

```bash
zig build test
```

## Python Installation

### Prerequisites

Ensure you have Python 3.9+ and pip installed.

### Install from Source

```bash
# The native library must be built from the repository root first.
zig build -Doptimize=ReleaseFast
pip install -e 'python[gymnasium]'
```

This installs Zeno in editable mode, allowing you to modify the source and see changes immediately.

### Verify Installation

```python
import zeno
print(zeno.__version__)
```

## Dependencies

### Zig (Core)

Zeno's core is written in pure Zig with no external dependencies beyond the Metal framework provided by macOS.

### Python Bindings

The Python bindings require:

- `numpy` - Array operations
- `cffi` - Foreign function interface

Optional dependencies:

- `gymnasium` - For Gym-compatible environments
- `stable-baselines3` - For the `make_sb3_env` adapter

Install all dependencies:

```bash
pip install -e 'python[all]'
```

## Benchmarks

To run the performance benchmarks:

```bash
# Zig benchmarks
zig build bench

# Python comparison with MuJoCo
pip install mujoco  # Optional, for comparison
python benchmarks/compare_mujoco.py --envs 1024 --steps 1000
```

The comparison reports a same-machine throughput ratio. It is not a simulator
speedup or physics-parity claim because model and solver semantics are not
matched. Use `python benchmarks/validate_physics.py` for the bounded ballistic
trajectory validation.

## Troubleshooting

### Metal Not Available

If you see errors about Metal not being available:

1. Ensure you're running macOS 13 or later
2. Check that your Mac has a Metal-compatible GPU
3. Run `system_profiler SPDisplaysDataType` to verify Metal support

### Build Errors

If the Zig build fails:

1. Verify Zig version: `zig version` (should be 0.16.x)
2. Clean and rebuild: `rm -rf .zig-cache && zig build`

### Python Import Errors

If `import zeno` fails:

1. Ensure the library was built: check for `zig-out/lib/libzeno.dylib`
2. Reinstall: `cd python && pip install -e . --force-reinstall`
