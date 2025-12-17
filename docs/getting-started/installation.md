# Installation

## Requirements

- **macOS 13+** (Ventura or later)
- **Zig 0.15+** ([download](https://ziglang.org/download/))
- **Apple Silicon** (M1/M2/M3/M4) recommended

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
cd python
pip install -e .
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

Install all dependencies:

```bash
pip install numpy cffi gymnasium
```

## Benchmarks

To run the performance benchmarks:

```bash
# Zig benchmarks
zig build bench

# Python comparison with MuJoCo
cd benchmarks
pip install mujoco  # Optional, for comparison
python compare_mujoco.py --envs 1024 --steps 1000
```

## Troubleshooting

### Metal Not Available

If you see errors about Metal not being available:

1. Ensure you're running macOS 13 or later
2. Check that your Mac has a Metal-compatible GPU
3. Run `system_profiler SPDisplaysDataType` to verify Metal support

### Build Errors

If the Zig build fails:

1. Verify Zig version: `zig version` (should be 0.15+)
2. Clean and rebuild: `rm -rf .zig-cache && zig build`

### Python Import Errors

If `import zeno` fails:

1. Ensure the library was built: check for `zig-out/lib/libzeno.dylib`
2. Reinstall: `cd python && pip install -e . --force-reinstall`
