#!/usr/bin/env bash
# Build libzeno.dylib and run Python tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIG_BIN="${ZIG:-zig}"
PYTHON_BIN="${PYTHON:-python3}"

if command -v "$ZIG_BIN" >/dev/null 2>&1; then
    ZIG_VERSION="$("$ZIG_BIN" version)"
else
    echo "Could not find Zig executable: $ZIG_BIN" >&2
    exit 1
fi

if [[ "$ZIG_VERSION" != 0.16.* ]]; then
    echo "Unsupported Zig version: $ZIG_VERSION. Use Zig 0.16.x or set ZIG=/path/to/zig-0.16." >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Could not find Python executable: $PYTHON_BIN" >&2
    exit 1
fi

echo "=== Building libzeno.dylib ==="
cd "$PROJECT_ROOT"
"$ZIG_BIN" build -Doptimize=ReleaseFast

LIB_PATH="$PROJECT_ROOT/zig-out/lib"
export DYLD_LIBRARY_PATH="${LIB_PATH}:${DYLD_LIBRARY_PATH:-}"

echo ""
echo "=== Running Python tests ==="
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"
"$PYTHON_BIN" -m pytest tests/test_python_integration.py -v "$@"

echo ""
echo "=== Done ==="
