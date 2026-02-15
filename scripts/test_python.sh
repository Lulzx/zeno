#!/usr/bin/env bash
# Build libzeno.dylib and run Python tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building libzeno.dylib ==="
cd "$PROJECT_ROOT"
zig build -Doptimize=ReleaseFast

LIB_PATH="$PROJECT_ROOT/zig-out/lib"
export DYLD_LIBRARY_PATH="${LIB_PATH}:${DYLD_LIBRARY_PATH:-}"

echo ""
echo "=== Running Python tests ==="
cd "$PROJECT_ROOT"
python -m pytest python/tests/ -v "$@"

echo ""
echo "=== Done ==="
