#!/usr/bin/env python3
"""
Validate that every zeno_* function declared in python/zeno/_ffi.py cdef
is exported by the built native library in zig-out/lib.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _extract_declared_functions(ffi_file: Path) -> list[str]:
    text = ffi_file.read_text(encoding="utf-8")
    match = re.search(r'ffi\.cdef\(\s*"""(.*?)"""\s*\)', text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not locate ffi.cdef block in {ffi_file}")

    cdef_text = match.group(1)
    names = sorted(set(re.findall(r"\b(zeno_[A-Za-z0-9_]+)\s*\(", cdef_text)))
    return names


def _read_exported_symbols(lib_path: Path) -> set[str]:
    if sys.platform == "darwin":
        cmd = ["nm", "-gU", str(lib_path)]
    else:
        cmd = ["nm", "-g", str(lib_path)]

    output = subprocess.check_output(cmd, text=True)
    exported: set[str] = set()
    for line in output.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        sym = parts[-1]
        if sym.startswith("_"):
            sym = sym[1:]
        if sym.startswith("zeno_"):
            exported.add(sym)
    return exported


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ffi_file = repo_root / "python" / "zeno" / "_ffi.py"

    if sys.platform == "darwin":
        lib_path = repo_root / "zig-out" / "lib" / "libzeno.dylib"
    else:
        lib_path = repo_root / "zig-out" / "lib" / "libzeno.so"

    if not lib_path.exists():
        print(f"ABI check failed: library not found at {lib_path}", file=sys.stderr)
        return 1

    declared = _extract_declared_functions(ffi_file)
    exported = _read_exported_symbols(lib_path)

    missing = [name for name in declared if name not in exported]
    if missing:
        print("ABI check failed: missing exported symbols for declared cdef functions:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1

    print(f"ABI check passed: {len(declared)} declared zeno_* functions are exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
