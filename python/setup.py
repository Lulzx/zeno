"""Build hooks for the native Zeno Python wheel."""

from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


PYTHON_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PYTHON_DIR.parent


class BuildNativeLibrary(build_py):
    """Compile and bundle the C ABI library before copying Python modules."""

    def run(self):
        if sys.platform != "darwin":
            raise RuntimeError("Zeno wheels currently support macOS only")

        sdk_path = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
        subprocess.check_call(
            [
                "zig",
                "build",
                "-Doptimize=ReleaseFast",
                "-Dtarget=aarch64-macos.13.0",
                "--sysroot",
                sdk_path,
            ],
            cwd=REPOSITORY_ROOT,
        )
        super().run()

        source = REPOSITORY_ROOT / "zig-out" / "lib" / "libzeno.dylib"
        destination = Path(self.build_lib) / "zeno" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        # Model-name factories and packaged examples must work without a source
        # checkout, so ship the small reference model/scenario collection.
        shutil.copytree(
            REPOSITORY_ROOT / "assets",
            Path(self.build_lib) / "zeno" / "assets",
            dirs_exist_ok=True,
        )


class PlatformWheel(bdist_wheel):
    """Mark the ABI-loaded dylib wheel as non-pure and macOS-specific."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "macosx_13_0_arm64"


setup(
    cmdclass={
        "build_py": BuildNativeLibrary,
        "bdist_wheel": PlatformWheel,
    }
)
