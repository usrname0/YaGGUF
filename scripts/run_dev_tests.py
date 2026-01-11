"""
Run development tests with version information header.

This script displays llama.cpp version information and then runs pytest
with markers to exclude slow and integration tests by default.
"""

import sys
import subprocess
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gguf_converter.llama_cpp_manager import LlamaCppManager
from gguf_converter.converter import GGUFConverter


def print_version_info():
    """Print llama.cpp version information."""
    print()
    print("=" * 80)
    print(" " * 32 + "DEV TESTS")
    print("=" * 80)
    print()

    # Expected version
    manager = LlamaCppManager()
    print(f"Expected llama.cpp: {manager.LLAMA_CPP_VERSION}")

    # Get binaries version
    try:
        imatrix_bin = manager.get_imatrix_path()
        result = subprocess.run(
            [str(imatrix_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stderr if result.stderr else result.stdout
        version = "unknown"
        for line in output.split("\n"):
            if line.startswith("version:"):
                match = re.search(r"\b(b?\d{4,5})\b", line)
                if match:
                    version = match.group(1) if match.group(1).startswith("b") else f"b{match.group(1)}"
                break
        print(f"Binaries version: {version} ({imatrix_bin.parent})")
    except Exception:
        print("Binaries version: unknown")

    # Get conversion scripts version
    try:
        converter = GGUFConverter()
        script = converter._find_convert_script()
        if script:
            llama_dir = script.parent
            result = subprocess.run(
                ["git", "describe", "--tags", "--always"],
                cwd=llama_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            scripts_ver = result.stdout.strip().split("-")[0] if result.returncode == 0 else "unknown"
            print(f"Conversion scripts: {scripts_ver} ({llama_dir})")
        else:
            print("Conversion scripts: not found")
    except Exception:
        print("Conversion scripts: unknown")

    print()
    print("=" * 80)
    print()


def main():
    """Main entry point."""
    # Print version info
    print_version_info()

    # Run pytest with default markers
    pytest_args = ["-m", "not slow and not integration"]

    # Pass through any additional arguments
    if len(sys.argv) > 1:
        pytest_args.extend(sys.argv[1:])

    # Run pytest
    sys.exit(subprocess.call([sys.executable, "-m", "pytest"] + pytest_args))


if __name__ == "__main__":
    main()
