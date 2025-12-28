#!/usr/bin/env python3
"""
Check and update llama.cpp binaries and conversion scripts if needed
Called by startup scripts before launching GUI
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gguf_converter.binary_manager import BinaryManager, remove_readonly


def check_git_available() -> bool:
    """
    Check if git is installed and available

    Returns:
        True if git is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_llama_cpp_repo(binary_manager: Any) -> bool:
    """
    Ensure llama.cpp repository exists (clone if missing)
    Does NOT auto-update - use Update tab in GUI for updates
    """
    llama_cpp_dir = project_root / "llama.cpp"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    # Only clone if missing
    if not llama_cpp_dir.exists() or not convert_script.exists():
        # Check if git is available before attempting clone
        if not check_git_available():
            print()
            print("WARNING: Git is not installed or not available in PATH.")
            print("The llama.cpp conversion scripts cannot be automatically downloaded.")
            print("The application will continue, but model conversions will fail.")
            print()
            print("To fix this:")
            print("  1. Install git from https://git-scm.com/downloads")
            print("  2. Restart the application")
            print()
            return False

        # Remove old repo if it exists but is incomplete
        if llama_cpp_dir.exists():
            print("Removing incomplete llama.cpp repository...")
            shutil.rmtree(llama_cpp_dir, onerror=remove_readonly)

        # Clone fresh copy
        expected_version = binary_manager.LLAMA_CPP_VERSION
        print(f"Cloning llama.cpp repository (version {expected_version})...")
        print("This may take a minute...")
        print()

        try:
            # Clone at the specific recommended version
            subprocess.run([
                "git", "clone",
                "https://github.com/ggml-org/llama.cpp.git",
                "--depth=1",
                "--branch", expected_version,
                str(llama_cpp_dir)
            ], check=True, capture_output=True, text=True)

            print(f"llama.cpp repository ready (version {expected_version})")
            return True
        except subprocess.CalledProcessError as e:
            print()
            print(f"ERROR: Failed to clone llama.cpp repository: {e.stderr if e.stderr else str(e)}")
            print("Please check your git installation and internet connection.")
            print("The application will continue, but conversions may fail.")
            return False
        except Exception as e:
            print()
            print(f"ERROR: Failed to clone llama.cpp repository: {e}")
            print("The application will continue, but conversions may fail.")
            return False

    return True


def main() -> int:
    """Check binary version and update if needed"""
    print("Checking llama.cpp binaries and conversion scripts...")
    print()

    manager = BinaryManager()

    # Check if binaries exist
    if manager._binaries_exist():
        print(f"Binaries up to date (version {manager.LLAMA_CPP_VERSION})")
    else:
        print("Binaries not installed")
        print("Downloading llama.cpp binaries...")
        print("This may take a few minutes...")
        print()

        try:
            manager.download_binaries(force=True)
            print()
            print("Binaries updated successfully!")
        except Exception as e:
            print()
            print(f"ERROR: Failed to download binaries: {e}")
            print("The application will continue, but conversions may fail.")
            print("You can manually download binaries later.")
            return 1

    print()

    # Ensure llama.cpp repository exists (clone if needed)
    ensure_llama_cpp_repo(manager)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
