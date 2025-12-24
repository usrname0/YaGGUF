#!/usr/bin/env python3
"""
Check and update llama.cpp binaries and conversion scripts if needed
Called by startup scripts before launching GUI
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gguf_converter.binary_manager import BinaryManager


def ensure_llama_cpp_repo(binary_manager):
    """
    Ensure llama.cpp repository exists (clone if missing)
    Does NOT auto-update - use Update tab in GUI for updates
    """
    llama_cpp_dir = project_root / "llama.cpp"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    # Only clone if missing
    if not llama_cpp_dir.exists() or not convert_script.exists():
        # Remove old repo if it exists but is incomplete
        if llama_cpp_dir.exists():
            print("Removing incomplete llama.cpp repository...")
            shutil.rmtree(llama_cpp_dir)

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


def update_llama_cpp_scripts():
    """Update llama.cpp conversion scripts via git pull"""
    llama_cpp_dir = project_root / "llama.cpp"

    if not llama_cpp_dir.exists():
        # Shouldn't happen if ensure_llama_cpp_repo() was called first
        print("Warning: llama.cpp directory not found")
        return True

    if not (llama_cpp_dir / ".git").exists():
        print("llama.cpp directory is not a git repository")
        return True

    try:
        print("Updating llama.cpp conversion scripts...")
        result = subprocess.run(
            ["git", "pull", "origin", "master"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            if "Already up to date" in result.stdout:
                print("Conversion scripts already up to date")
            else:
                print("Conversion scripts updated successfully")
            return True
        else:
            print(f"Warning: Failed to update conversion scripts: {result.stderr}")
            return False
    except Exception as e:
        print(f"Warning: Could not update conversion scripts: {e}")
        return False


def main():
    """Check binary version and update if needed"""
    print("Checking llama.cpp binaries and conversion scripts...")
    print()

    manager = BinaryManager()

    # Check if binaries exist
    binaries_updated = False
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
            binaries_updated = True
        except Exception as e:
            print()
            print(f"ERROR: Failed to download binaries: {e}")
            print("The application will continue, but conversions may fail.")
            print("You can manually download binaries later.")
            return 1

    print()

    # Ensure llama.cpp repository exists (clone if needed)
    ensure_llama_cpp_repo(manager)

    # Always try to update conversion scripts if binaries were updated
    if binaries_updated:
        print()
        update_llama_cpp_scripts()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
