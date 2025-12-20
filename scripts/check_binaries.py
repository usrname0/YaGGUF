#!/usr/bin/env python3
"""
Check and update llama.cpp binaries and conversion scripts if needed
Called by startup scripts before launching GUI
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gguf_converter.binary_manager import BinaryManager


def update_llama_cpp_scripts():
    """Update llama.cpp conversion scripts via git pull"""
    llama_cpp_dir = project_root / "llama.cpp"

    if not llama_cpp_dir.exists():
        print("llama.cpp directory not found, will be cloned on first conversion")
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

    # Always try to update conversion scripts if binaries were updated
    if binaries_updated:
        print()
        update_llama_cpp_scripts()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
