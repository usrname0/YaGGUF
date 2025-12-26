#!/usr/bin/env python3
"""
Script to bump version numbers for YaGUFF releases

Automates:
1. Update LLAMA_CPP_VERSION in binary_manager.py
2. Increment YaGUFF version in __init__.py
3. Create git commit and tag
4. Optionally push to remote

Usage:
    python scripts/bump_version.py b7600              # Auto-increment patch version
    python scripts/bump_version.py b7600 --version 1.0.10  # Specify version
    python scripts/bump_version.py b7600 --push       # Auto-push after tagging
"""

import sys
import re
import argparse
import subprocess
from pathlib import Path


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


def get_current_yaguff_version():
    """Read current version from __init__.py"""
    init_file = get_project_root() / "gguf_converter" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    return None


def get_current_llama_version():
    """Read current llama.cpp version from binary_manager.py"""
    binary_manager = get_project_root() / "gguf_converter" / "binary_manager.py"
    content = binary_manager.read_text()
    match = re.search(r'LLAMA_CPP_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    return None


def increment_version(version_str):
    """
    Increment patch version (1.0.8 -> 1.0.9)
    """
    parts = version_str.split('.')
    if len(parts) == 3:
        major, minor, patch = parts
        new_patch = int(patch) + 1
        return f"{major}.{minor}.{new_patch}"
    return version_str


def update_llama_version(new_llama_version):
    """Update LLAMA_CPP_VERSION in binary_manager.py"""
    binary_manager = get_project_root() / "gguf_converter" / "binary_manager.py"
    content = binary_manager.read_text()

    # Replace LLAMA_CPP_VERSION
    new_content = re.sub(
        r'LLAMA_CPP_VERSION\s*=\s*["\'][^"\']+["\']',
        f'LLAMA_CPP_VERSION = "{new_llama_version}"',
        content
    )

    binary_manager.write_text(new_content)
    print(f"✓ Updated LLAMA_CPP_VERSION to {new_llama_version}")


def update_yaguff_version(new_version):
    """Update __version__ in __init__.py"""
    init_file = get_project_root() / "gguf_converter" / "__init__.py"
    content = init_file.read_text()

    # Replace __version__
    new_content = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )

    init_file.write_text(new_content)
    print(f"✓ Updated YaGUFF version to {new_version}")


def git_commit_and_tag(yaguff_version, llama_version):
    """Create git commit and tag"""
    try:
        # Stage the changed files
        subprocess.run(
            ["git", "add", "gguf_converter/__init__.py", "gguf_converter/binary_manager.py"],
            check=True,
            cwd=get_project_root()
        )

        # Create commit
        commit_msg = f"Bump to v{yaguff_version} (llama.cpp {llama_version})"
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            cwd=get_project_root()
        )
        print(f"✓ Created commit: {commit_msg}")

        # Create tag
        tag_name = f"v{yaguff_version}"
        subprocess.run(
            ["git", "tag", tag_name],
            check=True,
            cwd=get_project_root()
        )
        print(f"✓ Created tag: {tag_name}")

        return tag_name

    except subprocess.CalledProcessError as e:
        print(f"✗ Git operation failed: {e}")
        return None


def git_push(tag_name):
    """Push commits and tags to remote"""
    try:
        # Push commits
        subprocess.run(
            ["git", "push", "origin", "main"],
            check=True,
            cwd=get_project_root()
        )
        print(f"✓ Pushed commits to origin/main")

        # Push tag
        subprocess.run(
            ["git", "push", "origin", tag_name],
            check=True,
            cwd=get_project_root()
        )
        print(f"✓ Pushed tag {tag_name}")

    except subprocess.CalledProcessError as e:
        print(f"✗ Push failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Bump version numbers for YaGUFF releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bump_version.py b7600                    # Auto-increment patch
  python scripts/bump_version.py b7600 --version 1.0.10   # Specify version
  python scripts/bump_version.py b7600 --push             # Auto-push
  python scripts/bump_version.py b7600 --dry-run          # Preview only
        """
    )

    parser.add_argument(
        "llama_version",
        help="New llama.cpp version (e.g., b7600)"
    )

    parser.add_argument(
        "--version",
        help="Specific YaGUFF version (default: auto-increment patch)"
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help="Push commits and tags to remote after creating"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Get current versions
    current_yaguff = get_current_yaguff_version()
    current_llama = get_current_llama_version()

    print(f"Current YaGUFF version: {current_yaguff}")
    print(f"Current llama.cpp version: {current_llama}")
    print()

    # Determine new YaGUFF version
    if args.version:
        new_yaguff = args.version
    else:
        new_yaguff = increment_version(current_yaguff)

    new_llama = args.llama_version

    print(f"New YaGUFF version: {new_yaguff}")
    print(f"New llama.cpp version: {new_llama}")
    print()

    if args.dry_run:
        print("DRY RUN - No changes made")
        print(f"Would update binary_manager.py: {current_llama} -> {new_llama}")
        print(f"Would update __init__.py: {current_yaguff} -> {new_yaguff}")
        print(f"Would create commit: 'Bump to v{new_yaguff} (llama.cpp {new_llama})'")
        print(f"Would create tag: v{new_yaguff}")
        if args.push:
            print("Would push to origin/main")
        return

    # Confirm with user
    response = input("Proceed with version bump? [y/N] ")
    if response.lower() != 'y':
        print("Aborted")
        return

    # Update versions
    update_llama_version(new_llama)
    update_yaguff_version(new_yaguff)

    # Create commit and tag
    tag_name = git_commit_and_tag(new_yaguff, new_llama)

    if tag_name and args.push:
        print()
        response = input("Push to remote? [y/N] ")
        if response.lower() == 'y':
            git_push(tag_name)

    print()
    print("=" * 50)
    print("Version bump complete!")
    print("=" * 50)
    print()
    print(f"YaGUFF version: {current_yaguff} -> {new_yaguff}")
    print(f"llama.cpp version: {current_llama} -> {new_llama}")
    print(f"Git tag: v{new_yaguff}")
    print()

    if not args.push:
        print("To push to remote, run:")
        print(f"  git push origin main")
        print(f"  git push origin v{new_yaguff}")


if __name__ == "__main__":
    main()
