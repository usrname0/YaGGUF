#!/usr/bin/env python3
"""
Interactive script to bump version numbers for YaGUFF releases

Workflow:
1. Shows recent version history
2. Prompts for new llama.cpp version (default: latest from GitHub)
3. Prompts for new YaGUFF version (default: auto-increment patch)
4. Verifies llama.cpp version exists on GitHub
5. Choose action: dry run, commit locally, or commit and push

Usage:
    python scripts/bump_version.py

The script will guide you through the process interactively.
"""

import sys
import re
import argparse
import subprocess
import json
from pathlib import Path
from urllib.request import urlopen


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def blue(text):
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def green(text):
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def yellow(text):
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def red(text):
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def bold(text):
        return f"{Colors.BOLD}{text}{Colors.RESET}"


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


def get_current_branch():
    """
    Get current git branch name

    Returns:
        Branch name or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root()
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_recent_versions(count=3):
    """
    Get recent YaGUFF version tags from git

    Args:
        count: Number of recent versions to retrieve

    Returns:
        List of (tag, yaguff_version, llama_version) tuples
    """
    try:
        # Get recent tags sorted by version
        result = subprocess.run(
            ["git", "tag", "-l", "v*", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_project_root()
        )

        tags = result.stdout.strip().split('\n')[:count]
        versions = []

        for tag in tags:
            if not tag:
                continue

            # Get commit message for this tag
            commit_result = subprocess.run(
                ["git", "log", "-1", "--format=%s", tag],
                capture_output=True,
                text=True,
                check=True,
                cwd=get_project_root()
            )

            commit_msg = commit_result.stdout.strip()
            # Extract llama.cpp version from commit message
            # Format: "Bump to v1.0.26 (llama.cpp b7548)"
            llama_match = re.search(r'\(llama\.cpp\s+([^)]+)\)', commit_msg)
            llama_version = llama_match.group(1) if llama_match else "unknown"

            # Extract YaGUFF version from tag
            yaguff_version = tag[1:] if tag.startswith('v') else tag

            versions.append((tag, yaguff_version, llama_version))

        return versions

    except subprocess.CalledProcessError:
        return []


def get_latest_llama_version():
    """
    Get the latest llama.cpp release tag from GitHub

    Returns:
        Latest release tag (e.g., "b7493") or None if fetch fails
    """
    try:
        api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        with urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            return data['tag_name']
    except Exception as e:
        print(Colors.yellow(f"Warning: Could not fetch latest llama.cpp version: {e}"))
        return None


def verify_llama_version_exists(version_tag):
    """
    Verify that a specific llama.cpp version tag exists on GitHub

    Args:
        version_tag: Version tag to check (e.g., "b7600")

    Returns:
        True if the tag exists, False otherwise
    """
    try:
        # Try to fetch the specific tag from GitHub API
        api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/git/refs/tags/{version_tag}"
        with urlopen(api_url) as response:
            if response.status == 200:
                return True
            return False
    except Exception:
        # If we get a 404 or any other error, the tag doesn't exist
        return False


def compare_llama_versions(version1, version2):
    """
    Compare two llama.cpp version strings (e.g., "b7600" vs "b7500")

    Args:
        version1: First version string
        version2: Second version string

    Returns:
        -1 if version1 < version2 (going backwards)
        0 if version1 == version2
        1 if version1 > version2 (going forward)
        None if versions can't be compared
    """
    try:
        # Extract numeric part after 'b' prefix
        if version1.startswith('b') and version2.startswith('b'):
            num1 = int(version1[1:])
            num2 = int(version2[1:])
            if num1 < num2:
                return -1
            elif num1 == num2:
                return 0
            else:
                return 1
    except (ValueError, AttributeError):
        pass
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
    print(Colors.green(f"Updated LLAMA_CPP_VERSION to {new_llama_version}"))


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
    print(Colors.green(f"Updated YaGUFF version to {new_version}"))


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
        print(Colors.green(f"Created commit: {commit_msg}"))

        # Create tag
        tag_name = f"v{yaguff_version}"
        subprocess.run(
            ["git", "tag", tag_name],
            check=True,
            cwd=get_project_root()
        )
        print(Colors.green(f"Created tag: {tag_name}"))

        return tag_name

    except subprocess.CalledProcessError as e:
        print(Colors.red(f"Git operation failed: {e}"))
        return None


def git_push(tag_name, branch_name):
    """
    Push commits and tags to remote

    Args:
        tag_name: Git tag to push
        branch_name: Branch name to push
    """
    try:
        # Push commits to current branch
        subprocess.run(
            ["git", "push", "origin", branch_name],
            check=True,
            cwd=get_project_root()
        )
        print(Colors.green(f"Pushed commits to origin/{branch_name}"))

        # Push tag
        subprocess.run(
            ["git", "push", "origin", tag_name],
            check=True,
            cwd=get_project_root()
        )
        print(Colors.green(f"Pushed tag {tag_name}"))

    except subprocess.CalledProcessError as e:
        print(Colors.red(f"Push failed: {e}"))


def main():
    parser = argparse.ArgumentParser(
        description="Bump version numbers for YaGUFF releases (interactive mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args = parser.parse_args()

    # Get current branch
    current_branch = get_current_branch()
    if current_branch:
        print("=" * 60)
        print(f"Current branch: {Colors.blue(current_branch)}")
        print("=" * 60)
    else:
        print(Colors.yellow("Warning: Could not detect git branch"))

    # Show recent versions
    print()
    print(Colors.bold("Recent YaGUFF Versions:"))
    print("-" * 60)

    recent = get_recent_versions(3)
    if recent:
        for tag, yaguff_ver, llama_ver in recent:
            print(f"  {Colors.blue(tag):12} - YaGUFF {yaguff_ver:8} - llama.cpp {llama_ver}")
    else:
        print(Colors.yellow("  No version tags found"))

    # Get current versions
    current_yaguff = get_current_yaguff_version()
    current_llama = get_current_llama_version()

    print()
    print(Colors.bold("Current versions:"))
    print(f"  YaGUFF:      {Colors.green(current_yaguff)}")
    print(f"  llama.cpp:   {Colors.green(current_llama)}")

    # Fetch latest llama.cpp version from GitHub
    print()
    print("Fetching latest llama.cpp version from GitHub...")
    latest_llama = get_latest_llama_version()
    if latest_llama:
        print(f"  Latest llama.cpp: {Colors.green(latest_llama)}")
        if latest_llama == current_llama:
            print(f"  {Colors.yellow('(already up to date)')}")
    else:
        latest_llama = None
        print(Colors.yellow("  Could not fetch latest llama.cpp version"))

    # Interactive prompts
    print()
    print("=" * 60)
    print()

    # Prompt for llama.cpp version
    if latest_llama:
        llama_prompt = f"New llama.cpp version [Enter={Colors.green(latest_llama)}]: "
    else:
        llama_prompt = "New llama.cpp version: "

    llama_input = input(llama_prompt).strip()
    new_llama = llama_input if llama_input else latest_llama

    if not new_llama:
        print(Colors.red("Error: llama.cpp version required"))
        return

    # Prompt for YaGUFF version
    suggested_yaguff = increment_version(current_yaguff)
    yaguff_prompt = f"New YaGUFF version [Enter={Colors.green(suggested_yaguff)}]: "
    yaguff_input = input(yaguff_prompt).strip()
    new_yaguff = yaguff_input if yaguff_input else suggested_yaguff

    # Summary
    print()
    print("=" * 60)
    print(Colors.bold("Summary:"))
    print(f"  YaGUFF:      {current_yaguff} -> {Colors.green(new_yaguff)}")
    print(f"  llama.cpp:   {current_llama} -> {Colors.green(new_llama)}")
    print("=" * 60)
    print()

    # Verify the llama.cpp version exists on GitHub
    print("Verifying llama.cpp version exists on GitHub...")
    version_exists = verify_llama_version_exists(new_llama)
    if not version_exists:
        print(Colors.red(f"ERROR: llama.cpp version '{new_llama}' does not exist on GitHub!"))
        print(Colors.red(f"       Please check https://github.com/ggml-org/llama.cpp/tags"))
        sys.exit(1)
    else:
        print(Colors.green(f"Version {new_llama} verified on GitHub"))

    # Check if going backwards in llama.cpp version
    version_comparison = compare_llama_versions(new_llama, current_llama)
    if version_comparison == -1:
        print()
        print(Colors.red("=" * 60))
        print(Colors.red("WARNING: Going backwards in llama.cpp version!"))
        print(Colors.red(f"         Current: {current_llama}"))
        print(Colors.red(f"         New:     {new_llama}"))
        print(Colors.red("=" * 60))
        print()
    elif version_comparison == 0:
        print(Colors.yellow(f"Note: Keeping llama.cpp version at {new_llama}"))

    print()

    # Prompt for action
    print("What would you like to do?")
    print("  1) Dry run (preview only, no changes)")
    print("  2) Commit and tag (local only)")
    print(f"  3) Commit, tag, and push to origin/{Colors.green(current_branch)}")
    print("  0) Cancel")
    print()

    action = input("Choose [0-3]: ").strip()

    if action == '0' or not action:
        print("Cancelled")
        return

    if action == '1':
        # Dry run
        print()
        print(Colors.yellow("=" * 60))
        print(Colors.yellow("DRY RUN - No changes made"))
        print(Colors.yellow("=" * 60))
        print()
        print(f"Would update binary_manager.py: {current_llama} -> {new_llama}")
        print(f"Would update __init__.py: {current_yaguff} -> {new_yaguff}")
        print(f"Would create commit: 'Bump to v{new_yaguff} (llama.cpp {new_llama})'")
        print(f"Would create tag: v{new_yaguff}")
        if action == '3':
            print(f"Would push to origin/{current_branch}")
        return

    if action not in ['2', '3']:
        print(Colors.red("Invalid choice"))
        return

    # Confirm
    print()
    response = input(f"{Colors.yellow('Proceed?')} [y/N] ")
    if response.lower() != 'y':
        print("Aborted")
        return

    # Update versions
    print()
    update_llama_version(new_llama)
    update_yaguff_version(new_yaguff)

    # Create commit and tag
    tag_name = git_commit_and_tag(new_yaguff, new_llama)

    if not tag_name:
        return

    # Push if requested
    if action == '3':
        print()
        git_push(tag_name, current_branch)

    print()
    print(Colors.green("=" * 60))
    print(Colors.green("Version bump complete!"))
    print(Colors.green("=" * 60))
    print()
    print(f"YaGUFF version: {current_yaguff} -> {Colors.bold(new_yaguff)}")
    print(f"llama.cpp version: {current_llama} -> {Colors.bold(new_llama)}")
    print(f"Git tag: {Colors.bold(f'v{new_yaguff}')}")
    print()

    if action == '2':
        print(Colors.yellow("To push to remote, run:"))
        print(f"  git push origin {current_branch}")
        print(f"  git push origin v{new_yaguff}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
