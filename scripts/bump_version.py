#!/usr/bin/env python3
"""
Interactive script to bump version numbers for YaGGUF releases

Workflow:
1. Shows recent version history
2. Prompts for new llama.cpp version (default: latest from GitHub)
3. Prompts for new YaGGUF version (default: auto-increment patch)
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


def get_current_yagguf_version():
    """Read current version from __init__.py"""
    init_file = get_project_root() / "gguf_converter" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    return None


def get_current_llama_version():
    """Read current llama.cpp version from llama_cpp_manager.py"""
    llama_cpp_manager = get_project_root() / "gguf_converter" / "llama_cpp_manager.py"
    content = llama_cpp_manager.read_text()
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
    Get recent YaGGUF version tags from git

    Args:
        count: Number of recent versions to retrieve

    Returns:
        List of (tag, yagguf_version, llama_version) tuples
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

            # Extract YaGGUF version from tag
            yagguf_version = tag[1:] if tag.startswith('v') else tag

            versions.append((tag, yagguf_version, llama_version))

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
    """Update LLAMA_CPP_VERSION in llama_cpp_manager.py"""
    llama_cpp_manager = get_project_root() / "gguf_converter" / "llama_cpp_manager.py"
    content = llama_cpp_manager.read_text()

    # Replace LLAMA_CPP_VERSION
    new_content = re.sub(
        r'LLAMA_CPP_VERSION\s*=\s*["\'][^"\']+["\']',
        f'LLAMA_CPP_VERSION = "{new_llama_version}"',
        content
    )

    llama_cpp_manager.write_text(new_content)
    print(Colors.green(f"Updated LLAMA_CPP_VERSION to {new_llama_version}"))


def update_yagguf_version(new_version):
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
    print(Colors.green(f"Updated YaGGUF version to {new_version}"))


def git_commit_and_tag(yagguf_version, llama_version):
    """Create git commit and tag"""
    try:
        # Stage the changed files
        subprocess.run(
            ["git", "add", "gguf_converter/__init__.py", "gguf_converter/llama_cpp_manager.py"],
            check=True,
            cwd=get_project_root()
        )

        # Create commit
        commit_msg = f"Bump to v{yagguf_version} (llama.cpp {llama_version})"
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            cwd=get_project_root()
        )
        print(Colors.green(f"Created commit: {commit_msg}"))

        # Create tag
        tag_name = f"v{yagguf_version}"
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


def get_venv_python():
    """
    Get path to venv Python interpreter

    Returns:
        Path to venv python or None if not found
    """
    project_root = get_project_root()
    venv_path = project_root / "venv"

    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if python_exe.exists():
        return python_exe
    return None


def run_tests():
    """
    Run test suite before version bump using venv Python

    Returns:
        True if tests pass, False otherwise
    """
    print()
    print("=" * 60)
    print("Running tests...")
    print("=" * 60)
    print()

    # Try to use venv python, fallback to system python
    venv_python = get_venv_python()
    if venv_python:
        python_cmd = str(venv_python)
        print(f"Using venv Python: {venv_python}")
    else:
        python_cmd = sys.executable
        print(Colors.yellow(f"Warning: venv not found, using {python_cmd}"))

    try:
        result = subprocess.run(
            [python_cmd, "-m", "pytest", "tests/", "-v"],
            cwd=get_project_root(),
            capture_output=False
        )

        if result.returncode == 0:
            print()
            print(Colors.green("All tests passed!"))
            return True
        else:
            print()
            print(Colors.red("Tests failed!"))
            print(Colors.red("Please fix failing tests before bumping version."))
            return False

    except FileNotFoundError:
        print(Colors.yellow("Warning: pytest not found, skipping tests"))
        response = input("Continue without running tests? [y/N] ")
        return response.lower() == 'y'


def main():
    parser = argparse.ArgumentParser(
        description="Bump version numbers for YaGGUF releases (interactive mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args = parser.parse_args()

    # Run tests first
    if not run_tests():
        sys.exit(1)

    # Get current branch
    current_branch = get_current_branch()
    if current_branch:
        print()
        print("=" * 60)
        print(f"Current branch: {Colors.blue(current_branch)}")
        print("=" * 60)
    else:
        print(Colors.yellow("Warning: Could not detect git branch"))

    # Show recent versions
    print()
    print(Colors.bold("Recent YaGGUF Versions:"))
    print("-" * 60)

    recent = get_recent_versions(3)
    if recent:
        for tag, yagguf_ver, llama_ver in recent:
            print(f"  {Colors.blue(tag):12} - YaGGUF {yagguf_ver:8} - llama.cpp {llama_ver}")
    else:
        print(Colors.yellow("  No version tags found"))

    # Get current versions
    current_yagguf = get_current_yagguf_version()
    current_llama = get_current_llama_version()

    print()
    print(Colors.bold("Current versions:"))
    print(f"  YaGGUF:      {Colors.green(current_yagguf)}")
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

    # Prompt for YaGGUF version
    suggested_yagguf = increment_version(current_yagguf)
    yagguf_prompt = f"New YaGGUF version [Enter={Colors.green(suggested_yagguf)}]: "
    yagguf_input = input(yagguf_prompt).strip()
    new_yagguf = yagguf_input if yagguf_input else suggested_yagguf

    # Summary
    print()
    print("=" * 60)
    print(Colors.bold("Summary:"))
    print(f"  YaGGUF:      {current_yagguf} -> {Colors.green(new_yagguf)}")
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
        print(f"Would update llama_cpp_manager.py: {current_llama} -> {new_llama}")
        print(f"Would update __init__.py: {current_yagguf} -> {new_yagguf}")
        print(f"Would create commit: 'Bump to v{new_yagguf} (llama.cpp {new_llama})'")
        print(f"Would create tag: v{new_yagguf}")
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
    update_yagguf_version(new_yagguf)

    # Create commit and tag
    tag_name = git_commit_and_tag(new_yagguf, new_llama)

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
    print(f"YaGGUF version: {current_yagguf} -> {Colors.bold(new_yagguf)}")
    print(f"llama.cpp version: {current_llama} -> {Colors.bold(new_llama)}")
    print(f"Git tag: {Colors.bold(f'v{new_yagguf}')}")
    print()

    if action == '2':
        print(Colors.yellow("To push to remote, run:"))
        print(f"  git push origin {current_branch}")
        print(f"  git push origin v{new_yagguf}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
