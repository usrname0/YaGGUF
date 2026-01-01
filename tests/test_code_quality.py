"""
Tests for code quality standards
"""

import re
import platform
from pathlib import Path


def test_no_emojis_in_code():
    """
    Verify that production code contains no emoji characters.

    According to CLAUDE.md: "No emojis or unicode characters in production code"
    """
    # Emoji regex pattern - matches most emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "]+",
        flags=re.UNICODE
    )

    project_root = Path(__file__).parent.parent
    python_files = list(project_root.glob("gguf_converter/**/*.py"))
    python_files.extend(project_root.glob("scripts/**/*.py"))

    violations = []

    for file_path in python_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            matches = emoji_pattern.findall(content)

            if matches:
                # Get line numbers for violations
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if emoji_pattern.search(line):
                        violations.append(f"{file_path.relative_to(project_root)}:{i} contains emoji: {line.strip()[:80]}")
        except Exception as e:
            # Skip files that can't be read
            print(f"Warning: Could not read {file_path}: {e}")
            continue

    assert not violations, f"Found emojis in production code:\n" + "\n".join(violations)


def test_platform_appropriate_paths_in_ui():
    """
    Verify that file paths displayed in the UI use platform-appropriate separators.

    On Windows: Should use backslashes (via str(path))
    On POSIX: Should use forward slashes (via str(path))

    This test checks that we're NOT forcing .as_posix() on all platforms,
    which would show forward slashes on Windows (confusing for users).
    """
    project_root = Path(__file__).parent.parent
    gui_files = list((project_root / "gguf_converter" / "gui_tabs").glob("*.py"))
    gui_files.append(project_root / "gguf_converter" / "gui.py")

    is_windows = platform.system() == "Windows"
    violations = []

    # On Windows, .as_posix() in user-facing messages is wrong
    # On POSIX, forcing backslashes would be wrong (but we don't do that)
    if is_windows:
        # Check for .as_posix() being used in user-facing UI elements
        # This would show forward slashes on Windows, which is confusing
        patterns_to_avoid = [
            (r'st\.code\([^)]*\.as_posix\(\)', "Don't use .as_posix() in UI on Windows - use str(path) for native separators"),
            (r'st\.markdown\(f["\'].*\.as_posix\(\)', "Don't use .as_posix() in markdown on Windows - use str(path)"),
            (r'st\.info\(f["\'].*\.as_posix\(\)', "Don't use .as_posix() in info on Windows - use str(path)"),
            (r'st\.success\(f["\'].*\.as_posix\(\)', "Don't use .as_posix() in success on Windows - use str(path)"),
            (r'st\.warning\(f["\'].*\.as_posix\(\)', "Don't use .as_posix() in warning on Windows - use str(path)"),
            (r'st\.error\(f["\'].*\.as_posix\(\)', "Don't use .as_posix() in error on Windows - use str(path)"),
        ]
    else:
        # On POSIX, check for hardcoded backslashes (shouldn't happen)
        patterns_to_avoid = [
            (r'st\.\w+\(f["\'].*\\\\', "Don't use hardcoded backslashes in UI on POSIX systems"),
        ]

    for file_path in gui_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith('#'):
                    continue

                # Skip if it's clearly not a path (e.g., markdown table separators)
                if '---' in line or ':::' in line:
                    continue

                for pattern, message in patterns_to_avoid:
                    if re.search(pattern, line):
                        violations.append(
                            f"{file_path.relative_to(project_root)}:{i} {message}\n"
                            f"  Line: {line.strip()[:100]}"
                        )
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue

    platform_name = "Windows" if is_windows else "POSIX"
    assert not violations, (
        f"Found platform-inappropriate path formatting on {platform_name}:\n" +
        "\n".join(violations) +
        f"\n\nPaths should use platform-native separators via str(path)"
    )
