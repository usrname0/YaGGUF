"""
Tests for code quality standards
"""

import re
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


def test_posix_paths_in_ui_code():
    """
    Verify that file paths displayed in the UI use forward slashes (POSIX format).

    Checks for patterns like:
    - st.code(str(path), ...)
    - st.markdown(f"...{path}...")
    - st.info(f"...{path}...")

    Should use path.as_posix() instead.
    """
    project_root = Path(__file__).parent.parent
    gui_files = list((project_root / "gguf_converter" / "gui_tabs").glob("*.py"))
    gui_files.append(project_root / "gguf_converter" / "gui.py")

    violations = []

    # Patterns to check for
    patterns_to_check = [
        (r'st\.code\(str\([^)]+\)', "Use .as_posix() instead of str() for path display"),
        (r'st\.markdown\(f["\'].*\{[^}]*_path\}', "Consider using .as_posix() for path in markdown"),
        (r'st\.info\(f["\'].*\{[^}]*_path\}', "Consider using .as_posix() for path in info"),
    ]

    for file_path in gui_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Skip comments
                if line.strip().startswith('#'):
                    continue

                for pattern, message in patterns_to_check:
                    if re.search(pattern, line):
                        # Check if .as_posix() is already used on this line or nearby
                        if '.as_posix()' not in line:
                            violations.append(
                                f"{file_path.relative_to(project_root)}:{i} {message}\n"
                                f"  Line: {line.strip()[:100]}"
                            )
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue

    assert not violations, f"Found POSIX path issues - use .as_posix() for path display:\n" + "\n".join(violations)
