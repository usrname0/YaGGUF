# Code Quality Tests

## Overview

The `test_code_quality.py` file contains automated checks for code quality standards defined in CLAUDE.md.

## Tests

### 1. `test_no_emojis_in_code`

**Purpose**: Enforce the "No emojis or unicode characters in production code" rule from CLAUDE.md

**What it checks**:
- Scans all Python files in `gguf_converter/` and `scripts/`
- Detects emoji characters using Unicode ranges
- Reports file path and line number for any violations

**Why**: Emojis in code can cause encoding issues and make code less professional.

**Example violation**:
```python
# Bad
print("✅ Success!")

# Good
print("Success!")
```

### 2. `test_platform_appropriate_paths_in_ui`

**Purpose**: Ensure file paths displayed in the UI use platform-native separators

**Platform-specific behavior**:

#### On Windows
- **Expects**: Backslashes (`C:\Users\...`)
- **Checks for**: Inappropriate use of `.as_posix()` in UI elements
- **Why**: Windows users expect to see `C:\Users\...`, not `C:/Users/...`

**Example violations on Windows**:
```python
# Bad - shows forward slashes on Windows
st.info(f"Path: {path.as_posix()}")

# Good - shows backslashes on Windows
st.info(f"Path: {path}")
```

#### On POSIX (Linux/Mac)
- **Expects**: Forward slashes (`/home/user/...`)
- **Checks for**: Hardcoded backslashes in UI elements
- **Why**: POSIX users expect forward slashes

**Example violations on POSIX**:
```python
# Bad - hardcoded backslashes
st.info(f"Path: C:\\Users\\...")

# Good - uses Path objects
st.info(f"Path: {path}")
```

### What the test catches

#### Streamlit UI elements checked:
- `st.code()` - Code display blocks
- `st.markdown()` - Markdown text
- `st.info()` - Info messages
- `st.success()` - Success messages
- `st.warning()` - Warning messages
- `st.error()` - Error messages

#### What's allowed:
- `str(path)` - Automatically uses platform-native separators
- Direct path variables - They use native separators
- Comments - Ignored by the test

#### What's flagged on Windows:
- `.as_posix()` in user-facing UI elements
- Forces forward slashes when backslashes are expected

#### What's flagged on POSIX:
- Hardcoded backslashes in UI elements
- Would show Windows-style paths on Linux/Mac

## Why Platform-Appropriate Paths Matter

### User Experience
Users expect to see paths in their platform's native format:
- **Windows users** are familiar with `C:\Users\...`
- **Linux/Mac users** are familiar with `/home/user/...`

Showing the wrong separator is confusing and looks unprofessional.

### Copy-Paste Behavior
Users often copy paths from the UI:
- On Windows, copying `C:\Users\...` works directly in File Explorer
- Copying `C:/Users/...` may not work in some Windows tools
- On Linux/Mac, `/home/user/...` is the standard

### Platform Consistency
The OS file browsers show:
- Windows Explorer: `C:\Users\Documents\model.gguf`
- Mac Finder: `/Users/username/Documents/model.gguf`
- Linux file managers: `/home/username/Documents/model.gguf`

Our UI should match what users see in their file browser.

## How to Fix Violations

### Windows-specific fixes

#### Before (wrong - shows forward slashes):
```python
st.code(str(file_path.as_posix()), language=None)
st.markdown(f"Path: {model_path.as_posix()}")
st.info(f"Saved to: {output_path.as_posix()}")
```

#### After (correct - shows backslashes on Windows):
```python
st.code(str(file_path), language=None)
st.markdown(f"Path: {model_path}")
st.info(f"Saved to: {output_path}")
```

### POSIX-specific fixes

#### Before (wrong - hardcoded backslashes):
```python
st.info(f"Path: C:\\Users\\model.gguf")
```

#### After (correct - uses Path):
```python
from pathlib import Path
model_path = Path("C:/Users/model.gguf")  # Path handles conversion
st.info(f"Path: {model_path}")
```

## When `.as_posix()` IS Appropriate

Use `.as_posix()` for:
1. **Log files** - Machine-readable logs often prefer forward slashes
2. **URLs** - Must always use forward slashes
3. **Configuration files** - Cross-platform config may prefer forward slashes
4. **Internal processing** - When you need consistent path format for comparison

**Don't use** `.as_posix()` for:
- User-facing messages in the UI
- Error messages shown to users
- Status displays
- File path displays in the GUI

## Running the Tests

```bash
# Run both code quality tests
pytest tests/test_code_quality.py -v

# Run just the emoji test
pytest tests/test_code_quality.py::test_no_emojis_in_code -v

# Run just the path test
pytest tests/test_code_quality.py::test_platform_appropriate_paths_in_ui -v
```

## Test Implementation Details

### Skipped Patterns
The test intelligently skips:
- Comment lines (starting with `#`)
- Markdown table separators (`---`, `:::`)
- Non-path content

### Regular Expression Patterns
The test uses regex to detect:
- `.as_posix()` calls in Streamlit functions (Windows)
- Escaped backslashes in f-strings (POSIX)

### Files Checked
- `gguf_converter/gui_tabs/*.py` - All GUI tab files
- `gguf_converter/gui.py` - Main GUI file

## Cross-Platform Testing

This test works on all platforms:
- **Windows**: Checks for `.as_posix()` abuse
- **Linux**: Checks for hardcoded backslashes
- **Mac**: Checks for hardcoded backslashes

CI/CD pipelines on different platforms will catch platform-specific issues.

## Contributing

When adding new UI code:
1. Use `str(path)` or just `{path}` in f-strings for user-facing displays
2. Avoid `.as_posix()` in Streamlit UI elements
3. Run `pytest tests/test_code_quality.py` to verify
4. Test on your platform to see actual separator behavior

## Examples

### Good Examples ✅

```python
# Display path in UI
model_path = Path("D:/models/llama.gguf")
st.success(f"Model loaded: {model_path}")  # Shows D:\models\llama.gguf on Windows

# File path in code block
output_file = Path("output/model.gguf")
st.code(str(output_file))  # Shows output\model.gguf on Windows

# Multiple paths
st.markdown(f"""
**Input**: {input_path}
**Output**: {output_path}
""")  # All use native separators
```

### Bad Examples ❌

```python
# Wrong - forces forward slashes on Windows
model_path = Path("D:/models/llama.gguf")
st.success(f"Model loaded: {model_path.as_posix()}")  # Shows D:/models/llama.gguf (confusing!)

# Wrong - hardcoded backslashes
st.info(f"Path: C:\\Users\\model.gguf")  # Won't work on Linux/Mac

# Wrong - .as_posix() in code display
st.code(str(file_path.as_posix()))  # Shows forward slashes on Windows
```

## FAQ

**Q: Why not always use `.as_posix()` for consistency?**
A: Users expect to see paths in their platform's native format. Showing forward slashes on Windows is confusing.

**Q: What if I need forward slashes for a URL?**
A: URLs are fine - use `.as_posix()` for URLs. This test only checks user-facing file path displays.

**Q: Does `str(path)` automatically use the right separator?**
A: Yes! `Path` objects automatically use backslashes on Windows and forward slashes on POSIX when converted to strings.

**Q: What about paths in error messages?**
A: Error messages shown to users should use native separators (`str(path)`), not `.as_posix()`.

**Q: Can I use `.as_posix()` in non-GUI code?**
A: Yes, the test only checks Streamlit UI elements. Use `.as_posix()` freely in internal processing, logs, etc.
