# llama.cpp Smoke Tests

## Overview

The smoke tests in `test_llama_cpp_smoke.py` verify that all llama.cpp binaries and conversion scripts work correctly. These tests are designed to catch breaking changes when updating llama.cpp to a new version.

## What Gets Tested

### Binaries
- **llama-quantize** - Model quantization tool
- **llama-imatrix** - Importance matrix generation tool
- **llama-cli** - Command-line interface for model inference

### Conversion Scripts
- **convert_hf_to_gguf.py** - Convert HuggingFace models to GGUF format (only script used by this project)

### Test Categories

1. **Binary Execution Tests**
   - Verify binaries can be invoked with --help
   - Check that --version flags work
   - Ensure help text is displayed

2. **Conversion Script Tests**
   - Test that scripts can be invoked with --help
   - Verify scripts have no syntax or import errors

3. **Version Detection Tests**
   - Check that version information can be extracted from binaries
   - Verify LlamaCppManager can detect installed versions

4. **Error Handling Tests**
   - Ensure binaries fail gracefully when called with missing arguments
   - Verify error messages are displayed

## Running the Tests

### Run all smoke tests
```bash
pytest tests/test_llama_cpp_smoke.py -v
```

### Run only tests marked as requires_binaries
```bash
pytest tests/test_llama_cpp_smoke.py -v -m requires_binaries
```

### Run a specific test class
```bash
# Test only binaries
pytest tests/test_llama_cpp_smoke.py::TestLlamaCppBinaries -v

# Test only conversion scripts
pytest tests/test_llama_cpp_smoke.py::TestConversionScripts -v

# Test only version detection
pytest tests/test_llama_cpp_smoke.py::TestBinaryVersions -v
```

### Run a specific test
```bash
pytest tests/test_llama_cpp_smoke.py::TestLlamaCppBinaries::test_llama_quantize_help -v
```

## When to Run These Tests

### After Updating llama.cpp
Run these tests immediately after updating llama.cpp binaries or conversion scripts:

```bash
# Update binaries (in GUI or via script)
# Then run smoke tests
pytest tests/test_llama_cpp_smoke.py -v
```

If any test fails, it indicates a breaking change in llama.cpp that needs investigation.

### During Development
Run these tests when making changes to:
- `llama_cpp_manager.py` - Binary path resolution or version detection
- Binary download/update logic
- Conversion workflow

### In CI/CD
Include these tests in your continuous integration pipeline to catch regressions early.

## Understanding Test Results

### All Tests Pass
All llama.cpp binaries and scripts are working correctly.

### Some Tests Skipped
Tests may be skipped if:
- Binaries are not installed (run setup first)
- llama.cpp directory is not cloned
- Specific scripts don't exist in your llama.cpp version

### Test Failures
If a test fails, check:
1. **Binary not found** - Run setup to download binaries
2. **Version incompatibility** - llama.cpp may have changed command-line interface
3. **Missing dependencies** - Conversion scripts may need additional Python packages
4. **Breaking changes** - llama.cpp may have removed or renamed flags

## Test Behavior Notes

### llama-quantize Quirks
- `llama-quantize --help` returns exit code 1 (not 0) but still outputs help text
- `llama-quantize` doesn't support `--version` flag
- Tests are designed to handle these quirks

### Conversion Scripts
- Some scripts may not support `--help` flag
- Tests verify scripts can at least be executed without syntax errors
- Import errors are checked to ensure dependencies are available

## Adding New Tests

When llama.cpp adds new binaries or scripts, add corresponding smoke tests:

```python
def test_new_binary_help(self, llama_manager):
    """
    Verify new-binary can be invoked with --help
    """
    binary_path = llama_manager.get_binary_path('new-binary')

    if not binary_path.exists():
        pytest.skip(f"new-binary not found at {binary_path}")

    result = subprocess.run(
        [str(binary_path), "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "usage:" in output.lower()
```

## Troubleshooting

### Tests timeout
Increase timeout value in subprocess.run() calls:
```python
timeout=30  # Increase from 10 seconds
```

### Tests fail on specific platform
Add platform-specific handling:
```python
import platform
if platform.system() == "Windows":
    # Windows-specific test behavior
```

### Version detection fails
Check that binaries actually support --version flag. Some tools may not.
