# YaGGUF Test Suite

Complete test suite for Yet Another GGUF Converter.

## Test Overview

### Test Statistics
- **Total Tests**: 135
- **Fast Tests**: 122 (run by default)
- **Integration Tests**: 13 (require model download, run explicitly)

### Test Files

| File | Tests | Type | Description |
|------|-------|------|-------------|
| `test_llama_cpp_smoke.py` | 14 | Smoke | Test llama.cpp binaries and scripts |
| `test_integration.py` | 13 | Integration | End-to-end tests with real model |
| `test_llama_cpp_manager.py` | 22 | Unit | Test binary manager functionality |
| `test_imatrix_stats.py` | 26 | Unit | Test imatrix statistics |
| `test_quantization_validation.py` | 16 | Validation | Verify quantization types |
| `test_gui_utils.py` | 12 | Unit | Test GUI utility functions |
| `test_version_utils.py` | 11 | Unit | Test version checking |
| `test_config.py` | 7 | Unit | Test configuration management |
| `test_path_utils.py` | 12 | Unit | Test path utilities |
| `test_code_quality.py` | 2 | Quality | Code quality checks |

## Quick Start

### Run all fast tests (recommended)
```bash
pytest
```

### Run specific test categories
```bash
# Smoke tests - verify binaries work (5-10 seconds)
pytest tests/test_llama_cpp_smoke.py -v

# Unit tests only (instant)
pytest tests/test_llama_cpp_manager.py -v
pytest tests/test_config.py -v

# Integration tests - full workflow (10-15 minutes)
pytest -m integration -v
```

### Skip slow tests
```bash
# Skip both integration and slow tests
pytest -m "not integration and not slow"
```

## Test Types

### 1. Smoke Tests (14 tests, ~5-10 seconds)
**File**: `test_llama_cpp_smoke.py`

Verify llama.cpp binaries and scripts can be executed.

**What's Tested**:
- Binary execution (llama-quantize, llama-imatrix, llama-cli)
- Conversion script (convert_hf_to_gguf.py)
- Version detection
- Error handling

**When to Run**: After updating llama.cpp

```bash
pytest tests/test_llama_cpp_smoke.py -v
```

See: [README_SMOKE_TESTS.md](README_SMOKE_TESTS.md)

### 2. Integration Tests (13 tests, ~10-15 minutes)
**File**: `test_integration.py`

End-to-end testing with real model (Qwen2.5-0.5B-Instruct).

**What's Tested**:
- Model download from HuggingFace
- GGUF conversion (F16, BF16, F32)
- Quantization (Q4_K_M, Q5_K_M, Q8_0, IQ3_XXS)
- Overwrite vs reuse behavior
- Multiple quantizations
- Imatrix generation and usage
- Edge cases

**Requirements**:
- 2-3GB disk space
- Internet connection (first run)
- 10-15 minutes time

**When to Run**: Before releases, after major changes

```bash
# Run all integration tests
pytest -m integration -v

# Run specific categories
pytest tests/test_integration.py::TestBasicConversion -v
pytest tests/test_integration.py::TestQuantization -v
```

See: [README_INTEGRATION_TESTS.md](README_INTEGRATION_TESTS.md)

### 3. Unit Tests (95 tests, instant)

Fast tests that don't require external resources.

**Categories**:
- **llama_cpp_manager** (22 tests) - Binary path resolution, platform detection
- **imatrix_stats** (26 tests) - Statistics calculation
- **quantization_validation** (16 tests) - Quantization type definitions
- **gui_utils** (12 tests) - GUI utility functions
- **version_utils** (11 tests) - Version checking
- **config** (7 tests) - Configuration management
- **path_utils** (12 tests) - Path handling
- **code_quality** (2 tests) - Code standards (no emojis, platform-appropriate paths)

```bash
# Run all unit tests
pytest -m "not integration and not slow"

# Run specific unit tests
pytest tests/test_llama_cpp_manager.py -v
pytest tests/test_config.py -v
```

## Common Commands

### Development Workflow
```bash
# Quick check while developing
pytest tests/test_llama_cpp_manager.py

# Run fast tests before commit
pytest -m "not integration"

# Full test suite before release
pytest && pytest -m integration
```

### CI/CD
```bash
# Fast tests in CI (every commit)
pytest -m "not integration and not slow"

# Nightly integration tests
pytest -m integration --tb=short
```

### Debugging
```bash
# Run single test with full output
pytest tests/test_integration.py::TestBasicConversion::test_convert_to_f16 -v -s

# Show local variables on failure
pytest --tb=long --showlocals

# Stop on first failure
pytest -x
```

## Test Markers

Tests can be marked with pytest markers:

- `@pytest.mark.integration` - Integration tests requiring model download
- `@pytest.mark.slow` - Slow tests (can be skipped)
- `@pytest.mark.requires_binaries` - Requires llama.cpp binaries installed

### Using Markers
```bash
# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"

# Run tests requiring binaries
pytest -m requires_binaries

# Complex marker expressions
pytest -m "not integration and not slow"
```

## Configuration

Test configuration is in `pytest.ini`:
- Test discovery limited to `tests/` directory
- Ignores `llama.cpp/` subdirectory tests
- Verbose output by default
- Strict marker enforcement

## Expected Test Times

| Test Type | First Run | Subsequent Runs | Notes |
|-----------|-----------|-----------------|-------|
| Unit Tests | <5s | <5s | Always fast |
| Smoke Tests | 5-10s | 5-10s | Invokes binaries |
| Integration (all) | 10-15m | 5-10m | Model download cached |
| Integration (single) | 30s-2m | 30s-2m | After model downloaded |

## Test Results

### All Pass ✅
All tests passed - system working correctly.

### Some Skip ⏭️
Expected skips:
- Integration tests (when using `-m "not integration"`)
- Imatrix tests (when calibration file missing)
- Binary tests (when binaries not installed)

### Failures ❌
If tests fail:
1. Check error message
2. Ensure dependencies installed: `pip install -r requirements.txt`
3. Ensure binaries installed: Run GUI or `python scripts/check_and_download_binaries.py`
4. Check disk space (integration tests need 2-3GB)
5. See specific test documentation for troubleshooting

## Best Practices

### During Development
1. **Run unit tests frequently** - They're instant
2. **Run smoke tests before commits** - Catch binary issues
3. **Run integration tests before PRs** - Ensure no regressions

### Before Releases
1. **Run all unit tests**: `pytest -m "not integration"`
2. **Run smoke tests**: `pytest tests/test_llama_cpp_smoke.py`
3. **Run integration tests**: `pytest -m integration`
4. **Verify all pass** before tagging release

### Adding New Tests
- **Unit tests** for pure Python logic
- **Smoke tests** for new binaries/scripts
- **Integration tests** for workflow changes
- Mark appropriately (`@pytest.mark.integration`, etc.)
- Update this README with new test counts

## Troubleshooting

### "No tests collected"
```bash
# Check test discovery
pytest --collect-only

# Ensure in project root
cd D:\Dev\YaGGUF
pytest
```

### "ModuleNotFoundError"
```bash
# Install project in development mode
pip install -e .

# Or ensure in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Integration tests timeout
```bash
# Increase timeout in test file
# Or run specific test class instead of all
pytest tests/test_integration.py::TestBasicConversion -v
```

### llama.cpp tests interfere
```bash
# Ensure pytest.ini exists in project root
# It configures testpaths = tests
ls pytest.ini
```

## Contributing

When adding new features:
1. Add unit tests for the feature
2. Add integration test if it changes workflow
3. Update test counts in this README
4. Ensure all existing tests still pass

## Support

For issues:
- Check specific test README files
- Review test output carefully
- Check GitHub issues
- Verify system requirements met
