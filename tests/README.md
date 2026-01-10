# YaGGUF Test Suite

Test suite for Yet Another GGUF Converter.

## Quick Start

```bash
# Run all fast tests (recommended for development)
pytest

# Run everything including integration tests (before releases)
pytest && pytest -m integration
```

## Test Overview

**Total Tests**: ~200
- Fast tests: ~185 (unit, smoke, validation)
- Integration tests: 13 (require model download)

### Test Files

| File | Tests | Time | Description |
|------|-------|------|-------------|
| `test_split_merge.py` | 33 | <3s | Shard analysis, splitting, merging, and resplitting (GGUF & Safetensors) |
| `test_model_quirks.py` | 17 | <1s | Model quirk detection (Mistral, Vision, Sentence-Transformers) |
| `test_split_file_handling.py` | 20 | <1s | Core shard validation utilities |
| `test_llama_cpp_manager.py` | 22 | <1s | Binary manager functionality |
| `test_imatrix_stats.py` | 26 | <1s | Imatrix statistics |
| `test_quantization_validation.py` | 16 | <1s | Quantization type validation |
| `test_llama_cpp_smoke.py` | 14 | 5-10s | Binary execution tests |
| `test_gui_utils.py` | 12 | <1s | GUI utility functions |
| `test_version_utils.py` | 11 | <1s | Version checking |
| `test_config.py` | 7 | <1s | Configuration management |
| `test_path_utils.py` | 12 | <1s | Path utilities |
| `test_code_quality.py` | 2 | <1s | Code standards |
| `test_integration.py` | 13 | 10-15m | End-to-end workflow |
| `test_source_dtype_detection.py` | 12 | <1s | Source datatype detection |
| `test_intermediate_detection.py` | 8 | <1s | Intermediate file detection |

## Common Commands

### Development
```bash
# Quick check while developing
pytest

# Run specific test file
pytest tests/test_split_merge.py -v

# Stop on first failure
pytest -x
```

### Before Commits
```bash
# All fast tests
pytest -m "not integration"

# Smoke tests only (verify binaries work)
pytest tests/test_llama_cpp_smoke.py -v
```

### Before Releases
```bash
# Fast tests
pytest

# Integration tests
pytest -m integration -v

# Keep integration outputs for inspection
pytest -m integration --test-output-dir ./test_outputs -v
```

### Debugging
```bash
# Single test with full output
pytest tests/test_integration.py::TestBasicConversion::test_convert_to_f16 -v -s

# Show local variables on failure
pytest --tb=long --showlocals

# Show what would run
pytest --collect-only
```

## Test Types

### Unit Tests (~180 tests, <10 seconds)
Fast tests of individual components:
- Split/Merge functionality (GGUF & Safetensors)
- Model quirks detection
- Shard validation
- Binary path resolution
- Configuration management
- Path utilities
- Statistics calculation

```bash
pytest -m "not integration and not slow"
```

### Smoke Tests (14 tests, 5-10 seconds)
Verify llama.cpp binaries execute correctly:
- llama-quantize --help
- llama-imatrix --help
- llama-cli --help
- convert_hf_to_gguf.py --help

Run after updating llama.cpp:
```bash
pytest tests/test_llama_cpp_smoke.py -v
```

### Integration Tests (13 tests, 10-15 minutes)
End-to-end workflow with real model (Qwen2.5-0.5B-Instruct):
- Model download from HuggingFace
- GGUF conversion (F16, BF16, F32)
- Quantization (Q4_K_M, Q5_K_M, Q8_0, IQ3_XXS)
- Overwrite vs reuse behavior
- Multiple quantizations in one run
- Imatrix generation and usage
- Functional correctness checksum validation 

### Note about test_q4_k_m_checksum
- Delete tests/golden_outputs/qwen2.5_0.5b_q4_k_m_checksum.txt to reset
- Note: first run will generate checksum / fail to compare

Requirements:
- 2-3GB disk space
- Internet connection (first run only)
- Calibration file for imatrix tests (optional)

```bash
# Run all integration tests
pytest -m integration -v

# Run specific test class
pytest tests/test_integration.py::TestBasicConversion -v

# Skip slow imatrix tests
pytest -m integration -k "not imatrix" -v
```

## Test Markers

Tests use pytest markers for categorization:

```bash
# Run only integration tests
pytest -m integration

# Skip integration tests
pytest -m "not integration"

# Skip slow tests
pytest -m "not slow"

# Run tests requiring binaries
pytest -m requires_binaries
```

## Integration Test Options

### Keep Test Outputs
```bash
# Keep in pytest temp directory
pytest -m integration --keep-test-outputs

# Save to specific directory
pytest -m integration --test-output-dir ./test_outputs
```

### Run Specific Tests
```bash
# Basic conversion only (fastest)
pytest tests/test_integration.py::TestBasicConversion -v

# Quantization tests
pytest tests/test_integration.py::TestQuantization -v

# Overwrite behavior
pytest tests/test_integration.py::TestOverwriteBehavior -v

# Multiple quantizations
pytest tests/test_integration.py::TestMultipleQuantizations -v

# Imatrix generation (requires calibration file)
pytest tests/test_integration.py::TestImatrixGeneration -v

# Functional correctness (checksum validation)
pytest tests/test_integration.py::TestFunctionalCorrectness -v
```

## Checksum Test

The functional correctness test validates quantized model output:

- Generates text from Q4_K_M model with fixed seed (temp=0.0)
- Compares SHA256 checksum against per-system baseline
- Catches broken quantization or corrupted models
- First run creates baseline, subsequent runs validate against it

**Per-system baseline:**
- Checksum file is NOT committed to git (in .gitignore)
- Each system/developer establishes their own baseline
- Detects regressions on that specific system
- Will differ across llama.cpp versions, platforms, or CPU architectures

**First run on a system:**
```bash
# First run creates the baseline and fails (expected)
pytest tests/test_integration.py::TestFunctionalCorrectness::test_q4_k_m_checksum -v -s

# Subsequent runs validate against this baseline
pytest tests/test_integration.py::TestFunctionalCorrectness::test_q4_k_m_checksum -v
```

**Reset baseline (after llama.cpp update):**
```bash
# Delete old checksum to regenerate
rm tests/golden_outputs/qwen2.5_0.5b_q4_k_m_checksum.txt

# Run test to create new baseline
pytest tests/test_integration.py::TestFunctionalCorrectness::test_q4_k_m_checksum -v -s
```

**When checksum fails:**
1. Inspect generated text in test output (run with `-s` flag)
2. If text looks reasonable, llama.cpp likely updated - reset baseline
3. If text is garbage/empty, investigate quantization issue

## Expected Behavior

### All Pass
System working correctly.

### Some Skip
Expected when:
- Integration tests skipped (using `-m "not integration"`)
- Calibration file missing (imatrix tests)
- Binaries not installed (smoke tests)

### Failures
Check:
1. Error message for specific issue
2. Dependencies installed: `pip install -r requirements.txt`
3. Binaries installed: Run GUI or `python scripts/check_and_download_binaries.py`
4. Disk space (integration tests need 2-3GB)

## Troubleshooting

### No tests collected
```bash
# Check test discovery
pytest --collect-only

# Ensure in project root
cd D:\Dev\YaGGUF
pytest
```

### ModuleNotFoundError
```bash
# Install in development mode
pip install -e .
```

### Integration tests timeout
Normal on slower systems. Run specific test classes instead:
```bash
pytest tests/test_integration.py::TestBasicConversion -v
```

### Calibration file missing
```bash
# Download calibration file to calibration_data/ directory
# Or skip imatrix tests
pytest -m integration -k "not imatrix"
```

## Best Practices

### During Development
- Run unit tests frequently (instant feedback)
- Run smoke tests before commits (catch binary issues)
- Run relevant integration tests for major changes

### Before Releases
1. Run all unit tests: `pytest -m "not integration"`
2. Run smoke tests: `pytest tests/test_llama_cpp_smoke.py`
3. Run integration tests: `pytest -m integration`
4. Verify all pass before tagging

### After llama.cpp Updates
1. Run smoke tests first (catch breaking changes)
2. If smoke tests pass, run integration tests
3. Update golden checksum if needed

### Adding New Tests
- Unit tests for pure Python logic
- Smoke tests for new binaries/scripts
- Integration tests for workflow changes
- Mark appropriately with `@pytest.mark.integration`, etc.
- Update test counts in this README

## Configuration

Test configuration in `pytest.ini`:
- Test discovery limited to `tests/` directory
- Ignores `llama.cpp/` subdirectory tests
- Verbose output by default
- Strict marker enforcement