# Integration Tests

## Overview

The integration tests in `test_integration.py` perform end-to-end testing of the entire conversion workflow using a real small model (Qwen2.5-0.5B-Instruct, ~500MB).

## What Gets Tested

### Full Workflow Tests
- **Model Download** - Download from HuggingFace
- **GGUF Conversion** - Convert HuggingFace model to GGUF format
- **Quantization** - Test various quantization types
- **Overwrite Behavior** - Test file reuse vs regeneration
- **Multiple Quantizations** - Create multiple quant types in one run
- **Importance Matrix** - Generate and use imatrix for better quality
- **Edge Cases** - Invalid inputs, different intermediate formats

### Specific Test Coverage

#### Quantization Types Tested
- Q4_K_M - Most common quantization
- Q5_K_M - Medium quality
- Q8_0 - High quality
- IQ3_XXS - Low-bit with importance matrix

#### Intermediate Formats Tested
- F16 (default)
- BF16
- F32

#### Settings Tested
- `overwrite_intermediates=True/False`
- `overwrite_quants=True/False`
- Multiple quantization types in single run
- Imatrix generation and usage

## Requirements

### Disk Space
Integration tests require significant disk space:
- Model download: ~500MB
- F16 conversion: ~1GB
- Quantizations: ~200-500MB each
- Imatrix files: ~50-100MB
- **Total: 2-3GB recommended**

### Time
- First run: 5-15 minutes (includes model download)
- Subsequent runs: 2-5 minutes (model cached)
- Individual test classes: 30 seconds - 2 minutes each

### Network
- Tests download model from HuggingFace on first run
- Model is cached in temp directory for subsequent tests
- Requires active internet connection for first run

### Calibration Data (for imatrix tests)
- Imatrix tests require a calibration file
- Place `.txt` files in `calibration_data/` directory
- Tests skip gracefully if no calibration file found

## Running Integration Tests

### Skip integration tests (default recommended)
```bash
# Run all tests except integration
pytest -m "not integration"

# Run only smoke tests (fast)
pytest tests/test_llama_cpp_smoke.py -v
```

### Run integration tests
```bash
# Run ALL integration tests (may take 10-15 minutes)
pytest tests/test_integration.py -v -m integration

# Run with output to see progress
pytest tests/test_integration.py -v -s -m integration
```

### Keep test outputs (inspect files after tests)
```bash
# Keep files in pytest temp directory
pytest -m integration --keep-test-outputs -v

# Save to specific directory
pytest -m integration --test-output-dir D:/test_outputs -v

# Combine with -s to see where files are saved
pytest -m integration --test-output-dir ./test_outputs -v -s
```

### Run specific test classes
```bash
# Test only basic conversion
pytest tests/test_integration.py::TestBasicConversion -v

# Test only quantization
pytest tests/test_integration.py::TestQuantization -v

# Test only overwrite behavior
pytest tests/test_integration.py::TestOverwriteBehavior -v

# Test multiple quantizations
pytest tests/test_integration.py::TestMultipleQuantizations -v

# Test imatrix (requires calibration file)
pytest tests/test_integration.py::TestImatrixGeneration -v

# Test edge cases
pytest tests/test_integration.py::TestEdgeCases -v
```

### Run specific tests
```bash
# Test F16 conversion only
pytest tests/test_integration.py::TestBasicConversion::test_convert_to_f16 -v

# Test Q4_K_M quantization only
pytest tests/test_integration.py::TestQuantization::test_quantize_q4_k_m -v

# Test overwrite behavior
pytest tests/test_integration.py::TestOverwriteBehavior::test_overwrite_intermediates_false -v
```

## Test Structure

### Module-Scoped Fixtures
Integration tests use module-scoped fixtures to optimize performance:

- **downloaded_model** - Model downloaded once, reused by all tests
- **test_output_dir** - Temp directory created once, cleaned up after all tests
- **converter** - Single GGUFConverter instance reused

This means:
1. Model is only downloaded once per test run
2. Tests share the same downloaded model
3. Each test class gets its own output subdirectory
4. All files cleaned up automatically after tests complete

## Expected Results

### All Tests Pass
All conversion workflows work correctly with the test model.

### Some Tests Skipped
Tests may skip if:
- **Calibration file missing** - Imatrix tests skip gracefully
- **Binaries not installed** - Run setup first
- **Disk space insufficient** - Free up space

### Test Failures
If tests fail, check:
1. **Disk space** - Need 2-3GB available
2. **Network connection** - Required for model download
3. **llama.cpp version** - Update may have breaking changes
4. **Dependencies** - Ensure all Python packages installed

## Interpreting Test Output

### Successful Test
```
test_convert_to_f16 PASSED
```
- Model converted successfully
- Output file created and validated
- File size within expected range

### Test with Setup Output
```
[SETUP] Downloading Qwen/Qwen2.5-0.5B-Instruct (this may take a few minutes)...
[SETUP] Model downloaded to /tmp/pytest-xxx/downloads/Qwen2.5-0.5B-Instruct
test_download_model PASSED
```
- Module fixture ran (only happens once)
- Model cached for subsequent tests

### Skipped Test
```
test_generate_imatrix SKIPPED (No calibration file found)
```
- Test requires resource not available
- Not a failure - test handled gracefully

## When to Run Integration Tests

### Before Releasing
Run full integration test suite before any release:
```bash
pytest tests/test_integration.py -v -m integration
```

### After llama.cpp Updates
After updating llama.cpp binaries or conversion scripts:
```bash
# Run smoke tests first (fast)
pytest tests/test_llama_cpp_smoke.py -v

# If smoke tests pass, run integration tests
pytest tests/test_integration.py -v -m integration
```

### When Changing Conversion Logic
If modifying `converter.py` or related code:
```bash
# Run relevant integration test class
pytest tests/test_integration.py::TestQuantization -v
```

### In CI/CD
Consider running integration tests:
- On pull requests to main branch
- Nightly builds
- Before releases

May want to cache downloaded model to speed up CI runs.

## Optimizing Test Speed

### Run Only What You Need
```bash
# Fast check - basic conversion only (1-2 min)
pytest tests/test_integration.py::TestBasicConversion -v

# Medium check - conversion + quantization (2-4 min)
pytest tests/test_integration.py::TestBasicConversion -v
pytest tests/test_integration.py::TestQuantization -v

# Full check - all except imatrix (4-8 min)
pytest tests/test_integration.py -v -m integration -k "not imatrix"
```

### Skip Slow Tests
```bash
# Skip imatrix tests (they're the slowest)
pytest tests/test_integration.py -v -m integration -k "not imatrix"
```

### Use Cached Model
After first run, model is cached in temp directory. Subsequent runs are much faster.

## Test Output Locations

### Default Behavior (Automatic Cleanup)
By default, tests save files to pytest's temporary directory and clean up after:

**Windows**:
```
C:\Users\<username>\AppData\Local\Temp\pytest-of-<username>\pytest-<N>\integration_tests<N>\
```

**Linux/Mac**:
```
/tmp/pytest-of-<username>/pytest-<N>/integration_tests<N>/
```

**Model cache** (persists across runs):
```
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/
```

### Keeping Test Outputs

#### Option 1: Keep in temp directory
```bash
pytest -m integration --keep-test-outputs -v
```
Files stay in pytest temp directory (visible in test output).

#### Option 2: Save to specific directory
```bash
pytest -m integration --test-output-dir D:/test_outputs -v
```
Files saved to `D:/test_outputs/` and preserved.

#### Option 3: See where files are saved
```bash
pytest tests/test_show_temp_dir.py -v -s
```
Prints exact paths to console.

### What Gets Saved

Test outputs are organized by test class:
```
integration_tests/
├── downloads/
│   └── Qwen2.5-0.5B-Instruct/      # Downloaded model
├── basic_conversion/
│   └── Qwen2.5-0.5B-Instruct_F16.gguf
├── quantization/
│   ├── Qwen2.5-0.5B-Instruct_F16.gguf
│   ├── Qwen2.5-0.5B-Instruct_Q4_K_M.gguf
│   └── Qwen2.5-0.5B-Instruct_Q8_0.gguf
├── overwrite_test/
│   ├── Qwen2.5-0.5B-Instruct_F16.gguf
│   └── Qwen2.5-0.5B-Instruct_Q4_K_M.gguf
├── multiple_quants/
│   ├── Qwen2.5-0.5B-Instruct_F16.gguf
│   ├── Qwen2.5-0.5B-Instruct_Q4_K_M.gguf
│   ├── Qwen2.5-0.5B-Instruct_Q5_K_M.gguf
│   └── Qwen2.5-0.5B-Instruct_Q8_0.gguf
└── imatrix_test/
    ├── Qwen2.5-0.5B-Instruct_F16.gguf
    ├── Qwen2.5-0.5B-Instruct.imatrix
    └── Qwen2.5-0.5B-Instruct_IQ3_XXS.gguf
```

## Cleanup

### Automatic Cleanup
By default, tests clean up after completion:
- Temp directories removed (unless `--keep-test-outputs`)
- Test outputs deleted (unless `--test-output-dir`)
- Model download remains cached in HuggingFace cache

### Manual Cleanup
To free disk space:
```bash
# Clear pytest cache
pytest --cache-clear

# Remove test outputs (if kept)
rm -rf ./test_outputs

# Remove HuggingFace model cache (if needed)
# Windows:
# rmdir /s C:\Users\<username>\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct
# Linux/Mac:
# rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
```

## Adding New Integration Tests

When adding new features, add corresponding integration tests:

```python
@pytest.mark.integration
@pytest.mark.slow
class TestNewFeature:
    """
    Test new feature description
    """

    def test_new_feature(self, converter, downloaded_model, test_output_dir):
        """
        Test specific aspect of new feature
        """
        output_dir = test_output_dir / "new_feature"
        output_dir.mkdir(exist_ok=True)

        # Test implementation
        result = converter.new_feature(
            model_path=downloaded_model,
            output_dir=output_dir
        )

        # Assertions
        assert result.exists()
        assert result.stat().st_size > 0
```

## Troubleshooting

### Model Download Fails
```python
# Error: Connection timeout
```
**Solution**: Check internet connection, try again, or manually download model

### Disk Space Errors
```python
# Error: Insufficient disk space
```
**Solution**: Free up 2-3GB, or reduce number of quantization types tested

### Tests Timeout
```python
# Error: Test exceeded timeout
```
**Solution**: Tests may be slow on some systems - this is normal for integration tests

### Calibration File Missing
```python
# SKIPPED: No calibration file found
```
**Solution**: Download a calibration file to `calibration_data/` or skip imatrix tests

### Import Errors
```python
# ModuleNotFoundError: No module named 'gguf_converter'
```
**Solution**: Install project in development mode:
```bash
pip install -e .
```

## Best Practices

1. **Run smoke tests first** - They're fast and catch most issues
2. **Run integration tests before releases** - Catch subtle issues
3. **Use specific test classes** - Don't run full suite unless necessary
4. **Check disk space first** - Need 2-3GB free
5. **Be patient** - Integration tests are slow by nature
6. **Monitor first run** - Model download can take time

## Performance Tips

### Parallel Testing
Pytest can run tests in parallel with pytest-xdist:
```bash
# WARNING: May not work well with module fixtures
pip install pytest-xdist
pytest tests/test_integration.py -n 2 -m integration
```

### Reuse Model
Model is automatically reused within a test session. For multiple sessions, consider:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/persistent/cache
pytest tests/test_integration.py -v -m integration
```
