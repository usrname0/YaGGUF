# YaGGUF Test Suite

## Installation

```bash
pip install -r tests/requirements-dev.txt
```

## Running Tests

```bash
# All fast tests
pytest

# Including integration tests (downloads model, ~15min)
pytest && pytest -m integration

# Via scripts (will prompt to install dependencies if needed)
scripts\run_tests.bat      # Windows
scripts/run_tests.sh       # Linux/Mac
```

**GUI:** Enable **Developer mode** in Info tab > Dev Options. A **Dev Tests** button appears in the sidebar.

## Test Overview

**243 tests total** - 230 fast (<10s), 13 integration (10-15min)

| File | Tests | Description |
|------|-------|-------------|
| `test_split_merge.py` | 33 | Shard analysis, split, merge, resplit (GGUF & safetensors) |
| `test_imatrix_stats.py` | 26 | Imatrix statistics parsing |
| `test_llama_cpp_manager.py` | 23 | Binary manager |
| `test_split_file_handling.py` | 20 | Shard validation utilities |
| `test_gui_utils.py` | 20 | GUI utilities (URL extraction, model detection) |
| `test_model_quirks.py` | 17 | Quirk detection (Mistral, Vision, Sentence-Transformers) |
| `test_quantization_validation.py` | 16 | Quantization type validation |
| `test_intermediate_detection.py` | 15 | Intermediate GGUF detection |
| `test_llama_cpp_smoke.py` | 14 | Binary execution (5-10s) |
| `test_integration.py` | 13 | End-to-end workflow with real model |
| `test_path_utils.py` | 12 | Path utilities |
| `test_source_dtype_detection.py` | 12 | Source dtype detection |
| `test_version_utils.py` | 11 | Version checking |
| `test_config.py` | 7 | Configuration management |
| `test_code_quality.py` | 2 | Code standards |
| `test_show_temp_dir.py` | 1 | Temp directory location |

## Common Commands

```bash
pytest tests/test_split_merge.py -v      # Specific file
pytest -x                                 # Stop on first failure
pytest -m "not integration"              # Skip integration
pytest -m integration -k "not imatrix"   # Integration without imatrix
pytest --collect-only                    # Show what would run
pytest --tb=long --showlocals            # Debug failures
```

## Test Markers

```bash
pytest -m integration          # Only integration tests
pytest -m "not integration"    # Skip integration tests
pytest -m "not slow"           # Skip slow tests
pytest -m requires_binaries    # Tests needing llama.cpp binaries
```

## Integration Tests

Require ~2-3GB disk space and internet (first run). Use SmolLM2-135M-Instruct.

```bash
pytest -m integration -v
pytest -m integration --test-output-dir ./test_outputs  # Keep outputs
```

### Checksum Test

Validates Q4_K_M output against a per-system baseline (not in git).

- First run: creates baseline, skips
- Subsequent runs: validates against baseline
- After llama.cpp update: delete `tests/golden_outputs/smollm2_135m_q4_k_m_checksum.txt` to regenerate

## Manual Variant Testing

Interactive script to test all GGUF variants with llama-server:

```bash
python tests/manual_variant_testing.py E:\LLM\my-model
python tests/manual_variant_testing.py E:\LLM\my-model --exclude F16 --exclude "*_K_S"
python tests/manual_variant_testing.py E:\LLM\my-model --port 9090
```

Loads each variant, opens browser, waits for ENTER to continue. Auto-detects mmproj files for vision models.

Also available via **Test Models** button in GUI sidebar (dev mode).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No tests collected | Run from project root, check `pytest --collect-only` |
| ModuleNotFoundError | `pip install -e .` |
| Integration timeout | Run specific class: `pytest tests/test_integration.py::TestBasicConversion -v` |
| Calibration file missing | Skip imatrix: `pytest -m integration -k "not imatrix"` |
| Binaries missing | Run GUI or `python scripts/check_and_download_binaries.py` |

## Configuration

`pytest.ini`: Test discovery in `tests/`, ignores `llama.cpp/`, verbose output, strict markers.
