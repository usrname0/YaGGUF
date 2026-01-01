"""
Shared pytest fixtures and configuration for all tests
"""

import pytest
from pathlib import Path
import shutil


def pytest_addoption(parser):
    """
    Add custom command line options
    """
    parser.addoption(
        "--keep-test-outputs",
        action="store_true",
        default=False,
        help="Keep integration test output files (don't clean up temp directory)"
    )
    parser.addoption(
        "--test-output-dir",
        action="store",
        default=None,
        help="Directory to save integration test outputs (default: pytest temp dir)"
    )


@pytest.fixture(scope="session")
def keep_outputs(request):
    """
    Check if test outputs should be kept
    """
    return request.config.getoption("--keep-test-outputs")


@pytest.fixture(scope="session")
def custom_test_output_dir(request):
    """
    Get custom test output directory if specified
    """
    return request.config.getoption("--test-output-dir")


@pytest.fixture
def sample_model_config():
    """
    Fixture providing a sample model config.json structure
    """
    return {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "tie_word_embeddings": False,
        "vocab_size": 32000
    }


@pytest.fixture
def sample_tied_embeddings_config():
    """
    Fixture providing a model config with tied embeddings
    """
    return {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "tie_word_embeddings": True,  # This triggers incompatibility
        "vocab_size": 32000
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """
    Fixture providing a temporary directory with model structure
    """
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    return model_dir


# Pytest configuration
def pytest_configure(config):
    """
    Configure pytest with custom markers
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring real model downloads and significant disk space/time"
    )
    config.addinivalue_line(
        "markers", "requires_binaries: marks tests that require llama.cpp binaries"
    )
