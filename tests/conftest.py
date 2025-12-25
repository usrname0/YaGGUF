"""
Shared pytest fixtures and configuration for all tests
"""

import pytest
from pathlib import Path


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
        "markers", "integration: marks tests as integration tests requiring real files/network"
    )
    config.addinivalue_line(
        "markers", "requires_binaries: marks tests that require llama.cpp binaries"
    )
