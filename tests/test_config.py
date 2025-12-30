"""
Tests for configuration management utilities
"""

import pytest
import json
from pathlib import Path
from gguf_converter.gui_utils import (
    get_default_config,
    save_config,
    load_config,
    reset_config,
    CONFIG_FILE
)


@pytest.fixture
def temp_config_file(tmp_path, monkeypatch):
    """
    Fixture to use a temporary config file instead of the real one
    """
    temp_config = tmp_path / "test_config.json"
    monkeypatch.setattr('gguf_converter.gui_utils.CONFIG_FILE', temp_config)
    return temp_config


def test_get_default_config():
    """
    Test that default config has expected structure and values
    """
    config = get_default_config()

    # Check essential keys exist
    assert "verbose" in config
    assert "use_imatrix" in config
    assert "num_threads" in config
    assert "model_path" in config
    assert "output_dir" in config

    # Check default values
    assert config["verbose"] is True
    assert config["use_imatrix"] is True
    assert config["num_threads"] is None  # Auto-detect
    assert config["model_path"] == ""
    assert config["output_dir"] == ""

    # Check quantization defaults
    assert "other_quants" in config
    assert config["other_quants"]["Q4_K_M"] is True  # Default quant


def test_save_and_load_config(temp_config_file):
    """
    Test that config saves to file and loads correctly
    """
    # Create a custom config
    config = {
        "verbose": False,
        "num_threads": 8,
        "model_path": "/path/to/model",
        "output_dir": "/path/to/output"
    }

    # Save it
    save_config(config)

    # Verify file was created
    assert temp_config_file.exists()

    # Load it back
    loaded_config = load_config()

    # Verify values
    assert loaded_config["verbose"] is False
    assert loaded_config["num_threads"] == 8
    assert loaded_config["model_path"] == "/path/to/model"
    assert loaded_config["output_dir"] == "/path/to/output"


def test_load_config_with_missing_file(temp_config_file):
    """
    Test that load_config returns defaults when file doesn't exist
    """
    # Ensure file doesn't exist
    assert not temp_config_file.exists()

    # Load config
    config = load_config()

    # Should return default config
    default = get_default_config()
    assert config == default


def test_load_config_merges_with_defaults(temp_config_file):
    """
    Test that loading config merges saved values with defaults
    This ensures new config keys added in updates get default values
    """
    # Save a partial config (simulating old config format)
    partial_config = {
        "verbose": False,
        "num_threads": 4
    }

    with open(temp_config_file, 'w') as f:
        json.dump(partial_config, f)

    # Load config
    loaded = load_config()

    # Should have saved values
    assert loaded["verbose"] is False
    assert loaded["num_threads"] == 4

    # Should also have default values for missing keys
    assert "model_path" in loaded
    assert "output_dir" in loaded
    assert loaded["use_imatrix"] is True  # Default value


def test_load_config_handles_corrupt_file(temp_config_file):
    """
    Test that corrupt config file falls back to defaults
    """
    # Create a corrupt JSON file
    with open(temp_config_file, 'w') as f:
        f.write("{ this is not valid json }")

    # Load config - should return defaults without crashing
    config = load_config()

    # Should return default config
    default = get_default_config()
    assert config == default


def test_reset_config(temp_config_file):
    """
    Test that reset_config resets to defaults
    """
    # Save a custom config
    custom_config = {
        "verbose": False,
        "num_threads": 16,
        "model_path": "/custom/path"
    }
    save_config(custom_config)

    # Reset config
    reset_config()

    # Load it back
    loaded = load_config()

    # Should be back to defaults
    assert loaded["verbose"] is True  # Default
    assert loaded["num_threads"] is None  # Default
    assert loaded["model_path"] == ""  # Default


def test_config_roundtrip(temp_config_file):
    """
    Test that config survives save/load/save/load cycle
    """
    # Get default config
    original = get_default_config()

    # Modify it
    original["verbose"] = False
    original["num_threads"] = 12
    original["model_path"] = "/test/path"

    # Save -> Load -> Save -> Load
    save_config(original)
    first_load = load_config()
    save_config(first_load)
    second_load = load_config()

    # Should be identical
    assert second_load["verbose"] == original["verbose"]
    assert second_load["num_threads"] == original["num_threads"]
    assert second_load["model_path"] == original["model_path"]
