"""
Tests for model incompatibility detection

These tests ensure the MODEL_INCOMPATIBILITIES registry works correctly.
This is critical because:
1. It prevents users from wasting time on failed quantizations
2. New llama.cpp versions might change incompatibility patterns
3. New model architectures might have new incompatibilities
"""

import pytest
import json
from pathlib import Path
from gguf_converter.converter import GGUFConverter


@pytest.fixture
def mock_converter(tmp_path, monkeypatch):
    """
    Create a converter instance without downloading binaries
    """
    # Mock the binary manager initialization to skip downloads
    def mock_init(self, custom_binaries_folder=None):
        from gguf_converter.binary_manager import BinaryManager
        self.binary_manager = BinaryManager(custom_binaries_folder=custom_binaries_folder)

    def mock_ensure_repo(self):
        pass

    monkeypatch.setattr(GGUFConverter, "__init__", mock_init)
    monkeypatch.setattr(GGUFConverter, "_ensure_llama_cpp_repo", mock_ensure_repo)

    converter = GGUFConverter()
    return converter


def create_model_config(tmp_path, config_data):
    """
    Helper to create a temporary model directory with config.json
    """
    model_dir = tmp_path / "test_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    config_file = model_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    return model_dir


def test_tied_embeddings_detection_via_flag(tmp_path, mock_converter):
    """
    Test that tied embeddings are detected via tie_word_embeddings flag
    """
    # Create model with tied embeddings
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": True,  # This should trigger detection
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Check incompatibility detection
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)

    # Should detect IQ quants as incompatible
    assert "IQ2_XXS" in incompatible
    assert "IQ3_XS" in incompatible
    assert "IQ4_NL" in incompatible
    assert "Q2_K_S" in incompatible


def test_tied_embeddings_detection_via_model_family(tmp_path, mock_converter):
    """
    Test that tied embeddings are detected via model family patterns (Qwen)
    """
    # Create Qwen model (known to have tied embeddings)
    config = {
        "architectures": ["QWenForCausalLM"],  # Pattern matching "QWen"
        "tie_word_embeddings": False,  # Even if flag is false
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Check incompatibility detection
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)

    # Should still detect based on model family
    assert "IQ2_XXS" in incompatible
    assert "IQ3_XS" in incompatible


def test_no_incompatibilities_for_normal_model(tmp_path, mock_converter):
    """
    Test that normal models without tied embeddings have no incompatibilities
    """
    # Create normal model
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": False,
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Check incompatibility detection
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)

    # Should have no incompatibilities
    assert len(incompatible) == 0


def test_incompatibility_info_structure(tmp_path, mock_converter):
    """
    Test that get_incompatibility_info returns correct structure
    """
    # Create model with tied embeddings
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": True,
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Get detailed info
    info = mock_converter.get_incompatibility_info(model_dir)

    # Check structure
    assert "has_incompatibilities" in info
    assert "types" in info
    assert "incompatible_quants" in info
    assert "alternatives" in info
    assert "reasons" in info

    # Check values
    assert info["has_incompatibilities"] is True
    assert "tied_embeddings" in info["types"]
    assert len(info["incompatible_quants"]) > 0
    assert len(info["alternatives"]) > 0
    assert len(info["reasons"]) > 0


def test_incompatibility_info_no_incompatibilities(tmp_path, mock_converter):
    """
    Test get_incompatibility_info for models without incompatibilities
    """
    # Create normal model
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": False,
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Get detailed info
    info = mock_converter.get_incompatibility_info(model_dir)

    # Should indicate no incompatibilities
    assert info["has_incompatibilities"] is False
    assert len(info["types"]) == 0
    assert len(info["incompatible_quants"]) == 0


def test_alternatives_provided(tmp_path, mock_converter):
    """
    Test that alternative quantizations are provided for incompatible models
    """
    # Create model with tied embeddings
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": True,
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Get incompatibility info
    info = mock_converter.get_incompatibility_info(model_dir)

    # Check alternatives are provided
    alternatives = info["alternatives"]
    assert len(alternatives) > 0

    # Alternatives should mention Q3_K, Q2_K, or Q4_K as safe options
    alternatives_text = " ".join(alternatives)
    assert any(quant in alternatives_text for quant in ["Q3_K", "Q2_K", "Q4_K"])


def test_missing_config_file(tmp_path, mock_converter):
    """
    Test handling of model directory without config.json
    """
    # Create empty model directory
    model_dir = tmp_path / "empty_model"
    model_dir.mkdir()

    # Should not crash, just return empty list
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)
    assert incompatible == []

    info = mock_converter.get_incompatibility_info(model_dir)
    assert info["has_incompatibilities"] is False


def test_corrupt_config_file(tmp_path, mock_converter):
    """
    Test handling of corrupt config.json
    """
    # Create model with corrupt JSON
    model_dir = tmp_path / "corrupt_model"
    model_dir.mkdir()
    config_file = model_dir / "config.json"

    with open(config_file, 'w') as f:
        f.write("{ this is not valid json }")

    # Should not crash, just return empty list
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)
    assert incompatible == []

    info = mock_converter.get_incompatibility_info(model_dir)
    assert info["has_incompatibilities"] is False


def test_all_iq_quants_flagged(tmp_path, mock_converter):
    """
    Test that all IQ quantization types are flagged for tied embeddings

    This is important because new IQ quants might be added to llama.cpp
    """
    # Create model with tied embeddings
    config = {
        "architectures": ["LlamaForCausalLM"],
        "tie_word_embeddings": True,
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Get incompatible quants
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)

    # All IQ types should be flagged
    iq_quants = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                 "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL"]

    for iq_quant in iq_quants:
        assert iq_quant in incompatible, f"{iq_quant} should be incompatible with tied embeddings"


def test_model_incompatibilities_registry_structure():
    """
    Test that MODEL_INCOMPATIBILITIES has the expected structure

    This catches accidental registry modifications
    """
    from gguf_converter.converter import GGUFConverter

    registry = GGUFConverter.MODEL_INCOMPATIBILITIES

    # Should have tied_embeddings entry
    assert "tied_embeddings" in registry

    tied = registry["tied_embeddings"]

    # Check required keys
    assert "description" in tied
    assert "detection" in tied
    assert "incompatible_quants" in tied
    assert "alternatives" in tied
    assert "reason" in tied

    # Check detection methods
    detection = tied["detection"]
    assert "config_flag" in detection or "model_families" in detection

    # Check incompatible quants list is not empty
    assert len(tied["incompatible_quants"]) > 0

    # Check alternatives are provided
    assert len(tied["alternatives"]) > 0


def test_qwen_variant_detection(tmp_path, mock_converter):
    """
    Test that different Qwen name variants are all detected
    """
    variants = ["Qwen", "QWen", "qwen"]

    for variant in variants:
        # Create model with variant naming
        config = {
            "architectures": [f"{variant}ForCausalLM"],
            "tie_word_embeddings": False,
            "hidden_size": 4096
        }
        model_dir = create_model_config(tmp_path / variant, config)

        # Should detect incompatibility
        incompatible = mock_converter.get_incompatible_quantizations(model_dir)
        assert len(incompatible) > 0, f"Failed to detect {variant} variant"
        assert "IQ2_XXS" in incompatible


def test_no_duplicate_incompatibilities(tmp_path, mock_converter):
    """
    Test that get_incompatible_quantizations doesn't return duplicates
    """
    # Create model with tied embeddings
    config = {
        "architectures": ["QWenForCausalLM"],  # Matches model family
        "tie_word_embeddings": True,  # Also matches config flag
        "hidden_size": 4096
    }
    model_dir = create_model_config(tmp_path, config)

    # Get incompatible quants
    incompatible = mock_converter.get_incompatible_quantizations(model_dir)

    # Check for duplicates
    assert len(incompatible) == len(set(incompatible)), "Found duplicate entries in incompatible quants"
