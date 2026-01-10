"""
Tests for ModelQuirks detection logic
"""

import pytest
import json
from pathlib import Path
from gguf_converter.model_quirks import ModelQuirks


class TestMistralFormat:
    """Tests for Mistral format detection"""

    def test_detects_consolidated_safetensors(self, tmp_path):
        """Should detect Mistral format if consolidated*.safetensors exists"""
        (tmp_path / "consolidated.safetensors").touch()
        assert ModelQuirks.uses_mistral_format(tmp_path) is True

    def test_detects_params_json(self, tmp_path):
        """Should detect Mistral format if params.json exists"""
        (tmp_path / "params.json").touch()
        assert ModelQuirks.uses_mistral_format(tmp_path) is True

    def test_detects_ministral3_in_config(self, tmp_path):
        """Should detect Mistral format via config for HF-converted models"""
        config = {
            "architectures": ["Mistral3ForConditionalGeneration"]
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.uses_mistral_format(tmp_path) is True

    def test_detects_ministral3_model_type(self, tmp_path):
        """Should detect Mistral format via model_type"""
        config = {
            "model_type": "ministral3"
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.uses_mistral_format(tmp_path) is True

    def test_negative_detection(self, tmp_path):
        """Should return False for standard models"""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").touch()
        assert ModelQuirks.uses_mistral_format(tmp_path) is False


class TestVisionModel:
    """Tests for Vision/Multimodal model detection"""

    def test_detects_vision_config(self, tmp_path):
        """Should detect vision model if vision_config is present"""
        config = {
            "vision_config": {"hidden_size": 1024}
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_vision_model(tmp_path) is True

    def test_detects_image_processor(self, tmp_path):
        """Should detect vision model if image_processor_type is present"""
        config = {
            "image_processor_type": "SiglipImageProcessor"
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_vision_model(tmp_path) is True

    def test_detects_vision_architectures(self, tmp_path):
        """Should detect known vision architectures"""
        architectures = [
            "LlavaForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "Gemma3ForConditionalGeneration",
            "Pixtral"
        ]
        
        for arch in architectures:
            config = {"architectures": [arch]}
            config_file = tmp_path / "config.json"
            config_file.write_text(json.dumps(config))
            assert ModelQuirks.is_vision_model(tmp_path) is True

    def test_detects_model_type_keywords(self, tmp_path):
        """Should detect keywords in model_type"""
        types = ["smolvlm", "pixtral-vision", "my-vision-model"]
        
        for mtype in types:
            config = {"model_type": mtype}
            config_file = tmp_path / "config.json"
            config_file.write_text(json.dumps(config))
            assert ModelQuirks.is_vision_model(tmp_path) is True

    def test_negative_detection(self, tmp_path):
        """Should return False for text-only models"""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"]
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_vision_model(tmp_path) is False


class TestSentenceTransformers:
    """Tests for Sentence Transformers detection"""

    def test_detects_dense_module(self, tmp_path):
        """Should detect if modules.json contains Dense module"""
        modules = [
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling"},
            {"type": "sentence_transformers.models.Dense"},
            {"type": "sentence_transformers.models.Normalize"}
        ]
        (tmp_path / "modules.json").write_text(json.dumps(modules))
        assert ModelQuirks.is_sentence_transformers_model(tmp_path) is True

    def test_negative_if_no_dense_module(self, tmp_path):
        """Should return False if modules.json exists but no Dense module"""
        modules = [
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling"}
        ]
        (tmp_path / "modules.json").write_text(json.dumps(modules))
        assert ModelQuirks.is_sentence_transformers_model(tmp_path) is False

    def test_negative_if_no_modules_json(self, tmp_path):
        """Should return False if modules.json is missing"""
        assert ModelQuirks.is_sentence_transformers_model(tmp_path) is False


class TestConversionFlags:
    """Tests for flag generation"""

    def test_mistral_flags(self, tmp_path):
        """Should include --mistral-format for Mistral models"""
        (tmp_path / "params.json").touch()
        
        flags = ModelQuirks.get_conversion_flags(tmp_path)
        assert "--mistral-format" in flags

    def test_vision_flags_mmproj(self, tmp_path):
        """Should include --mmproj only when requested for vision models"""
        config = {"vision_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        
        # Without explicit request
        flags = ModelQuirks.get_conversion_flags(tmp_path, include_mmproj=False)
        assert "--mmproj" not in flags
        
        # With explicit request
        flags = ModelQuirks.get_conversion_flags(tmp_path, include_mmproj=True)
        assert "--mmproj" in flags

    def test_sentence_transformers_flags(self, tmp_path):
        """Should include dense modules flag"""
        modules = [{"type": "sentence_transformers.models.Dense"}]
        (tmp_path / "modules.json").write_text(json.dumps(modules))
        
        flags = ModelQuirks.get_conversion_flags(tmp_path)
        assert "--sentence-transformers-dense-modules" in flags

    def test_combined_flags(self, tmp_path):
        """Should handle multiple quirks simultaneously"""
        # Create a "Frankenstein" model that triggers multiple quirks
        # (Unlikely in reality, but good for testing flag aggregation)
        (tmp_path / "params.json").touch()  # Mistral
        
        modules = [{"type": "sentence_transformers.models.Dense"}]
        (tmp_path / "modules.json").write_text(json.dumps(modules)) # Sentence Transformer
        
        flags = ModelQuirks.get_conversion_flags(tmp_path)
        assert "--mistral-format" in flags
        assert "--sentence-transformers-dense-modules" in flags
