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


class TestMultimodalModel:
    """Tests for multimodal model detection (vision and audio)"""

    def test_detects_vision_config(self, tmp_path):
        config = {"vision_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_image_processor(self, tmp_path):
        config = {"image_processor_type": "SiglipImageProcessor"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_vision_architectures(self, tmp_path):
        architectures = [
            "LlavaForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "Gemma3ForConditionalGeneration",
            "Pixtral",
        ]
        for arch in architectures:
            config = {"architectures": [arch]}
            (tmp_path / "config.json").write_text(json.dumps(config))
            assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_audio_config(self, tmp_path):
        config = {"audio_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_audio_processor(self, tmp_path):
        config = {"audio_processor_type": "WhisperFeatureExtractor"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_audio_architectures(self, tmp_path):
        architectures = [
            "Qwen2AudioForConditionalGeneration",
            "UltravoxModel",
            "Qwen2_5OmniModel",
        ]
        for arch in architectures:
            config = {"architectures": [arch]}
            (tmp_path / "config.json").write_text(json.dumps(config))
            assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_detects_model_type_keywords(self, tmp_path):
        types = ["smolvlm", "pixtral-vision", "my-vision-model", "qwen2-audio", "my-asr-model", "omni-chat"]
        for mtype in types:
            config = {"model_type": mtype}
            (tmp_path / "config.json").write_text(json.dumps(config))
            assert ModelQuirks.is_multimodal_model(tmp_path) is True

    def test_negative_detection(self, tmp_path):
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is False

    def test_is_vision_model_alias(self, tmp_path):
        """is_vision_model is a deprecated alias for is_multimodal_model"""
        config = {"vision_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_vision_model(tmp_path) == ModelQuirks.is_multimodal_model(tmp_path)


class TestProjectorOnlyModel:
    """Tests for projector-only model detection (e.g. Ultravox)"""

    def test_detects_ultravox(self, tmp_path):
        config = {"architectures": ["UltravoxModel"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_projector_only_model(tmp_path) is True

    def test_negative_for_full_model(self, tmp_path):
        config = {"architectures": ["LlamaForCausalLM"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_projector_only_model(tmp_path) is False

    def test_negative_for_standard_multimodal(self, tmp_path):
        config = {"architectures": ["Qwen2AudioForConditionalGeneration"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_projector_only_model(tmp_path) is False

    def test_projector_only_implies_multimodal(self, tmp_path):
        """Projector-only models should also be detected as multimodal"""
        config = {"architectures": ["UltravoxModel"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert ModelQuirks.is_multimodal_model(tmp_path) is True


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
