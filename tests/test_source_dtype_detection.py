"""
Tests for source dtype detection in convert tab
"""

import pytest
import json
from pathlib import Path
from gguf_converter.gui_tabs.convert import detect_source_dtype


class TestDetectSourceDtype:
    """Tests for detect_source_dtype function"""

    def test_detects_bfloat16(self, tmp_path):
        """Test detecting BF16 from config.json"""
        config = {"torch_dtype": "bfloat16"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result == "BF16"

    def test_detects_float16(self, tmp_path):
        """Test detecting F16 from config.json"""
        config = {"torch_dtype": "float16"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result == "F16"

    def test_detects_float32(self, tmp_path):
        """Test detecting F32 from config.json"""
        config = {"torch_dtype": "float32"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result == "F32"

    def test_detects_fp8(self, tmp_path):
        """Test detecting FP8 from config.json"""
        # Test float8_e4m3fn
        config = {"torch_dtype": "float8_e4m3fn"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        assert detect_source_dtype(tmp_path) == "FP8"

        # Test float8_e5m2
        config = {"torch_dtype": "float8_e5m2"}
        config_file.write_text(json.dumps(config))
        assert detect_source_dtype(tmp_path) == "FP8"

    def test_detects_dtype_field(self, tmp_path):
        """Test detecting dtype from 'dtype' field instead of 'torch_dtype'"""
        config = {"dtype": "bfloat16"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result == "BF16"

    def test_dtype_takes_precedence(self, tmp_path):
        """Test that 'dtype' takes precedence over 'torch_dtype'"""
        config = {"dtype": "float32", "torch_dtype": "bfloat16"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        # dtype should be checked first
        assert result == "F32"

    def test_missing_config_json(self, tmp_path):
        """Test returns None when config.json is missing"""
        result = detect_source_dtype(tmp_path)

        assert result is None

    def test_invalid_dtype_returns_none(self, tmp_path):
        """Test returns None for unsupported dtype"""
        config = {"torch_dtype": "int8"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result is None

    def test_missing_dtype_field(self, tmp_path):
        """Test returns None when dtype fields are missing"""
        config = {"model_type": "llama"}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result is None

    def test_malformed_json(self, tmp_path):
        """Test handles malformed JSON gracefully"""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")

        result = detect_source_dtype(tmp_path)

        # Should return None instead of raising exception
        assert result is None

    def test_case_insensitive(self, tmp_path):
        """Test that dtype matching is case-insensitive"""
        config = {"torch_dtype": "BFloat16"}  # Mixed case
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result == "BF16"

    def test_empty_dtype_value(self, tmp_path):
        """Test handles empty dtype value"""
        config = {"torch_dtype": ""}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = detect_source_dtype(tmp_path)

        assert result is None
