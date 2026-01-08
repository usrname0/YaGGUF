"""
Tests for GUI utility functions

These tests ensure URL extraction works correctly.
"""

import pytest
from gguf_converter.gui_utils import extract_repo_id_from_url


class TestExtractRepoIdFromUrl:
    """Tests for extract_repo_id_from_url function"""

    def test_full_huggingface_url(self):
        """Test extraction from full HuggingFace URL"""
        url = "https://huggingface.co/meta-llama/Llama-3.2-3B"
        assert extract_repo_id_from_url(url) == "meta-llama/Llama-3.2-3B"

    def test_huggingface_url_with_tree(self):
        """Test extraction from URL with /tree/main path"""
        url = "https://huggingface.co/meta-llama/Llama-3.2-3B/tree/main"
        assert extract_repo_id_from_url(url) == "meta-llama/Llama-3.2-3B"

    def test_short_hf_url(self):
        """Test extraction from short hf.co URL"""
        url = "hf.co/meta-llama/Llama-3.2-3B"
        assert extract_repo_id_from_url(url) == "meta-llama/Llama-3.2-3B"

    def test_http_url(self):
        """Test extraction from http:// URL"""
        url = "http://huggingface.co/username/model-name"
        assert extract_repo_id_from_url(url) == "username/model-name"

    def test_already_repo_id_format(self):
        """Test that valid repo ID is returned as-is"""
        repo_id = "meta-llama/Llama-3.2-3B"
        assert extract_repo_id_from_url(repo_id) == "meta-llama/Llama-3.2-3B"

    def test_url_with_extra_whitespace(self):
        """Test URL with leading/trailing whitespace"""
        url = "  https://huggingface.co/username/model  "
        assert extract_repo_id_from_url(url) == "username/model"

    def test_invalid_url_no_username_model(self):
        """Test that invalid URL returns None"""
        url = "https://huggingface.co/"
        assert extract_repo_id_from_url(url) is None

    def test_empty_string(self):
        """Test that empty string returns None"""
        assert extract_repo_id_from_url("") is None

    def test_none_input(self):
        """Test that None returns None"""
        assert extract_repo_id_from_url(None) is None

    def test_invalid_format(self):
        """Test that invalid format returns None"""
        assert extract_repo_id_from_url("just-a-string") is None

    def test_repo_id_with_numbers_and_dashes(self):
        """Test repo ID with numbers and dashes"""
        url = "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1"
        assert extract_repo_id_from_url(url) == "mistralai/Mixtral-8x7B-v0.1"

    def test_repo_id_with_underscores(self):
        """Test repo ID with underscores"""
        url = "https://huggingface.co/some_user/some_model"
        assert extract_repo_id_from_url(url) == "some_user/some_model"


class TestDetectAllModelFiles:
    """Tests for detect_all_model_files function"""

    def test_empty_directory(self, tmp_path):
        """Test with empty directory"""
        from gguf_converter.gui_utils import detect_all_model_files
        result = detect_all_model_files(tmp_path)
        assert result == {}

    def test_single_gguf_file(self, tmp_path):
        """Test detecting single GGUF file"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"x" * 1024 * 1024 * 100)  # 100MB
        
        result = detect_all_model_files(tmp_path)
        
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key]['type'] == 'single'
        assert result[key]['extension'] == 'gguf'
        assert result[key]['shard_count'] == 1
        assert 'model' in result[key]['display_name']

    def test_single_safetensors_file(self, tmp_path):
        """Test detecting single safetensors file"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"x" * 1024 * 1024 * 100)  # 100MB
        
        result = detect_all_model_files(tmp_path)
        
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key]['type'] == 'single'
        assert result[key]['extension'] == 'safetensors'
        # Safetensors should show full filename
        assert 'model.safetensors' in result[key]['display_name']

    def test_split_gguf_files(self, tmp_path):
        """Test detecting split GGUF files"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        # Create split files
        for i in range(1, 4):
            shard = tmp_path / f"model-{i:05d}-of-00003.gguf"
            shard.write_bytes(b"x" * 1024 * 1024 * 50)  # 50MB each
        
        result = detect_all_model_files(tmp_path)
        
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key]['type'] == 'split'
        assert result[key]['extension'] == 'gguf'
        assert result[key]['shard_count'] == 3
        assert '3 shards' in result[key]['display_name']

    def test_split_safetensors_files(self, tmp_path):
        """Test detecting split safetensors files"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        for i in range(1, 3):
            shard = tmp_path / f"model-{i:05d}-of-00002.safetensors"
            shard.write_bytes(b"x" * 1024 * 1024 * 50)
        
        result = detect_all_model_files(tmp_path)
        
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key]['type'] == 'split'
        assert result[key]['extension'] == 'safetensors'
        assert result[key]['shard_count'] == 2

    def test_incomplete_split_files(self, tmp_path):
        """Test that incomplete split files are not detected"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        # Only create 2 out of 3 shards
        (tmp_path / "model-00001-of-00003.gguf").write_bytes(b"x" * 100)
        (tmp_path / "model-00002-of-00003.gguf").write_bytes(b"x" * 100)
        # Missing 00003
        
        result = detect_all_model_files(tmp_path)
        
        # Should not detect incomplete sets
        assert len(result) == 0

    def test_multiple_models(self, tmp_path):
        """Test detecting multiple different models"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        # Single GGUF
        (tmp_path / "model_a.gguf").write_bytes(b"x" * 1024 * 1024)
        
        # Split safetensors
        (tmp_path / "model_b-00001-of-00002.safetensors").write_bytes(b"x" * 1024 * 1024)
        (tmp_path / "model_b-00002-of-00002.safetensors").write_bytes(b"x" * 1024 * 1024)
        
        result = detect_all_model_files(tmp_path)
        
        assert len(result) == 2

    def test_nonexistent_directory(self, tmp_path):
        """Test with non-existent directory"""
        from gguf_converter.gui_utils import detect_all_model_files
        
        result = detect_all_model_files(tmp_path / "nonexistent")
        assert result == {}
