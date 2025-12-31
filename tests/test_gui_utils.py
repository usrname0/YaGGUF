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
