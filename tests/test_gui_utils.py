"""
Tests for GUI utility functions

These tests ensure URL extraction and disk space checking work correctly.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from gguf_converter.gui_utils import extract_repo_id_from_url, check_disk_space_sufficient


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


class TestCheckDiskSpaceSufficient:
    """Tests for check_disk_space_sufficient function"""

    def test_sufficient_space(self, tmp_path):
        """Test when there is sufficient disk space"""
        required = 1024  # 1KB
        sufficient, free = check_disk_space_sufficient(tmp_path, required)

        # tmp_path should exist and have plenty of space
        assert sufficient is True
        assert free > 0

    def test_with_custom_buffer(self, tmp_path):
        """Test with custom buffer size"""
        required = 1024
        buffer = 2048
        sufficient, free = check_disk_space_sufficient(tmp_path, required, buffer)

        assert isinstance(sufficient, bool)
        assert isinstance(free, int)

    def test_nonexistent_path(self):
        """Test that nonexistent path returns False"""
        nonexistent = Path("/this/path/does/not/exist")
        sufficient, free = check_disk_space_sufficient(nonexistent, 1024)

        assert sufficient is False
        assert free == 0

    @patch('shutil.disk_usage')
    def test_insufficient_space(self, mock_disk_usage, tmp_path):
        """Test when there is insufficient disk space"""
        # Mock disk_usage to return low free space
        mock_stat = Mock()
        mock_stat.free = 100  # Only 100 bytes free
        mock_disk_usage.return_value = mock_stat

        required = 1024 * 1024 * 1024  # 1GB
        sufficient, free = check_disk_space_sufficient(tmp_path, required)

        assert sufficient is False
        assert free == 100

    @patch('shutil.disk_usage')
    def test_exact_space_match(self, mock_disk_usage, tmp_path):
        """Test when free space exactly matches requirement + buffer"""
        buffer = 500 * 1024 * 1024  # 500MB
        required = 1024 * 1024 * 1024  # 1GB
        total_needed = required + buffer

        mock_stat = Mock()
        mock_stat.free = total_needed  # Exact match
        mock_disk_usage.return_value = mock_stat

        sufficient, free = check_disk_space_sufficient(tmp_path, required, buffer)

        assert sufficient is True
        assert free == total_needed

    @patch('shutil.disk_usage')
    def test_exception_handling(self, mock_disk_usage, tmp_path):
        """Test that exceptions are handled gracefully"""
        mock_disk_usage.side_effect = OSError("Disk error")

        sufficient, free = check_disk_space_sufficient(tmp_path, 1024)

        assert sufficient is False
        assert free == 0

    def test_default_buffer_is_500mb(self, tmp_path):
        """Test that default buffer is 500MB"""
        # We can verify the buffer by mocking and checking the calculation
        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_stat = Mock()
            mock_stat.free = 600 * 1024 * 1024  # 600MB free
            mock_disk_usage.return_value = mock_stat

            required = 200 * 1024 * 1024  # 200MB required
            # With 500MB buffer, total needed is 700MB
            # 600MB < 700MB, so should be insufficient
            sufficient, _ = check_disk_space_sufficient(tmp_path, required)

            assert sufficient is False
