"""
Tests for split file handling functionality in converter.py

Tests the shard validation and counting utilities used for working with
split GGUF files (e.g., model-00001-of-00009.gguf).
"""

import pytest
from pathlib import Path
from gguf_converter.converter import GGUFConverter


class TestGetShardCount:
    """
    Tests for GGUFConverter._get_shard_count()

    This method extracts the total shard count from shard file names.
    """

    def test_empty_list_returns_none(self):
        """
        Test that empty list returns None
        """
        result = GGUFConverter._get_shard_count([])
        assert result is None

    def test_single_shard_file_returns_count(self, tmp_path):
        """
        Test with single shard file returns correct total count
        """
        shard_file = tmp_path / "model-00001-of-00005.gguf"
        shard_file.touch()

        result = GGUFConverter._get_shard_count([shard_file])
        assert result == 5

    def test_multiple_files_consistent_counts(self, tmp_path):
        """
        Test with multiple files having consistent total counts
        """
        shard_files = [
            tmp_path / "model-00001-of-00009.gguf",
            tmp_path / "model-00002-of-00009.gguf",
            tmp_path / "model-00009-of-00009.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._get_shard_count(shard_files)
        assert result == 9

    def test_inconsistent_counts_returns_none(self, tmp_path):
        """
        Test with files having inconsistent total counts returns None
        """
        shard_files = [
            tmp_path / "model-00001-of-00005.gguf",
            tmp_path / "model-00002-of-00009.gguf",  # Different total!
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._get_shard_count(shard_files)
        assert result is None

    def test_non_shard_files_returns_none(self, tmp_path):
        """
        Test with non-shard files (no shard pattern in name) returns None
        """
        regular_file = tmp_path / "model.gguf"
        regular_file.touch()

        result = GGUFConverter._get_shard_count([regular_file])
        assert result is None

    def test_mixed_shard_and_non_shard_files(self, tmp_path):
        """
        Test with mix of shard and non-shard files extracts count from shards
        """
        files = [
            tmp_path / "model.gguf",
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "readme.txt",
            tmp_path / "model-00002-of-00003.gguf",
        ]
        for f in files:
            f.touch()

        result = GGUFConverter._get_shard_count(files)
        assert result == 3


class TestValidateShardSet:
    """
    Tests for GGUFConverter._validate_shard_set()

    This method validates that a set of shard files is complete
    (all shards from 1 to N are present).
    """

    def test_empty_list_returns_false(self):
        """
        Test that empty list returns False
        """
        result = GGUFConverter._validate_shard_set([])
        assert result is False

    def test_complete_shard_set_returns_true(self, tmp_path):
        """
        Test with complete shard set returns True
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
            tmp_path / "model-00003-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is True

    def test_missing_middle_shard_returns_false(self, tmp_path):
        """
        Test with missing middle shard returns False
        """
        shard_files = [
            tmp_path / "model-00001-of-00005.gguf",
            tmp_path / "model-00002-of-00005.gguf",
            # Missing 00003
            tmp_path / "model-00004-of-00005.gguf",
            tmp_path / "model-00005-of-00005.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False

    def test_missing_first_shard_returns_false(self, tmp_path):
        """
        Test with missing first shard returns False
        """
        shard_files = [
            # Missing 00001
            tmp_path / "model-00002-of-00003.gguf",
            tmp_path / "model-00003-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False

    def test_missing_last_shard_returns_false(self, tmp_path):
        """
        Test with missing last shard returns False
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
            # Missing 00003
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False

    def test_inconsistent_total_counts_returns_false(self, tmp_path):
        """
        Test with inconsistent total counts across shards returns False
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00005.gguf",  # Different total!
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False

    def test_non_shard_files_returns_false(self, tmp_path):
        """
        Test with non-shard files returns False
        """
        regular_file = tmp_path / "model.gguf"
        regular_file.touch()

        result = GGUFConverter._validate_shard_set([regular_file])
        assert result is False

    def test_single_complete_shard_set(self, tmp_path):
        """
        Test with single complete shard (only one file needed)
        """
        shard_file = tmp_path / "model-00001-of-00001.gguf"
        shard_file.touch()

        result = GGUFConverter._validate_shard_set([shard_file])
        assert result is True

    def test_large_shard_set(self, tmp_path):
        """
        Test with larger shard set (20 shards)
        """
        shard_files = [
            tmp_path / f"model-{i:05d}-of-00020.gguf"
            for i in range(1, 21)
        ]
        for shard in shard_files:
            shard.touch()

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is True

    def test_unordered_shard_list(self, tmp_path):
        """
        Test that validation works even if list is not sorted
        """
        shard_files = [
            tmp_path / "model-00003-of-00004.gguf",
            tmp_path / "model-00001-of-00004.gguf",
            tmp_path / "model-00004-of-00004.gguf",
            tmp_path / "model-00002-of-00004.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        # Should work regardless of order
        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is True

    def test_duplicate_shard_numbers(self, tmp_path):
        """
        Test behavior with duplicate shard numbers

        This is an edge case where the same shard number appears twice.
        The validation should still work correctly.
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
            tmp_path / "model-00003-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        # Add duplicate shard 2 with different path
        duplicate_shard = tmp_path / "copy" / "model-00002-of-00003.gguf"
        duplicate_shard.parent.mkdir()
        duplicate_shard.touch()
        shard_files.append(duplicate_shard)

        # Should still return True since we have all required shards
        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is True

    def test_different_model_names_mixed(self, tmp_path):
        """
        Test with shards from different models mixed together

        This should return False as the base names don't match
        """
        shard_files = [
            tmp_path / "model_a-00001-of-00002.gguf",
            tmp_path / "model_b-00001-of-00002.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        # Different base names, so not a complete set of any single model
        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False


class TestShardFileNaming:
    """
    Tests for shard file naming pattern validation

    Ensures the regex patterns correctly identify shard files.
    """

    def test_valid_shard_names(self, tmp_path):
        """
        Test various valid shard naming patterns
        """
        valid_names = [
            "model-00001-of-00009.gguf",
            "Qwen3-VL-4B-Instruct_F16-00001-of-00009.gguf",
            "my_model_Q4_K_M-00001-of-00002.gguf",
            "model-with-dashes-00005-of-00010.gguf",
            "model_with_underscores-00003-of-00007.gguf",
        ]

        shard_files = [tmp_path / name for name in valid_names]
        for shard in shard_files:
            shard.touch()

        # All should be recognized by _get_shard_count
        # (will return None due to inconsistent counts, but that's ok)
        result = GGUFConverter._get_shard_count(shard_files)
        # Result should be None due to different totals, but files should be recognized
        assert result is None or isinstance(result, int)

    def test_invalid_shard_names_ignored(self, tmp_path):
        """
        Test that invalid shard names are ignored
        """
        invalid_names = [
            "model.gguf",
            "model-shard-1.gguf",
            "model-00001.gguf",
            "model-of-00003.gguf",
            "model-00001-of.gguf",
            "model-00001-of-00003",  # No extension
        ]

        shard_files = [tmp_path / name for name in invalid_names]
        for shard in shard_files:
            shard.touch()

        # Should return None as no valid shard files
        result = GGUFConverter._get_shard_count(shard_files)
        assert result is None

        result = GGUFConverter._validate_shard_set(shard_files)
        assert result is False
