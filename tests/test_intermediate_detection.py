"""
Tests for intermediate GGUF file detection in convert.py

Tests the detect_intermediate_gguf_files() function which scans directories
for intermediate GGUF files (F16, F32, BF16) in both single and split formats.
"""

import pytest
from pathlib import Path
from gguf_converter.gui_tabs.convert import detect_intermediate_gguf_files


class TestDetectIntermediateGGUFFiles:
    """Tests for detect_intermediate_gguf_files() function"""

    def test_empty_directory(self, tmp_path):
        """Test with empty directory returns empty dict"""
        result = detect_intermediate_gguf_files(tmp_path)
        assert result == {}

    def test_nonexistent_directory(self, tmp_path):
        """Test with nonexistent directory returns empty dict"""
        nonexistent = tmp_path / "does_not_exist"
        result = detect_intermediate_gguf_files(nonexistent)
        assert result == {}

    def test_single_f16_file(self, tmp_path):
        """Test detection of single F16 file"""
        f16_file = tmp_path / "model_F16.gguf"
        f16_file.write_bytes(b"dummy" * 100)  # 500 bytes

        result = detect_intermediate_gguf_files(tmp_path)

        assert "F16_single" in result
        assert result["F16_single"]["format"] == "F16"
        assert result["F16_single"]["type"] == "single"
        assert result["F16_single"]["shard_count"] == 1
        assert result["F16_single"]["primary_file"] == f16_file
        assert result["F16_single"]["total_size_gb"] > 0

    def test_single_f32_file(self, tmp_path):
        """Test detection of single F32 file"""
        f32_file = tmp_path / "model_F32.gguf"
        f32_file.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        assert "F32_single" in result
        assert result["F32_single"]["format"] == "F32"
        assert result["F32_single"]["type"] == "single"

    def test_single_bf16_file(self, tmp_path):
        """Test detection of single BF16 file"""
        bf16_file = tmp_path / "model_BF16.gguf"
        bf16_file.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        assert "BF16_single" in result
        assert result["BF16_single"]["format"] == "BF16"
        assert result["BF16_single"]["type"] == "single"

    def test_split_f16_complete_set(self, tmp_path):
        """Test detection of complete split F16 file set"""
        split_files = [
            tmp_path / "model_F16-00001-of-00003.gguf",
            tmp_path / "model_F16-00002-of-00003.gguf",
            tmp_path / "model_F16-00003-of-00003.gguf",
        ]
        for f in split_files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        assert "F16_split" in result
        assert result["F16_split"]["format"] == "F16"
        assert result["F16_split"]["type"] == "split"
        assert result["F16_split"]["shard_count"] == 3
        assert result["F16_split"]["primary_file"] == split_files[0]
        assert len(result["F16_split"]["files"]) == 3

    def test_split_f16_incomplete_set(self, tmp_path):
        """Test that incomplete split set is not detected"""
        # Missing shard 2
        split_files = [
            tmp_path / "model_F16-00001-of-00003.gguf",
            tmp_path / "model_F16-00003-of-00003.gguf",
        ]
        for f in split_files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # Incomplete set should not be in results
        assert "F16_split" not in result

    def test_both_single_and_split_same_format(self, tmp_path):
        """Test detection of both single and split F16 files"""
        # Single F16
        single_file = tmp_path / "model_single_F16.gguf"
        single_file.write_bytes(b"dummy" * 100)

        # Split F16
        split_files = [
            tmp_path / "model_split_F16-00001-of-00002.gguf",
            tmp_path / "model_split_F16-00002-of-00002.gguf",
        ]
        for f in split_files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # Both should be detected with different keys
        assert "F16_single" in result
        assert "F16_split" in result
        assert result["F16_single"]["format"] == "F16"
        assert result["F16_split"]["format"] == "F16"
        assert result["F16_single"]["type"] == "single"
        assert result["F16_split"]["type"] == "split"

    def test_multiple_formats(self, tmp_path):
        """Test detection of multiple different formats"""
        files = [
            tmp_path / "model_F16.gguf",
            tmp_path / "model_F32.gguf",
            tmp_path / "model_BF16.gguf",
        ]
        for f in files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        assert "F16_single" in result
        assert "F32_single" in result
        assert "BF16_single" in result

    def test_split_files_sorted_correctly(self, tmp_path):
        """Test that split files are sorted by shard number"""
        # Create in reverse order
        split_files = [
            tmp_path / "model_F16-00003-of-00003.gguf",
            tmp_path / "model_F16-00001-of-00003.gguf",
            tmp_path / "model_F16-00002-of-00003.gguf",
        ]
        for f in split_files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # Files should be sorted by shard number
        files = result["F16_split"]["files"]
        assert files[0].name == "model_F16-00001-of-00003.gguf"
        assert files[1].name == "model_F16-00002-of-00003.gguf"
        assert files[2].name == "model_F16-00003-of-00003.gguf"

    def test_non_intermediate_gguf_files_ignored(self, tmp_path):
        """Test that non-intermediate GGUF files are ignored"""
        ignored_files = [
            tmp_path / "model_Q4_K_M.gguf",
            tmp_path / "model_Q8_0.gguf",
            tmp_path / "model.gguf",
        ]
        for f in ignored_files:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # None of these should be detected
        assert result == {}

    def test_split_with_different_base_names(self, tmp_path):
        """Test detection of split files with different base names"""
        split_files_a = [
            tmp_path / "model_a_F16-00001-of-00002.gguf",
            tmp_path / "model_a_F16-00002-of-00002.gguf",
        ]
        split_files_b = [
            tmp_path / "model_b_F32-00001-of-00002.gguf",
            tmp_path / "model_b_F32-00002-of-00002.gguf",
        ]
        for f in split_files_a + split_files_b:
            f.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # Should detect both split sets
        assert "F16_split" in result
        assert "F32_split" in result

    def test_total_size_calculation(self, tmp_path):
        """Test that total_size_gb is calculated correctly"""
        # Create file with known size (1 MB = 1024 * 1024 bytes)
        f16_file = tmp_path / "model_F16.gguf"
        f16_file.write_bytes(b"x" * (1024 * 1024))  # 1 MB

        result = detect_intermediate_gguf_files(tmp_path)

        # Should be approximately 0.001 GB (allowing for float precision)
        assert 0.0009 < result["F16_single"]["total_size_gb"] < 0.0011

    def test_split_total_size_is_sum(self, tmp_path):
        """Test that split file total size is sum of all shards"""
        split_files = [
            tmp_path / "model_F16-00001-of-00002.gguf",
            tmp_path / "model_F16-00002-of-00002.gguf",
        ]
        # Each shard is 0.5 MB
        for f in split_files:
            f.write_bytes(b"x" * (512 * 1024))

        result = detect_intermediate_gguf_files(tmp_path)

        # Total should be ~1 MB = ~0.001 GB
        assert 0.0009 < result["F16_split"]["total_size_gb"] < 0.0011

    def test_mixed_valid_and_incomplete_splits(self, tmp_path):
        """Test with both valid and incomplete split sets"""
        # Complete F16 set
        f16_files = [
            tmp_path / "model_F16-00001-of-00002.gguf",
            tmp_path / "model_F16-00002-of-00002.gguf",
        ]
        for f in f16_files:
            f.write_bytes(b"dummy" * 100)

        # Incomplete F32 set (missing shard 1)
        f32_file = tmp_path / "model_F32-00002-of-00002.gguf"
        f32_file.write_bytes(b"dummy" * 100)

        result = detect_intermediate_gguf_files(tmp_path)

        # Only complete set should be detected
        assert "F16_split" in result
        assert "F32_split" not in result
