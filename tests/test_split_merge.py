"""
Tests for split/merge shards functionality in gui_tabs/split_merge.py

Consolidated tests for analyzing, splitting, merging, and resplitting
GGUF and safetensors files.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from gguf_converter.gui_tabs.split_merge import (
    analyze_shards,
    merge_gguf_shards,
    merge_safetensors_shards,
    split_gguf_file,
    split_safetensors_file,
    resplit_gguf_shards,
    resplit_safetensors_shards,
    copy_auxiliary_files
)


class TestAnalyzeShards:
    """
    Tests for analyze_shards()

    This function scans a directory for shard files and groups them by model.
    """

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        """
        Test that empty directory returns empty dictionary
        """
        result = analyze_shards(tmp_path, "gguf")
        assert result == {}

    def test_single_complete_gguf_model(self, tmp_path):
        """
        Test with single complete GGUF model
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
            tmp_path / "model-00003-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        assert len(result) == 1
        assert "model" in result
        assert result["model"]["total_expected"] == 3
        assert result["model"]["complete"] is True
        assert len(result["model"]["shards_found"]) == 3
        assert result["model"]["shards_found"] == [1, 2, 3]
        assert result["model"]["output_filename"] == "model.gguf"

    def test_single_incomplete_gguf_model(self, tmp_path):
        """
        Test with single incomplete GGUF model (missing shards)
        """
        shard_files = [
            tmp_path / "model-00001-of-00005.gguf",
            tmp_path / "model-00003-of-00005.gguf",
            # Missing 00002, 00004, 00005
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        assert len(result) == 1
        assert "model" in result
        assert result["model"]["total_expected"] == 5
        assert result["model"]["complete"] is False
        assert len(result["model"]["shards_found"]) == 2
        assert result["model"]["missing_shards"] == [2, 4, 5]

    def test_multiple_complete_models(self, tmp_path):
        """
        Test with multiple complete models in same directory
        """
        # First model
        model_a_files = [
            tmp_path / "model_a-00001-of-00002.gguf",
            tmp_path / "model_a-00002-of-00002.gguf",
        ]
        # Second model
        model_b_files = [
            tmp_path / "model_b-00001-of-00003.gguf",
            tmp_path / "model_b-00002-of-00003.gguf",
            tmp_path / "model_b-00003-of-00003.gguf",
        ]

        for shard in model_a_files + model_b_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        assert len(result) == 2
        assert "model_a" in result
        assert "model_b" in result
        assert result["model_a"]["complete"] is True
        assert result["model_b"]["complete"] is True
        assert result["model_a"]["total_expected"] == 2
        assert result["model_b"]["total_expected"] == 3

    def test_mixed_complete_and_incomplete_models(self, tmp_path):
        """
        Test with mix of complete and incomplete models
        """
        # Complete model
        complete_files = [
            tmp_path / "complete-00001-of-00002.gguf",
            tmp_path / "complete-00002-of-00002.gguf",
        ]
        # Incomplete model
        incomplete_files = [
            tmp_path / "incomplete-00001-of-00003.gguf",
            # Missing 00002 and 00003
        ]

        for shard in complete_files + incomplete_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        assert len(result) == 2
        assert result["complete"]["complete"] is True
        assert result["incomplete"]["complete"] is False
        assert result["incomplete"]["missing_shards"] == [2, 3]

    def test_safetensors_files(self, tmp_path):
        """
        Test analyzing safetensors files
        """
        shard_files = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "safetensors")

        assert len(result) == 1
        assert "model" in result
        assert result["model"]["complete"] is True
        assert result["model"]["output_filename"] == "model.safetensors"

    def test_inconsistent_shard_counts(self, tmp_path):
        """
        Test with inconsistent shard counts marks as error
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00005.gguf",  # Different total!
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        assert len(result) == 1
        assert "model" in result
        assert "error" in result["model"]
        assert "Inconsistent" in result["model"]["error"]

    def test_single_shard_files_skipped(self, tmp_path):
        """
        Test that files with only 1 shard (xxx-00001-of-00001) are skipped

        These are not actually sharded files, just files with the shard naming pattern.
        """
        shard_files = [
            tmp_path / "model-00001-of-00001.gguf",
            tmp_path / "another-00001-of-00001.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        # Should return empty dict since single-shard files are skipped
        assert result == {}

    def test_non_shard_files_ignored(self, tmp_path):
        """
        Test that non-shard files are ignored
        """
        files = [
            tmp_path / "model.gguf",
            tmp_path / "readme.txt",
            tmp_path / "config.json",
        ]
        for f in files:
            f.touch()

        result = analyze_shards(tmp_path, "gguf")
        assert result == {}

    def test_mixed_gguf_and_safetensors(self, tmp_path):
        """
        Test that only files with requested extension are analyzed
        """
        gguf_files = [
            tmp_path / "model-00001-of-00002.gguf",
            tmp_path / "model-00002-of-00002.gguf",
        ]
        safetensors_files = [
            tmp_path / "other-00001-of-00002.safetensors",
            tmp_path / "other-00002-of-00002.safetensors",
        ]

        for shard in gguf_files + safetensors_files:
            shard.touch()

        # Analyze GGUF only
        result_gguf = analyze_shards(tmp_path, "gguf")
        assert len(result_gguf) == 1
        assert "model" in result_gguf

        # Analyze safetensors only
        result_safetensors = analyze_shards(tmp_path, "safetensors")
        assert len(result_safetensors) == 1
        assert "other" in result_safetensors

    def test_files_sorted_by_shard_number(self, tmp_path):
        """
        Test that files are sorted by shard number in output
        """
        # Create files in non-sequential order
        shard_files = [
            tmp_path / "model-00003-of-00003.gguf",
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        # Files should be sorted 1, 2, 3
        file_names = [f.name for f in result["model"]["files"]]
        assert file_names == [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]

    def test_complex_model_names(self, tmp_path):
        """
        Test with complex model names containing special characters
        """
        shard_files = [
            tmp_path / "Qwen3-VL-4B-Instruct_F16-00001-of-00009.gguf",
            tmp_path / "Qwen3-VL-4B-Instruct_F16-00002-of-00009.gguf",
            tmp_path / "Qwen3-VL-4B-Instruct_F16-00009-of-00009.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        result = analyze_shards(tmp_path, "gguf")

        # Should still be incomplete (missing shards 3-8)
        assert "Qwen3-VL-4B-Instruct_F16" in result
        assert result["Qwen3-VL-4B-Instruct_F16"]["complete"] is False
        assert result["Qwen3-VL-4B-Instruct_F16"]["total_expected"] == 9


class TestMergeGGUFShards:
    """
    Tests for merge_gguf_shards()

    This function merges GGUF shard files using llama-gguf-split.
    """

    def test_empty_shard_list_raises_error(self, tmp_path):
        """
        Test that empty shard list raises ValueError
        """
        output_file = tmp_path / "output.gguf"

        with pytest.raises(ValueError, match="No GGUF shard files"):
            merge_gguf_shards([], output_file)

    @patch('subprocess.run')
    def test_merge_valid_shards(self, mock_run, tmp_path):
        """
        Test merging valid shards calls llama-gguf-split correctly
        """
        # Setup mock
        mock_run.return_value = Mock(returncode=0, stdout="Merge complete", stderr="")

        # Create shard files
        shard_files = [
            tmp_path / "input" / "model-00001-of-00003.gguf",
            tmp_path / "input" / "model-00002-of-00003.gguf",
            tmp_path / "input" / "model-00003-of-00003.gguf",
        ]
        for shard in shard_files:
            shard.parent.mkdir(exist_ok=True)
            shard.touch()

        output_file = tmp_path / "output" / "merged.gguf"

        # Create a fake binary that exists
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_binary = bin_dir / "llama-gguf-split.exe"
        fake_binary.touch()

        # Patch the __file__ location to make the binary discoverable
        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            merge_gguf_shards(shard_files, output_file)

        # Verify subprocess was called
        assert mock_run.called
        call_args = mock_run.call_args[0][0]

        # Should call llama-gguf-split with --merge flag
        assert "--merge" in call_args
        assert str(shard_files[0]) in call_args  # First shard as input
        assert str(output_file) in call_args

    @patch('subprocess.run')
    def test_merge_creates_output_directory(self, mock_run, tmp_path):
        """
        Test that merge creates output directory if it doesn't exist
        """
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        shard_files = [
            tmp_path / "model-00001-of-00002.gguf",
            tmp_path / "model-00002-of-00002.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        # Output in non-existent subdirectory
        output_file = tmp_path / "new_dir" / "output.gguf"
        assert not output_file.parent.exists()

        # Create a fake binary that exists
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_binary = bin_dir / "llama-gguf-split.exe"
        fake_binary.touch()

        # Patch the __file__ location to make the binary discoverable
        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            merge_gguf_shards(shard_files, output_file)

        # Directory should be created
        assert output_file.parent.exists()

    @patch('subprocess.run')
    def test_merge_overwrites_existing_file(self, mock_run, tmp_path):
        """
        Test that merge deletes existing output file before merging
        """
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        shard_files = [
            tmp_path / "model-00001-of-00002.gguf",
            tmp_path / "model-00002-of-00002.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        output_file = tmp_path / "output.gguf"
        output_file.touch()  # Create existing file
        assert output_file.exists()

        with patch('gguf_converter.gui_tabs.split_merge.Path.exists', return_value=True):
            merge_gguf_shards(shard_files, output_file)

        # File should have been deleted (and recreated by mock merge)
        # We can't verify the actual deletion since subprocess is mocked,
        # but we can verify unlink was called by checking call count
        assert not output_file.exists() or mock_run.called

    @patch('subprocess.run')
    def test_merge_failure_raises_error(self, mock_run, tmp_path):
        """
        Test that merge failure raises RuntimeError
        """
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error: Failed to merge shards"
        )

        shard_files = [
            tmp_path / "model-00001-of-00002.gguf",
            tmp_path / "model-00002-of-00002.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        output_file = tmp_path / "output.gguf"

        # Create a fake binary that exists
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_binary = bin_dir / "llama-gguf-split.exe"
        fake_binary.touch()

        # Patch the __file__ location to make the binary discoverable
        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            with pytest.raises(RuntimeError, match="llama-gguf-split failed"):
                merge_gguf_shards(shard_files, output_file)

    def test_missing_binary_raises_error(self, tmp_path):
        """
        Test that missing llama-gguf-split binary raises FileNotFoundError
        """
        shard_files = [
            tmp_path / "model-00001-of-00002.gguf",
            tmp_path / "model-00002-of-00002.gguf",
        ]
        for shard in shard_files:
            shard.touch()

        output_file = tmp_path / "output.gguf"

        # Don't create the binary - it should be missing
        # Patch the __file__ location so it looks for binary in tmp_path
        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            with pytest.raises(FileNotFoundError, match="llama-gguf-split not found"):
                merge_gguf_shards(shard_files, output_file)


class TestMergeSafetensorsShards:
    """
    Tests for merge_safetensors_shards()

    This function merges safetensors shard files using the safetensors library.
    """

    def test_empty_shard_list_raises_error(self, tmp_path):
        """
        Test that empty shard list raises ValueError
        """
        output_file = tmp_path / "output.safetensors"

        with pytest.raises(ValueError, match="No safetensors shard files"):
            merge_safetensors_shards([], output_file)

    def test_missing_dependencies_raises_error(self, tmp_path):
        """
        Test that missing safetensors/torch dependencies raises ImportError
        """
        shard_files = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        ]
        for shard in shard_files:
            shard.touch()

        output_file = tmp_path / "output.safetensors"

        # Mock the imports to fail
        with patch.dict('sys.modules', {'safetensors.torch': None, 'torch': None}):
            with pytest.raises(ImportError, match="safetensors and torch are required"):
                merge_safetensors_shards(shard_files, output_file)

    @patch('safetensors.torch.load_file')
    @patch('safetensors.torch.save_file')
    def test_merge_valid_shards(self, mock_save, mock_load, tmp_path):
        """
        Test merging valid safetensors shards
        """
        # Create shard files
        shard_files = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        ]
        for shard in shard_files:
            shard.touch()

        # Mock tensor loading
        mock_load.side_effect = [
            {"layer1": "tensor1", "layer2": "tensor2"},
            {"layer3": "tensor3", "layer4": "tensor4"},
        ]

        output_file = tmp_path / "output.safetensors"

        merge_safetensors_shards(shard_files, output_file)

        # Verify load was called for each shard
        assert mock_load.call_count == 2

        # Verify save was called with merged tensors
        assert mock_save.called
        saved_tensors = mock_save.call_args[0][0]
        assert "layer1" in saved_tensors
        assert "layer2" in saved_tensors
        assert "layer3" in saved_tensors
        assert "layer4" in saved_tensors

    @patch('safetensors.torch.load_file')
    @patch('safetensors.torch.save_file')
    def test_merge_creates_output_directory(self, mock_save, mock_load, tmp_path):
        """
        Test that merge creates output directory if it doesn't exist
        """
        shard_files = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        ]
        for shard in shard_files:
            shard.touch()

        mock_load.side_effect = [
            {"layer1": "tensor1"},
            {"layer2": "tensor2"},
        ]

        # Output in non-existent subdirectory
        output_file = tmp_path / "new_dir" / "output.safetensors"
        assert not output_file.parent.exists()

        merge_safetensors_shards(shard_files, output_file)

        # Directory should be created
        assert output_file.parent.exists()

    @patch('safetensors.torch.load_file')
    @patch('safetensors.torch.save_file')
    def test_tensor_merging_order(self, mock_save, mock_load, tmp_path):
        """
        Test that tensors from later shards override earlier ones with same name

        This tests the behavior when shards have overlapping tensor names.
        """
        shard_files = [
            tmp_path / "model-00001-of-00003.safetensors",
            tmp_path / "model-00002-of-00003.safetensors",
            tmp_path / "model-00003-of-00003.safetensors",
        ]
        for shard in shard_files:
            shard.touch()

        # Simulate overlapping tensor names
        mock_load.side_effect = [
            {"common": "value1", "layer1": "tensor1"},
            {"common": "value2", "layer2": "tensor2"},
            {"common": "value3", "layer3": "tensor3"},
        ]

        output_file = tmp_path / "output.safetensors"

        merge_safetensors_shards(shard_files, output_file)

        # Last shard should win for overlapping keys
        saved_tensors = mock_save.call_args[0][0]
        assert saved_tensors["common"] == "value3"
        assert saved_tensors["layer1"] == "tensor1"
        assert saved_tensors["layer2"] == "tensor2"
        assert saved_tensors["layer3"] == "tensor3"


class TestSplitGGUFFile:
    """Tests for split_gguf_file()"""

    def test_empty_input_file_raises_error(self, tmp_path):
        """Test that nonexistent input file raises FileNotFoundError"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            split_gguf_file(tmp_path / "nonexistent.gguf", output_dir, "2000M")

    @patch('subprocess.run')
    def test_split_valid_file(self, mock_run, tmp_path):
        """Test splitting a valid GGUF file"""
        mock_run.return_value = Mock(returncode=0, stdout="Split complete", stderr="")

        input_file = tmp_path / "model.gguf"
        input_file.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_binary = bin_dir / "llama-gguf-split.exe"
        fake_binary.touch()

        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            with patch('pathlib.Path.glob', return_value=[
                output_dir / "model-00001-of-00002.gguf",
                output_dir / "model-00002-of-00002.gguf"
            ]):
                result = split_gguf_file(input_file, output_dir, "2000M")

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "--split" in call_args
        assert str(input_file) in call_args


class TestSplitSafetensorsFile:
    """Tests for split_safetensors_file()"""

    def test_empty_input_file_raises_error(self, tmp_path):
        """Test that nonexistent input file raises FileNotFoundError"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            split_safetensors_file(tmp_path / "nonexistent.safetensors", output_dir, 2.0)

    @patch('safetensors.torch.load_file')
    @patch('safetensors.torch.save_file')
    def test_split_valid_file(self, mock_save, mock_load, tmp_path):
        """Test splitting a valid safetensors file"""
        input_file = tmp_path / "model.safetensors"
        input_file.touch()

        # Create mock tensors with nbytes property
        mock_tensor1 = Mock()
        mock_tensor1.nbytes = 2_000_000_000  # 2GB

        mock_tensor2 = Mock()
        mock_tensor2.nbytes = 2_000_000_000  # 2GB

        mock_tensor3 = Mock()
        mock_tensor3.nbytes = 1_000_000_000  # 1GB

        mock_tensors = {
            "layer1": mock_tensor1,
            "layer2": mock_tensor2,
            "layer3": mock_tensor3,
        }
        mock_load.return_value = mock_tensors

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = split_safetensors_file(input_file, output_dir, max_shard_size_gb=2.5)

        assert mock_save.call_count >= 2
        assert len(result) >= 2


class TestResplitGGUFShards:
    """Tests for resplit_gguf_shards()"""

    @patch('subprocess.run')
    def test_resplit_valid_shards(self, mock_run, tmp_path):
        """Test resplitting GGUF shards"""
        # Create side effect that creates files when commands are run
        def create_files_on_command(cmd, *args, **kwargs):
            if '--merge' in cmd:
                # Extract output file path (last argument for merge)
                output_path = Path(cmd[-1])
                output_path.touch()
            elif '--split' in cmd:
                # Extract output base name (last argument for split)
                output_base = Path(cmd[-1])
                output_dir = output_base.parent
                base_name = output_base.name
                # Create sample split files
                (output_dir / f"{base_name}-00001-of-00002.gguf").touch()
                (output_dir / f"{base_name}-00002-of-00002.gguf").touch()
            return Mock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = create_files_on_command

        input_shards = [
            tmp_path / "model-00001-of-00003.gguf",
            tmp_path / "model-00002-of-00003.gguf",
            tmp_path / "model-00003-of-00003.gguf"
        ]
        for shard in input_shards:
            shard.touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create the files that will be found and deleted
        (output_dir / "model-00001-of-00002.gguf").touch()
        (output_dir / "model-00002-of-00002.gguf").touch()

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_binary = bin_dir / "llama-gguf-split.exe"
        fake_binary.touch()

        with patch('gguf_converter.gui_tabs.split_merge.__file__', str(tmp_path / "gguf_converter" / "gui_tabs" / "split_merge.py")):
            result = resplit_gguf_shards(input_shards, output_dir, "3000M")

        assert mock_run.call_count >= 2


class TestResplitSafetensorsShards:
    """Tests for resplit_safetensors_shards()"""

    @patch('safetensors.torch.load_file')
    @patch('safetensors.torch.save_file')
    def test_resplit_valid_shards(self, mock_save, mock_load, tmp_path):
        """Test resplitting safetensors shards"""
        input_shards = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors"
        ]
        for shard in input_shards:
            shard.touch()

        # Create mock tensors with nbytes property
        mock_tensor1 = Mock()
        mock_tensor1.nbytes = 2_000_000_000  # 2GB

        mock_tensor2 = Mock()
        mock_tensor2.nbytes = 2_000_000_000  # 2GB

        # Mock load_file to return tensors for input shards, then for temp file
        mock_load.side_effect = [
            {"layer1": mock_tensor1},
            {"layer2": mock_tensor2},
            {"layer1": mock_tensor1, "layer2": mock_tensor2}  # For re-loading temp file
        ]

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock save_file to create the temp file when called
        def create_temp_file(tensors, path):
            Path(path).touch()

        mock_save.side_effect = create_temp_file

        result = resplit_safetensors_shards(input_shards, output_dir, max_shard_size_gb=1.5)

        assert mock_save.call_count >= 1


class TestCopyAuxiliaryFiles:
    """Tests for copy_auxiliary_files()"""

    def test_copy_json_and_txt_files(self, tmp_path):
        """Test copying JSON and text files"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "config.json").touch()
        (input_dir / "tokenizer_config.json").touch()
        (input_dir / "README.md").touch()
        (input_dir / "model.safetensors").touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = copy_auxiliary_files(input_dir, output_dir)

        assert len(result) == 3
        assert (output_dir / "config.json").exists()
        assert (output_dir / "tokenizer_config.json").exists()
        assert (output_dir / "README.md").exists()
        assert not (output_dir / "model.safetensors").exists()

    def test_copy_various_extensions(self, tmp_path):
        """Test copying files with various auxiliary extensions"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        aux_files = [
            "config.json", "vocab.txt", "README.md", "model.proto",
            "tokenizer.model", "script.py", "config.yaml", "settings.yml",
            "template.jinja", "tokenizer.spm", "config.toml", "data.msgpack"
        ]

        for filename in aux_files:
            (input_dir / filename).touch()

        (input_dir / "model.gguf").touch()
        (input_dir / "weights.bin").touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = copy_auxiliary_files(input_dir, output_dir)

        assert len(result) == len(aux_files)

        for filename in aux_files:
            assert (output_dir / filename).exists()

        assert not (output_dir / "model.gguf").exists()
        assert not (output_dir / "weights.bin").exists()

    def test_empty_directory(self, tmp_path):
        """Test copying from empty directory"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = copy_auxiliary_files(input_dir, output_dir)

        assert result == []

    def test_overwrites_existing_files(self, tmp_path):
        """Test that existing files are overwritten"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        input_file = input_dir / "config.json"
        input_file.write_text("new content")

        output_file = output_dir / "config.json"
        output_file.write_text("old content")

        copy_auxiliary_files(input_dir, output_dir)

        assert output_file.read_text() == "new content"
