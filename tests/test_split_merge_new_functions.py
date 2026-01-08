"""
Tests for new split/merge/resplit/copy functions in gui_tabs/split_merge.py
"""

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from gguf_converter.gui_tabs.split_merge import (
    split_gguf_file,
    split_safetensors_file,
    resplit_gguf_shards,
    resplit_safetensors_shards,
    copy_auxiliary_files
)


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
