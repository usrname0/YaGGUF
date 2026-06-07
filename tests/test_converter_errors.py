"""Tests for GGUFConverter error-message helpers (no binaries required)."""
import sys

from gguf_converter.converter import GGUFConverter


class TestTransformersUpgradeHint:
    """_transformers_upgrade_hint should only fire on transformers/version errors."""

    def test_unrecognized_architecture_triggers_hint(self):
        err = (
            "ValueError: The checkpoint you are trying to load has model type "
            "`gemma3` but Transformers does not recognize this architecture."
        )
        hint = GGUFConverter._transformers_upgrade_hint(err)
        assert hint
        # Must use YaGGUF's own interpreter, not a bare `pip`
        assert sys.executable in hint
        assert "-m pip install -U transformers tokenizers sentencepiece" in hint
        # And offer the git-source fallback for models too new for any release
        assert "git+https://github.com/huggingface/transformers.git" in hint

    def test_import_error_requires_latest_triggers_hint(self):
        err = "ImportError: This model requires the latest version of transformers."
        assert GGUFConverter._transformers_upgrade_hint(err)

    def test_unrelated_oom_error_no_hint(self):
        err = "RuntimeError: CUDA out of memory while loading tensors"
        assert GGUFConverter._transformers_upgrade_hint(err) == ""

    def test_unrelated_missing_file_no_hint(self):
        err = "FileNotFoundError: config.json not found"
        assert GGUFConverter._transformers_upgrade_hint(err) == ""

    def test_bare_transformers_mention_without_signal_no_hint(self):
        # Mentions transformers but not a version/architecture problem
        err = "Loaded transformers and started conversion successfully"
        assert GGUFConverter._transformers_upgrade_hint(err) == ""


class TestRunCapture:
    """_run_capture must capture output in both modes, and stream when verbose."""

    def test_non_verbose_captures_output_without_streaming(self, capsys):
        cmd = [
            sys.executable,
            "-c",
            "import sys; print('stdout line'); print('stderr line', file=sys.stderr); sys.exit(3)",
        ]
        returncode, output = GGUFConverter._run_capture(cmd, verbose=False)
        assert returncode == 3
        # Both streams are captured into the returned buffer
        assert "stdout line" in output
        assert "stderr line" in output
        # ...but nothing leaks to our own terminal in non-verbose mode
        assert "stdout line" not in capsys.readouterr().out

    def test_verbose_streams_and_still_captures(self, capsys):
        cmd = [
            sys.executable,
            "-c",
            "import sys; print('to stderr', file=sys.stderr); sys.exit(0)",
        ]
        returncode, output = GGUFConverter._run_capture(cmd, verbose=True)
        assert returncode == 0
        # Captured for hint analysis even though verbose was requested
        assert "to stderr" in output
        # And streamed to the terminal in real time
        assert "to stderr" in capsys.readouterr().out
