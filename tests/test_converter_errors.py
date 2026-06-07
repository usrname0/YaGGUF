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
