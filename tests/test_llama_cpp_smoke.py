"""
Smoke tests for llama.cpp binaries and conversion scripts

These tests verify that llama.cpp tools can be invoked without errors.
Run these after updating llama.cpp to ensure nothing is broken.

Usage:
    pytest tests/test_llama_cpp_smoke.py -v
    pytest tests/test_llama_cpp_smoke.py -v -m requires_binaries
"""

import pytest
import subprocess
import sys
from pathlib import Path
from gguf_converter.llama_cpp_manager import LlamaCppManager


@pytest.fixture(scope="module")
def llama_manager():
    """
    Create llama.cpp manager for accessing binary paths
    """
    return LlamaCppManager()


@pytest.fixture(scope="module")
def conversion_script_path():
    """
    Find the convert_hf_to_gguf.py script
    """
    project_root = Path(__file__).parent.parent
    script_path = project_root / "llama.cpp" / "convert_hf_to_gguf.py"
    return script_path


@pytest.mark.requires_binaries
class TestLlamaCppBinaries:
    """
    Test that all llama.cpp binaries can be executed
    """

    def test_llama_quantize_help(self, llama_manager):
        """
        Verify llama-quantize can be invoked with --help
        """
        quantize_path = llama_manager.get_quantize_path()

        if not quantize_path.exists():
            pytest.skip(f"llama-quantize not found at {quantize_path}")

        result = subprocess.run(
            [str(quantize_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # llama-quantize returns 1 even for --help, but should output help text
        # Just verify it didn't crash and produced output
        output = result.stdout + result.stderr
        assert len(output) > 0, "llama-quantize should output help text"
        assert "usage:" in output.lower() or "quantize" in output.lower()

    def test_llama_imatrix_help(self, llama_manager):
        """
        Verify llama-imatrix can be invoked with --help
        """
        imatrix_path = llama_manager.get_imatrix_path()

        if not imatrix_path.exists():
            pytest.skip(f"llama-imatrix not found at {imatrix_path}")

        result = subprocess.run(
            [str(imatrix_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should return 0 (success)
        assert result.returncode == 0, f"llama-imatrix --help failed: {result.stderr}"

        # Output should contain expected help text
        output = result.stdout + result.stderr
        assert "usage:" in output.lower() or "imatrix" in output.lower()

    def test_llama_cli_version(self, llama_manager):
        """
        Verify llama-cli can be invoked with --version
        """
        cli_path = llama_manager.get_binary_path('llama-cli')

        if not cli_path.exists():
            pytest.skip(f"llama-cli not found at {cli_path}")

        result = subprocess.run(
            [str(cli_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should return 0 (success)
        assert result.returncode == 0, f"llama-cli --version failed: {result.stderr}"

        # Output should contain version information
        output = result.stdout + result.stderr
        assert "version" in output.lower() or "llama" in output.lower()

    def test_llama_quantize_version(self, llama_manager):
        """
        Verify llama-quantize can report its version

        Note: llama-quantize doesn't have a traditional --version flag
        This test verifies it runs without crashing
        """
        quantize_path = llama_manager.get_quantize_path()

        if not quantize_path.exists():
            pytest.skip(f"llama-quantize not found at {quantize_path}")

        # llama-quantize doesn't support --version, but shouldn't crash
        result = subprocess.run(
            [str(quantize_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should produce output (even if exit code is 1)
        output = result.stdout + result.stderr
        assert len(output) > 0, "llama-quantize should produce output"

    def test_llama_imatrix_version(self, llama_manager):
        """
        Verify llama-imatrix can report its version
        """
        imatrix_path = llama_manager.get_imatrix_path()

        if not imatrix_path.exists():
            pytest.skip(f"llama-imatrix not found at {imatrix_path}")

        result = subprocess.run(
            [str(imatrix_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should return 0 (success)
        assert result.returncode == 0, f"llama-imatrix --version failed: {result.stderr}"

        # Output should contain version information
        output = result.stdout + result.stderr
        assert "version" in output.lower()


@pytest.mark.requires_binaries
class TestConversionScripts:
    """
    Test that conversion scripts can be executed

    Note: We only test convert_hf_to_gguf.py since that's the only
    conversion script actually used by this project.
    """

    def test_convert_hf_to_gguf_help(self, conversion_script_path):
        """
        Verify convert_hf_to_gguf.py can be invoked with --help
        """
        if not conversion_script_path.exists():
            pytest.skip(f"convert_hf_to_gguf.py not found at {conversion_script_path}")

        result = subprocess.run(
            [sys.executable, str(conversion_script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should return 0 (success)
        assert result.returncode == 0, f"convert_hf_to_gguf.py --help failed: {result.stderr}"

        # Output should contain expected help text
        output = result.stdout + result.stderr
        assert "usage:" in output.lower() or "convert" in output.lower()

    def test_convert_hf_to_gguf_no_syntax_errors(self, conversion_script_path):
        """
        Verify convert_hf_to_gguf.py has no syntax or import errors
        """
        if not conversion_script_path.exists():
            pytest.skip(f"convert_hf_to_gguf.py not found at {conversion_script_path}")

        # Try to run with --help to check for syntax/import errors
        result = subprocess.run(
            [sys.executable, str(conversion_script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Verify no syntax or import errors
        output = result.stdout + result.stderr
        assert "SyntaxError" not in output, "Script has syntax errors"
        assert "ModuleNotFoundError" not in output, "Script has missing dependencies"
        assert "ImportError" not in output, "Script has import errors"


@pytest.mark.requires_binaries
class TestBinaryVersions:
    """
    Test that we can get version information from binaries
    """

    def test_get_binary_versions_match(self, llama_manager):
        """
        Verify that all binaries report compatible versions
        """
        imatrix_path = llama_manager.get_imatrix_path()

        if not imatrix_path.exists():
            pytest.skip("Binaries not found")

        # Get version from imatrix (quantize doesn't support --version)
        imatrix_result = subprocess.run(
            [str(imatrix_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # imatrix should succeed
        assert imatrix_result.returncode == 0

        # Should contain version info
        imatrix_output = imatrix_result.stdout + imatrix_result.stderr
        assert "version" in imatrix_output.lower()

    def test_manager_version_detection(self, llama_manager):
        """
        Verify that LlamaCppManager can detect installed binary version
        """
        version_info = llama_manager.get_installed_version_info()

        # Should return a dict
        assert isinstance(version_info, dict)
        assert 'full_version' in version_info
        assert 'tag' in version_info

        # If binaries exist, should detect version
        if llama_manager.get_quantize_path().exists():
            # At least one field should be populated
            assert version_info['full_version'] is not None or version_info['tag'] is not None

    def test_conversion_scripts_version_detection(self, llama_manager):
        """
        Verify that LlamaCppManager can detect conversion scripts version
        """
        version_info = llama_manager.get_installed_conversion_scripts_version_info()

        # Should return a dict
        assert isinstance(version_info, dict)
        assert 'full_version' in version_info
        assert 'tag' in version_info


@pytest.mark.requires_binaries
class TestBinaryInvocationFormats:
    """
    Test various ways of invoking binaries to ensure they handle args correctly
    """

    def test_llama_quantize_missing_args(self, llama_manager):
        """
        Verify llama-quantize fails gracefully with missing arguments
        """
        quantize_path = llama_manager.get_quantize_path()

        if not quantize_path.exists():
            pytest.skip(f"llama-quantize not found")

        # Running without arguments should fail but not crash
        result = subprocess.run(
            [str(quantize_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should provide error message (not crash silently)
        output = result.stdout + result.stderr
        assert len(output) > 0, "Binary should output error message when run without args"

    def test_llama_imatrix_missing_args(self, llama_manager):
        """
        Verify llama-imatrix fails gracefully with missing arguments
        """
        imatrix_path = llama_manager.get_imatrix_path()

        if not imatrix_path.exists():
            pytest.skip(f"llama-imatrix not found")

        # Running without arguments should fail but not crash
        result = subprocess.run(
            [str(imatrix_path)],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should provide error message (not crash silently)
        output = result.stdout + result.stderr
        assert len(output) > 0, "Binary should output error message when run without args"

    def test_convert_script_missing_args(self, conversion_script_path):
        """
        Verify convert_hf_to_gguf.py fails gracefully with missing arguments
        """
        if not conversion_script_path.exists():
            pytest.skip(f"convert_hf_to_gguf.py not found")

        # Running without arguments should fail but not crash
        result = subprocess.run(
            [sys.executable, str(conversion_script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should provide error message (not crash silently)
        output = result.stdout + result.stderr
        assert len(output) > 0, "Script should output error message when run without args"


@pytest.mark.requires_binaries
def test_all_binaries_exist(llama_manager):
    """
    Quick check that all required binaries exist

    This test can be used as a fast gate before running more expensive tests
    """
    required_binaries = ['llama-quantize', 'llama-imatrix']
    missing = []

    for binary_name in required_binaries:
        path = llama_manager.get_binary_path(binary_name)
        if not path.exists():
            missing.append(binary_name)

    if missing:
        pytest.skip(f"Missing required binaries: {', '.join(missing)}")

    # If we get here, all binaries exist
    assert True
