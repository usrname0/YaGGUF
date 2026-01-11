"""
Tests for llama.cpp manager functionality

These tests ensure binary path resolution and detection works correctly.
This is critical because:
1. llama.cpp binary names might change
2. Platform detection must work on Windows/Linux/Mac
3. Custom binary paths must be handled correctly
"""

import pytest
import platform as platform_module
from pathlib import Path
from unittest.mock import patch, Mock
from gguf_converter.llama_cpp_manager import LlamaCppManager


@pytest.fixture
def mock_llama_cpp_manager(tmp_path, monkeypatch):
    """
    Create a llama.cpp manager without downloading binaries
    """
    # Set bin_dir to a temporary directory
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    manager = LlamaCppManager()
    monkeypatch.setattr(manager, 'bin_dir', bin_dir)

    return manager, bin_dir


def test_llama_cpp_manager_initialization():
    """
    Test that LlamaCppManager initializes without crashing
    """
    manager = LlamaCppManager()

    # Should have platform info
    assert hasattr(manager, 'platform_info')
    assert 'os' in manager.platform_info
    assert 'arch' in manager.platform_info


def test_platform_detection():
    """
    Test that platform is correctly detected
    """
    manager = LlamaCppManager()

    # Platform should be one of supported values
    assert manager.platform_info['os'] in ['win', 'ubuntu', 'macos']

    # Architecture should be detected
    assert manager.platform_info['arch'] in ['x64', 'arm64']


def test_binary_paths_format():
    """
    Test that binary paths are correctly formatted for the platform
    """
    manager = LlamaCppManager()

    # Get paths
    quantize_path = manager.get_binary_path('llama-quantize')
    imatrix_path = manager.get_binary_path('llama-imatrix')
    cli_path = manager.get_binary_path('llama-cli')

    # All should be Path objects
    assert isinstance(quantize_path, Path)
    assert isinstance(imatrix_path, Path)
    assert isinstance(cli_path, Path)

    # On Windows, should have .exe extension
    if manager.platform_info['os'] == 'win':
        assert quantize_path.name.endswith('.exe')
        assert imatrix_path.name.endswith('.exe')
        assert cli_path.name.endswith('.exe')
    else:
        # On Unix, should not have .exe
        assert not quantize_path.name.endswith('.exe')
        assert not imatrix_path.name.endswith('.exe')
        assert not cli_path.name.endswith('.exe')


def test_binary_names():
    """
    Test that binary names are correct
    """
    manager = LlamaCppManager()

    # Get binary names (without .exe)
    quantize_name = manager.get_binary_path('llama-quantize').stem
    imatrix_name = manager.get_binary_path('llama-imatrix').stem
    cli_name = manager.get_binary_path('llama-cli').stem

    # Names should match expected
    assert quantize_name == 'llama-quantize'
    assert imatrix_name == 'llama-imatrix'
    assert cli_name == 'llama-cli'


def test_custom_binaries_folder():
    """
    Test that custom_binaries_folder parameter is stored
    """
    custom_folder = "/custom/path/to/binaries"
    manager = LlamaCppManager(custom_binaries_folder=custom_folder)

    # Should store the custom folder setting
    assert manager.custom_binaries_folder == custom_folder


def test_system_path_fallback():
    """
    Test that empty string for custom binaries is stored
    """
    manager = LlamaCppManager(custom_binaries_folder="")

    # Should store the empty string (signals system PATH usage)
    assert manager.custom_binaries_folder == ""


def test_llama_cpp_version_defined():
    """
    Test that LLAMA_CPP_VERSION is defined and has expected format
    """
    manager = LlamaCppManager()

    # Should have version constant
    assert hasattr(manager, 'LLAMA_CPP_VERSION')
    assert isinstance(manager.LLAMA_CPP_VERSION, str)
    assert len(manager.LLAMA_CPP_VERSION) > 0

    # Should start with 'b' (build number format)
    assert manager.LLAMA_CPP_VERSION.startswith('b')


def test_get_quantize_path():
    """
    Test get_quantize_path convenience method
    """
    manager = LlamaCppManager()

    path = manager.get_quantize_path()

    # Should return a Path
    assert isinstance(path, Path)

    # Should contain 'llama-quantize'
    assert 'llama-quantize' in str(path)


def test_get_imatrix_path():
    """
    Test get_imatrix_path convenience method
    """
    manager = LlamaCppManager()

    path = manager.get_imatrix_path()

    # Should return a Path
    assert isinstance(path, Path)

    # Should contain 'llama-imatrix'
    assert 'llama-imatrix' in str(path)


@patch('platform.system')
def test_windows_platform_detection(mock_system):
    """
    Test that Windows is correctly detected
    """
    mock_system.return_value = 'Windows'

    manager = LlamaCppManager()

    assert manager.platform_info['os'] == 'win'


@patch('platform.system')
def test_linux_platform_detection(mock_system):
    """
    Test that Linux is correctly detected
    """
    mock_system.return_value = 'Linux'

    manager = LlamaCppManager()

    assert manager.platform_info['os'] == 'ubuntu'


@patch('platform.system')
def test_macos_platform_detection(mock_system):
    """
    Test that macOS is correctly detected
    """
    mock_system.return_value = 'Darwin'

    manager = LlamaCppManager()

    assert manager.platform_info['os'] == 'macos'


@patch('platform.machine')
def test_x64_architecture_detection(mock_machine):
    """
    Test that x64 architecture is correctly detected
    """
    mock_machine.return_value = 'x86_64'

    manager = LlamaCppManager()

    assert manager.platform_info['arch'] == 'x64'


@patch('platform.machine')
def test_arm64_architecture_detection(mock_machine):
    """
    Test that ARM64 architecture is correctly detected
    """
    mock_machine.return_value = 'arm64'

    manager = LlamaCppManager()

    assert manager.platform_info['arch'] == 'arm64'


def test_bin_dir_is_absolute_path():
    """
    Test that bin_dir is always an absolute path
    """
    manager = LlamaCppManager()

    if manager.bin_dir is not None:
        assert manager.bin_dir.is_absolute()


def test_binary_path_construction(mock_llama_cpp_manager):
    """
    Test that binary paths are correctly constructed
    """
    manager, bin_dir = mock_llama_cpp_manager

    path = manager.get_binary_path('llama-quantize')

    # Should be inside bin_dir
    assert path.parent == bin_dir

    # Should have correct name
    assert 'llama-quantize' in path.name


def test_unknown_binary_name(mock_llama_cpp_manager):
    """
    Test handling of unknown binary name
    """
    manager, bin_dir = mock_llama_cpp_manager

    # Should still return a path (even if binary doesn't exist)
    path = manager.get_binary_path('nonexistent-binary')

    assert isinstance(path, Path)
    assert 'nonexistent-binary' in path.name


def test_binaries_exist_check(mock_llama_cpp_manager):
    """
    Test _binaries_exist method
    """
    manager, bin_dir = mock_llama_cpp_manager

    # Initially should not exist (no binaries in temp dir)
    assert not manager._binaries_exist()

    # Create fake binaries
    (bin_dir / 'llama-quantize.exe').touch()
    (bin_dir / 'llama-imatrix.exe').touch()

    # Now should exist (at least some files in bin_dir)
    assert manager._binaries_exist()


def test_github_release_url_format():
    """
    Test that GitHub release URL is correctly formatted

    This is important because the URL format might change
    """
    manager = LlamaCppManager()

    # The URL construction is internal, but we can check the version format
    # that would be used in the URL
    version = manager.LLAMA_CPP_VERSION

    # Version should be in format suitable for GitHub releases
    assert len(version) > 0
    assert ' ' not in version  # No spaces in version
    assert '\n' not in version  # No newlines


def test_multiple_managers_independent():
    """
    Test that multiple LlamaCppManager instances can exist with different settings
    """
    manager1 = LlamaCppManager()
    manager2 = LlamaCppManager(custom_binaries_folder="/custom/path")

    # Should have different custom_binaries_folder settings
    assert manager1.custom_binaries_folder != manager2.custom_binaries_folder


def test_platform_info_structure():
    """
    Test that platform_info has expected structure
    """
    manager = LlamaCppManager()

    # Should have required keys
    assert 'os' in manager.platform_info
    assert 'arch' in manager.platform_info

    # Values should be strings
    assert isinstance(manager.platform_info['os'], str)
    assert isinstance(manager.platform_info['arch'], str)


def test_download_url_construction():
    """
    Test that download URL can be constructed for different platforms

    This doesn't test actual downloads, just URL construction logic
    """
    manager = LlamaCppManager()

    # Should be able to construct URL for current platform
    # (We don't call this directly, but test that platform_info is sufficient)
    os_name = manager.platform_info['os']
    arch = manager.platform_info['arch']

    # These should be valid values for URL construction
    assert os_name in ['win', 'linux', 'mac', 'ubuntu', 'macos']
    assert arch in ['x64', 'arm64', 'unknown']


def test_get_server_path():
    """
    Test get_server_path convenience method
    """
    manager = LlamaCppManager()

    path = manager.get_server_path()

    # Should return a Path
    assert isinstance(path, Path)

    # Should contain 'llama-server'
    assert 'llama-server' in str(path)

