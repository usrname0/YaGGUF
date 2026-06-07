"""
Tests for llama.cpp manager functionality

These tests ensure binary path resolution and detection works correctly.
This is critical because:
1. llama.cpp binary names might change
2. Platform detection must work on Windows/Linux/Mac
3. Custom binary paths must be handled correctly
"""

import pytest
import json
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
    import sys
    manager, bin_dir = mock_llama_cpp_manager

    # Initially should not exist (no binaries in temp dir)
    assert not manager._binaries_exist()

    # Create fake binaries with platform-specific names
    exe_ext = '.exe' if sys.platform == 'win32' else ''
    (bin_dir / f'llama-quantize{exe_ext}').touch()
    (bin_dir / f'llama-imatrix{exe_ext}').touch()

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



# Appended: backend discovery + version-parse + cleanup robustness tests


def _bare_manager(os_name='win', arch='x64'):
    """Build a manager without running __init__ (no binary download / network)."""
    m = LlamaCppManager.__new__(LlamaCppManager)
    m.platform_info = {
        'os': os_name, 'arch': arch, 'variant': arch,
        'filename': '', 'ext': 'zip' if os_name != 'ubuntu' else 'tar.gz',
    }
    return m


class TestCudaBackendDiscovery:
    """CUDA backends must be read from release assets as major families."""

    def test_discovers_cuda_families_from_assets(self):
        m = _bare_manager()
        assets = {
            "llama-b9544-bin-win-cpu-x64.zip",
            "llama-b9544-bin-win-cuda-12.4-x64.zip",
            "llama-b9544-bin-win-cuda-13.3-x64.zip",
            "llama-b9544-bin-win-vulkan-x64.zip",
        }
        with patch.object(m, '_get_release_asset_names', return_value=assets):
            backends = m._discover_cuda_backends('b9544')
        # Major families ("CUDA 13.x" -> "cuda-13"), not exact patch versions
        assert backends == [
            ("CUDA 12.x (Nvidia)", "cuda-12"),
            ("CUDA 13.x (Nvidia)", "cuda-13"),
        ]

    def test_collapses_multiple_patches_to_one_family(self):
        m = _bare_manager()
        assets = {
            "llama-b9999-bin-win-cuda-13.3-x64.zip",
            "llama-b9999-bin-win-cuda-13.5-x64.zip",
        }
        with patch.object(m, '_get_release_asset_names', return_value=assets):
            backends = m._discover_cuda_backends('b9999')
        assert backends == [("CUDA 13.x (Nvidia)", "cuda-13")]

    def test_falls_back_when_assets_unavailable(self):
        m = _bare_manager()
        with patch.object(m, '_get_release_asset_names', return_value=set()):
            backends = m._discover_cuda_backends('b9544')
        # Non-empty static fallback so the dropdown still offers CUDA options
        assert backends
        assert all(v.startswith("cuda-") and "." not in v for _, v in backends)


class TestCudaResolution:
    """A saved major-family preference must resolve to the right exact patch."""

    def test_family_helper(self):
        m = _bare_manager()
        assert m._cuda_family("cuda-13.3") == "cuda-13"
        assert m._cuda_family("cuda-13") == "cuda-13"
        assert m._cuda_family("cpu") == "cpu"
        assert m._cuda_family("vulkan") == "vulkan"

    def test_resolves_family_to_highest_patch(self):
        m = _bare_manager()
        assets = {
            "llama-b9999-bin-win-cuda-13.3-x64.zip",
            "llama-b9999-bin-win-cuda-13.5-x64.zip",
        }
        with patch.object(m, '_get_release_asset_names', return_value=assets):
            assert m._resolve_cuda_variant('b9999', 'cuda-13') == 'cuda-13.5'

    def test_legacy_exact_value_heals_to_current_patch(self):
        # User saved cuda-13.1, but the new release only ships cuda-13.3
        m = _bare_manager()
        assets = {"llama-b9544-bin-win-cuda-13.3-x64.zip"}
        with patch.object(m, '_get_release_asset_names', return_value=assets):
            assert m._resolve_cuda_variant('b9544', 'cuda-13.1') == 'cuda-13.3'

    def test_non_cuda_and_offline_pass_through(self):
        m = _bare_manager()
        with patch.object(m, '_get_release_asset_names', return_value=set()):
            assert m._resolve_cuda_variant('b9544', 'cpu') == 'cpu'
            # Offline (no assets): can't resolve, return input unchanged
            assert m._resolve_cuda_variant('b9544', 'cuda-13') == 'cuda-13'

    def test_available_backends_memoized_for_latest(self):
        m = _bare_manager()
        with patch.object(m, 'get_latest_version', return_value='b9544') as gl, \
             patch.object(m, '_get_release_asset_names',
                          return_value={"llama-b9544-bin-win-cuda-13.3-x64.zip"}):
            first = m.get_available_gpu_backends()
            second = m.get_available_gpu_backends()
        assert first == second
        # Cached: latest version resolved only once across both calls
        assert gl.call_count == 1
        assert ("CPU", "cpu") in first


class TestUpdateBinariesValidation:
    """A missing backend variant must fail clearly without wiping binaries."""

    def test_missing_family_raises_helpful_error(self, tmp_path):
        m = _bare_manager()
        m.bin_dir = tmp_path / "bin"
        m.bin_dir.mkdir()
        # An existing (working) binary that must NOT be removed on failure
        sentinel = m.bin_dir / "llama-cli.exe"
        sentinel.write_text("existing")

        # Release only ships CUDA 12.x / 13.x; user asks for an absent family
        assets = {
            "llama-b9544-bin-win-cuda-12.4-x64.zip",
            "llama-b9544-bin-win-cuda-13.3-x64.zip",
        }
        with patch.object(m, '_binaries_exist', return_value=False), \
             patch.object(m, '_get_release_asset_names', return_value=assets):
            with pytest.raises(RuntimeError) as exc:
                m.update_binaries(version='b9544', gpu_backend='cuda-11')

        msg = str(exc.value)
        assert 'cuda-11' in msg
        # Lists available families (not exact patches)
        assert 'cuda-13' in msg
        assert 'cuda-13.3' not in msg
        # Critical: existing binaries left intact (validation happens pre-cleanup)
        assert sentinel.exists()

    def test_legacy_patch_request_resolves_and_proceeds(self, tmp_path, monkeypatch):
        # A stale exact cuda-13.1 must heal to the available cuda-13.3 and
        # build the correct download URL rather than 404.
        m = _bare_manager()
        m.bin_dir = tmp_path / "bin"
        m.bin_dir.mkdir()
        assets = {"llama-b9544-bin-win-cuda-13.3-x64.zip"}

        urls = []

        def fake_retrieve(url, path, reporthook=None):
            urls.append(url)
            Path(path).write_text("zip")

        with patch.object(m, '_binaries_exist', return_value=False), \
             patch.object(m, '_get_release_asset_names', return_value=assets), \
             patch.object(m, '_check_disk_space', return_value=True), \
             patch.object(m, '_cleanup_old_binaries'), \
             patch.object(m, '_extract_archive'), \
             patch.object(m, '_check_binary_files_exist', return_value=True), \
             patch.object(m, '_write_bin_info'), \
             patch('gguf_converter.llama_cpp_manager.urlretrieve', side_effect=fake_retrieve):
            m.update_binaries(version='b9544', gpu_backend='cuda-13.1')

        # Main archive resolved to the available patch
        assert any('llama-b9544-bin-win-cuda-13.3-x64.zip' in u for u in urls)
        # CUDA runtime DLLs use the same resolved patch
        assert any('cudart-llama-bin-win-cuda-13.3-x64.zip' in u for u in urls)


class TestCleanupExclude:
    """_cleanup_old_binaries must be able to preserve a freshly downloaded archive."""

    def test_exclude_preserves_archive(self, tmp_path):
        m = _bare_manager()
        m.bin_dir = tmp_path / "bin"
        m.bin_dir.mkdir()
        old = m.bin_dir / "old-llama.exe"
        old.write_text("old")
        archive = m.bin_dir / "llama-b9544-bin-win-cuda-13.3-x64.zip"
        archive.write_text("zip")

        m._cleanup_old_binaries(exclude={archive})

        assert archive.exists()
        assert not old.exists()


class TestVersionParsing:
    """Version tag must come from the 'version:' line, not CUDA device numbers."""

    def test_ignores_cuda_device_numbers(self, tmp_path):
        m = _bare_manager()
        cli = tmp_path / "llama-cli.exe"
        cli.write_text("x")

        cuda_output = (
            "ggml_cuda_init: found 1 CUDA devices:\n"
            "  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, "
            "VMM: yes, 32606 MiB free\n"
            "version: 9544 (a1b2c3d)\n"
            "built with MSVC for x64\n"
        )
        fake = Mock(returncode=0, stdout="", stderr=cuda_output)
        with patch.object(m, 'get_binary_path', return_value=cli), \
             patch('subprocess.run', return_value=fake):
            info = m.get_installed_version_info()

        assert info['tag'] == 'b9544'  # not b32606


def _json_response(payload):
    """A urlopen() context-manager stand-in whose .read() yields JSON bytes."""
    cm = Mock()
    cm.__enter__ = Mock(return_value=Mock(read=Mock(return_value=json.dumps(payload).encode())))
    cm.__exit__ = Mock(return_value=False)
    return cm


class TestAssetPagination:
    """_get_release_asset_names must page past GitHub's 30-asset truncation."""

    def test_pages_through_assets_endpoint(self):
        m = _bare_manager()

        # Release-by-tag returns an id plus a truncated (30-item) embedded list;
        # the dedicated assets endpoint then serves the full set across pages.
        page1 = [{"name": f"asset-{i}.zip"} for i in range(100)]
        page2 = [
            {"name": "llama-b9544-bin-win-cuda-13.3-x64.zip"},
            {"name": "llama-b9544-bin-win-cpu-x64.zip"},
        ]

        def fake_urlopen(url, timeout=None):
            assert timeout == LlamaCppManager.GITHUB_API_TIMEOUT
            if "releases/tags/" in url:
                return _json_response({"id": 42, "assets": page1[:30]})
            if "page=1" in url:
                return _json_response(page1)
            if "page=2" in url:
                return _json_response(page2)
            return _json_response([])

        with patch("gguf_converter.llama_cpp_manager.urlopen", side_effect=fake_urlopen):
            names = m._get_release_asset_names("b9544")

        # Got assets from beyond the first (truncated) page
        assert "llama-b9544-bin-win-cuda-13.3-x64.zip" in names
        assert len(names) == 102

    def test_failure_returns_empty_and_is_not_cached(self):
        m = _bare_manager()
        with patch("gguf_converter.llama_cpp_manager.urlopen", side_effect=OSError("boom")):
            assert m._get_release_asset_names("b9544") == set()
        # Transient failure must not be cached
        assert "b9544" not in getattr(m, "_asset_cache", {})
